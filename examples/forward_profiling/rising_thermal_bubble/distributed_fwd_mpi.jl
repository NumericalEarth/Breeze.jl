# # Forward-Only Rising Thermal Bubble — Julia MPI Distributed Benchmark
#
# Benchmarks the forward model using native Julia GPU with MPI domain
# decomposition across multiple GPUs via Distributed(GPU()).
#
# Launch (4 GPUs):
#   LD_LIBRARY_PATH=/opt/amazon/openmpi5/lib:$LD_LIBRARY_PATH \
#   /opt/amazon/openmpi5/bin/mpirun -np 4 -x LD_LIBRARY_PATH \
#     julia --project=test examples/forward_profiling/rising_thermal_bubble_forward_mpi.jl

using MPI
MPI.Init()

const comm   = MPI.COMM_WORLD
const myrank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
const root   = myrank == 0

using Breeze
using Oceananigans
using Oceananigans.DistributedComputations
using CUDA
using BenchmarkTools
using Statistics: mean, median
using Printf
using Dates

function forward_loss(model, θ_initial, Δt, nsteps, loss_z_threshold, domain_z)
    FT = eltype(model.grid)
    set!(model; θ = θ_initial, ρ = FT(1))
    for _ in 1:nsteps
        time_step!(model, Δt)
    end
    θ_evolved = interior(model.temperature)
    k_start = ceil(Int, loss_z_threshold / domain_z * size(model.grid, 3)) + 1
    upper_θ = @view θ_evolved[:, :, k_start:end]
    return mean(upper_θ .^ 2)
end

function format_trial(trial::BenchmarkTools.Trial)
    med = median(trial).time / 1e9
    mn  = mean(trial).time / 1e9
    lo  = minimum(trial).time / 1e9
    hi  = maximum(trial).time / 1e9
    return (; median_s=med, mean_s=mn, min_s=lo, max_s=hi, nsamples=length(trial))
end

function main()
    root && @info "MPI configuration" nranks myrank
    root && @info "CUDA" runtime=CUDA.runtime_version() local_toolkit=CUDA.local_toolkit

    float_type = Float32
    Oceananigans.defaults.FloatType = float_type

    θ_background           = float_type(300.0)
    perturbation_amplitude = float_type(2.0)
    perturbation_radius    = float_type(1000.0)
    bubble_center_x        = float_type(5000.0)
    bubble_center_y        = float_type(5000.0)
    bubble_center_z        = float_type(2000.0)
    latitude               = float_type(45.0)

    domain_x = float_type(10000.0)
    domain_y = float_type(10000.0)
    domain_z = float_type(10000.0)
    topology = (Periodic, Bounded, Bounded)

    grid_sizes       = [(32, 32, 32)]
    loss_z_threshold = float_type(5000.0)
    nsteps           = 100
    bench_seconds    = 30
    bench_samples    = 10

    disable_bench = get(ENV, "BREEZE_DISABLE_BENCHMARK", "false") == "true"

    run_stamp = get(ENV, "BREEZE_PROFILE_RUN_STAMP",
                    Dates.format(now(Dates.UTC), "yyyy-mm-dd_HHMMSS"))
    run_dir = joinpath("benchmark_results", "run_" * run_stamp, "julia_mpi")
    root && mkpath(run_dir)
    MPI.Barrier(comm)

    arch = Distributed(GPU(); partition = Partition(nranks, 1, 1))
    root && @info "Architecture" arch nranks

    for (Nx, Ny, Nz) in grid_sizes
        GC.gc()
        MPI.Barrier(comm)
        root && @info "=" ^ 60
        root && @info @sprintf("Grid: %d × %d × %d  (%d cells, %d ranks)",
                               Nx, Ny, Nz, Nx*Ny*Nz, nranks)
        root && @info "=" ^ 60

        grid_kwargs = (; size=(Nx, Ny, Nz),
                         x=(0, domain_x), y=(0, domain_y), z=(0, domain_z),
                         topology)

        advection = WENO(order = 5)
        coriolis  = FPlane(; latitude)

        root && @info "Building grid + model..."
        grid  = RectilinearGrid(arch; grid_kwargs...)
        model = AtmosphereModel(grid; advection, coriolis,
                                dynamics = CompressibleDynamics())

        FT = eltype(grid)
        thermo = model.thermodynamic_constants
        Rᵈ  = thermo.molar_gas_constant / thermo.dry_air.molar_mass
        cᵖᵈ = thermo.dry_air.heat_capacity
        γ   = cᵖᵈ / (cᵖᵈ - Rᵈ)
        Δt  = FT(0.4 * domain_x / Nx / sqrt(γ * Rᵈ * θ_background))

        θ_perturbation(x, y, z) = FT(perturbation_amplitude) * exp(
            -((x - bubble_center_x)^2 + (y - bubble_center_y)^2 + (z - bubble_center_z)^2)
            / perturbation_radius^2)
        θ_initial_fn(x, y, z) = FT(θ_background) + θ_perturbation(x, y, z)

        θ_init = CenterField(grid)
        set!(θ_init, θ_initial_fn)

        grid_label = @sprintf("%dx%dx%d", Nx, Ny, Nz)

        # ── Warmup ───────────────────────────────────────────────────
        MPI.Barrier(comm)
        root && @info "Warmup run..."
        loss = forward_loss(model, θ_init, Δt, nsteps, loss_z_threshold, domain_z)
        CUDA.synchronize()
        MPI.Barrier(comm)
        root && @info @sprintf("  Loss = %.6e", loss)

        # ── Benchmark ────────────────────────────────────────────────
        if !disable_bench
            GC.gc()
            MPI.Barrier(comm)
            root && @info "Benchmarking ($nranks ranks, seconds=$bench_seconds, samples=$bench_samples)..."

            trial = @benchmark begin
                forward_loss($model, $θ_init, $Δt, $nsteps, $loss_z_threshold, $domain_z)
                CUDA.synchronize()
                MPI.Barrier($comm)
            end seconds=bench_seconds samples=bench_samples evals=1

            MPI.Barrier(comm)

            if root
                stats = format_trial(trial)
                @info @sprintf("  median: %.4f s  (n=%d, min=%.4f, max=%.4f)",
                               stats.median_s, stats.nsamples, stats.min_s, stats.max_s)

                open(joinpath(run_dir, "bench_$(grid_label).txt"), "w") do io
                    println(io, "# Julia MPI distributed  grid=$grid_label  nsteps=$nsteps  nranks=$nranks")
                    show(io, MIME"text/plain"(), trial); println(io)
                end

                summary_path = joinpath(run_dir, "summary.md")
                open(summary_path, "w") do io
                    println(io, "# Rising Thermal Bubble — Julia MPI Distributed Benchmark")
                    println(io, "\nRun: `$run_stamp`  |  nsteps=$nsteps  |  precision=$float_type  |  nranks=$nranks")
                    println(io, "\n| Grid | Ranks | Median (s) | Min (s) | Max (s) | Samples |")
                    println(io, "|------|-------|------------|---------|---------|---------|")
                    println(io, @sprintf("| %d×%d×%d | %d | %.4f | %.4f | %.4f | %d |",
                                         Nx, Ny, Nz, nranks, stats.median_s,
                                         stats.min_s, stats.max_s, stats.nsamples))
                end
                @info "Saved results to $run_dir"
            end
        end
    end

    MPI.Barrier(comm)
    root && @info "Done."
    MPI.Finalize()
end

main()
