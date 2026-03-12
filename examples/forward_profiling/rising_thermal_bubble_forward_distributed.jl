# # Forward-Only Rising Thermal Bubble — Reactant Distributed (Sharded) Benchmark
#
# Benchmarks the forward model using Reactant with raise=true on a
# Distributed(ReactantState()) architecture (XLA/PJRT sharding across devices).
#
# Note: for sharded inputs, Reactant always forces raise=true internally
# (Compiler.jl:1921-1927), so raise=false is not meaningfully different.
#
# Launch (GPU):
#   julia --project=test examples/forward_profiling/rising_thermal_bubble_forward_distributed.jl
#
# Launch (CPU testing):
#   BREEZE_BACKEND=cpu julia --project=test \
#     examples/forward_profiling/rising_thermal_bubble_forward_distributed.jl

const backend = get(ENV, "BREEZE_BACKEND", "gpu")

let xla_flags = get(ENV, "XLA_FLAGS", "")
    if backend == "cpu"
        ndevices = parse(Int, get(ENV, "BREEZE_NDEVICES", "4"))
        if !contains(xla_flags, "xla_force_host_platform_device_count")
            xla_flags *= " --xla_force_host_platform_device_count=$ndevices"
        end
    end
    if backend == "gpu" && !contains(xla_flags, "xla_gpu_enable_command_buffer")
        xla_flags *= " --xla_gpu_enable_command_buffer="
    end
    ENV["XLA_FLAGS"] = strip(xla_flags)
end

using Breeze
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Architectures: ReactantState
using CUDA
using Reactant
using Reactant: @trace
using BenchmarkTools
using Statistics: mean, median
using Printf
using CairoMakie
using Dates

Reactant.set_default_backend(backend)

function save_profile_log(profile_log_path, compiled_fn, args...)
    open(profile_log_path, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                old = (get(ENV, "LINES", nothing), get(ENV, "COLUMNS", nothing))
                ENV["LINES"] = "200000"; ENV["COLUMNS"] = "200000"
                try
                    result = Reactant.@profile compiled_fn(args...)
                    ioctx = IOContext(io, :limit => false, :displaysize => (200000, 200000))
                    show(ioctx, MIME"text/plain"(), result); println(io)
                finally
                    old[1] === nothing ? delete!(ENV, "LINES") : (ENV["LINES"] = old[1])
                    old[2] === nothing ? delete!(ENV, "COLUMNS") : (ENV["COLUMNS"] = old[2])
                end
            end
        end
    end
end

function format_trial(trial::BenchmarkTools.Trial)
    med = median(trial).time / 1e9
    mn  = mean(trial).time / 1e9
    lo  = minimum(trial).time / 1e9
    hi  = maximum(trial).time / 1e9
    return (; median_s=med, mean_s=mn, min_s=lo, max_s=hi, nsamples=length(trial))
end

function main()
    @info "Configuration" backend
    backend == "gpu" && @info "CUDA" runtime=CUDA.runtime_version() local_toolkit=CUDA.local_toolkit

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

    disable_profile = get(ENV, "BREEZE_DISABLE_REACTANT_PROFILE", "false") == "true"
    disable_bench   = get(ENV, "BREEZE_DISABLE_BENCHMARK", "false") == "true"
    disable_viz     = get(ENV, "BREEZE_DISABLE_VISUALIZATION", "false") == "true"

    run_stamp = get(ENV, "BREEZE_PROFILE_RUN_STAMP", Dates.format(now(Dates.UTC), "yyyy-mm-dd_HHMMSS"))
    run_dir   = joinpath("benchmark_results", "run_" * run_stamp, "reactant_sharded")
    mkpath(run_dir)
    @info "Results directory" run_dir

    devices  = Reactant.devices()
    ndevices = length(devices)
    arch = Distributed(ReactantState();
                       partition = Partition(ndevices, 1, 1),
                       devices = devices)
    @info "Reactant sharded architecture" arch ndevices

    function forward_loss(model, θ_initial, Δt, nsteps)
        FT = eltype(model.grid)
        set!(model; θ = θ_initial, ρ = FT(1))
        @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
            time_step!(model, Δt)
        end
        θ_evolved = interior(model.temperature)
        k_start = ceil(Int, loss_z_threshold / domain_z * size(model.grid, 3)) + 1
        upper_θ = @view θ_evolved[:, :, k_start:end]
        return mean(upper_θ .^ 2)
    end

    results = Dict{Tuple{Int,Int,Int}, NamedTuple}()

    for (Nx, Ny, Nz) in grid_sizes
        GC.gc()
        @info "=" ^ 60
        @info @sprintf("Grid: %d × %d × %d  (%d cells, %d devices)", Nx, Ny, Nz, Nx*Ny*Nz, ndevices)
        @info "=" ^ 60

        grid_kwargs = (; size=(Nx, Ny, Nz),
                         x=(0, domain_x), y=(0, domain_y), z=(0, domain_z),
                         topology)

        advection = WENO(order = 5)
        coriolis  = FPlane(; latitude)

        @info "Building grid + model..."
        @time grid  = RectilinearGrid(arch; grid_kwargs...)
        @time model = AtmosphereModel(grid; advection, coriolis, dynamics=CompressibleDynamics())

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

        @info "Initializing fields..."
        θ_init = CenterField(grid)
        cpu_grid = RectilinearGrid(CPU(); grid_kwargs...)
        θ_cpu = CenterField(cpu_grid)
        set!(θ_cpu, θ_initial_fn)
        copyto!(interior(θ_init), interior(θ_cpu))

        grid_label = @sprintf("%dx%dx%d", Nx, Ny, Nz)

        # ── Compile ──────────────────────────────────────────────────
        GC.gc()
        @info "Compiling (raise=true raise_first=true, $ndevices devices)..."
        t_compile = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true forward_loss(
            model, θ_init, Δt, nsteps)
        @info @sprintf("  Compile time: %.2f s", t_compile)

        # ── Warmup run ───────────────────────────────────────────────
        GC.gc()
        @info "Warmup run..."
        @time loss = compiled(model, θ_init, Δt, nsteps)
        @info @sprintf("  Loss = %.6e", loss)

        # ── Profile ──────────────────────────────────────────────────
        if !disable_profile
            profile_path = joinpath(run_dir, "profile_$(grid_label).log")
            @info "Profiling → $profile_path"
            GC.gc()
            save_profile_log(profile_path, compiled, model, θ_init, Δt, nsteps)
        end

        # ── Visualization ────────────────────────────────────────────
        if !disable_viz
            temp = Array(interior(model.temperature))
            if !any(isnan, temp)
                xc = range(0, domain_x, length=Nx)
                yc = range(0, domain_y, length=Ny)
                zc = range(domain_z/2Nz, domain_z - domain_z/2Nz, length=Nz)
                ic = argmin(abs.(xc .- bubble_center_x))
                jc = argmin(abs.(yc .- bubble_center_y))
                dev = maximum(abs, temp .- θ_background)
                cr = (θ_background - dev, θ_background + dev)

                fig = Figure(size=(1200, 450), fontsize=14)
                Label(fig[0,:], @sprintf("Rising thermal — %d×%d×%d, %d steps", Nx, Ny, Nz, nsteps),
                      fontsize=16, tellwidth=false)
                ax1 = Axis(fig[1,1]; xlabel="x (m)", ylabel="z (m)", aspect=DataAspect(),
                           title="θ x-z at y=$(Int(round(yc[jc]))) m")
                heatmap!(ax1, xc, zc, temp[:,jc,:]; colormap=:thermal, colorrange=cr)
                ax2 = Axis(fig[1,3]; xlabel="y (m)", ylabel="z (m)", aspect=DataAspect(),
                           title="θ y-z at x=$(Int(round(xc[ic]))) m")
                heatmap!(ax2, yc, zc, temp[ic,:,:]; colormap=:thermal, colorrange=cr)

                slice_file = joinpath(run_dir, "slices_$(grid_label).png")
                @time save(slice_file, fig; px_per_unit=2)
                @info "Saved $slice_file"
            else
                @warn "Temperature contains NaN — skipping visualization"
            end
        end

        # ── Benchmark ────────────────────────────────────────────────
        if !disable_bench
            GC.gc()
            @info "Benchmarking ($ndevices devices, seconds=$bench_seconds, samples=$bench_samples)..."
            trial = @benchmark $compiled($model, $θ_init, $Δt, $nsteps) seconds=bench_seconds samples=bench_samples
            stats = format_trial(trial)
            @info @sprintf("  median: %.4f s  (n=%d, min=%.4f, max=%.4f)",
                           stats.median_s, stats.nsamples, stats.min_s, stats.max_s)

            results[(Nx,Ny,Nz)] = (; stats.median_s, stats.min_s, stats.max_s, stats.nsamples,
                                     t_compile, ndevices)

            open(joinpath(run_dir, "bench_$(grid_label).txt"), "w") do io
                println(io, "# Reactant sharded (raise=true)  grid=$grid_label  nsteps=$nsteps  ndevices=$ndevices")
                show(io, MIME"text/plain"(), trial); println(io)
            end
        end
    end

    # ── Summary ──────────────────────────────────────────────────────
    if !disable_bench && !isempty(results)
        @info "=" ^ 60
        @info "Summary — Reactant Distributed(ReactantState()) sharded, raise=true"
        @info "=" ^ 60
        for sz in grid_sizes
            haskey(results, sz) || continue
            r = results[sz]
            @info @sprintf("  %d×%d×%d  median: %.4f s  compile: %.1f s  (%d devices)",
                           sz..., r.median_s, r.t_compile, r.ndevices)
        end

        summary_path = joinpath(run_dir, "summary.md")
        open(summary_path, "w") do io
            println(io, "# Rising Thermal Bubble — Reactant Sharded Benchmark")
            println(io, "\nRun: `$run_stamp`  |  nsteps=$nsteps  |  precision=$float_type  |  raise=true raise_first=true")
            println(io, "\n| Grid | Devices | Median (s) | Min (s) | Max (s) | Samples | Compile (s) |")
            println(io, "|------|---------|------------|---------|---------|---------|-------------|")
            for sz in grid_sizes
                haskey(results, sz) || continue
                r = results[sz]
                println(io, @sprintf("| %d×%d×%d | %d | %.4f | %.4f | %.4f | %d | %.1f |",
                                     sz..., r.ndevices, r.median_s, r.min_s, r.max_s, r.nsamples, r.t_compile))
            end
        end
        @info "Saved $summary_path"
    end
end

main()
