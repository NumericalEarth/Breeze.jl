# # Differentiable Rising Thermal Bubble — Distributed Forward + Backward Benchmark
#
# Benchmarks and profiles both the forward pass and the Enzyme reverse-mode
# backward pass (∂J/∂θ₀) on a Distributed(ReactantState()) architecture,
# sharded across N GPU devices in a (N, 1, 1) partition.
#
# Uses Centered advection (not WENO) for fast compilation.
#
# For each grid size we:
#   1. Compile + profile + benchmark the forward loss
#   2. Compile + profile + benchmark the backward (Enzyme adjoint)
#   3. Report the backward-to-forward ratio — the AD overhead multiplier
#
# The backward-to-forward ratio is the key metric: it tells you how much
# more expensive gradient computation is relative to the simulation itself.
# For adjoint-based optimization and data assimilation this ratio directly
# determines the cost of each gradient step.
#
# Launch (GPU, 4 devices):
#   julia --project=test examples/forward_profiling/rising_thermal_bubble_distributed.jl
#
# Launch (CPU testing):
#   BREEZE_BACKEND=cpu julia --project=test \
#     examples/forward_profiling/rising_thermal_bubble_distributed.jl

const backend = get(ENV, "BREEZE_BACKEND", "gpu")

# Reduce BFC allocator memory fraction so XLA has room to load compiled
# programs onto all devices. Default ~0.75 leaves only ~250 MB per T4;
# 0.5 leaves ~3.5 GB headroom which avoids collective rendezvous deadlocks.
if !haskey(ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION")
    ENV["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
end

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
using Enzyme
using BenchmarkTools
using Statistics: mean, median
using Printf
using CairoMakie
using Dates

Reactant.set_default_backend(backend)
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

# Workaround: Reactant is missing Base.zero for distributed ConcretePJRTNumber.
# Without this, Enzyme.make_zero fails on models whose Clock holds a sharded scalar.
function Base.zero(x::Reactant.ConcretePJRTNumber{T,D}) where {T,D}
    return Reactant.ConcreteRNumber{T}(
        T(0); client=Reactant.XLA.client(x), device=Reactant.XLA.device(x), x.sharding
    )
end

# ── Utilities ────────────────────────────────────────────────────────────

function save_profile_log(path, compiled_fn, args...)
    open(path, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                old = (get(ENV, "LINES", nothing), get(ENV, "COLUMNS", nothing))
                ENV["LINES"] = "200000"; ENV["COLUMNS"] = "200000"
                try
                    r = Reactant.@profile compiled_fn(args...)
                    show(IOContext(io, :limit => false, :displaysize => (200000, 200000)),
                         MIME"text/plain"(), r)
                    println(io)
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

# ── Main ─────────────────────────────────────────────────────────────────

function main()
    @info "Configuration" backend
    backend == "gpu" && @info "CUDA" runtime=CUDA.runtime_version() local_toolkit=CUDA.local_toolkit

    float_type = Float32
    Oceananigans.defaults.FloatType = float_type

    # Physics
    θ_background           = float_type(300.0)
    perturbation_amplitude = float_type(2.0)
    perturbation_radius    = float_type(1000.0)
    bubble_center_x        = float_type(5000.0)
    bubble_center_y        = float_type(5000.0)
    bubble_center_z        = float_type(2000.0)
    latitude               = float_type(45.0)

    # Domain
    domain_x = float_type(10000.0)
    domain_y = float_type(10000.0)
    domain_z = float_type(10000.0)
    topology = (Periodic, Bounded, Bounded)

    # Benchmark config
    grid_sizes       = [(128, 128, 32)]
    loss_z_threshold = float_type(5000.0)
    nsteps           = 100
    bench_seconds    = 30
    bench_samples    = 10

    disable_profile = get(ENV, "BREEZE_DISABLE_REACTANT_PROFILE", "false") == "true"
    disable_bench   = get(ENV, "BREEZE_DISABLE_BENCHMARK", "false") == "true"
    disable_viz     = get(ENV, "BREEZE_DISABLE_VISUALIZATION", "false") == "true"

    run_stamp = get(ENV, "BREEZE_PROFILE_RUN_STAMP", Dates.format(now(Dates.UTC), "yyyy-mm-dd_HHMMSS"))
    run_dir   = joinpath("benchmark_results", "run_" * run_stamp, "distributed_fwd_bwd")
    mkpath(run_dir)
    @info "Results directory" run_dir

    # Architecture: distributed across all available devices
    devices  = Reactant.devices()
    ndevices = length(devices)
    arch = Distributed(ReactantState();
                       partition = Partition(ndevices, 1, 1),
                       devices = devices)
    @info "Architecture" arch ndevices

    # ── Loss function (forward pass) ─────────────────────────────────────
    # Runs nsteps of the compressible atmosphere model, then computes a
    # scalar loss: mean squared θ in the upper half of the domain.

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

    # ── Gradient function (backward pass) ────────────────────────────────
    # Enzyme reverse-mode AD through the forward loss. Returns ∂J/∂θ₀ and
    # the loss value. The shadow_model accumulates model-parameter gradients
    # (unused here but required by Enzyme's Duplicated protocol).

    function compute_gradient(model, shadow_model, θ_initial, dθ_initial, Δt, nsteps)
        parent(dθ_initial) .= 0
        _, loss_value = Enzyme.autodiff(
            Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
            forward_loss, Enzyme.Active,
            Enzyme.Duplicated(model, shadow_model),
            Enzyme.Duplicated(θ_initial, dθ_initial),
            Enzyme.Const(Δt),
            Enzyme.Const(nsteps))
        return dθ_initial, loss_value
    end

    results = Dict{Tuple{Int,Int,Int}, NamedTuple}()

    for (Nx, Ny, Nz) in grid_sizes
        GC.gc()
        @info "=" ^ 60
        @info @sprintf("Grid: %d × %d × %d  (%d cells, %d devices)", Nx, Ny, Nz, Nx*Ny*Nz, ndevices)
        @info "=" ^ 60

        grid_kwargs = (; size=(Nx, Ny, Nz),
                         x=(0, domain_x), y=(0, domain_y), z=(0, domain_z), halo=(6, 6, 6),
                         topology)

        advection = WENO(order=5)
        coriolis  = FPlane(; latitude)

        @info "Building grid + model (Centered advection, $ndevices devices)..."
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

        # Initialize fields on CPU then copy (distributed grids need this)
        cpu_grid = RectilinearGrid(CPU(); grid_kwargs...)
        θ_cpu = CenterField(cpu_grid)
        set!(θ_cpu, θ_initial_fn)

        θ_init  = CenterField(grid); copyto!(interior(θ_init), interior(θ_cpu))
        dθ_cpu  = CenterField(cpu_grid); set!(dθ_cpu, FT(0))
        dθ_init = CenterField(grid); copyto!(interior(dθ_init), interior(dθ_cpu))

        grid_label = @sprintf("%dx%dx%d", Nx, Ny, Nz)

        # ── Forward: compile, warmup, profile, benchmark ─────────────
        @info "── Forward pass ──"

        GC.gc()
        @info "Generating forward HLO (raise=true raise_first=true, $ndevices devices)..."
        t_hlo_fwd = @elapsed hlo_fwd = Reactant.@code_hlo debug=true raise=true raise_first=true forward_loss(
            model, θ_init, Δt, nsteps)
        @info @sprintf("  HLO generation time: %.2f s", t_hlo_fwd)
        hlo_path = joinpath(run_dir, "hlo_fwd_$(grid_label).txt")
        open(hlo_path, "w") do io
            show(io, hlo_fwd)
        end
        @info "Saved forward HLO → $hlo_path"

        GC.gc()
        @info "Compiling forward (raise=true raise_first=true, $ndevices devices)..."
        t_compile_fwd = @elapsed compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true forward_loss(
            model, θ_init, Δt, nsteps)
        @info @sprintf("  Compile time: %.2f s", t_compile_fwd)

        GC.gc()
        @info "Warmup forward..."
        @time loss_val = compiled_fwd(model, θ_init, Δt, nsteps)
        @info @sprintf("  Loss = %.6e", loss_val)

        if !disable_profile
            p = joinpath(run_dir, "profile_fwd_$(grid_label).log")
            @info "Profiling forward → $p"
            GC.gc()
            save_profile_log(p, compiled_fwd, model, θ_init, Δt, nsteps)
        end

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
                Label(fig[0,:], @sprintf("Forward — %d×%d×%d, %d steps, %d devices", Nx, Ny, Nz, nsteps, ndevices),
                      fontsize=16, tellwidth=false)
                ax1 = Axis(fig[1,1]; xlabel="x (m)", ylabel="z (m)", aspect=DataAspect(),
                           title="θ x-z at y=$(Int(round(yc[jc]))) m")
                heatmap!(ax1, xc, zc, temp[:,jc,:]; colormap=:thermal, colorrange=cr)
                ax2 = Axis(fig[1,3]; xlabel="y (m)", ylabel="z (m)", aspect=DataAspect(),
                           title="θ y-z at x=$(Int(round(xc[ic]))) m")
                heatmap!(ax2, yc, zc, temp[ic,:,:]; colormap=:thermal, colorrange=cr)
                @time save(joinpath(run_dir, "fwd_slices_$(grid_label).png"), fig; px_per_unit=2)
                @info "Saved forward slices"
            end
        end

        stats_fwd = nothing
        if !disable_bench
            GC.gc()
            @info "Benchmarking forward ($ndevices devices, seconds=$bench_seconds)..."
            trial_fwd = @benchmark $compiled_fwd($model, $θ_init, $Δt, $nsteps) seconds=bench_seconds samples=bench_samples
            stats_fwd = format_trial(trial_fwd)
            @info @sprintf("  Forward median: %.4f s  (n=%d)", stats_fwd.median_s, stats_fwd.nsamples)
            open(joinpath(run_dir, "bench_fwd_$(grid_label).txt"), "w") do io
                println(io, "# Forward  grid=$grid_label  nsteps=$nsteps  ndevices=$ndevices  advection=Centered(2)")
                show(io, MIME"text/plain"(), trial_fwd); println(io)
            end
        end

        # ── Backward: compile, warmup, profile, benchmark ────────────
        @info "── Backward pass (Enzyme reverse mode) ──"

        GC.gc()
        @info "Creating shadow model (Enzyme.make_zero)..."
        @time shadow_model = Enzyme.make_zero(model)

        @info "Generating backward HLO (raise=true raise_first=true, $ndevices devices)..."
        t_hlo_bwd = @elapsed hlo_bwd = Reactant.@code_hlo debug=true raise=true raise_first=true compute_gradient(
            model, shadow_model, θ_init, dθ_init, Δt, nsteps)
        @info @sprintf("  HLO generation time: %.2f s", t_hlo_bwd)
        hlo_path_bwd = joinpath(run_dir, "hlo_bwd_$(grid_label).txt")
        open(hlo_path_bwd, "w") do io
            show(io, hlo_bwd)
        end
        @info "Saved backward HLO → $hlo_path_bwd"

        GC.gc()
        @info "Compiling backward (raise=true raise_first=true, $ndevices devices)..."
        t_compile_bwd = @elapsed compiled_bwd = Reactant.@compile raise=true raise_first=true sync=true compute_gradient(
            model, shadow_model, θ_init, dθ_init, Δt, nsteps)
        @info @sprintf("  Compile time: %.2f s", t_compile_bwd)

        GC.gc()
        @info "Warmup backward..."
        @time dθ_result, bwd_loss = compiled_bwd(model, shadow_model, θ_init, dθ_init, Δt, nsteps)
        sensitivity = Array(interior(dθ_result))
        @info @sprintf("  Loss = %.6e   Max |∂J/∂θ₀| = %.6e", bwd_loss, maximum(abs, sensitivity))

        if !disable_profile
            p = joinpath(run_dir, "profile_bwd_$(grid_label).log")
            @info "Profiling backward → $p"
            GC.gc()
            save_profile_log(p, compiled_bwd, model, shadow_model, θ_init, dθ_init, Δt, nsteps)
        end

        if !disable_viz && !any(isnan, sensitivity)
            xc = range(0, domain_x, length=Nx)
            yc = range(0, domain_y, length=Ny)
            zc = range(domain_z/2Nz, domain_z - domain_z/2Nz, length=Nz)
            ic = argmin(abs.(xc .- bubble_center_x))
            jc = argmin(abs.(yc .- bubble_center_y))
            smax = maximum(abs, sensitivity)
            sr = (-smax, smax)
            fig = Figure(size=(1200, 450), fontsize=14)
            Label(fig[0,:], @sprintf("∂J/∂θ₀ — %d×%d×%d, %d steps, %d devices", Nx, Ny, Nz, nsteps, ndevices),
                  fontsize=16, tellwidth=false)
            ax1 = Axis(fig[1,1]; xlabel="x (m)", ylabel="z (m)", aspect=DataAspect(),
                       title="∂J/∂θ₀ x-z at y=$(Int(round(yc[jc]))) m")
            heatmap!(ax1, xc, zc, sensitivity[:,jc,:]; colormap=:balance, colorrange=sr)
            ax2 = Axis(fig[1,3]; xlabel="y (m)", ylabel="z (m)", aspect=DataAspect(),
                       title="∂J/∂θ₀ y-z at x=$(Int(round(xc[ic]))) m")
            heatmap!(ax2, yc, zc, sensitivity[ic,:,:]; colormap=:balance, colorrange=sr)
            @time save(joinpath(run_dir, "bwd_slices_$(grid_label).png"), fig; px_per_unit=2)
            @info "Saved backward slices"
        end

        stats_bwd = nothing
        if !disable_bench
            GC.gc()
            @info "Benchmarking backward ($ndevices devices, seconds=$bench_seconds)..."
            trial_bwd = @benchmark $compiled_bwd($model, $shadow_model, $θ_init, $dθ_init, $Δt, $nsteps) seconds=bench_seconds samples=bench_samples
            stats_bwd = format_trial(trial_bwd)
            @info @sprintf("  Backward median: %.4f s  (n=%d)", stats_bwd.median_s, stats_bwd.nsamples)
            open(joinpath(run_dir, "bench_bwd_$(grid_label).txt"), "w") do io
                println(io, "# Backward  grid=$grid_label  nsteps=$nsteps  ndevices=$ndevices  advection=Centered(2)")
                show(io, MIME"text/plain"(), trial_bwd); println(io)
            end
        end

        # ── Compare ──────────────────────────────────────────────────
        if !disable_bench && stats_fwd !== nothing && stats_bwd !== nothing
            ratio = stats_bwd.median_s / stats_fwd.median_s
            @info @sprintf("  Backward / Forward = %.2f×  (fwd %.4f s, bwd %.4f s)",
                           ratio, stats_fwd.median_s, stats_bwd.median_s)
            results[(Nx,Ny,Nz)] = (; fwd=stats_fwd.median_s, bwd=stats_bwd.median_s,
                                     ratio, ndevices, t_compile_fwd, t_compile_bwd)
        end
    end

    # ── Summary ──────────────────────────────────────────────────────────
    if !disable_bench && !isempty(results)
        @info "=" ^ 60
        @info "Summary — Distributed forward + backward ($ndevices devices, Centered)"
        @info "=" ^ 60
        for sz in grid_sizes
            haskey(results, sz) || continue
            r = results[sz]
            @info @sprintf("  %d×%d×%d  fwd: %.4f s  bwd: %.4f s  ratio: %.2f×",
                           sz..., r.fwd, r.bwd, r.ratio)
        end

        open(joinpath(run_dir, "summary.md"), "w") do io
            println(io, "# Rising Thermal Bubble — Distributed Forward + Backward")
            println(io, "\nAdvection: Centered(order=2)  |  nsteps=$nsteps  |  precision=$float_type")
            println(io, "Devices: $ndevices  |  Partition: ($ndevices, 1, 1)  |  raise=true raise_first=true")
            println(io, "\n| Grid | Forward (s) | Backward (s) | Bwd/Fwd | Compile Fwd (s) | Compile Bwd (s) |")
            println(io, "|------|-------------|--------------|---------|-----------------|-----------------|")
            for sz in grid_sizes
                haskey(results, sz) || continue
                r = results[sz]
                println(io, @sprintf("| %d×%d×%d | %.4f | %.4f | %.2f× | %.1f | %.1f |",
                                     sz..., r.fwd, r.bwd, r.ratio, r.t_compile_fwd, r.t_compile_bwd))
            end
        end
        @info "Saved summary to $(joinpath(run_dir, "summary.md"))"
    end
end

main()
