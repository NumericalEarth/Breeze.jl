# # Baroclinic Wave — Weak-Scaling Forward Benchmark
#
# Measures weak scaling of the Jablonowski–Williamson baroclinic wave
# on a LatitudeLongitudeGrid with CompressibleDynamics + WENO(5).
#
# Weak scaling: specify the global grid for the max-device case (Nλ_max).
# The per-device local Nλ = Nλ_max ÷ ndevices_total stays constant as
# we scale down (partitioned along longitude via Partition(ndev, 1, 1)).
#
# The outer loop starts from the most devices (4) and works down to 1,
# so any sharding-related issues surface early.
#
# For each configuration: HLO generation → compile → warmup → profile → benchmark.
#
# Launch (GPU):
#   julia --project=test examples/forward_profiling/baroclinic_wave_distributed.jl
#
# Launch (CPU testing):
#   BREEZE_BACKEND=cpu julia --project=test \
#     examples/forward_profiling/baroclinic_wave_distributed.jl

const backend = get(ENV, "BREEZE_BACKEND", "gpu")

if !haskey(ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION")
    ENV["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
end

# XLA_FLAGS must be set BEFORE `using Reactant` — XLA reads them at init.
# xla_gpu_enable_priority_fusion is removed from the XLA proto, so it can
# only be controlled via this env var (not via @compile xla_debug_options).
let xla_flags = get(ENV, "XLA_FLAGS", "")
    if backend == "cpu"
        ndevices = parse(Int, get(ENV, "BREEZE_NDEVICES", "4"))
        if !contains(xla_flags, "xla_force_host_platform_device_count")
            xla_flags *= " --xla_force_host_platform_device_count=$ndevices"
        end
    end
    for (flag, val) in [
        "xla_gpu_enable_command_buffer"    => "",
        "xla_gpu_enable_priority_fusion"   => "false",
    ]
        if !contains(xla_flags, flag)
            xla_flags *= " --$flag=$val"
        end
    end
    ENV["XLA_FLAGS"] = strip(xla_flags)
end
@info "XLA_FLAGS = $(ENV["XLA_FLAGS"])"

using Breeze
using Oceananigans
using Oceananigans.Units
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

# ── Physics (Jablonowski–Williamson baroclinic wave) ─────────────────────

g_phys   = Float64(ThermodynamicConstants().gravitational_acceleration)
p₀       = 100_000.0
θ₀       = 300.0
N²_phys  = 1e-4
Δθ_ep    = 60.0
z_T      = 15_000.0

θᵇ(z) = θ₀ * exp(N²_phys * z / g_phys)

_constants = ThermodynamicConstants()
Rᵈ = dry_air_gas_constant(_constants)
cᵖ = _constants.dry_air.heat_capacity
κ  = Rᵈ / cᵖ
cᵥ_over_Rᵈ = (cᵖ - Rᵈ) / Rᵈ

λ_c = 90.0
φ_c = 45.0
σ_p = 10.0
Δθ_pert = 1.0

function uᵢ(λ, φ, z)
    Ω = 7.2921159e-5
    R = 6.371e6
    τ_bal = R * θ₀ * Ω / (g_phys * Δθ_ep)
    vertical_scale = ifelse(z ≤ z_T, z / 2 * (2 - z / z_T), z_T / 2)
    return (vertical_scale / τ_bal) * cosd(φ)
end

function θᵢ(λ, φ, z, H)
    θ_merid = -Δθ_ep * sind(φ) * max(0, 1 - z / z_T)
    r² = (λ - λ_c)^2 + (φ - φ_c)^2
    θ_pert = Δθ_pert * exp(-r² / (2σ_p^2)) * sin(π * z / H)
    return θᵇ(z) + θ_merid + θ_pert
end

function ρᵢ(λ, φ, z, H)
    nsteps_int = max(1, round(Int, z / 100))
    dz = z / nsteps_int
    Π = 1.0
    for n in 1:nsteps_int
        zn = (n - 1/2) * dz
        θn = θᵢ(λ, φ, zn, H)
        Π -= κ * g_phys / (Rᵈ * θn) * dz
    end
    θ = θᵢ(λ, φ, z, H)
    return p₀ * Π^cᵥ_over_Rᵈ / (Rᵈ * θ)
end

# ── Main ─────────────────────────────────────────────────────────────────

function main()
    @info "Configuration" backend
    backend == "gpu" && @info "CUDA" runtime=CUDA.runtime_version() local_toolkit=CUDA.local_toolkit

    FT = Float32
    Oceananigans.defaults.FloatType = FT

    # ── Weak-scaling configuration ────────────────────────────────────
    # Specify the GLOBAL grid for the max-device case (Nλ_max, Nφ, Nz).
    # Nλ_max must be divisible by the number of devices so that the local
    # Nλ per device stays constant as we scale down.
    Nλ_max = 512
    Nφ     = 256
    Nz     = 32

    H       = 30_000.0
    nsteps  = 100
    bench_seconds = 30
    bench_samples = 10

    disable_profile = get(ENV, "BREEZE_DISABLE_REACTANT_PROFILE", "false") == "true"
    disable_bench   = get(ENV, "BREEZE_DISABLE_BENCHMARK", "false") == "true"

    run_stamp = get(ENV, "BREEZE_PROFILE_RUN_STAMP",
                    Dates.format(now(Dates.UTC), "yyyy-mm-dd_HHMMSS"))
    summary_dir = joinpath("benchmark_results", "run_" * run_stamp, "baroclinic_wave_weak")
    mkpath(summary_dir)

    # Device counts: outer loop from most → fewest
    all_devices = Reactant.devices()
    ndevices_total = length(all_devices)
    device_counts = sort(unique([ndevices_total, ndevices_total ÷ 2, 1]); rev=true)
    filter!(n -> n ≥ 1, device_counts)

    Nλ_local = Nλ_max ÷ ndevices_total
    # @assert Nλ_max % ndevices_total == 0 "Nλ_max ($Nλ_max) must be divisible by ndevices_total ($ndevices_total)"

    # ── Closures that capture H ───────────────────────────────────────
    θᵢ_H(λ, φ, z) = θᵢ(λ, φ, z, H)
    ρᵢ_H(λ, φ, z) = ρᵢ(λ, φ, z, H)

    function forward_loss(model, θ_init, ρ_init, Δt, nsteps)
        set!(model; ρ = ρ_init)
        set!(model; θ = θ_init)
        @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
            time_step!(model, Δt)
        end
        v = model.velocities.v
        return mean(interior(v) .^ 2)
    end

    # ── Collect results ───────────────────────────────────────────────
    results = Vector{NamedTuple}()

    for ndev in device_counts
        devs = all_devices[1:ndev]
        Nλ = ndev * Nλ_local
        label = "$(ndev)dev"
        run_dir = joinpath(summary_dir, label)
        mkpath(run_dir)

        @info "=" ^ 70
        @info @sprintf("Weak scaling: %d device(s)  →  global grid %d × %d × %d  (local %d × %d × %d per device)",
                       ndev, Nλ, Nφ, Nz, Nλ_local, Nφ, Nz)
        @info "=" ^ 70

        arch = ndev > 1 ? Distributed(ReactantState();
                                       partition = Partition(ndev, 1, 1),
                                       devices = devs) :
                          ReactantState()

        GC.gc()
        grid_label = @sprintf("%dx%dx%d", Nλ, Nφ, Nz)

        grid_kwargs = (; size = (Nλ, Nφ, Nz),
                         halo = (6, 6, 6),
                         longitude = (0, 360),
                         latitude = (-85, 85),
                         z = (0, FT(H)))

        coriolis = HydrostaticSphericalCoriolis()

        γ  = cᵖ / (cᵖ - Rᵈ)
        cₛ = sqrt(γ * Rᵈ * θ₀)
        Δz = H / Nz
        Δt = FT(0.4 * Δz / cₛ)
        @info @sprintf("  Δt = %.4f s  (cₛ = %.1f m/s)", Δt, cₛ)

        @info "Building grid + model ($ndev device(s))..."
        @time grid = LatitudeLongitudeGrid(arch; grid_kwargs...)
        dynamics = CompressibleDynamics(
            ExplicitTimeStepping();
            surface_pressure = p₀,
            reference_potential_temperature = θᵇ)
        advection = WENO(order=5)
        @time model = AtmosphereModel(grid; dynamics, coriolis, advection)

        @info "Setting initial conditions..."
        cpu_grid = LatitudeLongitudeGrid(CPU(); grid_kwargs...)
        cpu_θ = CenterField(cpu_grid); set!(cpu_θ, θᵢ_H)
        cpu_ρ = CenterField(cpu_grid); set!(cpu_ρ, ρᵢ_H)

        θ_init = CenterField(grid)
        ρ_init = CenterField(grid)
        copyto!(interior(θ_init), interior(cpu_θ))
        copyto!(interior(ρ_init), interior(cpu_ρ))

        # ── HLO ──────────────────────────────────────────────────────
        GC.gc()
        @info "Generating forward HLO ($ndev device(s))..."
        t_hlo = @elapsed hlo = Reactant.@code_hlo raise=true raise_first=true forward_loss(
            model, θ_init, ρ_init, Δt, nsteps)
        @info @sprintf("  HLO generation time: %.2f s", t_hlo)
        hlo_path = joinpath(run_dir, "hlo_fwd_$(grid_label).txt")
        open(hlo_path, "w") do io; show(io, hlo); end
        @info "Saved forward HLO → $hlo_path"

        # ── Compile ──────────────────────────────────────────────────
        GC.gc()
        @info "Compiling forward ($ndev device(s))..."
        t_compile = @elapsed compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true forward_loss(
            model, θ_init, ρ_init, Δt, nsteps)
        @info @sprintf("  Compile time: %.2f s", t_compile)

        # ── Warmup ───────────────────────────────────────────────────
        GC.gc()
        @info "Warmup forward..."
        @time loss_val = compiled_fwd(model, θ_init, ρ_init, Δt, nsteps)
        @info @sprintf("  Loss (mean v²) = %.6e", loss_val)

        # ── Profile ──────────────────────────────────────────────────
        if !disable_profile
            p = joinpath(run_dir, "profile_fwd_$(grid_label).log")
            @info "Profiling forward → $p"
            GC.gc()
            save_profile_log(p, compiled_fwd, model, θ_init, ρ_init, Δt, nsteps)
        end

        # ── Benchmark ────────────────────────────────────────────────
        stats_fwd = nothing
        if !disable_bench
            GC.gc()
            @info "Benchmarking forward ($ndev device(s), seconds=$bench_seconds)..."
            trial = @benchmark $compiled_fwd($model, $θ_init, $ρ_init, $Δt, $nsteps) seconds=bench_seconds samples=bench_samples
            stats_fwd = format_trial(trial)
            @info @sprintf("  Forward median: %.4f s  (n=%d)",
                           stats_fwd.median_s, stats_fwd.nsamples)
            open(joinpath(run_dir, "bench_fwd_$(grid_label).txt"), "w") do io
                println(io, "# Forward  grid=$grid_label  nsteps=$nsteps  ndevices=$ndev")
                show(io, MIME"text/plain"(), trial); println(io)
            end
        end

        push!(results, (;
            ndevices    = ndev,
            Nλ, Nφ, Nz,
            grid_label,
            t_hlo,
            t_compile,
            loss        = Float64(loss_val),
            fwd_median  = stats_fwd === nothing ? NaN : stats_fwd.median_s,
            fwd_mean    = stats_fwd === nothing ? NaN : stats_fwd.mean_s,
            nsamples    = stats_fwd === nothing ? 0   : stats_fwd.nsamples,
        ))
    end # device_counts

    # ── Summary ──────────────────────────────────────────────────────────
    @info "=" ^ 70
    @info "Summary — Baroclinic wave weak-scaling benchmark"
    @info @sprintf("  Local grid per device: %d × %d × %d", Nλ_local, Nφ, Nz)
    @info "=" ^ 70

    for r in results
        @info @sprintf("  %d dev  grid %s  compile: %6.1f s  HLO: %6.1f s  fwd: %.4f s  (n=%d)",
                       r.ndevices, r.grid_label, r.t_compile, r.t_hlo,
                       r.fwd_median, r.nsamples)
    end

    ideal_time = results[end].fwd_median  # 1-device baseline
    if !isnan(ideal_time) && ideal_time > 0
        for r in results
            isnan(r.fwd_median) && continue
            eff = ideal_time / r.fwd_median * 100
            @info @sprintf("  %d dev → weak-scaling efficiency: %.1f%%", r.ndevices, eff)
        end
    end

    # ── Write summary.md ─────────────────────────────────────────────────
    open(joinpath(summary_dir, "summary.md"), "w") do io
        println(io, "# Baroclinic Wave — Weak-Scaling Forward Benchmark")
        println(io, "\nCompressible dynamics, WENO(5), ExplicitTimeStepping, HydrostaticSphericalCoriolis")
        println(io, @sprintf("Max global grid: %d × %d × %d  |  Local per device: %d × %d × %d  |  nsteps=%d  |  precision=%s",
                             Nλ_max, Nφ, Nz, Nλ_local, Nφ, Nz, nsteps, FT))
        println(io, "\n| Devices | Global Grid | HLO (s) | Compile (s) | Forward (s) | Samples | Efficiency | Loss |")
        println(io, "|---------|-------------|---------|-------------|-------------|---------|------------|------|")
        for r in results
            eff = (!isnan(ideal_time) && ideal_time > 0 && !isnan(r.fwd_median)) ?
                  @sprintf("%.1f%%", ideal_time / r.fwd_median * 100) : "—"
            println(io, @sprintf("| %d | %s | %.1f | %.1f | %.4f | %d | %s | %.4e |",
                                 r.ndevices, r.grid_label, r.t_hlo, r.t_compile,
                                 r.fwd_median, r.nsamples, eff, r.loss))
        end
    end
    @info "Saved summary to $(joinpath(summary_dir, "summary.md"))"

    # ── Plot ─────────────────────────────────────────────────────────────
    if !disable_bench && length(results) ≥ 2
        ndevs   = [r.ndevices for r in results]
        medians = [r.fwd_median for r in results]

        fig = Figure(size = (700, 450))
        ax = Axis(fig[1, 1];
            xlabel = "Number of devices",
            ylabel = "Forward pass time (s)",
            title  = @sprintf("Weak Scaling — Baroclinic Wave Forward\n(local %d×%d×%d per device, %d steps, %s, WENO5)",
                              Nλ_local, Nφ, Nz, nsteps, FT),
            xticks = ndevs)

        barplot!(ax, ndevs, medians; color = :steelblue, strokewidth = 1, strokecolor = :white)

        for (x, y) in zip(ndevs, medians)
            isnan(y) && continue
            eff = ideal_time / y * 100
            text!(ax, x, y; text = @sprintf("%.3fs\n(%.0f%%)", y, eff),
                  align = (:center, :bottom), fontsize = 11, offset = (0, 5))
        end

        hlines!(ax, [ideal_time]; color = :tomato, linestyle = :dash, linewidth = 1.5,
                label = @sprintf("ideal (1-dev baseline = %.3fs)", ideal_time))
        axislegend(ax; position = :lt, framevisible = false)

        plot_path = joinpath(summary_dir, "weak_scaling.png")
        save(plot_path, fig; px_per_unit = 2)
        @info "Saved plot → $plot_path"
    end
end

main()
