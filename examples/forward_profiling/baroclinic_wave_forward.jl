using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Units
using CUDA
using Reactant
using Reactant: @trace
using Statistics: mean
using Printf
using CairoMakie
using Dates

Reactant.set_default_backend("gpu")

function synchronize_device!(::Oceananigans.Architectures.CPU)
    return nothing
end

function synchronize_device!(::Oceananigans.Architectures.GPU)
    CUDA.synchronize()
    return nothing
end

function synchronize_device!(::ReactantState)
    # Reactant thunks in this script are compiled with `sync=true`, so calls
    # are already synchronized at return.
    return nothing
end

function benchmark_forward!(forward_call!, args...; warmup_steps, ntrials, arch)
    for _ in 1:warmup_steps
        forward_call!(args...)
    end

    synchronize_device!(arch)

    total_elapsed_ns = 0.0
    for _ in 1:ntrials
        t0 = time_ns()
        forward_call!(args...)
        synchronize_device!(arch)
        t1 = time_ns()
        total_elapsed_ns += (t1 - t0)
    end

    return total_elapsed_ns / ntrials / 1e9
end

function main()
    @info "CUDA runtime configuration" runtime = CUDA.runtime_version() local_toolkit = CUDA.local_toolkit

    float_type = Float32
    Oceananigans.defaults.FloatType = float_type
    @info "Using floating-point precision" float_type

    # Domain and benchmark configuration
    H = float_type(30kilometers)
    halo = (5, 5, 5)
    longitude = (float_type(0), float_type(360))
    latitude = (float_type(-85), float_type(85))
    grid_sizes = [(90, 43, 30), (180, 85, 30), (360, 170, 30)]

    nsteps = 4
    warmup_steps = 1
    ntrials = 3

    disable_reactant_profile = get(ENV, "BREEZE_DISABLE_REACTANT_PROFILE", "false") == "true"
    disable_benchmark = get(ENV, "BREEZE_DISABLE_BENCHMARK", "false") == "true"

    enable_reactant_profile = !disable_reactant_profile
    profile_run_stamp = get(ENV, "BREEZE_PROFILE_RUN_STAMP", Dates.format(now(Dates.UTC), "yyyy-mm-dd_HHMMSS"))
    profile_output_dir = joinpath("reactant_forward_profiles", "run_" * profile_run_stamp)
    profile_logs = Dict{Tuple{Int, Int, Int}, String}()
    if enable_reactant_profile
        mkpath(profile_output_dir)
    end

    # NOTE:
    # Nsight Compute orchestration is handled by
    # `examples/forward_profiling/baroclinic_wave_ncu_profile.jl`.
    # This script remains focused on benchmark timing + Reactant profiling.

    ncu_grid_idx = tryparse(Int, get(ENV, "BREEZE_NCU_GRID_IDX", ""))
    if ncu_grid_idx !== nothing
        if !(1 <= ncu_grid_idx <= length(grid_sizes))
            error("Invalid BREEZE_NCU_GRID_IDX=$(ncu_grid_idx), expected 1:$(length(grid_sizes)).")
        end
        grid_sizes = [grid_sizes[ncu_grid_idx]]
        @info "Running single-grid child pass for ncu" grid_sizes
    end

    benchmark_results = Dict{Tuple{Int, Int, Int}, NamedTuple}()

    for (Nλ, Nφ, Nz) in grid_sizes
        GC.gc()

        @info "=" ^ 60
        @info @sprintf("Grid resolution: %d × %d × %d  (%d cells)", Nλ, Nφ, Nz, Nλ * Nφ * Nz)
        @info "=" ^ 60

        grid_kwargs = (
            size = (Nλ, Nφ, Nz),
            halo = halo,
            longitude = longitude,
            latitude = latitude,
            z = (float_type(0), H),
        )

        @info "Building Reactant and GPU grids..."
        @time begin
            reactant_grid = LatitudeLongitudeGrid(ReactantState(); grid_kwargs...)
            gpu_grid = LatitudeLongitudeGrid(GPU(); grid_kwargs...)
        end
        @info "Grid floating-point types" reactant_eltype = eltype(reactant_grid) gpu_eltype = eltype(gpu_grid)

        FT = eltype(reactant_grid)

        constants = ThermodynamicConstants()
        g = FT(constants.gravitational_acceleration)
        p₀ = FT(100000)
        θ₀ = FT(300)
        N² = FT(1e-4)

        θᵇ(z) = θ₀ * exp(N² * z / g)

        coriolis = HydrostaticSphericalCoriolis()
        Ω = FT(coriolis.rotation_rate)
        Rplanet = FT(Oceananigans.defaults.planet_radius)
        Δθ_ep = FT(60)
        z_T = FT(15000)
        τ_bal = Rplanet * θ₀ * Ω / (g * Δθ_ep)

        λ_c = FT(90)
        φ_c = FT(45)
        σ = FT(10)
        Δθ = FT(1)

        function uᵢ(λ, φ, z)
            vertical_scale = ifelse(z ≤ z_T, z / FT(2) * (FT(2) - z / z_T), z_T / FT(2))
            return (vertical_scale / τ_bal) * cosd(φ)
        end

        function θᵢ(λ, φ, z)
            θ_merid = -Δθ_ep * sind(φ) * max(FT(0), FT(1) - z / z_T)
            r² = (λ - λ_c)^2 + (φ - φ_c)^2
            θ_pert = Δθ * exp(-r² / (FT(2) * σ^2)) * sin(FT(π) * z / H)
            return θᵇ(z) + θ_merid + θ_pert
        end

        Rᵈ = FT(dry_air_gas_constant(constants))
        cᵖ = FT(constants.dry_air.heat_capacity)
        κ = Rᵈ / cᵖ
        cᵥ_over_Rᵈ = (cᵖ - Rᵈ) / Rᵈ

        function ρᵢ(λ, φ, z)
            nz = max(1, round(Int, z / FT(100)))
            dz = z / nz
            Π = FT(1)
            for n in 1:nz
                zn = (n - FT(1) / FT(2)) * dz
                θn = θᵢ(λ, φ, zn)
                Π -= κ * g / (Rᵈ * θn) * dz
            end
            θ = θᵢ(λ, φ, z)
            return p₀ * Π^cᵥ_over_Rᵈ / (Rᵈ * θ)
        end

        make_dynamics() = CompressibleDynamics(ExplicitTimeStepping();
                                               surface_pressure = p₀,
                                               reference_potential_temperature = θᵇ)

        @info "Building atmosphere models..."
        @time begin
            reactant_model = AtmosphereModel(reactant_grid; dynamics = make_dynamics(), coriolis)
            gpu_model = AtmosphereModel(gpu_grid; dynamics = make_dynamics(), coriolis)
        end

        @info "Initializing fields..."
        @time begin
            ρ_initial_reactant = CenterField(reactant_grid); set!(ρ_initial_reactant, ρᵢ)
            ρ_initial_gpu = CenterField(gpu_grid); set!(ρ_initial_gpu, ρᵢ)

            θ_initial_reactant = CenterField(reactant_grid); set!(θ_initial_reactant, θᵢ)
            θ_initial_gpu = CenterField(gpu_grid); set!(θ_initial_gpu, θᵢ)
        end

        function forward_loss_julia(model, θ_initial, ρ_initial, Δt, nsteps)
            set!(model; ρ = ρ_initial)
            set!(model; u = uᵢ, θ = θ_initial)
            for _ in 1:nsteps
                time_step!(model, Δt)
            end
            v = model.velocities.v
            return mean(interior(v) .^ 2)
        end

        function forward_loss_reactant(model, θ_initial, ρ_initial, Δt, nsteps)
            set!(model; ρ = ρ_initial)
            set!(model; u = uᵢ, θ = θ_initial)
            @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
                time_step!(model, Δt)
            end
            v = model.velocities.v
            return mean(interior(v) .^ 2)
        end

        γ = cᵖ / (cᵖ - Rᵈ)
        cₛ = sqrt(γ * Rᵈ * θ₀)
        Δz = H / Nz
        Δt = FT(0.4 * Δz / cₛ)
        @info @sprintf("Δt = %.6f s  (acoustic CFL ≈ 0.4, sound speed ≈ %.1f m/s)", Δt, cₛ)

        GC.gc()
        @info "Compiling Reactant forward pass..."
        @time compiled_forward = Reactant.@compile raise=true raise_first=true sync=true forward_loss_reactant(
            reactant_model, θ_initial_reactant, ρ_initial_reactant, Δt, nsteps
        )

        GC.gc()
        @info "Running compiled forward pass once..."
        @time loss_value = compiled_forward(reactant_model, θ_initial_reactant, ρ_initial_reactant, Δt, nsteps)
        @info @sprintf("Forward loss J = %.6e", loss_value)

        if enable_reactant_profile
            profile_log_path = joinpath(profile_output_dir, @sprintf("baroclinic_profile_%dx%dx%d.log", Nλ, Nφ, Nz))
            @info "Profiling Reactant forward kernels with Reactant.@profile ..."
            @info "Saving profiling log to $profile_log_path"
            GC.gc()
            open(profile_log_path, "w") do io
                redirect_stdout(io) do
                    redirect_stderr(io) do
                        old_lines = get(ENV, "LINES", nothing)
                        old_columns = get(ENV, "COLUMNS", nothing)
                        ENV["LINES"] = "200000"
                        ENV["COLUMNS"] = "200000"
                        try
                            profile_result = Reactant.@profile compiled_forward(
                                reactant_model, θ_initial_reactant, ρ_initial_reactant, Δt, nsteps
                            )
                            ioctx = IOContext(io, :limit => false, :displaysize => (200000, 200000))
                            show(ioctx, MIME"text/plain"(), profile_result)
                            println(io)
                        finally
                            if old_lines === nothing
                                delete!(ENV, "LINES")
                            else
                                ENV["LINES"] = old_lines
                            end
                            if old_columns === nothing
                                delete!(ENV, "COLUMNS")
                            else
                                ENV["COLUMNS"] = old_columns
                            end
                        end
                    end
                end
            end
            profile_logs[(Nλ, Nφ, Nz)] = profile_log_path
        end

        if !disable_benchmark
            GC.gc()
            @info "Benchmarking forward pass (plain Julia GPU) — $ntrials trials..."
            time_forward_julia = benchmark_forward!(
                forward_loss_julia, gpu_model, θ_initial_gpu, ρ_initial_gpu, Δt, nsteps;
                warmup_steps, ntrials, arch = Oceananigans.Architectures.architecture(gpu_grid)
            )

            GC.gc()
            @info "Benchmarking forward pass (Reactant compiled GPU) — $ntrials trials..."
            time_forward_reactant = benchmark_forward!(
                compiled_forward, reactant_model, θ_initial_reactant, ρ_initial_reactant, Δt, nsteps;
                warmup_steps, ntrials, arch = Oceananigans.Architectures.architecture(reactant_grid)
            )

            forward_speedup = time_forward_julia / time_forward_reactant

            benchmark_results[(Nλ, Nφ, Nz)] = (;
                time_forward_julia,
                time_forward_reactant,
                forward_speedup,
            )

            @info @sprintf("  Forward (Julia GPU)    : %10.4f s", time_forward_julia)
            @info @sprintf("  Forward (Reactant GPU) : %10.4f s", time_forward_reactant)
            @info @sprintf("  Forward speedup        : %10.2f×", forward_speedup)
        end
    end

    if !disable_benchmark
        @info "=" ^ 60
        @info "Baroclinic wave forward-only benchmark summary"
        @info "=" ^ 60
        @info @sprintf("  %-16s  %14s  %16s  %10s", "Grid", "Julia GPU", "Reactant GPU", "Speedup")

        for sz in grid_sizes
            r = benchmark_results[sz]
            label = @sprintf("%d×%d×%d", sz...)
            @info @sprintf("  %-16s  %12.4f s  %14.4f s  %9.2f×",
                           label, r.time_forward_julia, r.time_forward_reactant, r.forward_speedup)
        end

        n = length(grid_sizes)
        xlabels = [@sprintf("%d×%d×%d", sz...) for sz in grid_sizes]
        t_julia = [benchmark_results[sz].time_forward_julia for sz in grid_sizes]
        t_react = [benchmark_results[sz].time_forward_reactant for sz in grid_sizes]

        xs = repeat(1:n, inner = 2)
        dodge = repeat(1:2, outer = n)
        heights = Float64[]
        for i in 1:n
            push!(heights, t_julia[i], t_react[i])
        end
        colors = repeat([:steelblue, :seagreen], outer = n)

        fig_bench = Figure(size = (860, 500), fontsize = 14)
        ax_bench = Axis(fig_bench[1, 1];
                        xticks = (1:n, xlabels),
                        ylabel = "Wall time (s)",
                        title = @sprintf("Baroclinic wave forward benchmark: Julia GPU vs Reactant GPU (nsteps = %d)", nsteps),
                        yscale = log10)

        barplot!(ax_bench, xs, heights;
                 dodge, color = colors,
                 bar_labels = :y,
                 label_formatter = x -> @sprintf("%.3g s", x))

        legend_labels = ["Forward (Julia GPU)", "Forward (Reactant GPU)"]
        legend_elements = [PolyElement(color = c) for c in [:steelblue, :seagreen]]
        Legend(fig_bench[1, 2], legend_elements, legend_labels; framevisible = false)

        speedup_strs = [@sprintf("%.2f×", benchmark_results[sz].forward_speedup) for sz in grid_sizes]
        summary_text = join([@sprintf("%s: speedup %s", xlabels[i], speedup_strs[i]) for i in 1:n], "   |   ")
        Label(fig_bench[2, :], summary_text; fontsize = 11, color = :gray30, tellwidth = false)

        save("baroclinic_wave_forward_benchmark.png", fig_bench; px_per_unit = 2)
        @info "Saved baroclinic_wave_forward_benchmark.png"
    end

    if enable_reactant_profile && !disable_benchmark
        profile_index_path = joinpath(profile_output_dir, "baroclinic_profile_index.md")
        open(profile_index_path, "w") do io
            println(io, "# Baroclinic Wave Forward Profiling Index")
            println(io)
            println(io, @sprintf("Run timestamp (UTC): `%s`", profile_run_stamp))
            println(io)
            println(io, "| Grid | Profile log | Julia GPU (s) | Reactant GPU (s) | Speedup |")
            println(io, "|------|-------------|---------------|------------------|---------|")
            for sz in grid_sizes
                r = benchmark_results[sz]
                log_path = get(profile_logs, sz, "N/A")
                grid_label = @sprintf("%d×%d×%d", sz...)
                println(io, @sprintf("| %s | `%s` | %.4f | %.4f | %.2f× |",
                                     grid_label, log_path,
                                     r.time_forward_julia, r.time_forward_reactant, r.forward_speedup))
            end
        end
        @info "Saved Reactant profiling index to $profile_index_path"
    end
end

main()

