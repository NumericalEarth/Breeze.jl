# # Forward-Only Rising Thermal Bubble — Three-Way Benchmark
#
# Benchmarks the forward model under three compilation modes:
#   1) Julia GPU  — plain CUDA.jl execution (no Reactant)
#   2) Reactant (raised) — `Reactant.@compile raise=true raise_first=true sync=true`
#   3) Reactant (standard) — `Reactant.@compile sync=true` without raise flags
#
# Uses BenchmarkTools.jl for statistically rigorous timing.
# Results are saved to a timestamped run folder with per-mode subfolders:
#   benchmark_results/run_<stamp>/julia_gpu/
#   benchmark_results/run_<stamp>/reactant_raised/
#   benchmark_results/run_<stamp>/reactant_standard/

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using CUDA
using Reactant
using Reactant: @trace
using BenchmarkTools
using Statistics: mean, median, std
using Printf
using CairoMakie
using Dates

Reactant.set_default_backend("gpu")

function save_profile_log(profile_log_path, compiled_fn, args...)
    open(profile_log_path, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                old_lines = get(ENV, "LINES", nothing)
                old_columns = get(ENV, "COLUMNS", nothing)
                ENV["LINES"] = "200000"
                ENV["COLUMNS"] = "200000"
                try
                    profile_result = Reactant.@profile compiled_fn(args...)
                    ioctx = IOContext(io, :limit => false, :displaysize => (200000, 200000))
                    show(ioctx, MIME"text/plain"(), profile_result)
                    println(io)
                finally
                    old_lines === nothing ? delete!(ENV, "LINES") : (ENV["LINES"] = old_lines)
                    old_columns === nothing ? delete!(ENV, "COLUMNS") : (ENV["COLUMNS"] = old_columns)
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
    n   = length(trial)
    return (; median_s=med, mean_s=mn, min_s=lo, max_s=hi, nsamples=n)
end

function main()
    @info "CUDA runtime configuration" runtime = CUDA.runtime_version() local_toolkit = CUDA.local_toolkit

    float_type = Float32
    Oceananigans.defaults.FloatType = float_type
    @info "Using floating-point precision" float_type

    # Physical parameters
    θ_background           = float_type(300.0)
    perturbation_amplitude = float_type(2.0)
    perturbation_radius    = float_type(1000.0)
    bubble_center_x        = float_type(5000.0)
    bubble_center_y        = float_type(5000.0)
    bubble_center_z        = float_type(2000.0)
    latitude               = float_type(45.0)

    domain_x, domain_y, domain_z = float_type(10000.0), float_type(10000.0), float_type(10000.0)
    topology = (Periodic, Bounded, Bounded)

    grid_sizes       = [(128, 128, 32), (128, 128, 128), (512, 512, 32), (512, 512, 128), (1024, 1024, 32), (1024, 1024, 128)]
    loss_z_threshold = float_type(5000.0)
    nsteps           = 100
    bench_seconds    = 30
    bench_samples    = 10

    disable_reactant_profile = get(ENV, "BREEZE_DISABLE_REACTANT_PROFILE", "false") == "true"
    disable_benchmark        = get(ENV, "BREEZE_DISABLE_BENCHMARK", "false") == "true"
    disable_visualization    = get(ENV, "BREEZE_DISABLE_VISUALIZATION", "false") == "true"

    enable_reactant_profile = !disable_reactant_profile
    run_stamp = get(ENV, "BREEZE_PROFILE_RUN_STAMP", Dates.format(now(Dates.UTC), "yyyy-mm-dd_HHMMSS"))

    run_dir = joinpath("benchmark_results", "run_" * run_stamp)
    julia_dir    = joinpath(run_dir, "julia_gpu")
    raised_dir   = joinpath(run_dir, "reactant_raised")
    standard_dir = joinpath(run_dir, "reactant_standard")
    mkpath(julia_dir)
    mkpath(raised_dir)
    mkpath(standard_dir)
    @info "Results directory" run_dir

    function forward_loss_julia(model, θ_initial, Δt, nsteps)
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

    function forward_loss_reactant(model, θ_initial, Δt, nsteps)
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

    benchmark_results = Dict{Tuple{Int, Int, Int}, NamedTuple}()

    for (Nx, Ny, Nz) in grid_sizes
        GC.gc()

        @info "=" ^ 60
        @info @sprintf("Grid resolution: %d × %d × %d  (%d cells)", Nx, Ny, Nz, Nx * Ny * Nz)
        @info "=" ^ 60

        grid_kwargs = (
            size = (Nx, Ny, Nz),
            x = (0, domain_x), y = (0, domain_y), z = (0, domain_z),
            topology = topology
        )

        @info "Building Reactant and GPU grids..."
        @time begin
            reactant_grid = RectilinearGrid(ReactantState(); grid_kwargs...)
            gpu_grid      = RectilinearGrid(GPU();           grid_kwargs...)
        end
        @info "Grid floating-point types" reactant_eltype = eltype(reactant_grid) gpu_eltype = eltype(gpu_grid)

        FT = eltype(reactant_grid)

        @info "Building atmosphere models..."
        @time begin
            advection = WENO(order = 5)
            coriolis = FPlane(; latitude)
            reactant_model = AtmosphereModel(reactant_grid; advection, coriolis, dynamics = CompressibleDynamics())
            gpu_model      = AtmosphereModel(gpu_grid;      advection, coriolis, dynamics = CompressibleDynamics())
        end

        thermo = reactant_model.thermodynamic_constants

        θ_perturbation(x, y, z) = FT(perturbation_amplitude) * exp(
            -((x - bubble_center_x)^2 + (y - bubble_center_y)^2 + (z - bubble_center_z)^2)
            / perturbation_radius^2
        )
        θ_initial_fn(x, y, z) = FT(θ_background) + θ_perturbation(x, y, z)

        @info "Initializing fields..."
        @time begin
            θ_initial_reactant = CenterField(reactant_grid); set!(θ_initial_reactant, θ_initial_fn)
            θ_initial_gpu      = CenterField(gpu_grid);      set!(θ_initial_gpu,      θ_initial_fn)
        end

        Rᵈ          = thermo.molar_gas_constant / thermo.dry_air.molar_mass
        cᵖᵈ         = thermo.dry_air.heat_capacity
        γ           = cᵖᵈ / (cᵖᵈ - Rᵈ)
        sound_speed = sqrt(γ * Rᵈ * θ_background)
        Δt          = FT(0.4 * domain_x / Nx / sound_speed)
        @info @sprintf("Δt = %.6f s  (acoustic CFL ≈ 0.4, sound speed ≈ %.1f m/s)", Δt, sound_speed)

        grid_label = @sprintf("%dx%dx%d", Nx, Ny, Nz)

        # ── Mode 1: Reactant raised ──────────────────────────────────
        GC.gc()
        @info "Compiling Reactant forward pass (raised)..."
        t_compile_raised = @elapsed compiled_raised = Reactant.@compile raise=true raise_first=true sync=true forward_loss_reactant(
            reactant_model, θ_initial_reactant, Δt, nsteps
        )
        @info @sprintf("  Compile time (raised): %.2f s", t_compile_raised)

        GC.gc()
        @info "Running raised forward pass..."
        @time loss_raised = compiled_raised(reactant_model, θ_initial_reactant, Δt, nsteps)
        @info @sprintf("  Loss (raised) = %.6e", loss_raised)

        if enable_reactant_profile
            profile_path = joinpath(raised_dir, "profile_$(grid_label).log")
            @info "Profiling raised forward → $profile_path"
            GC.gc()
            save_profile_log(profile_path, compiled_raised, reactant_model, θ_initial_reactant, Δt, nsteps)
        end

        # ── Mode 2: Reactant standard (no raise) ─────────────────────
        GC.gc()
        @info "Compiling Reactant forward pass (standard, no raise)..."
        t_compile_standard = @elapsed compiled_standard = Reactant.@compile sync=true forward_loss_reactant(
            reactant_model, θ_initial_reactant, Δt, nsteps
        )
        @info @sprintf("  Compile time (standard): %.2f s", t_compile_standard)

        GC.gc()
        @info "Running standard forward pass..."
        @time loss_standard = compiled_standard(reactant_model, θ_initial_reactant, Δt, nsteps)
        @info @sprintf("  Loss (standard) = %.6e", loss_standard)

        if enable_reactant_profile
            profile_path = joinpath(standard_dir, "profile_$(grid_label).log")
            @info "Profiling standard forward → $profile_path"
            GC.gc()
            save_profile_log(profile_path, compiled_standard, reactant_model, θ_initial_reactant, Δt, nsteps)
        end

        # ── Visualization (from raised path) ──────────────────────────
        temperature = Array(interior(reactant_model.temperature))
        xc = range(0, domain_x, length = Nx)
        yc = range(0, domain_y, length = Ny)
        zc = range(domain_z / 2Nz, domain_z - domain_z / 2Nz, length = Nz)
        i_center = argmin(abs.(xc .- bubble_center_x))
        j_center = argmin(abs.(yc .- bubble_center_y))

        if !disable_visualization
            θ_deviation_max = maximum(abs, temperature .- θ_background)
            θ_range_2d = (θ_background - θ_deviation_max, θ_background + θ_deviation_max)

            fig_slices = Figure(size = (1200, 450), fontsize = 14)
            Label(fig_slices[0, :],
                  @sprintf("Forward-only rising thermal — %d×%d×%d, %d steps, Δt = %.4f s",
                           Nx, Ny, Nz, nsteps, Δt),
                  fontsize = 16, tellwidth = false)

            ax_θ_xz = Axis(fig_slices[1, 1]; xlabel = "x (m)", ylabel = "z (m)",
                           title = "θ — x-z at y = $(Int(round(yc[j_center]))) m",
                           aspect = DataAspect())
            hm_θ_xz = heatmap!(ax_θ_xz, xc, zc, temperature[:, j_center, :];
                               colormap = :thermal, colorrange = θ_range_2d)
            Colorbar(fig_slices[1, 2], hm_θ_xz; label = "θ (K)")

            ax_θ_yz = Axis(fig_slices[1, 3]; xlabel = "y (m)", ylabel = "z (m)",
                           title = "θ — y-z at x = $(Int(round(xc[i_center]))) m",
                           aspect = DataAspect())
            hm_θ_yz = heatmap!(ax_θ_yz, yc, zc, temperature[i_center, :, :];
                               colormap = :thermal, colorrange = θ_range_2d)
            Colorbar(fig_slices[1, 4], hm_θ_yz; label = "θ (K)")

            slice_file = joinpath(raised_dir, "slices_$(grid_label).png")
            @time save(slice_file, fig_slices; px_per_unit = 2)
            @info "Saved $slice_file"
        end

        # ── Benchmark all three modes using BenchmarkTools ────────────
        if !disable_benchmark
            @info "Benchmarking with BenchmarkTools (seconds=$bench_seconds, samples=$bench_samples)..."

            # Julia GPU: run + synchronize
            GC.gc()
            @info "  Benchmarking Julia GPU..."
            trial_julia = @benchmark begin
                $forward_loss_julia($gpu_model, $θ_initial_gpu, $Δt, $nsteps)
                CUDA.synchronize()
            end seconds=bench_seconds samples=bench_samples
            stats_julia = format_trial(trial_julia)
            @info @sprintf("    Julia GPU    median: %.4f s  (n=%d, min=%.4f, max=%.4f)",
                           stats_julia.median_s, stats_julia.nsamples, stats_julia.min_s, stats_julia.max_s)

            # Reactant raised: compiled with sync=true, already synchronous
            GC.gc()
            @info "  Benchmarking Reactant (raised)..."
            trial_raised = @benchmark $compiled_raised($reactant_model, $θ_initial_reactant, $Δt, $nsteps) seconds=bench_seconds samples=bench_samples
            stats_raised = format_trial(trial_raised)
            @info @sprintf("    Raised       median: %.4f s  (n=%d, min=%.4f, max=%.4f)",
                           stats_raised.median_s, stats_raised.nsamples, stats_raised.min_s, stats_raised.max_s)

            # Reactant standard: compiled with sync=true, already synchronous
            GC.gc()
            @info "  Benchmarking Reactant (standard)..."
            trial_standard = @benchmark $compiled_standard($reactant_model, $θ_initial_reactant, $Δt, $nsteps) seconds=bench_seconds samples=bench_samples
            stats_standard = format_trial(trial_standard)
            @info @sprintf("    Standard     median: %.4f s  (n=%d, min=%.4f, max=%.4f)",
                           stats_standard.median_s, stats_standard.nsamples, stats_standard.min_s, stats_standard.max_s)

            speedup_raised   = stats_julia.median_s / stats_raised.median_s
            speedup_standard = stats_julia.median_s / stats_standard.median_s

            benchmark_results[(Nx, Ny, Nz)] = (;
                time_julia    = stats_julia.median_s,
                time_raised   = stats_raised.median_s,
                time_standard = stats_standard.median_s,
                min_julia     = stats_julia.min_s,
                min_raised    = stats_raised.min_s,
                min_standard  = stats_standard.min_s,
                max_julia     = stats_julia.max_s,
                max_raised    = stats_raised.max_s,
                max_standard  = stats_standard.max_s,
                n_julia       = stats_julia.nsamples,
                n_raised      = stats_raised.nsamples,
                n_standard    = stats_standard.nsamples,
                t_compile_raised, t_compile_standard,
                speedup_raised, speedup_standard,
            )

            @info @sprintf("  Julia GPU          : %10.4f s", stats_julia.median_s)
            @info @sprintf("  Reactant (raised)  : %10.4f s  (%.2f×)", stats_raised.median_s, speedup_raised)
            @info @sprintf("  Reactant (standard): %10.4f s  (%.2f×)", stats_standard.median_s, speedup_standard)
            @info @sprintf("  Compile (raised)   : %10.2f s", t_compile_raised)
            @info @sprintf("  Compile (standard) : %10.2f s", t_compile_standard)

            # Save BenchmarkTools output
            for (trial, name, dir) in [
                (trial_julia, "julia_gpu", julia_dir),
                (trial_raised, "reactant_raised", raised_dir),
                (trial_standard, "reactant_standard", standard_dir),
            ]
                open(joinpath(dir, "bench_$(grid_label).txt"), "w") do io
                    println(io, "# BenchmarkTools result: $name  grid=$grid_label  nsteps=$nsteps")
                    show(io, MIME"text/plain"(), trial)
                    println(io)
                end
            end
        end
    end

    # ── Summary ───────────────────────────────────────────────────────
    if !disable_benchmark
        @info "=" ^ 80
        @info "Three-way forward benchmark summary (median times)"
        @info "=" ^ 80
        @info @sprintf("  %-16s  %10s  %14s  %14s  %12s  %12s",
                       "Grid", "Julia GPU", "React raised", "React std", "↑raised", "↑standard")

        for sz in grid_sizes
            r = benchmark_results[sz]
            label = @sprintf("%d×%d×%d", sz...)
            @info @sprintf("  %-16s  %8.4f s  %12.4f s  %12.4f s  %10.2f×  %10.2f×",
                           label, r.time_julia, r.time_raised, r.time_standard,
                           r.speedup_raised, r.speedup_standard)
        end

        # Bar chart: 3 bars per grid (median times)
        n = length(grid_sizes)
        xlabels = [@sprintf("%d×%d×%d", sz...) for sz in grid_sizes]
        t_julia = [benchmark_results[sz].time_julia for sz in grid_sizes]
        t_raised = [benchmark_results[sz].time_raised for sz in grid_sizes]
        t_std = [benchmark_results[sz].time_standard for sz in grid_sizes]

        xs = repeat(1:n, inner = 3)
        dodge = repeat(1:3, outer = n)
        heights = Float64[]
        for i in 1:n
            push!(heights, t_julia[i], t_raised[i], t_std[i])
        end
        bar_colors = repeat([:steelblue, :seagreen, :darkorange], outer = n)

        fig_bench = Figure(size = (1000, 520), fontsize = 14)
        ax_bench = Axis(fig_bench[1, 1];
                        xticks = (1:n, xlabels),
                        ylabel = "Wall time (s) — median",
                        title = @sprintf("Forward benchmark: Julia vs Reactant raised vs standard (nsteps=%d)", nsteps),
                        yscale = log10)

        barplot!(ax_bench, xs, heights;
                 dodge, color = bar_colors,
                 bar_labels = :y,
                 label_formatter = x -> @sprintf("%.3g s", x))

        legend_labels = ["Julia GPU", "Reactant (raised)", "Reactant (standard)"]
        legend_elements = [PolyElement(color = c) for c in [:steelblue, :seagreen, :darkorange]]
        Legend(fig_bench[1, 2], legend_elements, legend_labels; framevisible = false)

        chart_path = joinpath(run_dir, "benchmark_chart.png")
        save(chart_path, fig_bench; px_per_unit = 2)
        @info "Saved $chart_path"

        # Compile-time bar chart
        fig_compile = Figure(size = (900, 420), fontsize = 14)
        ax_compile = Axis(fig_compile[1, 1];
                          xticks = (1:n, xlabels),
                          ylabel = "Compile time (s)",
                          title = "Reactant compile time: raised vs standard")

        c_raised = [benchmark_results[sz].t_compile_raised for sz in grid_sizes]
        c_std = [benchmark_results[sz].t_compile_standard for sz in grid_sizes]
        xs_c = repeat(1:n, inner = 2)
        dodge_c = repeat(1:2, outer = n)
        heights_c = Float64[]
        for i in 1:n
            push!(heights_c, c_raised[i], c_std[i])
        end
        compile_colors = repeat([:seagreen, :darkorange], outer = n)

        barplot!(ax_compile, xs_c, heights_c;
                 dodge = dodge_c, color = compile_colors,
                 bar_labels = :y,
                 label_formatter = x -> @sprintf("%.1f s", x))

        Legend(fig_compile[1, 2],
               [PolyElement(color = c) for c in [:seagreen, :darkorange]],
               ["Raised", "Standard"]; framevisible = false)

        compile_chart_path = joinpath(run_dir, "compile_time_chart.png")
        save(compile_chart_path, fig_compile; px_per_unit = 2)
        @info "Saved $compile_chart_path"

        # Summary markdown
        summary_path = joinpath(run_dir, "summary.md")
        open(summary_path, "w") do io
            println(io, "# Rising Thermal Bubble — Three-Way Forward Benchmark")
            println(io)
            println(io, @sprintf("Run: `%s`  |  nsteps=%d  |  precision=%s  |  bench_seconds=%d",
                                 run_stamp, nsteps, float_type, bench_seconds))
            println(io)
            println(io, "All times are **median** from BenchmarkTools.jl.")
            println(io)
            println(io, "| Grid | Julia GPU (s) | React raised (s) | React std (s) | ↑raised | ↑standard | n_julia | n_raised | n_std | Compile raised (s) | Compile std (s) |")
            println(io, "|------|---------------|------------------|---------------|---------|-----------|---------|----------|-------|--------------------|-----------------| ")
            for sz in grid_sizes
                r = benchmark_results[sz]
                label = @sprintf("%d×%d×%d", sz...)
                println(io, @sprintf("| %s | %.4f | %.4f | %.4f | %.2f× | %.2f× | %d | %d | %d | %.1f | %.1f |",
                                     label, r.time_julia, r.time_raised, r.time_standard,
                                     r.speedup_raised, r.speedup_standard,
                                     r.n_julia, r.n_raised, r.n_standard,
                                     r.t_compile_raised, r.t_compile_standard))
            end
        end
        @info "Saved $summary_path"
    end
end

main()
