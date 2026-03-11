# # Reactant Compilation Configuration Sweep — Rising Thermal Bubble
#
# Compares multiple Reactant compile configurations against each other on the
# same forward-pass workload. For each (config, grid_size) pair the script:
#   1) Dumps the HLO IR  (`@code_hlo`)
#   2) Compiles            (`Reactant.@compile`)
#   3) Benchmarks          (`BenchmarkTools.@benchmark`)
#   4) Profiles            (`Reactant.@profile`)
#
# Results are written to:
#   benchmark_results/reactant_config_sweep/run_<stamp>/<config_label>/
#     hlo_<grid>.txt          — StableHLO IR
#     bench_<grid>.txt        — BenchmarkTools trial
#     profile_<grid>.log      — Reactant profiler output
#   benchmark_results/reactant_config_sweep/run_<stamp>/
#     summary.md              — markdown table of all results
#     benchmark_chart.png     — grouped bar chart (median wall time)
#     compile_time_chart.png  — grouped bar chart (compile time)

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using CUDA
using Reactant
using Reactant: @trace, CompileOptions
using BenchmarkTools
using Statistics: mean, median
using Printf
using CairoMakie
using Dates

Reactant.set_default_backend("gpu")

# ─── Configurable section ─────────────────────────────────────────────────────
# Each entry is (label, CompileOptions).  Tweak / extend this list freely.
CONFIGS = [
    (
        label = "raised",
        opts  = CompileOptions(;
            raise = true, raise_first = true, sync = true,
        ),
    ),
    (
        label = "standard",
        opts  = CompileOptions(;
            sync = true,
        ),
    ),
    (
        label = "cudnn_hlo_optimize",
        opts  = CompileOptions(;
            raise = true, raise_first = true, sync = true,
            cudnn_hlo_optimize = true,
        ),
    ),
    (
    label = "raised_exhaustive_tiling",
    opts  = CompileOptions(;
        raise = true, raise_first = true, sync = true,
        xla_debug_options = (;
            xla_gpu_exhaustive_tiling_search = true,
        ),
    ),
    ),
    (
    label = "raised_block_rewriter",
    opts  = CompileOptions(;
        raise = true, raise_first = true, sync = true,
        xla_debug_options = (;
            xla_gpu_experimental_enable_fusion_block_level_rewriter = true,
        ),
    ),
    ),
]

GRID_SIZES = [
    (512, 512, 32),
]

NSTEPS          = 100
BENCH_SECONDS   = 30
BENCH_SAMPLES   = 10
FLOAT_TYPE      = Float32
# ─── End configurable section ─────────────────────────────────────────────────

function save_profile_log(path, compiled_fn, args...)
    open(path, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                old_lines   = get(ENV, "LINES", nothing)
                old_columns = get(ENV, "COLUMNS", nothing)
                ENV["LINES"]   = "200000"
                ENV["COLUMNS"] = "200000"
                try
                    result = Reactant.@profile compiled_fn(args...)
                    ioctx = IOContext(io, :limit => false, :displaysize => (200000, 200000))
                    show(ioctx, MIME"text/plain"(), result)
                    println(io)
                finally
                    old_lines   === nothing ? delete!(ENV, "LINES")   : (ENV["LINES"]   = old_lines)
                    old_columns === nothing ? delete!(ENV, "COLUMNS") : (ENV["COLUMNS"] = old_columns)
                end
            end
        end
    end
end

function format_trial(trial::BenchmarkTools.Trial)
    med = median(trial).time / 1e9
    mn  = mean(trial).time   / 1e9
    lo  = minimum(trial).time / 1e9
    hi  = maximum(trial).time / 1e9
    n   = length(trial)
    return (; median_s = med, mean_s = mn, min_s = lo, max_s = hi, nsamples = n)
end

function main()
    float_type = FLOAT_TYPE
    Oceananigans.defaults.FloatType = float_type

    # Physical parameters
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

    loss_z_threshold = float_type(5000.0)
    nsteps           = NSTEPS

    disable_profile   = get(ENV, "BREEZE_DISABLE_REACTANT_PROFILE", "false") == "true"
    disable_benchmark = get(ENV, "BREEZE_DISABLE_BENCHMARK",        "false") == "true"
    disable_hlo       = get(ENV, "BREEZE_DISABLE_HLO",              "false") == "true"

    run_stamp = get(ENV, "BREEZE_PROFILE_RUN_STAMP",
                    Dates.format(now(Dates.UTC), "yyyy-mm-dd_HHMMSS"))

    run_dir = joinpath("benchmark_results", "reactant_config_sweep", "run_" * run_stamp)
    config_dirs = Dict{String, String}()
    for cfg in CONFIGS
        d = joinpath(run_dir, cfg.label)
        mkpath(d)
        config_dirs[cfg.label] = d
    end
    @info "Results directory" run_dir

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

    # Store results keyed by (config_label, grid_tuple)
    all_results = Dict{Tuple{String, Tuple{Int,Int,Int}}, NamedTuple}()

    for (Nx, Ny, Nz) in GRID_SIZES
        GC.gc()
        grid_label = @sprintf("%dx%dx%d", Nx, Ny, Nz)

        @info "=" ^ 70
        @info @sprintf("Grid: %s  (%d cells)", grid_label, Nx * Ny * Nz)
        @info "=" ^ 70

        grid_kwargs = (
            size = (Nx, Ny, Nz),
            x = (0, domain_x), y = (0, domain_y), z = (0, domain_z),
            topology = topology,
        )

        @info "Building Reactant grid + model..."
        @time begin
            reactant_grid  = RectilinearGrid(ReactantState(); grid_kwargs...)
            advection      = WENO(order = 5)
            coriolis       = FPlane(; latitude)
            reactant_model = AtmosphereModel(reactant_grid;
                                             advection, coriolis,
                                             dynamics = CompressibleDynamics())
        end

        FT = eltype(reactant_grid)
        thermo = reactant_model.thermodynamic_constants

        θ_perturbation(x, y, z) = FT(perturbation_amplitude) * exp(
            -((x - bubble_center_x)^2 + (y - bubble_center_y)^2 +
              (z - bubble_center_z)^2) / perturbation_radius^2
        )
        θ_initial_fn(x, y, z) = FT(θ_background) + θ_perturbation(x, y, z)

        @info "Initializing θ field..."
        θ_initial_reactant = CenterField(reactant_grid)
        set!(θ_initial_reactant, θ_initial_fn)

        Rᵈ          = thermo.molar_gas_constant / thermo.dry_air.molar_mass
        cᵖᵈ         = thermo.dry_air.heat_capacity
        γ           = cᵖᵈ / (cᵖᵈ - Rᵈ)
        sound_speed = sqrt(γ * Rᵈ * θ_background)
        Δt          = FT(0.4 * domain_x / Nx / sound_speed)

        @info @sprintf("Δt = %.6f s  (sound speed ≈ %.1f m/s)", Δt, sound_speed)

        # ── Loop over configurations ──────────────────────────────────
        for cfg in CONFIGS
            label = cfg.label
            opts  = cfg.opts
            dir   = config_dirs[label]

            @info "-" ^ 50
            @info "  Config: $label"

            # 1) HLO IR
            if !disable_hlo
                @info "  Capturing HLO IR..."
                hlo_path = joinpath(dir, "hlo_$(grid_label).txt")
                try
                    hlo = Reactant.@code_hlo compile_options=opts forward_loss_reactant(
                        reactant_model, θ_initial_reactant, Δt, nsteps
                    )
                    open(hlo_path, "w") do io
                        print(io, String(hlo))
                    end
                    @info "    → $hlo_path"
                catch e
                    @warn "  @code_hlo failed for $label" exception = (e, catch_backtrace())
                    open(hlo_path, "w") do io
                        println(io, "# @code_hlo failed")
                        println(io, sprint(showerror, e))
                    end
                end
            end

            # 2) Compile
            GC.gc()
            @info "  Compiling..."
            t_compile = @elapsed compiled = Reactant.@compile compile_options=opts forward_loss_reactant(
                reactant_model, θ_initial_reactant, Δt, nsteps
            )
            @info @sprintf("    Compile time: %.2f s", t_compile)

            # Quick sanity run
            GC.gc()
            @info "  Running forward pass..."
            @time loss_val = compiled(reactant_model, θ_initial_reactant, Δt, nsteps)
            @info @sprintf("    Loss = %.6e", loss_val)

            # 3) Profile
            if !disable_profile
                profile_path = joinpath(dir, "profile_$(grid_label).log")
                @info "  Profiling → $profile_path"
                GC.gc()
                save_profile_log(profile_path, compiled,
                                 reactant_model, θ_initial_reactant, Δt, nsteps)
            end

            # 4) Benchmark
            stats = nothing
            if !disable_benchmark
                @info "  Benchmarking (seconds=$BENCH_SECONDS, samples=$BENCH_SAMPLES)..."
                GC.gc()
                trial = @benchmark $compiled(
                    $reactant_model, $θ_initial_reactant, $Δt, $nsteps
                ) seconds=BENCH_SECONDS samples=BENCH_SAMPLES

                stats = format_trial(trial)
                @info @sprintf("    Median: %.4f s  (n=%d, min=%.4f, max=%.4f)",
                               stats.median_s, stats.nsamples, stats.min_s, stats.max_s)

                bench_path = joinpath(dir, "bench_$(grid_label).txt")
                open(bench_path, "w") do io
                    println(io, "# BenchmarkTools: config=$label  grid=$grid_label  nsteps=$nsteps")
                    show(io, MIME"text/plain"(), trial)
                    println(io)
                end
            end

            all_results[(label, (Nx, Ny, Nz))] = (;
                t_compile,
                loss = Float64(loss_val),
                median_s  = stats === nothing ? NaN : stats.median_s,
                min_s     = stats === nothing ? NaN : stats.min_s,
                max_s     = stats === nothing ? NaN : stats.max_s,
                nsamples  = stats === nothing ? 0   : stats.nsamples,
            )
        end
    end

    # ── Summary ───────────────────────────────────────────────────────────────
    config_labels = [cfg.label for cfg in CONFIGS]
    n_configs = length(config_labels)
    n_grids   = length(GRID_SIZES)

    @info "=" ^ 80
    @info "Reactant configuration sweep summary"
    @info "=" ^ 80

    header = @sprintf("  %-30s  %10s  %10s  %10s  %8s", "Config / Grid", "Compile(s)", "Median(s)", "Min(s)", "Samples")
    @info header
    for (Nx, Ny, Nz) in GRID_SIZES
        gl = @sprintf("%d×%d×%d", Nx, Ny, Nz)
        @info "  Grid: $gl"
        for label in config_labels
            r = all_results[(label, (Nx, Ny, Nz))]
            @info @sprintf("    %-26s  %10.2f  %10.4f  %10.4f  %8d",
                           label, r.t_compile, r.median_s, r.min_s, r.nsamples)
        end
    end

    if !disable_benchmark
        # ── Bar chart: median wall time per config per grid ───────────────
        grid_labels = [@sprintf("%d×%d×%d", sz...) for sz in GRID_SIZES]

        xs      = Int[]
        dodges  = Int[]
        heights = Float64[]
        colors  = Symbol[]
        palette = [:steelblue, :seagreen, :darkorange, :firebrick, :mediumpurple,
                   :goldenrod, :hotpink, :teal, :slategray, :crimson]

        for (gi, sz) in enumerate(GRID_SIZES)
            for (ci, label) in enumerate(config_labels)
                push!(xs, gi)
                push!(dodges, ci)
                push!(heights, all_results[(label, sz)].median_s)
                push!(colors, palette[mod1(ci, length(palette))])
            end
        end

        fig_bench = Figure(size = (max(900, 200 * n_grids), 520), fontsize = 14)
        ax_bench = Axis(fig_bench[1, 1];
                        xticks = (1:n_grids, grid_labels),
                        ylabel = "Wall time (s) — median",
                        title  = @sprintf("Reactant config sweep  (nsteps=%d, %s)", nsteps, float_type),
                        yscale = log10)

        barplot!(ax_bench, xs, heights;
                 dodge = dodges, color = colors,
                 bar_labels = :y,
                 label_formatter = x -> @sprintf("%.3g", x))

        legend_elems = [PolyElement(color = palette[mod1(ci, length(palette))])
                        for ci in 1:n_configs]
        Legend(fig_bench[1, 2], legend_elems, config_labels; framevisible = false)

        chart_path = joinpath(run_dir, "benchmark_chart.png")
        save(chart_path, fig_bench; px_per_unit = 2)
        @info "Saved $chart_path"

        # ── Compile-time chart ────────────────────────────────────────────
        xs_c      = Int[]
        dodges_c  = Int[]
        heights_c = Float64[]
        colors_c  = Symbol[]
        for (gi, sz) in enumerate(GRID_SIZES)
            for (ci, label) in enumerate(config_labels)
                push!(xs_c, gi)
                push!(dodges_c, ci)
                push!(heights_c, all_results[(label, sz)].t_compile)
                push!(colors_c, palette[mod1(ci, length(palette))])
            end
        end

        fig_ct = Figure(size = (max(900, 200 * n_grids), 520), fontsize = 14)
        ax_ct = Axis(fig_ct[1, 1];
                     xticks = (1:n_grids, grid_labels),
                     ylabel = "Compile time (s)",
                     title  = "Reactant compile time by configuration")

        barplot!(ax_ct, xs_c, heights_c;
                 dodge = dodges_c, color = colors_c,
                 bar_labels = :y,
                 label_formatter = x -> @sprintf("%.1f", x))

        Legend(fig_ct[1, 2], legend_elems, config_labels; framevisible = false)

        compile_chart_path = joinpath(run_dir, "compile_time_chart.png")
        save(compile_chart_path, fig_ct; px_per_unit = 2)
        @info "Saved $compile_chart_path"
    end

    # ── Markdown summary ──────────────────────────────────────────────────
    summary_path = joinpath(run_dir, "summary.md")
    open(summary_path, "w") do io
        println(io, "# Reactant Configuration Sweep — Rising Thermal Bubble")
        println(io)
        println(io, @sprintf("Run: `%s`  |  nsteps=%d  |  precision=%s  |  bench_seconds=%d",
                             run_stamp, nsteps, float_type, BENCH_SECONDS))
        println(io)

        # Config descriptions
        println(io, "## Configurations\n")
        for cfg in CONFIGS
            o = cfg.opts
            flags = String[]
            o.raise       && push!(flags, "raise")
            o.raise_first && push!(flags, "raise_first")
            o.sync        && push!(flags, "sync")
            o.no_nan      && push!(flags, "no_nan")
            o.all_finite  && push!(flags, "all_finite")
            o.disable_licm_optimization_passes              && push!(flags, "disable_licm")
            o.disable_scatter_gather_optimization_passes    && push!(flags, "disable_scatter_gather")
            o.disable_reduce_slice_fusion_passes            && push!(flags, "disable_reduce_slice_fusion")
            o.disable_pad_optimization_passes               && push!(flags, "disable_pad")
            o.disable_loop_raising_passes                   && push!(flags, "disable_loop_raising")
            o.disable_slice_to_batch_passes                 && push!(flags, "disable_slice_to_batch")
            o.disable_concat_to_batch_passes                && push!(flags, "disable_concat_to_batch")
            println(io, "- **$(cfg.label)**: $(join(flags, ", "))")
        end
        println(io)

        # Results table
        println(io, "## Results\n")
        println(io, "All times are **median** from BenchmarkTools.jl.\n")

        header_cols = ["Grid", [l for l in config_labels]...,
                       ["Compile " * l for l in config_labels]...]
        println(io, "| " * join(header_cols, " | ") * " |")
        println(io, "| " * join(fill("---", length(header_cols)), " | ") * " |")

        for sz in GRID_SIZES
            gl = @sprintf("%d×%d×%d", sz...)
            row = [gl]
            for label in config_labels
                r = all_results[(label, sz)]
                push!(row, @sprintf("%.4f s", r.median_s))
            end
            for label in config_labels
                r = all_results[(label, sz)]
                push!(row, @sprintf("%.1f s", r.t_compile))
            end
            println(io, "| " * join(row, " | ") * " |")
        end
    end
    @info "Saved $summary_path"

    @info "Done."
end

main()
