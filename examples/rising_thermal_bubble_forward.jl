# # Forward-Only Rising Thermal Bubble — Julia GPU vs Reactant GPU
#
# This example is a forward-only companion to `rising_thermal_bubble.jl`.
# It removes all adjoint/sensitivity computations and benchmarks only the
# forward model on:
#   1) plain Julia GPU execution
#   2) Reactant-compiled GPU execution
#
# Benchmark timing follows the same pattern used in `benchmarking/src/utils.jl`:
# warmup, explicit device synchronization, and `time_ns` wall-clock timing.
#
# Optional kernel-level profiling is available via `Reactant.@profile` for the
# compiled forward path (GPU/TPU). Toggle with `enable_reactant_profile` below.

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
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

"""
    benchmark_forward!(forward_call!, args...; warmup_steps, ntrials, arch)

Benchmark a forward call using warmup + explicit synchronization, returning the
mean trial time in seconds.
"""
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

    # Physical parameters
    θ_background          = 300.0    # [K]
    perturbation_amplitude = 2.0     # [K]
    perturbation_radius    = 1000.0  # [m]
    bubble_center_x        = 5000.0  # [m]
    bubble_center_y        = 5000.0  # [m]
    bubble_center_z        = 2000.0  # [m]
    latitude               = 45.0    # [degrees] for f-plane Coriolis

    # Domain and benchmark configuration
    domain_x, domain_y, domain_z = 10000.0, 10000.0, 10000.0
    topology = (Periodic, Bounded, Bounded)

    grid_sizes       = [(128, 128, 128), (256, 256, 128), (512, 512, 128)]
    loss_z_threshold = 5000.0
    nsteps           = 2500
    warmup_steps     = 1
    ntrials          = 3

    # Optional Reactant kernel profiling.
    # When enabled, each grid is profiled before benchmarking and logs are saved
    # to per-grid files for side-by-side comparison.
    enable_reactant_profile = true
    profile_run_stamp = Dates.format(now(Dates.UTC), "yyyy-mm-dd_HHMMSS")
    profile_output_dir = joinpath("reactant_forward_profiles", "run_" * profile_run_stamp)
    profile_logs = Dict{Tuple{Int, Int, Int}, String}()
    if enable_reactant_profile
        mkpath(profile_output_dir)
    end

    # Forward objective for plain Julia path (no Reactant tracing)
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

    # Forward objective for Reactant path (tracing enabled for compilation)
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

        FT = eltype(reactant_grid)

        @info "Building atmosphere models..."
        @time begin
            advection = WENO(order = 5)
            coriolis = FPlane(; latitude)
            reactant_model = AtmosphereModel(
                reactant_grid;
                advection,
                coriolis,
                dynamics = CompressibleDynamics(),
            )
            gpu_model = AtmosphereModel(
                gpu_grid;
                advection,
                coriolis,
                dynamics = CompressibleDynamics(),
            )
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

        # Acoustic CFL for explicit compressible dynamics
        Rᵈ          = thermo.molar_gas_constant / thermo.dry_air.molar_mass
        cᵖᵈ         = thermo.dry_air.heat_capacity
        γ           = cᵖᵈ / (cᵖᵈ - Rᵈ)
        sound_speed = sqrt(γ * Rᵈ * θ_background)
        Δt          = FT(0.4 * domain_x / Nx / sound_speed)
        @info @sprintf("Δt = %.6f s  (acoustic CFL ≈ 0.4, sound speed ≈ %.1f m/s)", Δt, sound_speed)

        GC.gc()
        @info "Compiling Reactant forward pass..."
        @time compiled_forward = Reactant.@compile raise=true raise_first=true sync=true forward_loss_reactant(
            reactant_model, θ_initial_reactant, Δt, nsteps
        )

        # Forward-only state for visualization
        GC.gc()
        @info "Running compiled forward pass for evolved temperature..."
        @time loss_value = compiled_forward(reactant_model, θ_initial_reactant, Δt, nsteps)
        temperature = Array(interior(reactant_model.temperature))

        @info @sprintf("Forward loss J = %.6e", loss_value)

        # Coordinates
        xc = range(0, domain_x, length = Nx)
        yc = range(0, domain_y, length = Ny)
        zc = range(domain_z / 2Nz, domain_z - domain_z / 2Nz, length = Nz)
        i_center = argmin(abs.(xc .- bubble_center_x))
        j_center = argmin(abs.(yc .- bubble_center_y))

        # 2D theta slices
        θ_deviation_max = maximum(abs, temperature .- θ_background)
        θ_range_2d = (θ_background - θ_deviation_max, θ_background + θ_deviation_max)

        fig_slices = Figure(size = (1200, 450), fontsize = 14)
        Label(fig_slices[0, :],
              @sprintf("Forward-only rising thermal — %d×%d×%d, %d steps, Δt = %.4f s, J = %.6e",
                       Nx, Ny, Nz, nsteps, Δt, loss_value),
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

        slice_file = @sprintf("rising_thermal_forward_%dx%dx%d.png", Nx, Ny, Nz)
        @time save(slice_file, fig_slices; px_per_unit = 2)
        @info "Saved $slice_file"

        # 3D theta cut-away with visible-face-only color normalization
        top_face_θ   = temperature[:, :, end]
        front_face_θ = temperature[:, 1, :]
        left_face_θ  = temperature[1, :, :]

        visible_θ_max = max(maximum(abs, top_face_θ   .- θ_background),
                            maximum(abs, front_face_θ .- θ_background),
                            maximum(abs, left_face_θ  .- θ_background))
        θ_range_3d = (θ_background - visible_θ_max, θ_background + visible_θ_max)

        top_X   = [xc[i] for i in 1:Nx, j in 1:Ny]
        top_Y   = [yc[j] for i in 1:Nx, j in 1:Ny]
        top_Z   = fill(zc[end], Nx, Ny)
        front_X = [xc[i] for i in 1:Nx, k in 1:Nz]
        front_Y = fill(yc[1], Nx, Nz)
        front_Z = [zc[k] for i in 1:Nx, k in 1:Nz]
        left_X  = fill(xc[1], Ny, Nz)
        left_Y  = [yc[j] for j in 1:Ny, k in 1:Nz]
        left_Z  = [zc[k] for j in 1:Ny, k in 1:Nz]

        fig_3d = Figure(size = (800, 600), fontsize = 14)
        Label(fig_3d[0, :],
              @sprintf("Forward-only 3D cut-away — %d×%d×%d", Nx, Ny, Nz),
              fontsize = 16, tellwidth = false)

        ax_3d_θ = Axis3(fig_3d[1, 1]; xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)",
                        title = "Potential temperature θ", aspect = :data,
                        azimuth = 1.20π, elevation = 0.30)

        surface!(ax_3d_θ, top_X, top_Y, top_Z;
                 color = top_face_θ, colormap = :thermal,
                 colorrange = θ_range_3d, shading = NoShading)
        surface!(ax_3d_θ, front_X, front_Y, front_Z;
                 color = front_face_θ, colormap = :thermal,
                 colorrange = θ_range_3d, shading = NoShading)
        surf_θ = surface!(ax_3d_θ, left_X, left_Y, left_Z;
                          color = left_face_θ, colormap = :thermal,
                          colorrange = θ_range_3d, shading = NoShading)
        Colorbar(fig_3d[1, 2], surf_θ; label = "θ (K)")

        cube_file = @sprintf("rising_thermal_forward_3d_%dx%dx%d.png", Nx, Ny, Nz)
        @time save(cube_file, fig_3d; px_per_unit = 2)
        @info "Saved $cube_file"

        # Optional Reactant kernel profiling (before benchmarking)
        if enable_reactant_profile
            profile_log_path = joinpath(profile_output_dir, @sprintf("profile_%dx%dx%d.log", Nx, Ny, Nz))
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
                                reactant_model, θ_initial_reactant, Δt, nsteps
                            )
                            ioctx = IOContext(
                                io,
                                :limit => false,
                                :displaysize => (200000, 200000),
                            )
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
            profile_logs[(Nx, Ny, Nz)] = profile_log_path
        end

        # Forward benchmark only: Julia GPU vs Reactant GPU
        GC.gc()
        @info "Benchmarking forward pass (plain Julia GPU) — $ntrials trials..."
        time_forward_julia = benchmark_forward!(
            forward_loss_julia, gpu_model, θ_initial_gpu, Δt, nsteps;
            warmup_steps, ntrials, arch = Oceananigans.Architectures.architecture(gpu_grid)
        )

        GC.gc()
        @info "Benchmarking forward pass (Reactant compiled GPU) — $ntrials trials..."
        time_forward_reactant = benchmark_forward!(
            compiled_forward, reactant_model, θ_initial_reactant, Δt, nsteps;
            warmup_steps, ntrials, arch = Oceananigans.Architectures.architecture(reactant_grid)
        )

        forward_speedup = time_forward_julia / time_forward_reactant

        benchmark_results[(Nx, Ny, Nz)] = (;
            time_forward_julia,
            time_forward_reactant,
            forward_speedup,
        )

        @info @sprintf("  Forward (Julia GPU)    : %10.4f s", time_forward_julia)
        @info @sprintf("  Forward (Reactant GPU) : %10.4f s", time_forward_reactant)
        @info @sprintf("  Forward speedup        : %10.2f×", forward_speedup)
    end

    # Summary table
    @info "=" ^ 60
    @info "Forward-only benchmark summary"
    @info "=" ^ 60
    @info @sprintf("  %-16s  %14s  %16s  %10s", "Grid", "Julia GPU", "Reactant GPU", "Speedup")

    for sz in grid_sizes
        r = benchmark_results[sz]
        label = @sprintf("%d×%d×%d", sz...)
        @info @sprintf("  %-16s  %12.4f s  %14.4f s  %9.2f×",
                       label, r.time_forward_julia, r.time_forward_reactant, r.forward_speedup)
    end

    # Summary chart
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
                    title = "Forward-only benchmark: Julia GPU vs Reactant GPU",
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

    save("rising_thermal_forward_benchmark.png", fig_bench; px_per_unit = 2)
    @info "Saved rising_thermal_forward_benchmark.png"

    if enable_reactant_profile
        profile_index_path = joinpath(profile_output_dir, "profile_index.md")
        open(profile_index_path, "w") do io
            println(io, "# Reactant Forward Profiling Index")
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

