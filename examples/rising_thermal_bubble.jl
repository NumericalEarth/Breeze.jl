# # Differentiable Rising Thermal Bubble — Multi-Resolution Benchmark
#
# This example demonstrates how [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)
# and [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) can be combined to compute
# **fast, exact gradients** through a compressible atmosphere simulation.
#
# A warm Gaussian bubble in a neutrally stratified atmosphere rises under buoyancy,
# rolls up into a vortex ring, and eventually breaks into turbulence.  We define a
# scalar loss function on the evolved temperature field and differentiate it with
# respect to the initial potential-temperature perturbation using reverse-mode AD.
#
# Reactant compiles *both* the forward model and the Enzyme-generated adjoint code
# into fused XLA/StableHLO kernels.  This gives two advantages:
#
# 1. **Speed** — the compiled forward pass is significantly faster than plain Julia.
# 2. **Adjoint support** — Enzyme's reverse mode is compiled through the same path,
#    producing an efficient backward pass at a modest multiple of the forward cost.
#
# We benchmark three quantities across multiple grid resolutions:
#
# | Benchmark | Description |
# |-----------|-------------|
# | Forward (plain Julia) | uncompiled forward pass on `GPU()` |
# | Forward (Reactant)    | compiled forward pass on `ReactantState` |
# | Backward (Reactant)   | compiled adjoint (Enzyme reverse mode) on `ReactantState` |
#
# The backward-to-forward ratio quantifies the overhead of AD relative to the
# simulation itself — a key metric for adjoint-based optimization and data assimilation.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using Printf
using CairoMakie
using CUDA

Reactant.set_default_backend("gpu")

# ## Physical parameters
#
# A neutrally stratified dry atmosphere with a localised warm perturbation.

θ_background          = 300.0    # background potential temperature [K]
surface_pressure      = 101325.0 # surface pressure [Pa]
standard_pressure     = 1e5      # standard pressure [Pa]

perturbation_amplitude = 2.0     # warm bubble amplitude [K]
perturbation_radius    = 1000.0  # warm bubble e-folding radius [m]
bubble_center_x        = 5000.0  # bubble centre x [m]
bubble_center_y        = 5000.0  # bubble centre y [m]
bubble_center_z        = 2000.0  # bubble centre z [m]

# ## Domain, grid resolutions, and benchmark configuration
#
# The domain is a 10 km cube.  We sweep over several grid resolutions to study
# how wall-clock time scales with problem size.  For each resolution we create
# two grids: one on `ReactantState` for Reactant-compiled execution and one on
# `GPU()` for the plain-Julia baseline.

domain_x, domain_y, domain_z = 10000.0, 10000.0, 10000.0

topology = (Periodic, Bounded, Bounded)

grid_sizes       = [(32, 32, 32), (64, 64, 32), (128, 128, 32)]
loss_z_threshold = 5000.0
nsteps           = 2500
nwarmup          = 1
ntrials          = 3

# ## Loss function and adjoint
#
# The loss is the mean squared potential temperature in the upper half of the
# domain (above `loss_z_threshold`).  This is a simple proxy for "how much
# warm air has risen".
#
# `compute_gradient` wraps `Enzyme.autodiff` in reverse mode to compute
# ∂J/∂θ₀ — the sensitivity of the loss to the initial temperature field.
#
# Both functions are defined once and reused for every grid resolution.

function loss(model, θ_initial, Δt, nsteps)
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

function compute_gradient(model, shadow_model, θ_initial, dθ_initial, Δt, nsteps)
    parent(dθ_initial) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, shadow_model),
        Enzyme.Duplicated(θ_initial, dθ_initial),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dθ_initial, loss_value
end

# ## Multi-resolution loop
#
# For each grid resolution we:
#
# 1. Build `ReactantState` and `GPU()` grids and models.
# 2. Set initial conditions (Gaussian warm bubble).
# 3. Compile the forward and backward passes with Reactant.
# 4. Run the forward model and compute adjoint sensitivities.
# 5. Visualise evolved temperature and ∂J/∂θ₀ as 2-D slices and 3-D cut-aways.
# 6. Benchmark: plain-Julia forward, Reactant forward, Reactant backward.
#
# Timing results are collected into `benchmark_results` for the summary plot.

benchmark_results = Dict{Tuple{Int,Int,Int}, NamedTuple}()

for (Nx, Ny, Nz) in grid_sizes

    GC.gc()

    @info "=" ^ 60
    @info @sprintf("Grid resolution: %d × %d × %d  (%d cells)", Nx, Ny, Nz, Nx * Ny * Nz)
    @info "=" ^ 60

    # ── Grids and models ──────────────────────────────────────────────────

    grid_kwargs = (size = (Nx, Ny, Nz),
                   x = (0, domain_x), y = (0, domain_y), z = (0, domain_z),
                   topology = topology)

    @info "Building Reactant and GPU grids…"
    @time begin
        reactant_grid = RectilinearGrid(ReactantState(); grid_kwargs...)
        gpu_grid      = RectilinearGrid(GPU();           grid_kwargs...)
    end

    FT = eltype(reactant_grid)

    @info "Building atmosphere models (explicit compressible stepping)…"
    @time begin
        reactant_model = AtmosphereModel(reactant_grid; dynamics = CompressibleDynamics())
        gpu_model      = AtmosphereModel(gpu_grid;      dynamics = CompressibleDynamics())
    end

    thermo = reactant_model.thermodynamic_constants

    # ── Initial conditions ────────────────────────────────────────────────
    #
    # A Gaussian warm bubble centred at (bubble_center_x, bubble_center_y,
    # bubble_center_z).  The loss function internally resets model state
    # before each evaluation, so we only need the initial θ field.

    θ_perturbation(x, y, z) = FT(perturbation_amplitude) * exp(
        -((x - bubble_center_x)^2 + (y - bubble_center_y)^2 + (z - bubble_center_z)^2)
        / perturbation_radius^2)

    θ_init_func(x, y, z) = FT(θ_background) + θ_perturbation(x, y, z)

    @info "Initializing fields…"
    @time begin
        θ_initial_reactant = CenterField(reactant_grid); set!(θ_initial_reactant, θ_init_func)
        θ_initial_gpu      = CenterField(gpu_grid);      set!(θ_initial_gpu,      θ_init_func)
        dθ_initial         = CenterField(reactant_grid); set!(dθ_initial,         FT(0))
    end

    θ_reactant_arr = Array(interior(θ_initial_reactant))
    θ_gpu_arr      = Array(interior(θ_initial_gpu))

    @info "Initial θ diagnostics" minimum(θ_reactant_arr) maximum(θ_reactant_arr) maximum(abs, θ_reactant_arr .- θ_background)
    @info "Reactant vs GPU consistency" maximum(abs, θ_reactant_arr .- θ_gpu_arr)

    # ── Time step (CFL-limited by acoustic wave speed) ────────────────────

    Rᵈ          = thermo.molar_gas_constant / thermo.dry_air.molar_mass
    cᵖᵈ         = thermo.dry_air.heat_capacity
    γ           = cᵖᵈ / (cᵖᵈ - Rᵈ)
    sound_speed = sqrt(γ * Rᵈ * θ_background)
    Δt          = FT(0.4 * domain_x / Nx / sound_speed)

    @info @sprintf("Δt = %.6f s  (acoustic CFL ≈ 0.4, sound speed ≈ %.1f m/s)", Δt, sound_speed)

    # ── Reactant compilation ──────────────────────────────────────────────

    GC.gc()

    @info "Compiling forward pass with Reactant…"
    @time compiled_forward = Reactant.@compile raise=true raise_first=true sync=true loss(
        reactant_model, θ_initial_reactant, Δt, nsteps)

    GC.gc()

    @info "Compiling backward pass (Enzyme reverse mode) with Reactant…"
    shadow_model = Enzyme.make_zero(reactant_model)
    @time compiled_backward = Reactant.@compile raise=true raise_first=true sync=true compute_gradient(
        reactant_model, shadow_model, θ_initial_reactant, dθ_initial, Δt, nsteps)

    # ── Compute forward state and adjoint sensitivity ─────────────────────

    GC.gc()

    @info "Running forward pass to obtain evolved temperature…"
    @time compiled_forward(reactant_model, θ_initial_reactant, Δt, nsteps)
    temperature = Array(interior(reactant_model.temperature))

    GC.gc()

    @info "Running backward pass to obtain adjoint sensitivity ∂J/∂θ₀…"
    @time dθ_result, loss_value = compiled_backward(
        reactant_model, shadow_model, θ_initial_reactant, dθ_initial, Δt, nsteps)
    sensitivity = Array(interior(dθ_result))

    @info @sprintf("Loss J = %.6e", loss_value)
    @info @sprintf("Max |∂J/∂θ₀| = %.6e", maximum(abs, sensitivity))

    # ── Coordinate arrays for plotting ────────────────────────────────────

    xc = range(0, domain_x, length = Nx)
    yc = range(0, domain_y, length = Ny)
    zc = range(domain_z / 2Nz, domain_z - domain_z / 2Nz, length = Nz)

    i_center = argmin(abs.(xc .- bubble_center_x))
    j_center = argmin(abs.(yc .- bubble_center_y))

    # ── Visualisation: 2-D slices ─────────────────────────────────────────
    #
    # Top row: evolved potential temperature θ.
    # Bottom row: adjoint sensitivity ∂J/∂θ₀.
    # Each row shows an x–z slice (through bubble centre y) and a y–z slice
    # (through bubble centre x).

    θ_deviation_max = maximum(abs, temperature .- θ_background)
    θ_colorrange    = (θ_background - θ_deviation_max, θ_background + θ_deviation_max)

    sensitivity_max  = maximum(abs, sensitivity)
    sensitivity_range = (-sensitivity_max, sensitivity_max)

    fig_slices = Figure(size = (1200, 900), fontsize = 14)

    Label(fig_slices[0, :],
        @sprintf("Rising thermal bubble — %d×%d×%d, %d steps, Δt = %.4f s, J = %.6e",
                 Nx, Ny, Nz, nsteps, Δt, loss_value),
        fontsize = 16, tellwidth = false)

    ax_θ_xz = Axis(fig_slices[1, 1]; xlabel = "x (m)", ylabel = "z (m)",
                    title = "θ  — x–z at y = $(Int(round(yc[j_center]))) m",
                    aspect = DataAspect())
    hm_θ_xz = heatmap!(ax_θ_xz, xc, zc, temperature[:, j_center, :];
                        colormap = :thermal, colorrange = θ_colorrange)
    Colorbar(fig_slices[1, 2], hm_θ_xz; label = "θ (K)")

    ax_θ_yz = Axis(fig_slices[1, 3]; xlabel = "y (m)", ylabel = "z (m)",
                    title = "θ  — y–z at x = $(Int(round(xc[i_center]))) m",
                    aspect = DataAspect())
    hm_θ_yz = heatmap!(ax_θ_yz, yc, zc, temperature[i_center, :, :];
                        colormap = :thermal, colorrange = θ_colorrange)
    Colorbar(fig_slices[1, 4], hm_θ_yz; label = "θ (K)")

    ax_s_xz = Axis(fig_slices[2, 1]; xlabel = "x (m)", ylabel = "z (m)",
                    title = "∂J/∂θ₀ — x–z at y = $(Int(round(yc[j_center]))) m",
                    aspect = DataAspect())
    hm_s_xz = heatmap!(ax_s_xz, xc, zc, sensitivity[:, j_center, :];
                        colormap = :balance, colorrange = sensitivity_range)
    Colorbar(fig_slices[2, 2], hm_s_xz; label = "∂J/∂θ₀")

    ax_s_yz = Axis(fig_slices[2, 3]; xlabel = "y (m)", ylabel = "z (m)",
                    title = "∂J/∂θ₀ — y–z at x = $(Int(round(xc[i_center]))) m",
                    aspect = DataAspect())
    hm_s_yz = heatmap!(ax_s_yz, yc, zc, sensitivity[i_center, :, :];
                        colormap = :balance, colorrange = sensitivity_range)
    Colorbar(fig_slices[2, 4], hm_s_yz; label = "∂J/∂θ₀")

    fname_slices = @sprintf("rising_thermal_sensitivity_%dx%dx%d.png", Nx, Ny, Nz)
    @time save(fname_slices, fig_slices; px_per_unit = 2)
    @info "Saved $fname_slices"

    # ── Visualisation: 3-D cube cut-away ──────────────────────────────────
    #
    # Three exterior faces of the domain (top x–y, front x–z, left y–z)
    # form a single visible cube corner.  The color range is computed from
    # only the three visible face slices so that features on the boundary
    # are not washed out by interior extremes.

    top_face_θ   = temperature[:, :, end]
    front_face_θ = temperature[:, 1, :]
    left_face_θ  = temperature[1, :, :]

    visible_θ_max     = max(maximum(abs, top_face_θ   .- θ_background),
                            maximum(abs, front_face_θ  .- θ_background),
                            maximum(abs, left_face_θ   .- θ_background))
    θ_colorrange_3d   = (θ_background - visible_θ_max, θ_background + visible_θ_max)

    top_face_s   = sensitivity[:, :, end]
    front_face_s = sensitivity[:, 1, :]
    left_face_s  = sensitivity[1, :, :]

    visible_s_max     = max(maximum(abs, top_face_s),
                            maximum(abs, front_face_s),
                            maximum(abs, left_face_s))
    sensitivity_range_3d = (-visible_s_max, visible_s_max)

    top_X  = [xc[i] for i in 1:Nx, j in 1:Ny]
    top_Y  = [yc[j] for i in 1:Nx, j in 1:Ny]
    top_Z  = fill(zc[end], Nx, Ny)

    front_X = [xc[i] for i in 1:Nx, k in 1:Nz]
    front_Y = fill(yc[1], Nx, Nz)
    front_Z = [zc[k] for i in 1:Nx, k in 1:Nz]

    left_X = fill(xc[1], Ny, Nz)
    left_Y = [yc[j] for j in 1:Ny, k in 1:Nz]
    left_Z = [zc[k] for j in 1:Ny, k in 1:Nz]

    fig_3d = Figure(size = (1400, 600), fontsize = 14)

    Label(fig_3d[0, :],
        @sprintf("3-D cut-away — %d×%d×%d, %d steps, Δt = %.4f s",
                 Nx, Ny, Nz, nsteps, Δt),
        fontsize = 16, tellwidth = false)

    ax_3d_θ = Axis3(fig_3d[1, 1]; xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)",
                    title = "Potential temperature θ", aspect = :data,
                    azimuth = 1.20π, elevation = 0.30)

    surface!(ax_3d_θ, top_X, top_Y, top_Z;
             color = top_face_θ, colormap = :thermal,
             colorrange = θ_colorrange_3d, shading = NoShading)
    surface!(ax_3d_θ, front_X, front_Y, front_Z;
             color = front_face_θ, colormap = :thermal,
             colorrange = θ_colorrange_3d, shading = NoShading)
    surf_θ = surface!(ax_3d_θ, left_X, left_Y, left_Z;
                      color = left_face_θ, colormap = :thermal,
                      colorrange = θ_colorrange_3d, shading = NoShading)
    Colorbar(fig_3d[1, 2], surf_θ; label = "θ (K)")

    ax_3d_s = Axis3(fig_3d[1, 3]; xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)",
                    title = "Sensitivity ∂J/∂θ₀", aspect = :data,
                    azimuth = 1.20π, elevation = 0.30)

    surface!(ax_3d_s, top_X, top_Y, top_Z;
             color = top_face_s, colormap = :balance,
             colorrange = sensitivity_range_3d, shading = NoShading)
    surface!(ax_3d_s, front_X, front_Y, front_Z;
             color = front_face_s, colormap = :balance,
             colorrange = sensitivity_range_3d, shading = NoShading)
    surf_s = surface!(ax_3d_s, left_X, left_Y, left_Z;
                      color = left_face_s, colormap = :balance,
                      colorrange = sensitivity_range_3d, shading = NoShading)
    Colorbar(fig_3d[1, 4], surf_s; label = "∂J/∂θ₀")

    fname_3d = @sprintf("rising_thermal_3d_%dx%dx%d.png", Nx, Ny, Nz)
    @time save(fname_3d, fig_3d; px_per_unit = 2)
    @info "Saved $fname_3d"

    # ── Benchmarking ──────────────────────────────────────────────────────
    #
    # Reactant-compiled code is timed with `Reactant.Profiler.@timed`, which
    # synchronises with the device.  The plain-Julia GPU baseline uses
    # `Base.@elapsed` with CUDA.@sync for accurate wall-clock time.

    GC.gc()

    @info "Benchmarking forward pass (plain Julia on GPU) — $ntrials trials…"
    for _ in 1:nwarmup
        loss(gpu_model, θ_initial_gpu, Δt, nsteps)
    end
    time_forward_julia = @elapsed for _ in 1:ntrials
        loss(gpu_model, θ_initial_gpu, Δt, nsteps)
    end
    time_forward_julia /= ntrials

    GC.gc()

    @info "Benchmarking forward pass (Reactant compiled) — $ntrials trials…"
    profile_forward = Reactant.Profiler.@timed nrepeat=ntrials compiled_forward(
        reactant_model, θ_initial_reactant, Δt, nsteps)
    time_forward_reactant = profile_forward.runtime_ns / 1e9

    GC.gc()

    @info "Benchmarking backward pass (Reactant compiled) — $ntrials trials…"
    profile_backward = Reactant.Profiler.@timed nrepeat=ntrials compiled_backward(
        reactant_model, shadow_model, θ_initial_reactant, dθ_initial, Δt, nsteps)
    time_backward_reactant = profile_backward.runtime_ns / 1e9

    forward_speedup   = time_forward_julia / time_forward_reactant
    backward_to_forward = time_backward_reactant / time_forward_reactant

    benchmark_results[(Nx, Ny, Nz)] = (;
        time_forward_julia, time_forward_reactant, time_backward_reactant,
        forward_speedup, backward_to_forward)

    @info @sprintf("  Forward  (Julia GPU) : %10.4f s", time_forward_julia)
    @info @sprintf("  Forward  (Reactant)  : %10.4f s", time_forward_reactant)
    @info @sprintf("  Backward (Reactant)  : %10.4f s", time_backward_reactant)
    @info @sprintf("  Forward speedup      : %10.1f×",  forward_speedup)
    @info @sprintf("  Backward / Forward   : %10.1f×",  backward_to_forward)

end  # grid-size loop

# ## Benchmark summary
#
# Print a table and generate a grouped bar chart comparing wall-clock times
# across all grid resolutions.

@info "=" ^ 60
@info "Benchmark summary"
@info "=" ^ 60
@info @sprintf("  %-16s  %12s  %12s  %12s  %8s  %8s",
               "Grid", "Fwd (Julia)", "Fwd (React)", "Bwd (React)", "Speedup", "Bwd/Fwd")

for sz in grid_sizes
    r = benchmark_results[sz]
    label = @sprintf("%d×%d×%d", sz...)
    @info @sprintf("  %-16s  %10.4f s  %10.4f s  %10.4f s  %7.1f×  %7.1f×",
                   label, r.time_forward_julia, r.time_forward_reactant,
                   r.time_backward_reactant, r.forward_speedup, r.backward_to_forward)
end

# ## Benchmark bar chart
#
# Grouped bars: for each grid resolution, three bars show the plain-Julia GPU
# forward pass, the Reactant-compiled forward pass, and the Reactant backward
# pass.  Annotations show forward speedup and backward/forward ratio.

num_resolutions = length(grid_sizes)
resolution_labels = [@sprintf("%d×%d×%d", sz...) for sz in grid_sizes]

times_julia   = [benchmark_results[sz].time_forward_julia    for sz in grid_sizes]
times_reactant = [benchmark_results[sz].time_forward_reactant for sz in grid_sizes]
times_backward = [benchmark_results[sz].time_backward_reactant for sz in grid_sizes]

bar_positions = repeat(1:num_resolutions, inner = 3)
bar_dodge     = repeat(1:3, outer = num_resolutions)
bar_heights   = Float64[]
for i in 1:num_resolutions
    push!(bar_heights, times_julia[i], times_reactant[i], times_backward[i])
end
bar_colors = repeat([:steelblue, :seagreen, :coral], outer = num_resolutions)

fig_benchmark = Figure(size = (900, 500), fontsize = 14)

ax_benchmark = Axis(fig_benchmark[1, 1];
    xticks = (1:num_resolutions, resolution_labels),
    ylabel = "Wall time (s)",
    title  = "Rising Thermal Bubble — Reactant + Enzyme Benchmark",
    yscale = log10)

barplot!(ax_benchmark, bar_positions, bar_heights;
    dodge = bar_dodge, color = bar_colors,
    bar_labels = :y,
    label_formatter = x -> @sprintf("%.3g s", x))

legend_labels   = ["Forward (Julia GPU)", "Forward (Reactant)", "Backward (Reactant)"]
legend_elements = [PolyElement(color = c) for c in [:steelblue, :seagreen, :coral]]
Legend(fig_benchmark[1, 2], legend_elements, legend_labels; framevisible = false)

speedup_strings = [@sprintf("%.1f×", benchmark_results[sz].forward_speedup) for sz in grid_sizes]
ratio_strings   = [@sprintf("%.1f×", benchmark_results[sz].backward_to_forward) for sz in grid_sizes]
summary_text = join(
    [@sprintf("%s: speedup %s, bwd/fwd %s", resolution_labels[i], speedup_strings[i], ratio_strings[i])
     for i in 1:num_resolutions],
    "   |   ")

Label(fig_benchmark[2, :], summary_text;
      fontsize = 11, color = :gray30, tellwidth = false)

save("rising_thermal_benchmark.png", fig_benchmark; px_per_unit = 2)
@info "Saved rising_thermal_benchmark.png"

nothing #hide
