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

θ₀  = 300.0    # background potential temperature [K]
p₀  = 101325.0 # surface pressure [Pa]
pˢᵗ = 1e5      # standard pressure [Pa]

A   = 2.0      # perturbation amplitude [K]
R   = 1000.0   # perturbation radius [m]
x₀  = 5000.0   # bubble centre x [m]
y₀  = 5000.0   # bubble centre y [m]
z₀  = 2000.0   # bubble centre z [m]

# ## Domain, grid resolutions, and benchmark configuration
#
# The domain is a 10 km cube.  We sweep over three grid resolutions to study
# how wall-clock time scales with problem size.  For each resolution we create
# two grids: one on `ReactantState` for Reactant-compiled execution and one on
# `GPU()` for the plain-Julia baseline.

Lx, Ly, Lz = 10000.0, 10000.0, 10000.0

topo = (Periodic, Bounded, Bounded)

grid_sizes  = [(64, 64, 64), (256, 256, 256), (512, 512, 512)]
z_threshold = 5000.0
nsteps      = 2500
nwarmup     = 1
ntrials     = 3

# ## Loss function and adjoint
#
# The loss is the mean squared potential temperature in the upper half of the domain
# (above ``z_*``).  This is a simple proxy for "how much warm air has risen".
#
# `grad_loss` wraps `Enzyme.autodiff` in reverse mode to compute
# ``\partial J / \partial \theta_0`` — the sensitivity of the loss to the initial
# temperature field.
#
# Both functions are defined once and reused for every grid resolution.

function loss(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = FT(1))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    T = interior(model.temperature)
    k_start = ceil(Int, z_threshold / Lz * size(model.grid, 3)) + 1
    upper = @view T[:, :, k_start:end]
    return mean(upper .^ 2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    _, loss_val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dθ_init, loss_val
end

# ## Multi-resolution benchmark loop
#
# For each grid resolution we:
#
# 1. Build `ReactantState` and `GPU()` grids and models.
# 2. Set initial conditions (Gaussian warm bubble).
# 3. Compile the forward and backward passes with Reactant.
# 4. Benchmark: plain-Julia forward (GPU), Reactant forward, Reactant backward.
# 5. Extract evolved temperature and adjoint sensitivity for visualisation.
# 6. Save 2D-slice and 3D cube-face plots tagged by resolution.
#
# Timing results are collected into `benchmark_results` for the summary plot.

benchmark_results = Dict{Tuple{Int,Int,Int}, NamedTuple}()

for (Nx, Ny, Nz) in grid_sizes

    @info "=" ^ 60
    @info @sprintf("Grid resolution: %d × %d × %d  (%d cells)", Nx, Ny, Nz, Nx*Ny*Nz)
    @info "=" ^ 60

    # ── Grids and models ──────────────────────────────────────────────────

    grid_kwargs = (size = (Nx, Ny, Nz),
                   x = (0, Lx), y = (0, Ly), z = (0, Lz),
                   topology = topo)

    @info "Building grids…"
    @time begin
        grid     = RectilinearGrid(ReactantState(); grid_kwargs...)
        grid_gpu = RectilinearGrid(GPU();           grid_kwargs...)
    end

    FT = eltype(grid)

    @info "Building models…"
    @time begin
        model     = AtmosphereModel(grid;     dynamics = CompressibleDynamics())
        model_gpu = AtmosphereModel(grid_gpu; dynamics = CompressibleDynamics())
    end

    constants = model.thermodynamic_constants

    # ── Initial conditions ────────────────────────────────────────────────
    #
    # A Gaussian warm bubble centred at (x₀, y₀, z₀).  The loss function
    # internally resets model state before each evaluation, so we only need
    # the initial θ field (not full model state).

    θ_perturbation(x, y, z) = FT(A) * exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / R^2)
    θ_initial(x, y, z)      = FT(θ₀) + θ_perturbation(x, y, z)

    @info "Initializing fields…"
    @time begin
        θ_init     = CenterField(grid);     set!(θ_init,     θ_initial)
        θ_init_gpu = CenterField(grid_gpu); set!(θ_init_gpu, θ_initial)
        dθ_init    = CenterField(grid);     set!(dθ_init,    FT(0))
    end

    θ_init_arr     = Array(interior(θ_init))
    θ_init_gpu_arr = Array(interior(θ_init_gpu))
    @info "Initial θ diagnostics" minimum=minimum(θ_init_arr) maximum=maximum(θ_init_arr) max_perturbation=maximum(abs, θ_init_arr .- θ₀)
    @info "Reactant/GPU θ consistency" max_abs_diff=maximum(abs, θ_init_arr .- θ_init_gpu_arr)

    # ── Time step (CFL-limited by the acoustic sound speed) ───────────────

    Rᵈ  = constants.molar_gas_constant / constants.dry_air.molar_mass
    cᵖᵈ = constants.dry_air.heat_capacity
    γ   = cᵖᵈ / (cᵖᵈ - Rᵈ)
    cₛ  = sqrt(γ * Rᵈ * θ₀)
    Δt  = FT(0.4 * Lx / Nx / cₛ)

    @info @sprintf("Δt = %.6f s  (CFL ≈ 0.4, cₛ ≈ %.1f m/s)", Δt, cₛ)

    # ── Reactant compilation ──────────────────────────────────────────────

    @info "Compiling forward pass…"
    @time compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss(
        model, θ_init, Δt, nsteps)

    @info "Compiling backward pass (Enzyme reverse mode)…"
    dmodel = Enzyme.make_zero(model)
    @time compiled_bwd = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
        model, dmodel, θ_init, dθ_init, Δt, nsteps)

    # ── Benchmarking ──────────────────────────────────────────────────────
    #
    # Reactant-compiled code is timed with `Reactant.Profiler.@timed`, which
    # synchronises with the device.  The plain-Julia GPU baseline uses
    # `Base.@elapsed` with CUDA.@sync for accurate wall-clock time.

    @info "Benchmarking forward pass (plain Julia, GPU) — $ntrials trials…"
    for _ in 1:nwarmup
        loss(model_gpu, θ_init_gpu, Δt, nsteps)
    end
    t_fwd_julia = @elapsed for _ in 1:ntrials
        loss(model_gpu, θ_init_gpu, Δt, nsteps)
    end
    t_fwd_julia /= ntrials

    @info "Benchmarking forward pass (Reactant compiled) — $ntrials trials…"
    prof_fwd = Reactant.Profiler.@timed nrepeat=ntrials compiled_fwd(
        model, θ_init, Δt, nsteps)
    t_fwd_compiled = prof_fwd.runtime_ns / 1e9

    @info "Benchmarking backward pass (Reactant compiled) — $ntrials trials…"
    prof_bwd = Reactant.Profiler.@timed nrepeat=ntrials compiled_bwd(
        model, dmodel, θ_init, dθ_init, Δt, nsteps)
    t_bwd_compiled = prof_bwd.runtime_ns / 1e9

    fwd_speedup   = t_fwd_julia / t_fwd_compiled
    bwd_fwd_ratio = t_bwd_compiled / t_fwd_compiled

    benchmark_results[(Nx, Ny, Nz)] = (;
        t_fwd_julia, t_fwd_compiled, t_bwd_compiled,
        fwd_speedup, bwd_fwd_ratio)

    @info @sprintf("  Forward  (Julia GPU) : %10.4f s", t_fwd_julia)
    @info @sprintf("  Forward  (Reactant)  : %10.4f s", t_fwd_compiled)
    @info @sprintf("  Backward (Reactant)  : %10.4f s", t_bwd_compiled)
    @info @sprintf("  Forward speedup      : %10.1f×",  fwd_speedup)
    @info @sprintf("  Backward / Forward   : %10.1f×",  bwd_fwd_ratio)

    # ── Forward state and sensitivity field ───────────────────────────────

    @info "Running compiled forward pass for evolved state…"
    @time compiled_fwd(model, θ_init, Δt, nsteps)
    temperature = Array(interior(model.temperature))

    @info "Computing adjoint sensitivity…"
    @time dθ, J = compiled_bwd(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    sensitivity = Array(interior(dθ))

    @info "Loss value" J
    @info "Max |∂J/∂θ₀|" maximum_sensitivity=maximum(abs, sensitivity)

    # ── Visualisation: 2D slices ──────────────────────────────────────────
    #
    # Evolved potential temperature (top row) and adjoint sensitivity
    # ∂J/∂θ₀ (bottom row), sliced nearest the bubble centre.

    x = range(0, Lx, length = Nx)
    y = range(0, Ly, length = Ny)
    z = range(Lz / 2Nz, Lz - Lz / 2Nz, length = Nz)

    i0 = argmin(abs.(x .- x₀))
    j0 = argmin(abs.(y .- y₀))

    fig = Figure(size = (1200, 900), fontsize = 14)

    Label(fig[0, :],
        @sprintf("Rising thermal bubble — %d×%d×%d, %d steps, Δt = %.4f s, J = %.6e",
                 Nx, Ny, Nz, nsteps, Δt, J),
        fontsize = 16, tellwidth = false)

    θlim   = maximum(abs, temperature .- θ₀)
    θrange = (θ₀ - θlim, θ₀ + θlim)

    ax1 = Axis(fig[1, 1]; xlabel = "x (m)", ylabel = "z (m)",
               title = "θ  — x–z slice at y ≈ $(Int(round(y[j0]))) m",
               aspect = DataAspect())
    hm1 = heatmap!(ax1, x, z, temperature[:, j0, :];
                   colormap = :thermal, colorrange = θrange)
    Colorbar(fig[1, 2], hm1; label = "θ (K)")

    ax2 = Axis(fig[1, 3]; xlabel = "y (m)", ylabel = "z (m)",
               title = "θ  — y–z slice at x ≈ $(Int(round(x[i0]))) m",
               aspect = DataAspect())
    hm2 = heatmap!(ax2, y, z, temperature[i0, :, :];
                   colormap = :thermal, colorrange = θrange)
    Colorbar(fig[1, 4], hm2; label = "θ (K)")

    slimit = maximum(abs, sensitivity)

    ax3 = Axis(fig[2, 1]; xlabel = "x (m)", ylabel = "z (m)",
               title = "∂J/∂θ₀  — x–z slice at y ≈ $(Int(round(y[j0]))) m",
               aspect = DataAspect())
    hm3 = heatmap!(ax3, x, z, sensitivity[:, j0, :];
                   colormap = :balance, colorrange = (-slimit, slimit))
    Colorbar(fig[2, 2], hm3; label = "∂J/∂θ₀")

    ax4 = Axis(fig[2, 3]; xlabel = "y (m)", ylabel = "z (m)",
               title = "∂J/∂θ₀  — y–z slice at x ≈ $(Int(round(x[i0]))) m",
               aspect = DataAspect())
    hm4 = heatmap!(ax4, y, z, sensitivity[i0, :, :];
                   colormap = :balance, colorrange = (-slimit, slimit))
    Colorbar(fig[2, 4], hm4; label = "∂J/∂θ₀")

    fname_2d = @sprintf("rising_thermal_sensitivity_%dx%dx%d.png", Nx, Ny, Nz)
    @time save(fname_2d, fig; px_per_unit = 2)
    @info "Saved $fname_2d"

    # ── Visualisation: 3D cube-face view ──────────────────────────────────
    #
    # Three exterior faces of the domain (top, front, left) form a single
    # visible cube corner.

    xy_X = [x[i] for i in 1:Nx, j in 1:Ny]
    xy_Y = [y[j] for i in 1:Nx, j in 1:Ny]
    xy_Z = fill(z[end], Nx, Ny)

    xz_X = [x[i] for i in 1:Nx, k in 1:Nz]
    xz_Y = fill(y[1], Nx, Nz)
    xz_Z = [z[k] for i in 1:Nx, k in 1:Nz]

    yz_X = fill(x[1], Ny, Nz)
    yz_Y = [y[j] for j in 1:Ny, k in 1:Nz]
    yz_Z = [z[k] for j in 1:Ny, k in 1:Nz]

    fig3 = Figure(size = (1400, 600), fontsize = 14)

    Label(fig3[0, :],
        @sprintf("3D cut-away — %d×%d×%d, %d steps, Δt = %.4f s", Nx, Ny, Nz, nsteps, Δt),
        fontsize = 16, tellwidth = false)

    ax3d_θ = Axis3(fig3[1, 1]; xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)",
                   title = "Potential temperature θ", aspect = :data,
                   azimuth = 1.20π, elevation = 0.30)

    surface!(ax3d_θ, xy_X, xy_Y, xy_Z; color = temperature[:, :, end],
             colormap = :thermal, colorrange = θrange, shading = NoShading)
    surface!(ax3d_θ, xz_X, xz_Y, xz_Z; color = temperature[:, 1, :],
             colormap = :thermal, colorrange = θrange, shading = NoShading)
    sf_θ = surface!(ax3d_θ, yz_X, yz_Y, yz_Z; color = temperature[1, :, :],
                    colormap = :thermal, colorrange = θrange, shading = NoShading)
    Colorbar(fig3[1, 2], sf_θ; label = "θ (K)")

    ax3d_s = Axis3(fig3[1, 3]; xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)",
                   title = "Sensitivity ∂J/∂θ₀", aspect = :data,
                   azimuth = 1.20π, elevation = 0.30)

    surface!(ax3d_s, xy_X, xy_Y, xy_Z; color = sensitivity[:, :, end],
             colormap = :balance, colorrange = (-slimit, slimit), shading = NoShading)
    surface!(ax3d_s, xz_X, xz_Y, xz_Z; color = sensitivity[:, 1, :],
             colormap = :balance, colorrange = (-slimit, slimit), shading = NoShading)
    sf_s = surface!(ax3d_s, yz_X, yz_Y, yz_Z; color = sensitivity[1, :, :],
                    colormap = :balance, colorrange = (-slimit, slimit), shading = NoShading)
    Colorbar(fig3[1, 4], sf_s; label = "∂J/∂θ₀")

    fname_3d = @sprintf("rising_thermal_3d_%dx%dx%d.png", Nx, Ny, Nz)
    @time save(fname_3d, fig3; px_per_unit = 2)
    @info "Saved $fname_3d"

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
                   label, r.t_fwd_julia, r.t_fwd_compiled, r.t_bwd_compiled,
                   r.fwd_speedup, r.bwd_fwd_ratio)
end

# ## Benchmark bar chart
#
# Grouped bars: for each grid resolution, three bars show the plain-Julia GPU
# forward pass, the Reactant-compiled forward pass, and the Reactant backward
# pass.  Annotations show forward speedup and backward/forward ratio.

n = length(grid_sizes)
xlabels  = [@sprintf("%d³", sz[1]) for sz in grid_sizes]
t_julia  = [benchmark_results[sz].t_fwd_julia    for sz in grid_sizes]
t_react  = [benchmark_results[sz].t_fwd_compiled  for sz in grid_sizes]
t_bwd    = [benchmark_results[sz].t_bwd_compiled   for sz in grid_sizes]

xs     = repeat(1:n, inner = 3)
dodge  = repeat(1:3, outer = n)
heights = Float64[]
for i in 1:n
    push!(heights, t_julia[i], t_react[i], t_bwd[i])
end
colors = repeat([:steelblue, :seagreen, :coral], outer = n)

fig_bench = Figure(size = (900, 500), fontsize = 14)

ax_bench = Axis(fig_bench[1, 1];
    xticks = (1:n, xlabels),
    ylabel = "Wall time (s)",
    title  = "Rising Thermal Bubble — Reactant + Enzyme Benchmark",
    yscale = log10)

barplot!(ax_bench, xs, heights;
    dodge = dodge, color = colors,
    bar_labels = :y,
    label_formatter = x -> @sprintf("%.3g s", x))

labels = ["Forward (Julia GPU)", "Forward (Reactant)", "Backward (Reactant)"]
elements = [PolyElement(color = c) for c in [:steelblue, :seagreen, :coral]]
Legend(fig_bench[1, 2], elements, labels; framevisible = false)

speedup_strs = [@sprintf("%.1f×", benchmark_results[sz].fwd_speedup) for sz in grid_sizes]
ratio_strs   = [@sprintf("%.1f×", benchmark_results[sz].bwd_fwd_ratio) for sz in grid_sizes]
summary_text = join([@sprintf("%s: speedup %s, bwd/fwd %s", xlabels[i], speedup_strs[i], ratio_strs[i])
                     for i in 1:n], "   |   ")

Label(fig_bench[2, :], summary_text;
      fontsize = 11, color = :gray30, tellwidth = false)

save("rising_thermal_benchmark.png", fig_bench; px_per_unit = 2)
@info "Saved rising_thermal_benchmark.png"

nothing #hide
