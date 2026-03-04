# # Differentiable Rising Thermal Bubble
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
# We benchmark three quantities:
#
# | Benchmark | Description |
# |-----------|-------------|
# | Forward (plain Julia) | uncompiled forward pass on `CPU()` |
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

# ## Grid and model
#
# We use a small ``4 × 4 × 4`` grid to keep compilation and benchmarking fast.
# Two copies of the grid and model are created: one on `ReactantState` for
# Reactant-compiled execution and one on `CPU()` for the plain-Julia baseline.

Lx, Ly, Lz = 10000.0, 10000.0, 10000.0
Nx, Ny, Nz = 32, 32, 32

topo = (Periodic, Bounded, Bounded)
grid_kwargs = (size = (Nx, Ny, Nz), x = (0, Lx), y = (0, Ly), z = (0, Lz), topology = topo)

@info "Building grids…"
@time begin
    grid     = RectilinearGrid(ReactantState(); grid_kwargs...)
    grid_cpu = RectilinearGrid(CPU();           grid_kwargs...)
end

FT = eltype(grid)

@info "Building models…"
@time begin
    model     = AtmosphereModel(grid;     dynamics = CompressibleDynamics())
    model_cpu = AtmosphereModel(grid_cpu; dynamics = CompressibleDynamics())
end

constants = model.thermodynamic_constants

# ## Initial conditions
#
# A Gaussian warm bubble centred at ``(x_0, y_0, z_0)`` sits inside a hydrostatically
# balanced reference atmosphere.  Density is diagnosed from the adiabatic hydrostatic
# balance at each height.

θ_perturbation(x, y, z) = FT(A) * exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / R^2)
θ_initial(x, y, z) = FT(θ₀) + θ_perturbation(x, y, z)
ρ_initial(x, y, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

@info "Initializing fields…"
@time begin
    θ_init     = CenterField(grid);     set!(θ_init,     θ_initial)
    θ_init_cpu = CenterField(grid_cpu); set!(θ_init_cpu, θ_initial)
    dθ_init    = CenterField(grid);     set!(dθ_init,    FT(0))
end

@info "Setting initial model state…"
@time begin
    set!(model;     θ = θ_init,     ρ = ρ_initial)
    set!(model_cpu; θ = θ_init_cpu, ρ = ρ_initial)
end

θ_init_arr = Array(interior(θ_init))
θ_init_cpu_arr = Array(interior(θ_init_cpu))
@info "Initial θ field diagnostics" minimum=minimum(θ_init_arr) maximum=maximum(θ_init_arr) max_perturbation=maximum(abs, θ_init_arr .- θ₀)
@info "Reactant/CPU θ init consistency" max_abs_difference=maximum(abs, θ_init_arr .- θ_init_cpu_arr)

# ## Time step
#
# The CFL condition is set by the acoustic sound speed.

Rᵈ  = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ   = cᵖᵈ / (cᵖᵈ - Rᵈ)
cₛ  = sqrt(γ * Rᵈ * θ₀)
Δt  = FT(0.4 * Lx / Nx / cₛ)

# ## Loss function and adjoint
#
# The loss is the mean squared potential temperature in the upper half of the domain
# (above ``z_*``).  This is a simple proxy for "how much warm air has risen".
#
# `grad_loss` wraps `Enzyme.autodiff` in reverse mode to compute
# ``\partial J / \partial \theta_0`` — the sensitivity of the loss to the initial
# temperature field.

z_threshold = 5000.0
nsteps      = 2500

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

# ## Reactant compilation
#
# `Reactant.@compile` traces both the forward model and the Enzyme-generated adjoint
# into XLA/StableHLO, producing fused, optimized kernels.

@info "Compiling forward pass…"
@time compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss(
    model, θ_init, Δt, nsteps)

@info "Compiling backward pass (Enzyme reverse mode)…"
dmodel = Enzyme.make_zero(model)
@time compiled_bwd = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model, dmodel, θ_init, dθ_init, Δt, nsteps)

# ## Benchmarking
#
# Reactant-compiled code is timed with `Reactant.Profiler.@timed`, which synchronises
# with the device and returns accurate wall-clock time.  The plain-Julia baseline uses
# `Base.@elapsed` (CPU-synchronous by construction).

# nwarmup = 1
# ntrials = 5

# # ### Forward pass — plain Julia on CPU

# @info "Benchmarking forward pass (plain Julia, CPU)…"
# for _ in 1:nwarmup
#     loss(model_cpu, θ_init_cpu, Δt, nsteps)
# end
# t_fwd_julia = @elapsed for _ in 1:ntrials
#     loss(model_cpu, θ_init_cpu, Δt, nsteps)
# end
# t_fwd_julia /= ntrials

# # ### Forward pass — Reactant compiled

# @info "Benchmarking forward pass (Reactant compiled)…"
# prof_fwd = Reactant.Profiler.@timed nrepeat=ntrials compiled_fwd(model, θ_init, Δt, nsteps)
# t_fwd_compiled = prof_fwd.runtime_ns / 1e9

# # ### Backward pass — Reactant compiled (Enzyme reverse mode)

# @info "Benchmarking backward pass (Reactant compiled)…"
# prof_bwd = Reactant.Profiler.@timed nrepeat=ntrials compiled_bwd(model, dmodel, θ_init, dθ_init, Δt, nsteps)
# t_bwd_compiled = prof_bwd.runtime_ns / 1e9

# # ### Results

# fwd_speedup = t_fwd_julia / t_fwd_compiled
# bwd_fwd_ratio = t_bwd_compiled / t_fwd_compiled

# @info "Forward  (Julia)   " runtime = t_fwd_julia
# @info "Forward  (Reactant)" runtime = t_fwd_compiled
# @info "Backward (Reactant)" runtime = t_bwd_compiled
# @info "Forward speedup (Julia / Reactant)" speedup = fwd_speedup
# @info "Backward / Forward ratio (Reactant)" ratio = bwd_fwd_ratio

# ## Forward state and sensitivity field
#
# We run the compiled forward pass to obtain the evolved temperature, then the
# backward pass to obtain ``\partial J / \partial \theta_0``.

@info "Running forward pass for evolved state…"
@time compiled_fwd(model, θ_init, Δt, nsteps)
temperature = Array(interior(model.temperature))

@info "Computing sensitivity…"
@time dθ, J = compiled_bwd(model, dmodel, θ_init, dθ_init, Δt, nsteps)
sensitivity = Array(interior(dθ))

@info "Loss value" J
@info "Max |∂J/∂θ₀|" maximum_sensitivity = maximum(abs, interior(dθ))

# ## Visualisation
#
# A combined figure with the evolved potential temperature (top row) and the adjoint
# sensitivity ``\partial J / \partial \theta_0`` (bottom row). These 2D slices are
# chosen nearest to the bubble center ``(x₀, y₀, z₀)``.

x = range(0, Lx, length = Nx)
y = range(0, Ly, length = Ny)
z = range(Lz / 2Nz, Lz - Lz / 2Nz, length = Nz)

i0 = argmin(abs.(x .- x₀))
j0 = argmin(abs.(y .- y₀))

fig = Figure(size = (1200, 900), fontsize = 14)

Label(fig[0, :],
    @sprintf("Rising thermal bubble — %d steps, Δt = %.4f s, J = %.6e", nsteps, Δt, J),
    fontsize = 16, tellwidth = false)

# Top row: evolved potential temperature

θlim = maximum(abs, temperature .- θ₀)
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

# Bottom row: adjoint sensitivity ∂J/∂θ₀

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

@time save("rising_thermal_sensitivity.png", fig; px_per_unit = 2)
@info "Saved rising_thermal_sensitivity.png"

# ## 3D cube-face view
#
# To keep all planes perfectly aligned, we plot three *exterior* faces of the domain:
# top (`z = z_max`), front (`y = y_min`), and left (`x = x_min`). This forms a
# single visible cube corner. For now, the face colors also come from these boundary
# slices (rather than middle slices).

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
    @sprintf("3D cut-away — %d steps, Δt = %.4f s", nsteps, Δt),
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

@time save("rising_thermal_3d.png", fig3; px_per_unit = 2)
@info "Saved rising_thermal_3d.png"

# ## Timing bar chart
#
# Three bars: the plain-Julia forward pass, the Reactant forward pass, and the Reactant
# backward pass.  Annotations show the forward speedup and the backward/forward ratio.
#
# NOTE: disabled while iterating on model setup and visualisation. Re-enable for
# publication-style performance comparisons.
#
# fig2 = Figure(size = (700, 420), fontsize = 14)
# ax = Axis(fig2[1, 1];
#           xticks = ([1, 2, 3],
#                     ["Forward\n(Julia)", "Forward\n(Reactant)", "Backward\n(Reactant)"]),
#           ylabel = "Wall time (s)",
#           title  = "Rising Thermal Bubble — Reactant + Enzyme Benchmark")
#
# barplot!(ax, [1, 2, 3], [t_fwd_julia, t_fwd_compiled, t_bwd_compiled];
#          color = [:steelblue, :seagreen, :coral],
#          bar_labels = :y, label_formatter = x -> @sprintf("%.4f s", x))
#
# Label(fig2[2, 1],
#       @sprintf("Forward speedup: %.1f×   |   Backward / Forward ratio: %.1f×",
#                fwd_speedup, bwd_fwd_ratio),
#       fontsize = 12, color = :gray30, tellwidth = false)
#
# save("rising_thermal_benchmark.png", fig2; px_per_unit = 2)
# @info "Saved rising_thermal_benchmark.png"

nothing #hide
