# # Differentiable Rising Thermal Bubble
#
# A warm Gaussian bubble in a neutrally stratified atmosphere rises under buoyancy,
# rolls up into a vortex ring, and eventually breaks into 3D turbulence.  This is
# a classic compressible dynamics test case — simple to set up, physically rich, and
# clean for automatic differentiation.
#
# We use this example to benchmark **Reactant + Enzyme** for computing derivatives
# of a scalar loss through the fully compressible Euler equations.  The workflow:
#
# 1. Run the forward model (with and without Reactant compilation).
# 2. Define a loss ``J`` — mean squared potential temperature perturbation above a
#    threshold height — and differentiate it with respect to the initial condition.
# 3. Compare wall-clock times for the forward and backward passes.
# 4. Visualize the adjoint sensitivity field ``\partial J / \partial \theta_0``.
#
# ## Domain
#
# Channel geometry: periodic in ``x``, bounded (free-slip) in ``y`` and ``z``.
# The bubble rises in ``z`` and interacts with the ``y`` walls as it expands.
#
# ## Physics
#
# Compressible Euler with gravity.  No moisture, no Coriolis, no subgrid closure.
# Background: neutrally stratified (``\bar\theta = \mathrm{const}``), hydrostatically
# balanced, at rest.

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

# ## Parameters

const θ₀  = 300.0    # Background potential temperature [K]
const p₀  = 101325.0 # Surface pressure [Pa]
const pˢᵗ = 1e5      # Standard pressure [Pa]

const A   = 2.0      # Perturbation amplitude [K]
const R   = 1000.0   # Perturbation radius [m]
const x₀  = 5000.0   # Bubble center x [m]
const y₀  = 5000.0   # Bubble center y [m]
const z₀  = 2000.0   # Bubble center z [m]

const Lx  = 10000.0  # Domain length x [m]
const Ly  = 10000.0  # Domain length y [m]
const Lz  = 10000.0  # Domain height [m]

const Nx  = 4
const Ny  = 4
const Nz  = 4

const z_threshold = 5000.0  # Height above which the loss accumulates [m]
const nsteps      = 4      # Number of time steps (keep small for benchmarking)

# ## Grid

grid = RectilinearGrid(ReactantState();
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       topology = (Periodic, Bounded, Bounded))

FT = eltype(grid)

# ## Model

model = AtmosphereModel(grid; dynamics = CompressibleDynamics())
constants = model.thermodynamic_constants

# ## Initial condition
#
# Hydrostatically balanced density with a warm Gaussian potential temperature
# perturbation centered at ``(x_0, y_0, z_0)``:
#
# ```math
# \theta'(\mathbf{x}) = A \exp\!\left( -\frac{|\mathbf{x} - \mathbf{x}_0|^2}{R^2} \right)
# ```

θ_perturbation(x, y, z) = FT(A) * exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / R^2)
θ_initial(x, y, z) = FT(θ₀) + θ_perturbation(x, y, z)
ρ_initial(x, y, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

# Build θ_init as a concrete CenterField for the differentiable pipeline.
θ_init  = CenterField(grid); set!(θ_init, θ_initial)
dθ_init = CenterField(grid); set!(dθ_init, FT(0))

set!(model; θ=θ_init, ρ=ρ_initial)

# ## Loss function
#
# Mean squared potential temperature above ``z_*``:
#
# ```math
# J = \frac{1}{N_{z>z_*}} \sum_{k : z_k > z_*} \theta(i,j,k,T)^2
# ```
#
# In the linear regime ``J \propto A^2``, providing an analytic consistency check.
# Once the vortex ring breaks down, the adjoint sensitivity grows chaotically —
# exactly the regime that motivates ensemble / Girsanov approaches.

function loss(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=FT(1))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end

    T = interior(model.temperature)
    Nz_grid = size(model.grid, 3)
    k_start = ceil(Int, z_threshold / Lz * Nz_grid) + 1
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

# ## Time step
#
# CFL based on sound speed.  With explicit time stepping the acoustic CFL
# constrains Δt.

Rᵈ  = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ   = cᵖᵈ / (cᵖᵈ - Rᵈ)
cₛ  = sqrt(γ * Rᵈ * θ₀)

Δx  = Lx / Nx
Δt  = FT(0.4 * Δx / cₛ)

# ## Compile

@info "Compiling forward pass..."
compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss(
    model, θ_init, Δt, nsteps)

@info "Compiling backward pass (Enzyme reverse mode)..."
dmodel = Enzyme.make_zero(model)
compiled_bwd = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model, dmodel, θ_init, dθ_init, Δt, nsteps)

# ## Benchmark: forward pass

@info "Benchmarking forward pass..."
set!(model; θ=θ_init, ρ=ρ_initial)

t_fwd_compiled = @elapsed for _ in 1:3
    compiled_fwd(model, θ_init, Δt, nsteps)
end
t_fwd_compiled /= 3

t_fwd_julia = @elapsed for _ in 1:3
    loss(model, θ_init, Δt, nsteps)
end
t_fwd_julia /= 3

@info "Forward pass" compiled=t_fwd_compiled julia=t_fwd_julia speedup=t_fwd_julia/t_fwd_compiled

# ## Benchmark: backward pass

@info "Benchmarking backward pass..."

t_bwd_compiled = @elapsed for _ in 1:3
    compiled_bwd(model, dmodel, θ_init, dθ_init, Δt, nsteps)
end
t_bwd_compiled /= 3

@info "Backward pass (compiled)" time=t_bwd_compiled

# ## Compute the sensitivity field

@info "Computing sensitivity..."
dθ, J = compiled_bwd(model, dmodel, θ_init, dθ_init, Δt, nsteps)

@info "Loss value" J
@info "Max |∂J/∂θ₀|" maximum_sensitivity=maximum(abs, interior(dθ))

# ## Visualization
#
# Central cross-sections of the adjoint sensitivity ``\partial J / \partial \theta_0``
# through the bubble center, showing which initial temperature perturbations most
# affect the upper-atmosphere thermal variance.

sensitivity = Array(interior(dθ))

x = range(0, Lx, length=Nx)
y = range(0, Ly, length=Ny)
z = range(Lz/2Nz, Lz - Lz/2Nz, length=Nz)

jmid = Ny ÷ 2
imid = Nx ÷ 2

fig = Figure(size = (1200, 500), fontsize = 14)

# x–z slice through y = Ly/2
ax1 = Axis(fig[1, 1]; xlabel = "x (m)", ylabel = "z (m)",
           title = "∂J/∂θ₀  — x–z slice at y = $(Int(Ly/2)) m",
           aspect = DataAspect())

slimit = maximum(abs, sensitivity)
hm1 = heatmap!(ax1, x, z, sensitivity[:, jmid, :];
               colormap = :balance, colorrange = (-slimit, slimit))
Colorbar(fig[1, 2], hm1; label = "∂J/∂θ₀")

# y–z slice through x = Lx/2
ax2 = Axis(fig[1, 3]; xlabel = "y (m)", ylabel = "z (m)",
           title = "∂J/∂θ₀  — y–z slice at x = $(Int(Lx/2)) m",
           aspect = DataAspect())

hm2 = heatmap!(ax2, y, z, sensitivity[imid, :, :];
               colormap = :balance, colorrange = (-slimit, slimit))
Colorbar(fig[1, 4], hm2; label = "∂J/∂θ₀")

# Title
supertitle = Label(fig[0, :],
    @sprintf("Adjoint sensitivity — %d steps, Δt = %.4f s, J = %.6e", nsteps, Δt, J),
    fontsize = 16, tellwidth = false)

save("rising_thermal_sensitivity.png", fig; px_per_unit = 2)
@info "Saved rising_thermal_sensitivity.png"

# ## Timing summary

fig2 = Figure(size = (500, 350), fontsize = 14)
ax = Axis(fig2[1, 1];
          xticks = ([1, 2, 3], ["Forward\n(Julia)", "Forward\n(Reactant)", "Backward\n(Reactant)"]),
          ylabel = "Wall time (s)",
          title = "Rising Thermal Bubble — Compilation Benchmark")

barplot!(ax, [1, 2, 3], [t_fwd_julia, t_fwd_compiled, t_bwd_compiled];
         color = [:steelblue, :seagreen, :indianred],
         bar_labels = :y, label_formatter = x -> @sprintf("%.3f s", x))

save("rising_thermal_benchmark.png", fig2; px_per_unit = 2)
@info "Saved rising_thermal_benchmark.png"

nothing #hide
