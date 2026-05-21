# # Acoustic refraction in wind shear — a differentiable example
#
# This example does two things on top of the same compressible forward model.
# **First**, we simulate an acoustic pulse propagating through a wind shear
# layer using the fully compressible [Euler
# equations](https://en.wikipedia.org/wiki/Euler_equations_(fluid_dynamics)),
# and observe how the shear refracts the wave: waves traveling **with** the
# wind bend **downward** (trapped near the surface), while waves traveling
# **against** the wind bend **upward**.  The effective propagation speed for
# a wave in direction ``\hat{\boldsymbol{n}}`` is
# ```math
# \mathbb{C}^{ac} + \boldsymbol{u} \cdot \hat{\boldsymbol{n}}
# ```
# where ``ℂᵃᶜ`` is the acoustic sound speed and ``\boldsymbol{u}`` is the wind
# velocity.  Wavefronts tilt toward regions of lower effective propagation
# speed, "ducting" sound energy along the surface — which is why distant
# sounds are often heard more clearly downwind. For more on this topic, see
#
# ```@bibliography
# ostashev2015acoustics
# pierce2019acoustics
# ```
#
# **Second**, we use this setup as a minimal introduction to *differentiable*
# atmospheric simulation in Breeze.  After running the forward problem, we
# take a gradient through the entire compressible time-stepping — asking
# *which parts of the wind profile control how much acoustic energy ends up
# trapped near the surface?* — using
# [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for reverse-mode
# automatic differentiation and [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)
# to compile the model down to
# [XLA](https://en.wikipedia.org/wiki/Accelerated_Linear_Algebra).  The result
# is a 2D sensitivity field obtained in a single backward pass; the same
# answer via finite differences would cost one model rerun per grid cell.
#
# We use stable stratification to suppress [Kelvin-Helmholtz instability](https://en.wikipedia.org/wiki/Kelvin%E2%80%93Helmholtz_instability)
# and a logarithmic wind profile consistent with the atmospheric surface layer.

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie

# ## Grid and model setup

Nx, Nz = 128, 64
Lx, Lz = 1000, 200  # (m)

grid = RectilinearGrid(size = (Nx, Nz), x = (-Lx/2, Lx/2), z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

model = AtmosphereModel(grid; dynamics = CompressibleDynamics(ExplicitTimeStepping()))

# ## Background state
#
# We build a hydrostatically balanced reference state using [`ReferenceState`](@ref).
# This provides the background density and pressure profiles.

constants = model.thermodynamic_constants

θ₀ = 300      # Reference potential temperature (K)
p₀ = 101325   # Surface pressure (Pa)
pˢᵗ = 1e5     # Standard pressure (Pa)

reference = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀, standard_pressure=pˢᵗ)

# The sound speed at the surface determines the acoustic wave propagation speed.

Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
ℂᵃᶜ = sqrt(γ * Rᵈ * θ₀)

# The wind profile follows the classic log-law of the atmospheric surface layer.

U₀ = 20 # Surface velocity (m/s), u★/κ
ℓ = 1   # Roughness length (m), like, shrubs and stuff

Uᵢ(z) = U₀ * log((z + ℓ) / ℓ)

# ## Initial conditions
#
# We initialize a localized Gaussian density pulse representing an acoustic disturbance.
# For a rightward-propagating acoustic wave, the velocity perturbation is in phase with
# the density perturbation: ``u' = (ℂᵃᶜ / ρ₀) ρ'``.

δρ = 0.01    # Density perturbation amplitude (kg/m³)
σ = 20       # Pulse width (m)

gaussian(x, z) = exp(-(x^2 + z^2) / 2σ^2)
ρ₀ = interior(reference.density, 1, 1, 1)[]

ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants) + δρ * gaussian(x, z)
uᵢ(x, z) = Uᵢ(z) # + (ℂᵃᶜ / ρ₀) * δρ * gaussian(x, z)

set!(model, ρ=ρᵢ, θ=θ₀, u=uᵢ)


# ## Simulation setup
#
# Acoustic waves travel fast (``ℂᵃᶜ ≈ 347`` m/s), so we need a small time step.
# The [Courant–Friedrichs–Lewy (CFL) condition](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition) is based on the effective propagation speed ``ℂᵃᶜ + \mathrm{max}(U)``.

Δx, Δz = Lx / Nx, Lz / Nz
Δt = 0.5 * min(Δx, Δz) / (ℂᵃᶜ + Uᵢ(Lz))
stop_time = 0.5 # (s) — long enough for the wave to traverse the domain and for refraction to bend rays visibly

simulation = Simulation(model; Δt, stop_time)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

function progress(sim)
    u, v, w = sim.model.velocities
    msg = @sprintf("Iter: %d, t: %s, max|u|: %.2f m/s, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim),
                   maximum(abs, u), maximum(abs, w))
    @info msg
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Output
#
# We perturbation fields for density and x-velocity for visualization.

ρ = model.dynamics.density
u, v, w = model.velocities

ρᵇᵍ = CenterField(grid)
uᵇᵍ = XFaceField(grid)

set!(ρᵇᵍ, (x, z) -> adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants))
set!(uᵇᵍ, (x, z) -> Uᵢ(z))

ρ′ = Field(ρ - ρᵇᵍ)
u′ = Field(u - uᵇᵍ)

U = Average(u, dims = 1)
R = Average(ρ, dims = 1)
W² = Average(w^2, dims = 1)

filename = "acoustic_wave.jld2"
outputs = (; ρ′, u′, w, U, R, W²)

simulation.output_writers[:jld2] = JLD2Writer(model, outputs; filename,
                                              schedule = TimeInterval(0.01),
                                              overwrite_existing = true)

run!(simulation)

# ## Visualization
#
# Load the saved perturbation fields and create a snapshot.

ρ′ts = FieldTimeSeries(filename, "ρ′")
u′ts = FieldTimeSeries(filename, "u′")
wts = FieldTimeSeries(filename, "w")
Uts = FieldTimeSeries(filename, "U")
Rts = FieldTimeSeries(filename, "R")
W²ts = FieldTimeSeries(filename, "W²")

times = ρ′ts.times
Nt = length(times)

fig = Figure(size = (900, 600), fontsize = 12)

axρ = Axis(fig[1, 2]; aspect = 5, ylabel = "z (m)")
axw = Axis(fig[2, 2]; aspect = 5, ylabel = "z (m)")
axu = Axis(fig[3, 2]; aspect = 5, xlabel = "x (m)", ylabel = "z (m)")
axR = Axis(fig[1, 1]; xlabel = "⟨ρ⟩ (kg/m³)")
axW = Axis(fig[2, 1]; xlabel = "⟨w²⟩ (m²/s²)", limits = (extrema(W²ts), nothing))
axU = Axis(fig[3, 1]; xlabel = "⟨u⟩ (m/s)")

hidexdecorations!(axρ)
hidexdecorations!(axw)
colsize!(fig.layout, 1, Relative(0.2))

n = Observable(Nt)
ρ′n = @lift ρ′ts[$n]
u′n = @lift u′ts[$n]
wn = @lift wts[$n]
Un = @lift Uts[$n]
Rn = @lift Rts[$n]
W²n = @lift W²ts[$n]

ρlim = δρ / 4
ulim = 1

hmρ = heatmap!(axρ, ρ′n; colormap = :balance, colorrange = (-ρlim, ρlim))
hmw = heatmap!(axw, wn; colormap = :balance, colorrange = (-ulim, ulim))
hmu = heatmap!(axu, u′n; colormap = :balance, colorrange = (-ulim, ulim))

lines!(axR, Rn)
lines!(axW, W²n)
lines!(axU, Un)

Colorbar(fig[1, 3], hmρ; label = "ρ′ (kg/m³)")
Colorbar(fig[2, 3], hmw; label = "w (m/s)")
Colorbar(fig[3, 3], hmu; label = "u′ (m/s)")

title = @lift "Acoustic wave in log-layer shear — t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title, fontsize = 16, tellwidth = false)

CairoMakie.record(fig, "acoustic_wave.mp4", 1:Nt, framerate = 18) do nn
    n[] = nn
end
nothing #hide

# ![](acoustic_wave.mp4)

# ---
#
# # A differentiable workflow
#
# The forward simulation above gives us the physics; the rest of the example
# treats that same forward model as a *function* and takes its gradient.  This
# is the minimal pattern you'd reach for whenever you want to do
# data-assimilation, parameter calibration, or sensitivity analysis with a
# Breeze atmosphere — wrap the time-stepping in a scalar-valued `loss`,
# compile it with `Reactant.@compile`, and differentiate it with
# `Enzyme.autodiff`.
#
# The pattern is:
#
#   1. Rebuild the model on a `ReactantState` grid so all arrays are XLA buffers.
#   2. Choose a differentiated input (here, the initial wind field).
#   3. Define a scalar `loss` that re-initializes the model from that input,
#      runs `nsteps` of `time_step!` inside a `@trace` loop, and reduces to a
#      scalar diagnostic.
#   4. Wrap `loss` in a `grad_loss` that calls `Enzyme.autodiff(...)` with the
#      input as `Duplicated` and everything else as `Const`.
#   5. Compile once with `Reactant.@compile raise=true raise_first=true`; run
#      many times.
#
# ### Why Reactant?
#
# Reactant traces Julia code into an intermediate representation (StableHLO)
# that XLA can optimize and Enzyme can differentiate.  The key requirement is
# that the model lives on
# [`ReactantState`](https://clima.github.io/OceananigansDocumentation/stable/appendix/library#Oceananigans.Architectures.ReactantState)
# — Reactant's architecture in Oceananigans — so that all arrays are XLA
# buffers.  We therefore rebuild the *same* physical setup on a new grid whose
# architecture is `ReactantState()`.

using Reactant, CUDA    # CUDA is required for loading the Reactant extension
using Enzyme
using Statistics: mean
using Oceananigans.Architectures: ReactantState
using Reactant: @trace

Reactant.set_default_backend("cpu")

# Rebuild the grid and model on `ReactantState`.

grid_ad = RectilinearGrid(ReactantState(); size = (Nx, Nz),
                          x = (-Lx/2, Lx/2), z = (0, Lz),
                          topology = (Periodic, Flat, Bounded))

model_ad = AtmosphereModel(grid_ad; dynamics = CompressibleDynamics(ExplicitTimeStepping()))

# ### Fixed and varying fields
#
# In this experiment the initial density pulse and hydrostatic background are
# held fixed; only the wind profile varies.  We therefore precompute the total
# initial density once (background + Gaussian pulse) and use it as a `Const`.
# We also keep the standalone background ``\bar\rho(z)`` around so the loss can
# subtract it from the model density to isolate the acoustic perturbation.

ρᵇᵍ = CenterField(grid_ad)
set!(ρᵇᵍ, (x, z) -> adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants))

ρ_total = CenterField(grid_ad)
set!(ρ_total, ρᵢ)

# The initial wind field is the quantity we differentiate with respect to.
# Enzyme accumulates ``∂J / ∂u_i`` into the shadow buffer
# ``du_0``.

u₀  = XFaceField(grid_ad)
du₀ = XFaceField(grid_ad)
set!(u₀, (x, z) -> Uᵢ(z))
set!(du₀, 0)

# The shadow model stores accumulated adjoints for every prognostic field.

dmodel_ad = Enzyme.make_zero(model_ad)

# ### Time step and integration length
#
# We reuse the CFL-based time step and the exact number of iterations from the
# forward simulation above.  The `Simulation` API is not used here because
# Reactant compiles a fixed-length traced loop instead.  Gradient
# checkpointing requires a perfect-square step count, so we round up to the
# next perfect square.

Nt = simulation.model.clock.iteration
Nsteps = (isqrt(Nt - 1) + 1)^2

# ### Defining the objective
#
# We measure the mean squared acoustic density anomaly along the bottom of
# the domain:
#
# ```math
# J \;=\; \frac{1}{N_x}\sum_{i}\bigl[\rho(x_i, z_1) - \bar\rho(x_i, z_1)\bigr]^2
# ```
#
# This is a global measure of how much acoustic energy ends up trapped near
# the surface.  Averaging along the whole bottom row gives a sensitivity field
# that lights up wherever the wind affects *any* part of the surface response,
# making the ducting pattern visible across the entire domain.
#
# The `set!` inside `loss` is what re-initializes the model from the current
# wind field on every backward evaluation.  Without it, AD would differentiate
# a stale trajectory.

function loss(model, u₀, ρ_total, ρᵇᵍ, θ₀, Δt, nsteps)
    set!(model; ρ = ρ_total, θ = θ₀, u = u₀)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    ρ₀  = interior(model.dynamics.density, :, :, 1)
    ρᵇ₀ = interior(ρᵇᵍ, :, :, 1)
    return mean((ρ₀ .- ρᵇ₀).^2)
end

# ### The gradient wrapper
#
# `grad_loss` zeroes the adjoint buffer and calls `Enzyme.autodiff` in reverse
# mode.  The model and the initial wind are `Duplicated` (primal + shadow);
# everything else is `Const`.

function grad_loss(model, dmodel, u₀, du₀, ρ_total, ρᵇᵍ, θ₀, Δt, nsteps)
    parent(du₀) .= 0
    _, J = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(u₀, du₀),
        Enzyme.Const(ρ_total),
        Enzyme.Const(ρᵇᵍ),
        Enzyme.Const(θ₀),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return du₀, J
end

# ### Compilation and execution
#
# `Reactant.@compile` traces the function once to build an XLA executable.
# The flags `raise=true` and `raise_first=true` ensure that every
# KernelAbstractions kernel is "raised" to StableHLO before Enzyme
# differentiates through it — a requirement for the backward pass.

@info "Compiling differentiated model — this may take a minute..."
compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model_ad, dmodel_ad, u₀, du₀,
    ρ_total, ρᵇᵍ, θ₀, Δt, Nsteps)

@info "Running gradient..."
du, J = compiled_grad(
    model_ad, dmodel_ad, u₀, du₀,
    ρ_total, ρᵇᵍ, θ₀, Δt, Nsteps)

xs_u = xnodes(grid_ad, Face())
zs   = znodes(grid_ad, Center())

@info @sprintf("Surface-mean (ρ - ρ̄)² = %.6e after %d steps", Float64(only(J)), Nsteps)

# ### Sensitivity visualization
#
# The heatmap shows ``\partial J / \partial u_i(x,z)``: positive values are
# wind perturbations that would *increase* surface acoustic energy, negative
# values would decrease it.  Because ``J`` integrates along the entire bottom,
# the pattern reveals which parts of the wind profile feed energy into the
# surface duct from anywhere along it.

sensitivity = Array(interior(du, :, 1, :))
abs_max     = maximum(abs, sensitivity)

fig_sens = Figure(size = (800, 350), fontsize = 12)
Label(fig_sens[0, :],
      "∂J / ∂u₀  (J = ⟨(ρ - ρ̄)²⟩ at surface,  t=$(prettytime(Nsteps * Δt))",
      fontsize = 14, tellwidth = false)
ax_sens = Axis(fig_sens[1, 1]; xlabel = "x (m)", ylabel = "z (m)")
hm = heatmap!(ax_sens, xs_u, zs, sensitivity; colormap = :balance,
              colorrange = (-abs_max, abs_max))
Colorbar(fig_sens[1, 2], hm; label = "∂J / ∂u₀")

save("acoustic_wave_wind_sensitivity.png", fig_sens; px_per_unit = 2) #src
fig_sens
