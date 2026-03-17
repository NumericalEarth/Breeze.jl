# # Acoustic wave refraction by wind shear
#
# This example simulates an acoustic pulse propagating through a wind shear layer
# using the fully compressible [Euler equations](https://en.wikipedia.org/wiki/Euler_equations_(fluid_dynamics)).
# When wind speed increases with height, sound waves are refracted: waves traveling **with**
# the wind bend **downward** (trapped near the surface), while waves traveling **against**
# the wind bend **upward**.
#
# The effective propagation speed for a wave traveling in direction ``\hat{\boldsymbol{n}}`` is
# ```math
# \mathbb{C}^{ac} + \boldsymbol{u} \cdot \hat{\boldsymbol{n}}
# ```
# where ``ℂᵃᶜ`` is the acoustic sound speed and ``\boldsymbol{u}`` is the wind velocity.
# This causes wavefronts to tilt toward regions of lower effective propagation speed.
#
# This phenomenon explains why distant sounds are often heard more clearly downwind
# of a source, as sound energy is "ducted" along the surface. For more on this topic, see
#
# ```@bibliography
# ostashev2015acoustics
# pierce2019acoustics
# ```
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
Lx, Lz = 1000, 200  # meters

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

U₀ = 20 # Surface velocity (m/s, u★ / κ)
ℓ = 1  # Roughness length [m] -- like, shrubs and stuff

Uᵢ(z) = U₀ * log((z + ℓ) / ℓ)

# ## Initial conditions
#
# We initialize a localized Gaussian density pulse representing an acoustic disturbance.
# For a rightward-propagating acoustic wave, the velocity perturbation is in phase with
# the density perturbation: ``u' = (ℂᵃᶜ / ρ₀) ρ'``.

δρ = 0.01         # Density perturbation amplitude (kg/m³)
σ = 20            # Pulse width (m)

gaussian(x, z) = exp(-(x^2 + z^2) / 2σ^2)
ρ₀ = interior(reference.density, 1, 1, 1)[]

ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants) + δρ * gaussian(x, z)
uᵢ(x, z) = Uᵢ(z) #+ (ℂᵃᶜ / ρ₀) * δρ * gaussian(x, z)

set!(model, ρ=ρᵢ, θ=θ₀, u=uᵢ)


# ## Simulation setup
#
# Acoustic waves travel fast (``ℂᵃᶜ ≈ 347`` m/s), so we need a small time step.
# The [Courant–Friedrichs–Lewy (CFL) condition](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition) is based on the effective propagation speed ``ℂᵃᶜ + \mathrm{max}(U)``.

Δx, Δz = Lx / Nx, Lz / Nz
Δt = 0.5 * min(Δx, Δz) / (ℂᵃᶜ + Uᵢ(Lz))
stop_time = 1  # seconds

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

# ## Differentiability: sensitivity to the initial perturbation
#
# A natural follow-up question is: *how sensitive is the acoustic field at some
# distant observation point to the shape of the initial density pulse?*
# Answering this with finite differences would require re-running the simulation
# once per grid cell. Automatic differentiation (AD) gives us the full
# sensitivity field in a single backward pass.
#
# We use [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for reverse-mode AD
# and [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) to compile the
# model to XLA so that we can target multiple accelerators (GPU, TPU, etc...) and
# differentiate through it with Enzyme.
#
# ### Why Reactant?
#
# Reactant traces Julia code into an intermediate representation (StableHLO) that
# XLA can optimize and Enzyme can differentiate.  The key requirement is that the
# model lives on `ReactantState` — Reactant's architecture — so that all arrays
# are XLA buffers.  We therefore rebuild the *same* physical setup on a new grid
# whose architecture is `ReactantState()`.  Everything else — domain, resolution,
# thermodynamic constants, wind profile, perturbation shape — is identical.

using CUDA       # required for Reactant extension loading
using Reactant
using Enzyme
using Oceananigans.Architectures: ReactantState
using Reactant: @trace

Reactant.set_default_backend("cpu")

# Rebuild the grid and model on `ReactantState`.

grid_ad = RectilinearGrid(ReactantState(); size = (Nx, Nz),
                          x = (-Lx/2, Lx/2), z = (0, Lz),
                          topology = (Periodic, Flat, Bounded))

model_ad = AtmosphereModel(grid_ad; dynamics = CompressibleDynamics(ExplicitTimeStepping()))

# ### Background fields
#
# The hydrostatic background density and the log-layer wind profile do not
# change between evaluations of the objective, so we precompute them once.
# These are *not* recomputed inside the loss function.

ρᵇᵍ = CenterField(grid_ad)
uᵇᵍ = XFaceField(grid_ad)
set!(ρᵇᵍ, (x, z) -> adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants))
set!(uᵇᵍ, (x, z) -> Uᵢ(z))

# The initial density perturbation is the quantity we differentiate with respect
# to.  We also allocate its adjoint (shadow) field, which Enzyme will fill with
# the gradient ``\partial J / \partial \rho'_0``.

δρᵢ  = CenterField(grid_ad)
dδρᵢ = CenterField(grid_ad)
set!(δρᵢ, (x, z) -> δρ * gaussian(x, z))
set!(dδρᵢ, 0)

# A scratch field for the total initial density (background + perturbation).

ρᵗ = CenterField(grid_ad)

# The shadow model stores accumulated adjoints for every prognostic field.

dmodel_ad = Enzyme.make_zero(model_ad)

# ### Time step and observation point
#
# We reuse the CFL-based time step and the exact number of iterations from the
# forward simulation above.  The `Simulation` API is not used here because
# Reactant compiles a fixed-length traced loop instead.  Gradient checkpointing
# requires a perfect-square step count, so we round up to the next perfect square.

Nt = simulation.model.clock.iteration
Nsteps = (isqrt(Nt - 1) + 1)^2
target_i = round(Int, 0.75Nx)
target_k = round(Int, 0.35Nz)

# ### Defining the objective
#
# The loss function must contain `set!` so that the model is re-initialized from
# the current perturbation field on every evaluation. Without this, the backward
# pass would differentiate a stale trajectory.
#
# Only two things are recomputed each call:
#
# 1. The total initial density ``\rho_0 = \bar\rho(z) + \rho'_0(x,z)`` — because
#    the perturbation field is the input we vary.
# 2. The forward trajectory — because time stepping mutates the model in place.
#
# Everything else (grid, background fields, constants, compiled artifacts) is
# allocated once and reused.

function loss(model, δρᵢ, ρᵗ, ρᵇᵍ, uᵇᵍ, θ₀, Δt, nsteps, it, kt)
    parent(ρᵗ) .= parent(ρᵇᵍ) .+ parent(δρᵢ)
    set!(model; ρ = ρᵗ, θ = θ₀, u = uᵇᵍ)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return interior(model.dynamics.density, it, 1, kt)
end

# ### The gradient wrapper
#
# `grad_loss` zeroes the adjoint buffer and calls `Enzyme.autodiff` in reverse
# mode.  The model and the perturbation field are `Duplicated` (primal + shadow);
# everything else is `Const` (no gradient needed).

function grad_loss(model, dmodel, δρᵢ, dδρᵢ,
                   ρᵗ, ρᵇᵍ, uᵇᵍ, θ₀, Δt, nsteps, it, kt)
    parent(dδρᵢ) .= 0
    _, J = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(δρᵢ, dδρᵢ),
        Enzyme.Const(ρᵗ),
        Enzyme.Const(ρᵇᵍ),
        Enzyme.Const(uᵇᵍ),
        Enzyme.Const(θ₀),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps),
        Enzyme.Const(it),
        Enzyme.Const(kt))
    return dδρᵢ, J
end

# ### Compilation and execution
#
# `Reactant.@compile` traces the function once to build an XLA executable.
# The flags `raise=true` and `raise_first=true` ensure that every
# KernelAbstractions kernel is "raised" to StableHLO before Enzyme
# differentiates through it — a requirement for the backward pass.

@info "Compiling differentiated model — this may take a minute..."
compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model_ad, dmodel_ad, δρᵢ, dδρᵢ,
    ρᵗ, ρᵇᵍ, uᵇᵍ, θ₀, Δt, Nsteps, target_i, target_k)

@info "Running gradient..."
dδρ, J = compiled_grad(
    model_ad, dmodel_ad, δρᵢ, dδρᵢ,
    ρᵗ, ρᵇᵍ, uᵇᵍ, θ₀, Δt, Nsteps, target_i, target_k)

xs = xnodes(grid_ad, Center())
zs = znodes(grid_ad, Center())
x_target = xs[target_i]
z_target = zs[target_k]

@info @sprintf("Receiver density J = %.6e  at (x=%.1f m, z=%.1f m) after %d steps",
               Float64(only(J)), x_target, z_target, Nsteps)

# ### Sensitivity visualization

sensitivity = Array(interior(dδρ, :, 1, :))
sens_min, sens_max = minimum(sensitivity), maximum(sensitivity)

fig_sens = Figure(size = (800, 350), fontsize = 12)
Label(fig_sens[0, :],
      @sprintf("∂ρ / ∂ρ′₀  (receiver at x=%.0f m, z=%.0f m, t=%d Δt)",
               x_target, z_target, Nsteps),
      fontsize = 14, tellwidth = false)
ax_sens = Axis(fig_sens[1, 1]; xlabel = "x (m)", ylabel = "z (m)")
hm = heatmap!(ax_sens, xs, zs, sensitivity; colormap = :balance, colorrange = (sens_min, sens_max))
scatter!(ax_sens, [x_target], [z_target]; color = :black, marker = :star5,
         markersize = 14, label = "receiver")
axislegend(ax_sens; position = :rt)
Colorbar(fig_sens[1, 2], hm; label = "∂J/∂ρ′₀")

save("acoustic_wave_sensitivity.png", fig_sens; px_per_unit = 2)
@info "Saved acoustic_wave_sensitivity.png"

nothing #hide
