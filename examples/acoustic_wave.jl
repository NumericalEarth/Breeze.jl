# # Acoustic wave refraction by wind shear
#
# This example simulates an acoustic pulse propagating through a wind shear layer
# using the fully compressible Euler equations. When wind speed increases with height,
# sound waves are refracted: waves traveling **with** the wind bend **downward**
# (trapped near the surface), while waves traveling **against** the wind bend **upward**.
#
# The sound speed for a wave traveling in direction ``\hat{n}`` is
# ```math
# 𝕌ˢ = 𝕌ˢⁱ + \mathbf{u} \cdot \hat{n}
# ```
# where ``𝕌ˢⁱ`` is the intrinsic wave speed and ``\mathbf{u}`` is the wind velocity.
# This causes wavefronts to tilt toward regions of lower effective sound speed.
#
# This phenomenon explains why distant sounds are often heard more clearly downwind
# of a source, as sound energy is "ducted" along the surface. For more on this topic, see
# * Ostashev and Wilson (2015), *Acoustics in Moving Inhomogeneous Media*, CRC Press.
# * Pierce (2019), *Acoustics: An Introduction to Its Physical Principles and Applications*, Springer.
#
# We use stable stratification to suppress Kelvin-Helmholtz instability and a logarithmic
# wind profile consistent with the atmospheric surface layer.

using Breeze
using Breeze.Thermodynamics: adiabatic_hydrostatic_density
using Oceananigans.Units
using Printf
using CairoMakie

# ## Grid and model setup

Nx, Nz = 256, 128
Lx, Lz = 2000, 200  # meters

grid = RectilinearGrid(size = (Nx, Nz), x = (-Lx/2, Lx/2), z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

model = AtmosphereModel(grid; dynamics = CompressibleDynamics())

# ## Background state
#
# We build a hydrostatically balanced reference state using `ReferenceState`.
# This provides the background density and pressure profiles.

constants = model.thermodynamic_constants

θ₀ = 300      # Reference potential temperature (K)
p₀ = 101325   # Surface pressure (Pa)

reference = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)

# The sound speed at the surface determines the acoustic wave propagation speed.

Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
𝕌ˢⁱ = sqrt(γ * Rᵈ * θ₀)

# The wind profile follows the classic log-law of the atmospheric surface layer.

u★ = 10 # Friction velocity (m/s)
κ = 0.4  # von Kármán constant
ℓ = 1.0  # Roughness length [m] -- like, shrubs and stuff

Uᵢ(z) = (u★ / κ) * log((z + ℓ) / ℓ)

# ## Initial conditions
#
# We initialize a localized Gaussian density pulse representing an acoustic disturbance.
# For a rightward-propagating acoustic wave, the velocity perturbation is in phase with
# the density perturbation: ``u' = (𝕌ˢ / ρ₀) ρ'``.

δρ = 0.01         # Density perturbation amplitude (kg/m³)
σ = 20            # Pulse width (m)

gaussian(x, z) = exp(-(x^2 + z^2) / 2σ^2)
ρ₀ = interior(reference.density, 1, 1, 1)[]

ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, constants) + δρ * gaussian(x, z)
uᵢ(x, z) = Uᵢ(z) + (𝕌ˢⁱ / ρ₀) * δρ * gaussian(x, z)

set!(model, ρ=ρᵢ, θ=θ₀, u=uᵢ)


# ## Simulation setup
#
# Acoustic waves travel fast (``𝕌ˢⁱ ≈ 347`` m/s), so we need a small time step.
# The CFL condition is based on the effective sound speed ``𝕌ˢ = 𝕌ˢⁱ + max(U)``.

Δx, Δz = Lx / Nx, Lz / Nz
𝕌ˢ = 𝕌ˢⁱ + Uᵢ(Lz)
Δt = 0.1 * min(Δx, Δz) / 𝕌ˢ
stop_time = 1  # seconds

simulation = Simulation(model; Δt, stop_time)

function progress(sim)
    u, v, w = sim.model.velocities
    msg = @sprintf("Iter: %d, t: %s, max|u|: %.2f m/s, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim),
                   maximum(abs, u), maximum(abs, w))
    @info msg
end

add_callback!(simulation, progress, IterationInterval(500))

# ## Output
#
# We perturbation fields for density and x-velocity for visualization.

ρ = model.dynamics.density
u, v, w = model.velocities

ρᵇᵍ = CenterField(grid)
uᵇᵍ = XFaceField(grid)

set!(ρᵇᵍ, (x, z) -> adiabatic_hydrostatic_density(z, p₀, θ₀, constants))
set!(uᵇᵍ, (x, z) -> Uᵢ(z))

ρ′ = Field(ρ - ρᵇᵍ)
u′ = Field(u - uᵇᵍ)

U = Average(u, dims = 1)
R = Average(ρ, dims = 1)
W² = Average(w^2, dims = 1)

filename = "acoustic_wave.jld2"
outputs = (; ρ′, u′, w, U, R, W²)

simulation.output_writers[:jld2] = JLD2Writer(model, outputs; filename,
                                              schedule = TimeInterval(0.005),
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

axρ = Axis(fig[1, 2]; aspect = 10, ylabel = "z (m)", title = "Density perturbation ρ′",
            xticklabelsvisible = false)
axw = Axis(fig[2, 2]; aspect = 10, xlabel = "x (m)", ylabel = "z (m)", title = "Vertical velocity w")
axu = Axis(fig[3, 2]; aspect = 10, xlabel = "x (m)", ylabel = "z (m)", title = "Velocity perturbation u′")
axR = Axis(fig[1, 1]; width = Relative(0.2), xlabel = "x (m)", ylabel = "z (m)", title = "Horizontal average of density ρ")
axW² = Axis(fig[2, 1]; width = Relative(0.2), xlabel = "x (m)", ylabel = "z (m)", title = "Horizontal average of vertical velocity squared W²")
axU = Axis(fig[3, 1]; width = Relative(0.2), xlabel = "x (m)", ylabel = "z (m)", title = "Horizontal average of velocity u")

n = Observable(Nt)
ρ′n = @lift ρ′ts[$n]
u′n = @lift u′ts[$n]
Un = @lift Uts[$n]
Rn = @lift Rts[$n]
W²n = @lift W²ts[$n]

ρlim = δρ / 2
ulim = 1.5
wlim = 1.5

hmρ = heatmap!(axρ, ρ′n; colormap = :balance, colorrange = (-ρlim, ρlim))
hmw = heatmap!(axw, wn; colormap = :balance, colorrange = (-wlim, wlim))
hmu = heatmap!(axu, u′n; colormap = :balance, colorrange = (-ulim, ulim))

lines(axU, Un; colormap = :balance, colorrange = (-Ulim, Ulim))
lines(axR, Rn; colormap = :balance, colorrange = (-Rlim, Rlim))
lines(axW², W²n; colormap = :balance, colorrange = (-W²lim, W²lim))

Colorbar(fig[1, 3], hmρ; label = "ρ′ (kg/m³)", height = Relative(0.2))
Colorbar(fig[2, 3], hmw; label = "w (m/s)", height = Relative(0.2))
Colorbar(fig[3, 3], hmu; label = "u′ (m/s)", height = Relative(0.2))

title = @lift "Acoustic wave with log-layer shear — t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title, fontsize = 16, tellwidth = false)

CairoMakie.record(fig, "acoustic_wave.mp4", 1:Nt, framerate = 18) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end
nothing #hide

# ![](acoustic_wave.mp4)
