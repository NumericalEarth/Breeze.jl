# # Baroclinic wave on the sphere
#
# This example simulates the growth of a baroclinic wave on a near-global
# `LatitudeLongitudeGrid`, inspired by the dynamical core benchmark described
# by [JablonowskiWilliamson2006](@citet).
# A midlatitude jet in thermal wind balance with a meridional temperature
# gradient is seeded with a localized perturbation that triggers baroclinic
# instability, producing growing Rossby waves over roughly ten days.
#
# This is the first spherical-geometry example in Breeze, exercising
# `CompressibleDynamics` with `ExplicitTimeStepping`
# and `HydrostaticSphericalCoriolis` on a latitude-longitude grid spanning
# 85° S to 85° N.
#
# ## Physical setup
#
# The background atmosphere is stably stratified with a constant Brunt-Väisälä
# frequency ``N``, giving a potential-temperature profile
#
# ```math
# θ^{\rm b}(z) = θ_0 \exp\!\left(\frac{N^2 z}{g}\right)
# ```
#
# with ``θ_0 = 300\,{\rm K}`` and ``N^2 = 10^{-4}\,{\rm s^{-2}}``.
#
# ### Meridional temperature gradient
#
# A pole-to-equator temperature difference ``Δθ = 60\,{\rm K}``
# drives the baroclinic instability. The temperature gradient is confined
# to the troposphere (below the tropopause height ``z_T = 15\,{\rm km}``):
#
# ```math
# θ(φ, z) = θ^{\rm b}(z) - Δθ \sin^2 φ \max(0,\, 1 - z/z_T)
# ```
#
# This creates a cold pole / warm equator contrast at the surface that
# weakens linearly with height and vanishes at the tropopause.
#
# ### Balanced zonal jet
#
# The zonal wind is derived from the meridional temperature gradient
# via thermal wind balance. The thermal wind relation on the sphere,
#
# ```math
# f \frac{∂u}{∂z} = -\frac{g}{a θ_0} \frac{∂θ}{∂φ}
# ```
#
# yields a jet in geostrophic balance with the temperature field:
#
# ```math
# u(φ, z) = \frac{g\, Δθ}{a\, θ_0\, Ω}\, \cos φ
#            \times \begin{cases}
#              z - \dfrac{z^2}{2 z_T} & z \le z_T \\[6pt]
#              \dfrac{z_T}{2} & z > z_T
#            \end{cases}
# ```
#
# The ``\cos φ`` factor gives a broad jet that peaks at the equator (~32 m/s)
# and is roughly 22 m/s at 45° latitude.
# By initializing with a balanced state we avoid spurious gravity-wave transients and
# allows baroclinic instability to develop cleanly from the perturbation.
#
# ### Perturbation
#
# A localized potential-temperature Gaussian bump centered at
# ``(λ_c, φ_c) = (90°, 45°)`` seeds the instability:
#
# ```math
# θ'(λ, φ, z) = Δθ \exp\!\left(-\frac{(λ - λ_c)^2 + (φ - φ_c)^2}{2σ^2}\right)
#                \sin\!\left(\frac{π z}{H}\right)
# ```
#
# with amplitude ``Δθ = 1\,{\rm K}`` and width ``σ = 10°``.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie
using CUDA

# ## Domain and grid
#
# We use a near-global latitude-longitude grid at roughly 2° horizontal
# resolution, excluding the poles to avoid the coordinate singularity.
# The domain extends from the surface to 30 km with 30 vertical levels.

Nλ = 180
Nφ = 85
Nz = 30
H  = 30kilometers

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-85, 85),
                             z = (0, H))

# ## Physical parameters

constants = ThermodynamicConstants()
g  = constants.gravitational_acceleration
p₀ = 100000 # Pa — surface pressure
θ₀ = 300    # K — surface potential temperature
N² = 1e-4   # s⁻² — Brunt-Väisälä frequency squared

# Background potential temperature with stable stratification:

θᵇ(z) = θ₀ * exp(N² * z / g)

# ## Model configuration
#
# We use fully explicit compressible dynamics — all tendencies including
# acoustic modes are advanced together, so the time step must resolve
# sound waves (``Δt ≲ Δz / c_s ≈ 3`` s for 30 levels over 30 km).
# The reference state uses the stratified ``θᵇᵍ(z)`` profile, so the buoyancy
# force is computed as a perturbation ``ρ b = -g(ρ - ρ_r)`` for accuracy.
# `HydrostaticSphericalCoriolis` retains the traditional ``f = 2Ω\sin φ``
# Coriolis terms.

coriolis = HydrostaticSphericalCoriolis()

dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵇ)

model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO())

# ## Initial conditions
#
# The temperature field combines the background stratification, a meridional
# gradient, and a localized perturbation. The zonal wind is derived analytically
# from the thermal wind relation for the meridional gradient.

Ω     = coriolis.rotation_rate               # s⁻¹ — Earth rotation rate
a     = Oceananigans.defaults.planet_radius  # m — Earth radius
Δθ    = 60                                   # K — equator-to-pole θ difference
z_T   = 15_000                               # m — tropopause height
τ_bal = g * Δθ / (a * θ₀ * Ω)                # s⁻¹ — thermal wind parameter

# Perturbation parameters:
λ_c = 90  # degrees — perturbation center longitude
φ_c = 45  # degrees — perturbation center latitude
σ   = 10  # degrees — Gaussian half-width
Δθ  = 1   # K — perturbation amplitude

# Balanced zonal wind from the thermal wind relation:

function uᵢ(λ, φ, z)
    φ_rad = φ * π / 180
    vertical = ifelse(z ≤ z_T, z - z^2 / 2z_T, z_T / 2)
    return τ_bal * vertical * cos(φ_rad) # m/s
end

# Potential temperature: background + meridional gradient + perturbation:

function θᵢ(λ, φ, z)
    θ_merid = - Δθ * sind(φ)^2 * max(0, 1 - z / z_T)

    r² = (λ - λ_c)^2 + (φ - φ_c)^2
    θ_pert = Δθ * exp(-r² / 2σ^2) * sin(π * z / H)
    return θᵇ(z) + θ_merid + θ_pert
end

# ### Hydrostatic density
#
# The density must be in hydrostatic balance with the full ``θ(φ, z)`` field
# (not just the 1D reference profile). We integrate the Exner function
# from the surface for each column:
#
# ```math
# \frac{\mathrm{d}Π}{\mathrm{d}z} = -\frac{κ\, g}{R^d\, θ(φ, z)}
# ```
#
# and then recover ``ρ = p_0\, Π^{c_v/R^d} / (R^d\, θ)``.

Rᵈ = dry_air_gas_constant(constants)
cᵖ = constants.dry_air.heat_capacity
κ  = Rᵈ / cᵖ
cᵥ_over_Rᵈ = (cᵖ - Rᵈ) / Rᵈ

function ρᵢ(λ, φ, z)
    nsteps = max(1, round(Int, z / 100)) # ~100 m steps
    dz = z / nsteps
    Π = 1.0 # Exner at surface (pˢᵗ = p₀)
    for n in 1:nsteps
        zn = (n - 1/2) * dz
        θn = θᵢ(λ, φ, zn)
        Π -= κ * g / (Rᵈ * θn) * dz
    end
    θ = θᵢ(λ, φ, z)
    return p₀ * Π^cᵥ_over_Rᵈ / (Rᵈ * θ)
end

set!(model, θ=θᵢ, u=uᵢ, ρ=ρᵢ)

# ## Time-stepping
#
# With explicit time stepping, the time step is limited by the acoustic CFL:
# ``Δt ≲ Δz / c_s``, where ``c_s ≈ 340`` m/s is the speed of sound.
# For ``Δz = 1`` km this gives ``Δt ≈ 3`` s. We run for 10 days to
# observe baroclinic wave growth.

Δt = 2seconds
stop_time = 10days

simulation = Simulation(model; Δt, stop_time)

# Progress callback:

function progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("Iter %5d | t = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# ## Output
#
# We save the velocities and the potential temperature perturbation (i.e., the
# departure from background stratification) for visualization.

θ = PotentialTemperature(model)

θᵇᵍ = CenterField(grid)
set!(θᵇᵍ, (λ, φ, z) -> θᵇ(z))
θ′ = θ - θᵇᵍ

outputs = merge(model.velocities, (; θ′))

simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                              filename = "baroclinic_wave",
                                              schedule = TimeInterval(3hours),
                                              overwrite_existing = true)

# ## Run

run!(simulation)

# ## Visualization
#
# We plot the potential-temperature perturbation ``θ'`` (departure from the
# horizontally uniform background ``θ^{\rm b}(z)``) and the zonal wind
# on the sphere. Oceananigans' Makie extension converts fields on a
# `LatitudeLongitudeGrid` to spherical coordinates automatically when
# plotted with `surface!` on an `Axis3`.

θ′_ts = FieldTimeSeries("baroclinic_wave.jld2", "θ′")
u_ts  = FieldTimeSeries("baroclinic_wave.jld2", "u")
times = θ′_ts.times
Nt = length(times)

# Select the mid-level index for horizontal slices:
k_mid = Nz ÷ 2
z_mid = znode(k_mid, grid, Center())

# ### Final snapshot on the sphere

fig = Figure(size = (1200, 600))
sphere_kw = (elevation = π/6, azimuth = -π/2, aspect = :data)

ax1 = Axis3(fig[1, 1];
            title = "θ′ at z = $(z_mid/1e3) km, t = $(prettytime(times[Nt]))", sphere_kw...)
plt1 = surface!(ax1, view(θ′_ts[Nt], :, :, k_mid); colormap = :balance, shading = NoShading)
Colorbar(fig[1, 2], plt1; label = "θ′ (K)")

ax2 = Axis3(fig[1, 3];
            title = "u at z = $(z_mid/1e3) km, t = $(prettytime(times[Nt]))", sphere_kw...)
plt2 = surface!(ax2, view(u_ts[Nt], :, :, k_mid); colormap = :balance, shading = NoShading)
Colorbar(fig[1, 4], plt2; label = "u (m/s)")

for ax in (ax1, ax1)
    hidedecorations!(ax)
    hidespines!(ax)
end

current_figure()

# ### Animation
#
# Animate the potential-temperature perturbation on the sphere over
# the full simulation:

n = Observable(1)
θ′_n = @lift view(θ′_ts[$n], :, :, k_mid)

fig = Figure(size = (800, 600))

title = @lift "θ′ at z = $(z_mid/1e3) km, t = $(prettytime(times[$n]))"

ax = Axis3(fig[1, 1]; title, sphere_kw...)
hm = surface!(ax, θ′_n; colormap = :balance, shading = NoShading)

hidedecorations!(ax)
hidespines!(ax)
Colorbar(fig[1, 2], hm; label = "θ′ (K)")

CairoMakie.record(fig, "baroclinic_wave.mp4", 1:Nt; framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](baroclinic_wave.mp4)
