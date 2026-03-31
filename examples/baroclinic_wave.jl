# # Baroclinic wave on the sphere
#
# This example simulates the growth of a baroclinic wave on a near-global
# `LatitudeLongitudeGrid` following the DCMIP2016 specification
# [UllrichEtAl2016](@citet), which extends the classic
# [JablonowskiWilliamson2006](@citet) test case.
# A midlatitude jet in thermal-wind balance with a meridional temperature
# gradient is seeded with a localized zonal-wind perturbation that triggers
# baroclinic instability, producing growing Rossby waves over roughly ten days.
#
# This example exercises `CompressibleDynamics` with `ExplicitTimeStepping`
# and `HydrostaticSphericalCoriolis` on a latitude-longitude grid spanning
# 85° S to 85° N.
#
# ## Physical setup
#
# The background state is an analytic steady-state solution of the dry,
# adiabatic, inviscid primitive equations in height coordinates.
# The temperature field has two parts: a horizontally uniform stratification
# controlled by a lapse rate ``Λ`` and a meridional gradient that creates
# warm equator / cold pole contrast:
#
# ```math
# T(φ, z) = \frac{1}{τ_1(z) - τ_2(z)\, F(φ)}
# ```
#
# where ``τ_1`` and ``τ_2`` encode the vertical structure and
# ``F(φ) = \cos^K φ - \frac{K}{K+2} \cos^{K+2} φ`` is the meridional shape
# with jet-width parameter ``K = 3``.
#
# ### Balanced zonal jet
#
# The zonal wind is derived analytically from gradient-wind balance,
# producing a subtropical jet peaking near 30 m/s at 45° latitude
# in the upper troposphere.
#
# ### Perturbation
#
# A localized zonal-wind perturbation centered at
# ``(λ_c, φ_c) = (20°\text{E}, 40°\text{N})`` seeds the instability.
# The perturbation decays exponentially with great-circle distance from the
# center and is tapered smoothly to zero above 15 km:
#
# ```math
# u'(λ, φ, z) = u_p \, \mathcal{T}(z) \, \exp\!\left(-\left(\frac{d}{r_p}\right)^2\right)
# ```
#
# where ``d`` is the great-circle distance, ``r_p = 0.1\,a``, ``u_p = 1`` m/s,
# and ``\mathcal{T}(z) = 1 - 3(z/z_p)^2 + 2(z/z_p)^3`` for ``z < z_p``.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie
using CUDA

# ## DCMIP2016 parameters
#
# All parameters follow the DCMIP2016 test case document
# [UllrichEtAl2016](@citet). We set the Oceananigans defaults and build a
# custom [`ThermodynamicConstants`](@ref) matching the DCMIP specification
# so that the grid, Coriolis, and model thermodynamics are all consistent
# with the analytic initial conditions.

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220.0
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5    # Pa — surface pressure
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

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

## Temperature profile parameters
Tᴱ = 310.0   # K — equatorial surface temperature
Tᴾ = 240.0   # K — polar surface temperature
Tₘ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005    # K/m — lapse rate
K  = 3        # jet width parameter
b  = 2        # vertical half-width parameter

# ## Analytic initial conditions
#
# The temperature and pressure are computed from the DCMIP2016 analytic
# formulas. The vertical structure functions ``τ_1, τ_2`` and their
# integrals encode the stratification and meridional gradient.

## Vertical structure functions (shallow atmosphere, X = 1)
function τ_and_integrals(z)
    Hₛ = Rᵈ * Tₘ / g
    η  = z / (b * Hₛ)
    e  = exp(-η^2)

    A = (Tₘ - Tᴾ) / (Tₘ * Tᴾ)
    C = (K + 2) / 2 * (Tᴱ - Tᴾ) / (Tᴱ * Tᴾ)

    τ₁  = exp(Γ * z / Tₘ) / Tₘ + A * (1 - 2η^2) * e
    τ₂  = C * (1 - 2η^2) * e
    ∫τ₁ = (exp(Γ * z / Tₘ) - 1) / Γ + A * z * e
    ∫τ₂ = C * z * e

    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

## Meridional shape functions
F(φ)  = cosd(φ)^K - K / (K + 2) * cosd(φ)^(K + 2)
dF(φ) = cosd(φ)^(K - 1) - cosd(φ)^(K + 1)

## Temperature: T(φ, z) = 1 / (τ₁ - τ₂ F(φ))
function temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end

## Pressure: p(φ, z) = p₀ exp(-g/Rᵈ (∫τ₁ - ∫τ₂ F(φ)))
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end

## Density from the ideal gas law
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * temperature(λ, φ, z))

## Potential temperature: θ = T (p₀/p)^κ
function potential_temperature(λ, φ, z)
    T = temperature(λ, φ, z)
    p = pressure(λ, φ, z)
    return T * (p₀ / p)^κ
end

# ### Balanced zonal wind
#
# The zonal wind satisfies gradient-wind balance with the temperature field.
# For the shallow atmosphere (``r = a``):
#
# ```math
# u = -Ω a \cos φ + \sqrt{Ω^2 a^2 \cos^2 φ + a \cos φ \, U(φ, z)}
# ```
#
# where ``U = (g/a) K \int τ_2 \, T \, (\cos^{K-1} φ - \cos^{K+1} φ)``.

function zonal_velocity(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    T = temperature(λ, φ, z)

    ## Gradient-wind balance
    U = g / a * K * ∫τ₂ * dF(φ) * T
    rcosφ  = a * cosd(φ)
    Ωrcosφ = Ω * rcosφ
    u_balanced = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)

    ## Localized perturbation (DCMIP2016 §3.3)
    uₚ = 1.0       # m/s — amplitude
    rₚ = 0.1       # perturbation radius (Earth radii)
    λₚ = π / 9     # 20°E center longitude
    φₚ = 2π / 9    # 40°N center latitude
    zₚ = 15000.0   # m — height cap

    φʳ = deg2rad(φ)
    λʳ = deg2rad(λ)
    great_circle = acos(sin(φₚ) * sin(φʳ) + cos(φₚ) * cos(φʳ) * cos(λʳ - λₚ)) / rₚ

    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, 0.0)
    u_perturbation = ifelse(great_circle < 1, uₚ * taper * exp(-great_circle^2), 0.0)

    return u_balanced + u_perturbation
end

# ## Model configuration
#
# We use fully explicit compressible dynamics with no reference state
# subtraction. The vertical momentum equation computes the full pressure
# gradient and gravitational force directly: ``∂(ρw)/∂t = -∂p/∂z - ρg + \ldots``
# This avoids errors from a 1D reference state that cannot capture the
# meridional density structure of the balanced jet.
# `HydrostaticSphericalCoriolis` retains the traditional ``f = 2Ω \sin φ``
# Coriolis terms.

coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)

dynamics = CompressibleDynamics(ExplicitTimeStepping(); surface_pressure = p₀)

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO())

# ## Set initial conditions

set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density)

# ## Time-stepping
#
# With explicit time stepping the time step is limited by the acoustic CFL.
# For ``Δx ≈ 200`` km and sound speed ``c_s ≈ 340`` m/s,
# the acoustic CFL gives ``Δt ≈ 2`` s.
# We run for 15 days to observe baroclinic wave growth; the instability
# becomes visible around day 4 and develops explosive cyclogenesis near day 8.

Δt = 2seconds
stop_time = 15days

simulation = Simulation(model; Δt, stop_time)

# ## Polar filter
#
# On a latitude-longitude grid the zonal grid spacing shrinks as
# ``Δx(φ) = a \cos φ \, Δλ``, reaching roughly one-eleventh of the equatorial
# value at 85°. The polar filter damps unresolvable high-wavenumber zonal
# modes poleward of 60° using a batched spectral truncation, following the
# WRF approach ([Skamarock et al., 2008](@cite Skamarock2008Description)).

add_polar_filter!(simulation; threshold_latitude=60)

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
# We save the velocities and the potential temperature for visualization.
# Also save surface pressure (bottom-level pressure) for comparison with
# published DCMIP reference solutions.

θ = PotentialTemperature(model)

## Background θ from the initial profile for computing perturbation θ′
θ_bg = CenterField(grid)
set!(θ_bg, potential_temperature)
θ′ = θ - θ_bg

outputs = merge(model.velocities, (; θ′))

simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                              filename = "baroclinic_wave",
                                              schedule = TimeInterval(1hours),
                                              overwrite_existing = true)

# ## Run

run!(simulation)

# ## Visualization
#
# We plot the potential-temperature perturbation ``θ'`` (departure from the
# equatorial background ``θ^{\rm ref}(z)``) and the zonal wind on the sphere.

θ′_ts = FieldTimeSeries("baroclinic_wave.jld2", "θ′")
u_ts = FieldTimeSeries("baroclinic_wave.jld2", "u")
w_ts = FieldTimeSeries("baroclinic_wave.jld2", "w")
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
hm1 = surface!(ax1, view(θ′_ts[Nt], :, :, k_mid); colormap = :balance, shading = NoShading)
Colorbar(fig[1, 2], hm1; label = "θ′ (K)")

ax2 = Axis3(fig[1, 3];
            title = "u at z = $(z_mid/1e3) km, t = $(prettytime(times[Nt]))", sphere_kw...)
hm2 = surface!(ax2, view(u_ts[Nt], :, :, k_mid); colormap = :speed, shading = NoShading)
Colorbar(fig[1, 4], hm2; label = "u (m/s)")

for ax in (ax1, ax2)
    hidedecorations!(ax)
    hidespines!(ax)
end

current_figure()

# ### Animation
#
# Animate the potential-temperature perturbation and the vertical velocity
# on the sphere over the full simulation:

n = Observable(1)
θ′n = @lift view(θ′_ts[$n], :, :, k_mid)
wn = @lift view(w_ts[$n], :, :, k_mid)

fig = Figure(size = (1200, 600))
sphere_kw = (elevation = π/6, azimuth = -π/2, aspect = :data)

title = @lift "z = $(z_mid/1e3) km, t = $(prettytime(times[$n]))"

ax1 = Axis3(fig[1, 1]; title = "θ′", sphere_kw...)
hm1 = surface!(ax1, θ′n; colormap = :balance, colorrange = (-2, 2), shading = NoShading)
Colorbar(fig[1, 2], hm1; label = "θ′ (K)")

ax2 = Axis3(fig[1, 3]; title = "w", sphere_kw...)
hm2 = surface!(ax2, wn; colormap = :balance, colorrange = (-1, 1), shading = NoShading)
Colorbar(fig[1, 4], hm2; label = "w (m/s)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

for ax in (ax1, ax2)
    hidedecorations!(ax)
    hidespines!(ax)
end

CairoMakie.record(fig, "baroclinic_wave.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](baroclinic_wave.mp4)
