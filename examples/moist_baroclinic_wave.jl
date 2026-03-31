# # Moist baroclinic wave on the sphere
#
# This example extends the dry [`baroclinic_wave`](@ref) test case with moisture
# and warm-phase saturation adjustment microphysics, following test case 4
# of the DCMIP2016 specification [UllrichEtAl2016](@citet).
#
# The dry dynamical core setup — balanced jet, temperature field, and
# localized perturbation — is identical to the dry version. The additions
# are:
#
# 1. An initial specific humidity field concentrated in the lower
#    troposphere with Gaussian decay in both latitude and pressure:
#
#    ```math
#    q^t(φ, z) = q_0 \exp\!\left[-\left(\frac{φ}{φ_w}\right)^4\right]
#                    \exp\!\left[-\left(\frac{(η - 1)\, p_0}{p_w}\right)^2\right]
#    ```
#
#    where ``η = p / p_0`` is the pressure coordinate, ``q_0 = 0.018`` kg/kg,
#    ``φ_w = 40°`` is the meridional width, and ``p_w = 340`` hPa the vertical
#    width. Above the tropopause (``η < 0.1``), moisture is set to a trace value.
#
# 2. One-moment warm-rain bulk microphysics from CloudMicrophysics.jl with
#    non-equilibrium cloud formation. Vapor condenses into cloud liquid via
#    relaxation toward saturation, and rain forms through autoconversion
#    and accretion. This avoids the saturation-adjustment GPU limitations
#    while providing physically meaningful precipitation.
#
# Moisture modifies the baroclinic development through latent heat release
# in frontal ascent regions, intensifying the cyclones and sharpening fronts
# compared to the dry case.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie
using CUDA
using CloudMicrophysics: CloudMicrophysics

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
# The temperature, pressure, density, potential temperature, and zonal
# wind are identical to the dry baroclinic wave test case.

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

## Temperature
function temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end

## Pressure
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

# ### Initial moisture
#
# Specific humidity peaks at the surface near the equator and decays
# with both latitude and altitude. Above the tropopause (``η < 0.1``),
# moisture is set to a trace value ``q_t = 10^{-12}`` kg/kg.

function specific_humidity(λ, φ, z)
    q₀  = 0.018    # kg/kg — surface maximum
    qₜ  = 1e-12    # kg/kg — stratospheric trace value
    φʷ  = 2π / 9   # rad — meridional e-folding width (≈ 40°)
    pʷ  = 34000    # Pa — vertical e-folding pressure width

    p = pressure(λ, φ, z)
    η = p / p₀
    φʳ = deg2rad(φ)

    q_troposphere = q₀ * exp(-(φʳ / φʷ)^4) * exp(-((η - 1) * p₀ / pʷ)^2)

    return ifelse(η > 0.1, q_troposphere, qₜ)
end

# ## Model configuration
#
# We use fully explicit compressible dynamics with one-moment warm-rain
# bulk microphysics from CloudMicrophysics.jl. Non-equilibrium cloud
# formation relaxes vapor toward saturation on a microphysical timescale,
# producing prognostic cloud liquid and rain.

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)

dynamics = CompressibleDynamics(ExplicitTimeStepping(); surface_pressure=p₀)

cloud_formation = NonEquilibriumCloudFormation(nothing, nothing)
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics,
                        thermodynamic_constants = constants,
                        advection = WENO())

# ## Set initial conditions

set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density, qᵛ=specific_humidity)

# ## Time-stepping
#
# With explicit time stepping the time step is limited by the acoustic CFL.
# At 2° resolution, ``Δx ≈ 200`` km and ``c_s ≈ 340`` m/s give ``Δt ≈ 2`` s.

Δt = 2seconds
stop_time = 15days

simulation = Simulation(model; Δt, stop_time)

# ## Polar filter

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
# We save the potential temperature perturbation and cloud liquid water.
# Cloud liquid water ``q^l`` highlights frontal zones where ascending
# moist air condenses.

θ = PotentialTemperature(model)

## Background θ from the initial profile for computing perturbation θ′
θ_bg = CenterField(grid)
set!(θ_bg, potential_temperature)
θ′ = θ - θ_bg

qᶜˡ = model.microphysical_fields.qᶜˡ

outputs = merge(model.velocities, (; θ′, qᶜˡ))

simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                              filename = "moist_baroclinic_wave",
                                              schedule = TimeInterval(1hours),
                                              overwrite_existing = true)

# ## Run

run!(simulation)

# ## Visualization
#
# We compare the potential-temperature perturbation with the cloud
# liquid water field, which reveals condensation along the frontal bands.

θ′_ts = FieldTimeSeries("moist_baroclinic_wave.jld2", "θ′")
qᶜˡ_ts = FieldTimeSeries("moist_baroclinic_wave.jld2", "qᶜˡ")
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
            title = "qᶜˡ at z = $(z_mid/1e3) km, t = $(prettytime(times[Nt]))", sphere_kw...)
hm2 = surface!(ax2, view(qᶜˡ_ts[Nt], :, :, k_mid); colormap = Reverse(:grays), shading = NoShading)
Colorbar(fig[1, 4], hm2; label = "qᶜˡ (kg/kg)")

for ax in (ax1, ax2)
    hidedecorations!(ax)
    hidespines!(ax)
end

current_figure()

# ### Animation
#
# Animate the potential-temperature perturbation and cloud liquid water
# on the sphere over the full simulation:

n = Observable(1)
θ′n = @lift view(θ′_ts[$n], :, :, k_mid)
qᶜˡn = @lift view(qᶜˡ_ts[$n], :, :, k_mid)

fig = Figure(size = (1200, 600))
sphere_kw = (elevation = π/6, azimuth = -π/2, aspect = :data)

title = @lift "z = $(z_mid/1e3) km, t = $(prettytime(times[$n]))"

ax1 = Axis3(fig[1, 1]; title = "θ′", sphere_kw...)
hm1 = surface!(ax1, θ′n; colormap = :balance, colorrange = (-2, 2), shading = NoShading)
Colorbar(fig[1, 2], hm1; label = "θ′ (K)")

ax2 = Axis3(fig[1, 3]; title = "qᶜˡ", sphere_kw...)
hm2 = surface!(ax2, qᶜˡn; colormap = Reverse(:grays), colorrange = (0, 1e-4), shading = NoShading)
Colorbar(fig[1, 4], hm2; label = "qᶜˡ (kg/kg)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

for ax in (ax1, ax2)
    hidedecorations!(ax)
    hidespines!(ax)
end

CairoMakie.record(fig, "moist_baroclinic_wave.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](moist_baroclinic_wave.mp4)
