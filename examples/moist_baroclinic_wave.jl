# # Moist baroclinic wave on the sphere
#
# This example extends the dry [`baroclinic_wave`](@ref) test case with moisture
# and mixed-phase microphysics, following test case 1-1 of the DCMIP2016
# specification [UllrichEtAl2016](@citet). It uses the same fully compressible
# dynamics with [`SplitExplicitTimeDiscretization`](@ref) acoustic substepping
# as the dry baroclinic wave, but adds:
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
# 2. **Virtual temperature balance**: the analytic balanced state is expressed
#    in virtual temperature ``T_v(\\varphi, z)``. The actual temperature
#    ``T = T_v / (1 + \\epsilon q)`` is colder than the dry value at the same
#    ``T_v``, but the pressure-gradient force is unchanged because density uses
#    ``\\rho = p / (R_d T_v)``.
#
# 3. **One-moment mixed-phase bulk microphysics** from CloudMicrophysics.jl with
#    non-equilibrium cloud formation including ice. Vapor relaxes toward
#    saturation on a microphysical timescale, producing prognostic cloud liquid,
#    cloud ice, rain, and snow.
#
# 4. **Bulk surface fluxes** of momentum, sensible heat, and water vapor over
#    a prescribed sea-surface temperature equal to the analytic surface virtual
#    temperature. Latent heat release from the surface fluxes intensifies the
#    cyclogenesis and sharpens the moist fronts compared to the dry case.

using Breeze
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
# 1° resolution on `latitude=(-80, 80)` matches the dry [`baroclinic_wave`](@ref)
# example, which avoids the polar Δx_min trap and lets us run with a larger
# outer time step. The vertical resolution `Nz = 64` (Δz ≈ 470 m) is fine enough
# for the moist physics; the WENO(5) advection needs `halo = (5, 5, 5)`.

Nλ = 360
Nφ = 160
Nz = 64
H  = 30kilometers

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-80, 80),
                             z = (0, H))

## Temperature profile parameters
Tᴱ = 310.0   # K — equatorial surface temperature
Tᴾ = 240.0   # K — polar surface temperature
Tₘ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005    # K/m — lapse rate
K  = 3        # jet width parameter
b  = 2        # vertical half-width parameter
ε_v = 0.608   # virtual-temperature factor (R_v/R_d − 1)

# ## Analytic initial conditions
#
# The DCMIP2016 balanced state is given in *virtual* temperature ``T_v``.
# In the dry case ``T_v = T``; in the moist case the actual temperature
# ``T = T_v / (1 + \\epsilon q)`` is colder. Density always uses ``T_v``
# in the ideal gas law so that the analytic pressure-gradient force is
# preserved.

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

## Virtual temperature (DCMIP2016 Eq. 9)
function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end

## Pressure (DCMIP2016 Eq. 10)
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end

## Moist density: ρ = p / (Rᵈ Tᵥ)
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

## Actual potential temperature: θ = T (p₀/p)^κ where T = Tᵥ / (1 + ε q)
function potential_temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z)
    p  = pressure(λ, φ, z)
    q  = specific_humidity(λ, φ, z)
    T  = Tᵥ / (1 + ε_v * q)
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
# We use fully compressible dynamics with **acoustic substepping** via
# [`SplitExplicitTimeDiscretization`](@ref) and the [`AcousticRungeKutta3`](@ref)
# (Wicker–Skamarock RK3) outer loop, exactly as in the dry
# [`baroclinic_wave`](@ref) example. The default damping
# ([`PressureProjectionDamping`](@ref) at ``\\beta_d = 0.5``) is the BCI-tuned
# coefficient from the empirical sweep in
# `appendix/bw_dt_sweep_results.md`.
#
# Microphysics: one-moment **mixed-phase** bulk scheme from
# CloudMicrophysics.jl with non-equilibrium cloud formation. Both liquid
# and ice phases are active, so we get cloud liquid, cloud ice, rain, and
# snow as prognostic species.
#
# We use a hydrostatically-balanced isothermal reference state at
# ``T_0 = 250`` K (matching the dry example) so the substepper's slow
# tendencies see only perturbations from the background.

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)

T₀_ref = 250.0
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θ_ref)

# Use a `ConstantRateCondensateFormation` with τ_relax = 200 s for both the
# liquid and ice phases. The CloudMicrophysics default τ_relax (≈ 40 s) is
# shorter than our outer Δt, which makes the explicit WS-RK3 update overshoot
# the saturation value by ~50% per outer step and the moist BW blows up within
# a few outer steps. 200 s is comfortably non-stiff at Δt ≤ 30 s and still
# resolves the BCI moist physics over the day-7-to-15 lifecycle. (Future work:
# subcycle the microphysics inside one outer Δt so that the default τ_relax can
# be used safely.)
τ_relax = 200.0
relaxation = ConstantRateCondensateFormation(1 / τ_relax)
cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

# ## Surface fluxes
#
# Bulk drag, sensible heat flux, and water-vapor flux at the bottom of the
# atmosphere over a prescribed sea-surface temperature equal to the analytic
# surface virtual temperature. ``C_D = 10^{-3}`` is a typical bulk exchange
# coefficient and ``U_g = 10^{-2}`` m/s is a small gustiness floor.

Cᴰ = 1e-3
Uᵍ = 1e-2
T_surface(λ, φ) = virtual_temperature(λ, φ, 0.0)

ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρθ_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))

boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵛ=ρqᵛ_bcs)

# WENO with `bounds=(0, 1)` is bounds-preserving for the moisture mass
# fractions: it forbids the spurious negative or super-unity overshoots that
# vanilla WENO can produce on a sharp tropopause moisture jump, which would
# otherwise feed an unphysical microphysics tendency at a single grid point.
# Momentum and ρθ use the standard WENO since they have no `[0, 1]` bound.
weno = WENO()
bounds_preserving_weno = WENO(order=5, bounds=(0, 1))
momentum_advection = weno
scalar_advection = (ρθ  = weno,
                    ρqᵛ  = bounds_preserving_weno,
                    ρqᶜˡ = bounds_preserving_weno,
                    ρqᶜⁱ = bounds_preserving_weno,
                    ρqʳ  = bounds_preserving_weno,
                    ρqˢ  = bounds_preserving_weno)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics, boundary_conditions,
                        thermodynamic_constants = constants,
                        momentum_advection,
                        scalar_advection,
                        timestepper = :AcousticRungeKutta3)

# ## Set initial conditions

set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density, qᵛ=specific_humidity)

# ## Time-stepping
#
# Acoustic substepping eliminates the acoustic CFL constraint on the outer Δt,
# so we can run at a much larger ``Δt`` than the original moist baroclinic
# wave example (which used ``Δt ≈ 2`` s for explicit acoustics). On this 1°
# `latitude=(-80, 80)` grid the *advective* CFL with ``U_{\\max} ≈ 60`` m/s at
# the BCI peak comfortably allows ``Δt ≈ 60`` s; the binding constraint here
# is the **microphysics-dynamics coupling stiffness** instead. With
# ``τ_{\\rm relax} = 200`` s and explicit (per-stage) microphysics tendencies,
# the rain autoconversion + accretion processes have an effective stiffness
# limit of ``Δt ≈ 20`` s on this grid: at ``Δt = 60`` s the integration NaNs
# within ~25 outer steps, at ``Δt = 30`` s a slow ``2 Δt`` oscillation grows,
# and at ``Δt = 20`` s the integration is rock-stable with ``\\max |w|`` of a
# few cm/s. This is still ~10× larger than the explicit acoustic-CFL ``Δt`` of
# the original moist BW example. (Future work: subcycle the microphysics
# inside one outer Δt so the dynamics CFL is the only limit.)

Δt = 20.0
stop_time = 15days

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
