# # Dry and moist baroclinic waves on the sphere
#
# This example simulates the growth of a baroclinic wave on a near-global
# `LatitudeLongitudeGrid` following the DCMIP2016 specification
# [UllrichEtAl2016](@citet), which extends the classic
# [JablonowskiWilliamson2006](@citet) test case.
# A midlatitude jet in thermal-wind balance with a meridional temperature
# gradient is seeded with a localized zonal-wind perturbation that triggers
# baroclinic instability, producing growing Rossby waves over roughly ten days.
#
# We run the test case in **two configurations**:
#
# 1. **Dry** — the original DCMIP2016 §4.1 setup: no moisture, no microphysics,
#    no surface fluxes. The instability grows from gradient-wind balance only.
# 2. **Moist** — DCMIP2016 §1.1 with mixed-phase one-moment microphysics from
#    [CloudMicrophysics.jl](https://clima.github.io/CloudMicrophysics.jl/) and
#    bulk surface fluxes of momentum, sensible heat, and water vapor over a
#    prescribed sea-surface temperature. Latent heat release sharpens the
#    moist fronts and intensifies cyclogenesis compared to the dry case.
#
# Both configurations exercise `CompressibleDynamics` with
# `SplitExplicitTimeDiscretization` (acoustic substepping via
# [`AcousticRungeKutta3`](@ref)) and `HydrostaticSphericalCoriolis` on a 1°
# latitude-longitude grid spanning 80° S to 80° N. The substepper handles
# the fast acoustic-mode pressure gradient and buoyancy via a vertically-
# implicit inner loop, so the outer time step is set by the *advective* CFL
# rather than the much-tighter acoustic CFL.
#
# ## Physical setup
#
# The DCMIP2016 background state is an analytic steady-state solution of
# the dry, adiabatic, inviscid primitive equations in height coordinates,
# expressed in **virtual** temperature ``T_v(\varphi, z)``:
#
# ```math
# T_v(φ, z) = \frac{1}{τ_1(z) - τ_2(z)\, F(φ)}
# ```
#
# where ``τ_1`` and ``τ_2`` encode the vertical structure and
# ``F(φ) = \cos^K φ - \frac{K}{K+2} \cos^{K+2} φ`` is the meridional shape
# with jet-width parameter ``K = 3``.
# In the dry case, ``T = T_v``; in the moist case, the actual temperature is
# colder, ``T = T_v / (1 + \epsilon q^v)`` with ``\epsilon = R_v/R_d - 1
# \approx 0.608``. Density uses ``T_v`` in the ideal gas law in both cases,
# so the analytic pressure-gradient force is preserved.
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
# center and is tapered smoothly to zero above 15 km.
#
# ### Initial moisture (moist case)
#
# Specific humidity peaks at the surface near the equator and decays
# with both latitude and altitude:
#
# ```math
# q^v(φ, z) = q_0 \exp\!\left[-\left(\frac{φ}{φ_w}\right)^4\right]
#                 \exp\!\left[-\left(\frac{(η - 1)\, p_0}{p_w}\right)^2\right]
# ```
#
# where ``η = p / p_0`` is the pressure coordinate, ``q_0 = 0.018`` kg/kg,
# ``φ_w = 40°``, and ``p_w = 340`` hPa. Above the tropopause (``η < 0.1``),
# moisture is set to a trace value.

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
# We use a 1° latitude-longitude grid spanning 80° S to 80° N. Capping the
# domain at ±80° (rather than ±85°) keeps the polar `Δx_min` at a manageable
# `a · cos 80° · 2π/Nλ ≈ 19.3 km` instead of the ≈ 9.7 km we would have at
# 85°. The domain extends from the surface to 30 km with 64 vertical
# levels (Δz ≈ 470 m). The same grid is used by both the dry and moist runs.

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
# We define a single set of pointwise functions that work for both dry and
# moist cases — moisture enters through the `specific_humidity` argument to
# `potential_temperature`. The dry run will pass ``q^v = 0``; the moist run
# will pass the DCMIP profile.

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

## Virtual temperature: Tᵥ(φ, z) = 1 / (τ₁ - τ₂ F(φ))
function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end

## Pressure: p(φ, z) = p₀ exp(-g/Rᵈ (∫τ₁ - ∫τ₂ F(φ)))
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end

## Density (uses virtual temperature in the ideal gas law)
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

## DCMIP2016 specific humidity profile (used by the moist run only)
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

## Potential temperature: θ = T (p₀/p)^κ where T = Tᵥ / (1 + ε qᵛ)
dry_potential_temperature(λ, φ, z) =
    virtual_temperature(λ, φ, z) * (p₀ / pressure(λ, φ, z))^κ

function moist_potential_temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z)
    p  = pressure(λ, φ, z)
    q  = specific_humidity(λ, φ, z)
    T  = Tᵥ / (1 + ε_v * q)
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
# where ``U = (g/a) K \int τ_2 \, T_v \, (\cos^{K-1} φ - \cos^{K+1} φ)``.

function zonal_velocity(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    Tᵥ = virtual_temperature(λ, φ, z)

    ## Gradient-wind balance
    U = g / a * K * ∫τ₂ * dF(φ) * Tᵥ
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

# ## Shared model configuration
#
# Both dry and moist runs use fully compressible dynamics with **acoustic
# substepping** via [`SplitExplicitTimeDiscretization`](@ref) and the
# [`AcousticRungeKutta3`](@ref) (Wicker–Skamarock RK3) outer loop.
# We use a hydrostatically-balanced isothermal reference state at
# ``T_0 = 250`` K (matching the MPAS convention) so that the substepper's
# slow tendencies see only perturbations from the background.
# `HydrostaticSphericalCoriolis` retains the traditional ``f = 2Ω \sin φ``
# Coriolis terms.

coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)

T₀_ref = 250.0
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θ_ref)

# # Run 1: Dry baroclinic wave
#
# ## Dry model
#
# The dry case has no microphysics and no surface fluxes. The outer time
# step is set by the advective CFL with ``U_{\max} ≈ 60`` m/s at the BCI
# peak: ``Δt ≈ 0.7 \cdot Δx_{\min} / U_{\max} \approx 225`` s. This is
# roughly **110× larger** than the acoustic-CFL-limited Δt = 2 s the fully
# explicit solver requires for the same grid. We run for 14 days to observe
# the full BCI lifecycle — the instability becomes visible around day 4
# and develops explosive cyclogenesis near day 8.

dry_model = AtmosphereModel(grid; dynamics, coriolis,
                            thermodynamic_constants = constants,
                            advection = WENO(),
                            timestepper = :AcousticRungeKutta3)

set!(dry_model, θ=dry_potential_temperature, u=zonal_velocity, ρ=density)

dry_simulation = Simulation(dry_model; Δt = 225seconds, stop_time = 14days)
Oceananigans.Diagnostics.erroring_NaNChecker!(dry_simulation)

function dry_progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("[dry] Iter %5d | t = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, w))
    return nothing
end
add_callback!(dry_simulation, dry_progress, IterationInterval(100))

# Save θ′ (departure from the equatorial background) and the velocities for
# visualization:

dry_θ = PotentialTemperature(dry_model)
dry_θ_bg = CenterField(grid)
set!(dry_θ_bg, dry_potential_temperature)
dry_θ′ = dry_θ - dry_θ_bg

dry_simulation.output_writers[:jld2] = JLD2Writer(dry_model,
    merge(dry_model.velocities, (; θ′ = dry_θ′));
    filename = "baroclinic_wave_dry",
    schedule = TimeInterval(1hours),
    overwrite_existing = true)

run!(dry_simulation)

# # Run 2: Moist baroclinic wave
#
# ## Moist model with microphysics + surface fluxes
#
# We add four ingredients to the dry setup:
#
# 1. an initial specific humidity field (DCMIP2016 §1.1),
# 2. one-moment **mixed-phase** bulk microphysics with non-equilibrium
#    cloud formation (cloud liquid + cloud ice + rain + snow),
# 3. bulk surface fluxes of momentum, sensible heat, and water vapor over
#    a prescribed sea-surface temperature equal to the analytic surface
#    virtual temperature, and
# 4. bounds-preserving WENO for the moisture mass fractions to forbid
#    spurious overshoots from a sharp tropopause moisture jump.
#
# We use a `ConstantRateCondensateFormation` with `τ_relax = 200` s for
# both liquid and ice phases. The CloudMicrophysics default τ_relax (≈ 40
# s) is shorter than our outer Δt, which makes the explicit per-stage
# microphysics tendency overshoot the saturation value and the moist BW
# blow up within a few outer steps. 200 s is comfortably non-stiff at
# Δt ≤ 30 s. (Future work: subcycle microphysics inside one outer Δt.)

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

τ_relax = 200.0
relaxation = ConstantRateCondensateFormation(1 / τ_relax)
cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

# ### Surface fluxes
#
# Bulk drag, sensible heat flux, and water-vapor flux at the bottom of the
# atmosphere over a prescribed sea-surface temperature equal to the
# analytic surface virtual temperature. ``C_D = 10^{-3}`` is a typical
# bulk exchange coefficient and ``U_g = 10^{-2}`` m/s is a small gustiness
# floor.

Cᴰ = 1e-3
Uᵍ = 1e-2
T_surface(λ, φ) = virtual_temperature(λ, φ, 0.0)

ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρθ_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))

boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵛ=ρqᵛ_bcs)

# WENO with `bounds=(0, 1)` forbids the spurious negative or super-unity
# overshoots that vanilla WENO can produce on a sharp tropopause moisture
# jump, which would otherwise feed an unphysical microphysics tendency.

weno = WENO()
bounds_preserving_weno = WENO(order=5, bounds=(0, 1))
momentum_advection = weno
scalar_advection = (ρθ  = weno,
                    ρqᵛ  = bounds_preserving_weno,
                    ρqᶜˡ = bounds_preserving_weno,
                    ρqᶜⁱ = bounds_preserving_weno,
                    ρqʳ  = bounds_preserving_weno,
                    ρqˢ  = bounds_preserving_weno)

moist_model = AtmosphereModel(grid; dynamics, coriolis, microphysics, boundary_conditions,
                              thermodynamic_constants = constants,
                              momentum_advection,
                              scalar_advection,
                              timestepper = :AcousticRungeKutta3)

set!(moist_model, θ=moist_potential_temperature, u=zonal_velocity, ρ=density, qᵛ=specific_humidity)

# ### Time-stepping
#
# In the moist case the binding constraint is the **microphysics-dynamics
# coupling stiffness** rather than the advective CFL. With ``τ_{\rm relax} =
# 200`` s and explicit (per-stage) microphysics tendencies, the moist BW
# is rock-stable at ``Δt = 20`` s and starts to blow up around ``Δt = 30``
# s. We use ``Δt = 20`` s here — still ~10× larger than the explicit
# acoustic-CFL Δt of the original moist BW example.

moist_simulation = Simulation(moist_model; Δt = 20.0, stop_time = 15days)
Oceananigans.Diagnostics.erroring_NaNChecker!(moist_simulation)

function moist_progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("[moist] Iter %5d | t = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, w))
    return nothing
end
add_callback!(moist_simulation, moist_progress, IterationInterval(1000))

# Save θ′ and cloud liquid water for visualization:

moist_θ = PotentialTemperature(moist_model)
moist_θ_bg = CenterField(grid)
set!(moist_θ_bg, moist_potential_temperature)
moist_θ′ = moist_θ - moist_θ_bg

qᶜˡ = moist_model.microphysical_fields.qᶜˡ

moist_simulation.output_writers[:jld2] = JLD2Writer(moist_model,
    merge(moist_model.velocities, (; θ′ = moist_θ′, qᶜˡ));
    filename = "baroclinic_wave_moist",
    schedule = TimeInterval(1hours),
    overwrite_existing = true)

run!(moist_simulation)

# # Visualization
#
# We compare the dry and moist runs side-by-side. The dry run shows the
# canonical BCI θ′ pattern with its zonal-wave-five structure; the moist
# run shows the same pattern with cloud liquid water tracing the frontal
# zones where rising moist air condenses.

dry_θ′_ts = FieldTimeSeries("baroclinic_wave_dry.jld2", "θ′")
dry_u_ts  = FieldTimeSeries("baroclinic_wave_dry.jld2", "u")
moist_θ′_ts = FieldTimeSeries("baroclinic_wave_moist.jld2", "θ′")
moist_qᶜˡ_ts = FieldTimeSeries("baroclinic_wave_moist.jld2", "qᶜˡ")

k_mid = Nz ÷ 2
z_mid = znode(k_mid, grid, Center())

# ## Final snapshots on the sphere

fig = Figure(size = (1600, 800))
sphere_kw = (elevation = π/6, azimuth = -π/2, aspect = :data)

ax1 = Axis3(fig[1, 1]; title = "dry: θ′ at z = $(z_mid/1e3) km, t = $(prettytime(dry_θ′_ts.times[end]))", sphere_kw...)
hm1 = surface!(ax1, view(dry_θ′_ts[end], :, :, k_mid); colormap = :balance, shading = NoShading)
Colorbar(fig[1, 2], hm1; label = "θ′ (K)")

ax2 = Axis3(fig[1, 3]; title = "dry: u", sphere_kw...)
hm2 = surface!(ax2, view(dry_u_ts[end], :, :, k_mid); colormap = :speed, shading = NoShading)
Colorbar(fig[1, 4], hm2; label = "u (m/s)")

ax3 = Axis3(fig[2, 1]; title = "moist: θ′ at z = $(z_mid/1e3) km, t = $(prettytime(moist_θ′_ts.times[end]))", sphere_kw...)
hm3 = surface!(ax3, view(moist_θ′_ts[end], :, :, k_mid); colormap = :balance, shading = NoShading)
Colorbar(fig[2, 2], hm3; label = "θ′ (K)")

ax4 = Axis3(fig[2, 3]; title = "moist: qᶜˡ", sphere_kw...)
hm4 = surface!(ax4, view(moist_qᶜˡ_ts[end], :, :, k_mid); colormap = Reverse(:grays), shading = NoShading)
Colorbar(fig[2, 4], hm4; label = "qᶜˡ (kg/kg)")

for ax in (ax1, ax2, ax3, ax4)
    hidedecorations!(ax)
    hidespines!(ax)
end

current_figure()

# ## Animation
#
# Animate the moist run's θ′ alongside cloud liquid water, which reveals
# condensation along the frontal bands.

times = moist_θ′_ts.times
Nt = length(times)

n = Observable(1)
θ′n  = @lift view(moist_θ′_ts[$n],  :, :, k_mid)
qᶜˡn = @lift view(moist_qᶜˡ_ts[$n], :, :, k_mid)

fig = Figure(size = (1200, 600))
title = @lift "moist BW, z = $(z_mid/1e3) km, t = $(prettytime(times[$n]))"

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

CairoMakie.record(fig, "baroclinic_wave.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](baroclinic_wave.mp4)
