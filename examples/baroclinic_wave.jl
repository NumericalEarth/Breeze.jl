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
# 2. **Moist** — DCMIP2016 §1.1 with mixed-phase saturation-adjustment
#    microphysics (diagnostic cloud liquid ``q^\ell`` and cloud ice
#    ``q^{i}``, no precipitation). No surface fluxes — the wave evolves
#    from the initial moisture only. Latent heat release sharpens the
#    moist fronts and intensifies cyclogenesis compared to the dry case.
#
# Both configurations exercise `CompressibleDynamics` with
# [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization)
# (acoustic substepping via [`AcousticRungeKutta3`](@ref)) and
# `SphericalCoriolis` (non-hydrostatic) on a coarse latitude-longitude grid
# (Δλ ≈ 1.4°, Δφ ≈ 1.2°) spanning 75° S to 75° N. Acoustic substepping lets
# the outer time step be set by the *advective* CFL rather than the much-
# tighter acoustic CFL — here a time-step wizard floats Δt at advective
# CFL ≈ 1.4 against the polar `Δx_min ≈ 40.5 km`, capped at 12 min.
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
# In the dry case ``T = T_v``; in the moist case the actual temperature is
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
# center and is tapered smoothly to zero above 15 km:
#
# ```math
# u'(λ, φ, z) = u_p \, \mathcal{T}(z) \, \exp\!\left(-\left(\frac{d}{r_p}\right)^2\right)
# ```
#
# where ``d`` is the great-circle distance, ``r_p = 0.1\,a``, ``u_p = 1`` m/s,
# and ``\mathcal{T}(z) = 1 - 3(z/z_p)^2 + 2(z/z_p)^3`` for ``z < z_p``.
#
# ### Initial moisture (moist run only)
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
Oceananigans.defaults.planet_radius = 6371220
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5    # Pa — surface pressure
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

# ## Domain and grid
#
# A coarse 256×128 lat-lon grid (Δλ ≈ 1.4°, Δφ ≈ 1.2°) spanning 75° S to
# 75° N. Capping the domain at ±75° keeps the polar `Δx_min` manageable:
# `a · cos 75° · 2π/Nλ ≈ 40.5 km`. The domain extends from the surface to
# 30 km with 32 vertical levels (Δz ≈ 940 m). The same grid is shared by
# the dry and moist runs.

Nλ = 256
Nφ = 128
Nz = 32
H  = 30kilometers

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-75, 75),
                             z = (0, H))

## Temperature profile parameters
Tᴱ  = 310     # K — equatorial surface temperature
Tᴾ  = 240     # K — polar surface temperature
Tᴹ  = (Tᴱ + Tᴾ) / 2
Γ   = 0.005   # K/m — lapse rate
K   = 3       # jet width parameter
b   = 2       # vertical half-width parameter
εᵥ  = 0.608   # virtual-temperature factor R_v/R_d − 1

# ## Analytic initial conditions
#
# The DCMIP2016 balanced state is given in *virtual* temperature ``T_v``.
# We define a single set of pointwise functions usable for both dry and
# moist cases. In the dry case ``T = T_v``; in the moist case the actual
# temperature ``T = T_v / (1 + \epsilon q^v)`` is colder than ``T_v``.
# Density always uses ``T_v`` in the ideal gas law so the analytic
# pressure-gradient force is preserved.

## Vertical structure functions (shallow atmosphere, X = 1)
function τ_and_integrals(z)
    Hˢ = Rᵈ * Tᴹ / g
    η  = z / (b * Hˢ)
    e  = exp(-η^2)

    A = (Tᴹ - Tᴾ) / (Tᴹ * Tᴾ)
    C = (K + 2) * (Tᴱ - Tᴾ) / (2 * Tᴱ * Tᴾ)

    τ₁  = A * (1 - 2η^2) * e + exp(Γ * z / Tᴹ) / Tᴹ
    ∫τ₁ = A * z * e + (exp(Γ * z / Tᴹ) - 1) / Γ

    τ₂  = C * (1 - 2η^2) * e
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

## Density (uses Tᵥ in the ideal gas law; in the dry case T = Tᵥ).
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

## DCMIP2016 §1.1 specific humidity profile (used by the moist run).
function initial_specific_humidity(λ, φ, z)
    q₀ = 0.018    # kg/kg — surface maximum
    qₜ = 1e-12    # kg/kg — stratospheric trace value
    φʷ = 2π / 9   # rad — meridional e-folding width (≈ 40°)
    pʷ = 34000    # Pa — vertical e-folding pressure width

    p  = pressure(λ, φ, z)
    η  = p / p₀
    φʳ = deg2rad(φ)

    q_troposphere = q₀ * exp(-(φʳ / φʷ)^4) * exp(-((η - 1) * p₀ / pʷ)^2)

    return ifelse(η > 0.1, q_troposphere, qₜ)
end

## Dry potential temperature: θ = Tᵥ (p₀/p)^κ.
dry_potential_temperature(λ, φ, z) =
    virtual_temperature(λ, φ, z) * (p₀ / pressure(λ, φ, z))^κ

## Moist potential temperature: θ = T (p₀/p)^κ with T = Tᵥ / (1 + εᵥ qᵛ).
function moist_potential_temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z)
    p  = pressure(λ, φ, z)
    qᵛ = initial_specific_humidity(λ, φ, z)
    T  = Tᵥ / (1 + εᵥ * qᵛ)
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
    uₚ = 1         # m/s — amplitude
    rₚ = 0.1       # perturbation radius (Earth radii)
    λₚ = π / 9     # 20°E center longitude
    φₚ = 2π / 9    # 40°N center latitude
    zₚ = 15000     # m — height cap

    φʳ = deg2rad(φ)
    λʳ = deg2rad(λ)
    great_circle = acos(sin(φₚ) * sin(φʳ) + cos(φₚ) * cos(φʳ) * cos(λʳ - λₚ)) / rₚ

    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, zero(z))
    u_perturbation = ifelse(great_circle < 1, uₚ * taper * exp(-great_circle^2), zero(z))

    return u_balanced + u_perturbation
end

# ## Shared model configuration
#
# Both runs use fully compressible dynamics with **acoustic substepping**
# via [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization)
# and the [`AcousticRungeKutta3`](@ref) (Wicker–Skamarock RK3) outer loop.
# The substepper handles the fast acoustic-mode pressure gradient and
# buoyancy via a vertically-implicit inner loop, so the outer time step is
# limited only by the *advective* CFL — about 100× larger than the
# acoustic-CFL-limited Δt the fully explicit solver requires for the same
# grid.
#
# We use a hydrostatically-balanced isothermal reference state at
# `T₀ᵣ = 250 K` (matching the MPAS convention) so the substepper's slow
# tendencies see only perturbations from the background.
# `SphericalCoriolis` is the non-hydrostatic spherical form, which retains
# both the traditional ``f = 2Ω \sin φ`` and the non-traditional
# ``2Ω \cos φ`` cross-terms that couple horizontal momentum to ``w``. Breeze
# evolves prognostic ``ρw`` so the non-traditional terms are physically
# required for self-consistent dynamics on the sphere.

coriolis = SphericalCoriolis(rotation_rate=Ω)

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵣ)

# # Run 1: Dry baroclinic wave
#
# No microphysics and no surface fluxes — the canonical DCMIP2016 §4.1 case.
# A time-step wizard targets advective CFL ≈ 1.4 against the polar
# `Δx_min ≈ 40.5 km`, capped at Δt = 12 min. We run for 30 days to capture
# the full BCI life cycle: visible perturbations by day 4, explosive
# cyclogenesis around day 8, and saturation afterward.

dry_model = AtmosphereModel(grid; dynamics, coriolis,
                            thermodynamic_constants = constants,
                            advection = WENO())

set!(dry_model, θ=dry_potential_temperature, u=zonal_velocity, ρ=density)

dry_simulation = Simulation(dry_model; Δt=12minutes, stop_time=30days)
conjure_time_step_wizard!(dry_simulation; cfl=1.4, max_Δt=12minutes)
Oceananigans.Diagnostics.erroring_NaNChecker!(dry_simulation)

function dry_progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("[dry] Iter %5d | t = %s | Δt = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, w))
    return nothing
end

add_callback!(dry_simulation, dry_progress, IterationInterval(200))

# Output the velocities, full potential temperature ``θ`` (the classic
# surface synoptic diagnostic for cold/warm sectors), and the vertical
# vorticity ``ζ``, sliced at two levels: ``k = 1`` (surface) and ``k = 16``
# (mid-troposphere, ~5 km).

using Oceananigans.Operators: ζ₃ᶠᶠᶜ
dry_u, dry_v, dry_w = dry_model.velocities
dry_ζ = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, dry_model.grid, dry_u, dry_v)
dry_θ = PotentialTemperature(dry_model)
dry_outputs = merge(dry_model.velocities, (; ζ = dry_ζ, θ = dry_θ))

for k in (1, 16)
    filename = "baroclinic_wave_dry_k$k"
    dry_simulation.output_writers[Symbol(filename)] = JLD2Writer(dry_model, dry_outputs;
                                                                 filename,
                                                                 indices = (:, :, k),
                                                                 schedule = TimeInterval(6hours),
                                                                 overwrite_existing = true)
end

run!(dry_simulation)

# # Run 2: Moist baroclinic wave
#
# We add two ingredients to the dry setup:
#
# 1. an initial vapor-specific-humidity field (DCMIP2016 §1.1), and
# 2. **mixed-phase saturation-adjustment** microphysics
#    ([`SaturationAdjustment`](@ref) with [`MixedPhaseEquilibrium`](@ref)),
#    which instantaneously partitions any super-saturated vapor into
#    diagnostic cloud liquid ``q^\ell`` and cloud ice ``q^{i}`` and
#    releases the corresponding latent heat into ``θ``.
#
# Saturation adjustment has no relaxation timescale — equilibrium is
# instantaneous — so unlike a one-moment scheme there is no microphysics
# stiffness limit on Δt. The moist run uses the same advective-CFL wizard
# and 12-min Δt cap as the dry run.

microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())

# WENO with `bounds=(0, 1)` forbids the spurious negative or super-unity
# overshoots that vanilla WENO can produce on a sharp tropopause moisture
# jump, which would otherwise drive an unphysical saturation adjustment at
# a single grid point. Momentum and ``ρθ`` use the standard WENO since
# they have no ``[0, 1]`` bound.

momentum_advection = WENO()
scalar_advection = (ρθ = WENO(), ρqᵉ = WENO(order=5, bounds=(0, 1)))

moist_model = AtmosphereModel(grid; dynamics, coriolis, microphysics,
                              thermodynamic_constants = constants,
                              momentum_advection,
                              scalar_advection)

set!(moist_model, θ=moist_potential_temperature, u=zonal_velocity, ρ=density,
                  qᵛ=initial_specific_humidity)

moist_simulation = Simulation(moist_model; Δt=12minutes, stop_time=30days)
conjure_time_step_wizard!(moist_simulation; cfl=1.4, max_Δt=12minutes)
Oceananigans.Diagnostics.erroring_NaNChecker!(moist_simulation)

function moist_progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("[moist] Iter %5d | t = %s | Δt = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, w))
    return nothing
end

add_callback!(moist_simulation, moist_progress, IterationInterval(200))

# Output the velocities, ``θ``, ``ζ``, and the diagnostic cloud liquid
# water ``q^\ell`` (which traces ascending saturated air along the moist
# fronts), sliced at the surface and mid-troposphere.

moist_u, moist_v, moist_w = moist_model.velocities
moist_ζ = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, moist_model.grid, moist_u, moist_v)
moist_θ = PotentialTemperature(moist_model)
qˡ = moist_model.microphysical_fields.qˡ
moist_outputs = merge(moist_model.velocities, (; ζ = moist_ζ, θ = moist_θ, qˡ))

for k in (1, 16)
    filename = "baroclinic_wave_moist_k$k"
    moist_simulation.output_writers[Symbol(filename)] = JLD2Writer(moist_model, moist_outputs;
                                                                   filename,
                                                                   indices = (:, :, k),
                                                                   schedule = TimeInterval(6hours),
                                                                   overwrite_existing = true)
end

run!(moist_simulation)

# # Visualization
#
# We make two animations on the sphere — one per run. The **dry** view
# plots the surface potential temperature ``θ_{\rm sfc}`` (the classic
# synoptic diagnostic for the cold/warm sectors), the surface vertical
# vorticity ``ζ`` (which reveals the cyclones and anticyclones), and the
# mid-level vertical velocity ``w`` (which highlights the warm conveyor
# belt and the wave's vertical structure). The **moist** view replaces
# ``w`` with the diagnostic cloud liquid water ``q^\ell``, which traces
# condensation along the frontal bands.

k_sfc = 1
k_mid = 16

sphere_kw = (elevation = π/6, azimuth = π/2, aspect = :data)
ζlim = 1e-4
wlim = 0.06
qˡlim = 1e-4

θ_kw  = (colormap = :thermal, colorrange = (260, 310))
ζ_kw  = (colormap = :balance, colorrange = (-ζlim, ζlim))
w_kw  = (colormap = :balance, colorrange = (-wlim, wlim))
qˡ_kw = (colormap = Reverse(:grays), colorrange = (0, qˡlim))

# ### Dry animation

dry_θ_ts = FieldTimeSeries("baroclinic_wave_dry_k1.jld2",  "θ")
dry_ζ_ts = FieldTimeSeries("baroclinic_wave_dry_k1.jld2",  "ζ")
dry_w_ts = FieldTimeSeries("baroclinic_wave_dry_k16.jld2", "w")
dry_times = dry_θ_ts.times
Nt_dry = length(dry_times)

n = Observable(1)
θn = @lift view(dry_θ_ts[$n], :, :, k_sfc)
ζn = @lift view(dry_ζ_ts[$n], :, :, k_sfc)
wn = @lift view(dry_w_ts[$n], :, :, k_mid)

fig = Figure(size = (1800, 700))

title = @lift "dry, t = $(prettytime(dry_times[$n]))"
fig[0, 1:6] = Label(fig, title, fontsize=22, tellwidth=false)

ax1 = Axis3(fig[1, 1]; title = "θ at surface", sphere_kw...)
hm1 = surface!(ax1, θn; shading = NoShading, θ_kw...)
Colorbar(fig[1, 2], hm1; label = "θ (K)", height=Relative(0.5))

ax2 = Axis3(fig[1, 3]; title = "ζ at surface", sphere_kw...)
hm2 = surface!(ax2, ζn; shading = NoShading, ζ_kw...)
Colorbar(fig[1, 4], hm2; label = "ζ (1/s)", height=Relative(0.5))

ax3 = Axis3(fig[1, 5]; title = "w at mid-level", sphere_kw...)
hm3 = surface!(ax3, wn; shading = NoShading, w_kw...)
Colorbar(fig[1, 6], hm3; label = "w (m/s)", height=Relative(0.5))

for ax in (ax1, ax2, ax3)
    hidedecorations!(ax)
    hidespines!(ax)
end

CairoMakie.record(fig, "baroclinic_wave_dry.mp4", 1:Nt_dry; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](baroclinic_wave_dry.mp4)

# ### Moist animation

moist_θ_ts  = FieldTimeSeries("baroclinic_wave_moist_k1.jld2",  "θ")
moist_ζ_ts  = FieldTimeSeries("baroclinic_wave_moist_k1.jld2",  "ζ")
moist_qˡ_ts = FieldTimeSeries("baroclinic_wave_moist_k16.jld2", "qˡ")
moist_times = moist_θ_ts.times
Nt_moist = length(moist_times)

n = Observable(1)
θn  = @lift view(moist_θ_ts[$n],  :, :, k_sfc)
ζn  = @lift view(moist_ζ_ts[$n],  :, :, k_sfc)
qˡn = @lift view(moist_qˡ_ts[$n], :, :, k_mid)

fig = Figure(size = (1800, 700))

title = @lift "moist, t = $(prettytime(moist_times[$n]))"
fig[0, 1:6] = Label(fig, title, fontsize=22, tellwidth=false)

ax1 = Axis3(fig[1, 1]; title = "θ at surface", sphere_kw...)
hm1 = surface!(ax1, θn; shading = NoShading, θ_kw...)
Colorbar(fig[1, 2], hm1; label = "θ (K)", height=Relative(0.5))

ax2 = Axis3(fig[1, 3]; title = "ζ at surface", sphere_kw...)
hm2 = surface!(ax2, ζn; shading = NoShading, ζ_kw...)
Colorbar(fig[1, 4], hm2; label = "ζ (1/s)", height=Relative(0.5))

ax3 = Axis3(fig[1, 5]; title = "qˡ at mid-level", sphere_kw...)
hm3 = surface!(ax3, qˡn; shading = NoShading, qˡ_kw...)
Colorbar(fig[1, 6], hm3; label = "qˡ (kg/kg)", height=Relative(0.5))

for ax in (ax1, ax2, ax3)
    hidedecorations!(ax)
    hidespines!(ax)
end

CairoMakie.record(fig, "baroclinic_wave_moist.mp4", 1:Nt_moist; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](baroclinic_wave_moist.mp4)
