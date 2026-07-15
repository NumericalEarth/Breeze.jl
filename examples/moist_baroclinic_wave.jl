# # Moist baroclinic wave on the sphere
#
# This example simulates the growth of a *moist* baroclinic wave on a near-global
# `LatitudeLongitudeGrid` following the DCMIP2016 specification
# [UllrichEtAl2016](@citet), which extends the classic dry
# [JablonowskiWilliamson2006](@citet) test case with a background moisture
# distribution and Kessler warm-rain microphysics. It is the moist companion to
# the dry `baroclinic_wave.jl` example, and it completes Breeze's coverage of the
# DCMIP2016 moist test suite alongside the supercell (`splitting_supercell.jl`)
# and tropical-cyclone cases.
#
# The dynamics are identical to the dry case: a midlatitude jet in thermal-wind
# balance is seeded with a localized zonal-wind perturbation that triggers
# baroclinic instability, producing growing Rossby waves over roughly ten days.
# What is new here is moisture. The balanced state is prescribed in **virtual**
# temperature ``T_v``, so the pressure and density fields are *unchanged* from the
# dry case — moisture enters only by (i) adding a background specific humidity
# ``q^v(\varphi, z)``, and (ii) recovering the actual temperature
# ``T = T_v / (1 + \epsilon\, q^v)`` from ``T_v``. As the wave develops, the warm
# conveyor belt lifts moist boundary-layer air, condensation releases latent heat,
# clouds form, and rain falls out — a self-consistent moist cyclogenesis.
#
# As in the dry case we use `CompressibleDynamics` with
# [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization)
# (acoustic substepping via [`AcousticRungeKutta3`](@ref)) and `SphericalCoriolis`
# (non-hydrostatic) on a 1° latitude-longitude grid spanning 75° S to 75° N.
# The same acoustic substepper transports the moisture and condensate tracers, so
# the outer time step is still set by the *advective* CFL rather than the acoustic
# one.
#
# !!! warning "Runtime"
#     The moist substepper is more expensive per step than the dry one — it carries
#     the moisture-dependent pressure-gradient coefficient and two extra prognostic
#     tracers — and moist convection can demand a smaller ``Δt`` than the dry
#     advective CFL alone. If instabilities appear, lower `cfl` / `max_Δt` in the
#     time-step wizard below. A full 30-day sphere run is a workstation-GPU-scale
#     computation, not a quick doctest.
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
#
# ### Background moisture
#
# DCMIP2016 prescribes an analytic specific-humidity field that decays away from
# the moist tropical lower troposphere:
#
# ```math
# q^v(φ, η) = q_0 \, \exp\!\left[-\left(\frac{φ}{φ_w}\right)^4\right]
#                    \exp\!\left[-\left(\frac{(η - 1)\, p_0}{p_w}\right)^2\right]
# ```
#
# where ``η = p / p_0`` is the pressure coordinate and ``p_w = 340`` hPa the
# vertical e-folding width, capped to a tiny stratospheric value ``q_t`` above
# ``η = 0.1``. With ``q^v`` known, the actual temperature follows from the virtual
# temperature via ``T = T_v / (1 + \epsilon\, q^v)`` with
# ``\epsilon = R^v/R^d - 1 \approx 0.608``.
#
# ### Perturbation
#
# A localized zonal-wind perturbation centered at
# ``(λ_c, φ_c) = (20°\text{E}, 40°\text{N})`` seeds the instability, exactly as in
# the dry case:
#
# ```math
# u'(λ, φ, z) = u_p \, \mathcal{T}(z) \, \exp\!\left(-\left(\frac{d}{r_p}\right)^2\right)
# ```
#
# where ``d`` is the great-circle distance, ``r_p = 0.1\,a``, ``u_p = 1`` m/s,
# and ``\mathcal{T}(z) = 1 - 3(z/z_p)^2 + 2(z/z_p)^3`` for ``z < z_p``.

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: φnode
using Printf
using CairoMakie
using CUDA

# ## DCMIP2016 parameters
#
# All parameters follow the DCMIP2016 test case document
# [UllrichEtAl2016](@citet). We set the Oceananigans defaults and build a
# custom [`ThermodynamicConstants`](@ref) matching the DCMIP specification
# so that the grid, Coriolis, and model thermodynamics are all consistent
# with the analytic initial conditions. The Kessler microphysics scheme diagnoses
# saturation from the [`TetensFormula`](@ref), so we select it here (as in the
# DCMIP2016 supercell example) rather than the default Clausius–Clapeyron form.

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    saturation_vapor_pressure = TetensFormula(),
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
Rᵛ  = vapor_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
ε   = Rᵛ / Rᵈ - 1   # ≈ 0.608 — the DCMIP2016 virtual-temperature coefficient Mᵥ
p₀  = 1e5    # Pa — surface pressure
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

# ## Domain and grid
#
# We use a 1° latitude-longitude grid spanning 75° S to 75° N. Capping the
# domain at ±75° (rather than the poles) keeps the polar `Δx_min` manageable:
# `a · cos 75° · 2π/Nλ ≈ 28.8 km`. The domain extends from the surface to
# 30 km with 64 vertical levels, exponentially stretched toward the surface
# with `ExponentialDiscretization`: the interfaces are clustered near
# the ground (`bias = :left`) so the smallest cells sit at the surface
# (`Δz ≈ 150 m`) and coarsen to `≈ 1070 m` at the model top. The e-folding
# `scale = H/2` sets how quickly the spacing grows with height. The near-surface
# refinement matters more here than in the dry case: it resolves the moist
# boundary layer that feeds the warm conveyor belt.

Nλ = 360
Nφ = 150
Nz = 64
H  = 30kilometers

z_faces = ExponentialDiscretization(Nz, 0, H; scale = H/2, bias = :left)

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-75, 75),
                             z = z_faces)

# ## Temperature and moisture profile parameters

Tᴱ = 310     # K — equatorial surface temperature
Tᴾ = 240     # K — polar surface temperature
Tᴹ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005    # K/m — lapse rate
K  = 3        # jet width parameter
b  = 2        # vertical half-width parameter

q₀   = 0.018    # kg/kg — maximum (equatorial, surface) specific humidity
φʷ   = 2π / 9   # rad — moisture meridional half-width (≈ 40°)
pʷ   = 34000    # Pa — moisture vertical scale (pressure e-folding width)
qᵗᵒᵖ = 1e-12    # kg/kg — background (stratospheric) specific humidity

# ## Analytic initial conditions
#
# The DCMIP2016 balanced state is given in *virtual* temperature ``T_v``, so the
# pressure and density profiles below are identical to the dry example. Moisture
# is layered on top through `vapor_mass_fraction` and the ``T_v \to T`` conversion
# inside `potential_temperature`.

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

## Density uses Tᵥ in the ideal gas law for moist air: p = ρ Rᵈ Tᵥ, so this is
## unchanged from the dry case.
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

## DCMIP2016 §1.1 background specific humidity qᵛ(φ, η) in the pressure
## coordinate η = p / p₀, tapered to the stratospheric value qᵗᵒᵖ above η = 0.1.
function vapor_mass_fraction(λ, φ, z)
    φʳ = deg2rad(φ)
    η  = pressure(λ, φ, z) / p₀
    qᵛ = q₀ * exp(-(φʳ / φʷ)^4) * exp(-((η - 1) * p₀ / pʷ)^2)
    return ifelse(η > 0.1, qᵛ, qᵗᵒᵖ)
end

## Potential temperature θ = T (p₀/p)^κ, using the *actual* temperature
## T = Tᵥ / (1 + ε qᵛ). With no condensate at t = 0, θ = θˡⁱ.
function potential_temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z)
    qᵛ = vapor_mass_fraction(λ, φ, z)
    T  = Tᵥ / (1 + ε * qᵛ)
    return T * (p₀ / pressure(λ, φ, z))^κ
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

# ## Model configuration
#
# The dynamical core is identical to the dry case — fully compressible dynamics
# with **acoustic substepping** and a hydrostatically-balanced isothermal
# reference state at `T₀ᵣ = 250 K`. We keep the reference state *dry*: it exists
# only to remove the balanced hydrostatic pressure gradient from the slow
# tendencies, and the prognostic moisture then rides on top of it. `SphericalCoriolis`
# retains both the traditional ``f = 2Ω \sin φ`` and non-traditional
# ``2Ω \cos φ`` cross-terms, physically required because Breeze evolves prognostic ``ρw``.

coriolis = SphericalCoriolis(rotation_rate=Ω)

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵣ)

# For microphysics we use the DCMIP2016 Kessler warm-rain scheme — the same scheme
# used across all three DCMIP2016 moist test cases. It carries prognostic cloud
# water ``q^{cl}`` and rain water ``q^r`` alongside vapor ``q^v``, with
# autoconversion, accretion, rain evaporation, and sedimentation.

microphysics = DCMIP2016KesslerMicrophysics()

# Tracer and momentum advection use fifth-order `WENO` reconstruction. We apply
# bounds-preserving WENO to the moisture and condensate densities so they stay
# non-negative through the sharp gradients of the developing fronts.

momentum_advection = WENO(order=5)
scalar_advection = (ρθ  = WENO(order=5),
                    ρqᵛ  = WENO(order=5, bounds=(0, 1)),
                    ρqᶜˡ = WENO(order=5, bounds=(0, 1)),
                    ρqʳ  = WENO(order=5, bounds=(0, 1)))

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        microphysics,
                        momentum_advection, scalar_advection)

# ## Set initial conditions
#
# `set!` accepts the pointwise IC functions. For compressible dynamics the density
# must be provided (moisture partial densities are weighted by the total ``ρ``);
# `set!` orders the operations internally so the specific humidity `qᵛ` is
# converted to `ρqᵛ` with the density we pass. Condensate starts at zero.

set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density, qᵛ=vapor_mass_fraction)

# ## Time-stepping
#
# Substepping eliminates the acoustic CFL constraint on the outer Δt; the advective
# CFL remains. A time-step wizard targets advective CFL ≈ 0.7 against the polar
# `Δx_min ≈ 28.8 km`, capped at Δt = 12 min (the same target as the dry case).
#
# We start from a gentle `Δt = 1 minute` and let the wizard ramp it (`max_change =
# 1.08` per adjustment) up to the 12 min cap. The moist balanced state launches a
# brief adjustment transient in the first steps; at full resolution the sharp
# near-surface layer amplifies it, so taking the first ~hour of simulation at a
# small step lets it damp out before the step grows. Jumping straight to 12 min on
# step 1 instead re-triggers the transient into a vertical-velocity shock.

Δt = 1minute
stop_time = 30days

simulation = Simulation(model; Δt, stop_time)
conjure_time_step_wizard!(simulation; cfl=0.7, max_Δt=12minutes, max_change=1.08)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# ## Progress callback

function progress(sim)
    u, v, w = sim.model.velocities
    qᵛ  = sim.model.microphysical_fields.qᵛ
    qᶜˡ = sim.model.microphysical_fields.qᶜˡ
    qʳ  = sim.model.microphysical_fields.qʳ
    @info @sprintf("Iter %5d | t = %s | Δt = %s | max|u| = %.1f m/s | max|w| = %.4f m/s | max qᵛ = %.4f | max qᶜˡ = %.2e | max qʳ = %.2e",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, w),
                   maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(50))

# ## Output
#
# We save the velocities, potential temperature ``θ``, vertical vorticity ``ζ``,
# and the three moisture species — water vapor ``q^v``, cloud water ``q^{cl}``,
# and rain water ``q^r`` — sliced at three levels: k = 1 (surface),
# k = 16 (lower troposphere, ~2.9 km, where clouds and rain are most vigorous),
# and k = 38 (the 250 hPa jet level, ~10.5 km).

using Oceananigans.Operators: ζ₃ᶠᶠᶜ
u, v, w = model.velocities
ζ = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, model.grid, u, v)

θ   = PotentialTemperature(model)
qᵛ  = model.microphysical_fields.qᵛ
qᶜˡ = model.microphysical_fields.qᶜˡ
qʳ  = model.microphysical_fields.qʳ

outputs = merge(model.velocities, (; ζ, θ, qᵛ, qᶜˡ, qʳ))

for k in (1, 16, 38)
    filename = "moist_baroclinic_wave_k$k"
    ow = JLD2Writer(model, outputs; filename,
                    indices = (:, :, k),
                    schedule = TimeInterval(6hours),
                    overwrite_existing = true)

    simulation.output_writers[Symbol(filename)] = ow
end

# ## Run

run!(simulation)

# ## Visualization
#
# We tell the moist cyclogenesis story with six near-surface / lower-tropospheric
# diagnostics on the sphere. Top row: the classic dry synoptics — surface
# **potential temperature** ``θ`` (cold/warm sectors), surface **vertical
# vorticity** ``ζ`` (cyclones/anticyclones), and surface **water vapor** ``q^v``
# (the moisture reservoir wrapping into the comma head). Bottom row: the moist
# response — **cloud water** ``q^{cl}`` and **rain water** ``q^r`` at ~2.9 km
# (marking the warm conveyor belt and cold-frontal rain band), and the
# lower-tropospheric **vertical velocity** ``w`` that drives the condensation.

θ_ts   = FieldTimeSeries("moist_baroclinic_wave_k1.jld2",  "θ")
ζ_ts   = FieldTimeSeries("moist_baroclinic_wave_k1.jld2",  "ζ")
qᵛ_ts  = FieldTimeSeries("moist_baroclinic_wave_k1.jld2",  "qᵛ")
qᶜˡ_ts = FieldTimeSeries("moist_baroclinic_wave_k16.jld2", "qᶜˡ")
qʳ_ts  = FieldTimeSeries("moist_baroclinic_wave_k16.jld2", "qʳ")
w_ts   = FieldTimeSeries("moist_baroclinic_wave_k16.jld2", "w")
times  = θ_ts.times
Nt = length(times)

k_sfc = 1
k_mid = 16

# Sphere view: rotate so the developing wave faces the camera.
sphere_kw = (elevation = π/6, azimuth = π/2, aspect = :data)
ζlim   = 1e-4
qᵛlim  = 0.018     # kg/kg — matches the equatorial-surface maximum
qᶜˡlim = 1e-3      # kg/kg — cloud water is small
qʳlim  = 5e-4      # kg/kg — rain water is smaller still
wlim   = 0.06

θ_kw   = (colormap = :thermal, colorrange = (260, 310))
ζ_kw   = (colormap = :balance, colorrange = (-ζlim, ζlim))
qᵛ_kw  = (colormap = :dense,   colorrange = (0, qᵛlim))
qᶜˡ_kw = (colormap = :Greens,  colorrange = (0, qᶜˡlim))
qʳ_kw  = (colormap = :amp,     colorrange = (0, qʳlim))
w_kw   = (colormap = :balance, colorrange = (-wlim, wlim))

# ### Animation

n = Observable(1)
θn   = @lift view(θ_ts[$n],   :, :, k_sfc)
ζn   = @lift view(ζ_ts[$n],   :, :, k_sfc)
qᵛn  = @lift view(qᵛ_ts[$n],  :, :, k_sfc)
qᶜˡn = @lift view(qᶜˡ_ts[$n], :, :, k_mid)
qʳn  = @lift view(qʳ_ts[$n],  :, :, k_mid)
wn   = @lift view(w_ts[$n],   :, :, k_mid)

fig = Figure(size = (1800, 1300))

title = @lift "t = $(prettytime(times[$n]))"
fig[0, 1:6] = Label(fig, title, fontsize=22, tellwidth=false)

## Top row: near-surface synoptics + moisture reservoir
ax1 = Axis3(fig[1, 1]; title = "θ at surface", sphere_kw...)
hm1 = surface!(ax1, θn; shading = NoShading, θ_kw...)
Colorbar(fig[1, 2], hm1; label = "θ (K)", height=Relative(0.5))

ax2 = Axis3(fig[1, 3]; title = "ζ at surface", sphere_kw...)
hm2 = surface!(ax2, ζn; shading = NoShading, ζ_kw...)
Colorbar(fig[1, 4], hm2; label = "ζ (1/s)", height=Relative(0.5))

ax3 = Axis3(fig[1, 5]; title = "qᵛ at surface", sphere_kw...)
hm3 = surface!(ax3, qᵛn; shading = NoShading, qᵛ_kw...)
Colorbar(fig[1, 6], hm3; label = "qᵛ (kg/kg)", height=Relative(0.5))

## Bottom row: moist response in the lower troposphere
ax4 = Axis3(fig[2, 1]; title = "qᶜˡ at 2.9 km", sphere_kw...)
hm4 = surface!(ax4, qᶜˡn; shading = NoShading, qᶜˡ_kw...)
Colorbar(fig[2, 2], hm4; label = "qᶜˡ (kg/kg)", height=Relative(0.5))

ax5 = Axis3(fig[2, 3]; title = "qʳ at 2.9 km", sphere_kw...)
hm5 = surface!(ax5, qʳn; shading = NoShading, qʳ_kw...)
Colorbar(fig[2, 4], hm5; label = "qʳ (kg/kg)", height=Relative(0.5))

ax6 = Axis3(fig[2, 5]; title = "w at 2.9 km", sphere_kw...)
hm6 = surface!(ax6, wn; shading = NoShading, w_kw...)
Colorbar(fig[2, 6], hm6; label = "w (m/s)", height=Relative(0.5))

for ax in (ax1, ax2, ax3, ax4, ax5, ax6)
    hidedecorations!(ax)
    hidespines!(ax)
end

CairoMakie.record(fig, "moist_baroclinic_wave.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](moist_baroclinic_wave.mp4)
