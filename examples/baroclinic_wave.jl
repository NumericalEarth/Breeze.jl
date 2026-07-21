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
# This example exercises `CompressibleDynamics` with
# [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization)
# (acoustic substepping via [`AcousticRungeKutta3`](@ref)) and
# `SphericalCoriolis` (non-hydrostatic) on a 1° latitude-longitude grid spanning
# 75° S to 75° N. Acoustic substepping lets the outer time step be set by
# the *advective* CFL rather than the much-tighter acoustic CFL — here a
# time-step wizard floats Δt at advective CFL ≈ 1.4 against the polar
# `Δx_min ≈ 28.8 km`, capped at 12 min.
#
# A future moist version (one-moment mixed-phase microphysics + bulk surface
# fluxes) will be added once the moist substepper supports the larger ``Δt``
# needed for a tractable runtime. The dry case here runs for 30 days and
# captures the full BCI life cycle: visible perturbations by day 4,
# explosive cyclogenesis around day 8, and saturation afterward.
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
# with jet-width parameter ``K = 3``. In the dry case, ``T = T_v``.
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
# We use a 1° latitude-longitude grid spanning 75° S to 75° N. Capping the
# domain at ±75° (rather than the poles) keeps the polar `Δx_min` manageable:
# `a · cos 75° · 2π/Nλ ≈ 28.8 km`. The domain extends from the surface to
# 30 km with 64 vertical levels, exponentially stretched toward the surface
# with `ExponentialDiscretization`: the interfaces are clustered near
# the ground (`bias = :left`) so the smallest cells sit at the surface
# (`Δz ≈ 150 m`) and coarsen to `≈ 1070 m` at the model top. The e-folding
# `scale = H/2` sets how quickly the spacing grows with height.

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

# ## Temperature profile parameters

Tᴱ = 310     # K — equatorial surface temperature
Tᴾ = 240     # K — polar surface temperature
Tᴹ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005    # K/m — lapse rate
K  = 3        # jet width parameter
b  = 2        # vertical half-width parameter

# ## Analytic initial conditions
#
# The DCMIP2016 balanced state is given in *virtual* temperature ``T_v``.
# We define the IC in terms of ``T_v`` so a future moist version (where
# ``T = T_v / (1 + \epsilon q^v)``) can reuse the same functions.

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

## Density (uses Tᵥ in the ideal gas law; in the dry case, T = Tᵥ).
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

## Potential temperature: θ = Tᵥ (p₀/p)^κ in the dry case.
potential_temperature(λ, φ, z) = virtual_temperature(λ, φ, z) * (p₀ / pressure(λ, φ, z))^κ

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
# We use fully compressible dynamics with **acoustic substepping** via
# [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization)
# and the [`AcousticRungeKutta3`](@ref)
# (Wicker–Skamarock RK3) outer loop. Acoustic substepping handles the
# fast acoustic-mode pressure gradient and buoyancy via a vertically-implicit
# inner loop, so the outer time step is limited only by the *advective*
# CFL — about 100× larger than the acoustic-CFL-limited Δt the fully
# explicit solver requires for the same grid.
#
# We use a hydrostatically-balanced isothermal reference state at
# `T₀ᵣ = 250 K` (matching the MPAS convention) so that the substepper's
# slow tendencies see only perturbations from the background.
# `SphericalCoriolis` is the non-hydrostatic spherical form, which retains
# both the traditional ``f = 2Ω \sin φ`` and the non-traditional
# ``2Ω \cos φ`` cross-terms that couple horizontal momentum to ``w``. Breeze
# evolves prognostic ``ρw`` so the non-traditional terms are physically
# required for self-consistent dynamics on the sphere.
#
# Tracer and momentum advection uses fifth-order `WENO` reconstruction. No
# explicit closure is applied: WENO's implicit dissipation suffices on this
# poleward-refined, near-surface-stretched grid.

coriolis = SphericalCoriolis(rotation_rate=Ω)

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵣ)

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO(order=5))

# ## Set initial conditions

set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density)

# ## Time-stepping
#
# Substepping eliminates the acoustic CFL constraint on the outer Δt; only
# the advective CFL remains. A time-step wizard targets advective CFL ≈ 1.4
# against the polar `Δx_min ≈ 28.8 km`, capped at Δt = 12 min:
#
# ```math
# Δt = \min\!\left(1.4 \cdot Δx_{\min} / U_{\max},\ 720 \text{ s}\right).
# ```
#
# This is many times larger than the acoustic-CFL-limited Δt a fully
# explicit solver would require. We run for 30 days to capture the full
# BCI life cycle.

Δt = 12minutes
stop_time = 30days
cfl = 1.4

simulation = Simulation(model; Δt, stop_time)
conjure_time_step_wizard!(simulation; cfl, max_Δt=12minutes)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# ## Progress callback

function progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("Iter %5d | t = %s | Δt = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(50))

# ## Output
#
# We save the velocities, the full potential temperature ``θ`` (the
# classic surface synoptic diagnostic for the cold/warm sectors during
# cyclogenesis), and the vertical vorticity ``ζ``, sliced at two levels:
# k = 1 (surface) and k = 16 (lower troposphere, ~2.9 km).

using Oceananigans.Operators: ζ₃ᶠᶠᶜ
u, v, w = model.velocities
ζ = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, model.grid, u, v)

θ = PotentialTemperature(model)

outputs = merge(model.velocities, (; ζ, θ))

for k in (1, 16)
    filename = "baroclinic_wave_k$k"
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
# We plot three near-surface diagnostics on the sphere: the **surface
# potential temperature** ``θ_{\rm sfc}`` (the classic diagnostic for the
# cold/warm sectors), the **surface vertical vorticity** ``ζ`` (which reveals
# the cyclones and anticyclones), and the **lower-tropospheric vertical
# velocity** ``w`` at ~2.9 km (the warm conveyor belt).

θ_ts = FieldTimeSeries("baroclinic_wave_k1.jld2",  "θ")
ζ_ts = FieldTimeSeries("baroclinic_wave_k1.jld2",  "ζ")
w_ts = FieldTimeSeries("baroclinic_wave_k16.jld2", "w")
times = θ_ts.times
Nt = length(times)

k_sfc = 1
k_mid = 16

# Sphere view: rotate so the developing wave faces the camera.
sphere_kw = (elevation = π/6, azimuth = π/2, aspect = :data)
ζlim = 1e-4
wlim = 0.06

θ_kw = (colormap = :thermal, colorrange = (260, 310))
ζ_kw = (colormap = :balance, colorrange = (-ζlim, ζlim))
w_kw = (colormap = :balance, colorrange = (-wlim, wlim))

# ### Animation

n = Observable(1)
θn = @lift view(θ_ts[$n], :, :, k_sfc)
ζn = @lift view(ζ_ts[$n], :, :, k_sfc)
wn = @lift view(w_ts[$n], :, :, k_mid)

fig = Figure(size = (1800, 700))

title = @lift "t = $(prettytime(times[$n]))"
fig[0, 1:6] = Label(fig, title, fontsize=22, tellwidth=false)

ax1 = Axis3(fig[1, 1]; title = "θ at surface", sphere_kw...)
hm1 = surface!(ax1, θn; shading = NoShading, θ_kw...)
Colorbar(fig[1, 2], hm1; label = "θ (K)", height=Relative(0.5))

ax2 = Axis3(fig[1, 3]; title = "ζ at surface", sphere_kw...)
hm2 = surface!(ax2, ζn; shading = NoShading, ζ_kw...)
Colorbar(fig[1, 4], hm2; label = "ζ (1/s)", height=Relative(0.5))

ax3 = Axis3(fig[1, 5]; title = "w at 2.9 km", sphere_kw...)
hm3 = surface!(ax3, wn; shading = NoShading, w_kw...)
Colorbar(fig[1, 6], hm3; label = "w (m/s)", height=Relative(0.5))

for ax in (ax1, ax2, ax3)
    hidedecorations!(ax)
    hidespines!(ax)
end

CairoMakie.record(fig, "baroclinic_wave.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](baroclinic_wave.mp4)
