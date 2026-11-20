#####
##### SAM's ocean surface flux routine (`oceflx.f90`), evaluated per column
#####
##### `sam_ocean_fluxes` is a line-by-line transcription of SAM's `oceflx.f90` (Large & Pond
##### neutral 10-m laws shifted to the lowest model level with two Monin–Obukhov iterations),
##### together with SAM's saturation humidities `qsatw`/`qsati` (`sat.f90`, Flatau et al. 1992).
##### `SAMOceanSurfaceFlux` feeds it per column with the inputs `surface.f90` passes when
##### `UNIFORM_SFC_FLX = .false.`: the scalar fluxes at cell centres from centred winds, each
##### stress component at its own velocity point from averaged scalars. The algorithm and the
##### mapping of SAM's inputs onto Breeze's state are written down in
##### `docs/cases/sam_oceflx_parity.md`.
#####

using Adapt: Adapt, adapt
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: BoundaryConditions as OceananigansBC, BoundaryCondition, Flux
using Oceananigans.Grids: AbstractGrid, Center, znode
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ
using Breeze.AtmosphereModels: grid_moisture_fractions
using Breeze.Thermodynamics: mixture_heat_capacity

#####
##### SAM's saturation specific humidities (sat.f90), pressure in mb
#####

# esatw: saturation vapor pressure over liquid [mb]
@inline function sam_liquid_saturation_vapor_pressure(T)
    FT = typeof(T)
    ΔT = max(-80, T - FT(273.16))
    return evalpoly(ΔT, (FT(6.11239921), FT(0.443987641), FT(0.142986287e-1),
                         FT(0.264847430e-3), FT(0.302950461e-5), FT(0.206739458e-7),
                         FT(0.640689451e-10), FT(-0.952447341e-13), FT(-0.976195544e-15)))
end

# esati: saturation vapor pressure over ice [mb], with esatw above 273.15 K and a
# quadratic below 185 K
@inline function sam_ice_saturation_vapor_pressure(T)
    FT = typeof(T)
    ΔT = T - FT(273.16)
    eⁱ = evalpoly(ΔT, (FT(6.11147274), FT(0.503160820), FT(0.188439774e-1),
                       FT(0.420895665e-3), FT(0.615021634e-5), FT(0.602588177e-7),
                       FT(0.385852041e-9), FT(0.146898966e-11), FT(0.252751365e-14)))
    ΔT_cold = max(-100, ΔT)
    eᶜ = FT(0.00763685) + ΔT_cold * (FT(0.000151069) + ΔT_cold * FT(7.48215e-07))
    eˡ = sam_liquid_saturation_vapor_pressure(T)
    return ifelse(T > FT(273.15), eˡ, ifelse(T > 185, eⁱ, eᶜ))
end

"""
    sam_surface_saturation_specific_humidity(Tₛ, p)

SAM's sea-surface saturation specific humidity `ssq` (`oceflx.f90`): `qsatw(Tₛ, p)` over
water when `Tₛ > 271` K and `qsati(Tₛ, p)` otherwise, with `p` in mb.
"""
@inline function sam_surface_saturation_specific_humidity(Tₛ, p)
    eˡ = sam_liquid_saturation_vapor_pressure(Tₛ)
    eⁱ = sam_ice_saturation_vapor_pressure(Tₛ)
    e = ifelse(Tₛ > 271, eˡ, eⁱ)
    FT = typeof(Tₛ)
    return FT(0.622) * e / max(e, p - e)
end

#####
##### oceflx.f90
#####

# Statement functions of oceflx.f90
@inline sam_neutral_drag(U₁₀) = typeof(U₁₀)(0.0027) / U₁₀ + typeof(U₁₀)(0.000142) + typeof(U₁₀)(0.0000764) * U₁₀
@inline sam_unstable_momentum_stability(x) =
    log((1 + x * (2 + x)) * (1 + x * x) / 8) - 2 * atan(x) + typeof(x)(1.571)
@inline sam_unstable_scalar_stability(x) = 2 * log((1 + x * x) / 2)

# Square root of the neutral 10-m Stanton number: 0.0327 (unstable), 0.0180 (stable)
@inline sam_neutral_stanton_root(stable) =
    (1 - stable) * typeof(stable)(0.0327) + stable * typeof(stable)(0.018)

"""
One Monin–Obukhov iteration of `oceflx.f90`: z/L from `(u★, θ★, q★)`, the Businger–Dyer
stability functions, the 10-m neutral wind, the updated neutral coefficients and their shift to
the lowest level. Returns the new `(rdn, rd, rh, re, stable)`.
"""
@inline function sam_ocean_flux_iteration(rdn, u★, θ★, q★, θᵥ, qᵇ, zᵇ, ℓ, U)
    FT = typeof(U)
    κ = FT(0.4)
    g = FT(9.81)

    # Stability, stable if θ★ > 0 or z/L > 0, with |z/L| ≤ 10
    ζ = κ * g * zᵇ * (θ★ / θᵥ + q★ / (1 / FT(0.606) + qᵇ)) / u★^2
    ζ = copysign(min(abs(ζ), 10), ζ)
    stable = (1 + copysign(one(FT), ζ)) / 2
    x² = max(sqrt(abs(1 - 16 * ζ)), 1)
    x = sqrt(x²)
    ψᵐ = -5 * ζ * stable + (1 - stable) * sam_unstable_momentum_stability(x)
    ψʰ = -5 * ζ * stable + (1 - stable) * sam_unstable_scalar_stability(x)

    # Shift the 10-m neutral wind with the previous neutral drag coefficient
    rd = rdn / (1 + rdn / κ * (ℓ - ψᵐ))
    U₁₀ = U * rd / rdn

    # Neutral 10-m coefficients at the new wind, then shifted to zᵇ and the stability
    rdn = sqrt(sam_neutral_drag(U₁₀))
    ren = FT(0.0346)
    rhn = sam_neutral_stanton_root(stable)
    rd = rdn / (1 + rdn / κ * (ℓ - ψᵐ))
    rh = rhn / (1 + rhn / κ * (ℓ - ψʰ))
    re = ren / (1 + ren / κ * (ℓ - ψʰ))

    return rdn, rd, rh, re, stable
end

"""
    sam_ocean_fluxes(p, u, v, Tᵇ, qᵇ, θᵇ, zᵇ, Tₛ)

SAM's `oceflx.f90` (Large & Pond, CCM3.5; adopted for LES by Khairoutdinov) for one column.
Inputs, in SAM's units: pressure `p` [mb] and wind `(u, v)` [m s⁻¹] at the lowest level,
its temperature `Tᵇ` [K] (SAM passes `t - gamaz`), specific humidity `qᵇ`, "potential"
temperature `θᵇ` [K] (SAM passes the liquid-ice static energy over cₚ, `t`), height `zᵇ` [m]
and the sea-surface temperature `Tₛ` [K].

Returns `(; sensible, vapor, τˣ, τʸ, surface_humidity)`: the kinematic heat flux
`-u★θ★` [K m s⁻¹] and vapor flux `-u★q★` [m s⁻¹] (positive upward), the stress components
`-ρᵇ u★² (u, v) / U` [N m⁻²] with `ρᵇ = 100 p / (287 Tᵇ)` and `U = max(1, |u|)`, and the surface
saturation humidity `ssq`. Generic in the floating-point type: in `Float32` it is SAM's
default single precision.
"""
@inline function sam_ocean_fluxes(p, u, v, Tᵇ, qᵇ, θᵇ, zᵇ, Tₛ)
    p, u, v, Tᵇ, qᵇ, θᵇ, zᵇ, Tₛ = promote(p, u, v, Tᵇ, qᵇ, θᵇ, zᵇ, Tₛ)
    FT = typeof(p)
    κ = FT(0.4)
    z_reference = FT(10)
    U_minimum = FT(1)

    ρᵇ = p * 100 / (287 * Tᵇ)
    U = max(U_minimum, sqrt(u^2 + v^2))
    θᵥ = θᵇ * (1 + FT(0.606) * qᵇ)
    Δθ = θᵇ - Tₛ
    qˢ = sam_surface_saturation_specific_humidity(Tₛ, p)
    Δq = qᵇ - qˢ
    ℓ = log(zᵇ / z_reference)

    # Initial guess: z/L = 0, U₁₀ ≈ U, stable if θᵇ ≥ Tₛ
    stable = (1 + copysign(one(FT), Δθ)) / 2
    rdn = sqrt(sam_neutral_drag(U))
    rhn = sam_neutral_stanton_root(stable)
    ren = FT(0.0346)
    u★ = rdn * U
    θ★ = rhn * Δθ
    q★ = ren * Δq

    # First iteration
    rdn, rd, rh, re, stable = sam_ocean_flux_iteration(rdn, u★, θ★, q★, θᵥ, qᵇ, zᵇ, ℓ, U)
    u★ = rd * U
    θ★ = rh * Δθ
    q★ = re * Δq

    # Second iteration
    rdn, rd, rh, re, stable = sam_ocean_flux_iteration(rdn, u★, θ★, q★, θᵥ, qᵇ, zᵇ, ℓ, U)
    u★ = rd * U
    θ★ = rh * Δθ
    q★ = re * Δq

    # Fluxes (the 2-m reference temperature `tref` of oceflx.f90 is diagnostic only)
    τ = ρᵇ * u★ * u★
    τˣ = -τ * u / U
    τʸ = -τ * v / U
    sensible = -τ * θ★ / u★ / ρᵇ
    vapor = -τ * q★ / u★ / ρᵇ

    return (; sensible, vapor, τˣ, τʸ, surface_humidity = qˢ)
end

#####
##### Per-column boundary condition
#####

struct SAMSensibleHeatFlux end
struct SAMVaporFlux end
struct SAMXStress end
struct SAMYStress end

"""
    SAMOceanSurfaceFlux(component, surface_temperature, microphysics, thermodynamic_constants, surface_density)

Bottom flux condition evaluating [`sam_ocean_fluxes`](@ref) per column, as SAM's `surface.f90`
does with `UNIFORM_SFC_FLX = .false.`. `component` selects the returned flux:

- `SAMSensibleHeatFlux()`, for the `ρE` key: the energy flux `cᵖᵐ ρʷ shf`. Breeze converts an
  energy flux to `ρθ` as `Jᶿ = 𝒬 / (cᵖᵐ Π)` with the same `cᵖᵐ`, so the lowest cell warms by
  `ρʷ shf`, SAM's `fluxbt × rhow(1)`;
- `SAMVaporFlux()`: the vapor mass flux `ρʷ lhf` (SAM's `fluxbq × rhow(1)`);
- `SAMXStress()`, `SAMYStress()`: the stress `τˣ`, `τʸ` of `oceflx` (SAM's `fluxbu × rhow(1)`).

`ρʷ` is `surface_density`, SAM's `rhow(1)`. The inputs per column are SAM's: the reference
pressure and height of the lowest cell, the ground-relative wind, `t = T + g z / cₚ -
(ℒˡ qˡ + ℒⁱ qⁱ) / cₚ` with SAM's constants as `θᵇ` and `t - g z / cₚ` as `Tᵇ`, the vapor mass
fraction, and the SST field `surface_temperature`. The scalar fluxes are evaluated at cell
centres with the wind averaged there; each stress component at its velocity point with the
scalars and SST averaged onto it and the other wind component averaged over four points.
"""
struct SAMOceanSurfaceFlux{Q, T, M, C, FT}
    component :: Q
    surface_temperature :: T
    microphysics :: M
    thermodynamic_constants :: C
    surface_density :: FT
end

Adapt.adapt_structure(to, flux::SAMOceanSurfaceFlux) =
    SAMOceanSurfaceFlux(flux.component, adapt(to, flux.surface_temperature), adapt(to, flux.microphysics),
                        adapt(to, flux.thermodynamic_constants), flux.surface_density)

Base.summary(flux::SAMOceanSurfaceFlux) =
    string("SAMOceanSurfaceFlux(", nameof(typeof(flux.component)), ", ρʷ=", flux.surface_density, ")")

# Moisture mass fractions of a cell, as Breeze's energy-flux conversion forms them
@inline function sam_moisture_fractions(i, j, k, grid, flux, fields, dynamics_fields)
    ρ = @inbounds dynamics_fields.ρ[i, j, k]
    qᵛ = @inbounds fields.qᵛ[i, j, k]
    return grid_moisture_fractions(i, j, k, grid, flux.microphysics, ρ, qᵛ, fields)
end

# SAM's prognostic t: liquid-ice static energy over cₚ, with SAM's params.f90 constants
@inline function sam_static_temperature(i, j, k, grid, flux, fields, dynamics_fields)
    q = sam_moisture_fractions(i, j, k, grid, flux, fields, dynamics_fields)
    T = @inbounds fields.T[i, j, k]
    FT = typeof(T)
    cᵖ = FT(1004)
    z = znode(i, j, k, grid, Center(), Center(), Center())
    return T + FT(9.81) / cᵖ * z - FT(2.5104e6) / cᵖ * q.liquid - FT(2.8440e6) / cᵖ * q.ice
end

@inline sam_vapor(i, j, k, grid, fields) = @inbounds fields.qᵛ[i, j, k]
@inline sam_sea_surface_temperature(i, j, k, grid, Tₛ) = @inbounds Tₛ[i, j, 1]

# oceflx with the SAM inputs averaged by `interpolate` (identity at centres)
@inline function sam_ocean_fluxes(i, j, grid, flux, fields, dynamics_fields, u, v, interpolate)
    FT = eltype(grid)
    p = @inbounds dynamics_fields.p[i, j, 1] / 100
    zᵇ = znode(i, j, 1, grid, Center(), Center(), Center())
    t = interpolate(i, j, 1, grid, sam_static_temperature, flux, fields, dynamics_fields)
    qᵇ = interpolate(i, j, 1, grid, sam_vapor, fields)
    Tₛ = interpolate(i, j, 1, grid, sam_sea_surface_temperature, flux.surface_temperature)
    Tᵇ = t - FT(9.81) / FT(1004) * zᵇ
    return sam_ocean_fluxes(p, u, v, Tᵇ, qᵇ, t, zᵇ, Tₛ)
end

@inline centred(i, j, k, grid, f, args...) = f(i, j, k, grid, args...)

@inline function sam_centre_fluxes(i, j, grid, flux, fields, dynamics_fields)
    u = ℑxᶜᵃᵃ(i, j, 1, grid, fields.u)
    v = ℑyᵃᶜᵃ(i, j, 1, grid, fields.v)
    return sam_ocean_fluxes(i, j, grid, flux, fields, dynamics_fields, u, v, centred)
end

@inline function OceananigansBC.getbc(flux::SAMOceanSurfaceFlux{SAMSensibleHeatFlux}, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields, dynamics_fields)
    fluxes = sam_centre_fluxes(i, j, grid, flux, fields, dynamics_fields)
    q = sam_moisture_fractions(i, j, 1, grid, flux, fields, dynamics_fields)
    cᵖᵐ = mixture_heat_capacity(q, flux.thermodynamic_constants)
    return cᵖᵐ * flux.surface_density * fluxes.sensible
end

@inline function OceananigansBC.getbc(flux::SAMOceanSurfaceFlux{SAMVaporFlux}, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields, dynamics_fields)
    fluxes = sam_centre_fluxes(i, j, grid, flux, fields, dynamics_fields)
    return flux.surface_density * fluxes.vapor
end

@inline function OceananigansBC.getbc(flux::SAMOceanSurfaceFlux{SAMXStress}, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields, dynamics_fields)
    u = @inbounds fields.u[i, j, 1]
    v = ℑxyᶠᶜᵃ(i, j, 1, grid, fields.v)
    return sam_ocean_fluxes(i, j, grid, flux, fields, dynamics_fields, u, v, ℑxᶠᵃᵃ).τˣ
end

@inline function OceananigansBC.getbc(flux::SAMOceanSurfaceFlux{SAMYStress}, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields, dynamics_fields)
    u = ℑxyᶜᶠᵃ(i, j, 1, grid, fields.u)
    v = @inbounds fields.v[i, j, 1]
    return sam_ocean_fluxes(i, j, grid, flux, fields, dynamics_fields, u, v, ℑyᵃᶠᵃ).τʸ
end

"""
    sam_surface_interface_density(pressure, z_centers, g)

SAM's surface-interface density `rhow(1)` (`setdata.f90`): the hydrostatic interface densities
`ρʷₖ = (pₖ₋₁ - pₖ) / ((zₖ - zₖ₋₁) g)` at the second and third interfaces, extrapolated
linearly, `ρʷ₁ = 2ρʷ₂ - ρʷ₃`. `pressure` [Pa] is the reference pressure at the cell centres
`z_centers`.
"""
function sam_surface_interface_density(pressure, z_centers, g)
    ρʷ₂ = (pressure[1] - pressure[2]) / (z_centers[2] - z_centers[1]) / g
    ρʷ₃ = (pressure[2] - pressure[3]) / (z_centers[3] - z_centers[2]) / g
    return 2ρʷ₂ - ρʷ₃
end

"""
    sam_ocean_surface_flux_boundary_conditions(grid, surface_temperature; moisture_name, microphysics,
                                               thermodynamic_constants, surface_density)

Bottom boundary conditions for `ρu`, `ρv`, `ρE` and the moisture density from SAM's per-column
`oceflx` ([`SAMOceanSurfaceFlux`](@ref)).
"""
function sam_ocean_surface_flux_boundary_conditions(grid, surface_temperature; moisture_name, microphysics,
                                                    thermodynamic_constants, surface_density)
    FT = eltype(grid)
    ρʷ = convert(FT, surface_density)
    # The model moves its microphysics (P3 lookup tables) to the device only when it is built; this copy
    # lives in the boundary condition, so it must be on the grid's architecture before kernels adapt it.
    microphysics = on_architecture(architecture(grid), microphysics)
    condition(component) = BoundaryCondition(Flux(), SAMOceanSurfaceFlux(component, surface_temperature, microphysics,
                                                                         thermodynamic_constants, ρʷ))
    bcs = (; ρu = FieldBoundaryConditions(bottom=condition(SAMXStress())),
             ρv = FieldBoundaryConditions(bottom=condition(SAMYStress())),
             ρE = FieldBoundaryConditions(bottom=condition(SAMSensibleHeatFlux())))
    moisture_density_name = Symbol("ρ", moisture_name)
    return merge(bcs, NamedTuple{(moisture_density_name,)}((FieldBoundaryConditions(bottom=condition(SAMVaporFlux())),)))
end
