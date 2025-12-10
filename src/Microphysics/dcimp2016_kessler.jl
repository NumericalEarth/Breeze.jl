using ..Thermodynamics:
    MoistureMassFractions,
    mixture_heat_capacity,
    dry_air_gas_constant,
    vapor_gas_constant,
    PlanarLiquidSurface,
    saturation_vapor_pressure,
    temperature,
    density,
    is_absolute_zero,
    with_moisture,
    total_specific_moisture,
    AbstractThermodynamicState

using Oceananigans: Oceananigans, CenterField, ZFaceField
using Oceananigans.Grids: znode

using DocStringExtensions: TYPEDSIGNATURES

"""
$(TYPEDSIGNATURES)

DCMIP2016 
Kessler (1969) warm-rain bulk microphysics scheme following Klemp and Wilhelmson (1978).

Fortran reference: https://gitlab.in2p3.fr/ipsl/projets/dynamico/dynamico/-/blob/master/src/dcmip2016_kessler_physic.f90

This scheme represents three moisture categories:
- Water vapor (`qᵛ`)
- Cloud water (`qᶜ`) - liquid water that moves with the flow
- Rain water (`qʳ`) - liquid water that falls relative to the surrounding air

Constants from `kessler.f90`:
- `f2x = 17.27`: Clausius-Clapeyron coefficient
- `f5 = 237.3 * f2x * 2500000 / 1003`: Saturation adjustment coefficient
- `xk = 0.2875`: kappa (R/cp)
- `psl = 1000`: Reference pressure in mb
- `rhoqr = 1000`: Density of liquid water (kg/m³)

The microphysical rates are pre-computed in `update_microphysical_fields!` to avoid
redundant calculations across tendency functions. The rates stored are:
- `Sᶜᵒⁿᵈ`: Net condensation rate (vapor → cloud), can be negative for evaporation from cloud
- `Sᵃᵘᵗᵒ`: Cloud-to-rain conversion rate (autoconversion + accretion combined, using implicit formula)
- `Sᵃᶜᶜʳ`: Reserved (set to 0, accretion is combined with autoconversion)
- `Sᵉᵛᵃᵖ`: Evaporation rate (rain → vapor)
- `wʳ`: Rain terminal velocity (negative = downward) for sedimentation

The cloud-to-rain conversion uses the original Kessler implicit formula from KW eq. 2.13:
```
qrprod = qc - (qc - Δt*max(k_auto*(qc - qc_thresh), 0)) / (1 + Δt*k_accr*qr^0.875)
```
This implicit formulation ensures numerical stability and positivity for large timesteps.

Rain sedimentation follows KW eq. 2.15 for terminal velocity:
```
wʳ = -36.34 * (qʳ * 0.001 * ρ)^0.1364 * sqrt(ρ₀/ρ)
```
The negative sign indicates downward motion. The sedimentation is handled by adding
the terminal velocity to the advection velocity for rain water.
"""
struct KesslerMicrophysics end

const KM = KesslerMicrophysics

prognostic_field_names(::KM) = (:ρqᵛ, :ρqᶜ, :ρqʳ)

function materialize_microphysical_fields(::KM, grid, boundary_conditions)
    # Prognostic fields (density-weighted)
    ρqᵛ = CenterField(grid, boundary_conditions=boundary_conditions.ρqᵛ)
    ρqᶜ = CenterField(grid, boundary_conditions=boundary_conditions.ρqᶜ)
    ρqʳ = CenterField(grid, boundary_conditions=boundary_conditions.ρqʳ)
    # Diagnostic mixing ratios
    qᵛ = CenterField(grid)
    qᶜ = CenterField(grid)
    qʳ = CenterField(grid)
    # Pre-computed microphysical rates (to avoid redundant calculations)
    Sᶜᵒⁿᵈ = CenterField(grid)  # Net condensation rate
    Sᵃᵘᵗᵒ = CenterField(grid)  # Cloud-to-rain rate (autoconversion + accretion combined)
    Sᵃᶜᶜʳ = CenterField(grid)  # Reserved (set to 0)
    Sᵉᵛᵃᵖ = CenterField(grid)  # Rain evaporation rate
    # Rain sedimentation velocity (negative = downward)
    wʳ = ZFaceField(grid)
    return (; ρqᵛ, ρqᶜ, ρqʳ, qᵛ, qᶜ, qʳ, Sᶜᵒⁿᵈ, Sᵃᵘᵗᵒ, Sᵃᶜᶜʳ, Sᵉᵛᵃᵖ, wʳ)
end

#
# ρ = pᵣ / (Rᵐ T)
# p′ = 
# ρ = ρᵣ + ρ′
# ∂t ρ + ∇⋅(ρ u) = ∇ ⋅ (ρᵣ u) + ∂t ρ′ + ∇ ⋅ (ρ′ u) + ∇ ⋅ (ρᵣ u′) = 0

# O(0): ∇ ⋅ (ρᵣ u) = 0
# O(ϵ): + ∂t ρ′ + ∇ ⋅ (ρ′ u) + ∇ ⋅ (ρᵣ u′) = 0

@inline function update_microphysical_fields!(μ, ::KM, i, j, k, grid, ρ, 𝒰, p′, constants, Δt)
    T = temperature(𝒰, constants)
    pᵣ = 𝒰.reference_pressure
    p = pᵣ + p′  # Full pressure = reference + perturbation

    @inbounds begin
        # Compute specific humidities from prognostic density-weighted fields
        qᵛ = μ.ρqᵛ[i, j, k] / ρ
        qᶜ = μ.ρqᶜ[i, j, k] / ρ
        qʳ = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = qᵛ
        μ.qᶜ[i, j, k] = qᶜ
        μ.qʳ[i, j, k] = qʳ

        # Total specific humidity for conversion to mixing ratio
        qᵗ = qᵛ + qᶜ + qʳ

        # Convert specific humidities to mixing ratios for Kessler physics
        rᵛ = specific_humidity_to_mixing_ratio(qᵛ, qᵗ)
        rᶜ = specific_humidity_to_mixing_ratio(qᶜ, qᵗ)
        rʳ = specific_humidity_to_mixing_ratio(qʳ, qᵗ)

        # Compute microphysical rates in mixing ratio space
        Sᶜᵒⁿᵈ_r, Sʳᵃⁱⁿ_r, Sᵉᵛᵃᵖ_r = kessler_microphysical_rates(rᵛ, rᶜ, rʳ, ρ, T, p, Δt)

        # Convert rates from mixing ratio to specific humidity
        # The conversion factor is (1 - qᵗ) since dr/dt = dq/dt / (1 - qᵗ) for small changes
        # Therefore dq/dt = dr/dt * (1 - qᵗ)
        conversion_factor = 1 - qᵗ
        Sᶜᵒⁿᵈ = Sᶜᵒⁿᵈ_r * conversion_factor
        Sʳᵃⁱⁿ = Sʳᵃⁱⁿ_r * conversion_factor
        Sᵉᵛᵃᵖ = Sᵉᵛᵃᵖ_r * conversion_factor

        μ.Sᶜᵒⁿᵈ[i, j, k] = Sᶜᵒⁿᵈ
        μ.Sᵃᵘᵗᵒ[i, j, k] = Sʳᵃⁱⁿ  # Combined cloud-to-rain rate (autoconversion + accretion)
        μ.Sᵃᶜᶜʳ[i, j, k] = 0      # No longer computed separately
        μ.Sᵉᵛᵃᵖ[i, j, k] = Sᵉᵛᵃᵖ

        # Compute rain terminal velocity at cell center (negative = downward)
        # Following KW eq. 2.15: velqr = 36.34 * (qr * r)^0.1364 * sqrt(ρ₀/ρ)
        # where r = 0.001 * ρ
        # Note: terminal velocity uses mixing ratio (rʳ) not specific humidity
        # For the density ratio, we use ρ₀ = ρ (simplified; assumes near-surface reference)
        # This can be improved by passing in the surface density
        wʳ_center = -kessler_terminal_velocity(rʳ, ρ)
        
        # Store at cell center - will be interpolated to face during advection
        # For now, store at face k (bottom face of cell k)
        μ.wʳ[i, j, k] = wʳ_center
    end
    return nothing
end

@inline function compute_moisture_fractions(i, j, k, grid, ::KM, ρ, qᵗ, μ)
    @inbounds begin
        qᵛ = μ.ρqᵛ[i, j, k] / ρ
        qᶜ = μ.ρqᶜ[i, j, k] / ρ
        qʳ = μ.ρqʳ[i, j, k] / ρ
    end
    return MoistureMassFractions(qᵛ, qᶜ + qʳ)
end

@inline maybe_adjust_thermodynamic_state(𝒰, ::KM, μ, qᵗ, constants) = 𝒰

#@inline microphysical_velocities(::KM, ::Val{:ρqʳ}, μ) = (u = nothing, v = nothing, w = μ.wʳ)
@inline microphysical_velocities(::KM, name, μ) = nothing

#####
##### Kessler scheme functions following kessler.f90
#####

# Constants from kessler.f90
const kessler_f2x = 17.27
const kessler_xk = 0.2875  # kappa (R/cp)
const kessler_psl = 1000.0  # pressure at sea level (mb)
const kessler_rhoqr = 1000.0  # density of liquid water (kg/m³)

#####
##### Conversion between specific humidity and mixing ratio
#####
# Kessler scheme uses mixing ratio (mass of hydrometeor / mass of dry air)
# Breeze uses specific humidity (mass of hydrometeor / total mass of moist air)
# Conversion: r = q / (1 - qᵗ)  where qᵗ is total specific humidity
#             q = r / (1 + rᵗ)  where rᵗ is total mixing ratio
#####

"""
    specific_humidity_to_mixing_ratio(q, qᵗ)

Convert specific humidity `q` to mixing ratio `r`.
`qᵗ` is the total specific humidity (sum of all moisture species).

The conversion is: r = q / (1 - qᵗ)
"""
@inline specific_humidity_to_mixing_ratio(q, qᵗ) = q / (1 - qᵗ)

"""
    mixing_ratio_to_specific_humidity(r, rᵗ)

Convert mixing ratio `r` to specific humidity `q`.
`rᵗ` is the total mixing ratio (sum of all moisture species).

The conversion is: q = r / (1 + rᵗ)
"""
@inline mixing_ratio_to_specific_humidity(r, rᵗ) = r / (1 + rᵗ)

"""
    kessler_saturation_mixing_ratio(T, p)

Compute saturation vapor mixing ratio (kg/kg, w.r.t. dry air) following KW eq. 2.11.
Uses temperature T (K) and pressure p (Pa).
In the original Fortran: qvs = pc * exp(f2x * (Π*θ - 273) / (Π*θ - 36))
where pc = 3.8 / (Π^(1/xk) * psl) = 3.8 / (p/p0 * psl) with p in suitable units.

Note: This returns mixing ratio (mass of vapor / mass of dry air), not specific humidity.
"""
@inline function kessler_saturation_mixing_ratio(T, p)
    # Convert pressure from Pa to mb for consistency with Fortran
    p_mb = p / 100
    # pc = 3.8 / p_mb (since Π^(1/xk) * psl = (p/p0)^(1/xk*xk) * psl = p/p0 * psl ≈ p_mb for p0=1000mb)
    pc = 3.8 / p_mb
    qvs = pc * exp(kessler_f2x * (T - 273) / (T - 36))
    return qvs
end

"""
    kessler_terminal_velocity(qʳ, ρ, ρ₀)

Compute liquid water terminal velocity (m/s) following KW eq. 2.15.
Uses three-argument form with explicit reference density.
"""
@inline function kessler_terminal_velocity(qʳ, ρ, ρ₀)
    r = 0.001 * ρ  # r(k) = 0.001 * rho(k) in Fortran
    rhalf = sqrt(ρ₀ / ρ)
    return 36.34 * (qʳ * r)^0.1364 * rhalf
end

"""
    kessler_terminal_velocity(qʳ, ρ)

Compute liquid water terminal velocity (m/s) following KW eq. 2.15.
Simplified two-argument form assuming ρ₀ ≈ ρ (valid near surface).
The full formula includes a density correction factor sqrt(ρ₀/ρ).
"""
@inline function kessler_terminal_velocity(qʳ, ρ)
    r = 0.001 * ρ
    # Simplified: assume rhalf ≈ 1 (near-surface approximation)
    # For better accuracy, pass the surface density explicitly
    return 36.34 * (qʳ * r)^0.1364
end

"""
    kessler_microphysical_rates(rᵛ, rᶜ, rʳ, ρ, T, pᵣ, Δt)

Compute all Kessler microphysical process rates at once.
All moisture inputs (rᵛ, rᶜ, rʳ) must be mixing ratios (w.r.t. dry air mass).

Returns (Sᶜᵒⁿᵈ, Sʳᵃⁱⁿ, Sᵉᵛᵃᵖ) as rates in mixing ratio units:
- Sᶜᵒⁿᵈ: Net condensation rate (vapor → cloud), limited by available cloud for evaporation
- Sʳᵃⁱⁿ: Cloud-to-rain conversion rate (autoconversion + accretion, using implicit formula)
- Sᵉᵛᵃᵖ: Rain evaporation rate (rain → vapor)

The cloud-to-rain conversion uses the original Kessler implicit formula from KW:
```
rʳprod = rᶜ - (rᶜ - Δt*max(k_auto*(rᶜ - rc_thresh), 0)) / (1 + Δt*k_accr*rʳ^0.875)
```
where k_auto = 0.001 s⁻¹, rc_thresh = 0.001, and k_accr = 2.2 s⁻¹.
The rate is then Sʳᵃⁱⁿ = rʳprod / Δt.

These rates are related to the tendencies as (in mixing ratio space):
- Sᵛ = -Sᶜᵒⁿᵈ + Sᵉᵛᵃᵖ  (vapor tendency)
- Sᶜ = Sᶜᵒⁿᵈ - Sʳᵃⁱⁿ  (cloud tendency)
- Sʳ = Sʳᵃⁱⁿ - Sᵉᵛᵃᵖ  (rain tendency)
- Sᵉ = ρ * Lv * (Sᶜᵒⁿᵈ - Sᵉᵛᵃᵖ)  (energy tendency)

Note: Rates must be converted from mixing ratio to specific humidity before use in Breeze.
"""

function microphysics_model_update!(km::KM, model)
    grid = model.grid
    arch = grid.architecture
    Δt = model.clock.last_Δt

    # Prognostic fields updated by Kessler scheme.
    ρθ = model.formulation.thermodynamics.potential_temperature_density
    ρqᵛ = model.microphysical_fields.ρqᵛ
    ρqʳ = model.microphysical_fields.ρqʳ
    ρqᶜˡ = model.microphysical_fields.ρqᶜˡ

    # Diagnostic fields updated by Kessler scheme.
    θ = model.formulation.thermodynamics.potential_temperature
    qᵛ = model.microphysical_fields.qᵛ
    qʳ = model.microphysical_fields.qʳ
    qᶜˡ = model.microphysical_fields.qᶜˡ
    T = model.temperature
    Pʳ = model.microphysical_fields.precipitation_rate

    fields_to_update = (ρθ, ρqᵛ, ρqʳ, ρqᶜˡ, θ, qᵛ, qʳ, qᶜˡ, T, Pʳ)
    launch!(arch, grid, :xy, _kessler_microphysical_update!,
            fields_to_update, grid, other_needed_fields...)

    return nothing
end

@kernel function _kessler_microphysical_update!(fields, grid, everything_else...)
    i, j = @index(Global, NTuple)

    for k = 1:grid.Nz

        # Saturation mixing ratio following KW eq. 2.11
        rᵛˢ = kessler_saturation_mixing_ratio(T[i, j, k], pᵣ[i, j, k])

        # Saturation adjustment: prod = (rv - rvs) / (1 + rvs*f5/(T - 36)^2)
        prod = (rᵛ - rᵛˢ) / (1 + rᵛˢ * (4093 * 2.5e6 / 1003) / (T - 36)^2) 

        # Net condensation rate (limited by available cloud water for evaporation)
        # From Fortran: rc = max(rc + max(prod, -rc), 0)
        # This means condensation is max(prod, -rc), i.e., if prod < 0, we can only evaporate up to rc
        Sᶜᵒⁿᵈ = max(prod, -rᶜ) / Δt

        # Cloud-to-rain conversion rate (autoconversion + accretion) following KW eq. 2.13a,b
        # Original Fortran implicit formula:
        # rrprod = rc - (rc - dt*max(0.001*(rc-0.001),0)) / (1 + dt*2.2*rr^0.875)
        # This is an implicit Euler discretization that guarantees positivity.
        # We use Δt to compute the effective rate.

        # Implicit formula for rrprod (amount converted from cloud to rain in Δt)
        rrprod = rᶜ - (rᶜ - Δt * max(0.001 * (rᶜ - 0.001), 0)) / (1 + Δt * 2.2 * rʳ^0.875)

        # Convert to a rate (per unit time)
        Sʳᵃⁱⁿ = rrprod / Δt

        # Rain evaporation rate following KW eq. 2.14a,b
        # Only occurs when subsaturated (rvs > rv)
        r = 0.001 * ρ
        rrr = r * rʳ  # Product of r and rain mixing ratio
        numerator = (1.6 + 124.9 * rrr^0.2046) * rrr^0.525

        p_mb = pᵣ / 100
        pc = 3.8 / p_mb
        subsaturation = max(rᵛˢ - rᵛ, 0)
        denomerator = 2550000 * pc / (3.8 * rᵛˢ) + 540000
        ern_rate = numerator / denomerator * subsaturation / (r * rᵛˢ + 1e-20)

        # Evaporation is limited by available rain and available subsaturation
        # From Fortran: ern = min(dt*(ern_rate), max(-prod - rc, 0), rr)
        # The original Fortran computes ern as an amount, we want the rate
        ern_max = max(-prod - rᶜ, 0)  # Maximum evaporable amount based on subsaturation
        Sᵉᵛᵃᵖ = min(ern_rate, ern_max / Δt, rʳ / Δt)

        #return Sᶜᵒⁿᵈ, Sʳᵃⁱⁿ, Sᵉᵛᵃᵖ
    end
end