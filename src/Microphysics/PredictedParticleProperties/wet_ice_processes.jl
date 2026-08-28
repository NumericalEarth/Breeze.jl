"""
$(TYPEDSIGNATURES)

Compute the density of newly accreted cloud rime from the rime-impact parameter.

Diagnose the cloud gamma PSD from `qᶜˡ` and `Nᶜˡ`, compute the droplet impact speed
relative to falling ice, form the rime-impact parameter `Ri`, and apply the
piecewise density fit of [Cober and List (1993)](@cite CoberList1993). When cloud
riming is inactive or the air is above freezing, the fallback value `400 kg m⁻³`
is used.

# Arguments
- `p3`: P3 microphysics scheme
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `cloud_rim`: Cloud-riming mass tendency [kg/kg/s]
- `T`: Temperature [K]
- `vᵢ`: Ice particle fall speed [m/s]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants
- `transport`: Air transport properties at `(T, P)`

# Returns
- Rime density [kg/m³]
"""
@inline function rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport,
                      μᶜˡ, λᶜˡ)
    FT = typeof(T)
    parameters = p3.process_rates
    qsmall = p3.minimum_mass_mixing_ratio

    minimum_rime_density = parameters.minimum_rime_density
    maximum_rime_density = parameters.maximum_rime_density
    T₀ = parameters.freezing_temperature
    ρᴸ = parameters.liquid_water_density

    # Dynamic viscosity of air. Written `η`, not `μⁱ`, which is the PSD shape
    # parameter everywhere else in this module.
    η = transport.ν * ρ
    g = p3_gravitational_acceleration(constants, FT)

    # The droplet impact speed is the mass-weighted Stokes velocity of the cloud
    # DSD, shared with `cloud_terminal_velocities`.
    stokes_prefactor = cloud_stokes_prefactor(g, ρᴸ, η, parameters.floors)
    cloud_terminal_velocity = cloud_mass_weighted_stokes_velocity(stokes_prefactor, μᶜˡ, λᶜˡ)
    cloud_mean_diameter = (μᶜˡ + 4) / λᶜˡ

    # Riming impact parameter Ri = c Dᶜ |vᵢ - Vt_qc| / (T₀ - T): large drops striking
    # fast at weak supercooling pack dense rime. The supercooling floor keeps Ri
    # finite as T → T₀, and the clamp holds Ri inside the range the fit below covers.
    inverse_supercooling = inv(min(-parameters.minimum_riming_supercooling, T - T₀))
    Ri = clamp(-(parameters.rime_impact_coefficient * cloud_mean_diameter) *
               abs(vᵢ - cloud_terminal_velocity) * inverse_supercooling,
               parameters.minimum_rime_impact, parameters.maximum_rime_impact)

    # Cober-List rime-density fit (see the docstring for the citation): a quadratic
    # in g/cm³ (hence the ×10³) below Ri = 8, a linear extension above it.
    #   Ri ≤ 8:  ρᶠ = (0.051 + 0.114 Ri - 0.0055 Ri²) × 10³
    #   Ri > 8:  ρᶠ = 611 + 72.25 (Ri - 8)
    # These are the coefficients of a published fit, not independently tunable
    # parameters, so they stay inline with the formula they belong to. What *is*
    # settable is the range of Ri over which the fit is trusted, clamped above.
    ρ_rime_Ri = ifelse(
        Ri <= FT(8),
        (FT(0.051) + FT(0.114) * Ri - FT(0.0055) * Ri^2) * FT(1000),
        FT(611) + FT(72.25) * (Ri - FT(8))
    )

    active_cloud_riming = (cloud_rim >= qsmall) & (qᶜˡ >= qsmall) & (T < T₀)
    ρᶠ = ifelse(active_cloud_riming, ρ_rime_Ri, parameters.unrimed_rime_density)

    return clamp(ρᶠ, minimum_rime_density, maximum_rime_density)
end

#####
##### Phase 2: Shedding and Refreezing (liquid fraction dynamics)
#####

"""
$(TYPEDSIGNATURES)

Compute liquid shedding rate from ice particles following
[Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

PSD-integrated shedding of liquid from mixed-phase ice particles with D ≥ 9 mm
(Rasmussen et al. 2011):

```math
q_{lshd} = F^f \\times f_{1pr28} \\times N^i \\times F^l
```

where `f1pr28 = ∫_{D≥9mm} m(D) N'(D) dD` (lookup table, Fl-blended mass),
`Fr = qirim / (qitot - qiliq)` is the rime fraction of ice-only mass, and
`Fl = qiliq / qitot` is the liquid fraction.

# Arguments
- `p3`: P3 microphysics scheme (provides shedding table)
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime fraction (= qᶠ/qⁱ) [-]
- `Fˡ`: Liquid fraction (= qʷⁱ/(qⁱ+qʷⁱ)) [-]
- `lookups`: [`P3IceLookups`](@ref) of the population

# Returns
- Rate of liquid → rain shedding [kg/kg/s]
"""
@inline function shedding_rate(p3, qʷⁱ, nⁱ, Fᶠ, Fˡ, lookups::P3IceLookups)
    qʷⁱ_eff = max(0, qʷⁱ)
    nⁱ_eff = max(0, nⁱ)

    # Lookup ∫_{D≥9mm} m(D) N'(D) dD (normalized per particle)
    f1pr28 = evaluate_at(p3.ice.bulk_properties.shedding, lookups.prep)

    # Fᶠ is the rime fraction of the ice-only mass, since qⁱ excludes qʷⁱ.
    rate = Fᶠ * f1pr28 * nⁱ_eff * Fˡ

    # Bound by available liquid: qlshd ≤ qwi / dt_safety
    rate = max(0, rate)
    τ_safety = p3.process_rates.sink_limiting_timescale
    rate = min(rate, qʷⁱ_eff / τ_safety)

    return rate
end

"""
$(TYPEDSIGNATURES)

Compute rain number source from shedding.

Shed liquid forms rain drops of approximately 1 mm diameter.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `shed_rate`: Liquid shedding mass rate [kg/kg/s]

# Returns
- Rate of rain number increase [1/kg/s]
"""
@inline function shedding_number_rate(p3, shed_rate)
    # Liquid-fraction shedding carries its own drop mass so it can be tuned
    # separately; by default it equals `shed_drop_mass`, the mass of a 1 mm drop.
    m_shed = p3.process_rates.shed_drop_mass_liqfrac

    return shed_rate / m_shed
end

"""
$(TYPEDSIGNATURES)

Compute the wet growth freezing capacity following
[Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

The wet growth capacity is the maximum rate at which collected
hydrometeors can be frozen, determined by the ventilated heat balance:

```math
q_{wgrth} = C f^{ve} \\left[Kᵃ(T_0-T) + \\frac{2π}{ℒᶠᵘˢ} ℒⁱ Dᵛ(ρ^{v+}-ρ^v)\\right] × N^i
```

When the collection rate (cloud + rain riming) exceeds this capacity,
the excess collected water stays liquid and is redirected into qʷⁱ.

# Arguments
- `p3`: P3 microphysics scheme
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `P`: Pressure [Pa]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants
- `transport`: Pre-computed air transport properties `(; Dᵛ, Kᵃ, ν)`

# Returns
- Wet growth capacity [kg/kg/s] (positive; zero when T ≥ T₀)
"""
@inline function wet_growth_capacity(p3, qⁱ, qʷⁱ, nⁱ, T, qᵛ, Fᶠ, ρᶠ, ρ,
                                     constants, transport,
                                     lookups = p3_ice_lookups(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, ρ))
    FT = typeof(qⁱ)
    parameters = p3.process_rates

    nⁱ_eff = max(0, nⁱ)

    T₀ = parameters.freezing_temperature
    below_freezing = T < T₀

    ℒᶠᵘˢ = fusion_latent_heat(constants, T)
    ℒⁱ = sublimation_latent_heat(constants, T)

    Kᵃ = transport.Kᵃ
    Dᵛ = transport.Dᵛ
    ν  = transport.ν

    q_sat0 = freezing_point_saturation_mass_fraction(constants, T₀, ρ)

    # Ventilation integral (same as deposition/refreezing)
    C_fv = ventilation_from_terms(lookups.ventilation, lookups.ventilation_enhanced,
                                  ν, Dᵛ, lookups.ρ_correction, parameters.floors)

    # Heat balance: sensible + latent
    Q_sensible = Kᵃ * (T₀ - T)
    Q_latent = ℒⁱ * Dᵛ * ρ * (q_sat0 - qᵛ)

    # 2π/ℒᶠᵘˢ multiplies only the latent term; the sensible-conduction term uses
    # the capm convention directly.
    qwgrth = C_fv * (Q_sensible + 2 * FT(π) * Q_latent / ℒᶠᵘˢ) * nⁱ_eff

    return ifelse(below_freezing, max(0, qwgrth), zero(FT))
end

"""
$(TYPEDSIGNATURES)

Compute refreezing rate of liquid on ice using the heat-balance formula.

Below freezing, liquid coating on ice particles refreezes. The rate is
determined by the heat flux at the particle surface:

```math
\\frac{dm}{dt} = C f^{ve} \\left[Kᵃ(T_0-T) + \\frac{2π}{ℒᶠᵘˢ} ρ ℒⁱ Dᵛ (q^{v+}_0 - q^v)\\right]
```

That is the same particle heat balance that sets the wet-growth capacity, so this
is [`wet_growth_capacity`](@ref) capped by the liquid available on the ice. Above
freezing the capacity is already zero, which carries the temperature gate here.
See [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization)
appendix C, section i (and Mason 1971 for the underlying heat-balance form).

# Arguments
- `p3`: P3 microphysics scheme
- `qⁱ`: Ice mass fraction [kg/kg]
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants
- `transport`: Pre-computed air transport properties `(; Dᵛ, Kᵃ, ν)`

# Returns
- Rate of liquid → ice refreezing [kg/kg/s]
"""
@inline function refreezing_rate(p3, qⁱ, qʷⁱ, nⁱ, T, qᵛ, Fᶠ, ρᶠ, ρ,
                                 constants, transport)
    capacity = wet_growth_capacity(p3, qⁱ, qʷⁱ, nⁱ, T, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport)
    return refreezing_rate_from_capacity(p3, qʷⁱ, capacity)
end

@inline refreezing_rate_from_capacity(p3, qʷⁱ, capacity) =
    min(capacity, max(0, qʷⁱ) / p3.process_rates.sink_limiting_timescale)
