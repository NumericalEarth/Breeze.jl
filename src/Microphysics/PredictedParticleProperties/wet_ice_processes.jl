"""
$(TYPEDSIGNATURES)

Compute the density of newly accreted cloud rime from the rime-impact parameter.

Take the already-diagnosed cloud gamma PSD parameters `μᶜˡ` and `λᶜˡ`, compute the
droplet impact speed relative to falling ice, form the rime-impact parameter `Ri`, and
apply the piecewise density fit of [Cober and List (1993)](@cite CoberList1993). When cloud
riming is inactive or the air is above freezing, the fallback value `400 kg m⁻³`
is used.

# Arguments
- `p3`: P3 microphysics scheme
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `cloud_rim`: Cloud-riming mass tendency [kg/kg/s]
- `T`: Temperature [K]
- `𝕎ⁱ`: Ice particle fall speed [m/s]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants
- `transport`: Air transport properties at `(T, P)`
- `μᶜˡ`, `λᶜˡ`: Cloud gamma PSD shape and slope parameters, diagnosed upstream

# Returns
- Rime density [kg/m³]
"""
@inline function rime_density(p3, qᶜˡ, cloud_rim, T, 𝕎ⁱ, ρ, constants, transport,
                      μᶜˡ, λᶜˡ)
    FT = typeof(T)
    parameters = p3.process_rates
    qsmall = p3.minimum_mass_mixing_ratio

    ℂʳⁱᵐᵉ₃ = parameters.minimum_rime_density
    ℂʳⁱᵐᵉ₄ = parameters.maximum_rime_density
    ℂʳⁱᵐᵉ₅ = parameters.rime_impact_coefficient
    ℂʳⁱᵐᵉ₆ = parameters.minimum_rime_impact
    ℂʳⁱᵐᵉ₇ = parameters.maximum_rime_impact
    ℂʳⁱᵐᵉ₈ = parameters.unrimed_rime_density
    T₀ = parameters.freezing_temperature
    ρᴸ = parameters.liquid_water_density

    # Dynamic viscosity of air. Written `η`, not `μⁱ`, which is the PSD shape
    # parameter everywhere else in this module.
    η = transport.ν * ρ
    g = p3_gravitational_acceleration(constants, FT)

    # The droplet impact speed is the mass-weighted Stokes velocity of the cloud
    # DSD, shared with `cloud_terminal_velocities`.
    stokes_prefactor = cloud_stokes_prefactor(g, ρᴸ, η, parameters.floors)
    𝕎ᶜˡ = cloud_mass_weighted_stokes_velocity(stokes_prefactor, μᶜˡ, λᶜˡ)
    cloud_mean_diameter = (μᶜˡ + 4) / λᶜˡ

    # Riming impact parameter Ri = c Dᶜ |𝕎ⁱ - 𝕎ᶜˡ| / (T₀ - T): large drops striking
    # fast at weak supercooling pack dense rime. The supercooling floor keeps Ri
    # finite as T → T₀, and the clamp holds Ri inside the range the fit below covers.
    inverse_supercooling = inv(min(-parameters.minimum_riming_supercooling, T - T₀))
    Ri = clamp(-(ℂʳⁱᵐᵉ₅ * cloud_mean_diameter) *
               abs(𝕎ⁱ - 𝕎ᶜˡ) * inverse_supercooling,
               ℂʳⁱᵐᵉ₆, ℂʳⁱᵐᵉ₇)

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
    ρᶠ = ifelse(active_cloud_riming, ρ_rime_Ri, ℂʳⁱᵐᵉ₈)

    return clamp(ρᶠ, ℂʳⁱᵐᵉ₃, ℂʳⁱᵐᵉ₄)
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
    qʷⁱ₊ = max(0, qʷⁱ)
    nⁱ₊ = max(0, nⁱ)

    # Lookup ∫_{D≥9mm} m(D) N'(D) dD (normalized per particle)
    f1pr28 = evaluate_at(p3.ice.bulk.shedding, lookups.prep)

    # Fᶠ is the rime fraction of the ice-only mass, since qⁱ excludes qʷⁱ.
    rate = Fᶠ * f1pr28 * nⁱ₊ * Fˡ

    # Bound by available liquid: qlshd ≤ qwi / dt_safety
    rate = max(0, rate)
    τ_safety = p3.process_rates.sink_limiting_timescale
    rate = min(rate, qʷⁱ₊ / τ_safety)

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
    ℂˢʰᵉᵈ₂ = p3.process_rates.shed_drop_mass_liqfrac

    return shed_rate / ℂˢʰᵉᵈ₂
end

"""
$(TYPEDSIGNATURES)

Compute the wet growth and refreezing capacities, the maximum rates at which
collected water and the liquid coating can freeze, from the ventilated heat
balance of an ice particle whose surface is held at `T₀`. Both share the
conduction and vapor-diffusion terms; see [`wet_growth_capacity`](@ref) and
[`refreezing_rate`](@ref) for what each freezes.

# Returns
- `(wet_growth, refreezing)` capacities [kg/kg/s], zero when T ≥ T₀
"""
@inline function freezing_capacities(p3, nⁱ, T, qᵛ, ρ, constants, transport, lookups)
    FT = typeof(T)
    parameters = p3.process_rates
    T₀ = parameters.freezing_temperature

    q_sat0 = freezing_point_saturation_mass_fraction(constants, T₀, ρ)

    # Ventilation integral (same as deposition/melting)
    ventilated_capacitance = ventilation_from_terms(lookups.ventilation, lookups.enhanced_ventilation,
                                                    transport.ν, transport.Dᵛ, lookups.ρ_correction,
                                                    parameters.floors)

    # 2π (not 4π) because ventilated_capacitance stores capm = 2 × capacitance, as in melting.
    surface_factor = 2 * FT(π) * ventilated_capacitance * max(0, nⁱ)
    conduction = surface_factor * transport.Kᵃ * (T₀ - T)          # [W/kg]
    vapor_flux = surface_factor * transport.Dᵛ * ρ * (q_sat0 - qᵛ) # [kg/kg/s]

    # Wet growth: collected drops warm from T to T₀, then freeze at a wet surface,
    # so the latent heats are taken at T₀. Drops are no colder than homogeneous
    # freezing, which keeps the denominator positive.
    ℒˡ₀ = vaporization_latent_heat(constants, T₀)
    ℒᶠʳ₀ = fusion_latent_heat(constants, T₀)
    cˡ = constants.liquid.heat_capacity
    drop_temperature = max(T, parameters.homogeneous_freezing_temperature)
    wet_growth = (conduction + ℒˡ₀ * vapor_flux) / (ℒᶠʳ₀ + cˡ * (drop_temperature - T₀))

    # Refreezing: the coating needs no warming.
    ℒⁱ = sublimation_latent_heat(constants, T)
    ℒᶠʳ = fusion_latent_heat(constants, T)
    refreezing = (conduction + ℒⁱ * vapor_flux) / ℒᶠʳ

    below_freezing = T < T₀
    return ifelse(below_freezing, max(0, wet_growth), zero(wet_growth)),
           ifelse(below_freezing, max(0, refreezing), zero(refreezing))
end

"""
$(TYPEDSIGNATURES)

Compute the wet growth freezing capacity following
[Musil (1970)](@cite Musil1970), Eq. (A9), as in P3 v4.

The wet growth capacity is the maximum rate at which collected hydrometeors can
be frozen. Collected drops are warmed from `T` to `T₀` and freeze at a wet
surface, so the heat balance uses the vaporization and fusion latent heats at `T₀`:

```math
\\dot{q}_\\text{wet} = 2π C f^{ve} \\frac{Kᵃ(T_0-T) + ℒˡ_0 Dᵛ(ρ^{v+}_0-ρ^v)}{ℒᶠʳ_0 + cˡ (T - T_0)} × N^i
```

When the collection rate (cloud + rain riming) exceeds this capacity,
the excess collected water stays liquid and is redirected into qʷⁱ.

# Arguments
- `p3`: P3 microphysics scheme
- `qⁱ`: Ice mass fraction [kg/kg]
- `qʷⁱ`: Liquid water on ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants
- `transport`: Pre-computed air transport properties `(; Dᵛ, Kᵃ, ν)`

# Returns
- Wet growth capacity [kg/kg/s] (positive; zero when T ≥ T₀)
"""
@inline wet_growth_capacity(p3, qⁱ, qʷⁱ, nⁱ, T, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport,
                            lookups = p3_ice_lookups(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, ρ)) =
    first(freezing_capacities(p3, nⁱ, T, qᵛ, ρ, constants, transport, lookups))

"""
$(TYPEDSIGNATURES)

Compute the heat-limited refreezing capacity of the liquid coating on ice.
See [`refreezing_rate`](@ref).
"""
@inline refreezing_capacity(p3, qⁱ, qʷⁱ, nⁱ, T, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport,
                            lookups = p3_ice_lookups(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, ρ)) =
    last(freezing_capacities(p3, nⁱ, T, qᵛ, ρ, constants, transport, lookups))

"""
$(TYPEDSIGNATURES)

Compute refreezing rate of liquid on ice using the heat-balance formula.

Below freezing, liquid coating on ice particles refreezes. The rate is
determined by the heat flux at the particle surface,
[Cholette et al. (2019)](@cite Cholette2019parameterization), Eq. (A5):

```math
\\frac{dm}{dt} = \\frac{2π}{ℒᶠʳ} C f^{ve} \\left[Kᵃ(T_0-T) + ρ ℒⁱ Dᵛ (q^{v+}_0 - q^v)\\right]
```

The conduction and vapor terms are those of [`wet_growth_capacity`](@ref), but the
coating is already at `T₀`, so there is no drop-warming term. The capacity is capped
by the liquid available on the ice; above freezing it is zero, which carries the
temperature gate here.

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
    capacity = refreezing_capacity(p3, qⁱ, qʷⁱ, nⁱ, T, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport)
    return refreezing_rate_from_capacity(p3, qʷⁱ, capacity)
end

@inline refreezing_rate_from_capacity(p3, qʷⁱ, capacity) =
    min(capacity, max(0, qʷⁱ) / p3.process_rates.sink_limiting_timescale)
