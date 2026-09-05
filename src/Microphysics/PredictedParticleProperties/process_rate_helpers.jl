#####
##### P3 Process Rates
#####
##### Microphysical process rate calculations for the P3 scheme.
##### All rate functions take the P3 scheme as first positional argument
##### to access parameters. No keyword arguments (GPU compatibility).
#####
##### Notation follows docs/src/appendix/notation.md
#####

#####
##### Utility functions
#####

"""
$(TYPEDSIGNATURES)

Compute proportional rescaling factor for sink rates so that
`total_sink × dt_safety` does not exceed `available_mass`.

Returns 1 when sinks are within budget, or `available_mass / (total_sink × dt_safety)`
when they exceed it. All arguments must be positive or zero.
GPU-compatible: uses `ifelse` instead of branching.
"""
@inline function sink_limiting_factor(total_sink, available_mass, dt_safety)
    projected = total_sink * dt_safety
    return ifelse(projected > available_mass,
                  available_mass / max(projected, eps(typeof(available_mass))),
                  one(typeof(available_mass)))
end

@inline function p3_ice_saturation_specific_humidity(T, ρ, constants, freezing_temperature, qᵛ⁺ˡ)
    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
    return ifelse(T >= freezing_temperature, qᵛ⁺ˡ, qᵛ⁺ⁱ)
end

@inline function p3_ice_saturation_specific_humidity(T, ρ, constants, freezing_temperature)
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    return p3_ice_saturation_specific_humidity(T, ρ, constants, freezing_temperature, qᵛ⁺ˡ)
end

@inline function p3_adjustment_ice_saturation_specific_humidity(T, P, qᵗ, constants, freezing_temperature)
    qᵛ⁺ˡ = adjustment_saturation_specific_humidity(T, P, qᵗ, constants, PlanarLiquidSurface())
    qᵛ⁺ⁱ = adjustment_saturation_specific_humidity(T, P, qᵗ, constants, PlanarIceSurface())
    return ifelse(T >= freezing_temperature, qᵛ⁺ˡ, qᵛ⁺ⁱ)
end

"""
$(TYPEDSIGNATURES)

Cap vapor sinks and sources against the moist-adiabatic saturation-adjustment
budget.

Defining the liquid saturation-adjustment increment `δqˡ = (qᵛ - qᵛ⁺ˡ) / ξˡ` with
the moist-static feedback factor `ξˡ = 1 + ℒˡ² qᵛ⁺ˡ / (cᵖᵈ Rᵛ T²)`:

- Liquid-phase condensation sinks (`cond > 0`, `ccn_activation_mass`, `rain_cond`,
  `coat_cond`) cannot exceed `max(0, δqˡ)`.
- Liquid-phase evaporation sources (`cond < 0`, `rain_evap`, `coat_evap`)
  cannot exceed `max(0, -δqˡ)`.

The rescaled liquid tendencies are then carried into a post-liquid state
`(qᵛ_after, T_after)`, and `qᵛ⁺ⁱ_after` is recomputed at `T_after` to evaluate
`ξⁱ_after = 1 + ℒⁱ_after² qᵛ⁺ⁱ_after / (cᵖᵈ Rᵛ T_after²)`. With the ice
increment `δqⁱ = (qᵛ_after - qᵛ⁺ⁱ_after) / ξⁱ_after`:

- Ice-phase deposition sinks (`dep > 0`, `nuc_q`) cannot exceed
  `max(0, δqⁱ)`.
- Ice-phase sublimation sources (`dep < 0`) cannot exceed
  `max(0, -δqⁱ)`.

Number rates `ccn_activation_number` and `nuc_n` are scaled by the same factor as
their companion mass rates to preserve mean particle mass.

Returns a NamedTuple of the possibly-rescaled rates.
"""
@inline function limit_vapor_rates(cond, ccn_activation_mass, ccn_activation_number,
                                   rain_cond, rain_evap, dep, coat_cond, coat_evap,
                                   nuc_q, nuc_n, qᵛ, qᵛ⁺ˡ, T, P, qᵗ, constants,
                                   dt_safety, freezing_temperature)
    FT = typeof(qᵛ)
    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    ξˡ = liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)

    # Liquid-phase saturation-adjustment caps. In the SCF=1 limit the condensation
    # and evaporation budgets collapse to the same signed increment `δqˡ`;
    # condensation sees the positive part, evaporation the negative part.
    δqˡ = (qᵛ - qᵛ⁺ˡ) / ξˡ
    qcon_cap = max(0, δqˡ)
    qevp_cap = max(0, -δqˡ)

    # Condensation cap
    cond_sink_total = max(0, cond) + ccn_activation_mass + rain_cond + coat_cond
    f_cond = sink_limiting_factor(cond_sink_total, qcon_cap, dt_safety)

    ccn_activation_mass = ccn_activation_mass * f_cond
    ccn_activation_number = ccn_activation_number * f_cond
    rain_cond = rain_cond * f_cond
    coat_cond = coat_cond * f_cond

    # Evaporation cap: zero when supersaturated, otherwise rescale the lumped
    # evaporation rates to fit within `qevp_cap`.
    evp_total = max(0, -cond) + rain_evap + coat_evap
    f_evp = sink_limiting_factor(evp_total, qevp_cap, dt_safety)

    cond = max(0, cond) * f_cond + min(0, cond) * f_evp
    rain_evap = rain_evap * f_evp
    coat_evap = coat_evap * f_evp

    # Ice-phase cap, after netting the rescaled liquid tendencies into qᵛ and T.
    net_liquid = max(0, cond) + ccn_activation_mass + rain_cond + coat_cond -
                 rain_evap - coat_evap - max(0, -cond)
    qᵛ_after = qᵛ - net_liquid * dt_safety
    T_after = T + net_liquid * ℒˡ * dt_safety / cᵖᵈ
    qᵛ⁺ⁱ_after = p3_adjustment_ice_saturation_specific_humidity(T_after, P, qᵗ, constants, freezing_temperature)
    ℒⁱ_after = sublimation_latent_heat(constants, T_after)
    ξⁱ_after = ice_psychrometric_correction(constants, ℒⁱ_after, qᵛ⁺ⁱ_after, Rᵛ, T_after)

    # Ice-phase deposition / sublimation caps.
    δqⁱ = (qᵛ_after - qᵛ⁺ⁱ_after) / ξⁱ_after
    qdep_cap = max(0, δqⁱ)
    qsub_cap = max(0, -δqⁱ)

    # Deposition cap
    dep_sink_total = max(0, dep) + nuc_q
    f_dep = sink_limiting_factor(dep_sink_total, qdep_cap, dt_safety)

    nuc_q = nuc_q * f_dep
    nuc_n = nuc_n * f_dep

    # Sublimation cap
    sub_total = max(0, -dep)
    f_sub = sink_limiting_factor(sub_total, qsub_cap, dt_safety)

    dep = max(0, dep) * f_dep + min(0, dep) * f_sub

    return (; cond, ccn_activation_mass, ccn_activation_number, rain_cond, rain_evap,
            dep, coat_cond, coat_evap, nuc_q, nuc_n)
end

"""
$(TYPEDSIGNATURES)

Mass [kg] of a newly activated cloud droplet, the sphere of radius
`parameters.activated_droplet_radius` at the liquid water density. It converts a
CCN activation *number* rate into the matching *mass* rate wherever one of the two
is diagnosed from the other.
"""
@inline function activated_droplet_mass(parameters, FT)
    r₀ = FT(parameters.activated_droplet_radius)
    return 4 * FT(π) / 3 * FT(parameters.liquid_water_density) * r₀^3
end

"""
$(TYPEDSIGNATURES)

Cloud droplets per unit cloud mass, ``Nᶜˡ / (ρ qᶜˡ) = nᶜˡ / qᶜˡ`` [1/kg]. `Nᶜˡ` is
volumetric [1/m³] and `qᶜˡ` is a mass fraction [kg/kg], so this is the factor that
turns a cloud mass rate [kg/kg/s] into its companion number rate [1/kg/s], keeping
the two consistent in mean droplet mass.

Zero where the cloud mass is below `minimum_mass_mixing_ratio`, the threshold under
which the scheme treats cloud water as absent. The threshold matters: advection and
sedimentation leave positive but negligible `qᶜˡ` (down to subnormal values) in
cloud-free cells while the DSD diagnosis floors `Nᶜˡ` above zero, so the unguarded
quotient overflows to `Inf` and any companion rate of zero then yields `Inf × 0 = NaN`.
The guard also avoids evaluating the division at all for absent cloud.
"""
@inline function cloud_number_per_cloud_mass(Nᶜˡ, ρ, qᶜˡ, minimum_mass_mixing_ratio)
    FT = typeof(qᶜˡ)
    absent = qᶜˡ < minimum_mass_mixing_ratio
    qᶜˡ_safe = ifelse(absent, one(FT), qᶜˡ)
    return ifelse(absent, zero(FT), Nᶜˡ / (ρ * qᶜˡ_safe))
end

"""
$(TYPEDSIGNATURES)

Cloud-droplet number budget, ``∂n^{cl}/∂t`` [1/kg/s], before homogeneous freezing:
CCN activation is the only source, and the sinks are autoconversion, accretion,
self-collection, riming, heterogeneous freezing, and above-freezing collection by
ice. Each number sink is the companion of a mass rate, so the two stay consistent
in mean droplet mass.
"""
@inline function cloud_number_tendency_before_homogeneous_freezing(p3, ρ, qᶜˡ, Nᶜˡ,
                                                                  ccn_activation_mass,
                                                                  ccn_activation_number,
                                                                  autoconversion, accretion,
                                                                  self_collection,
                                                                  riming_number,
                                                                  freezing_number,
                                                                  warm_collection_number)
    FT = typeof(ρ)
    parameters = p3.process_rates
    seed_drop_mass = activated_droplet_mass(parameters, FT)
    activation_number = ifelse(iszero(ccn_activation_number),
                               ccn_activation_mass / seed_drop_mass,
                               ccn_activation_number)
    autoconversion_number = cloud_number_loss_from_autoconversion(p3, autoconversion, qᶜˡ, Nᶜˡ, ρ)
    # The mass threshold keeps the quotient bounded by Nᶜˡ / (ρ qmin) and zero for absent cloud.
    collection_number = accretion * cloud_number_per_cloud_mass(Nᶜˡ, ρ, qᶜˡ, p3.minimum_mass_mixing_ratio)
    number_loss = autoconversion_number + collection_number + self_collection +
                  riming_number + freezing_number + warm_collection_number
    return activation_number - number_loss
end

"""
$(TYPEDSIGNATURES)

Rain-drop number budget, ``∂n^r/∂t`` [1/kg/s], before homogeneous freezing.
Sources: autoconversion, melting ice, drop breakup, shedding, wet-growth shedding,
and — outside liquid-fraction mode — cloud water swept up by melting ice and shed
as drops. Sinks: evaporation, self-collection, riming, heterogeneous freezing, and
above-freezing collection by ice.
"""
@inline function rain_number_tendency_before_homogeneous_freezing(p3, autoconversion,
                                                                  melting_number,
                                                                  evaporation_number,
                                                                  self_collection, breakup,
                                                                  riming_number,
                                                                  freezing_number,
                                                                  shedding_number,
                                                                  cloud_warm_collection,
                                                                  warm_collection_number,
                                                                  wet_growth_shedding_number)
    FT = typeof(autoconversion)
    parameters = p3.process_rates
    number_from_autoconversion = autoconversion / rain_seed_drop_mass(p3)
    number_from_melting = melting_number
    cloud_warm_rain_number = ifelse(parameters.liquid_fraction_active, zero(FT),
                                    cloud_warm_collection / parameters.shed_drop_mass)
    number_gain = number_from_autoconversion + number_from_melting + breakup +
                  shedding_number + cloud_warm_rain_number + wet_growth_shedding_number
    number_loss = evaporation_number + self_collection + riming_number +
                  freezing_number + warm_collection_number
    return number_gain - number_loss
end

# Fall-speed correction for ambient air density, `(ρ₀ / ρ)^α`. The default exponent α is
# the [Heymsfield et al. (2007)](@cite HeymsfieldEtAl2007) fit and the default density
# floor only bites above ~30 km, where there is no condensate to fall; both are settable
# on [`ProcessRate`](@ref).
@inline function ice_air_density_correction(parameters, reference_air_density, air_density)
    FT = typeof(reference_air_density)
    ρ_floor = FT(parameters.minimum_fall_speed_air_density)
    α = FT(parameters.fall_speed_density_correction_exponent)
    return (reference_air_density / max(air_density, ρ_floor))^α
end

"""
$(TYPEDSIGNATURES)

Apply a bulk rime-density consistency pass to the prognostic rime state.
Returns corrected `qᶠ`, `bᶠ`, rime fraction `Fᶠ`, and rime density `ρᶠ`.

The rime-volume threshold is `minimum_mass_mixing_ratio / maximum_rime_density`,
so that it scales with the scheme's mass floor rather than being a fixed literal.

`qⁱ` is the dry ice mass, so it is already the bound on rime; there is no `qʷⁱ` argument,
unlike the reference implementation, which passes total ice and subtracts it here.
"""
@inline function consistent_rime_state(p3, qⁱ, qᶠ, bᶠ)
    FT = typeof(qⁱ)
    parameters = p3.process_rates

    qⁱ_dry = max(0, qⁱ)
    qᶠ_raw = max(0, qᶠ)
    bᶠ_raw = max(0, bᶠ)

    # Rime volume floor [m³/kg]: the volume a just-significant rime mass occupies at
    # the densest admissible packing. Below it, zero the pair rather than divide by a
    # vanishing bᶠ. `rime_not_small` below is the mass counterpart.
    has_rime_volume = bᶠ_raw >= p3.minimum_mass_mixing_ratio / parameters.maximum_rime_density
    ρᶠ_raw = safe_divide(qᶠ_raw, bᶠ_raw, zero(FT))
    ρᶠ_bounded = clamp(ρᶠ_raw, parameters.minimum_rime_density, parameters.maximum_rime_density)

    qᶠ_after_volume = ifelse(has_rime_volume, qᶠ_raw, zero(FT))
    bᶠ_after_volume = ifelse(has_rime_volume,
                             safe_divide(qᶠ_after_volume, ρᶠ_bounded, zero(FT)),
                             zero(FT))
    ρᶠ = ifelse(has_rime_volume, ρᶠ_bounded, zero(FT))

    rime_not_small = qᶠ_after_volume >= p3.minimum_mass_mixing_ratio
    qᶠ_after_small = ifelse(rime_not_small, qᶠ_after_volume, zero(FT))
    bᶠ_after_small = ifelse(rime_not_small, bᶠ_after_volume, zero(FT))

    # bound rime mass by dry ice mass, not total ice mass
    exceeds_dry_ice = (qᶠ_after_small > qⁱ_dry) & (ρᶠ > zero(FT))
    qᶠ_consistent = ifelse(exceeds_dry_ice, qⁱ_dry, qᶠ_after_small)
    bᶠ_consistent = ifelse(exceeds_dry_ice,
                           safe_divide(qᶠ_consistent, ρᶠ, zero(FT)),
                           bᶠ_after_small)
    Fᶠ = safe_divide(qᶠ_consistent, qⁱ_dry, zero(FT))

    return (; qᶠ = qᶠ_consistent, bᶠ = bᶠ_consistent, Fᶠ, ρᶠ)
end

@inline total_ice_mass(qⁱ, qʷⁱ) = max(0, qⁱ) + max(0, qʷⁱ)

@inline function liquid_fraction_on_ice(qⁱ, qʷⁱ, floors)
    FT = typeof(qⁱ)
    qⁱ_total = max(total_ice_mass(qⁱ, qʷⁱ), FT(floors.mass_scale))
    return max(0, qʷⁱ) / qⁱ_total
end

@inline active_liquid_on_ice(p3, qʷⁱ) = ifelse(p3.process_rates.liquid_fraction_active, qʷⁱ, zero(qʷⁱ))

@inline function mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ, floors)
    FT = typeof(qⁱ)
    return safe_divide(max(total_ice_mass(qⁱ, qʷⁱ), FT(floors.mass_scale)),
                       max(nⁱ, FT(floors.number_scale)),
                       FT(floors.mass_scale))
end

# Table-1 bracket of the *diagnostic* population (total ice mass, pre-limiter number),
# shared by the λ-limiter and the mean-density read. `qⁱ_total` already includes the coating.
@inline diagnostic_ice_bracket(limiter::IceLambdaLimiter, qⁱ_total, nⁱ, Fᶠ, Fˡ, ρᶠ, floors) =
    ice_table_bracket(limiter.large_q, mean_total_ice_mass(qⁱ_total, zero(qⁱ_total), nⁱ, floors),
                      Fᶠ, Fˡ, ρᶠ, floors)

@inline bounded_ice_number(limiter::IceLambdaLimiter, qⁱ_total, nⁱ, Fᶠ, Fˡ, ρᶠ, floors) =
    bounded_ice_number(limiter, diagnostic_ice_bracket(limiter, qⁱ_total, nⁱ, Fᶠ, Fˡ, ρᶠ, floors),
                       qⁱ_total, nⁱ, floors)

@inline function bounded_ice_number(limiter::IceLambdaLimiter, prep, qⁱ_total, nⁱ, floors)
    FT = typeof(qⁱ_total)
    qⁱ_eff = max(0, qⁱ_total)
    nⁱ_eff = max(0, nⁱ)
    nⁱ_min = evaluate_at(limiter.large_q, prep) * qⁱ_eff
    nⁱ_max = evaluate_at(limiter.small_q, prep) * qⁱ_eff
    bounded = clamp(nⁱ_eff, nⁱ_min, nⁱ_max)
    return ifelse(qⁱ_eff > FT(floors.mass_scale), bounded, zero(FT))
end

# Un-materialized scheme (no tables, `prep === nothing`): the only bound is positivity.
@inline bounded_ice_number(::IceLambdaLimiter{Nothing, Nothing}, prep, qⁱ_total, nⁱ, floors) =
    ifelse(qⁱ_total > zero(qⁱ_total), max(0, nⁱ), zero(nⁱ))

@inline function bounded_ice_number(p3, qⁱ_total, nⁱ, Fᶠ, Fˡ, ρᶠ)
    return bounded_ice_number(p3.ice.lambda_limiter, qⁱ_total, nⁱ, Fᶠ, Fˡ, ρᶠ,
                              p3.process_rates.floors)
end

# Exponential rain PSD: λʳ = (π ρʷ nʳ / qʳ)^(1/3). `bounded_rain_number` needs the
# unclamped value to tell whether the clamp fired.
@inline function unbounded_rain_slope_parameter(qʳ, nʳ, parameters)
    FT = typeof(qʳ)
    qʳ_eff = max(0, qʳ)
    nʳ_eff = max(0, nʳ)
    λʳ_cubed = FT(π) * parameters.liquid_water_density * nʳ_eff /
               max(qʳ_eff, FT(parameters.floors.mass_scale))
    return cbrt(λʳ_cubed)
end

"""
$(TYPEDSIGNATURES)

Return the exponential rain particle size distribution slope parameter ``λʳ``
diagnosed from the rain mass concentration `qʳ` and number concentration `nʳ`.
The result is clamped between `parameters.minimum_rain_slope` and
`parameters.maximum_rain_slope`.
"""
@inline rain_slope_parameter(qʳ, nʳ, parameters) =
    clamp(unbounded_rain_slope_parameter(qʳ, nʳ, parameters),
          parameters.minimum_rain_slope, parameters.maximum_rain_slope)

@inline function rain_number_from_slope(qʳ, λʳ, parameters)
    FT = typeof(qʳ)
    qʳ_eff = max(0, qʳ)
    return qʳ_eff * λʳ^3 / (FT(π) * parameters.liquid_water_density)
end

@inline function bounded_rain_number(nʳ, qʳ, parameters)
    qʳ_eff = max(0, qʳ)
    nʳ_eff = max(0, nʳ)
    unbounded_slope = unbounded_rain_slope_parameter(qʳ_eff, nʳ_eff, parameters)
    λʳ = clamp(unbounded_slope, parameters.minimum_rain_slope, parameters.maximum_rain_slope)
    nʳ_bounded = rain_number_from_slope(qʳ_eff, λʳ, parameters)
    needs_adjustment = (unbounded_slope < parameters.minimum_rain_slope) |
                       (unbounded_slope > parameters.maximum_rain_slope)
    return ifelse(needs_adjustment, nʳ_bounded, nʳ_eff)
end

# Bulk ice density from Table 1 at the diagnostic-population bracket.
@inline ice_mean_density(bulk::IceBulk, prep::PreparedInterpolation) =
    evaluate_at(bulk.mean_density, prep)

#####
##### Ice shape parameter (μⁱ) from Table 1
#####

"""
$(TYPEDSIGNATURES)

Compute the ice PSD shape parameter μⁱ from the lookup tables.

μⁱ is looked up directly from Table 1 (`bulk.shape`), which stores the
shape parameter computed when the table was generated.
"""
@inline function compute_ice_shape_parameter(p3, qⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ)
    FT = typeof(qⁱ)
    m̄ = safe_divide(qⁱ, nⁱ, one(FT))
    log_m = log10(ifelse(m̄ > 0, m̄, one(FT)))
    return p3.ice.bulk.shape(log_m, Fᶠ, Fˡ, ρᶠ)
end

#####
##### Thermodynamic latent heat helpers
#####
##### Use T-dependent latent heats for energy-budget consistency with the
##### condensation path.
#####

@inline sublimation_latent_heat(constants, T) = ice_latent_heat(T, constants)

@inline vaporization_latent_heat(constants, T) = liquid_latent_heat(T, constants)

@inline p3_dry_air_heat_capacity(constants, FT) = FT(constants.dry_air.heat_capacity)

@inline p3_gravitational_acceleration(constants, FT) = FT(constants.gravitational_acceleration)

@inline fusion_latent_heat(constants, T) = sublimation_latent_heat(constants, T) - vaporization_latent_heat(constants, T)

#####
##### Psychrometric corrections ξˡ, ξⁱ
#####
##### Account for the latent-heat feedback that reduces the effective
##### supersaturation drive during condensation (ξˡ) and ice deposition (ξⁱ).
##### `psychrometric_correction` itself is not P3-specific and lives beside its
##### mixture-heat-capacity counterpart `thermodynamic_adjustment_factor` in
##### `Microphysics/bulk_microphysics.jl`; only the two phase-named wrappers below,
##### which fix the dry-air heat capacity P3 uses, stay here.
#####

# Named for the phase each caller drives, so call sites still read as ξˡ / ξⁱ.
@inline liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T) =
    psychrometric_correction(ℒˡ, qᵛ⁺ˡ, p3_dry_air_heat_capacity(constants, typeof(T)), Rᵛ, T)

@inline ice_psychrometric_correction(constants, ℒⁱ, qᵛ⁺ⁱ, Rᵛ, T) =
    psychrometric_correction(ℒⁱ, qᵛ⁺ⁱ, p3_dry_air_heat_capacity(constants, typeof(T)), Rᵛ, T)

#####
##### Saturation vapor pressure at freezing (M6)
#####
##### Derive e_s(T₀) from the Clausius-Clapeyron or Tetens formula.
#####

@inline function saturation_vapor_pressure_at_freezing(constants, T₀)
    return saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface())
end

# Saturation vapor mass fraction at the melting point T₀. Breeze's qᵛ is a
# total-air mass fraction (ρᵛ/ρ), so this must use the same basis:
# q_sat0 = ρᵛ⁺(T₀)/ρ = e_s0 / (Rᵛ T₀ ρ). With this convention the diffusion term
# ℒ Dᵥ ρ (qᵛ - q_sat0) reduces to the exact vapor-density difference ρᵛ - ρᵛ⁺(T₀).
# A dry-air mixing ratio ε e_s0/(P - e_s0) would only be correct against a vapor
# variable that is itself a dry-air mixing ratio; mixing the two mass bases would
# bias the melting and refreezing heat balances, so all three call sites share this.
@inline function freezing_point_saturation_mass_fraction(constants, T₀, ρ)
    Rᵛ = typeof(ρ)(vapor_gas_constant(constants))
    return saturation_vapor_pressure_at_freezing(constants, T₀) / (Rᵛ * T₀ * ρ)
end
