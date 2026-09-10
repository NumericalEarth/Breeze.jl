@inline function ice_rain_collection_lookup(table::IceRainCollection,
                                            m̄, λr, Fᶠ, Fˡ, ρᶠ)
    log_m = log10(m̄)
    log_λ = log10(λr)
    # Both rain-ice tables share `(log_m, log_λ, Fᶠ, Fˡ, ρᶠ)` axes
    # by construction, so prep indices once and reuse across evaluations.
    prep = prepare_interpolation(table.mass, log_m, log_λ, Fᶠ, Fˡ, ρᶠ)
    # The table stores rain-ice mass and number kernels as log10;
    # exponentiate to recover physical values.
    return exp10(evaluate_at(table.mass, prep)),
           exp10(evaluate_at(table.number, prep))
end

#####
##### Phase 2: Ice aggregation
#####

"""
$(TYPEDSIGNATURES)

Compute ice self-collection (aggregation) rate using proper collision kernel.

Ice particles collide and stick together, reducing number concentration
without changing total mass. The collision kernel is:

```math
K(D_1, D_2) = E^{ii} × \\frac{π}{4}(D_1 + D_2)^2 × |V_1 - V_2|
```

The number tendency is:

```math
\\frac{dn^i}{dt} = -\\frac{ρ}{2} ∫∫ K(D_1, D_2) N'(D_1) N'(D_2) dD_1 dD_2
```

The ρ factor converts the volumetric collision kernel [m³/s] to the
mass-specific number tendency [1/kg/s] when nⁱ is in [1/kg].

The sticking efficiency ``E^{ii}`` increases with temperature (more sticky near 0°C).
See [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Rate of ice number loss [1/kg/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline function ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, qʷⁱ = zero(typeof(qⁱ)),
                                      lookups = p3_ice_lookups(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, ρ))
    FT = typeof(qⁱ)
    parameters = p3.process_rates

    maximum_efficiency = parameters.maximum_aggregation_efficiency
    ramp_start_temperature = parameters.aggregation_efficiency_ramp_start_temperature
    ramp_end_temperature = parameters.aggregation_efficiency_ramp_end_temperature

    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    nⁱ_eff = max(nⁱ, p3.minimum_number_mixing_ratio)

    # Aggregation is gated on bulk ice mass only, with the number floored at
    # `minimum_number_mixing_ratio` before the collection kernel is evaluated.
    aggregation_active = qⁱ_total >= p3.minimum_mass_mixing_ratio

    # Temperature-dependent sticking efficiency (linear ramp)
    # Cold ice is less sticky, near-melting ice is very sticky
    minimum_efficiency = parameters.minimum_aggregation_efficiency
    aggregation_efficiency = clamp(minimum_efficiency +
                                   (T - ramp_start_temperature) /
                                   (ramp_end_temperature - ramp_start_temperature) *
                                   (maximum_efficiency - minimum_efficiency),
                                   minimum_efficiency, maximum_efficiency)

    # Rime-fraction limiter (Eii_fact): shut off aggregation for heavily rimed ice
    # Eii_fact = 1 for Fᶠ<0.6, a linear ramp to 0 for 0.6≤Fᶠ<0.9, and 0 for Fᶠ≥0.9
    minimum_rime_fraction = parameters.minimum_aggregation_rime_fraction
    maximum_rime_fraction = parameters.maximum_aggregation_rime_fraction
    rime_fraction_factor = clamp(1 - (Fᶠ - minimum_rime_fraction) /
                                     (maximum_rime_fraction - minimum_rime_fraction), 0, 1)
    aggregation_efficiency *= rime_fraction_factor

    # PSD-integrated self-collection kernel (E-free) from lookup table.
    aggregation_kernel_value = evaluate_at(p3.ice.collection.aggregation, lookups.prep)

    # Collection kernel with temperature-dependent sticking efficiency
    mean_collection_kernel = aggregation_efficiency * aggregation_kernel_value

    # Number loss rate: ρ × K × n² × rhofaci (positive magnitude)
    # The ρ factor converts the volumetric kernel [m³/s] to mass-specific
    # tendency [1/kg/s]. The 1/2 self-collection factor is already included
    # in the kernel (table stores half-integral, analytical path includes 0.5 factor).
    # Sign convention (M7): returns positive; caller subtracts in tendency assembly.
    # The density correction uses the ice reference density (P=600 hPa, T=-20°C), not
    # the rain reference; see `p3_ice_lookups`.
    rate = ρ * mean_collection_kernel * nⁱ_eff^2 * lookups.ρ_correction

    return ifelse(aggregation_active, rate, zero(FT))
end
