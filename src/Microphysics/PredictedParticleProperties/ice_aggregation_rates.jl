@inline function ice_rain_collection_lookup(table::P3RainIceCollectionTable, m̄, λr, Fᶠ, Fˡ, ρᶠ, μ = zero(typeof(m̄)))
    log_m = log10(m̄)
    log_λ = log10(λr)
    # Both rain-ice tables share `(log_m, log_λ, Fᶠ, Fˡ, ρᶠ, μ)` axes
    # by construction, so prep indices once and reuse across evaluations.
    prep = prepare_6d(table.mass, log_m, log_λ, Fᶠ, Fˡ, ρᶠ, μ)
    # Fortran table stores rain-ice mass and number kernels as log10;
    # exponentiate to recover physical values (Fortran runtime: 10.**proc).
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
K(D_1, D_2) = E_{ii} × \\frac{π}{4}(D_1 + D_2)^2 × |V_1 - V_2|
```

The number tendency is:

```math
\\frac{dn^i}{dt} = -\\frac{ρ}{2} ∫∫ K(D_1, D_2) N'(D_1) N'(D_2) dD_1 dD_2
```

The ρ factor converts the volumetric collision kernel [m³/s] to the
mass-specific number tendency [1/kg/s] when nⁱ is in [1/kg].

The sticking efficiency E_ii increases with temperature (more sticky near 0°C).
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
function ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ, qʷⁱ = zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates

    Eᵢᵢ_max = prp.aggregation_efficiency_max
    T_low = prp.aggregation_efficiency_temperature_low
    T_high = prp.aggregation_efficiency_temperature_high

    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = max(clamp_positive(nⁱ), p3.minimum_number_mixing_ratio)

    # Fortran gates aggregation on bulk ice mass only. It floors the active
    # category's number to nsmall before evaluating the collection kernel.
    aggregation_active = qⁱ_total >= p3.minimum_mass_mixing_ratio

    # Temperature-dependent sticking efficiency (linear ramp)
    # Cold ice is less sticky, near-melting ice is very sticky
    Eᵢᵢ_cold = prp.aggregation_efficiency_min
    Eᵢᵢ = ifelse(T < T_low, Eᵢᵢ_cold,
                  ifelse(T > T_high, Eᵢᵢ_max,
                         Eᵢᵢ_cold + (T - T_low) / (T_high - T_low) * (Eᵢᵢ_max - Eᵢᵢ_cold)))

    # Rime-fraction limiter (Eii_fact): shut off aggregation for heavily rimed ice
    # Fortran P3: Eii_fact = 1 for Fr<0.6, linear ramp to 0 for 0.6≤Fr<0.9, 0 for Fr≥0.9
    Fᶠ_low = prp.aggregation_rime_fraction_low
    Fᶠ_high = prp.aggregation_rime_fraction_high
    Eᵢᵢ_fact = ifelse(Fᶠ < Fᶠ_low, FT(1),
                       ifelse(Fᶠ > Fᶠ_high, FT(0),
                              FT(1) - (Fᶠ - Fᶠ_low) / (Fᶠ_high - Fᶠ_low)))
    Eᵢᵢ = Eᵢᵢ * Eᵢᵢ_fact

    # Mean particle properties
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    # PSD-integrated self-collection kernel (E-free) from lookup table.
    AV_kernel = aggregation_kernel(p3.ice.collection.aggregation,
                                     m_mean, Fᶠ, Fˡ, ρᶠ, μ)

    # Collection kernel with temperature-dependent sticking efficiency
    K_mean = Eᵢᵢ * AV_kernel

    # Number loss rate: ρ × K × n² × rhofaci (positive magnitude)
    # The ρ factor converts the volumetric kernel [m³/s] to mass-specific
    # tendency [1/kg/s]. The 1/2 self-collection factor is already included
    # in the kernel (table stores half-integral, analytical path includes 0.5 factor).
    # Sign convention (M7): returns positive; caller subtracts in tendency assembly.
    # Use ice reference density (Fortran rhosui, P=600 hPa, T=-20°C), not rain reference.
    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = ice_air_density_correction(ρ₀, ρ)
    rate = ρ * K_mean * nⁱ_eff^2 * rhofaci

    return ifelse(aggregation_active, rate, zero(FT))
end
