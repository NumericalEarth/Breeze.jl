#####
##### Phase 2: Ice aggregation
#####

"""
    ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ)

Compute ice self-collection (aggregation) rate using proper collision kernel.

Ice particles collide and stick together, reducing number concentration
without changing total mass. The collision kernel is:

```math
K(D_1, D_2) = E_{ii} × \\frac{π}{4}(D_1 + D_2)^2 × |V_1 - V_2|
```

The number tendency is:

```math
\\frac{dn^i}{dt} = -\\frac{1}{2} ∫∫ K(D_1, D_2) N'(D_1) N'(D_2) dD_1 dD_2
```

The sticking efficiency E_ii increases with temperature (more sticky near 0°C).
See [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]

# Returns
- Rate of ice number reduction [1/kg/s]
"""
@inline function ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    Eᵢᵢ_max = prp.aggregation_efficiency_max
    T_low = prp.aggregation_efficiency_temperature_low
    T_high = prp.aggregation_efficiency_temperature_high

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # Thresholds
    qⁱ_threshold = FT(1e-8)
    nⁱ_threshold = FT(1e2)

    aggregation_active = (qⁱ_eff > qⁱ_threshold) & (nⁱ_eff > nⁱ_threshold)

    # Temperature-dependent sticking efficiency (linear ramp)
    # Cold ice is less sticky, near-melting ice is very sticky
    Eᵢᵢ_cold = FT(0.001)
    Eᵢᵢ = ifelse(T < T_low, Eᵢᵢ_cold,
                  ifelse(T > T_high, Eᵢᵢ_max,
                         Eᵢᵢ_cold + (T - T_low) / (T_high - T_low) * (Eᵢᵢ_max - Eᵢᵢ_cold)))

    # Rime-fraction limiter (Eii_fact): shut off aggregation for heavily rimed ice
    # Fortran P3: Eii_fact = 1 for Fr<0.6, linear ramp to 0 for 0.6≤Fr<0.9, 0 for Fr≥0.9
    Eᵢᵢ_fact = ifelse(Fᶠ < FT(0.6), FT(1),
                       ifelse(Fᶠ > FT(0.9), FT(0),
                              FT(1) - (Fᶠ - FT(0.6)) / FT(0.3)))
    Eᵢᵢ = Eᵢᵢ * Eᵢᵢ_fact

    # Mean particle properties
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # Self-collection kernel: dispatches to PSD-integrated table or
    # mean-mass path. Returns E-free kernel (A × ΔV per particle pair).
    AV_kernel = _aggregation_kernel(p3.ice.collection.aggregation,
                                     m_mean, Fᶠ, ρᶠ, prp)

    # Collection kernel with temperature-dependent sticking efficiency
    K_mean = Eᵢᵢ * AV_kernel

    # Number tendency: dn/dt = -0.5 × K × n²
    rate = -FT(0.5) * K_mean * nⁱ_eff^2

    return ifelse(aggregation_active, rate, zero(FT))
end

#####
##### Phase 2: Riming (cloud and rain collection by ice)
#####

"""
    cloud_riming_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)

Compute cloud droplet collection (riming) by ice particles using the
continuous collection equation with the collision kernel integrated
over the ice particle size distribution.

The collection rate is:
```math
\\frac{dq_c}{dt} = -E_{ci} q_c ρ n_i ⟨A V⟩
```
where ⟨A V⟩ is the PSD-averaged product of projected area and terminal
velocity, approximated using the mean-mass diameter with a correction
factor for the exponential PSD.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Rate of cloud → ice conversion [kg/kg/s] (also equals rime mass gain rate)
"""
@inline function cloud_riming_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    Eᶜⁱ = prp.cloud_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-8)
    n_threshold = FT(1)
    below_freezing = T < T₀
    active = below_freezing & (qᶜˡ_eff > q_threshold) & (qⁱ_eff > q_threshold) & (nⁱ_eff > n_threshold)

    # Mean particle mass
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # Collection kernel ⟨A×V⟩: dispatches to PSD-integrated table or
    # mean-mass path with psd_correction. The RainCollectionNumber integral
    # computes ∫ V(D) A(D) N'(D) dD with E=1, giving the geometric kernel.
    AV_per_particle = _collection_kernel_per_particle(p3.ice.collection.rain_collection,
                                                       m_mean, Fᶠ, ρᶠ, prp)

    # Air density correction: tables computed at reference conditions
    # (ρ₀ ≈ 0.826 kg/m³), then scaled by (ρ₀/ρ)^0.54.
    ρ₀ = prp.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Collection rate = E × qc × ni × ρ × rhofaci × ⟨A×V⟩
    rate = Eᶜⁱ * qᶜˡ_eff * nⁱ_eff * ρ * rhofaci * AV_per_particle

    return ifelse(active, rate, zero(FT))
end

"""
    cloud_riming_number_rate(qᶜˡ, Nᶜ, riming_rate)

Compute cloud droplet number sink from riming.

# Arguments
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜ`: Cloud droplet number concentration [1/m³]
- `riming_rate`: Cloud riming mass rate [kg/kg/s]

# Returns
- Rate of cloud number reduction [1/m³/s]
"""
@inline function cloud_riming_number_rate(qᶜˡ, Nᶜ, riming_rate)
    FT = typeof(qᶜˡ)

    ratio = safe_divide(Nᶜ, qᶜˡ, zero(FT))

    return -ratio * riming_rate
end

"""
    rain_riming_rate(p3, qʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)

Compute rain collection (riming) by ice particles using the continuous
collection equation with collision kernel integrated over the ice PSD.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Rate of rain → ice conversion [kg/kg/s] (also equals rime mass gain rate)
"""
@inline function rain_riming_rate(p3, qʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    Eʳⁱ = prp.rain_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qʳ_eff = clamp_positive(qʳ)
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-8)
    n_threshold = FT(1)
    below_freezing = T < T₀
    # Only ice collects rain when qi >= qr (Mizuno et al. 1990).
    # When qr > qi, rain collects ice — mass goes the other way (handled in driver).
    ice_dominant = qⁱ_eff >= qʳ_eff
    active = below_freezing & ice_dominant & (qʳ_eff > q_threshold) & (qⁱ_eff > q_threshold) & (nⁱ_eff > n_threshold)

    # Mean particle mass
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # Collection kernel ⟨A×V⟩: dispatches to PSD-integrated table or
    # mean-mass path with psd_correction (same kernel as cloud riming).
    AV_per_particle = _collection_kernel_per_particle(p3.ice.collection.rain_collection,
                                                       m_mean, Fᶠ, ρᶠ, prp)

    # Air density correction (same as cloud riming)
    ρ₀ = prp.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Collection rate = E × qr × ni × ρ × rhofaci × ⟨A×V⟩
    rate = Eʳⁱ * qʳ_eff * nⁱ_eff * ρ * rhofaci * AV_per_particle

    return ifelse(active, rate, zero(FT))
end

"""
    rain_riming_number_rate(qʳ, nʳ, riming_rate)

Compute rain number sink from riming.

# Arguments
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `riming_rate`: Rain riming mass rate [kg/kg/s]

# Returns
- Rate of rain number reduction [1/kg/s]
"""
@inline function rain_riming_number_rate(qʳ, nʳ, riming_rate)
    FT = typeof(qʳ)

    ratio = safe_divide(nʳ, qʳ, zero(FT))

    return -ratio * riming_rate
end

"""
    rime_density_cober_list(p3, T, vᵢ, D_drop, D_ice, lwc)

Compute rime density using the full Cober & List (1993) parameterization.

The rime density depends on the impact conditions:

```math
ρ_f = ρ_0 × exp(a × K^b)
```

where K is a dimensionless impact parameter that depends on:
- Impact velocity (v_i)
- Cloud droplet diameter (D_drop)
- Surface temperature

For wet growth conditions (T > -3°C, high LWC), rime density approaches
the density of liquid water (soaking).

# Arguments
- `p3`: P3 microphysics scheme
- `T`: Temperature [K]
- `vᵢ`: Ice particle fall speed [m/s]
- `D_drop`: Median cloud droplet diameter [m] (default 20 μm)
- `D_ice`: Ice particle diameter [m] (for Reynolds number)
- `lwc`: Liquid water content [kg/m³] (for wet growth check)

# Returns
- Rime density [kg/m³]

# References
[Cober and List (1993)](@cite CoberList1993)
"""
@inline function rime_density_cober_list(p3, T, vᵢ, D_drop, D_ice, lwc)
    FT = typeof(T)
    prp = p3.process_rates

    ρ_rim_min = prp.minimum_rime_density
    ρ_rim_max = prp.maximum_rime_density
    T₀ = prp.freezing_temperature
    ρ_water = p3.water_density

    # Temperature in Celsius
    Tc = T - T₀

    # Clamp temperature to supercooled range
    Tc_clamped = clamp(Tc, FT(-40), FT(0))

    # Impact velocity (approximately fall speed minus droplet fall speed)
    v_impact = max(vᵢ, FT(0.1))

    # Droplet Stokes number (St = ρ_w × D_drop² × v_impact / (18 × μ × D_ice))
    # Simplified: use dimensionless impact parameter K
    μ = FT(1.8e-5)  # Dynamic viscosity of air [Pa·s]
    K = ρ_water * D_drop^2 * v_impact / (18 * μ * max(D_ice, FT(1e-5)))

    # Cober & List (1993) empirical fit for dry growth regime
    # ρ_f = 110 + 290 × (1 - exp(-1.25 × K^0.75))
    # This asymptotes to ~400 kg/m³ for high K (dense rime/graupel)
    # and to ~110 kg/m³ for low K (fluffy rime)
    K_clamped = clamp(K, FT(0.01), FT(100))
    ρ_dry = FT(110) + FT(290) * (1 - exp(-FT(1.25) * K_clamped^FT(0.75)))

    # Temperature correction: slightly denser rime near 0°C
    T_factor = 1 + FT(0.1) * (Tc_clamped + FT(40)) / FT(40)
    ρ_dry = ρ_dry * T_factor

    # Wet growth regime: when T > -10°C and high LWC
    # Rime density approaches water density (spongy graupel)
    is_wet_growth = (Tc > FT(-10)) & (lwc > FT(0.5e-3))
    wet_fraction = clamp((Tc + FT(10)) / FT(10), zero(FT), one(FT))
    ρ_wet = ρ_dry * (1 - wet_fraction) + ρ_water * FT(0.8) * wet_fraction

    ρᶠ = ifelse(is_wet_growth, ρ_wet, ρ_dry)

    return clamp(ρᶠ, ρ_rim_min, ρ_rim_max)
end

# Simplified version for backward compatibility
@inline function rime_density(p3, T, vᵢ)
    FT = typeof(T)
    prp = p3.process_rates

    ρ_rim_min = prp.minimum_rime_density
    ρ_rim_max = prp.maximum_rime_density
    T₀ = prp.freezing_temperature

    # Default droplet and ice properties
    D_drop = FT(20e-6)  # 20 μm cloud droplets
    D_ice = FT(1e-3)    # 1 mm ice particle
    lwc = FT(0.3e-3)    # 0.3 g/m³ typical LWC

    return rime_density_cober_list(p3, T, vᵢ, D_drop, D_ice, lwc)
end

#####
##### Phase 2: Shedding and Refreezing (liquid fraction dynamics)
#####

"""
    shedding_rate(p3, qʷⁱ, qⁱ, T)

Compute liquid shedding rate from ice particles.

When ice particles carry too much liquid coating (from partial melting
or warm riming), excess liquid is shed as rain drops.
See [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `T`: Temperature [K]

# Returns
- Rate of liquid → rain shedding [kg/kg/s]
"""
@inline function shedding_rate(p3, qʷⁱ, qⁱ, T)
    FT = typeof(qʷⁱ)
    prp = p3.process_rates

    τ_shed = prp.shedding_timescale
    qʷⁱ_max_frac = prp.maximum_liquid_fraction
    T₀ = prp.freezing_temperature

    qʷⁱ_eff = clamp_positive(qʷⁱ)
    qⁱ_eff = clamp_positive(qⁱ)

    # Total particle mass
    qᵗᵒᵗ = qⁱ_eff + qʷⁱ_eff

    # Maximum liquid that can be retained
    qʷⁱ_max = qʷⁱ_max_frac * qᵗᵒᵗ

    # Excess liquid sheds
    qʷⁱ_excess = clamp_positive(qʷⁱ_eff - qʷⁱ_max)

    # Enhanced shedding above freezing
    T_factor = ifelse(T > T₀, FT(3), FT(1))

    return T_factor * qʷⁱ_excess / τ_shed
end

"""
    shedding_number_rate(p3, shed_rate)

Compute rain number source from shedding.

Shed liquid forms rain drops of approximately 1 mm diameter.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `shed_rate`: Liquid shedding mass rate [kg/kg/s]

# Returns
- Rate of rain number increase [1/kg/s]
"""
@inline function shedding_number_rate(p3, shed_rate)
    m_shed = p3.process_rates.shed_drop_mass

    return shed_rate / m_shed
end

"""
    refreezing_rate(p3, qʷⁱ, T)

Compute refreezing rate of liquid on ice particles.

Below freezing, liquid coating on ice particles refreezes,
transferring mass from liquid-on-ice to ice+rime.
See [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `T`: Temperature [K]

# Returns
- Rate of liquid → ice refreezing [kg/kg/s]
"""
@inline function refreezing_rate(p3, qʷⁱ, T)
    FT = typeof(qʷⁱ)
    prp = p3.process_rates

    τ_frz = prp.refreezing_timescale
    T₀ = prp.freezing_temperature

    qʷⁱ_eff = clamp_positive(qʷⁱ)

    # Only refreeze below freezing
    below_freezing = T < T₀

    # Faster refreezing at colder temperatures
    ΔT = clamp_positive(T₀ - T)
    T_factor = FT(1) + FT(0.1) * ΔT

    rate = ifelse(below_freezing & (qʷⁱ_eff > FT(1e-10)),
                   T_factor * qʷⁱ_eff / τ_frz,
                   zero(FT))

    return rate
end
