@inline function ice_rain_collection_lookup(table::P3LookupTable2, m̄, λr, Fᶠ, Fˡ, ρᶠ)
    return table.mass(log10(m̄), log10(λr), Fᶠ, Fˡ, ρᶠ),
           table.number(log10(m̄), log10(λr), Fᶠ, Fˡ, ρᶠ),
           table.sixth_moment(log10(m̄), log10(λr), Fᶠ, Fˡ, ρᶠ)
end

#####
##### Phase 2: Ice aggregation
#####

"""
    ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)

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
@inline function ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
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
    AV_kernel = aggregation_kernel(p3.ice.collection.aggregation,
                                     m_mean, Fᶠ, ρᶠ, prp, p3)

    # Collection kernel with temperature-dependent sticking efficiency
    K_mean = Eᵢᵢ * AV_kernel

    # Number loss rate: ρ × K × n² (positive magnitude)
    # The ρ factor converts the volumetric kernel [m³/s] to mass-specific
    # tendency [1/kg/s]. The 1/2 self-collection factor is already included
    # in the kernel (table stores half-integral, analytical path includes 0.5 factor).
    # Sign convention (M7): returns positive; caller subtracts in tendency assembly.
    rate = ρ * K_mean * nⁱ_eff^2

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
    AV_per_particle = collection_kernel_per_particle(p3.ice.collection.rain_collection,
                                                       m_mean, Fᶠ, ρᶠ, prp, p3)

    # Air density correction for ice particle fall speed (Heymsfield et al. 2006):
    # ρfaci = (ρ₀_ice / ρ)^0.54, where ρ₀_ice = 60000/(287.15×253.15) ≈ 0.826 kg/m³
    # (Fortran P3: rhosui — NOT the surface/rain reference density rhosur ≈ 1.275).
    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Collection rate = E × qc × ni × ρ × rhofaci × ⟨A×V⟩
    rate = Eᶜⁱ * qᶜˡ_eff * nⁱ_eff * ρ * rhofaci * AV_per_particle

    return ifelse(active, rate, zero(FT))
end

"""
    cloud_warm_collection_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)

Compute above-freezing cloud collection by melting ice (Fortran qcshd/ncshdc pathway).

When `T > T₀`, ice particles still sweep up cloud droplets via the same collection
kernel as riming, but the collected water is immediately shed as rain drops (not frozen).
The number of new rain drops assumes 1mm shed drops (Fortran: ncshdc = qcshd × 1.923e6).

# Returns
- `(mass_rate, number_rate)`: Cloud → rain mass rate [kg/kg/s] and rain number source [1/kg/s]
"""
@inline function cloud_warm_collection_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    Eᶜⁱ = prp.cloud_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-8)
    n_threshold = FT(1)
    above_freezing = T >= T₀
    active = above_freezing & (qᶜˡ_eff > q_threshold) & (qⁱ_eff > q_threshold) & (nⁱ_eff > n_threshold)

    # Same collection kernel as cloud_riming_rate
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))
    AV_per_particle = collection_kernel_per_particle(p3.ice.collection.rain_collection,
                                                       m_mean, Fᶠ, ρᶠ, prp, p3)
    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    mass_rate = Eᶜⁱ * qᶜˡ_eff * nⁱ_eff * ρ * rhofaci * AV_per_particle
    # Fortran: ncshdc = qcshd * 1.923e6 (shed as 1mm drops: m = π/6 × 1000 × 0.001³ ≈ 5.2e-7 kg)
    number_rate = mass_rate * FT(1.923e6)

    return (ifelse(active, mass_rate, zero(FT)),
            ifelse(active, number_rate, zero(FT)))
end

"""
    rain_warm_collection_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)

Compute above-freezing rain collection by melting ice (Fortran qrcoll pathway).

When `T > T₀` and liquid fraction is active, rain drops collected by ice
contribute to the liquid coating (qʷⁱ) rather than to rime.
Uses the same collection kernel as rain riming.
See [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

# Returns
- Rain mass rate collected onto ice [kg/kg/s]
"""
@inline function rain_warm_collection_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    Eʳⁱ = prp.rain_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-8)
    n_threshold = FT(1)
    above_freezing = T >= T₀
    active = above_freezing & (qʳ_eff > q_threshold) & (qⁱ_eff > q_threshold) & (nⁱ_eff > n_threshold)

    # Same collection kernel as rain_riming_rate
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))
    AV_per_particle = collection_kernel_per_particle(p3.ice.collection.rain_collection,
                                                       m_mean, Fᶠ, ρᶠ, prp, p3)
    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Rain-DSD cross-section correction (C5)
    λ_r_cubed = FT(π) * prp.liquid_water_density * nʳ_eff / max(qʳ_eff, FT(1e-15))
    λ_r = clamp(cbrt(λ_r_cubed), prp.rain_lambda_min, prp.rain_lambda_max)
    D_r_mean = 1 / λ_r
    ρ_eff = (1 - Fᶠ) * prp.ice_effective_density_unrimed + Fᶠ * max(ρᶠ, FT(50))
    D_i_mean = cbrt(6 * m_mean / (FT(π) * max(ρ_eff, FT(50))))
    D_i_mean = clamp(D_i_mean, prp.ice_diameter_min, prp.ice_diameter_max)
    r_ratio = D_r_mean / max(D_i_mean, FT(prp.ice_diameter_min))
    rain_dsd_correction = 1 + 8 * r_ratio + 20 * r_ratio^2
    rain_dsd_correction = ifelse(nʳ_eff > FT(1), rain_dsd_correction, one(FT))

    rate = Eʳⁱ * qʳ_eff * nⁱ_eff * ρ * rhofaci * AV_per_particle * rain_dsd_correction

    return ifelse(active, rate, zero(FT))
end

"""
    cloud_riming_number_rate(qᶜˡ, Nᶜ, riming_rate)

Compute cloud droplet number sink from riming.

Returns `(Nᶜ / qᶜˡ) * riming_rate`, which has units [1/m³/s] because Nᶜ
is in [1/m³] while qᶜˡ and riming_rate are in [kg/kg] and [kg/kg/s].
Note: this rate is currently computed but unused by the tendency kernel
(cloud droplet number is prescribed, not predicted, in the P3 scheme).

# Arguments
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜ`: Cloud droplet number concentration [1/m³]
- `riming_rate`: Cloud riming mass rate [kg/kg/s]

# Returns
- Rate of cloud number loss [1/m³/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline function cloud_riming_number_rate(qᶜˡ, Nᶜ, riming_rate)
    FT = typeof(qᶜˡ)

    ratio = safe_divide(Nᶜ, qᶜˡ, zero(FT))

    return ratio * riming_rate
end

"""
    rain_riming_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)

Compute rain collection (riming) by ice particles using the continuous
collection equation with collision kernel integrated over the ice PSD,
plus a correction for the rain drop size distribution (C5 fix).

**C5 correction (double-PSD integration):**

The Fortran P3 f1pr07/f1pr08 lookup entries integrate over *both* the ice PSD
and the rain PSD, capturing how rain drop size affects the collision geometry.
The geometric cross section is ``π/4 (D_i + D_r)^2``, not just ``π/4 D_i^2``.
For an exponential rain PSD (μ_r = 0) the exact cross-section correction to the
single-PSD ice-side integral is:

```math
C = 1 + 8 \\frac{D_r^{\\rm mean}}{D_i^{\\rm mean}} + 20 \\left(\\frac{D_r^{\\rm mean}}{D_i^{\\rm mean}}\\right)^2
```

where ``D_r^{\\rm mean} = 1/λ_r`` and ``D_i^{\\rm mean}`` is the mean ice diameter.
When ``n_r = 0`` the correction is 1 (no change from the legacy path).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]; use 0 to disable C5 correction
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Rate of rain → ice conversion [kg/kg/s] (also equals rime mass gain rate)
"""
@inline function rain_riming_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    Eʳⁱ = prp.rain_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-8)
    n_threshold = FT(1)
    below_freezing = T < T₀
    # Fortran P3 v5.5.0: rain-ice collection proceeds whenever both species
    # are present and T < T₀ (no qi >= qr condition). Removed Mizuno (1990) gate.
    active = below_freezing & (qʳ_eff > q_threshold) & (qⁱ_eff > q_threshold) & (nⁱ_eff > n_threshold)

    # Mean particle mass
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # Collection kernel ⟨A×V⟩: dispatches to PSD-integrated table or
    # mean-mass path with psd_correction (same kernel as cloud riming).
    AV_per_particle = collection_kernel_per_particle(p3.ice.collection.rain_collection,
                                                       m_mean, Fᶠ, ρᶠ, prp, p3)

    # Air density correction for ice particle fall speed (same convention as cloud riming):
    # uses ice reference density ρ₀_ice ≈ 0.826 kg/m³, NOT rain reference ≈ 1.275 kg/m³.
    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # C5: Rain-DSD cross-section correction for double-PSD integration.
    # The Fortran f1pr07/f1pr08 table integrates over BOTH ice and rain PSDs,
    # capturing the (D_i + D_r)² collision geometry. The single-PSD path above
    # only uses D_i². For exponential rain (μ_r = 0), the exact correction is:
    #   correction = 1 + 8*(D_r_mean/D_i_mean) + 20*(D_r_mean/D_i_mean)²
    # (derived from the ratio of double-PSD to single-PSD cross-section integrals;
    # see P3_FORTRAN_COMPARISON.md, issue C5).
    #
    # Rain mean diameter: λ_r³ = π ρ_w nʳ / qʳ  (exponential PSD, μ_r = 0)
    # D_r_mean = 1/λ_r (number-weighted mean diameter for μ_r = 0)
    λ_r_cubed = FT(π) * prp.liquid_water_density * nʳ_eff / max(qʳ_eff, FT(1e-15))
    λ_r = clamp(cbrt(λ_r_cubed), prp.rain_lambda_min, prp.rain_lambda_max)
    D_r_mean = 1 / λ_r

    # Ice mean diameter at mean mass (mean-mass approximation)
    ρ_eff = (1 - Fᶠ) * prp.ice_effective_density_unrimed + Fᶠ * max(ρᶠ, FT(50))
    D_i_mean = cbrt(6 * m_mean / (FT(π) * max(ρ_eff, FT(50))))
    D_i_mean = clamp(D_i_mean, prp.ice_diameter_min, prp.ice_diameter_max)

    r_ratio = D_r_mean / max(D_i_mean, FT(prp.ice_diameter_min))
    rain_dsd_correction = 1 + 8 * r_ratio + 20 * r_ratio^2

    # Only apply the rain-DSD correction when rain number is available (nʳ > 0).
    # When nʳ = 0 the correction is 1 (matches the legacy 8-argument overload behavior).
    rain_dsd_correction = ifelse(nʳ_eff > FT(1), rain_dsd_correction, one(FT))

    # Collection rate = E × qr × ni × ρ × rhofaci × ⟨A×V⟩ × rain_dsd_correction
    rate = Eʳⁱ * qʳ_eff * nⁱ_eff * ρ * rhofaci * AV_per_particle * rain_dsd_correction

    return ifelse(active, rate, zero(FT))
end

"""
    rain_riming_rate(p3, qʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)

Backward-compatible 8-argument overload of `rain_riming_rate` without rain DSD correction.
Passes `nʳ = 0`, which disables the C5 double-PSD cross-section correction.
Prefer the 9-argument form `rain_riming_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)`.
"""
@inline function rain_riming_rate(p3, qʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    FT = typeof(qʳ)
    return rain_riming_rate(p3, qʳ, zero(FT), qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
end

"""
    rain_riming_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)

Compute rain number loss from riming using the tabulated number-weighted
collection kernel (RainCollectionNumber / Fortran f1pr07).

Replaces the monodisperse approximation `(nʳ/qʳ) × mass_rate` with an
independent PSD-integrated number collection rate.

# Arguments
- `p3`: P3 microphysics scheme
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Rate of rain number loss [1/kg/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline function rain_riming_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    Eʳⁱ = prp.rain_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-8)
    n_threshold = FT(1)
    below_freezing = T < T₀
    active = below_freezing & (qʳ_eff > q_threshold) & (qⁱ_eff > q_threshold) & (nʳ_eff > n_threshold) & (nⁱ_eff > n_threshold)

    # Mean ice particle mass (same as in rain_riming_rate)
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # H6: Use number-weighted collection kernel from RainCollectionNumber table
    # (Fortran f1pr07), instead of monodisperse approximation (nʳ/qʳ × mass_rate).
    AV_number = collection_kernel_per_particle(p3.ice.collection.rain_collection,
                                                m_mean, Fᶠ, ρᶠ, prp, p3)

    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Number collection rate = E × nʳ × nⁱ × ρ × rhofaci × ⟨A×V⟩_number
    rate = Eʳⁱ * nʳ_eff * nⁱ_eff * ρ * rhofaci * AV_number

    return ifelse(active, rate, zero(FT))
end

"""
    rain_riming_number_rate(qʳ, nʳ, riming_rate)

Backward-compatible fallback: compute rain number loss from riming using the
monodisperse approximation `(nʳ/qʳ) × mass_rate`.
Prefer the 9-argument form `rain_riming_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)`.

# Arguments
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `riming_rate`: Rain riming mass rate [kg/kg/s]

# Returns
- Rate of rain number loss [1/kg/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline function rain_riming_number_rate(qʳ, nʳ, riming_rate)
    FT = typeof(qʳ)

    ratio = safe_divide(nʳ, qʳ, zero(FT))

    return ratio * riming_rate
end

"""
    rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport)

Compute the density of newly accreted cloud rime using the Fortran P3 Ri fit.

This follows the `p3_main` cloud-riming branch: diagnose the cloud gamma PSD
from `qᶜˡ` and prescribed `Nᶜ`, compute the droplet impact speed relative to
falling ice, form the rime-impact parameter `Ri`, and apply the same piecewise
fit for `ρ_rime`. When cloud riming is inactive or the air is above freezing,
the Fortran fallback value `400 kg m⁻³` is used.

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
@inline function rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport)
    FT = typeof(T)
    prp = p3.process_rates
    qsmall = p3.minimum_mass_mixing_ratio

    ρ_rim_min = prp.minimum_rime_density
    ρ_rim_max = prp.maximum_rime_density
    T₀ = prp.freezing_temperature
    ρ_water = prp.liquid_water_density
    μ_c = p3.cloud.shape_parameter
    Nᶜ = p3.cloud.number_concentration

    qᶜˡ_abs = clamp_positive(qᶜˡ) * ρ
    μ_air = transport.nu * ρ
    g = constants.gravitational_acceleration

    λ_c_uncapped = cbrt(
        FT(π) * ρ_water * Nᶜ * (μ_c + 3) * (μ_c + 2) * (μ_c + 1) /
        (FT(6) * max(qᶜˡ_abs, FT(1e-20)))
    )
    λ_c = clamp(λ_c_uncapped, (μ_c + 1) * FT(2.5e4), (μ_c + 1) * FT(1e6))

    # Fortran get_cloud_dsd2 / p3_main: bcn = 2 and Γ(μ+6)/Γ(μ+4) = (μ+5)(μ+4).
    a_cn = g * ρ_water / (FT(18) * max(μ_air, FT(1e-20)))
    Vt_qc = a_cn * (μ_c + 5) * (μ_c + 4) / λ_c^2
    D_c = (μ_c + 4) / λ_c
    inverse_supercooling = inv(min(FT(-0.001), T - T₀))
    Ri = clamp(-(FT(0.5e6) * D_c) * abs(vᵢ - Vt_qc) * inverse_supercooling, FT(1), FT(12))

    ρ_rime_Ri = ifelse(
        Ri <= FT(8),
        (FT(0.051) + FT(0.114) * Ri - FT(0.0055) * Ri^2) * FT(1000),
        FT(611) + FT(72.25) * (Ri - FT(8))
    )

    active_cloud_riming = (cloud_rim >= qsmall) & (qᶜˡ >= qsmall) & (T < T₀)
    ρᶠ = ifelse(active_cloud_riming, ρ_rime_Ri, FT(400))

    return clamp(ρᶠ, ρ_rim_min, ρ_rim_max)
end

#####
##### Phase 2: Shedding and Refreezing (liquid fraction dynamics)
#####

"""
    shedding_rate(p3, qʷⁱ, qⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ, m_mean)

Compute liquid shedding rate from ice particles following
[Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

PSD-integrated shedding of liquid from mixed-phase ice particles with D ≥ 9 mm
(Rasmussen et al. 2011). Matches Fortran P3 v5.5.0:

```math
q_{lshd} = F_r \\times f_{1pr28} \\times N_i \\times F_l
```

where `f1pr28 = ∫_{D≥9mm} m(D) N'(D) dD` (lookup table, Fl-blended mass),
`Fr = qirim / (qitot - qiliq)` is the rime fraction of ice-only mass, and
`Fl = qiliq / qitot` is the liquid fraction.

# Arguments
- `p3`: P3 microphysics scheme (provides shedding table)
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg] (dry ice, excluding qʷⁱ)
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime fraction (= qᶠ/qⁱ) [-]
- `Fˡ`: Liquid fraction (= qʷⁱ/(qⁱ+qʷⁱ)) [-]
- `ρᶠ`: Rime density [kg/m³]
- `m_mean`: Mean ice particle mass [kg]

# Returns
- Rate of liquid → rain shedding [kg/kg/s]
"""
@inline function shedding_rate(p3, qʷⁱ, qⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ, m_mean)
    FT = typeof(qʷⁱ)

    qʷⁱ_eff = clamp_positive(qʷⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # Lookup ∫_{D≥9mm} m(D) N'(D) dD (normalized per particle)
    f1pr28 = shedding_integral(p3.ice.bulk_properties.shedding, m_mean, Fᶠ, Fˡ, ρᶠ)

    # Fortran: qlshd = Fr × f1pr28 × ni × Fl
    # Fr = rime fraction of ice-only mass (= Fᶠ in Julia convention since qⁱ excludes qʷⁱ)
    rate = Fᶠ * f1pr28 * nⁱ_eff * Fˡ

    # Bound by available liquid: qlshd ≤ qwi / dt_safety
    rate = clamp_positive(rate)
    τ_safety = FT(1)  # [s]
    rate = min(rate, qʷⁱ_eff / τ_safety)

    return rate
end

"""
    shedding_integral(table, m_mean, Fᶠ, Fˡ, ρᶠ)

Lookup the PSD-integrated shedding mass for D ≥ 9 mm particles.
Dispatches on table type (TabulatedFunction4D or analytical fallback).
"""
@inline function shedding_integral(table::TabulatedFunction4D, m_mean, Fᶠ, Fˡ, ρᶠ)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, FT(1e-20)))
    return table(log_m, Fᶠ, Fˡ, ρᶠ)
end

# Analytical fallback: zero shedding when table is not available
@inline function shedding_integral(::Any, m_mean, Fᶠ, Fˡ, ρᶠ)
    return zero(m_mean)
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
    wet_growth_capacity(p3, qⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport)

Compute the wet growth freezing capacity following
[Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

The wet growth capacity is the maximum rate at which collected
hydrometeors can be frozen, determined by the ventilated heat balance:

```math
q_{wgrth} = C f_v \\left[K_a(T_0-T) + \\frac{2π}{L_f} L_s D_v(ρ_{vs}-ρ_v)\\right] × N_i
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
- `constants`: Thermodynamic constants (or `nothing`)
- `transport`: Pre-computed air transport properties `(; D_v, K_a, nu)`

# Returns
- Wet growth capacity [kg/kg/s] (positive; zero when T ≥ T₀)
"""
@inline function wet_growth_capacity(p3, qⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    T₀ = prp.freezing_temperature
    below_freezing = T < T₀

    L_f = fusion_latent_heat(constants, T)
    L_s = sublimation_latent_heat(constants, T)
    thermodynamic_constants = isnothing(constants) ? ThermodynamicConstants(FT) : constants
    Rᵛ = FT(vapor_gas_constant(thermodynamic_constants))

    K_a = transport.K_a
    D_v = transport.D_v
    nu  = transport.nu

    # Saturation mixing ratio at T₀ over liquid (Fortran: qsat0)
    e_s0 = saturation_vapor_pressure_at_freezing(constants, T₀)
    ρ_vs = e_s0 / (Rᵛ * T₀)
    ρ_v = qᵛ * ρ

    # Mean ice particle mass
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)

    # Ventilation integral (same as deposition/refreezing)
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                    p3.ice.deposition.ventilation_enhanced,
                                    m_mean, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3)

    # Heat balance: sensible + latent
    Q_sensible = K_a * (T₀ - T)
    Q_latent = L_s * D_v * (ρ_vs - ρ_v)

    # Fortran applies 2π/Lf only to the latent term; the sensible-conduction
    # term uses the capm convention directly.
    qwgrth = C_fv * (Q_sensible + FT(2π) * Q_latent / L_f) * nⁱ_eff

    return ifelse(below_freezing, clamp_positive(qwgrth), zero(FT))
end

"""
    refreezing_rate(p3, qʷⁱ, qⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport)

Compute refreezing rate of liquid on ice using the heat-balance formula.

Below freezing, liquid coating on ice particles refreezes. The rate is
determined by the heat flux at the particle surface:

```math
\\frac{dm}{dt} = C f_v \\left[K_a(T_0-T) + \\frac{2π}{L_f} L_s D_v (\\rho_{vs} - \\rho_v)\\right]
```

This mirrors the melting formula with reversed temperature gradient.
See [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 44.

# Arguments
- `p3`: P3 microphysics scheme
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `P`: Pressure [Pa] (unused; reserved for future transport recomputation)
- `qᵛ`: Vapor mass fraction [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants (or `nothing` for Fortran-matched hardcoded values)
- `transport`: Pre-computed air transport properties `(; D_v, K_a, nu)`

# Returns
- Rate of liquid → ice refreezing [kg/kg/s]
"""
@inline function refreezing_rate(p3, qʷⁱ, qⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport)
    FT = typeof(qʷⁱ)
    prp = p3.process_rates

    qʷⁱ_eff = clamp_positive(qʷⁱ)
    qⁱ_eff  = clamp_positive(qⁱ)
    nⁱ_eff  = clamp_positive(nⁱ)

    T₀ = prp.freezing_temperature
    below_freezing = T < T₀
    ΔT = T₀ - T  # positive when below freezing

    L_f = fusion_latent_heat(constants, T)
    L_s = sublimation_latent_heat(constants, T)
    thermodynamic_constants = isnothing(constants) ? ThermodynamicConstants(FT) : constants
    Rᵛ = FT(vapor_gas_constant(thermodynamic_constants))

    K_a = transport.K_a
    D_v = transport.D_v
    nu  = transport.nu

    # Saturation vapor density at T₀ (liquid surface at melting point)
    e_s0 = saturation_vapor_pressure_at_freezing(constants, T₀)
    ρ_vs = e_s0 / (Rᵛ * T₀)

    # Ambient vapor density
    ρ_v = qᵛ * ρ

    # Mean ice particle mass
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)

    # Ventilation integral (ice-particle capacitance; same path as deposition)
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                    p3.ice.deposition.ventilation_enhanced,
                                    m_mean, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3)

    # Heat balance for refreezing:
    # Conductive: K_a × (T₀ - T) removes heat from liquid → promotes freezing
    Q_sensible = K_a * ΔT

    # Vapor: L_s × D_v × (ρ_vs - ρ_v)
    # Subsaturated (ρ_vs > ρ_v): evaporation cools particle → promotes freezing
    # Supersaturated (ρ_vs < ρ_v): condensation warms particle → opposes freezing
    Q_latent = L_s * D_v * (ρ_vs - ρ_v)

    # Only refreeze when net heat balance favors it. As in the Fortran wet-growth
    # and refreezing paths, 2π/Lf multiplies only the latent-diffusion term.
    dm_dt_refrz = clamp_positive(C_fv * (Q_sensible + FT(2π) * Q_latent / L_f))

    refrz_rate = nⁱ_eff * dm_dt_refrz

    # Limit to available liquid on ice
    τ_safety = FT(1)  # [s]
    max_refrz = qʷⁱ_eff / τ_safety
    refrz_rate = min(refrz_rate, max_refrz)

    return ifelse(below_freezing, refrz_rate, zero(FT))
end
