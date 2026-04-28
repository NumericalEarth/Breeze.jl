@inline function ice_rain_collection_lookup(table::P3LookupTable2, m̄, λr, Fᶠ, Fˡ, ρᶠ, μ = zero(typeof(m̄)))
    FT = typeof(m̄)
    log_m = log10(m̄)
    log_λ = log10(λr)
    z_val = _ice_rain_sixth_moment_lookup(table.sixth_moment, log_m, log_λ, Fᶠ, Fˡ, ρᶠ, μ, FT)
    # Fortran table stores rain-ice mass and number kernels as log10;
    # exponentiate to recover physical values (Fortran runtime: 10.**proc).
    # Sixth moment (m6collr) is NOT log10.
    return exp10(table.mass(log_m, log_λ, Fᶠ, Fˡ, ρᶠ, μ)),
           exp10(table.number(log_m, log_λ, Fᶠ, Fˡ, ρᶠ, μ)),
           z_val
end

@inline _ice_rain_sixth_moment_lookup(table, log_m, log_λ, Fᶠ, Fˡ, ρᶠ, μ, FT) = table(log_m, log_λ, Fᶠ, Fˡ, ρᶠ, μ)
@inline _ice_rain_sixth_moment_lookup(::Nothing, log_m, log_λ, Fᶠ, Fˡ, ρᶠ, μ, FT) = zero(FT)

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
    nⁱ_eff = clamp_positive(nⁱ)

    # Thresholds
    qⁱ_threshold = FT(1e-14)
    nⁱ_threshold = FT(1e2)

    aggregation_active = (qⁱ_total > qⁱ_threshold) & (nⁱ_eff > nⁱ_threshold)

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
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    # PSD-integrated self-collection kernel (E-free) from lookup table.
    AV_kernel = aggregation_kernel(p3.ice.collection.aggregation,
                                     m_mean, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)

    # Collection kernel with temperature-dependent sticking efficiency
    K_mean = Eᵢᵢ * AV_kernel

    # Number loss rate: ρ × K × n² × rhofaci (positive magnitude)
    # The ρ factor converts the volumetric kernel [m³/s] to mass-specific
    # tendency [1/kg/s]. The 1/2 self-collection factor is already included
    # in the kernel (table stores half-integral, analytical path includes 0.5 factor).
    # Sign convention (M7): returns positive; caller subtracts in tendency assembly.
    # Use ice reference density (Fortran rhosui, P=600 hPa, T=-20°C), not rain reference.
    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)
    rate = ρ * K_mean * nⁱ_eff^2 * rhofaci

    return ifelse(aggregation_active, rate, zero(FT))
end

#####
##### Phase 2: Riming (cloud and rain collection by ice)
#####

"""
$(TYPEDSIGNATURES)

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
function cloud_riming_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ, qʷⁱ = zero(typeof(qⁱ)))
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    Eᶜⁱ = prp.cloud_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-14)
    n_threshold = FT(1)
    # D3: Fortran uses T <= trplpt for below-freezing riming
    below_freezing = T <= T₀
    active = below_freezing & (qᶜˡ_eff > q_threshold) & (qⁱ_total > q_threshold) & (nⁱ_eff > n_threshold)

    # Mean particle mass
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    # PSD-integrated collection kernel ⟨A×V⟩ from lookup table.
    # Computes ∫ V(D) A(D) N'(D) dD with E=1 (geometric kernel).
    AV_per_particle = collection_kernel_per_particle(p3.ice.collection.rain_collection,
                                                       m_mean, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)

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
$(TYPEDSIGNATURES)

Compute above-freezing cloud collection by melting ice (Fortran qcshd/ncshdc pathway).

When `T > T₀`, ice particles still sweep up cloud droplets via the same collection
kernel as riming, but the collected water is immediately shed as rain drops (not frozen).
The number of new rain drops assumes 1mm shed drops (Fortran: ncshdc = qcshd × 1.923e6).

# Returns
- `(mass_rate, number_rate)`: Cloud → rain mass rate [kg/kg/s] and rain number source [1/kg/s]
"""
@inline function cloud_warm_collection_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ, qʷⁱ = zero(typeof(qⁱ)))
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    Eᶜⁱ = prp.cloud_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-14)
    n_threshold = FT(1)
    # D3: Fortran uses T > trplpt for above-freezing collection
    above_freezing = T > T₀
    active = above_freezing & (qᶜˡ_eff > q_threshold) & (qⁱ_total > q_threshold) & (nⁱ_eff > n_threshold)

    # Same collection kernel as cloud_riming_rate
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    AV_per_particle = collection_kernel_per_particle(p3.ice.collection.rain_collection,
                                                       m_mean, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)
    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    mass_rate = Eᶜⁱ * qᶜˡ_eff * nⁱ_eff * ρ * rhofaci * AV_per_particle
    # Fortran: ncshdc = qcshd * 1.923e6 (shed as 1mm drops: m = π/6 × 1000 × 0.001³ ≈ 5.2e-7 kg)
    number_rate = mass_rate * FT(1.923e6)

    return (ifelse(active, mass_rate, zero(FT)),
            ifelse(active, number_rate, zero(FT)))
end

"""
$(TYPEDSIGNATURES)

Compute above-freezing rain collection by melting ice (Fortran qrcoll pathway).

When `T > T₀` and liquid fraction is active, rain drops collected by ice
contribute to the liquid coating (qʷⁱ) rather than to rime.
Uses the same collection kernel as rain riming.
See [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

# Returns
- Rain mass rate collected onto ice [kg/kg/s]
"""
@inline function rain_warm_collection_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ = zero(typeof(qʳ)), qʷⁱ = zero(typeof(qⁱ)))
    FT = typeof(qʳ)
    prp = p3.process_rates

    Eʳⁱ = prp.rain_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-14)
    n_threshold = FT(1)
    # D3: Fortran uses T > trplpt for above-freezing collection
    above_freezing = T > T₀
    active = above_freezing & (qʳ_eff > q_threshold) & (qⁱ_total > q_threshold) & (nⁱ_eff > n_threshold)

    # D5: Use Table 2 (double-PSD kernel) for above-freezing rain-ice collection,
    # matching the below-freezing rain_riming_rate path and Fortran P3 convention.
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Diagnose rain lambda for Table 2 lookup
    λ_r = rain_slope_parameter(qʳ_eff, nʳ_eff, prp)

    mass_kernel = _rain_riming_mass_kernel(lookup_table_2(p3),
        m_mean, λ_r, nʳ_eff, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)

    # Fortran convention: qrcoll = 10^(f1pr08 + logn0r) × ni × env.
    # N0r = nr × λr (for μr=0).
    N0r = nʳ_eff * λ_r
    rate = Eʳⁱ * N0r * nⁱ_eff * ρ * rhofaci * mass_kernel

    return ifelse(active, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)    cloud_riming_number_rate(qᶜˡ, Nᶜ, ρ, riming_rate)

Compute cloud droplet number sink from riming.

Returns `(Nᶜ / (ρ * qᶜˡ)) * riming_rate` [1/kg/s]: the per-mass cloud
number removal proportional to the rimed cloud mass fraction.

# Arguments
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜ`: Cloud droplet number concentration [1/m³]
- `ρ`: Air density [kg/m³]
- `riming_rate`: Cloud riming mass rate [kg/kg/s]

# Returns
- Rate of cloud number loss [1/kg/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline function cloud_riming_number_rate(qᶜˡ, Nᶜ, ρ, riming_rate)
    FT = typeof(qᶜˡ)

    # Nᶜ [#/m³] / (ρ [kg/m³] × qᶜˡ [kg/kg]) = nᶜˡ/qᶜˡ [#/kg],
    # matching Fortran nc/qc.
    ratio = safe_divide(Nᶜ, ρ * qᶜˡ, zero(FT))

    return ratio * riming_rate
end

"""
$(TYPEDSIGNATURES)

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
function rain_riming_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ = zero(typeof(qʳ)), qʷⁱ = zero(typeof(qⁱ)))
    FT = typeof(qʳ)
    prp = p3.process_rates

    Eʳⁱ = prp.rain_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-14)
    n_threshold = FT(1)
    # D3: Fortran uses T <= trplpt for below-freezing riming
    below_freezing = T <= T₀
    active = below_freezing & (qʳ_eff > q_threshold) & (qⁱ_total > q_threshold) & (nⁱ_eff > n_threshold)

    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Diagnose rain DSD slope parameter
    λ_r = rain_slope_parameter(qʳ_eff, nʳ_eff, prp)

    # H6: Use Table 2 (double-PSD kernel) for ice-rain mass collection.
    # Fortran convention: qrcol = 10^(f1pr08 + logn0r) × ni × ρ × rhofaci × E
    # The table stores the double-PSD integral with N0r factored out.
    # N0r = nr × λr (for μr=0 used in table generation).
    mass_kernel = _rain_riming_mass_kernel(lookup_table_2(p3),
        m_mean, λ_r, nʳ_eff, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)

    N0r = nʳ_eff * λ_r
    rate = Eʳⁱ * N0r * nⁱ_eff * ρ * rhofaci * mass_kernel

    return ifelse(active, rate, zero(FT))
end

# H6: Table 2 path — use the dedicated ice-rain mass collection table (Fortran f1pr07).
@inline function _rain_riming_mass_kernel(table2::P3LookupTable2,
                                           m_mean, λ_r, nʳ, Fᶠ, Fˡ, ρᶠ, prp, p3,
                                           μ = zero(typeof(m_mean)))
    mass_kernel, _, _ = ice_rain_collection_lookup(table2, m_mean, λ_r, Fᶠ, Fˡ, ρᶠ, μ)
    return mass_kernel
end

"""
$(TYPEDSIGNATURES)

Backward-compatible 8-argument overload of `rain_riming_rate` without rain DSD correction.
Passes `nʳ = 0`, which disables the C5 double-PSD cross-section correction.
Prefer the 9-argument form `rain_riming_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)`.
"""
function rain_riming_rate(p3, qʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    FT = typeof(qʳ)
    return rain_riming_rate(p3, qʳ, zero(FT), qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
end

"""
$(TYPEDSIGNATURES)

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
function rain_riming_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ = zero(typeof(qʳ)), qʷⁱ = zero(typeof(qⁱ)))
    FT = typeof(qʳ)
    prp = p3.process_rates

    Eʳⁱ = prp.rain_ice_collection_efficiency
    T₀ = prp.freezing_temperature

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    q_threshold = FT(1e-14)
    n_threshold = FT(1)
    # m14: Fortran uses .le. for both mass and number riming
    below_freezing = T <= T₀
    active = below_freezing & (qʳ_eff > q_threshold) & (qⁱ_total > q_threshold) & (nʳ_eff > n_threshold) & (nⁱ_eff > n_threshold)

    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Diagnose rain DSD slope parameter
    λ_r = rain_slope_parameter(qʳ_eff, nʳ_eff, prp)

    # H6: Use Table 2 (number-weighted kernel) for ice-rain number collection.
    # Fortran convention: nrcol = 10^(f1pr07 + logn0r) × ni × ρ × rhofaci × E
    # N0r = nr × λr (for μr=0).
    number_kernel = _rain_riming_number_kernel(lookup_table_2(p3),
        m_mean, λ_r, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)

    N0r = nʳ_eff * λ_r
    rate = Eʳⁱ * N0r * nⁱ_eff * ρ * rhofaci * number_kernel

    return ifelse(active, rate, zero(FT))
end

# H6: Table 2 path — use the dedicated ice-rain number collection table (Fortran f1pr08).
@inline function _rain_riming_number_kernel(table2::P3LookupTable2,
                                             m_mean, λ_r, Fᶠ, Fˡ, ρᶠ, prp, p3,
                                             μ = zero(typeof(m_mean)))
    _, number_kernel, _ = ice_rain_collection_lookup(table2, m_mean, λ_r, Fᶠ, Fˡ, ρᶠ, μ)
    return number_kernel
end

"""
$(TYPEDSIGNATURES)

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
function rain_riming_number_rate(qʳ, nʳ, riming_rate)
    FT = typeof(qʳ)

    ratio = safe_divide(nʳ, qʳ, zero(FT))

    return ratio * riming_rate
end

"""
$(TYPEDSIGNATURES)

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
function rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport,
                      μ_c, λ_c)
    FT = typeof(T)
    prp = p3.process_rates
    qsmall = p3.minimum_mass_mixing_ratio

    ρ_rim_min = prp.minimum_rime_density
    ρ_rim_max = prp.maximum_rime_density
    T₀ = prp.freezing_temperature
    ρ_water = prp.liquid_water_density

    qᶜˡ_abs = clamp_positive(qᶜˡ) * ρ
    μ_air = transport.nu * ρ
    g = constants.gravitational_acceleration

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

# Backward-compatible 8-arg method: uses prescribed cloud DSD (μ_c, Nᶜ from p3.cloud).
# The full 10-arg form takes locally diagnosed (μ_c, λ_c) per Fortran p3_main parity.
function rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport)
    FT = typeof(T)
    μ_c = p3.cloud.shape_parameter
    Nᶜ = p3.cloud.number_concentration
    ρ_water = p3.process_rates.liquid_water_density
    qᶜˡ_abs = clamp_positive(qᶜˡ) * ρ
    λ_c_uncapped = cbrt(
        FT(π) * ρ_water * Nᶜ * (μ_c + 3) * (μ_c + 2) * (μ_c + 1) /
        (FT(6) * max(qᶜˡ_abs, FT(1e-20)))
    )
    λ_c = clamp(λ_c_uncapped, (μ_c + 1) * FT(2.5e4), (μ_c + 1) * FT(1e6))
    return rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport, μ_c, λ_c)
end

#####
##### Phase 2: Shedding and Refreezing (liquid fraction dynamics)
#####

"""
$(TYPEDSIGNATURES)

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
function shedding_rate(p3, qʷⁱ, qⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ, m_mean, μ)
    FT = typeof(qʷⁱ)

    qʷⁱ_eff = clamp_positive(qʷⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # Lookup ∫_{D≥9mm} m(D) N'(D) dD (normalized per particle)
    f1pr28 = shedding_integral(p3.ice.bulk_properties.shedding, m_mean, Fᶠ, Fˡ, ρᶠ, μ)

    # Fortran: qlshd = Fr × f1pr28 × ni × Fl
    # Fr = rime fraction of ice-only mass (= Fᶠ in Julia convention since qⁱ excludes qʷⁱ)
    rate = Fᶠ * f1pr28 * nⁱ_eff * Fˡ

    # Bound by available liquid: qlshd ≤ qwi / dt_safety
    rate = clamp_positive(rate)
    τ_safety = p3.process_rates.sink_limiting_timescale
    rate = min(rate, qʷⁱ_eff / τ_safety)

    return rate
end

"""
$(TYPEDSIGNATURES)

Lookup the PSD-integrated shedding mass for D ≥ 9 mm particles
from tabulated `TabulatedFunction5D`.
"""
@inline function shedding_integral(table::P3Table5D, m_mean, Fᶠ, Fˡ, ρᶠ, μ)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, FT(1e-20)))
    return table(log_m, Fᶠ, Fˡ, ρᶠ, μ)
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
    # Liquid-fraction shedding uses 1.928e6 drops/kg (Fortran nlshd, line 3350),
    # slightly different from cloud/wet-growth shedding (1.923e6).
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
function wet_growth_capacity(p3, qⁱ, qʷⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport, μ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    T₀ = prp.freezing_temperature
    below_freezing = T < T₀

    L_f = fusion_latent_heat(constants, T)
    L_s = sublimation_latent_heat(constants, T)
    Rᵛ = FT(vapor_gas_constant(constants))

    K_a = transport.K_a
    D_v = transport.D_v
    nu  = transport.nu

    # M10: use mixing ratio convention (Fortran: rho*Ls*Dv*(qsat0-Qv))
    Rᵈ = FT(dry_air_gas_constant(constants))
    ε = Rᵈ / Rᵛ
    e_s0 = saturation_vapor_pressure_at_freezing(constants, T₀)
    q_sat0 = ε * e_s0 / max(P - e_s0, FT(1))

    # Mean ice particle mass
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)

    # Ventilation integral (same as deposition/refreezing)
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                    p3.ice.deposition.ventilation_enhanced,
                                    m_mean, Fᶠ, Fˡ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)

    # Heat balance: sensible + latent
    Q_sensible = K_a * (T₀ - T)
    Q_latent = L_s * D_v * ρ * (q_sat0 - qᵛ)

    # Fortran applies 2π/Lf only to the latent term; the sensible-conduction
    # term uses the capm convention directly.
    qwgrth = C_fv * (Q_sensible + FT(2π) * Q_latent / L_f) * nⁱ_eff

    return ifelse(below_freezing, clamp_positive(qwgrth), zero(FT))
end

"""
$(TYPEDSIGNATURES)

Compute refreezing rate of liquid on ice using the heat-balance formula.

Below freezing, liquid coating on ice particles refreezes. The rate is
determined by the heat flux at the particle surface:

```math
\\frac{dm}{dt} = C f_v \\left[K_a(T_0-T) + \\frac{2π}{L_f} ρ L_s D_v (q_{sat0} - q_v)\\right]
```

This mirrors the melting formula with reversed temperature gradient.
See [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 44.

# Arguments
- `p3`: P3 microphysics scheme
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `P`: Pressure [Pa]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants (or `nothing` for Fortran-matched hardcoded values)
- `transport`: Pre-computed air transport properties `(; D_v, K_a, nu)`

# Returns
- Rate of liquid → ice refreezing [kg/kg/s]
"""
function refreezing_rate(p3, qʷⁱ, qⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport, μ)
    FT = typeof(qʷⁱ)
    prp = p3.process_rates

    qʷⁱ_eff = clamp_positive(qʷⁱ)
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff  = clamp_positive(nⁱ)

    T₀ = prp.freezing_temperature
    below_freezing = T < T₀
    ΔT = T₀ - T  # positive when below freezing

    L_f = fusion_latent_heat(constants, T)
    L_s = sublimation_latent_heat(constants, T)
    Rᵛ = FT(vapor_gas_constant(constants))

    K_a = transport.K_a
    D_v = transport.D_v
    nu  = transport.nu

    # M10: use mixing ratio convention (Fortran: rho*Ls*Dv*(qsat0-Qv))
    Rᵈ = FT(dry_air_gas_constant(constants))
    ε = Rᵈ / Rᵛ
    e_s0 = saturation_vapor_pressure_at_freezing(constants, T₀)
    q_sat0 = ε * e_s0 / max(P - e_s0, FT(1))

    # Mean ice particle mass
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)

    # Ventilation integral (ice-particle capacitance; same path as deposition)
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                    p3.ice.deposition.ventilation_enhanced,
                                    m_mean, Fᶠ, Fˡ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)

    # Heat balance for refreezing:
    # Conductive: K_a × (T₀ - T) removes heat from liquid → promotes freezing
    Q_sensible = K_a * ΔT

    # Vapor: L_s × D_v × ρ × (q_sat0 - qᵛ)
    # Subsaturated (q_sat0 > qᵛ): evaporation cools particle → promotes freezing
    # Supersaturated (q_sat0 < qᵛ): condensation warms particle → opposes freezing
    Q_latent = L_s * D_v * ρ * (q_sat0 - qᵛ)

    # Only refreeze when net heat balance favors it. As in the Fortran wet-growth
    # and refreezing paths, 2π/Lf multiplies only the latent-diffusion term.
    dm_dt_refrz = clamp_positive(C_fv * (Q_sensible + FT(2π) * Q_latent / L_f))

    refrz_rate = nⁱ_eff * dm_dt_refrz

    # Limit to available liquid on ice
    τ_safety = p3.process_rates.sink_limiting_timescale
    max_refrz = qʷⁱ_eff / τ_safety
    refrz_rate = min(refrz_rate, max_refrz)

    return ifelse(below_freezing, refrz_rate, zero(FT))
end
