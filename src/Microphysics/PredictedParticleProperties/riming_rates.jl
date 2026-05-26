
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
    # Fortran uses T <= trplpt for below-freezing riming
    below_freezing = T <= T₀
    active = below_freezing & (qᶜˡ_eff > q_threshold) & (qⁱ_total > q_threshold) & (nⁱ_eff > n_threshold)

    # Mean particle mass
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    # PSD-integrated collection kernel ⟨A×V⟩ from lookup table.
    # Computes ∫ V(D) A(D) N'(D) dD with E=1 (geometric kernel).
    AV_per_particle = collection_kernel_per_particle(p3.ice.collection.rain_collection,
                                                       m_mean, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)

    # Air density correction for ice particle fall speed (Heymsfield et al. 2007):
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
    # Fortran uses T > trplpt for above-freezing collection
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
    # Fortran uses T > trplpt for above-freezing collection
    above_freezing = T > T₀
    active = above_freezing & (qʳ_eff > q_threshold) & (qⁱ_total > q_threshold) & (nⁱ_eff > n_threshold)

    # Use Table 2 (double-PSD kernel) for above-freezing rain-ice collection,
    # matching the below-freezing rain_riming_rate path and Fortran P3 convention.
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Diagnose rain lambda for Table 2 lookup
    λ_r = rain_slope_parameter(qʳ_eff, nʳ_eff, prp)
    nʳ_bounded = rain_number_from_slope(qʳ_eff, λ_r, prp)

    mass_kernel = rain_riming_mass_kernel(rain_ice_collection_table(p3),
        m_mean, λ_r, nʳ_bounded, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)

    # Fortran convention: qrcoll = 10^(f1pr08 + logn0r) × ni × env.
    # N0r = nr × λr (for μr=0).
    N0r = nʳ_bounded * λ_r
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
    # Fortran uses T <= trplpt for below-freezing riming
    below_freezing = T <= T₀
    active = below_freezing & (qʳ_eff > q_threshold) & (qⁱ_total > q_threshold) & (nⁱ_eff > n_threshold)

    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Diagnose rain DSD slope parameter
    λ_r = rain_slope_parameter(qʳ_eff, nʳ_eff, prp)
    nʳ_bounded = rain_number_from_slope(qʳ_eff, λ_r, prp)

    # Use Table 2 (double-PSD kernel) for ice-rain mass collection.
    # Fortran convention: qrcol = 10^(f1pr08 + logn0r) × ni × ρ × rhofaci × E
    # The table stores the double-PSD integral with N0r factored out.
    # N0r = nr × λr (for μr=0 used in table generation).
    mass_kernel = rain_riming_mass_kernel(rain_ice_collection_table(p3),
        m_mean, λ_r, nʳ_bounded, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)

    N0r = nʳ_bounded * λ_r
    rate = Eʳⁱ * N0r * nⁱ_eff * ρ * rhofaci * mass_kernel

    return ifelse(active, rate, zero(FT))
end

# Rain-ice collection table path — uses the dedicated ice-rain mass collection table (Fortran f1pr07).
@inline function rain_riming_mass_kernel(rain_ice_table::P3RainIceCollectionTable,
                                           m_mean, λ_r, nʳ, Fᶠ, Fˡ, ρᶠ, prp, p3,
                                           μ = zero(typeof(m_mean)))
    mass_kernel, _, _ = ice_rain_collection_lookup(rain_ice_table, m_mean, λ_r, Fᶠ, Fˡ, ρᶠ, μ)
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
    nʳ_bounded = rain_number_from_slope(qʳ_eff, λ_r, prp)

    # Use Table 2 (number-weighted kernel) for ice-rain number collection.
    # Fortran convention: nrcol = 10^(f1pr07 + logn0r) × ni × ρ × rhofaci × E
    # N0r = nr × λr (for μr=0).
    number_kernel = rain_riming_number_kernel(rain_ice_collection_table(p3),
        m_mean, λ_r, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)

    N0r = nʳ_bounded * λ_r
    rate = Eʳⁱ * N0r * nⁱ_eff * ρ * rhofaci * number_kernel

    return ifelse(active, rate, zero(FT))
end

# Rain-ice collection table path — uses the dedicated ice-rain number collection table (Fortran f1pr08).
@inline function rain_riming_number_kernel(rain_ice_table::P3RainIceCollectionTable,
                                             m_mean, λ_r, Fᶠ, Fˡ, ρᶠ, prp, p3,
                                             μ = zero(typeof(m_mean)))
    _, number_kernel, _ = ice_rain_collection_lookup(rain_ice_table, m_mean, λ_r, Fᶠ, Fˡ, ρᶠ, μ)
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
