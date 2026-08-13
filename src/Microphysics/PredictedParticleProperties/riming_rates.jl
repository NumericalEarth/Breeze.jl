
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
\\frac{dq^{cl}}{dt} = -E^{ci} q^{cl} ρ n^i ⟨A V⟩
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
@inline cloud_riming_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μⁱ, qʷⁱ = zero(typeof(qⁱ))) =
    # Riming is the below-freezing branch, T ≤ T₀
    cloud_collection_mass_rate(p3, qᶜˡ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ,
                               T <= p3.process_rates.freezing_temperature, μⁱ, qʷⁱ)

"""
$(TYPEDSIGNATURES)

Ice sweep-out of cloud water, gated by `temperature_active`. Below freezing the
collected water rimes onto the ice; above freezing it is shed as rain. Both use
the same kernel, so they differ only in the gate and in what the caller does with
the result — see [`cloud_riming_rate`](@ref) and [`cloud_warm_collection_rate`](@ref).
"""
@inline function cloud_collection_mass_rate(p3, qᶜˡ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ,
                                            temperature_active, μⁱ,
                                            qʷⁱ = zero(typeof(qⁱ)))
    FT = typeof(qᶜˡ)
    parameters = p3.process_rates

    Eᶜⁱ = parameters.cloud_ice_collection_efficiency

    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = max(clamp_positive(nⁱ), p3.minimum_number_mixing_ratio)

    active = temperature_active &
             (qᶜˡ_eff >= p3.minimum_mass_mixing_ratio) &
             (qⁱ_total >= p3.minimum_mass_mixing_ratio)

    # Mean particle mass
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    # PSD-integrated cloud-water collection kernel ⟨A×V⟩ from lookup table
    # ∫ V(D) A(D) N'(D) dD with E=1 (geometric kernel).
    collection_kernel = collection_kernel_per_particle(p3.ice.collection.cloud_collection,
                                                        m_mean, Fᶠ, Fˡ, ρᶠ, μⁱ)

    # Air density correction for ice particle fall speed (Heymsfield et al. 2007):
    # ρfaci = (ρ₀_ice / ρ)^0.54, where ρ₀_ice = 60000/(287.15×253.15) ≈ 0.826 kg/m³.
    # This is the ice reference density, NOT the surface/rain reference ≈ 1.275.
    ρ₀ = p3.ice.fall_speed.reference_air_density
    density_correction = ice_air_density_correction(ρ₀, ρ)

    # Collection rate = E × qc × ni × ρ × rhofaci × ⟨A×V⟩
    rate = Eᶜⁱ * qᶜˡ_eff * nⁱ_eff * ρ * density_correction * collection_kernel

    return ifelse(active, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Compute above-freezing cloud collection by melting ice.

When `T > T₀`, ice particles still sweep up cloud droplets via the same collection
kernel as riming, but the collected water is immediately shed as rain drops (not frozen).
The number of new rain drops follows `process_rates.shed_drop_mass`, whose default
is the mass of a 1 mm drop, ``π/6 ρ^L D³ ≈ 5.24 × 10⁻⁷`` kg.

# Returns
- `(mass_rate, number_rate)`: Cloud → rain mass rate [kg/kg/s] and rain number source [1/kg/s]
"""
@inline function cloud_warm_collection_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μⁱ, qʷⁱ = zero(typeof(qⁱ)))
    # Collection above freezing is the T > T₀ branch
    above_freezing = T > p3.process_rates.freezing_temperature
    mass_rate = cloud_collection_mass_rate(p3, qᶜˡ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ,
                                           above_freezing, μⁱ, qʷⁱ)
    # Collected water is shed as 1 mm drops: m = π/6 × 1000 × 0.001³ ≈ 5.2e-7 kg.
    # The gate is already applied to `mass_rate`, so the quotient carries it.
    return (mass_rate, mass_rate / p3.process_rates.shed_drop_mass)
end

"""
$(TYPEDSIGNATURES)

Compute above-freezing rain collection by melting ice.

When `T > T₀` and liquid fraction is active, rain drops collected by ice
contribute to the liquid coating (qʷⁱ) rather than to rime.
Uses the same collection kernel as rain riming.
See [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

# Returns
- Rain mass rate collected onto ice [kg/kg/s]
"""
@inline rain_warm_collection_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ,
                                  μⁱ = zero(typeof(qʳ)), qʷⁱ = zero(typeof(qⁱ))) =
    # Collection above freezing is the T > T₀ branch. It uses the same Table 2
    # double-PSD kernel as the below-freezing path.
    rain_collection_mass_rate(p3, qʳ, nʳ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ,
                              T > p3.process_rates.freezing_temperature, μⁱ, qʷⁱ)

"""
    $(TYPEDSIGNATURES)    cloud_riming_number_rate(qᶜˡ, Nᶜˡ, ρ, riming_rate)

Compute cloud droplet number sink from riming.

Returns `(Nᶜˡ / (ρ * qᶜˡ)) * riming_rate` [1/kg/s]: the per-mass cloud
number removal proportional to the rimed cloud mass fraction.

# Arguments
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜˡ`: Cloud droplet number concentration [1/m³]
- `ρ`: Air density [kg/m³]
- `riming_rate`: Cloud riming mass rate [kg/kg/s]

# Returns
- Rate of cloud number loss [1/kg/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline function cloud_riming_number_rate(qᶜˡ, Nᶜˡ, ρ, riming_rate)
    FT = typeof(qᶜˡ)

    # Nᶜˡ [#/m³] / (ρ [kg/m³] × qᶜˡ [kg/kg]) = nᶜˡ/qᶜˡ [#/kg].
    ratio = safe_divide(Nᶜˡ, ρ * qᶜˡ, zero(FT))

    return ratio * riming_rate
end

"""
$(TYPEDSIGNATURES)

Compute rain collection (riming) by ice particles using the continuous
collection equation with collision kernel integrated over the ice PSD,
plus a correction for the rain drop size distribution (C5 fix).

**C5 correction (double-PSD integration):**

The ice-rain collection tables integrate over *both* the ice PSD and the rain PSD,
capturing how rain drop size affects the collision geometry.
The geometric cross section is ``π/4 (D^i + D^r)^2``, not just ``π/4 (D^i)^2``.
For an exponential rain PSD (``μ^r = 0``) the exact cross-section correction to the
single-PSD ice-side integral is:

```math
C = 1 + 8 \\frac{\\bar{D}^r}{\\bar{D}^i} + 20 \\left(\\frac{\\bar{D}^r}{\\bar{D}^i}\\right)^2
```

where ``\\bar{D}^r = 1/λ^r`` and ``\\bar{D}^i`` is the mean ice diameter.
When ``n^r = 0`` the correction is 1 (no change from the legacy path).

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
@inline rain_riming_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ,
                         μⁱ = zero(typeof(qʳ)), qʷⁱ = zero(typeof(qⁱ))) =
    # Riming is the below-freezing branch, T ≤ T₀
    rain_collection_mass_rate(p3, qʳ, nʳ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ,
                              T <= p3.process_rates.freezing_temperature, μⁱ, qʷⁱ)

"""
$(TYPEDSIGNATURES)

Ice sweep-out of rain water, gated by `temperature_active`. Below freezing the
collected rain rimes onto the ice; above freezing it becomes liquid coating. The
mass counterpart of [`rain_collection_number_rate`](@ref); see
[`rain_riming_rate`](@ref) and [`rain_warm_collection_rate`](@ref) for the gates.
"""
@inline function rain_collection_mass_rate(p3, qʳ, nʳ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ,
                                           temperature_active, μⁱ = zero(typeof(qʳ)),
                                           qʷⁱ = zero(typeof(qⁱ)))
    FT = typeof(qʳ)
    parameters = p3.process_rates

    Eʳⁱ = parameters.rain_ice_collection_efficiency

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(clamp_positive(nʳ), p3.minimum_number_mixing_ratio)
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = max(clamp_positive(nⁱ), p3.minimum_number_mixing_ratio)

    active = temperature_active &
             (qʳ_eff >= p3.minimum_mass_mixing_ratio) &
             (qⁱ_total >= p3.minimum_mass_mixing_ratio)

    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    ρ₀ = p3.ice.fall_speed.reference_air_density
    density_correction = ice_air_density_correction(ρ₀, ρ)

    # Diagnose rain DSD slope parameter
    λʳ = rain_slope_parameter(qʳ_eff, nʳ_eff, parameters)
    nʳ_bounded = rain_number_from_slope(qʳ_eff, λʳ, parameters)

    # Use Table 2 (double-PSD kernel) for ice-rain mass collection.
    # The table stores the double-PSD integral with N₀ʳ factored out, so the rate is
    # kernel × N₀ʳ × nⁱ × ρ × (density correction) × E, with N₀ʳ = nʳ λʳ at μʳ = 0.
    mass_kernel = rain_riming_mass_kernel(rain_ice_collection_table(p3),
        m_mean, λʳ, nʳ_bounded, Fᶠ, Fˡ, ρᶠ, parameters, p3, μⁱ)

    Nʳ₀ = nʳ_bounded * λʳ
    rate = Eʳⁱ * Nʳ₀ * nⁱ_eff * ρ * density_correction * mass_kernel

    return ifelse(active, rate, zero(FT))
end

# Rain-ice collection table path — uses the dedicated ice-rain mass collection table.
@inline function rain_riming_mass_kernel(rain_ice_table::P3RainIceCollectionTable,
                                            m_mean, λʳ, nʳ, Fᶠ, Fˡ, ρᶠ, parameters, p3,
                                            μⁱ = zero(typeof(m_mean)))
    mass_kernel, _ = ice_rain_collection_lookup(rain_ice_table, m_mean, λʳ, Fᶠ, Fˡ, ρᶠ, μⁱ)
    return mass_kernel
end

"""
$(TYPEDSIGNATURES)

Compute rain number loss from rain-ice collection using the tabulated
number-weighted collection kernel (`RainCollectionNumber`) when
`temperature_active` is true.

Replaces the monodisperse approximation `(nʳ/qʳ) × mass_rate` with an
independent PSD-integrated number collection rate.
"""
@inline function rain_collection_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ,
                                             temperature_active, μⁱ = zero(typeof(qʳ)),
                                             qʷⁱ = zero(typeof(qⁱ)))
    FT = typeof(qʳ)
    parameters = p3.process_rates

    Eʳⁱ = parameters.rain_ice_collection_efficiency

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(clamp_positive(nʳ), p3.minimum_number_mixing_ratio)
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = max(clamp_positive(nⁱ), p3.minimum_number_mixing_ratio)

    active = temperature_active &
             (qʳ_eff >= p3.minimum_mass_mixing_ratio) &
             (qⁱ_total >= p3.minimum_mass_mixing_ratio)

    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    ρ₀ = p3.ice.fall_speed.reference_air_density
    density_correction = ice_air_density_correction(ρ₀, ρ)

    # Diagnose rain DSD slope parameter
    λʳ = rain_slope_parameter(qʳ_eff, nʳ_eff, parameters)
    nʳ_bounded = rain_number_from_slope(qʳ_eff, λʳ, parameters)

    # Use Table 2 (number-weighted kernel) for ice-rain number collection.
    # As for the mass kernel, N₀ʳ = nʳ λʳ at μʳ = 0 is factored back in here.
    number_kernel = rain_riming_number_kernel(rain_ice_collection_table(p3),
        m_mean, λʳ, Fᶠ, Fˡ, ρᶠ, parameters, p3, μⁱ)

    Nʳ₀ = nʳ_bounded * λʳ
    rate = Eʳⁱ * Nʳ₀ * nⁱ_eff * ρ * density_correction * number_kernel

    return ifelse(active, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Compute below-freezing rain number loss from riming using the tabulated
number-weighted collection kernel (`RainCollectionNumber`).
"""
@inline function rain_riming_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μⁱ = zero(typeof(qʳ)), qʷⁱ = zero(typeof(qⁱ)))
    # Riming is the T ≤ T₀ branch.
    below_freezing = T <= p3.process_rates.freezing_temperature
    return rain_collection_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ,
                                       below_freezing, μⁱ, qʷⁱ)
end

"""
$(TYPEDSIGNATURES)

Compute above-freezing rain number loss using the tabulated number-weighted
collection kernel (`RainCollectionNumber`).
"""
@inline function rain_warm_collection_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ,
                                                  μⁱ = zero(typeof(qʳ)), qʷⁱ = zero(typeof(qⁱ)))
    above_freezing = T > p3.process_rates.freezing_temperature
    return rain_collection_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ,
                                       above_freezing, μⁱ, qʷⁱ)
end

# Rain-ice collection table path — uses the dedicated ice-rain number collection table.
@inline function rain_riming_number_kernel(rain_ice_table::P3RainIceCollectionTable,
                                           m_mean, λʳ, Fᶠ, Fˡ, ρᶠ, parameters, p3,
                                           μⁱ = zero(typeof(m_mean)))
    _, number_kernel = ice_rain_collection_lookup(rain_ice_table, m_mean, λʳ,
                                                   Fᶠ, Fˡ, ρᶠ, μⁱ)
    return number_kernel
end
