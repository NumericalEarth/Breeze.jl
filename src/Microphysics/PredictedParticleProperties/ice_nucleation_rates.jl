#####
##### Ice nucleation (deposition and immersion freezing)
#####

"""
    deposition_nucleation_rate(p3, T, qᵛ, qᵛ⁺ⁱ, nⁱ, ρ)

Compute ice nucleation rate from deposition/condensation freezing.

New ice crystals nucleate when temperature is below a threshold and the air
is supersaturated with respect to ice. Uses [Cooper (1986)](@cite Cooper1986).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `T`: Temperature [K]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ⁱ`: Saturation vapor mass fraction over ice [kg/kg]
- `nⁱ`: Current ice number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Tuple (Q_nuc, N_nuc): mass rate [kg/kg/s] and number rate [1/kg/s]
"""
@inline function deposition_nucleation_rate(p3, T, qᵛ, qᵛ⁺ⁱ, nⁱ, ρ)
    FT = typeof(T)
    prp = p3.process_rates

    T_threshold = prp.nucleation_temperature_threshold
    Sⁱ_threshold = prp.nucleation_supersaturation_threshold
    N_max = prp.nucleation_maximum_concentration
    τ_nuc = prp.nucleation_timescale
    T₀ = prp.freezing_temperature
    mᵢ₀ = prp.nucleated_ice_mass
    c_nuc = prp.nucleation_coefficient

    # Ice supersaturation
    Sⁱ = (qᵛ - qᵛ⁺ⁱ) / max(qᵛ⁺ⁱ, FT(1e-10))

    # Conditions for nucleation
    nucleation_active = (T < T_threshold) & (Sⁱ > Sⁱ_threshold)

    # Cooper (1986): N_ice = c_nuc × exp(0.304 × (T₀ - T)) [1/m³]
    # Default c_nuc = 5.0 /m³ = 0.005 /L from Cooper (1986), divided by ρ for [1/kg]
    ΔT = T₀ - T
    N_cooper = c_nuc * exp(FT(0.304) * ΔT) / ρ

    # Limit to maximum and subtract existing ice
    N_equilibrium = min(N_cooper, N_max / ρ)

    # Nucleation rate: relaxation toward equilibrium
    N_nuc = clamp_positive(N_equilibrium - nⁱ) / τ_nuc

    # Mass nucleation rate
    Q_nuc = N_nuc * mᵢ₀

    # Zero out if conditions not met
    N_nuc = ifelse(nucleation_active & (N_nuc > FT(1e-20)), N_nuc, zero(FT))
    Q_nuc = ifelse(nucleation_active & (Q_nuc > FT(1e-30)), Q_nuc, zero(FT))

    return Q_nuc, N_nuc
end

"""
    immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)

Compute immersion freezing rate of cloud droplets using the
[Barklie and Gokhale (1959)](@cite BarklieGokhale1959) stochastic volume-dependent
freezing parameterization, following Fortran P3 v5.5.0.

The probability per droplet per second of freezing is ``J₀ V_{\\text{drop}} \\exp(a ΔT)``,
where ``J₀ ≈ 2`` m⁻³s⁻¹ is the nucleation rate coefficient (``a = 0.65``) and
``V_{\\text{drop}}`` is the individual droplet volume. For monodisperse cloud droplets
this gives a mass freezing rate proportional to ``q_c^2 / N_c``, making freezing
negligible for small droplets.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜ`: Cloud droplet number concentration [1/m³]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]

# Returns
- Tuple (Q_frz, N_frz): mass rate [kg/kg/s] and number rate [1/kg/s]
"""
@inline function immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    T_max = prp.immersion_freezing_temperature_max
    aimm = prp.immersion_freezing_coefficient
    T₀ = prp.freezing_temperature
    ρ_water = FT(prp.liquid_water_density)
    bimm = prp.immersion_freezing_nucleation_coefficient

    # H2: Compute μ_c dynamically from local Nᶜ (already [1/m³]) via Liu-Daum (2000),
    # then derive the PSD correction C(μ_c) = Γ(μ+7)Γ(μ+1)/Γ(μ+4)².
    # This replaces the precomputed construction-time value, allowing the correction
    # to vary spatially with the local droplet population.
    μ_c = liu_daum_shape_parameter(Nᶜ)
    psd_correction = psd_correction_spherical_volume(μ_c)

    qᶜˡ_eff = clamp_positive(qᶜˡ)

    # Conditions for freezing
    freezing_active = (T < T_max) & (qᶜˡ_eff > FT(1e-8))

    # Barklie-Gokhale (1959) stochastic immersion freezing.
    # Per-drop freezing probability: P(D) = bimm × V_drop × exp(aimm × ΔT)
    # For a gamma PSD, the PSD-integrated mass rate is boosted by C(μ_c),
    # but the number rate has C_N = 1 (no PSD correction).
    ΔT = max(T₀ - T, zero(FT))

    # Individual droplet mass and volume (monodisperse assumption)
    # Nᶜ is [1/m³]; convert to per-kg: nᶜ = Nᶜ/ρ [1/kg]
    nᶜ = max(Nᶜ / ρ, FT(1))
    m_drop = qᶜˡ_eff / nᶜ                     # [kg]
    V_drop = m_drop / ρ_water                   # [m³]

    # H1: Per-drop freezing probability per second (NO psd_correction).
    # The PSD correction applies only to the mass (6th moment) rate,
    # not the number (3rd moment) rate, matching Fortran P3 v5.5.0.
    prob_per_s = bimm * V_drop * exp(aimm * ΔT)

    # Mass freezing rate [kg/kg/s]: boosted by PSD correction (large drops freeze first)
    Q_frz = qᶜˡ_eff * psd_correction * prob_per_s

    # Number freezing rate [1/kg/s]: no PSD correction (C_N = 1)
    N_frz = nᶜ * prob_per_s

    Q_frz = ifelse(freezing_active, Q_frz, zero(FT))
    N_frz = ifelse(freezing_active, N_frz, zero(FT))

    return Q_frz, N_frz
end

"""
    immersion_freezing_rain_rate(p3, qʳ, nʳ, T)

Compute immersion freezing rate of rain drops.

Rain drops freeze when temperature is below a threshold. Uses
[Barklie and Gokhale (1959)](@cite BarklieGokhale1959) stochastic freezing
parameterization, following Fortran P3 v5.5.0.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `T`: Temperature [K]

# Returns
- Tuple (Q_frz, N_frz): mass rate [kg/kg/s] and number rate [1/kg/s]
"""
@inline function immersion_freezing_rain_rate(p3, qʳ, nʳ, T)
    FT = typeof(qʳ)
    prp = p3.process_rates

    T_max = prp.immersion_freezing_temperature_max
    aimm = prp.immersion_freezing_coefficient
    T₀ = prp.freezing_temperature
    ρ_water = FT(prp.liquid_water_density)
    bimm = prp.immersion_freezing_nucleation_coefficient
    psd_correction = prp.freezing_rain_psd_correction

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # Conditions for freezing
    freezing_active = (T < T_max) & (qʳ_eff > FT(1e-8))

    # Barklie-Gokhale (1959) stochastic volume-dependent freezing.
    # PSD correction for rain (broader PSD than cloud, μ_r ≈ 1-3).
    ΔT = max(T₀ - T, zero(FT))

    # Individual rain drop mass and volume (monodisperse assumption)
    nʳ_safe = max(nʳ_eff, FT(1))
    m_drop = qʳ_eff / nʳ_safe          # [kg]
    V_drop = m_drop / ρ_water            # [m³]

    # H1: Per-drop freezing probability per second (NO psd_correction).
    # The PSD correction applies only to the mass (6th moment) rate,
    # not the number (3rd moment) rate, matching Fortran P3 v5.5.0.
    prob_per_s = bimm * V_drop * exp(aimm * ΔT)

    # Mass freezing rate: boosted by PSD correction (large drops freeze first)
    Q_frz = qʳ_eff * psd_correction * prob_per_s

    # Number freezing rate: no PSD correction (C_N = 1)
    N_frz = nʳ_eff * prob_per_s

    Q_frz = ifelse(freezing_active, Q_frz, zero(FT))
    N_frz = ifelse(freezing_active, N_frz, zero(FT))

    return Q_frz, N_frz
end

#####
##### Homogeneous freezing
#####

"""
$(TYPEDSIGNATURES)

Compute homogeneous freezing rate of cloud droplets.

Below −40°C (233.15 K) all supercooled cloud liquid freezes instantaneously.
The frozen mass deposits as dense rime at ``ρ_{\\text{rim}} = 900`` kg/m³
(solid ice sphere), following the Fortran P3 v5.5.0 treatment of
[Morrison and Milbrandt (2015)](@cite Morrison2015parameterization).

The number rate ``N_{\\text{hom}}`` is capped by a mass-number consistency bound:
at most one ice particle per minimum-size cloud droplet
(`ProcessRateParameters.minimum_cloud_drop_mass`) can form from the frozen mass.
This prevents an ni explosion when `Nᶜ` is prescribed (continental aerosol loading)
and `qᶜˡ` is trace at ``T < -40°\\text{C}``.

**Fortran parity note:** This cap is not present in the Fortran P3 v5.5.0 reference,
where `Nᶜ` is prognostic and naturally depletes with cloud consumption. When
prognostic `Nᶜ` is implemented in Breeze, this cap can be removed.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜ`: Cloud droplet number concentration [1/m³]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]

# Returns
- Tuple (Q_hom, N_hom):
  - `Q_hom`: Mass rate cloud → ice [kg/kg/s]
  - `N_hom`: Number rate cloud → ice [1/kg/s], capped by mass-number consistency

# Example

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties:
    homogeneous_freezing_cloud_rate
p3 = PredictedParticlePropertiesMicrophysics()
Q, N = homogeneous_freezing_cloud_rate(p3, 1e-3, 100e6, 230.0, 1.2)
typeof(Q)

# output
Float64
```
"""
@inline function homogeneous_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    T_threshold = FT(prp.homogeneous_freezing_temperature)
    τ_hom = FT(prp.homogeneous_freezing_timescale)

    qᶜˡ_eff = clamp_positive(qᶜˡ)

    # Guard: temperature below threshold AND sufficient cloud liquid present
    freezing_active = (T < T_threshold) & (qᶜˡ_eff > FT(1e-8))

    # Instantaneous conversion: rate = mixing ratio / timescale
    Q_hom = qᶜˡ_eff / τ_hom

    # Number rate: Nᶜ is [1/m³] → divide by ρ for [1/kg]
    N_hom = Nᶜ / ρ / τ_hom

    # Mass-number consistency cap: cannot produce more particles than the number
    # of minimum-size droplets that could have frozen (prevents ni explosion when
    # Nᶜ is prescribed and qᶜˡ is trace).
    min_drop_mass = FT(prp.minimum_cloud_drop_mass)
    N_hom_max = Q_hom / min_drop_mass
    N_hom = min(N_hom, N_hom_max)

    Q_hom = ifelse(freezing_active, Q_hom, zero(FT))
    N_hom = ifelse(freezing_active, N_hom, zero(FT))

    return Q_hom, N_hom
end

"""
$(TYPEDSIGNATURES)

Compute homogeneous freezing rate of rain drops.

Below −40°C (233.15 K) all supercooled rain freezes instantaneously.
The frozen mass deposits as dense rime at ``ρ_{\\text{rim}} = 900`` kg/m³,
following the Fortran P3 v5.5.0 treatment of
[Morrison and Milbrandt (2015)](@cite Morrison2015parameterization).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `T`: Temperature [K]

# Returns
- Tuple (Q_hom, N_hom):
  - `Q_hom`: Mass rate rain → ice [kg/kg/s]
  - `N_hom`: Number rate rain → ice [1/kg/s]

# Example

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties:
    homogeneous_freezing_rain_rate
p3 = PredictedParticlePropertiesMicrophysics()
Q, N = homogeneous_freezing_rain_rate(p3, 1e-3, 1e4, 220.0)
typeof(Q)

# output
Float64
```
"""
@inline function homogeneous_freezing_rain_rate(p3, qʳ, nʳ, T)
    FT = typeof(qʳ)
    prp = p3.process_rates

    T_threshold = FT(prp.homogeneous_freezing_temperature)
    τ_hom = FT(prp.homogeneous_freezing_timescale)

    qʳ_eff = clamp_positive(qʳ)

    # Guard: temperature below threshold AND sufficient rain present
    freezing_active = (T < T_threshold) & (qʳ_eff > FT(1e-8))

    # Instantaneous conversion: rate = mixing ratio / timescale
    Q_hom = qʳ_eff / τ_hom

    # Number rate: nʳ already in [1/kg]
    N_hom = clamp_positive(nʳ) / τ_hom

    Q_hom = ifelse(freezing_active, Q_hom, zero(FT))
    N_hom = ifelse(freezing_active, N_hom, zero(FT))

    return Q_hom, N_hom
end

#####
##### Rime splintering (Hallett-Mossop secondary ice production)
#####

"""
    rime_splintering_rate(p3, cloud_riming, rain_riming, T, D_ice, Fˡ, surface_T, qᶠ)

Compute secondary ice production from rime splintering (Hallett-Mossop effect).

When rimed ice particles accrete supercooled drops, ice splinters are
ejected. This occurs only in a narrow temperature range around -5°C.
See [Hallett and Mossop (1974)](@cite HallettMossop1974).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `cloud_riming`: Cloud droplet riming rate [kg/kg/s]
- `rain_riming`: Rain riming rate [kg/kg/s]
- `T`: Temperature [K]
- `D_ice`: Mean ice diameter [m]
- `Fˡ`: Liquid fraction on ice [-]
- `surface_T`: Surface-temperature proxy for the warm-season shutoff [K]
- `qᶠ`: Existing rimed-ice mass [kg/kg]

# Returns
- Tuple (Q_spl, N_spl): ice mass rate [kg/kg/s] and number rate [1/kg/s]
"""
@inline function rime_splintering_rate(p3, cloud_riming, rain_riming, T, D_ice, Fˡ, surface_T, qᶠ)
    FT = typeof(T)
    prp = p3.process_rates

    T_low = prp.splintering_temperature_low
    T_high = prp.splintering_temperature_high
    T_peak = prp.splintering_temperature_peak
    c_splinter = prp.splintering_rate
    mᵢ₀ = prp.nucleated_ice_mass

    warm_branch = clamp((T - T_low) / (T_peak - T_low), zero(FT), one(FT))
    cold_branch = clamp((T_high - T) / (T_high - T_peak), zero(FT), one(FT))
    efficiency = ifelse(T <= T_peak, warm_branch, cold_branch)

    # Fortran nCat=1 path includes BOTH cloud and rain riming in splintering
    # (microphy_p3.f90 lines 3547-3574: HM_cloud + HM_rain blocks).
    total_riming = clamp_positive(cloud_riming) + clamp_positive(rain_riming)
    has_rime = qᶠ >= p3.minimum_mass_mixing_ratio
    active = (D_ice ≥ prp.splintering_diameter_threshold) &
             has_rime &
             (Fˡ < prp.splintering_liquid_fraction_max) &
             (surface_T < prp.splintering_surface_temperature_max)

    # Number of splinters produced
    N_spl = ifelse(active, efficiency * c_splinter * total_riming, zero(FT))

    # Mass of splinters
    Q_spl = N_spl * mᵢ₀

    return Q_spl, N_spl
end
