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

    # Ice supersaturation
    Sⁱ = (qᵛ - qᵛ⁺ⁱ) / max(qᵛ⁺ⁱ, FT(1e-10))

    # Conditions for nucleation
    nucleation_active = (T < T_threshold) & (Sⁱ > Sⁱ_threshold)

    # Cooper (1986): N_ice = 0.005 × exp(0.304 × (T₀ - T))
    ΔT = T₀ - T
    N_cooper = FT(0.005) * exp(FT(0.304) * ΔT) * FT(1000) / ρ

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
    psd_correction = prp.freezing_cloud_psd_correction

    qᶜˡ_eff = clamp_positive(qᶜˡ)

    # Conditions for freezing
    freezing_active = (T < T_max) & (qᶜˡ_eff > FT(1e-8))

    # Barklie-Gokhale (1959) stochastic immersion freezing.
    # Per-drop freezing probability: P(D) = bimm × V_drop × exp(aimm × ΔT)
    # For a gamma PSD, the PSD-integrated rate is boosted by Γ(7+μ)Γ(1+μ)/Γ(4+μ)²
    # relative to monodisperse: ≈20× for μ=0, ≈3× for μ=10.
    ΔT = max(T₀ - T, zero(FT))

    # Individual droplet mass and volume (monodisperse assumption)
    # Nᶜ is [1/m³]; convert to per-kg: nᶜ = Nᶜ/ρ [1/kg]
    nᶜ = max(Nᶜ / ρ, FT(1))
    m_drop = qᶜˡ_eff / nᶜ                     # [kg]
    V_drop = m_drop / ρ_water                   # [m³]

    # Per-drop freezing probability per second
    prob_per_s = bimm * psd_correction * V_drop * exp(aimm * ΔT)

    # Mass freezing rate [kg/kg/s]: each drop freezes with its own mass
    Q_frz = qᶜˡ_eff * prob_per_s

    # Number freezing rate [1/kg/s]
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

    # Per-drop freezing probability per second: bimm × psd × V_drop × exp(a × ΔT)
    prob_per_s = bimm * psd_correction * V_drop * exp(aimm * ΔT)

    # Mass freezing rate: qʳ × prob (each drop freezes with its own mass)
    Q_frz = qʳ_eff * prob_per_s

    # Number freezing rate: nʳ × prob
    N_frz = nʳ_eff * prob_per_s

    Q_frz = ifelse(freezing_active, Q_frz, zero(FT))
    N_frz = ifelse(freezing_active, N_frz, zero(FT))

    return Q_frz, N_frz
end

#####
##### Rime splintering (Hallett-Mossop secondary ice production)
#####

"""
    rime_splintering_rate(p3, cloud_riming, rain_riming, T)

Compute secondary ice production from rime splintering (Hallett-Mossop effect).

When rimed ice particles accrete supercooled drops, ice splinters are
ejected. This occurs only in a narrow temperature range around -5°C.
See [Hallett and Mossop (1974)](@cite HallettMossop1974).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `cloud_riming`: Cloud droplet riming rate [kg/kg/s]
- `rain_riming`: Rain riming rate [kg/kg/s]
- `T`: Temperature [K]

# Returns
- Tuple (Q_spl, N_spl): ice mass rate [kg/kg/s] and number rate [1/kg/s]
"""
@inline function rime_splintering_rate(p3, cloud_riming, rain_riming, T)
    FT = typeof(T)
    prp = p3.process_rates

    T_low = prp.splintering_temperature_low
    T_high = prp.splintering_temperature_high
    T_peak = prp.splintering_temperature_peak
    T_width = prp.splintering_temperature_width
    c_splinter = prp.splintering_rate
    mᵢ₀ = prp.nucleated_ice_mass

    # Hallett-Mossop temperature window
    in_HM_window = (T > T_low) & (T < T_high)

    # Efficiency peaks at T_peak, tapers to zero at boundaries
    efficiency = exp(-((T - T_peak) / T_width)^2)

    # Total riming rate
    total_riming = clamp_positive(cloud_riming + rain_riming)

    # Number of splinters produced
    N_spl = ifelse(in_HM_window,
                    efficiency * c_splinter * total_riming,
                    zero(FT))

    # Mass of splinters
    Q_spl = N_spl * mᵢ₀

    return Q_spl, N_spl
end
