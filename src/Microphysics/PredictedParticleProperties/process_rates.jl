#####
##### P3 Process Rates
#####
##### Microphysical process rate calculations for the P3 scheme.
##### All rate functions take the P3 scheme as first positional argument
##### to access parameters. No keyword arguments (GPU compatibility).
#####
##### Notation follows docs/src/appendix/notation.md
#####

using Oceananigans: Oceananigans

using Breeze.Thermodynamics: temperature

#####
##### Utility functions
#####

"""
    clamp_positive(x)

Return max(0, x) for numerical stability.
"""
@inline clamp_positive(x) = max(0, x)

"""
    safe_divide(a, b, default)

Safe division returning `default` when b ≈ 0.
All arguments must be positional (GPU kernel compatibility).
"""
@inline function safe_divide(a, b, default)
    FT = typeof(a)
    ε = eps(FT)
    return ifelse(abs(b) < ε, default, a / b)
end

# Convenience overload for common case
@inline safe_divide(a, b) = safe_divide(a, b, zero(a))

#####
##### Rain processes
#####

"""
    rain_autoconversion_rate(p3, qᶜˡ, Nᶜ)

Compute rain autoconversion rate following [Khairoutdinov and Kogan (2000)](@citet KhairoutdinovKogan2000).

Cloud droplets larger than a threshold undergo collision-coalescence to form rain.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜ`: Cloud droplet number concentration [1/m³]

# Returns
- Rate of cloud → rain conversion [kg/kg/s]
"""
@inline function rain_autoconversion_rate(p3, qᶜˡ, Nᶜ)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    # No autoconversion below threshold
    qᶜˡ_eff = clamp_positive(qᶜˡ - prp.autoconversion_threshold)

    # Scale droplet concentration
    Nᶜ_scaled = Nᶜ / prp.autoconversion_reference_concentration
    Nᶜ_scaled = max(Nᶜ_scaled, FT(0.01))

    # Khairoutdinov-Kogan (2000): ∂qʳ/∂t = k₁ × qᶜˡ^α × (Nᶜ/Nᶜ_ref)^β
    k₁ = prp.autoconversion_coefficient
    α = prp.autoconversion_exponent_cloud
    β = prp.autoconversion_exponent_droplet

    return k₁ * qᶜˡ_eff^α * Nᶜ_scaled^β
end

"""
    rain_accretion_rate(p3, qᶜˡ, qʳ)

Compute rain accretion rate following [Khairoutdinov and Kogan (2000)](@citet KhairoutdinovKogan2000).

Falling rain drops collect cloud droplets via gravitational sweep-out.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `qʳ`: Rain mass fraction [kg/kg]

# Returns
- Rate of cloud → rain conversion [kg/kg/s]
"""
@inline function rain_accretion_rate(p3, qᶜˡ, qʳ)
    prp = p3.process_rates

    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qʳ_eff = clamp_positive(qʳ)

    # KK2000: ∂qʳ/∂t = k₂ × (qᶜˡ × qʳ)^α
    k₂ = prp.accretion_coefficient
    α = prp.accretion_exponent

    return k₂ * (qᶜˡ_eff * qʳ_eff)^α
end

"""
    rain_self_collection_rate(p3, qʳ, nʳ, ρ)

Compute rain self-collection rate (number tendency only).

Large rain drops collect smaller ones, reducing number but conserving mass.
Follows [Seifert and Beheng (2001)](@citet SeifertBeheng2001).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Rate of rain number reduction [1/kg/s]
"""
@inline function rain_self_collection_rate(p3, qʳ, nʳ, ρ)
    prp = p3.process_rates

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # ∂nʳ/∂t = -k_rr × ρ × qʳ × nʳ
    k_rr = prp.self_collection_coefficient

    return -k_rr * ρ * qʳ_eff * nʳ_eff
end

"""
    rain_evaporation_rate(p3, qʳ, nʳ, qᵛ, qᵛ⁺ˡ, T, ρ)

Compute rain evaporation rate using ventilation-enhanced diffusion.

Rain drops evaporate when the ambient air is subsaturated (qᵛ < qᵛ⁺ˡ).
The evaporation rate is enhanced by ventilation (air flow around falling drops):

```math
\\frac{dm}{dt} = \\frac{4πD f_v (S - 1)}{\\frac{L_v}{K_a T}(\\frac{L_v}{R_v T} - 1) + \\frac{R_v T}{e_s D_v}}
```

where D is the drop diameter and f_v is the ventilation factor.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ˡ`: Saturation vapor mass fraction over liquid [kg/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]

# Returns
- Rate of rain → vapor conversion [kg/kg/s] (negative = evaporation)
"""
@inline function rain_evaporation_rate(p3, qʳ, nʳ, qᵛ, qᵛ⁺ˡ, T, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # Only evaporate in subsaturated conditions
    S = qᵛ / max(qᵛ⁺ˡ, FT(1e-10))
    is_subsaturated = S < 1

    # Thermodynamic constants
    R_v = FT(461.5)           # Gas constant for water vapor [J/kg/K]
    L_v = FT(2.5e6)           # Latent heat of vaporization [J/kg]
    K_a = FT(2.5e-2)          # Thermal conductivity of air [W/m/K]
    D_v = FT(2.5e-5)          # Diffusivity of water vapor [m²/s]

    # Saturation vapor pressure
    T₀ = prp.freezing_temperature
    e_s0 = FT(611)  # Pa at 273.15 K
    e_s = e_s0 * exp(L_v / R_v * (1 / T₀ - 1 / T))

    # Mean drop properties
    m_mean = safe_divide(qʳ_eff, nʳ_eff, FT(1e-12))
    ρ_water = p3.water_density
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_water))

    # Terminal velocity for rain drops (power law)
    V = FT(130) * D_mean^FT(0.5)  # Simplified Gunn-Kinzer

    # Ventilation factor
    ν = FT(1.5e-5)
    Re_term = sqrt(V * D_mean / ν)
    f_v = FT(0.78) + FT(0.31) * Re_term  # Different coefficients for drops

    # Thermodynamic resistance
    A = L_v / (K_a * T) * (L_v / (R_v * T) - 1)
    B = R_v * T / (e_s * D_v)
    thermodynamic_factor = A + B

    # Evaporation rate per drop (negative for evaporation)
    dm_dt = FT(4π) * (D_mean / 2) * f_v * (S - 1) / thermodynamic_factor

    # Total rate
    evap_rate = nʳ_eff * dm_dt

    # Cannot evaporate more than available
    τ_evap = prp.rain_evaporation_timescale
    max_evap = -qʳ_eff / τ_evap

    evap_rate = max(evap_rate, max_evap)

    return ifelse(is_subsaturated, evap_rate, zero(FT))
end

# Backward compatibility: simplified version without T, ρ
@inline function rain_evaporation_rate(p3, qʳ, qᵛ, qᵛ⁺ˡ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    qʳ_eff = clamp_positive(qʳ)
    τ_evap = prp.rain_evaporation_timescale

    # Subsaturation
    S = qᵛ - qᵛ⁺ˡ

    # Only evaporate in subsaturated conditions
    S_sub = min(S, zero(FT))

    # Relaxation toward saturation
    evap_rate = S_sub / τ_evap

    # Cannot evaporate more than available
    max_evap = -qʳ_eff / τ_evap

    return max(evap_rate, max_evap)
end

#####
##### Ice deposition and sublimation
#####

"""
    ice_deposition_rate(p3, qⁱ, qᵛ, qᵛ⁺ⁱ)

Compute ice deposition/sublimation rate.

Ice grows by vapor deposition when supersaturated with respect to ice,
and sublimates when subsaturated.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ⁱ`: Saturation vapor mass fraction over ice [kg/kg]

# Returns
- Rate of vapor → ice conversion [kg/kg/s] (positive = deposition)
"""
@inline function ice_deposition_rate(p3, qⁱ, qᵛ, qᵛ⁺ⁱ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    τ_dep = prp.ice_deposition_timescale

    # Supersaturation with respect to ice
    Sⁱ = qᵛ - qᵛ⁺ⁱ

    # Relaxation toward saturation
    dep_rate = Sⁱ / τ_dep

    # Limit sublimation to available ice
    is_sublimation = Sⁱ < 0
    max_sublim = -qⁱ_eff / τ_dep

    return ifelse(is_sublimation, max(dep_rate, max_sublim), dep_rate)
end

"""
    ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P)

Compute ventilation-enhanced ice deposition/sublimation rate.

Following Morrison & Milbrandt (2015a) Eq. 30, the deposition rate is:

```math
\\frac{dm}{dt} = \\frac{4πC f_v (S_i - 1)}{\\frac{L_s}{K_a T}(\\frac{L_s}{R_v T} - 1) + \\frac{R_v T}{e_{si} D_v}}
```

where f_v is the ventilation factor and C is the capacitance.

The bulk rate integrates over the size distribution:

```math
\\frac{dq^i}{dt} = ∫ \\frac{dm}{dt}(D) N'(D) dD
```

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ⁱ`: Saturation vapor mass fraction over ice [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `T`: Temperature [K]
- `P`: Pressure [Pa]

# Returns
- Rate of vapor → ice conversion [kg/kg/s] (positive = deposition)
"""
@inline function ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # Thermodynamic constants
    R_v = FT(461.5)           # Gas constant for water vapor [J/kg/K]
    L_s = FT(2.835e6)         # Latent heat of sublimation [J/kg]
    K_a = FT(2.5e-2)          # Thermal conductivity of air [W/m/K]
    D_v = FT(2.5e-5)          # Diffusivity of water vapor [m²/s]

    # Saturation vapor pressure over ice (simplified Clausius-Clapeyron)
    T₀ = prp.freezing_temperature
    e_si0 = FT(611)  # Pa at 273.15 K
    e_si = e_si0 * exp(L_s / R_v * (1 / T₀ - 1 / T))

    # Supersaturation ratio with respect to ice
    S_i = qᵛ / max(qᵛ⁺ⁱ, FT(1e-10))

    # Mean particle mass
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # Effective density depends on riming
    ρⁱ = prp.pure_ice_density
    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    ρ_eff = (1 - Fᶠ) * ρ_eff_unrimed + Fᶠ * ρᶠ

    # Mean diameter
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_eff))

    # Capacitance (regime-dependent)
    D_threshold = prp.ice_diameter_threshold
    C = ifelse(D_mean < D_threshold, D_mean / 2, FT(0.48) * D_mean)

    # Ventilation factor: f_v = a + b × Re^(1/2) × Sc^(1/3)
    # Simplified: f_v ≈ 0.65 + 0.44 × √(V × D / ν)
    ν = FT(1.5e-5)  # kinematic viscosity [m²/s]
    # Estimate terminal velocity (simplified power law)
    V = FT(11.72) * D_mean^FT(0.41)
    Re_term = sqrt(V * D_mean / ν)
    f_v = FT(0.65) + FT(0.44) * Re_term

    # Denominator: thermodynamic resistance terms
    # A = L_s/(K_a × T) × (L_s/(R_v × T) - 1)
    # B = R_v × T / (e_si × D_v)
    A = L_s / (K_a * T) * (L_s / (R_v * T) - 1)
    B = R_v * T / (e_si * D_v)
    thermodynamic_factor = A + B

    # Deposition rate per particle (Eq. 30 from MM15a)
    dm_dt = FT(4π) * C * f_v * (S_i - 1) / thermodynamic_factor

    # Total rate
    dep_rate = nⁱ_eff * dm_dt

    # Limit sublimation to available ice
    τ_dep = prp.ice_deposition_timescale
    is_sublimation = S_i < 1
    max_sublim = -qⁱ_eff / τ_dep

    return ifelse(is_sublimation, max(dep_rate, max_sublim), dep_rate)
end

# Backward compatibility: version without T, P uses simplified form
@inline function ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ)
    FT = typeof(qⁱ)
    # Use default T = 250 K, P = 50000 Pa for backward compatibility
    return ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, FT(250), FT(50000))
end

#####
##### Melting
#####

"""
    ice_melting_rate(p3, qⁱ, nⁱ, T, qᵛ, qᵛ⁺, Fᶠ, ρᶠ)

Compute ice melting rate using the heat balance equation from
Morrison & Milbrandt (2015a) Eq. 44.

The melting rate is determined by the heat flux to the particle:

```math
\\frac{dm}{dt} = -\\frac{4πC}{L_f} × [K_a(T-T_0) + L_v D_v(ρ_v - ρ_{vs})] × f_v
```

where:
- C is the capacitance
- L_f is the latent heat of fusion
- K_a is thermal conductivity of air
- T_0 is the freezing temperature
- L_v is latent heat of vaporization
- D_v is diffusivity of water vapor
- ρ_v, ρ_vs are vapor density and saturation vapor density
- f_v is the ventilation factor

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺`: Saturation vapor mass fraction over liquid [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]

# Returns
- Rate of ice → rain conversion [kg/kg/s]
"""
@inline function ice_melting_rate(p3, qⁱ, nⁱ, T, qᵛ, qᵛ⁺, Fᶠ, ρᶠ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    T₀ = prp.freezing_temperature

    # Only melt above freezing
    ΔT = T - T₀
    is_melting = ΔT > 0

    # Thermodynamic constants
    L_f = FT(3.34e5)          # Latent heat of fusion [J/kg]
    L_v = FT(2.5e6)           # Latent heat of vaporization [J/kg]
    K_a = FT(2.5e-2)          # Thermal conductivity of air [W/m/K]
    D_v = FT(2.5e-5)          # Diffusivity of water vapor [m²/s]
    R_v = FT(461.5)           # Gas constant for water vapor [J/kg/K]

    # Vapor density terms
    # At T₀, ρ_vs corresponds to saturation at melting point
    e_s0 = FT(611)  # Saturation vapor pressure at 273.15 K [Pa]
    P_atm = FT(1e5)  # Reference pressure [Pa]
    ρ_vs = e_s0 / (R_v * T₀)  # Saturation vapor density at T₀

    # Ambient vapor density (from mixing ratio)
    ρ_air = P_atm / (FT(287) * T)  # Approximate air density
    ρ_v = qᵛ * ρ_air

    # Mean particle properties
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # Effective density
    ρⁱ = prp.pure_ice_density
    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    ρ_eff = (1 - Fᶠ) * ρ_eff_unrimed + Fᶠ * ρᶠ

    # Mean diameter
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_eff))

    # Capacitance
    D_threshold = prp.ice_diameter_threshold
    C = ifelse(D_mean < D_threshold, D_mean / 2, FT(0.48) * D_mean)

    # Ventilation factor
    ν = FT(1.5e-5)
    V = FT(11.72) * D_mean^FT(0.41)
    Re_term = sqrt(V * D_mean / ν)
    f_v = FT(0.65) + FT(0.44) * Re_term

    # Heat flux terms (Eq. 44 from MM15a)
    # Sensible heat: K_a × (T - T₀)
    Q_sensible = K_a * ΔT

    # Latent heat: L_v × D_v × (ρ_v - ρ_vs)
    # When subsaturated, this is negative and opposes melting
    Q_latent = L_v * D_v * (ρ_v - ρ_vs)

    # Total heat flux
    Q_total = Q_sensible + Q_latent

    # Melting rate per particle (negative dm/dt → positive melt rate)
    dm_dt_melt = FT(4π) * C * f_v * Q_total / L_f

    # Clamp to positive (only melting, not refreezing here)
    dm_dt_melt = clamp_positive(dm_dt_melt)

    # Total rate
    melt_rate = nⁱ_eff * dm_dt_melt

    # Limit to available ice
    τ_melt = prp.ice_melting_timescale
    max_melt = qⁱ_eff / τ_melt

    melt_rate = min(melt_rate, max_melt)

    return ifelse(is_melting, melt_rate, zero(FT))
end

# Backward compatibility: simplified version
@inline function ice_melting_rate(p3, qⁱ, T)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    T₀ = prp.freezing_temperature
    τ_melt = prp.ice_melting_timescale

    # Temperature excess above freezing
    ΔT = T - T₀
    ΔT_pos = clamp_positive(ΔT)

    # Melting rate proportional to temperature excess (normalized to 1K)
    rate_factor = ΔT_pos

    return qⁱ_eff * rate_factor / τ_melt
end

"""
    ice_melting_number_rate(qⁱ, nⁱ, qⁱ_melt_rate)

Compute ice number tendency from melting.

Number of melted particles equals number of rain drops produced.

# Arguments
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `qⁱ_melt_rate`: Ice mass melting rate [kg/kg/s]

# Returns
- Rate of ice number reduction [1/kg/s]
"""
@inline function ice_melting_number_rate(qⁱ, nⁱ, qⁱ_melt_rate)
    FT = typeof(qⁱ)

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # ∂nⁱ/∂t = (nⁱ/qⁱ) × ∂qⁱ_melt/∂t
    ratio = safe_divide(nⁱ_eff, qⁱ_eff, zero(FT))

    return -ratio * qⁱ_melt_rate
end

#####
##### Ice nucleation (deposition and immersion freezing)
#####

"""
    deposition_nucleation_rate(p3, T, qᵛ, qᵛ⁺ⁱ, nⁱ, ρ)

Compute ice nucleation rate from deposition/condensation freezing.

New ice crystals nucleate when temperature is below a threshold and the air
is supersaturated with respect to ice. Uses [Cooper (1986)](@citet Cooper1986).

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
    nucleation_active = (T < T_threshold) && (Sⁱ > Sⁱ_threshold)

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
    N_nuc = ifelse(nucleation_active && N_nuc > FT(1e-20), N_nuc, zero(FT))
    Q_nuc = ifelse(nucleation_active && Q_nuc > FT(1e-30), Q_nuc, zero(FT))

    return Q_nuc, N_nuc
end

"""
    immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T)

Compute immersion freezing rate of cloud droplets.

Cloud droplets freeze when temperature is below a threshold. Uses
[Bigg (1953)](@citet Bigg1953) stochastic freezing parameterization.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜ`: Cloud droplet number concentration [1/m³]
- `T`: Temperature [K]

# Returns
- Tuple (Q_frz, N_frz): mass rate [kg/kg/s] and number rate [1/kg/s]
"""
@inline function immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    T_max = prp.immersion_freezing_temperature_max
    aimm = prp.immersion_freezing_coefficient
    τ_base = prp.immersion_freezing_timescale_cloud
    T₀ = prp.freezing_temperature

    qᶜˡ_eff = clamp_positive(qᶜˡ)

    # Conditions for freezing
    freezing_active = (T < T_max) && (qᶜˡ_eff > FT(1e-8))

    # Bigg (1953): J = exp(aimm × (T₀ - T))
    ΔT = T₀ - T
    J = exp(aimm * ΔT)

    # Timescale decreases as J increases
    τ_frz = τ_base / max(J, FT(1))

    # Freezing rate
    N_frz = ifelse(freezing_active, Nᶜ / τ_frz, zero(FT))
    Q_frz = ifelse(freezing_active, qᶜˡ_eff / τ_frz, zero(FT))

    return Q_frz, N_frz
end

"""
    immersion_freezing_rain_rate(p3, qʳ, nʳ, T)

Compute immersion freezing rate of rain drops.

Rain drops freeze when temperature is below a threshold. Uses
[Bigg (1953)](@citet Bigg1953) stochastic freezing parameterization.

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
    τ_base = prp.immersion_freezing_timescale_rain
    T₀ = prp.freezing_temperature

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # Conditions for freezing
    freezing_active = (T < T_max) && (qʳ_eff > FT(1e-8))

    # Bigg (1953)
    ΔT = T₀ - T
    J = exp(aimm * ΔT)

    # Rain freezes faster due to larger volume
    τ_frz = τ_base / max(J, FT(1))

    # Freezing rate
    N_frz = ifelse(freezing_active, nʳ_eff / τ_frz, zero(FT))
    Q_frz = ifelse(freezing_active, qʳ_eff / τ_frz, zero(FT))

    return Q_frz, N_frz
end

"""
    contact_freezing_rate(p3, qᶜˡ, Nᶜ, T, N_IN)

Compute contact freezing nucleation rate.

Contact freezing occurs when ice nuclei (IN) collide with supercooled droplets.
This is often a more efficient ice nucleation mechanism than deposition
at temperatures warmer than -15°C.

The rate is proportional to:
- IN concentration (N_IN)
- Cloud droplet surface area (∝ D² × N_cloud)
- Collection efficiency (Brownian + phoretic)

Following [Cotton et al. (1986)](@cite CottonEtAl1986) and [Meyers et al. (1992)](@cite MeyersEtAl1992):

```math
\\frac{dN^i}{dt} = 4π D_c^2 N_c N_{IN} D_{IN} (1 + 0.4 Re^{0.5} Sc^{0.33})
```

where D_IN is the IN diffusivity and the parenthetical term is the
phoretic enhancement.

# Arguments
- `p3`: P3 microphysics scheme
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜ`: Cloud droplet number concentration [1/m³]
- `T`: Temperature [K]
- `N_IN`: Ice nuclei concentration [1/m³] (optional, defaults to Meyers parameterization)

# Returns
- Tuple (Q_frz, N_frz): mass rate [kg/kg/s] and number rate [1/kg/s]
"""
@inline function contact_freezing_rate(p3, qᶜˡ, Nᶜ, T, N_IN)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    T₀ = prp.freezing_temperature
    T_max = FT(268)  # Contact freezing inactive above -5°C

    qᶜˡ_eff = clamp_positive(qᶜˡ)

    # Conditions for contact freezing
    freezing_active = (T < T₀) && (T < T_max) && (qᶜˡ_eff > FT(1e-8))

    # Cloud droplet properties
    ρ_water = p3.water_density
    # Mean cloud droplet diameter (from cloud properties)
    m_drop = qᶜˡ_eff / max(Nᶜ, FT(1e6))
    D_c = cbrt(6 * m_drop / (FT(π) * ρ_water))
    D_c = clamp(D_c, FT(5e-6), FT(50e-6))

    # IN diffusivity (approximately Brownian for submicron particles)
    # D_IN ~ k_B T / (3 π μ D_IN_particle) ~ 2e-11 m²/s for 0.5 μm particles
    D_IN = FT(2e-11)

    # Contact kernel: K = 4π D_c² D_IN × ventilation_factor
    # Simplified ventilation factor for cloud droplets (small Re)
    vent_factor = FT(1.2)

    K_contact = FT(4π) * D_c^2 * D_IN * vent_factor

    # Freezing rate
    N_frz = K_contact * Nᶜ * N_IN

    # Mass rate: each frozen droplet becomes ice of same mass
    Q_frz = m_drop * N_frz

    # Apply conditions
    N_frz = ifelse(freezing_active, N_frz, zero(FT))
    Q_frz = ifelse(freezing_active, Q_frz, zero(FT))

    return Q_frz, N_frz
end

# Version with Meyers IN parameterization
@inline function contact_freezing_rate(p3, qᶜˡ, Nᶜ, T)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates
    T₀ = prp.freezing_temperature

    # Meyers et al. (1992) IN parameterization (contact nuclei)
    # N_IN = exp(-2.80 - 0.262 × (T₀ - T)) per liter
    ΔT = T₀ - T
    ΔT_clamped = clamp(ΔT, FT(0), FT(40))
    N_IN = exp(FT(-2.80) - FT(0.262) * ΔT_clamped) * FT(1000)  # per m³

    return contact_freezing_rate(p3, qᶜˡ, Nᶜ, T, N_IN)
end

#####
##### Rime splintering (Hallett-Mossop secondary ice production)
#####

"""
    rime_splintering_rate(p3, cloud_riming, rain_riming, T)

Compute secondary ice production from rime splintering (Hallett-Mossop effect).

When rimed ice particles accrete supercooled drops, ice splinters are
ejected. This occurs only in a narrow temperature range around -5°C.
See [Hallett and Mossop (1974)](@citet HallettMossop1974).

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
    in_HM_window = (T > T_low) && (T < T_high)

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

    aggregation_active = (qⁱ_eff > qⁱ_threshold) && (nⁱ_eff > nⁱ_threshold)

    # Temperature-dependent sticking efficiency (linear ramp)
    # Cold ice is less sticky, near-melting ice is very sticky
    Eᵢᵢ_cold = FT(0.1)
    Eᵢᵢ = ifelse(T < T_low, Eᵢᵢ_cold,
                  ifelse(T > T_high, Eᵢᵢ_max,
                         Eᵢᵢ_cold + (T - T_low) / (T_high - T_low) * (Eᵢᵢ_max - Eᵢᵢ_cold)))

    # Mean particle properties
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # Effective density
    ρⁱ = prp.pure_ice_density
    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    ρ_eff = (1 - Fᶠ) * ρ_eff_unrimed + Fᶠ * ρᶠ

    # Mean diameter
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_eff))

    # Mean terminal velocity (regime-dependent approximation)
    a_V_unrimed = FT(11.72)
    b_V_unrimed = FT(0.41)
    a_V_rimed = FT(19.3)
    b_V_rimed = FT(0.37)
    a_V = (1 - Fᶠ) * a_V_unrimed + Fᶠ * a_V_rimed
    b_V = (1 - Fᶠ) * b_V_unrimed + Fᶠ * b_V_rimed
    V_mean = a_V * D_mean^b_V

    # Mean projected area (regime-dependent)
    γ = FT(0.2285)
    σ = FT(1.88)
    A_aggregate = γ * D_mean^σ
    A_sphere = FT(π) / 4 * D_mean^2
    A_mean = (1 - Fᶠ) * A_aggregate + Fᶠ * A_sphere

    # Self-collection kernel approximation:
    # K ≈ E_ii × A_mean × ΔV, where ΔV ≈ 0.5 × V_mean for self-collection
    ΔV = FT(0.5) * V_mean
    K_mean = Eᵢᵢ * A_mean * ΔV

    # Number tendency: dn/dt = -0.5 × K × n²
    rate = -FT(0.5) * K_mean * nⁱ_eff^2

    return ifelse(aggregation_active, rate, zero(FT))
end

# Backward compatibility: simplified version without rime properties
@inline function ice_aggregation_rate(p3, qⁱ, nⁱ, T)
    FT = typeof(qⁱ)
    return ice_aggregation_rate(p3, qⁱ, nⁱ, T, zero(FT), FT(400))
end

#####
##### Phase 2: Riming (cloud and rain collection by ice)
#####

"""
    cloud_riming_rate(p3, qᶜˡ, qⁱ, T)

Compute cloud droplet collection (riming) by ice particles.

Cloud droplets are swept up by falling ice particles and freeze onto them.
This increases ice mass and rime mass.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `T`: Temperature [K]

# Returns
- Rate of cloud → ice conversion [kg/kg/s] (also equals rime mass gain rate)
"""
@inline function cloud_riming_rate(p3, qᶜˡ, qⁱ, T)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    Eᶜⁱ = prp.cloud_ice_collection_efficiency
    τ_rim = prp.cloud_riming_timescale
    T₀ = prp.freezing_temperature

    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qⁱ_eff = clamp_positive(qⁱ)

    # Thresholds
    q_threshold = FT(1e-8)

    # Only rime below freezing
    below_freezing = T < T₀

    # ∂qᶜˡ/∂t = -Eᶜⁱ × qᶜˡ × qⁱ / τ_rim
    rate = ifelse(below_freezing && qᶜˡ_eff > q_threshold && qⁱ_eff > q_threshold,
                   Eᶜⁱ * qᶜˡ_eff * qⁱ_eff / τ_rim,
                   zero(FT))

    return rate
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
    rain_riming_rate(p3, qʳ, qⁱ, T)

Compute rain collection (riming) by ice particles.

Rain drops are swept up by falling ice particles and freeze onto them.
This increases ice mass and rime mass.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `T`: Temperature [K]

# Returns
- Rate of rain → ice conversion [kg/kg/s] (also equals rime mass gain rate)
"""
@inline function rain_riming_rate(p3, qʳ, qⁱ, T)
    FT = typeof(qʳ)
    prp = p3.process_rates

    Eʳⁱ = prp.rain_ice_collection_efficiency
    τ_rim = prp.rain_riming_timescale
    T₀ = prp.freezing_temperature

    qʳ_eff = clamp_positive(qʳ)
    qⁱ_eff = clamp_positive(qⁱ)

    # Thresholds
    q_threshold = FT(1e-8)

    # Only rime below freezing
    below_freezing = T < T₀

    rate = ifelse(below_freezing && qʳ_eff > q_threshold && qⁱ_eff > q_threshold,
                   Eʳⁱ * qʳ_eff * qⁱ_eff / τ_rim,
                   zero(FT))

    return rate
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
    is_wet_growth = (Tc > FT(-10)) && (lwc > FT(0.5e-3))
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
See [Milbrandt et al. (2025)](@citet MilbrandtEtAl2025liquidfraction).

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
See [Milbrandt et al. (2025)](@citet MilbrandtEtAl2025liquidfraction).

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

    rate = ifelse(below_freezing && qʷⁱ_eff > FT(1e-10),
                   T_factor * qʷⁱ_eff / τ_frz,
                   zero(FT))

    return rate
end

#####
##### Combined P3 tendency calculation
#####

"""
    P3ProcessRates

Container for computed P3 process rates.
Includes Phase 1 (rain, deposition, melting), Phase 2 (aggregation, riming, shedding, nucleation).
"""
struct P3ProcessRates{FT}
    # Phase 1: Rain tendencies
    autoconversion :: FT           # Cloud → rain mass [kg/kg/s]
    accretion :: FT                # Cloud → rain mass (via rain sweep-out) [kg/kg/s]
    rain_evaporation :: FT         # Rain → vapor mass [kg/kg/s]
    rain_self_collection :: FT     # Rain number reduction [1/kg/s]

    # Phase 1: Ice tendencies
    deposition :: FT               # Vapor → ice mass [kg/kg/s]
    melting :: FT                  # Ice → rain mass [kg/kg/s]
    melting_number :: FT           # Ice number reduction from melting [1/kg/s]

    # Phase 2: Ice aggregation
    aggregation :: FT              # Ice number reduction from self-collection [1/kg/s]

    # Phase 2: Riming
    cloud_riming :: FT             # Cloud → ice via riming [kg/kg/s]
    cloud_riming_number :: FT      # Cloud number reduction [1/kg/s]
    rain_riming :: FT              # Rain → ice via riming [kg/kg/s]
    rain_riming_number :: FT       # Rain number reduction [1/kg/s]
    rime_density_new :: FT         # Density of new rime [kg/m³]

    # Phase 2: Shedding and refreezing
    shedding :: FT                 # Liquid on ice → rain [kg/kg/s]
    shedding_number :: FT          # Rain number from shedding [1/kg/s]
    refreezing :: FT               # Liquid on ice → rime [kg/kg/s]

    # Ice nucleation (deposition + immersion freezing)
    nucleation_mass :: FT          # New ice mass from deposition nucleation [kg/kg/s]
    nucleation_number :: FT        # New ice number from deposition nucleation [1/kg/s]
    cloud_freezing_mass :: FT      # Cloud → ice mass from immersion freezing [kg/kg/s]
    cloud_freezing_number :: FT    # Cloud number to ice number [1/kg/s]
    rain_freezing_mass :: FT       # Rain → ice mass from immersion freezing [kg/kg/s]
    rain_freezing_number :: FT     # Rain number to ice number [1/kg/s]

    # Rime splintering (Hallett-Mossop)
    splintering_mass :: FT         # New ice mass from splintering [kg/kg/s]
    splintering_number :: FT       # New ice number from splintering [1/kg/s]
end

"""
    compute_p3_process_rates(i, j, k, grid, p3, μ, ρ, 𝒰, constants)

Compute all P3 process rates (Phase 1 and Phase 2).

# Arguments
- `i, j, k`: Grid indices
- `grid`: Computational grid
- `p3`: P3 microphysics scheme
- `μ`: Microphysical fields (prognostic and diagnostic)
- `ρ`: Air density [kg/m³]
- `𝒰`: Thermodynamic state
- `constants`: Thermodynamic constants

# Returns
- `P3ProcessRates` containing all computed rates
"""
@inline function compute_p3_process_rates(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    FT = eltype(grid)
    prp = p3.process_rates
    T₀ = prp.freezing_temperature

    # Extract fields (density-weighted → specific)
    qᶜˡ = @inbounds μ.ρqᶜˡ[i, j, k] / ρ
    qʳ = @inbounds μ.ρqʳ[i, j, k] / ρ
    nʳ = @inbounds μ.ρnʳ[i, j, k] / ρ
    qⁱ = @inbounds μ.ρqⁱ[i, j, k] / ρ
    nⁱ = @inbounds μ.ρnⁱ[i, j, k] / ρ
    qᶠ = @inbounds μ.ρqᶠ[i, j, k] / ρ
    bᶠ = @inbounds μ.ρbᶠ[i, j, k] / ρ
    qʷⁱ = @inbounds μ.ρqʷⁱ[i, j, k] / ρ

    # Rime properties
    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))
    ρᶠ = safe_divide(qᶠ, bᶠ, FT(400))

    # Thermodynamic state
    T = temperature(𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor

    # Saturation vapor mixing ratios (simplified Clausius-Clapeyron)
    # TODO: Replace with proper thermodynamic interface
    eₛ_liquid = FT(611.2) * exp(FT(17.67) * (T - T₀) / (T - FT(29.65)))
    eₛ_ice = FT(611.2) * exp(FT(21.87) * (T - T₀) / (T - FT(7.66)))

    Rᵈ = FT(287.0)
    Rᵛ = FT(461.5)
    ε = Rᵈ / Rᵛ
    p = ρ * Rᵈ * T
    qᵛ⁺ˡ = ε * eₛ_liquid / (p - (1 - ε) * eₛ_liquid)
    qᵛ⁺ⁱ = ε * eₛ_ice / (p - (1 - ε) * eₛ_ice)

    # Cloud droplet number concentration
    Nᶜ = p3.cloud.number_concentration

    # =========================================================================
    # Phase 1: Rain processes
    # =========================================================================
    autoconv = rain_autoconversion_rate(p3, qᶜˡ, Nᶜ)
    accr = rain_accretion_rate(p3, qᶜˡ, qʳ)
    rain_evap = rain_evaporation_rate(p3, qʳ, qᵛ, qᵛ⁺ˡ)
    rain_self = rain_self_collection_rate(p3, qʳ, nʳ, ρ)

    # =========================================================================
    # Phase 1: Ice deposition/sublimation and melting
    # =========================================================================
    dep = ice_deposition_rate(p3, qⁱ, qᵛ, qᵛ⁺ⁱ)
    melt = ice_melting_rate(p3, qⁱ, T)
    melt_n = ice_melting_number_rate(qⁱ, nⁱ, melt)

    # =========================================================================
    # Phase 2: Ice aggregation
    # =========================================================================
    agg = ice_aggregation_rate(p3, qⁱ, nⁱ, T)

    # =========================================================================
    # Phase 2: Riming
    # =========================================================================
    cloud_rim = cloud_riming_rate(p3, qᶜˡ, qⁱ, T)
    cloud_rim_n = cloud_riming_number_rate(qᶜˡ, Nᶜ, cloud_rim)

    rain_rim = rain_riming_rate(p3, qʳ, qⁱ, T)
    rain_rim_n = rain_riming_number_rate(qʳ, nʳ, rain_rim)

    # Rime density for new rime
    vᵢ = FT(1)  # Placeholder fall speed [m/s]
    ρᶠ_new = rime_density(p3, T, vᵢ)

    # =========================================================================
    # Phase 2: Shedding and refreezing
    # =========================================================================
    shed = shedding_rate(p3, qʷⁱ, qⁱ, T)
    shed_n = shedding_number_rate(p3, shed)
    refrz = refreezing_rate(p3, qʷⁱ, T)

    # =========================================================================
    # Ice nucleation (deposition nucleation and immersion freezing)
    # =========================================================================
    nuc_q, nuc_n = deposition_nucleation_rate(p3, T, qᵛ, qᵛ⁺ⁱ, nⁱ, ρ)
    cloud_frz_q, cloud_frz_n = immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T)
    rain_frz_q, rain_frz_n = immersion_freezing_rain_rate(p3, qʳ, nʳ, T)

    # =========================================================================
    # Rime splintering (Hallett-Mossop secondary ice production)
    # =========================================================================
    spl_q, spl_n = rime_splintering_rate(p3, cloud_rim, rain_rim, T)

    return P3ProcessRates(
        # Phase 1: Rain
        autoconv, accr, rain_evap, rain_self,
        # Phase 1: Ice
        dep, melt, melt_n,
        # Phase 2: Aggregation
        agg,
        # Phase 2: Riming
        cloud_rim, cloud_rim_n, rain_rim, rain_rim_n, ρᶠ_new,
        # Phase 2: Shedding and refreezing
        shed, shed_n, refrz,
        # Ice nucleation
        nuc_q, nuc_n, cloud_frz_q, cloud_frz_n, rain_frz_q, rain_frz_n,
        # Rime splintering
        spl_q, spl_n
    )
end

#####
##### Individual field tendencies
#####
##### These functions combine process rates into tendencies for each prognostic field.
##### Phase 1 processes: autoconversion, accretion, evaporation, deposition, melting
##### Phase 2 processes: aggregation, riming, shedding, refreezing
#####

"""
    tendency_ρqᶜˡ(rates)

Compute cloud liquid mass tendency from P3 process rates.

Cloud liquid is consumed by:
- Autoconversion (Phase 1)
- Accretion by rain (Phase 1)
- Riming by ice (Phase 2)
- Immersion freezing (Phase 2)
"""
@inline function tendency_ρqᶜˡ(rates::P3ProcessRates, ρ)
    # Phase 1: autoconversion and accretion
    # Phase 2: cloud riming by ice, immersion freezing
    loss = rates.autoconversion + rates.accretion + rates.cloud_riming + rates.cloud_freezing_mass
    return -ρ * loss
end

"""
    tendency_ρqʳ(rates)

Compute rain mass tendency from P3 process rates.

Rain gains from:
- Autoconversion (Phase 1)
- Accretion (Phase 1)
- Melting (Phase 1)
- Shedding (Phase 2)

Rain loses from:
- Evaporation (Phase 1)
- Riming (Phase 2)
- Immersion freezing (Phase 2)
"""
@inline function tendency_ρqʳ(rates::P3ProcessRates, ρ)
    # Phase 1: gains from autoconv, accr, melt; loses from evap
    # Phase 2: gains from shedding; loses from riming and freezing
    gain = rates.autoconversion + rates.accretion + rates.melting + rates.shedding
    loss = -rates.rain_evaporation + rates.rain_riming + rates.rain_freezing_mass  # evap is negative
    return ρ * (gain - loss)
end

"""
    tendency_ρnʳ(rates, ρ, qᶜˡ, Nc, m_drop)

Compute rain number tendency from P3 process rates.

Rain number gains from:
- Autoconversion (Phase 1)
- Melting (Phase 1)
- Shedding (Phase 2)

Rain number loses from:
- Self-collection (Phase 1)
- Riming (Phase 2)
- Immersion freezing (Phase 2)
"""
@inline function tendency_ρnʳ(rates::P3ProcessRates, ρ, nⁱ, qⁱ;
                               m_rain_init = 5e-10)  # Initial rain drop mass [kg]
    FT = typeof(ρ)

    # Phase 1: New drops from autoconversion
    n_from_autoconv = rates.autoconversion / m_rain_init

    # Phase 1: New drops from melting (conserve number)
    n_from_melt = safe_divide(nⁱ * rates.melting, qⁱ, zero(FT))

    # Phase 1: Self-collection reduces number (already negative)
    # Phase 2: Shedding creates new drops
    # Phase 2: Riming removes rain drops (already negative)

    return ρ * (n_from_autoconv + n_from_melt +
                rates.rain_self_collection +
                rates.shedding_number +
                rates.rain_riming_number)
end

"""
    tendency_ρqⁱ(rates)

Compute ice mass tendency from P3 process rates.

Ice gains from:
- Deposition (Phase 1)
- Cloud riming (Phase 2)
- Rain riming (Phase 2)
- Refreezing (Phase 2)
- Deposition nucleation (Phase 2)
- Immersion freezing of cloud/rain (Phase 2)
- Rime splintering (Phase 2)

Ice loses from:
- Melting (Phase 1)
"""
@inline function tendency_ρqⁱ(rates::P3ProcessRates, ρ)
    # Phase 1: deposition, melting
    # Phase 2: riming (cloud + rain), refreezing, nucleation, freezing, splintering
    gain = rates.deposition + rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.nucleation_mass + rates.cloud_freezing_mass + rates.rain_freezing_mass +
           rates.splintering_mass
    loss = rates.melting
    return ρ * (gain - loss)
end

"""
    tendency_ρnⁱ(rates)

Compute ice number tendency from P3 process rates.

Ice number gains from:
- Deposition nucleation (Phase 2)
- Immersion freezing of cloud/rain (Phase 2)
- Rime splintering (Phase 2)

Ice number loses from:
- Melting (Phase 1)
- Aggregation (Phase 2)
"""
@inline function tendency_ρnⁱ(rates::P3ProcessRates, ρ)
    # Gains from nucleation, freezing, splintering
    gain = rates.nucleation_number + rates.cloud_freezing_number +
           rates.rain_freezing_number + rates.splintering_number
    # melting_number and aggregation are already negative (represent losses)
    loss_rates = rates.melting_number + rates.aggregation
    return ρ * (gain + loss_rates)
end

"""
    tendency_ρqᶠ(rates)

Compute rime mass tendency from P3 process rates.

Rime mass gains from:
- Cloud riming (Phase 2)
- Rain riming (Phase 2)
- Refreezing (Phase 2)
- Immersion freezing (frozen cloud/rain becomes rimed ice) (Phase 2)

Rime mass loses from:
- Melting (proportional to rime fraction) (Phase 1)
"""
@inline function tendency_ρqᶠ(rates::P3ProcessRates, ρ, Fᶠ)
    # Phase 2: gains from riming, refreezing, and freezing
    # Frozen cloud/rain becomes fully rimed ice (100% rime fraction for new frozen particles)
    gain = rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.cloud_freezing_mass + rates.rain_freezing_mass
    # Phase 1: melts proportionally with ice mass
    loss = Fᶠ * rates.melting
    return ρ * (gain - loss)
end

"""
    tendency_ρbᶠ(rates, Fᶠ, ρᶠ)

Compute rime volume tendency from P3 process rates.

Rime volume changes with rime mass: ∂bᶠ/∂t = ∂qᶠ/∂t / ρ_rime
"""
@inline function tendency_ρbᶠ(rates::P3ProcessRates, ρ, Fᶠ, ρᶠ)
    FT = typeof(ρ)

    ρᶠ_safe = max(ρᶠ, FT(100))
    ρ_rim_new_safe = max(rates.rime_density_new, FT(100))

    # Phase 2: Volume gain from new rime (cloud + rain riming + refreezing)
    # Use density of new rime for fresh rime, current density for refreezing
    volume_gain = (rates.cloud_riming + rates.rain_riming) / ρ_rim_new_safe +
                   rates.refreezing / ρᶠ_safe

    # Phase 1: Volume loss from melting (proportional to rime fraction)
    volume_loss = Fᶠ * rates.melting / ρᶠ_safe

    return ρ * (volume_gain - volume_loss)
end

"""
    tendency_ρzⁱ(rates, μ, λ)

Compute ice sixth moment tendency from P3 process rates.

The sixth moment (reflectivity) changes with:
- Deposition (growth) (Phase 1)
- Melting (loss) (Phase 1)
- Riming (growth) (Phase 2)
- Nucleation (growth) (Phase 2)
- Aggregation (redistribution) (Phase 2)

For P3 3-moment, Z tendencies are computed more accurately using
size distribution integrals. This simplified version uses proportional scaling.
"""
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ)
    FT = typeof(ρ)

    # Simplified: Z changes proportionally to mass changes
    # More accurate version would use full integral formulation
    ratio = safe_divide(zⁱ, qⁱ, zero(FT))

    # Net mass change for ice
    mass_change = rates.deposition - rates.melting +
                  rates.cloud_riming + rates.rain_riming + rates.refreezing

    return ρ * ratio * mass_change
end

"""
    tendency_ρqʷⁱ(rates)

Compute liquid on ice tendency from P3 process rates.

Liquid on ice:
- Gains from partial melting above freezing (currently in melting rate)
- Loses from shedding (Phase 2)
- Loses from refreezing (Phase 2)
"""
@inline function tendency_ρqʷⁱ(rates::P3ProcessRates, ρ)
    # Phase 2: loses from shedding and refreezing
    # Gains: In full P3, partial melting above freezing adds to qʷⁱ
    # For now, melting goes directly to rain; this is a placeholder
    return -ρ * (rates.shedding + rates.refreezing)
end

#####
##### Phase 3: Terminal velocities
#####
##### Terminal velocity calculations for rain and ice sedimentation.
##### Uses power-law relationships with air density correction.
#####

"""
    rain_terminal_velocity_mass_weighted(p3, qʳ, nʳ, ρ)

Compute mass-weighted terminal velocity for rain.

Uses the power-law relationship v(D) = a × D^b × √(ρ₀/ρ).
See [Seifert and Beheng (2006)](@citet SeifertBeheng2006).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Mass-weighted fall speed [m/s] (positive downward)
"""
@inline function rain_terminal_velocity_mass_weighted(p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    a = prp.rain_fall_speed_coefficient
    b = prp.rain_fall_speed_exponent
    ρ₀ = prp.reference_air_density
    ρʷ = prp.liquid_water_density
    D_min = prp.rain_diameter_min
    D_max = prp.rain_diameter_max
    v_min = prp.rain_velocity_min
    v_max = prp.rain_velocity_max

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, FT(1))

    # Mean rain drop mass
    m̄ = qʳ_eff / nʳ_eff

    # Mass-weighted mean diameter: m = (π/6) ρʷ D³
    D̄ₘ = cbrt(6 * m̄ / (FT(π) * ρʷ))

    # Density correction factor
    ρ_correction = sqrt(ρ₀ / ρ)

    # Clamp diameter to physical range
    D̄ₘ_clamped = clamp(D̄ₘ, D_min, D_max)

    # Terminal velocity
    vₜ = a * D̄ₘ_clamped^b * ρ_correction

    return clamp(vₜ, v_min, v_max)
end

"""
    rain_terminal_velocity_number_weighted(p3, qʳ, nʳ, ρ)

Compute number-weighted terminal velocity for rain.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Number-weighted fall speed [m/s] (positive downward)
"""
@inline function rain_terminal_velocity_number_weighted(p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    # Number-weighted velocity is smaller than mass-weighted
    ratio = prp.velocity_ratio_number_to_mass
    vₘ = rain_terminal_velocity_mass_weighted(p3, qʳ, nʳ, ρ)

    return ratio * vₘ
end

"""
    ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

Compute mass-weighted terminal velocity for ice.

Uses regime-dependent fall speeds following [Mitchell (1996)](@citet Mitchell1996)
and [Morrison and Milbrandt (2015)](@citet Morrison2015parameterization).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Mass-weighted fall speed [m/s] (positive downward)
"""
@inline function ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    ρ₀ = prp.reference_air_density
    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    D_threshold = prp.ice_diameter_threshold
    D_min = prp.ice_diameter_min
    D_max = prp.ice_diameter_max
    v_min = prp.ice_velocity_min
    v_max = prp.ice_velocity_max
    ρᶠ_min = prp.minimum_rime_density
    ρᶠ_max = prp.maximum_rime_density

    a_unrimed = prp.ice_fall_speed_coefficient_unrimed
    b_unrimed = prp.ice_fall_speed_exponent_unrimed
    a_rimed = prp.ice_fall_speed_coefficient_rimed
    b_rimed = prp.ice_fall_speed_exponent_rimed
    c_small = prp.ice_small_particle_coefficient

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1))

    # Mean ice particle mass
    m̄ = qⁱ_eff / nⁱ_eff

    # Effective density depends on riming
    Fᶠ_clamped = clamp(Fᶠ, FT(0), FT(1))
    ρᶠ_clamped = clamp(ρᶠ, ρᶠ_min, ρᶠ_max)
    ρ_eff = ρ_eff_unrimed + Fᶠ_clamped * (ρᶠ_clamped - ρ_eff_unrimed)

    # Effective diameter
    D̄ₘ = cbrt(6 * m̄ / (FT(π) * ρ_eff))
    D_clamped = clamp(D̄ₘ, D_min, D_max)

    # Coefficients interpolated based on riming
    a = a_unrimed + Fᶠ_clamped * (a_rimed - a_unrimed)
    b = b_unrimed + Fᶠ_clamped * (b_rimed - b_unrimed)

    # Density correction
    ρ_correction = sqrt(ρ₀ / ρ)

    # Terminal velocity (large particle regime)
    vₜ_large = a * D_clamped^b * ρ_correction

    # Small particle (Stokes) regime
    vₜ_small = c_small * D_clamped^2 * ρ_correction

    # Blend between regimes
    vₜ = ifelse(D_clamped < D_threshold, vₜ_small, vₜ_large)

    return clamp(vₜ, v_min, v_max)
end

"""
    ice_terminal_velocity_number_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

Compute number-weighted terminal velocity for ice.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Number-weighted fall speed [m/s] (positive downward)
"""
@inline function ice_terminal_velocity_number_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)
    prp = p3.process_rates
    ratio = prp.velocity_ratio_number_to_mass
    vₘ = ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

    return ratio * vₘ
end

"""
    ice_terminal_velocity_reflectivity_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

Compute reflectivity-weighted (Z-weighted) terminal velocity for ice.

Needed for the sixth moment (reflectivity) sedimentation in 3-moment P3.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Reflectivity-weighted fall speed [m/s] (positive downward)
"""
@inline function ice_terminal_velocity_reflectivity_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)
    prp = p3.process_rates
    ratio = prp.velocity_ratio_reflectivity_to_mass
    vₘ = ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

    return ratio * vₘ
end
