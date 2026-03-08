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

using Breeze.Thermodynamics: temperature,
                             saturation_specific_humidity,
                             PlanarLiquidSurface,
                             PlanarIceSurface,
                             liquid_latent_heat,
                             mixture_heat_capacity,
                             vapor_gas_constant,
                             MoistureMassFractions

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
##### Cloud condensation/evaporation
#####

"""
    cloud_condensation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, constants)

Compute cloud liquid condensation/evaporation rate using relaxation-to-saturation.

When the air is supersaturated (qᵛ > qᵛ⁺ˡ), excess vapor condenses onto cloud
droplets. When subsaturated, cloud liquid evaporates back to vapor. The rate
follows a relaxation timescale with a thermodynamic (psychrometric) correction
factor that accounts for latent heating during phase change.

# Arguments
- `p3`: P3 microphysics scheme (provides condensation timescale)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ˡ`: Saturation vapor mass fraction over liquid [kg/kg]
- `T`: Temperature [K]
- `q`: Moisture mass fractions (vapor, liquid, ice)
- `constants`: Thermodynamic constants

# Returns
- Rate of vapor → cloud liquid conversion [kg/kg/s]
  (positive = condensation, negative = evaporation)
"""
@inline function cloud_condensation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, constants)
    FT = typeof(qᶜˡ)
    τᶜˡ = p3.cloud.condensation_timescale

    # Thermodynamic adjustment factor (psychrometric correction)
    ℒˡ = liquid_latent_heat(T, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺_dT = qᵛ⁺ˡ * (ℒˡ / (Rᵛ * T^2) - 1 / T)
    Γˡ = 1 + (ℒˡ / cᵖᵐ) * dqᵛ⁺_dT

    # Relaxation toward saturation
    Sᶜᵒⁿᵈ = (qᵛ - qᵛ⁺ˡ) / (Γˡ * τᶜˡ)

    # Limit evaporation to available cloud liquid
    Sᶜᵒⁿᵈ_min = -max(0, qᶜˡ) / τᶜˡ
    return max(Sᶜᵒⁿᵈ, Sᶜᵒⁿᵈ_min)
end

#####
##### Rain processes
#####

"""
    rain_autoconversion_rate(p3, qᶜˡ, Nᶜ)

Compute rain autoconversion rate following [Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000).

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

Compute rain accretion rate following [Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000).

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

    # KK2000 Eq. 33 (Fortran P3 form): ∂qʳ/∂t = k₂ × qᶜˡ × qʳ^α
    k₂ = prp.accretion_coefficient
    α = prp.accretion_exponent

    return k₂ * qᶜˡ_eff * qʳ_eff^α
end

"""
    rain_self_collection_rate(p3, qʳ, nʳ, ρ)

Compute rain self-collection rate (number tendency only).

Large rain drops collect smaller ones, reducing number but conserving mass.
Follows [Seifert and Beheng (2001)](@cite SeifertBeheng2001).

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
    rain_breakup_rate(p3, qʳ, nʳ, self_collection)

Compute rain breakup rate following [Seifert and Beheng (2006)](@cite SeifertBeheng2006).

Large rain drops spontaneously break up into smaller fragments, producing
a number source that counterbalances self-collection. Uses a three-piece
function of the volume-mean drop diameter ``D_r``:

1. ``D_r < D_{th}`` (0.35 mm): No effect (``Φ_{br} = -1``)
2. ``D_{th} ≤ D_r ≤ D_{eq}`` (0.35–0.9 mm): Linear transition
3. ``D_r > D_{eq}`` (0.9 mm): Exponential breakup dominates

The breakup rate is ``-(Φ_{br} + 1) \\times`` self-collection rate.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `self_collection`: Self-collection rate [1/kg/s] (negative)

# Returns
- Breakup rate [1/kg/s] (positive = number source)
"""
@inline function rain_breakup_rate(p3, qʳ, nʳ, self_collection)
    FT = typeof(qʳ)
    prp = p3.process_rates

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # Volume-mean drop diameter: D_r = (6 qʳ / (π ρ_w nʳ))^(1/3)
    ρ_water = prp.liquid_water_density
    mean_mass = safe_divide(qʳ_eff, nʳ_eff, FT(1e-10))
    D_r = cbrt(FT(6) * mean_mass / (FT(π) * ρ_water))

    # Clamp to physical maximum (~2.5mm mean diameter, matching SB2006 xr_max)
    D_r = min(D_r, FT(2.5e-3))

    # Three-piece breakup function (Seifert & Beheng 2006, Eq. 13)
    D_eq = prp.rain_breakup_diameter_threshold  # 0.9mm: equilibrium diameter
    κ_br = prp.rain_breakup_coefficient         # 2300 m⁻¹: exponential coefficient
    D_th = FT(0.35e-3)                          # transition diameter
    k_br = FT(1000)                             # linear coefficient [1/m]
    ΔD = D_r - D_eq

    Φ_br = ifelse(D_r < D_th,
                   FT(-1),
                   ifelse(D_r ≤ D_eq,
                          k_br * ΔD,
                          FT(2) * (exp(κ_br * ΔD) - FT(1))))

    # Breakup rate: -(Φ_br + 1) × self_collection (Eq. 13 from SB2006)
    return -(Φ_br + FT(1)) * self_collection
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

    # Saturation vapor pressure derived from qᵛ⁺ˡ
    # From ideal gas law: ρ_v⁺ = e_s / (R_v × T)
    # And ρ_v⁺ ≈ ρ × qᵛ⁺ˡ for small qᵛ⁺ˡ
    e_s = ρ * qᵛ⁺ˡ * R_v * T

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
    R_d = FT(287.0)           # Gas constant for dry air [J/kg/K]
    L_s = FT(2.835e6)         # Latent heat of sublimation [J/kg]
    K_a = FT(2.5e-2)          # Thermal conductivity of air [W/m/K]
    D_v = FT(2.5e-5)          # Diffusivity of water vapor [m²/s]

    # Saturation vapor pressure over ice
    # Derived from qᵛ⁺ⁱ: qᵛ⁺ⁱ = ε × e_si / (P - (1-ε) × e_si)
    # Rearranging: e_si = P × qᵛ⁺ⁱ / (ε + qᵛ⁺ⁱ × (1 - ε))
    ε = R_d / R_v
    e_si = P * qᵛ⁺ⁱ / (ε + qᵛ⁺ⁱ * (1 - ε))

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
    # Estimate terminal velocity (simplified power law, unrimed)
    V = prp.ice_fall_speed_coefficient_unrimed * D_mean^prp.ice_fall_speed_exponent_unrimed
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

    # PSD correction factor: PSD integration gives <C×fv>/<C(Dm)×fv(Dm)>.
    # For the mean-mass approximation, psd_correction = 1.0.
    # The Fortran P3 uses lookup tables that integrate over the full PSD.
    # With our 41-level grid (vs Fortran's 90), we use 1.0 to avoid
    # overestimating deposition, keeping riming as the dominant growth mechanism.
    psd_correction = FT(1)
    dep_rate = psd_correction * nⁱ_eff * dm_dt

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
    ice_melting_rate(p3, qⁱ, nⁱ, T, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ)

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
@inline function ice_melting_rate(p3, qⁱ, nⁱ, T, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ)
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
    ρ_vs = e_s0 / (R_v * T₀)  # Saturation vapor density at T₀

    # Ambient vapor density (from mixing ratio and actual air density)
    ρ_v = qᵛ * ρ

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
    V = prp.ice_fall_speed_coefficient_unrimed * D_mean^prp.ice_fall_speed_exponent_unrimed
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

    # Limit melting rate: physical heat-transfer rate is the true limiter.
    # Guard against numerical overflow with a 1-second safety timescale,
    # meaning at most all ice can melt per second. The driver or time
    # integrator must additionally limit melting to available ice per dt.
    τ_safety = FT(1)  # [s] — CFL-like constraint, not a physical timescale
    max_melt = qⁱ_eff / τ_safety
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
    ice_melting_rates(p3, qⁱ, nⁱ, qʷⁱ, T, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ)

Compute partitioned ice melting rates following Milbrandt et al. (2025).

Above freezing, ice particles melt. The meltwater is partitioned:
- **Partial melting** (large particles): Meltwater stays on ice as liquid coating (qʷⁱ)
- **Complete melting** (small particles): Meltwater sheds directly to rain

The partitioning is based on a maximum liquid fraction capacity. Once the
particle reaches this capacity, additional meltwater sheds to rain.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `T`: Temperature [K]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺`: Saturation vapor mass fraction over liquid [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]

# Returns
- NamedTuple with `partial_melting` and `complete_melting` rates [kg/kg/s]
"""
@inline function ice_melting_rates(p3, qⁱ, nⁱ, qʷⁱ, T, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    # Get total melting rate
    total_melt = ice_melting_rate(p3, qⁱ, nⁱ, T, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ)

    # Maximum liquid fraction capacity (from Milbrandt et al. 2025)
    # Spongy ice can hold about 14% liquid by mass
    max_liquid_fraction = prp.maximum_liquid_fraction

    # Total ice mass (ice + liquid coating)
    qⁱ_total = qⁱ + qʷⁱ
    qⁱ_total_safe = max(qⁱ_total, FT(1e-20))

    # Current liquid fraction
    current_liquid_fraction = qʷⁱ / qⁱ_total_safe

    # Partition melting based on liquid fraction capacity
    # If below capacity: melting goes to liquid coating
    # If at/above capacity: melting sheds to rain
    fraction_to_coating = clamp_positive(max_liquid_fraction - current_liquid_fraction) / max_liquid_fraction

    # Limit to [0, 1]
    fraction_to_coating = clamp(fraction_to_coating, FT(0), FT(1))

    partial = total_melt * fraction_to_coating
    complete = total_melt * (1 - fraction_to_coating)

    return (partial_melting = partial, complete_melting = complete)
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
- Tuple (Q_frz, N_frz): mass rate [kg/kg/s] and number rate [1/m³/s]
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
    Nᶜ_eff = max(Nᶜ, FT(1))
    m_drop = ρ * qᶜˡ_eff / Nᶜ_eff           # [kg]
    V_drop = m_drop / ρ_water                  # [m³]

    # Mass freezing rate [kg/kg/s]:
    # = (Nᶜ/ρ) × bimm × psd × exp(a × ΔT) × V_drop × m_drop
    Q_frz = (Nᶜ_eff / ρ) * bimm * psd_correction * exp(aimm * ΔT) * V_drop * m_drop

    # Number freezing rate [1/m³/s]:
    # = Nᶜ × bimm × psd × exp(a × ΔT) × V_drop
    N_frz = Nᶜ_eff * bimm * psd_correction * exp(aimm * ΔT) * V_drop

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

Following [Meyers et al. (1992)](@cite MeyerEtAl1992icenucleation):

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
    freezing_active = (T < T₀) & (T < T_max) & (qᶜˡ_eff > FT(1e-8))

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

    # Effective density
    ρⁱ = prp.pure_ice_density
    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    ρ_eff = (1 - Fᶠ) * ρ_eff_unrimed + Fᶠ * ρᶠ

    # Mean diameter
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_eff))

    # Mean terminal velocity (regime-dependent, from prp)
    a_V = (1 - Fᶠ) * prp.ice_fall_speed_coefficient_unrimed + Fᶠ * prp.ice_fall_speed_coefficient_rimed
    b_V = (1 - Fᶠ) * prp.ice_fall_speed_exponent_unrimed + Fᶠ * prp.ice_fall_speed_exponent_rimed
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

    # Mean particle mass and effective density
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))
    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    ρ_eff = (1 - Fᶠ) * ρ_eff_unrimed + Fᶠ * ρᶠ

    # Mean diameter (clamped to physical range)
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_eff))
    D_mean = clamp(D_mean, prp.ice_diameter_min, prp.ice_diameter_max)

    # Mean terminal velocity (regime-dependent, from prp)
    a_V = (1 - Fᶠ) * prp.ice_fall_speed_coefficient_unrimed + Fᶠ * prp.ice_fall_speed_coefficient_rimed
    b_V = (1 - Fᶠ) * prp.ice_fall_speed_exponent_unrimed + Fᶠ * prp.ice_fall_speed_exponent_rimed
    V_mean = a_V * D_mean^b_V

    # Projected area (regime-dependent: aggregate vs sphere)
    γ = FT(0.2285)
    σ = FT(1.88)
    A_agg = γ * D_mean^σ
    A_sphere = FT(π) / 4 * D_mean^2
    A_mean = (1 - Fᶠ) * A_agg + Fᶠ * A_sphere

    # Air density correction: Fortran P3 lookup tables computed at reference
    # conditions (ρ₀ ≈ 0.826 kg/m³) then scaled by (ρ₀/ρ)^0.54.
    ρ₀ = prp.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Collection rate = E × qc × ρ × ni × rhofaci × <A×V> × psd_correction
    # PSD correction accounts for the PSD-integrated collection kernel
    # being larger than the mean-mass value.
    psd_correction = prp.riming_psd_correction
    rate = Eᶜⁱ * qᶜˡ_eff * nⁱ_eff * ρ * rhofaci * A_mean * V_mean * psd_correction

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

    # Mean particle mass and effective density
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))
    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    ρ_eff = (1 - Fᶠ) * ρ_eff_unrimed + Fᶠ * ρᶠ

    # Mean diameter (clamped)
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_eff))
    D_mean = clamp(D_mean, prp.ice_diameter_min, prp.ice_diameter_max)

    # Mean terminal velocity (from prp, consistent with aggregation and cloud riming)
    a_V = (1 - Fᶠ) * prp.ice_fall_speed_coefficient_unrimed + Fᶠ * prp.ice_fall_speed_coefficient_rimed
    b_V = (1 - Fᶠ) * prp.ice_fall_speed_exponent_unrimed + Fᶠ * prp.ice_fall_speed_exponent_rimed
    V_mean = a_V * D_mean^b_V

    # Projected area
    γ = FT(0.2285)
    σ = FT(1.88)
    A_agg = γ * D_mean^σ
    A_sphere = FT(π) / 4 * D_mean^2
    A_mean = (1 - Fᶠ) * A_agg + Fᶠ * A_sphere

    # Air density correction (same as cloud riming)
    ρ₀ = prp.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)

    # Collection rate = E × qr × ρ × ni × rhofaci × <A×V> × psd_correction
    psd_correction = prp.riming_psd_correction
    rate = Eʳⁱ * qʳ_eff * nⁱ_eff * ρ * rhofaci * A_mean * V_mean * psd_correction

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

#####
##### Combined P3 tendency calculation
#####

"""
    P3ProcessRates

Container for computed P3 process rates.
Includes Phase 1 (rain, deposition, melting), Phase 2 (aggregation, riming, shedding, nucleation).

Following Milbrandt et al. (2025), melting is partitioned:
- `partial_melting`: Meltwater stays on ice as liquid coating (large particles)
- `complete_melting`: Meltwater sheds to rain (small particles)
"""
struct P3ProcessRates{FT}
    # Phase 1: Cloud condensation/evaporation
    condensation :: FT             # Vapor → cloud liquid [kg/kg/s] (positive = condensation, negative = evaporation)

    # Phase 1: Rain tendencies
    autoconversion :: FT           # Cloud → rain mass [kg/kg/s]
    accretion :: FT                # Cloud → rain mass (via rain sweep-out) [kg/kg/s]
    rain_evaporation :: FT         # Rain → vapor mass [kg/kg/s]
    rain_self_collection :: FT     # Rain number reduction [1/kg/s]
    rain_breakup :: FT             # Rain number increase from breakup [1/kg/s]

    # Phase 1: Ice tendencies
    deposition :: FT               # Vapor → ice mass [kg/kg/s]
    partial_melting :: FT          # Ice → liquid coating (stays on ice) [kg/kg/s]
    complete_melting :: FT         # Ice → rain mass (sheds) [kg/kg/s]
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
    compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)

Compute all P3 process rates (Phase 1 and Phase 2) from a microphysical state.

This is the gridless version that accepts a `P3MicrophysicalState` directly,
suitable for use in GPU kernels where grid indexing is handled externally.

# Arguments
- `p3`: P3 microphysics scheme
- `ρ`: Air density [kg/m³]
- `ℳ`: P3MicrophysicalState containing all mixing ratios
- `𝒰`: Thermodynamic state
- `constants`: Thermodynamic constants

# Returns
- `P3ProcessRates` containing all computed rates
"""
@inline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
    FT = typeof(ρ)
    prp = p3.process_rates
    T₀ = prp.freezing_temperature

    # Extract from microphysical state (already specific, not density-weighted)
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nʳ = ℳ.nʳ
    qⁱ = ℳ.qⁱ
    nⁱ = ℳ.nⁱ
    qᶠ = ℳ.qᶠ
    bᶠ = ℳ.bᶠ
    qʷⁱ = ℳ.qʷⁱ

    # Rime properties
    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))
    ρᶠ = safe_divide(qᶠ, bᶠ, FT(400))

    # Thermodynamic state
    T = temperature(𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor

    # Saturation vapor mixing ratios using Breeze thermodynamics
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())

    # Moisture mass fractions for thermodynamic calculations
    q = 𝒰.moisture_mass_fractions

    # Cloud droplet number concentration
    Nᶜ = p3.cloud.number_concentration

    # =========================================================================
    # Phase 1: Cloud condensation/evaporation
    # =========================================================================
    cond = cloud_condensation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, constants)

    # =========================================================================
    # Phase 1: Rain processes
    # =========================================================================
    autoconv = rain_autoconversion_rate(p3, qᶜˡ, Nᶜ)
    accr = rain_accretion_rate(p3, qᶜˡ, qʳ)
    rain_evap = rain_evaporation_rate(p3, qʳ, nʳ, qᵛ, qᵛ⁺ˡ, T, ρ)
    rain_self = rain_self_collection_rate(p3, qʳ, nʳ, ρ)
    rain_br = rain_breakup_rate(p3, qʳ, nʳ, rain_self)

    # =========================================================================
    # Phase 1: Ice deposition/sublimation and melting
    # =========================================================================
    P = 𝒰.reference_pressure
    dep = ifelse(qⁱ > FT(1e-20),
                 ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P),
                 ice_deposition_rate(p3, qⁱ, qᵛ, qᵛ⁺ⁱ))

    # Partitioned melting: partial stays on ice, complete goes to rain
    melt_rates = ice_melting_rates(p3, qⁱ, nⁱ, qʷⁱ, T, qᵛ, qᵛ⁺ˡ, Fᶠ, ρᶠ, ρ)
    partial_melt = melt_rates.partial_melting
    complete_melt = melt_rates.complete_melting
    # Only complete melting removes ice particles; partial melting keeps particles as ice
    melt_n = ice_melting_number_rate(qⁱ, nⁱ, complete_melt)

    # =========================================================================
    # Phase 2: Ice aggregation
    # =========================================================================
    agg = ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ)

    # =========================================================================
    # Phase 2: Riming
    # =========================================================================
    cloud_rim = cloud_riming_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    cloud_rim_n = cloud_riming_number_rate(qᶜˡ, Nᶜ, cloud_rim)

    rain_rim = rain_riming_rate(p3, qʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    rain_rim_n = rain_riming_number_rate(qʳ, nʳ, rain_rim)

    # Rime density for new rime (use actual ice fall speed, not placeholder)
    vᵢ = ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)
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
    cloud_frz_q, cloud_frz_n_vol = immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)
    # Convert cloud_frz_n from [1/m³/s] to [1/kg/s] (Nᶜ is in 1/m³)
    cloud_frz_n = cloud_frz_n_vol / ρ
    rain_frz_q, rain_frz_n = immersion_freezing_rain_rate(p3, qʳ, nʳ, T)

    # =========================================================================
    # Rime splintering (Hallett-Mossop secondary ice production)
    # =========================================================================
    spl_q, spl_n = rime_splintering_rate(p3, cloud_rim, rain_rim, T)

    return P3ProcessRates(
        # Phase 1: Condensation
        cond,
        # Phase 1: Rain
        autoconv, accr, rain_evap, rain_self, rain_br,
        # Phase 1: Ice
        dep, partial_melt, complete_melt, melt_n,
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

Cloud liquid gains from:
- Condensation (Phase 1)

Cloud liquid is consumed by:
- Autoconversion (Phase 1)
- Accretion by rain (Phase 1)
- Riming by ice (Phase 2)
- Immersion freezing (Phase 2)
"""
@inline function tendency_ρqᶜˡ(rates::P3ProcessRates, ρ)
    # Phase 1: condensation (positive = cloud forms)
    gain = rates.condensation
    # Phase 1: autoconversion and accretion
    # Phase 2: cloud riming by ice, immersion freezing
    loss = rates.autoconversion + rates.accretion + rates.cloud_riming + rates.cloud_freezing_mass
    return ρ * (gain - loss)
end

"""
    tendency_ρqʳ(rates)

Compute rain mass tendency from P3 process rates.

Rain gains from:
- Autoconversion (Phase 1)
- Accretion (Phase 1)
- Complete melting (Phase 1) - meltwater that sheds from ice
- Shedding (Phase 2) - liquid coating shed from ice

Rain loses from:
- Evaporation (Phase 1)
- Riming (Phase 2)
- Immersion freezing (Phase 2)
"""
@inline function tendency_ρqʳ(rates::P3ProcessRates, ρ)
    # Phase 1: gains from autoconv, accr, complete_melt; loses from evap
    # Phase 2: gains from shedding; loses from riming and freezing
    # Note: partial_melting stays on ice as liquid coating, only complete_melting goes to rain
    gain = rates.autoconversion + rates.accretion + rates.complete_melting + rates.shedding
    loss = -rates.rain_evaporation + rates.rain_riming + rates.rain_freezing_mass  # evap is negative
    return ρ * (gain - loss)
end

"""
    tendency_ρnʳ(rates, ρ, qᶜˡ, Nc, m_drop)

Compute rain number tendency from P3 process rates.

Rain number gains from:
- Autoconversion (Phase 1)
- Complete melting (Phase 1) - new rain drops from melted ice
- Breakup (Phase 1) - large drops fragment into smaller ones
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

    # Phase 1: New drops from complete melting (conserve number)
    # Only complete_melting produces new rain drops; partial_melting stays on ice
    n_from_melt = safe_divide(nⁱ * rates.complete_melting, qⁱ, zero(FT))

    # Phase 1: Self-collection (negative) + breakup (positive)
    # Phase 2: Shedding creates new drops
    # Phase 2: Riming removes rain drops (already negative)

    return ρ * (n_from_autoconv + n_from_melt +
                rates.rain_self_collection +
                rates.rain_breakup +
                rates.shedding_number +
                rates.rain_riming_number +
                rates.rain_freezing_number)
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
- Partial melting (Phase 1) - becomes liquid coating
- Complete melting (Phase 1) - sheds to rain
"""
@inline function tendency_ρqⁱ(rates::P3ProcessRates, ρ)
    # Phase 1: deposition, melting (both partial and complete reduce ice mass)
    # Phase 2: riming (cloud + rain), refreezing, nucleation, freezing, splintering
    # Splintering mass is already part of the riming mass (splinters fragment existing rime),
    # so it is NOT added here. Instead, it is subtracted from rime mass in tendency_ρqᶠ.
    gain = rates.deposition + rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.nucleation_mass + rates.cloud_freezing_mass + rates.rain_freezing_mass
    # Total melting reduces ice mass (partial stays as liquid coating, complete sheds)
    loss = rates.partial_melting + rates.complete_melting
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
    # Splintering mass is subtracted from rime (splinters fragment existing rime)
    loss = Fᶠ * (rates.partial_melting + rates.complete_melting) + rates.splintering_mass
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

    ρ_water = FT(1000)  # physical constant [kg/m³]

    # Phase 2: Volume gain from new rime (cloud + rain riming + refreezing)
    # Use density of new rime for fresh rime, current density for refreezing
    # Frozen cloud/rain drops are dense ice at approximately water density
    volume_gain = (rates.cloud_riming + rates.rain_riming) / ρ_rim_new_safe +
                   rates.refreezing / ρᶠ_safe +
                   (rates.cloud_freezing_mass + rates.rain_freezing_mass) / ρ_water

    # Phase 1: Volume loss from melting (proportional to rime fraction)
    volume_loss = Fᶠ * (rates.partial_melting + rates.complete_melting) / ρᶠ_safe

    return ρ * (volume_gain - volume_loss)
end

"""
    tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ)

Compute ice sixth moment tendency from P3 process rates.

The sixth moment (reflectivity) changes with:
- Deposition (growth) (Phase 1)
- Melting (loss) (Phase 1)
- Riming (growth) (Phase 2)
- Nucleation (growth) (Phase 2)
- Aggregation (redistribution) (Phase 2)

This simplified version uses proportional scaling (Z/q ratio).
For more accurate 3-moment treatment, use the version that accepts
the p3 scheme to access tabulated sixth moment integrals.
"""
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ)
    FT = typeof(ρ)

    # Simplified: Z changes proportionally to mass changes
    # More accurate version would use full integral formulation
    ratio = safe_divide(zⁱ, qⁱ, zero(FT))

    # Net mass change for ice
    # Total melting (partial + complete) reduces ice mass
    total_melting = rates.partial_melting + rates.complete_melting
    mass_change = rates.deposition - total_melting +
                  rates.cloud_riming + rates.rain_riming + rates.refreezing

    return ρ * ratio * mass_change
end

"""
    tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, p3)

Compute ice sixth moment tendency using tabulated integrals when available.

Following Milbrandt et al. (2021, 2024), the sixth moment tendency is
computed by integrating the contribution of each process over the
size distribution, properly accounting for how different processes
affect particles of different sizes.

When tabulated integrals are available via `tabulate(p3, arch)`, uses
pre-computed lookup tables. Otherwise, falls back to proportional scaling.

# Arguments
- `rates`: P3ProcessRates containing mass tendencies
- `ρ`: Air density [kg/m³]
- `qⁱ`: Ice mass mixing ratio [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `zⁱ`: Ice sixth moment [m⁶/kg]
- `Fᶠ`: Rime fraction [-]
- `Fˡ`: Liquid fraction [-]
- `p3`: P3 microphysics scheme (for accessing tabulated integrals)

# Returns
- Tendency of density-weighted sixth moment [kg/m³ × m⁶/kg / s]
"""
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, p3)
    FT = typeof(ρ)

    # Mean ice particle mass for table lookup
    m̄ = safe_divide(qⁱ, nⁱ, FT(1e-20))
    log_mean_mass = log10(max(m̄, FT(1e-20)))

    # Try to use tabulated sixth moment integrals
    z_tendency = _tabulated_z_tendency(
        p3.ice.sixth_moment, log_mean_mass, Fᶠ, Fˡ, rates, ρ, qⁱ, nⁱ, zⁱ
    )

    return z_tendency
end

# Tabulated version: use TabulatedFunction3D lookups for each process
@inline function _tabulated_z_tendency(sixth::IceSixthMoment{<:TabulatedFunction3D}, log_m, Fᶠ, Fˡ, rates, ρ, qⁱ, nⁱ, zⁱ)
    FT = typeof(ρ)

    # Look up normalized Z contribution for each process
    z_dep = sixth.deposition(log_m, Fᶠ, Fˡ)
    z_melt = sixth.melt1(log_m, Fᶠ, Fˡ) + sixth.melt2(log_m, Fᶠ, Fˡ)
    z_rime = sixth.rime(log_m, Fᶠ, Fˡ)
    z_agg = sixth.aggregation(log_m, Fᶠ, Fˡ)
    z_shed = sixth.shedding(log_m, Fᶠ, Fˡ)
    z_sub = sixth.sublimation(log_m, Fᶠ, Fˡ) + sixth.sublimation1(log_m, Fᶠ, Fˡ)

    # Total melting
    total_melting = rates.partial_melting + rates.complete_melting

    # Compute Z tendency from tabulated integrals
    # Each integral gives the normalized Z rate per unit mass rate
    z_rate = z_dep * rates.deposition +
             z_rime * (rates.cloud_riming + rates.rain_riming) +
             z_agg * rates.aggregation * safe_divide(qⁱ, nⁱ, FT(1e-12)) +  # agg is number rate
             z_shed * rates.shedding -
             z_melt * total_melting

    # Sublimation (when deposition is negative)
    is_sublimating = rates.deposition < 0
    z_rate = z_rate + ifelse(is_sublimating, z_sub * abs(rates.deposition), zero(FT))

    return ρ * z_rate
end

# Fallback: use proportional scaling when integrals are not tabulated
@inline function _tabulated_z_tendency(::Any, log_m, Fᶠ, Fˡ, rates, ρ, qⁱ, nⁱ, zⁱ)
    # Fall back to the simple proportional scaling
    FT = typeof(ρ)
    ratio = safe_divide(zⁱ, qⁱ, zero(FT))
    total_melting = rates.partial_melting + rates.complete_melting
    mass_change = rates.deposition - total_melting +
                  rates.cloud_riming + rates.rain_riming + rates.refreezing
    return ρ * ratio * mass_change
end

"""
    tendency_ρqʷⁱ(rates)

Compute liquid on ice tendency from P3 process rates.

Liquid on ice:
- Gains from partial melting above freezing (meltwater stays on ice)
- Loses from shedding (Phase 2) - liquid sheds to rain
- Loses from refreezing (Phase 2) - liquid refreezes to ice

Following Milbrandt et al. (2025), partial melting adds to the liquid coating
while complete melting sheds directly to rain.
"""
@inline function tendency_ρqʷⁱ(rates::P3ProcessRates, ρ)
    # Gains from partial melting (meltwater stays on ice as liquid coating)
    # Loses from shedding (liquid sheds to rain) and refreezing (liquid refreezes)
    gain = rates.partial_melting
    loss = rates.shedding + rates.refreezing
    return ρ * (gain - loss)
end

"""
    tendency_ρqᵛ(rates)

Compute vapor mass tendency from P3 process rates.

Vapor is consumed by:
- Condensation (vapor → cloud liquid)
- Deposition (vapor → ice)
- Deposition nucleation (vapor → ice)

Vapor is produced by:
- Cloud evaporation (negative condensation)
- Rain evaporation
- Sublimation (negative deposition)
"""
@inline function tendency_ρqᵛ(rates::P3ProcessRates, ρ)
    # Condensation: positive = vapor loss, negative = vapor gain (evap)
    # Deposition: positive = vapor loss (dep), negative = vapor gain (sublimation)
    # Rain evaporation: negative = rain loss = vapor gain
    # Nucleation: always vapor loss
    return ρ * (-rates.condensation - rates.deposition - rates.nucleation_mass - rates.rain_evaporation)
end

#####
##### Fallback methods for Nothing rates
#####
##### These are safety fallbacks that return zero tendency when rates
##### have not been computed (e.g., during incremental development).
#####

@inline tendency_ρqᶜˡ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqʳ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnʳ(::Nothing, ρ, nⁱ, qⁱ; kwargs...) = zero(ρ)
@inline tendency_ρqⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqᶠ(::Nothing, ρ, Fᶠ) = zero(ρ)
@inline tendency_ρbᶠ(::Nothing, ρ, Fᶠ, ρᶠ) = zero(ρ)
@inline tendency_ρzⁱ(::Nothing, ρ, qⁱ, nⁱ, zⁱ) = zero(ρ)
@inline tendency_ρqʷⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqᵛ(::Nothing, ρ) = zero(ρ)

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
See [Seifert and Beheng (2006)](@cite SeifertBeheng2006).

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

    # Density correction factor (Heymsfield et al. 2006)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

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
    ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))

Compute mass-weighted terminal velocity for ice.

When tabulated integrals are available (via `tabulate(p3, arch)`), uses
pre-computed lookup tables for accurate size-distribution integration.
Otherwise, uses regime-dependent fall speeds following [Mitchell (1996)](@cite Mitchell1996powerlaws)
and [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `Fˡ`: Liquid fraction (optional, for tabulated lookup)

# Returns
- Mass-weighted fall speed [m/s] (positive downward)
"""
@inline function ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density
    v_min = prp.ice_velocity_min
    v_max = prp.ice_velocity_max

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1))

    # Mean ice particle mass
    m̄ = qⁱ_eff / nⁱ_eff

    # Density correction factor (Heymsfield et al. 2006)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # Try to use tabulated fall speed if available
    vₜ = _tabulated_mass_weighted_fall_speed(fs.mass_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated version: use TabulatedFunction3D lookup
@inline function _tabulated_mass_weighted_fall_speed(table::TabulatedFunction3D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    FT = typeof(m̄)
    # Compute log mean mass (guarding against log(0))
    log_mean_mass = log10(max(m̄, FT(1e-20)))
    # Look up normalized velocity from table
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ)
    return vₜ_norm * ρ_correction
end

# Fallback: use analytical approximation when not tabulated
@inline function _tabulated_mass_weighted_fall_speed(::Any, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    FT = typeof(m̄)

    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    D_threshold = prp.ice_diameter_threshold
    D_min = prp.ice_diameter_min
    D_max = prp.ice_diameter_max
    ρᶠ_min = prp.minimum_rime_density
    ρᶠ_max = prp.maximum_rime_density

    a_unrimed = prp.ice_fall_speed_coefficient_unrimed
    b_unrimed = prp.ice_fall_speed_exponent_unrimed
    a_rimed = prp.ice_fall_speed_coefficient_rimed
    b_rimed = prp.ice_fall_speed_exponent_rimed
    c_small = prp.ice_small_particle_coefficient

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

    # Terminal velocity (large particle regime)
    vₜ_large = a * D_clamped^b * ρ_correction

    # Small particle (Stokes) regime
    vₜ_small = c_small * D_clamped^2 * ρ_correction

    # Mass-weighted PSD correction (analytical fallback only — the tabulated
    # path already returns PSD-integrated values). For an inverse exponential
    # PSD (μ=0), the mass-weighted velocity is Γ(4+b)/(Γ(4)×λ^(-b)) ≈ 1.9×
    # the single-particle velocity at D_mean. Correction = Γ(4+b)/(6×1.817^b).
    mass_weight_factor = FT(1.9)

    # Blend between regimes
    vₜ = ifelse(D_clamped < D_threshold, vₜ_small, vₜ_large)
    return vₜ * mass_weight_factor
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
@inline function ice_terminal_velocity_number_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density
    v_min = prp.ice_velocity_min
    v_max = prp.ice_velocity_max

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1))
    m̄ = qⁱ_eff / nⁱ_eff
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # Try to use tabulated fall speed if available
    vₜ = _tabulated_number_weighted_fall_speed(fs.number_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated version: use TabulatedFunction3D lookup
@inline function _tabulated_number_weighted_fall_speed(table::TabulatedFunction3D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    FT = typeof(m̄)
    log_mean_mass = log10(max(m̄, FT(1e-20)))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ)
    return vₜ_norm * ρ_correction
end

# Fallback: use ratio to mass-weighted velocity
@inline function _tabulated_number_weighted_fall_speed(::Any, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    ratio = prp.velocity_ratio_number_to_mass
    vₘ = _tabulated_mass_weighted_fall_speed(nothing, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    return ratio * vₘ
end

"""
    ice_terminal_velocity_reflectivity_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=0)

Compute reflectivity-weighted (Z-weighted) terminal velocity for ice.

Needed for the sixth moment (reflectivity) sedimentation in 3-moment P3.
When tabulated integrals are available, uses pre-computed lookup tables.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `Fˡ`: Liquid fraction (optional, for tabulated lookup)

# Returns
- Reflectivity-weighted fall speed [m/s] (positive downward)
"""
@inline function ice_terminal_velocity_reflectivity_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density
    v_min = prp.ice_velocity_min
    v_max = prp.ice_velocity_max

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1))
    m̄ = qⁱ_eff / nⁱ_eff
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # Try to use tabulated fall speed if available
    vₜ = _tabulated_reflectivity_weighted_fall_speed(fs.reflectivity_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated version: use TabulatedFunction3D lookup
@inline function _tabulated_reflectivity_weighted_fall_speed(table::TabulatedFunction3D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    FT = typeof(m̄)
    log_mean_mass = log10(max(m̄, FT(1e-20)))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ)
    return vₜ_norm * ρ_correction
end

# Fallback: use ratio to mass-weighted velocity
@inline function _tabulated_reflectivity_weighted_fall_speed(::Any, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    ratio = prp.velocity_ratio_reflectivity_to_mass
    vₘ = _tabulated_mass_weighted_fall_speed(nothing, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    return ratio * vₘ
end
