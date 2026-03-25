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

    # KK2000 uses cloud liquid directly (no threshold subtraction)
    qᶜˡ_eff = clamp_positive(qᶜˡ)

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

    # KK2000 Eq. 5 (Fortran P3 form): ∂qʳ/∂t = k₂ × (qᶜˡ × qʳ)^α
    k₂ = prp.accretion_coefficient
    α = prp.accretion_exponent

    return k₂ * (qᶜˡ_eff * qʳ_eff)^α
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
- Rate of rain number loss [1/kg/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline function rain_self_collection_rate(p3, qʳ, nʳ, ρ)
    prp = p3.process_rates

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # |∂nʳ/∂t| = k_rr × ρ × qʳ × nʳ (positive magnitude)
    # Sign convention (M7): returns positive; caller subtracts in tendency assembly.
    k_rr = prp.self_collection_coefficient

    return k_rr * ρ * qʳ_eff * nʳ_eff
end

"""
    rain_breakup_rate(p3, qʳ, nʳ, self_collection)

Compute rain breakup rate following Fortran P3 v5.5.0.

Large rain drops spontaneously break up into smaller fragments, producing
a number source that counterbalances self-collection. Uses a two-piece
function of ``D_r = (q_r / (π ρ_w n_r))^{1/3} = 1/λ_r`` (Fortran convention,
no factor of 6; this equals the mean-mass diameter for an exponential DSD):

1. ``D_r < D_{th}``: No breakup effect (``dum = 1``, breakup = 0)
2. ``D_r ≥ D_{th}``: ``dum = 2 - \\exp(κ_{br} (D_r - D_{th}))``, breakup > 0

The breakup rate is ``(1 - dum) \\times`` self-collection rate.

Note: ``D_r`` here uses the Fortran 1/λ_r convention (no factor of 6), which
is smaller than the physical volume-mean diameter by ``6^{1/3} ≈ 1.82``.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `self_collection`: Self-collection rate [1/kg/s] (positive magnitude)

# Returns
- Breakup rate [1/kg/s] (positive = number source)
"""
@inline function rain_breakup_rate(p3, qʳ, nʳ, self_collection)
    FT = typeof(qʳ)
    prp = p3.process_rates

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # Fortran P3 convention: D_r = (qr / (π ρ_w nr))^(1/3) = 1/λ_r
    # (no factor of 6; this is the exponential-DSD mean diameter at μ=0)
    ρ_water = prp.liquid_water_density
    mean_mass = safe_divide(qʳ_eff, nʳ_eff, FT(1e-10))
    D_r = cbrt(mean_mass / (FT(π) * ρ_water))

    # Two-piece breakup function (Fortran P3 v5.5.0)
    D_th = prp.rain_breakup_diameter_threshold  # 280 μm: breakup threshold (1/λ_r convention)
    κ_br = prp.rain_breakup_coefficient         # 2300 m⁻¹: exponential coefficient

    dum = ifelse(D_r < D_th,
                  FT(1),
                  FT(2) - exp(κ_br * (D_r - D_th)))

    # Breakup rate: (1 - dum) × self_collection
    # When D_r < D_th: dum=1 → breakup=0 (no effect)
    # When D_r ≥ D_th: dum < 1 → breakup > 0 (number source)
    # self_collection is positive magnitude (M7); breakup is positive (number source).
    return (FT(1) - dum) * self_collection
end

"""
    rain_evaporation_rate(p3, qʳ, nʳ, qᵛ, qᵛ⁺ˡ, T, ρ)

Compute rain evaporation rate using ventilation-enhanced diffusion.

Rain drops evaporate when the ambient air is subsaturated (qᵛ < qᵛ⁺ˡ).
The evaporation rate is enhanced by ventilation (air flow around falling drops).

Dispatches to either the tabulated PSD integral path or the mean-mass
approximation path depending on `p3.rain.evaporation`:

- **Tabulated** (`TabulatedFunction1D`): Computes λ_r from (q_r, N_r), looks up
  the ventilation integral `I_evap(λ_r) = ∫ D f_v(D) exp(-λ_r D) dD`, then
  applies `dq^r/dt = 2π × N_0 × I_evap × (S-1) / thermo_factor`
  (Mason 1971, capacitance C = D/2 so 4πC = 2πD).
- **Mean-mass** (`RainEvaporation`): Uses a single representative drop of
    diameter `D_mean = (6 m_mean / (π ρ_w))^(1/3)` and the same piecewise
    rain fall-speed law as the tabulated path.

```math
\\frac{dm}{dt} = \\frac{4\\pi C f_v (S - 1)}{\\frac{L_v}{K_a T}(\\frac{L_v}{R_v T} - 1)
               + \\frac{R_v T}{e_s D_v}},\\quad C = D/2
```

# Arguments
- `p3`: P3 microphysics scheme (provides parameters and evaporation table)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ˡ`: Saturation vapor mass fraction over liquid [kg/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `P`: Air pressure [Pa]

# Returns
- Rate of rain evaporation [kg/kg/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline function rain_evaporation_rate(p3, qʳ, nʳ, qᵛ, qᵛ⁺ˡ, T, ρ, P,
                                       transport=air_transport_properties(T, P))
    FT = typeof(qʳ)
    prp = p3.process_rates

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # Only evaporate in subsaturated conditions
    S = qᵛ / max(qᵛ⁺ˡ, FT(1e-10))
    is_subsaturated = S < 1

    # Thermodynamic constants
    Rᵛ = FT(vapor_gas_constant(ThermodynamicConstants()))
    Rᵈ = FT(dry_air_gas_constant(ThermodynamicConstants()))
    L_v = FT(2.5e6)           # Latent heat of vaporization [J/kg]
    # T,P-dependent transport properties (pre-computed or computed on demand)
    K_a = transport.K_a       # Thermal conductivity of air [W/m/K]
    D_v = transport.D_v       # Diffusivity of water vapor [m²/s]
    nu  = transport.nu        # Kinematic viscosity [m²/s]

    # Saturation vapor pressure derived from qᵛ⁺ˡ via inversion of
    # qᵛ⁺ˡ = ε × e_s / (P - (1 - ε) × e_s), consistent with ice deposition path
    ε = Rᵈ / Rᵛ
    qᵛ⁺ˡ_safe = max(qᵛ⁺ˡ, FT(1e-30))
    e_s = P * qᵛ⁺ˡ_safe / (ε + qᵛ⁺ˡ_safe * (1 - ε))

    # Thermodynamic resistance (Mason 1971)
    A = L_v / (K_a * T) * (L_v / (Rᵛ * T) - 1)
    B = Rᵛ * T / (e_s * D_v)
    thermodynamic_factor = max(A + B, FT(1e-10))

    # Internal helpers return negative (S - 1 < 0 when subsaturated).
    # Negate to get positive magnitude (M7 sign convention).
    evap_rate = -rain_evaporation_rate(p3.rain.evaporation, qʳ_eff, nʳ_eff, S,
                                        thermodynamic_factor, p3, prp, nu, D_v, ρ, FT)

    # Cannot evaporate more than available
    τ_evap = prp.rain_evaporation_timescale
    max_evap = qʳ_eff / τ_evap
    evap_rate = min(evap_rate, max_evap)

    return ifelse(is_subsaturated, evap_rate, zero(FT))
end

# Tabulated path: use PSD-integrated ventilation integral I_evap(λ_r)
@inline function rain_evaporation_rate(table::TabulatedFunction1D, qʳ, nʳ, S,
                                        thermodynamic_factor, p3, prp, nu, D_v, ρ, FT)
    ρ_water = p3.water_density

    # Diagnose λ_r from (q_r, N_r) for exponential DSD (μ_r = 0):
    #   q_r = N_r * <m> = N_r * (π/6) ρ_w / λ_r³  ⟹  λ_r = (π ρ_w N_r / (6 q_r))^(1/3)
    m_mean = safe_divide(qʳ, nʳ, FT(1e-12))
    λ_r = cbrt(FT(π) * ρ_water / (6 * max(m_mean, FT(1e-15))))
    # H6: Clamp λ_r to Fortran P3 bounds
    λ_r = clamp(λ_r, prp.rain_lambda_min, prp.rain_lambda_max)

    # Intercept N_0 = N_r * λ_r  (for exponential DSD N'(D) = N_0 exp(-λ D))
    N_0 = nʳ * λ_r

    log_λ = log10(λ_r)
    I_evap = table(log_λ)

    # Evaporation rate (Mason 1971, PSD-integrated):
    #   dm/dt per drop = 4π × C × f_v × (S-1)/Φ,  C = D/2 (spherical capacitance)
    #   dq^r/dt = N_0 × ∫ 4π × (D/2) × f_v × exp(-λD) dD × (S-1)/Φ
    #           = 2π × N_0 × I_evap × (S-1) / Φ,  I_evap = ∫ D × f_v × exp(-λD) dD
    return FT(2π) * N_0 * I_evap * (S - 1) / thermodynamic_factor
end

# Mean-mass fallback (used when evaporation field is not tabulated).
# NOTE: The tabulated path (via `tabulate(p3, :rain, CPU())`) is recommended
# for production use. It integrates D × f_v(D) × N(D) dD exactly over the
# PSD using the physical piecewise Gunn-Kinzer/Beard fall speed law.
# This fallback now uses the same piecewise Gunn-Kinzer/Beard formula
# (rain_fall_speed in quadrature.jl) for consistency with the tabulated path
# and terminal_velocities.jl.
@inline function rain_evaporation_rate(::AbstractRainIntegral, qʳ, nʳ, S,
                                        thermodynamic_factor, p3, prp, nu, D_v, ρ, FT)
    ρ_water = p3.water_density

    # Mean drop properties
    m_mean = safe_divide(qʳ, nʳ, FT(1e-12))
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_water))

    # Terminal velocity: 4-regime Gunn-Kinzer/Beard piecewise formula
    ρ₀ = prp.reference_air_density
    ρ_correction = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)
    V = rain_fall_speed(D_mean, ρ_correction)

    # Ventilation factor (Fortran P3 convention: Sc^(1/3) baked into RAIN_F2R=0.308)
    # Use reference viscosity RAIN_NU (not runtime nu) to match the table convention
    Re_term = sqrt(V * D_mean / FT(RAIN_NU))
    f_v = FT(0.78) + FT(RAIN_F2R) * Re_term

    # Evaporation rate per drop (negative for evaporation)
    dm_dt = FT(4π) * (D_mean / 2) * f_v * (S - 1) / thermodynamic_factor

    return nʳ * dm_dt
end
