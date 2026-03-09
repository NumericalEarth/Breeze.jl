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
    # Note: The Fortran P3 computes T,P-dependent transport properties
    # (dv = 8.794e-5*T^1.81/P, kap = 1414*mu). These constants represent
    # near-surface values. With PSD lookup tables (Phase 5), transport
    # properties should use air_transport_properties(T, P) instead.
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

    # Terminal velocity for rain drops
    # Note: The Fortran P3 uses ar=842, br=0.8, f1r=0.78, f2r=0.32 with
    # PSD-integrated ventilation via lookup tables. For the mean-mass
    # approximation, V=130*D^0.5 gives better PSD-effective ventilation
    # because it overestimates V for small drops, partially compensating
    # for the PSD tail where small drops evaporate efficiently.
    # TODO: Switch to ar/br when PSD lookup tables are implemented.
    V = FT(130) * D_mean^FT(0.5)

    # Ventilation factor
    ν = FT(1.5e-5)
    Re_term = sqrt(V * D_mean / ν)
    f_v = FT(0.78) + FT(0.31) * Re_term

    # Thermodynamic resistance (Mason 1971)
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
