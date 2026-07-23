#####
##### Rain processes
#####
##### Autoconversion / accretion / self-collection are dispatched on
##### `p3.warm_rain_scheme` (default `KhairoutdinovKogan2000`). The three
##### implementations mirror Fortran P3 v5.5.0 `autoAccr_param` 1–3.
#####

"""
$(TYPEDSIGNATURES)

Compute rain autoconversion rate, dispatched on `p3.warm_rain_scheme`.

Cloud droplets larger than a threshold undergo collision-coalescence to form rain.

Available schemes:
- [`KhairoutdinovKogan2000`](@ref) (default): power-law in (qᶜˡ, Nᶜ)
- [`SeifertBeheng2001`](@ref): Long (1974) kernel with universal function
- [`Kogan2013`](@ref): updated power-law in (qᶜˡ, Nᶜ)

# Arguments
- `p3`: P3 microphysics scheme (provides parameters and scheme selector)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜ`: Cloud droplet number concentration [1/m³]
- `ρ`: Air density [kg/m³]
- `qʳ`: Rain mass fraction [kg/kg] (only consumed by `SeifertBeheng2001`;
        defaults to 0, which reduces SB2001 to its dry-cloud limit)

# Returns
- Rate of cloud → rain conversion [kg/kg/s]
"""
@inline rain_autoconversion_rate(p3, qᶜˡ, Nᶜ, ρ, qʳ = zero(qᶜˡ)) =
    rain_autoconversion_rate(p3.warm_rain_scheme, p3, qᶜˡ, Nᶜ, ρ, qʳ)

@inline function rain_autoconversion_rate(::KhairoutdinovKogan2000, p3, qᶜˡ, Nᶜ, ρ, qʳ)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    # Fortran P3 v5.5.0: no autoconversion when in-cloud qc < qsmall_dry1 (1e-8 kg/kg).
    qᶜˡ_eff = ifelse(qᶜˡ >= prp.autoconversion_threshold, clamp_positive(qᶜˡ), zero(FT))

    # Fortran KK2000 uses (nc × rho × 1e-6)^β where nc is per-mass [1/kg].
    # The nc × rho product is a unit conversion to per-volume [1/m³], so no
    # reference-density normalization is needed — Julia's Nᶜ is already per-volume.
    Nᶜ_scaled = Nᶜ / prp.autoconversion_reference_concentration

    # Khairoutdinov-Kogan (2000): ∂qʳ/∂t = k₁ × qᶜˡ^α × (Nᶜ/Nᶜ_ref)^β
    k₁ = prp.autoconversion_coefficient
    α = prp.autoconversion_exponent_cloud
    β = prp.autoconversion_exponent_droplet

    return k₁ * qᶜˡ_eff^α * Nᶜ_scaled^β
end

@inline function rain_autoconversion_rate(::Kogan2013, p3, qᶜˡ, Nᶜ, ρ, qʳ)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates
    qᶜˡ_eff = ifelse(qᶜˡ >= prp.autoconversion_threshold, clamp_positive(qᶜˡ), zero(FT))

    # Fortran: qcaut = 7.98e10 × qc^4.22 × (nc·1e-6·ρ)^(-3.01)
    # Julia Nᶜ is per-volume; Fortran applies `nc = max(nc, nsmall)` in
    # get_cloud_dsd2, so mirror that to keep Nᶜ^(-3.01) bounded.
    Nᶜ_eff = max(Nᶜ, ρ * p3.minimum_number_mixing_ratio)
    Nᶜ_cm3 = Nᶜ_eff * FT(1e-6)
    return FT(7.98e10) * qᶜˡ_eff^FT(4.22) * Nᶜ_cm3^FT(-3.01)
end

@inline function rain_autoconversion_rate(sb::SeifertBeheng2001, p3, qᶜˡ, Nᶜ, ρ, qʳ)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates
    qᶜˡ_eff = ifelse(qᶜˡ >= prp.autoconversion_threshold, clamp_positive(qᶜˡ), zero(FT))
    qʳ_eff = clamp_positive(qʳ)

    # Fortran kc = 9.44e9 (Long 1974 collection kernel coefficient).
    kc = FT(9.44e9)
    ν = sb2001_shape_parameter(sb, Nᶜ)

    # SB2001 universal function with x = qʳ / (qᶜˡ + qʳ) (dimensionless rain mass fraction).
    qsum_safe = max(qᶜˡ_eff + qʳ_eff, FT(1e-30))
    x = qʳ_eff / qsum_safe                              # Fortran: 1 - qc/(qc+qr)
    x68 = x^FT(0.68)
    Φau = FT(600) * x68 * (FT(1) - x68)^3               # Fortran 'dum1'

    # Universal-function denominator (1 - x)² → guard against x → 1.
    one_minus_x_sq = max((FT(1) - x)^2, FT(1e-30))

    # Fortran applies `nc = max(nc, nsmall)` in get_cloud_dsd2 before entering this
    # branch; mirror that so divisions by (ρ·nc·1e-6)² stay bounded when Nᶜ→0.
    Nᶜ_eff = max(Nᶜ, ρ * p3.minimum_number_mixing_ratio)
    ρqᶜ_g_cm3 = ρ * qᶜˡ_eff * FT(1e-3)                  # g/cm³ (Fortran units in formula)
    Nᶜ_cm3 = Nᶜ_eff * FT(1e-6)                          # cm⁻³ (Julia Nᶜ already per-volume)

    F_ν = (ν + FT(2)) * (ν + FT(4)) / (ν + FT(1))^2

    # Fortran:
    #   qcaut = kc × 1.9230769e-5 × F(ν) × (ρ qc · 1e-3)^4 / (ρ nc · 1e-6)^2
    #         × (1 + Φau / (1 - x)²) × 1000 / ρ
    return kc * FT(1.9230769e-5) * F_ν *
           ρqᶜ_g_cm3^4 / Nᶜ_cm3^2 *
           (FT(1) + Φau / one_minus_x_sq) * FT(1000) / ρ
end

"""
$(TYPEDSIGNATURES)

Compute rain accretion rate, dispatched on `p3.warm_rain_scheme`.

Falling rain drops collect cloud droplets via gravitational sweep-out. Available
schemes correspond to Fortran P3 v5.5.0 `autoAccr_param` 1–3; see
[`rain_autoconversion_rate`](@ref) for the scheme menu.

# Arguments
- `p3`: P3 microphysics scheme
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `qʳ`: Rain mass fraction [kg/kg]
- `ρ`: Air density [kg/m³] (only consumed by `SeifertBeheng2001`; defaults to 1)

# Returns
- Rate of cloud → rain conversion [kg/kg/s]
"""
@inline rain_accretion_rate(p3, qᶜˡ, qʳ, ρ = one(qᶜˡ)) =
    rain_accretion_rate(p3.warm_rain_scheme, p3, qᶜˡ, qʳ, ρ)

@inline function rain_accretion_rate(::KhairoutdinovKogan2000, p3, qᶜˡ, qʳ, ρ)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates
    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qʳ_eff = clamp_positive(qʳ)
    active = (qᶜˡ_eff >= p3.minimum_mass_mixing_ratio) &
             (qʳ_eff >= p3.minimum_mass_mixing_ratio)

    # KK2000 Eq. 5 (Fortran P3 form): ∂qʳ/∂t = k₂ × (qᶜˡ × qʳ)^α
    k₂ = prp.accretion_coefficient
    α = prp.accretion_exponent

    rate = k₂ * (qᶜˡ_eff * qʳ_eff)^α
    return ifelse(active, rate, zero(FT))
end

@inline function rain_accretion_rate(::Kogan2013, p3, qᶜˡ, qʳ, ρ)
    FT = typeof(qᶜˡ)
    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qʳ_eff = clamp_positive(qʳ)
    active = (qᶜˡ_eff >= p3.minimum_mass_mixing_ratio) &
             (qʳ_eff >= p3.minimum_mass_mixing_ratio)

    # Fortran: qcacc = 8.53 × qc^1.05 × qr^0.98
    rate = FT(8.53) * qᶜˡ_eff^FT(1.05) * qʳ_eff^FT(0.98)
    return ifelse(active, rate, zero(FT))
end

@inline function rain_accretion_rate(::SeifertBeheng2001, p3, qᶜˡ, qʳ, ρ)
    FT = typeof(qᶜˡ)
    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qʳ_eff = clamp_positive(qʳ)
    active = (qᶜˡ_eff >= p3.minimum_mass_mixing_ratio) &
             (qʳ_eff >= p3.minimum_mass_mixing_ratio)

    # Fortran kr = 5.78e3 (Long 1974 accretion kernel coefficient).
    kr = FT(5.78e3)

    # Universal function τ = 1 - qᶜˡ / (qᶜˡ + qʳ)
    qsum_safe = max(qᶜˡ_eff + qʳ_eff, FT(1e-30))
    τ = qʳ_eff / qsum_safe
    Φac = (τ / (τ + FT(5e-4)))^4  # Fortran 'dum1' in accretion branch

    # Fortran: qcacc = kr × ρ × 1e-3 × qᶜˡ × qʳ × Φac
    rate = kr * ρ * FT(1e-3) * qᶜˡ_eff * qʳ_eff * Φac
    return ifelse(active, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Compute rain self-collection rate (number tendency only). Dispatches on
`p3.warm_rain_scheme`.

Large rain drops collect smaller ones, reducing number but conserving mass.
KK2000 (default) and SB2001 share the same linear form `k_rr × ρ × qʳ × nʳ`
(Fortran `kr × 1e-3 = 5.78`); [Kogan (2013)](@cite Kogan2013) uses a separate power-law form.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters and scheme selector)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Rate of rain number loss [1/kg/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline rain_self_collection_rate(p3, qʳ, nʳ, ρ) =
    rain_self_collection_rate(p3.warm_rain_scheme, p3, qʳ, nʳ, ρ)

@inline function rain_self_collection_rate(::Union{KhairoutdinovKogan2000, SeifertBeheng2001},
                                           p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates
    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = bounded_rain_number(nʳ, qʳ_eff, prp)
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio

    # KK2000 / SB2001: |∂nʳ/∂t| = k_rr × ρ × qʳ × nʳ
    k_rr = prp.rain_self_collection_coefficient
    rate = k_rr * ρ * qʳ_eff * nʳ_eff
    return ifelse(active, rate, zero(FT))
end

@inline function rain_self_collection_rate(::Kogan2013, p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates
    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = bounded_rain_number(nʳ, qʳ_eff, prp)
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio

    # Fortran: nrslf_base = 205. × qr^1.55 × (nr × 1e-6 × ρ)^0.6 × 1e6 / ρ
    # (nrslf is multiplied by the Verlinde-Cotton breakup modifier 'dum' downstream
    # in `rain_breakup_rate`; here we return the unmodified base rate.)
    nʳ_per_volume = max(nʳ_eff * ρ * FT(1e-6), FT(1e-30))
    rate = FT(205) * qʳ_eff^FT(1.55) * nʳ_per_volume^FT(0.6) * FT(1e6) / max(ρ, eps(FT))
    return ifelse(active, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Compute rain breakup rate following Fortran P3 v5.5.0.

Large rain drops spontaneously break up into smaller fragments, producing
a number source that counterbalances self-collection. Uses a two-piece
function of ``D_r = (q_r / (π ρ_w n_r))^{1/3} = 1/λ_r`` (Fortran convention,
no factor of 6; this equals the mean-mass diameter for an exponential DSD):

1. ``D_r < D_{th}``: No breakup effect (modifier = 1, breakup = 0)
2. ``D_r ≥ D_{th}``: ``\\text{modifier} = 2 - \\exp(κ_{br} (D_r - D_{th}))``, breakup > 0

The breakup rate is ``(1 - \\text{modifier}) \\times`` self-collection rate.

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
    nʳ_eff = bounded_rain_number(nʳ, qʳ_eff, prp)

    # Fortran P3 convention: D_r = 1/λ_r after `get_rain_dsd2` applies
    # the rain lambda limiter and recomputes the DSD-consistent number.
    λ_r = rain_slope_parameter(qʳ_eff, nʳ_eff, prp)
    D_r = 1 / λ_r

    # Two-piece breakup function (Fortran P3 v5.5.0)
    D_th = prp.rain_breakup_diameter_threshold  # 280 μm: breakup threshold (1/λ_r convention)
    κ_br = prp.rain_breakup_coefficient         # 2300 m⁻¹: exponential coefficient

    # Clamp exp argument to prevent Float32 overflow (exp(88.7) ≈ 3.4e38 = maxfloat).
    # Without the clamp, LLVM PTX may fuse the ifelse and multiply, producing
    # (Inf - 1) * 0 = NaN when D_r is large but self_collection ≈ 0.
    exp_arg = min(κ_br * (D_r - D_th), FT(80))
    breakup_modifier = ifelse(D_r < D_th,
                              FT(1),
                              FT(2) - exp(exp_arg))

    # Breakup rate: (1 - breakup_modifier) × self_collection
    # When D_r < D_th: modifier = 1 → breakup = 0 (no effect)
    # When D_r ≥ D_th: modifier < 1 → breakup > 0 (number source)
    # self_collection is positive magnitude (M7); breakup is positive (number source).
    rate = (FT(1) - breakup_modifier) * self_collection
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio
    return ifelse(active, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

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
\\frac{dm}{dt} = \\frac{4\\pi C f_v (S - 1)}{\\frac{ℒˡ}{K_a T}(\\frac{ℒˡ}{R_v T} - 1)
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
    Rᵛ = FT(VAPOR_GAS_CONSTANT)
    Rᵈ = FT(DRY_AIR_GAS_CONSTANT)
    ℒˡ = vaporization_latent_heat(nothing, T)  # Latent heat of vaporization [J/kg]
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
    A = ℒˡ / (K_a * T) * (ℒˡ / (Rᵛ * T) - 1)
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
    # Diagnose λ_r from (q_r, N_r) for exponential DSD (μ_r = 0):
    #   q_r = N_r * <m> = N_r * π ρ_w / λ_r³  ⟹  λ_r = (π ρ_w / m̄)^(1/3)
    λ_r = rain_slope_parameter(qʳ, nʳ, prp)
    nʳ_bounded = rain_number_from_slope(qʳ, λ_r, prp)

    # Intercept N_0 = N_r * λ_r  (for exponential DSD N'(D) = N_0 exp(-λ D))
    N_0 = nʳ_bounded * λ_r

    log_λ = log10(λ_r)
    I_VD = table(log_λ)

    # Combine constant + velocity-diameter terms with T,P-dependent transport.
    # Constant term: f1r × ∫ D × exp(-λD) dD = f1r / λ² (analytical for μ_r=0)
    I_const = FT(RAIN_F1R) / (λ_r * λ_r)
    # Table stores ∫ D √(V×D) exp(-λD) dD (no ν); apply 1/√ν at runtime.
    Sc_cbrt = cbrt(nu / max(D_v, FT(1e-10)))
    inv_sqrt_nu = 1 / sqrt(max(nu, FT(1e-10)))
    I_evap = I_const + FT(RAIN_F2R) * Sc_cbrt * inv_sqrt_nu * I_VD

    # Evaporation rate (Mason 1971, PSD-integrated):
    #   dm/dt per drop = 4π × C × f_v × (S-1)/Φ,  C = D/2 (spherical capacitance)
    #   dq^r/dt = N_0 × ∫ 4π × (D/2) × f_v × exp(-λD) dD × (S-1)/Φ
    #           = 2π × N_0 × I_evap × (S-1) / Φ,  I_evap = ∫ D × f_v × exp(-λD) dD
    return 2 * FT(π) * N_0 * I_evap * (S - 1) / thermodynamic_factor
end

"""
$(TYPEDSIGNATURES)

Compute rain condensation rate (vapor → rain) when the air is supersaturated.

Uses the same Mason (1971) diffusional growth framework as rain evaporation:
when the saturation ratio ``S > 1``, the growth rate is positive, representing
direct condensation of vapor onto existing rain drops. This mirrors the Fortran
P3 v5.5.0 semi-analytic framework where ``q_{rcon}`` can be positive.

# Returns
- Rate of vapor → rain condensation [kg/kg/s] (positive magnitude)
"""
@inline function rain_condensation_rate(p3, qʳ, nʳ, qᵛ, qᵛ⁺ˡ, T, ρ, P,
                                        transport=air_transport_properties(T, P))
    FT = typeof(qʳ)
    prp = p3.process_rates

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # Only condense in supersaturated conditions with existing rain
    S = qᵛ / max(qᵛ⁺ˡ, FT(1e-10))
    is_supersaturated = (S > 1) & (qʳ_eff > FT(1e-14))

    # Thermodynamic constants (same as rain evaporation)
    Rᵛ = FT(VAPOR_GAS_CONSTANT)
    Rᵈ = FT(DRY_AIR_GAS_CONSTANT)
    ℒˡ = vaporization_latent_heat(nothing, T)
    K_a = transport.K_a
    D_v = transport.D_v
    nu  = transport.nu

    ε = Rᵈ / Rᵛ
    qᵛ⁺ˡ_safe = max(qᵛ⁺ˡ, FT(1e-30))
    e_s = P * qᵛ⁺ˡ_safe / (ε + qᵛ⁺ˡ_safe * (1 - ε))

    # Thermodynamic resistance (Mason 1971)
    A = ℒˡ / (K_a * T) * (ℒˡ / (Rᵛ * T) - 1)
    B = Rᵛ * T / (e_s * D_v)
    thermodynamic_factor = max(A + B, FT(1e-10))

    # Diffusional growth rate (reuse evaporation ventilation integral)
    # Positive when S > 1 (condensation)
    raw_rate = rain_evaporation_rate(p3.rain.evaporation, qʳ_eff, nʳ_eff, S,
                                      thermodynamic_factor, p3, prp, nu, D_v, ρ, FT)
    cond_rate = clamp_positive(raw_rate)

    # Limit condensation to available vapor (Fortran: min(qrcon, qv*i_dt))
    τ = prp.sink_limiting_timescale
    max_cond = clamp_positive(qᵛ - qᵛ⁺ˡ) / τ
    cond_rate = min(cond_rate, max_cond)

    return ifelse(is_supersaturated, cond_rate, zero(FT))
end

#####
##### Scheme-dependent helpers shared by autoconv/accretion/number tendencies
#####

"""
$(TYPEDSIGNATURES)

Cloud-droplet self-collection rate (number loss in cloud, not rain).

Dispatched on `p3.warm_rain_scheme`. For KK2000 and Kogan2013 this is zero
(Fortran sets `ncslf = 0` in those branches). For SB2001:

    |∂Nᶜ/∂t|_self = kc × (ρ qᶜˡ × 1e-3)² × (ν+2)/(ν+1) × 1e6 / ρ

following Fortran P3 v5.5.0. Returned as a positive magnitude.

Note: the Fortran formula contains a `+ncautc` term that cancels the
double-count of autoconversion-derived cloud-number loss when assembled with
`-ncautc` in the final tendency. Here we return only the genuine self-collection
magnitude; the autoconversion-derived loss is handled separately by
[`cloud_number_loss_from_autoconversion`](@ref). The asymmetry with
[`rain_self_collection_rate`](@ref) — which uses `bounded_rain_number` —
is intentional: there is no analogous DSD-shape bound for the cloud branch in
Fortran's `ncslf` formula.
"""
@inline cloud_self_collection_rate(p3, qᶜˡ, Nᶜ, ρ) =
    cloud_self_collection_rate(p3.warm_rain_scheme, p3, qᶜˡ, Nᶜ, ρ)

@inline cloud_self_collection_rate(::Union{KhairoutdinovKogan2000, Kogan2013},
                                   p3, qᶜˡ, Nᶜ, ρ) = zero(qᶜˡ)

@inline function cloud_self_collection_rate(sb::SeifertBeheng2001, p3, qᶜˡ, Nᶜ, ρ)
    FT = typeof(qᶜˡ)
    qᶜˡ_eff = clamp_positive(qᶜˡ)
    active = qᶜˡ_eff >= p3.minimum_mass_mixing_ratio
    kc = FT(9.44e9)
    ν = sb2001_shape_parameter(sb, Nᶜ)

    ρqᶜ_g_cm3 = ρ * qᶜˡ_eff * FT(1e-3)
    rate = kc * ρqᶜ_g_cm3^2 * (ν + FT(2)) / (ν + FT(1)) * FT(1e6) / ρ
    return ifelse(active, rate, zero(FT))
end

@inline function sb2001_shape_parameter(::SeifertBeheng2001{Nothing}, Nᶜ)
    FT = typeof(Nᶜ)
    μ_c = liu_daum_shape_parameter(Nᶜ)
    dnu = (FT(-0.947), FT(-0.871), FT(-0.783), FT(-0.688),
           FT(-0.588), FT(-0.486), FT(-0.382), FT(-0.277),
           FT(-0.171), FT(-0.064), FT(0.044), FT(0.152),
           FT(0.260), FT(0.369), FT(0.478), FT(0.588))
    index = min(Int(floor(μ_c)) + 1, 15)
    return dnu[index] + (dnu[index + 1] - dnu[index]) * (μ_c - index)
end

@inline sb2001_shape_parameter(sb::SeifertBeheng2001, Nᶜ) =
    oftype(Nᶜ, sb.ν)

"""
$(TYPEDSIGNATURES)

Cloud-droplet number loss from autoconversion (mass → drop count conversion),
dispatched on `p3.warm_rain_scheme`. Returned as a positive magnitude.

Fortran convention:
- KK2000 / Kogan2013: `ncautc = qcaut × Nᶜ / qᶜˡ` (cloud number lost in proportion
  to mass lost).
- SB2001: no net cloud-number loss from autoconversion, because Fortran
  assembles `-ncautc + ncslf` and `ncslf` contains a matching `+ncautc`.
"""
@inline cloud_number_loss_from_autoconversion(p3, qcaut, qᶜˡ, Nᶜ, ρ) =
    cloud_number_loss_from_autoconversion(p3.warm_rain_scheme, p3, qcaut, qᶜˡ, Nᶜ, ρ)

@inline function cloud_number_loss_from_autoconversion(::Union{KhairoutdinovKogan2000, Kogan2013},
                                                       p3, qcaut, qᶜˡ, Nᶜ, ρ)
    FT = typeof(qcaut)
    # Fortran ncautc = qcaut × nc / qc, where nc = Nᶜ/ρ. The Julia equivalent is
    # qcaut × Nᶜ / (ρ qᶜˡ); safe_divide guards qᶜˡ = 0.
    nc_over_qc = safe_divide(Nᶜ, ρ * qᶜˡ, zero(FT))
    return qcaut * nc_over_qc
end

@inline cloud_number_loss_from_autoconversion(::SeifertBeheng2001, p3, qcaut, qᶜˡ, Nᶜ, ρ) =
    zero(qcaut)

"""
$(TYPEDSIGNATURES)

Mass per newly-formed rain drop produced by autoconversion, dispatched on
`p3.warm_rain_scheme`. Used to convert autoconversion mass rate into a rain
number source.

Fortran values:
- KK2000: mass of 25 μm radius drop ≈ 6.545e-11 kg (`cons3⁻¹`); uses
  `p3.process_rates.initial_rain_drop_mass` so the radius is user-configurable.
- Kogan2013: mass of 40 μm radius drop ≈ 2.681e-10 kg (`cons8⁻¹`); hardcoded
  to match Fortran.
- SB2001: `2 / 7.6923076e9` ≈ 2.6e-10 kg. Fortran assembles
  `nr += 0.5 × ncautc × dt` with `ncautc = qcaut × 7.6923076e9`, so the
  effective seed mass is `2 / 7.6923076e9`.
"""
@inline rain_seed_drop_mass(p3) = rain_seed_drop_mass(p3.warm_rain_scheme, p3)

@inline rain_seed_drop_mass(::KhairoutdinovKogan2000, p3) = p3.process_rates.initial_rain_drop_mass

@inline function rain_seed_drop_mass(::Kogan2013, p3)
    FT = typeof(p3.process_rates.initial_rain_drop_mass)
    ρʷ = p3.process_rates.liquid_water_density
    return FT(4) * FT(π) / FT(3) * ρʷ * FT(40e-6)^3
end

@inline function rain_seed_drop_mass(::SeifertBeheng2001, p3)
    FT = typeof(p3.process_rates.initial_rain_drop_mass)
    return FT(2 / 7.6923076e9)
end
