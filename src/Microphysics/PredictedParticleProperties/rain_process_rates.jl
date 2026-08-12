#####
##### Rain processes
#####
##### Autoconversion / accretion / self-collection are dispatched on
##### `p3.warm_rain_scheme` (default and only scheme:
##### `KhairoutdinovKogan2000`, Fortran P3 v5.5.0 `autoAccr_param = 2`).
#####

"""
$(TYPEDSIGNATURES)

Compute rain autoconversion rate, dispatched on `p3.warm_rain_scheme`.

Cloud droplets larger than a threshold undergo collision-coalescence to form rain.

Available schemes:
- [`KhairoutdinovKogan2000`](@ref) (default): power-law in (qᶜˡ, Nᶜ)

# Arguments
- `p3`: P3 microphysics scheme (provides parameters and scheme selector)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜ`: Cloud droplet number concentration [1/m³]
- `ρ`: Air density [kg/m³]
- `qʳ`: Rain mass fraction [kg/kg] (unused by KK2000; retained for the
        scheme-dispatch signature)

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

"""
$(TYPEDSIGNATURES)

Compute rain accretion rate, dispatched on `p3.warm_rain_scheme`.

Falling rain drops collect cloud droplets via gravitational sweep-out. See
[`rain_autoconversion_rate`](@ref) for the scheme menu.

# Arguments
- `p3`: P3 microphysics scheme
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `qʳ`: Rain mass fraction [kg/kg]
- `ρ`: Air density [kg/m³] (unused by KK2000; defaults to 1)

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

"""
$(TYPEDSIGNATURES)

Compute rain self-collection rate (number tendency only). Dispatches on
`p3.warm_rain_scheme`.

Large rain drops collect smaller ones, reducing number but conserving mass.
KK2000 uses the linear form `k_rr × ρ × qʳ × nʳ` (Fortran `kr × 1e-3 = 5.78`).

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

@inline function rain_self_collection_rate(::KhairoutdinovKogan2000, p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates
    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = bounded_rain_number(nʳ, qʳ_eff, prp)
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio

    # KK2000: |∂nʳ/∂t| = k_rr × ρ × qʳ × nʳ
    k_rr = prp.rain_self_collection_coefficient
    rate = k_rr * ρ * qʳ_eff * nʳ_eff
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

# Mason (1971) thermodynamic resistance Φ = A + B for diffusional growth at a
# liquid surface, shared by rain evaporation and rain condensation. `e_s` is
# recovered by inverting qᵛ⁺ˡ = ε e_s / (P - (1 - ε) e_s), consistent with the
# ice deposition path.
@inline function mason_thermodynamic_factor(qᵛ⁺ˡ, T, P, transport, prp, FT)
    Rᵛ = FT(VAPOR_GAS_CONSTANT)
    Rᵈ = FT(DRY_AIR_GAS_CONSTANT)
    ℒˡ = vaporization_latent_heat(nothing, T)  # Latent heat of vaporization [J/kg]
    Kᵃ = transport.Kᵃ                        # Thermal conductivity of air [W/m/K]
    Dᵛ = transport.Dᵛ                        # Diffusivity of water vapor [m²/s]

    ε = Rᵈ / Rᵛ
    qᵛ⁺ˡ_safe = max(qᵛ⁺ˡ, FT(prp.floors.divisor))
    e_s = P * qᵛ⁺ˡ_safe / (ε + qᵛ⁺ˡ_safe * (1 - ε))

    A = ℒˡ / (Kᵃ * T) * (ℒˡ / (Rᵛ * T) - 1)
    B = Rᵛ * T / (e_s * Dᵛ)
    # Φ is only ever a divisor, so the floor is the generic division guard rather than
    # a physical threshold. It never binds: A ≈ 7×10⁶ and B ≈ 9×10⁶ near freezing at
    # 1 atm, and the degenerate limits raise the sum rather than lower it (Kᵃ → 0 sends
    # A → ∞, Dᵛ → 0 sends B → ∞).
    return max(A + B, FT(prp.floors.divisor))
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
\\frac{dm}{dt} = \\frac{4\\pi C f_v (S - 1)}{\\frac{ℒˡ}{Kᵃ T}(\\frac{ℒˡ}{R_v T} - 1)
               + \\frac{R_v T}{e_s Dᵛ}},\\quad C = D/2
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
    S = qᵛ / max(qᵛ⁺ˡ, FT(prp.floors.saturation_mass_fraction))
    is_subsaturated = S < 1

    # T,P-dependent transport properties (pre-computed or computed on demand)
    Dᵛ = transport.Dᵛ       # Diffusivity of water vapor [m²/s]
    ν  = transport.ν        # Kinematic viscosity [m²/s]

    thermodynamic_factor = mason_thermodynamic_factor(qᵛ⁺ˡ, T, P, transport, prp, FT)

    # Internal helpers return negative (S - 1 < 0 when subsaturated).
    # Negate to get positive magnitude (M7 sign convention).
    evap_rate = -rain_evaporation_rate(p3.rain.evaporation, qʳ_eff, nʳ_eff, S,
                                        thermodynamic_factor, p3, prp, ν, Dᵛ, ρ, FT)

    # Cannot evaporate more than available
    τ_evap = prp.rain_evaporation_timescale
    max_evap = qʳ_eff / τ_evap
    evap_rate = min(evap_rate, max_evap)

    return ifelse(is_subsaturated, evap_rate, zero(FT))
end

# Tabulated path: use PSD-integrated ventilation integral I_evap(λ_r)
@inline function rain_evaporation_rate(table::TabulatedFunction1D, qʳ, nʳ, S,
                                        thermodynamic_factor, p3, prp, ν, Dᵛ, ρ, FT)
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
    coefficient_floor = FT(prp.floors.transport_coefficient)
    Sc_cbrt = cbrt(ν / max(Dᵛ, coefficient_floor))
    inv_sqrt_nu = 1 / sqrt(max(ν, coefficient_floor))
    I_evap = I_const + FT(RAIN_F2R) * Sc_cbrt * inv_sqrt_nu * I_VD

    # Evaporation rate (Mason 1971, PSD-integrated):
    #   dm/dt per drop = 4π × C × f_v × (S-1)/Φ,  C = D/2 (spherical capacitance)
    #   dq^r/dt = N_0 × ∫ 4π × (D/2) × f_v × exp(-λD) dD × (S-1)/Φ
    #           = 2π × N_0 × I_evap × (S-1) / Φ,  I_evap = ∫ D × f_v × exp(-λD) dD
    return 2 * FT(π) * N_0 * I_evap * (S - 1) / thermodynamic_factor
end

#####
##### Scheme-dependent helpers shared by autoconv/accretion/number tendencies
#####

"""
$(TYPEDSIGNATURES)

Cloud-droplet self-collection rate (number loss in cloud, not rain).

Dispatched on `p3.warm_rain_scheme`. Zero for KK2000 (Fortran sets
`ncslf = 0` in that branch). Returned as a positive magnitude.
"""
@inline cloud_self_collection_rate(p3, qᶜˡ, Nᶜ, ρ) =
    cloud_self_collection_rate(p3.warm_rain_scheme, p3, qᶜˡ, Nᶜ, ρ)

@inline cloud_self_collection_rate(::KhairoutdinovKogan2000, p3, qᶜˡ, Nᶜ, ρ) = zero(qᶜˡ)

"""
$(TYPEDSIGNATURES)

Cloud-droplet number loss from autoconversion (mass → drop count conversion),
dispatched on `p3.warm_rain_scheme`. Returned as a positive magnitude.

Fortran convention for KK2000: `ncautc = qcaut × Nᶜ / qᶜˡ` (cloud number lost in
proportion to mass lost).
"""
@inline cloud_number_loss_from_autoconversion(p3, qcaut, qᶜˡ, Nᶜ, ρ) =
    cloud_number_loss_from_autoconversion(p3.warm_rain_scheme, p3, qcaut, qᶜˡ, Nᶜ, ρ)

@inline function cloud_number_loss_from_autoconversion(::KhairoutdinovKogan2000,
                                                       p3, qcaut, qᶜˡ, Nᶜ, ρ)
    FT = typeof(qcaut)
    # Fortran ncautc = qcaut × nc / qc, where nc = Nᶜ/ρ. The Julia equivalent is
    # qcaut × Nᶜ / (ρ qᶜˡ); safe_divide guards qᶜˡ = 0.
    nc_over_qc = safe_divide(Nᶜ, ρ * qᶜˡ, zero(FT))
    return qcaut * nc_over_qc
end

"""
$(TYPEDSIGNATURES)

Mass per newly-formed rain drop produced by autoconversion, dispatched on
`p3.warm_rain_scheme`. Used to convert autoconversion mass rate into a rain
number source.

Fortran value for KK2000: mass of a 25 μm radius drop ≈ 6.545e-11 kg
(`cons3⁻¹`); read from `p3.process_rates.initial_rain_drop_mass` so the radius
is user-configurable.
"""
@inline rain_seed_drop_mass(p3) = rain_seed_drop_mass(p3.warm_rain_scheme, p3)

@inline rain_seed_drop_mass(::KhairoutdinovKogan2000, p3) = p3.process_rates.initial_rain_drop_mass
