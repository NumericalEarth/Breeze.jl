#####
##### Rain processes
#####
##### Autoconversion / accretion / self-collection are dispatched on
##### `p3.warm_rain_scheme` (default and only scheme: `KhairoutdinovKogan2000`).
#####

"""
$(TYPEDSIGNATURES)

Compute rain autoconversion rate, dispatched on `p3.warm_rain_scheme`.

Cloud droplets larger than a threshold undergo collision-coalescence to form rain.
For the KK2000 branch,

```math
\\dot q^{r}_{\\mathrm{auto}} = \\mathbb{C}_{\\mathrm{auto},1}
    (q^{cl})^{\\mathbb{C}_{\\mathrm{auto},2}}
    \\left(\\frac{N^{cl}}{N^{cl}_r}\\right)^{\\mathbb{C}_{\\mathrm{auto},3}},
```

gated to zero below ``q^{cl} = \\mathbb{C}_{\\mathrm{auto},4}``. The reference
concentration ``N^{cl}_r`` fixes units and is not an independently identifiable
free parameter.

Available schemes:
- [`KhairoutdinovKogan2000`](@ref) (default): power-law in (qᶜˡ, Nᶜˡ)

# Arguments
- `p3`: P3 microphysics scheme (provides parameters and scheme selector)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜˡ`: Cloud droplet number concentration [1/m³]
- `ρ`: Air density [kg/m³]
- `qʳ`: Rain mass fraction [kg/kg] (unused by KK2000; retained for the
        scheme-dispatch signature)

# Returns
- Rate of cloud → rain conversion [kg/kg/s]
"""
@inline rain_autoconversion_rate(p3, qᶜˡ, Nᶜˡ, ρ, qʳ = zero(qᶜˡ)) =
    rain_autoconversion_rate(p3.warm_rain_scheme, p3, qᶜˡ, Nᶜˡ, ρ, qʳ)

@inline function rain_autoconversion_rate(::KhairoutdinovKogan2000, p3, qᶜˡ, Nᶜˡ, ρ, qʳ)
    FT = typeof(qᶜˡ)
    parameters = p3.process_rates

    ℂᵃᵘᵗᵒ₁ = parameters.autoconversion_coefficient
    ℂᵃᵘᵗᵒ₂ = parameters.autoconversion_exponent_cloud
    ℂᵃᵘᵗᵒ₃ = parameters.autoconversion_exponent_droplet
    ℂᵃᵘᵗᵒ₄ = parameters.autoconversion_threshold

    qᶜˡ_eff = ifelse(qᶜˡ >= ℂᵃᵘᵗᵒ₄, max(0, qᶜˡ), zero(FT))

    # KK2000 scales with droplet number per volume, so no reference-density
    # normalization is needed — Nᶜˡ is already per-volume [1/m³].
    scaled_cloud_number = Nᶜˡ / parameters.autoconversion_reference_concentration

    return ℂᵃᵘᵗᵒ₁ * qᶜˡ_eff^ℂᵃᵘᵗᵒ₂ * scaled_cloud_number^ℂᵃᵘᵗᵒ₃
end

"""
$(TYPEDSIGNATURES)

Compute rain accretion rate, dispatched on `p3.warm_rain_scheme`.

Falling rain drops collect cloud droplets via gravitational sweep-out. See
[`rain_autoconversion_rate`](@ref) for the scheme menu.

```math
\\dot q^{r}_{\\mathrm{accr}} = \\mathbb{C}_{\\mathrm{accr},1}
    (q^{cl} q^r)^{\\mathbb{C}_{\\mathrm{accr},2}}.
```

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
    parameters = p3.process_rates
    qᶜˡ_eff = max(0, qᶜˡ)
    qʳ_eff = max(0, qʳ)
    active = (qᶜˡ_eff >= p3.minimum_mass_mixing_ratio) &
             (qʳ_eff >= p3.minimum_mass_mixing_ratio)

    ℂᵃᶜᶜʳ₁ = parameters.accretion_coefficient
    ℂᵃᶜᶜʳ₂ = parameters.accretion_exponent

    rate = ℂᵃᶜᶜʳ₁ * (qᶜˡ_eff * qʳ_eff)^ℂᵃᶜᶜʳ₂
    return ifelse(active, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Compute rain self-collection rate (number tendency only). Dispatches on
`p3.warm_rain_scheme`.

Large rain drops collect smaller ones, reducing number but conserving mass.
KK2000 uses ``\\dot n^r_{\\mathrm{self}} = \\mathbb{C}_{\\mathrm{self},1} ρ q^r n^r``,
with ``\\mathbb{C}_{\\mathrm{self},1} = 5.78`` m³ kg⁻¹ s⁻¹ by default.

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
    parameters = p3.process_rates
    qʳ_eff = max(0, qʳ)
    nʳ_eff = bounded_rain_number(nʳ, qʳ_eff, parameters)
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio

    ℂˢᵉˡᶠ₁ = parameters.rain_self_collection_coefficient
    rate = ℂˢᵉˡᶠ₁ * ρ * qʳ_eff * nʳ_eff
    return ifelse(active, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Compute rain breakup rate.

Large rain drops spontaneously break up into smaller fragments, producing
a number source that counterbalances self-collection. Uses a two-piece
function of ``\\bar D^r = (q^r / (π ρ^L n^r))^{1/3} = 1/λ^r``. For the
exponential rain DSD this is the number-mean diameter; the diameter of the mean
particle mass is ``6^{1/3} \\bar D^r``.

1. ``\\bar D^r < \\mathbb{C}_{\\mathrm{brkp},1}``: no breakup effect.
2. ``\\bar D^r ≥ \\mathbb{C}_{\\mathrm{brkp},1}``:
   ``f_{brkp} = 2 - \\exp[\\mathbb{C}_{\\mathrm{brkp},2}
   (\\bar D^r - \\mathbb{C}_{\\mathrm{brkp},1})]``.

The breakup source is ``(1 - f_{brkp})`` times the self-collection sink. The
net rain-number tendency changes sign only when ``f_{brkp} = 0``, at
``\\bar D^r = \\mathbb{C}_{\\mathrm{brkp},1} + \\log(2) / \\mathbb{C}_{\\mathrm{brkp},2}``.

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
    parameters = p3.process_rates

    qʳ_eff = max(0, qʳ)
    nʳ_eff = bounded_rain_number(nʳ, qʳ_eff, parameters)

    # Dʳ = 1/λʳ, evaluated after the rain lambda limiter has been applied and the
    # DSD-consistent number recomputed.
    λʳ = rain_slope_parameter(qʳ_eff, nʳ_eff, parameters)
    mean_rain_diameter = 1 / λʳ

    ℂᵇʳᵏᵖ₁ = parameters.rain_breakup_diameter_threshold
    ℂᵇʳᵏᵖ₂ = parameters.rain_breakup_coefficient

    # Clamp exp argument to prevent Float32 overflow (exp(88.7) ≈ 3.4e38 = maxfloat).
    # Without the clamp, LLVM PTX may fuse the ifelse and multiply, producing
    # (Inf - 1) * 0 = NaN when Dʳ is large but self_collection ≈ 0.
    exponential_argument = min(ℂᵇʳᵏᵖ₂ * (mean_rain_diameter - ℂᵇʳᵏᵖ₁), FT(80))
    breakup_modifier = ifelse(mean_rain_diameter < ℂᵇʳᵏᵖ₁,
                              FT(1),
                              FT(2) - exp(exponential_argument))

    # Breakup rate: (1 - breakup_modifier) × self_collection
    # When Dʳ < Dᵗʰ: modifier = 1 → breakup = 0 (no effect)
    # When Dʳ ≥ Dᵗʰ: modifier < 1 → breakup > 0 (number source)
    # self_collection is positive magnitude (M7); breakup is positive (number source).
    rate = (FT(1) - breakup_modifier) * self_collection
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio
    return ifelse(active, rate, zero(FT))
end

# Mason (1971) thermodynamic resistance Φ = A + B for diffusional growth at a
# liquid surface, shared by rain evaporation and rain condensation. `e_s` is
# recovered by inverting qᵛ⁺ˡ = ε e_s / (P - (1 - ε) e_s), consistent with the
# ice deposition path.
@inline function mason_thermodynamic_factor(qᵛ⁺ˡ, T, P, constants, transport, parameters, FT)
    Rᵛ = FT(vapor_gas_constant(constants))
    Rᵈ = FT(dry_air_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)  # Latent heat of vaporization [J/kg]
    Kᵃ = transport.Kᵃ                        # Thermal conductivity of air [W/m/K]
    Dᵛ = transport.Dᵛ                        # Diffusivity of water vapor [m²/s]

    ε = Rᵈ / Rᵛ
    qᵛ⁺ˡ_safe = max(qᵛ⁺ˡ, FT(parameters.floors.divisor))
    e_s = P * qᵛ⁺ˡ_safe / (ε + qᵛ⁺ˡ_safe * (1 - ε))

    A = ℒˡ / (Kᵃ * T) * (ℒˡ / (Rᵛ * T) - 1)
    B = Rᵛ * T / (e_s * Dᵛ)
    # Φ is only ever a divisor, so the floor is the generic division guard rather than
    # a physical threshold. It never binds: A ≈ 7×10⁶ and B ≈ 9×10⁶ near freezing at
    # 1 atm, and the degenerate limits raise the sum rather than lower it (Kᵃ → 0 sends
    # A → ∞, Dᵛ → 0 sends B → ∞).
    return max(A + B, FT(parameters.floors.divisor))
end

"""
$(TYPEDSIGNATURES)

Compute rain evaporation rate using ventilation-enhanced diffusion.

Rain drops evaporate when the ambient air is subsaturated (qᵛ < qᵛ⁺ˡ).
The evaporation rate is enhanced by ventilation (air flow around falling drops).

`p3.rain.evaporation` is the tabulated ventilation integral built by
`tabulate_rain_from_quadrature`. The inner method computes λʳ from (qʳ, Nʳ), looks up
`I_evap(λʳ) = ∫ D fᵛᵉ(D) exp(-λʳ D) dD`, then applies
`dqʳ/dt = 2π × Nʳ₀ × I_evap × (S-1) / thermo_factor`
(Mason 1971, capacitance C = D/2 so 4πC = 2πD).

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
- `constants`: Thermodynamic constants

# Returns
- Rate of rain evaporation [kg/kg/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline function rain_evaporation_rate(p3, qʳ, nʳ, qᵛ, qᵛ⁺ˡ, T, ρ, P,
                                       constants,
                                       transport=air_transport_properties(T, P, constants))
    FT = typeof(qʳ)
    parameters = p3.process_rates

    qʳ_eff = max(0, qʳ)
    nʳ_eff = max(0, nʳ)

    # Only evaporate in subsaturated conditions
    S = qᵛ / max(qᵛ⁺ˡ, FT(parameters.floors.saturation_mass_fraction))
    is_subsaturated = S < 1

    # T,P-dependent transport properties (pre-computed or computed on demand)
    Dᵛ = transport.Dᵛ       # Diffusivity of water vapor [m²/s]
    ν  = transport.ν        # Kinematic viscosity [m²/s]

    thermodynamic_factor = mason_thermodynamic_factor(qᵛ⁺ˡ, T, P, constants, transport, parameters, FT)

    # Internal helpers return negative (S - 1 < 0 when subsaturated).
    # Negate to get positive magnitude.
    evap_rate = -rain_evaporation_rate(p3.rain.evaporation, p3.rain.ventilation, qʳ_eff, nʳ_eff, S,
                                       thermodynamic_factor, parameters, ν, Dᵛ, FT)

    # Cannot evaporate more than available
    τ_evap = parameters.rain_evaporation_timescale
    max_evap = qʳ_eff / τ_evap
    evap_rate = min(evap_rate, max_evap)

    return ifelse(is_subsaturated, evap_rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Rain ventilation integral and the slope quantities that go with it:

```math
I_{evap}(λ^r) = \\frac{\\mathbb{C}_{\\mathrm{vent},1}}{(λ^r)^2}
              + \\mathbb{C}_{\\mathrm{vent},2} \\, \\frac{Sc^{1/3}}{\\sqrt{ν}} \\, I_{VD}(λ^r)
```

`I_VD` comes from the tabulated `table`, which stores
``∫ D \\sqrt{V D} e^{-λ^r D} dD`` with neither `ν` nor the Schmidt number baked
in, so both T,P-dependent factors are applied here. ``\\mathbb{C}_{\\mathrm{vent},1}`` and
``\\mathbb{C}_{\\mathrm{vent},2}`` come from `ventilation`, a [`RainVentilation`](@ref), for the same reason:
neither is baked into the table, so both remain configurable at runtime. Returns
`(; λʳ, Nʳ₀, integral)`, since every caller needs the intercept
``N^r_0 = n^r λ^r`` alongside the integral.

Consumed by [`rain_evaporation_rate`](@ref) and by the coupled
saturation-adjustment relaxation coefficient, both of which pass `p3.rain.ventilation`.
"""
@inline function rain_ventilation_integral(table, ventilation, qʳ, nʳ, ν, Dᵛ, parameters)
    FT = typeof(qʳ)
    coefficient_floor = FT(parameters.floors.transport_coefficient)

    # Diagnose λʳ from (qʳ, Nʳ) for exponential DSD (μʳ = 0):
    #   qʳ = Nʳ * <m> = Nʳ * π ρᴸ / λʳ³  ⟹  λʳ = (π ρᴸ / m̄)^(1/3)
    λʳ = rain_slope_parameter(qʳ, nʳ, parameters)
    # Intercept Nʳ₀ = Nʳ * λʳ  (for exponential DSD N'(D) = Nʳ₀ exp(-λ D))
    Nʳ₀ = rain_number_from_slope(qʳ, λʳ, parameters) * λʳ

    ℂᵛᵉⁿᵗ₁ = FT(ventilation.constant_coefficient)
    ℂᵛᵉⁿᵗ₂ = FT(ventilation.reynolds_coefficient)

    # Constant term: ℂᵛᵉⁿᵗ₁ ∫ D exp(-λD) dD = ℂᵛᵉⁿᵗ₁ / λ² for μʳ = 0.
    constant_integral = ℂᵛᵉⁿᵗ₁ / λʳ^2
    schmidt_factor = cbrt(ν / max(Dᵛ, coefficient_floor))
    inverse_sqrt_viscosity = 1 / sqrt(max(ν, coefficient_floor))
    integral = constant_integral + ℂᵛᵉⁿᵗ₂ * schmidt_factor *
                                      inverse_sqrt_viscosity * table(log10(λʳ))

    return (; λʳ, Nʳ₀, integral)
end

# Tabulated path: use PSD-integrated ventilation integral I_evap(λʳ)
@inline function rain_evaporation_rate(table::TabulatedFunction1D, ventilation,
                                       qʳ, nʳ, S,
                                       thermodynamic_factor, parameters, ν, Dᵛ, FT)
    integrals = rain_ventilation_integral(table, ventilation, qʳ, nʳ, ν, Dᵛ, parameters)

    # Evaporation rate (Mason 1971, PSD-integrated):
    #   dm/dt per drop = 4π × C × f_v × (S-1)/Φ,  C = D/2 (spherical capacitance)
    #   dq^r/dt = N_0 × ∫ 4π × (D/2) × f_v × exp(-λD) dD × (S-1)/Φ
    #           = 2π × N_0 × I_evap × (S-1) / Φ,  I_evap = ∫ D × f_v × exp(-λD) dD
    return 2 * FT(π) * integrals.Nʳ₀ * integrals.integral * (S - 1) / thermodynamic_factor
end

#####
##### Scheme-dependent helpers shared by autoconv/accretion/number tendencies
#####

"""
$(TYPEDSIGNATURES)

Cloud-droplet self-collection rate (number loss in cloud, not rain).

Dispatched on `p3.warm_rain_scheme`. Zero for KK2000, which carries no
cloud-droplet self-collection. Returned as a positive magnitude.
"""
@inline cloud_self_collection_rate(p3, qᶜˡ, Nᶜˡ, ρ) =
    cloud_self_collection_rate(p3.warm_rain_scheme, p3, qᶜˡ, Nᶜˡ, ρ)

@inline cloud_self_collection_rate(::KhairoutdinovKogan2000, p3, qᶜˡ, Nᶜˡ, ρ) = zero(qᶜˡ)

"""
$(TYPEDSIGNATURES)

Cloud-droplet number loss from autoconversion (mass → drop count conversion),
dispatched on `p3.warm_rain_scheme`. Returned as a positive magnitude.

For KK2000 the loss is `autoconversion × Nᶜˡ / qᶜˡ`: cloud number is lost in
proportion to the cloud mass lost.
"""
@inline cloud_number_loss_from_autoconversion(p3, autoconversion, qᶜˡ, Nᶜˡ, ρ) =
    cloud_number_loss_from_autoconversion(p3.warm_rain_scheme, p3,
                                          autoconversion, qᶜˡ, Nᶜˡ, ρ)

@inline function cloud_number_loss_from_autoconversion(::KhairoutdinovKogan2000,
                                                       p3, autoconversion, qᶜˡ, Nᶜˡ, ρ)
    return autoconversion * cloud_number_per_cloud_mass(Nᶜˡ, ρ, qᶜˡ)
end

"""
$(TYPEDSIGNATURES)

Mass per newly-formed rain drop produced by autoconversion, dispatched on
`p3.warm_rain_scheme`. Used to convert autoconversion mass rate into a rain
number source.

For KK2000 this is the mass of a 25 μm radius drop ≈ 6.545e-11 kg, read from
`p3.process_rates.initial_rain_drop_mass` so the radius is user-configurable.
"""
@inline rain_seed_drop_mass(p3) = rain_seed_drop_mass(p3.warm_rain_scheme, p3)

@inline rain_seed_drop_mass(::KhairoutdinovKogan2000, p3) = p3.process_rates.initial_rain_drop_mass
