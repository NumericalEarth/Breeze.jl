#####
##### Aerosol Activation (Prognostic CCN)
#####
##### Morrison and Grabowski (2007) equilibrium Kohler theory activation
##### with multi-mode lognormal aerosol support.
#####

"""
    AerosolMode

One lognormal aerosol mode for CCN activation.
See [`AerosolMode`](@ref) constructor for details.
"""
struct AerosolMode{FT}
    number_mixing_ratio :: FT        # Na [kg⁻¹], per unit mass of air (not per volume)
    mean_radius :: FT                # rm [m]
    geometric_std :: FT              # σg [-]
    vant_hoff_factor :: FT           # νi [-]
    osmotic_potential :: FT          # φs [-]
    mass_fraction_soluble :: FT      # εm [-]
    aerosol_density :: FT            # ρa [kg/m³]
    molecular_weight_aerosol :: FT   # Ma [kg/mol]
    solute_activity :: FT            # βact [-] (precomputed)
end

"""
$(TYPEDSIGNATURES)

Construct an `AerosolMode` representing one lognormal aerosol population.

The solute activity parameter ``β_{act} = ν_i ϕ_s ε_m M_w ρ_a / (M_a ρ_w)``
is precomputed at construction time from the chemistry parameters.

Default chemistry is ammonium sulfate (NH₄)₂SO₄.

# Keyword Arguments

- `number_mixing_ratio`: Aerosol number *per unit mass of air* [kg⁻¹], default 300×10⁶.
  This is the basis of the whole activation path: [`activated_number`](@ref),
  [`total_activated_number`](@ref) and `sum_aerosol_number` are all [kg⁻¹], and the
  activation cap compares them against the per-mass `nᶜˡ = ρnᶜˡ/ρ` and `nᵃ = ρnᵃ/ρ`.

  The prognostic reservoir `ρnᵃ` holds the ρ-weighted counterpart. Nothing needs to be
  initialized by hand: `AtmosphereModel` construction and every `set!` write it as the air
  density times this field summed over all of an [`AerosolActivation`](@ref)'s modes, so a
  multi-mode population is seeded from its own parameters and stays consistent with them.
  Pass `nᵃ` [kg⁻¹] or `ρnᵃ` [m⁻³] to `set!` to override, which is also how a partly depleted
  reservoir survives a `set!`.
- `mean_radius`: Geometric mean radius [m], default 0.05 μm
- `geometric_std`: Geometric standard deviation [-], default 2.0
- `vant_hoff_factor`: van't Hoff factor [-], default 3.0
- `osmotic_potential`: Osmotic potential [-], default 1.0
- `mass_fraction_soluble`: Mass fraction soluble [-], default 0.9
- `aerosol_density`: Aerosol density [kg/m³], default 1777.0
- `molecular_weight_aerosol`: Molecular weight of aerosol [kg/mol], default 0.132
- `molecular_weight_water`: Molecular weight of the condensate [kg/mol], default 0.018
- `liquid_water_density`: Density of the condensate [kg/m³], default 1000.0

The last two enter only through ``β_{act}``. They are the condensate-specific
inputs to this mode, so a species other than water is configured by overriding
them here and the surface-tension fit in [`AerosolActivation`](@ref).

# References

[Morrison and Grabowski (2007)](@cite MorrisonGrabowski2007)

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: AerosolMode
mode = AerosolMode()
mode.mean_radius

# output
5.0e-8
```
"""
function AerosolMode(FT::Type{<:AbstractFloat} = Float64;
                     number_mixing_ratio = 300e6,
                     mean_radius = 0.05e-6,
                     geometric_std = 2.0,
                     vant_hoff_factor = 3.0,
                     osmotic_potential = 1.0,
                     mass_fraction_soluble = 0.9,
                     aerosol_density = 1777.0,
                     molecular_weight_aerosol = 0.132,
                     molecular_weight_water = 0.018,
                     liquid_water_density = 1000.0)
    solute_activity = FT(vant_hoff_factor) * FT(osmotic_potential) * FT(mass_fraction_soluble) *
                      FT(molecular_weight_water) * FT(aerosol_density) /
                      (FT(molecular_weight_aerosol) * FT(liquid_water_density))
    return AerosolMode(FT(number_mixing_ratio), FT(mean_radius), FT(geometric_std),
                       FT(vant_hoff_factor), FT(osmotic_potential), FT(mass_fraction_soluble),
                       FT(aerosol_density), FT(molecular_weight_aerosol), solute_activity)
end

Base.summary(::AerosolMode) = "AerosolMode"

function Base.show(io::IO, m::AerosolMode)
    print(io, summary(m), "(")
    print(io, "Na=", m.number_mixing_ratio, " kg⁻¹, ")
    print(io, "rm=", m.mean_radius, " m, ")
    print(io, "σg=", m.geometric_std, ")")
end

"""
    AerosolActivation

Container for multi-mode aerosol activation parameters.
See [`AerosolActivation`](@ref) constructor for details.
"""
struct AerosolActivation{FT, M}
    modes :: M                       # Tuple of AerosolMode{FT}
    molecular_weight_water :: FT     # Mw [kg/mol]
    universal_gas_constant :: FT     # R [J/(mol·K)]
    activation_timescale :: FT       # τ_act [s]
    liquid_water_density :: FT       # ρᴸ [kg/m³]

    # Surface tension of the condensate, linear in temperature:
    # σ(T) = σ₀ + (dσ/dT) (T - T_ref). Defaults are the water fit used by Fortran P3.
    surface_tension_reference :: FT              # σ₀ [N/m]
    surface_tension_temperature_derivative :: FT # dσ/dT [N/(m·K)]
    surface_tension_reference_temperature :: FT  # T_ref [K]

    lognormal_activation_factor :: FT            # denominator factor in the erf argument [-]
    activated_droplet_radius :: FT               # radius of a newly activated droplet [m]
    activation_supersaturation_threshold :: FT   # S above which activation proceeds [-]
    minimum_supersaturation :: FT                # floor on S beneath the logarithm [-]
    minimum_saturation_mass_fraction :: FT       # floor on qᵛ⁺ˡ in the supersaturation denominator [kg/kg]
end

"""
$(TYPEDSIGNATURES)

Construct an `AerosolActivation` from one or more [`AerosolMode`](@ref)s.

The activation timescale ``τ_{act}`` controls how quickly the cloud
droplet number relaxes toward the activated equilibrium. Default 1.0 s.

Everything else the activation physics needs is a keyword here rather than a
literal in [`activated_number`](@ref): the condensate density and molecular
weight, the linear surface-tension fit ``σ(T) = σ_0 + (dσ/dT)(T - T_{ref})``
whose defaults are water's, the ``3\\sqrt{2}`` factor of the lognormal
activation integral (`lognormal_activation_factor`, held at the Fortran
reference's rounded 4.242), the radius of a newly activated droplet, the
supersaturation above which activation proceeds, and two floors that keep the
supersaturation finite. A condensable species other than water is configured by
overriding the first group here and in [`AerosolMode`](@ref).

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: AerosolActivation, AerosolMode
aerosol = AerosolActivation(AerosolMode())
length(aerosol.modes)

# output
1
```

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: AerosolActivation, AerosolMode
aerosol = AerosolActivation(
    AerosolMode(number_mixing_ratio=100e6, mean_radius=0.08e-6),
    AerosolMode(number_mixing_ratio=50e6,  mean_radius=1.0e-6, geometric_std=2.5);
    activation_timescale = 2.0
)
length(aerosol.modes)

# output
2
```
"""
function AerosolActivation(mode1::AerosolMode{FT}, rest::AerosolMode{FT}...;
                           molecular_weight_water = 0.018,
                           universal_gas_constant = 8.3145,
                           activation_timescale = 1.0,
                           liquid_water_density = 1000.0,
                           # Water surface tension [N/m], linear in T about 0°C (Fortran `sigvl`)
                           surface_tension_reference = 0.0761,
                           surface_tension_temperature_derivative = -1.55e-4,
                           surface_tension_reference_temperature = 273.15,
                           # 3√2 = 4.24264… in the lognormal activation integral; the
                           # Fortran reference rounds it to 4.242, kept here for parity.
                           lognormal_activation_factor = 4.242,
                           activated_droplet_radius = 1e-6,
                           activation_supersaturation_threshold = 1e-6,
                           minimum_supersaturation = 1e-20,
                           minimum_saturation_mass_fraction = 1e-20) where FT
    modes = (mode1, rest...)
    return AerosolActivation(modes, FT(molecular_weight_water),
                             FT(universal_gas_constant), FT(activation_timescale),
                             FT(liquid_water_density),
                             FT(surface_tension_reference),
                             FT(surface_tension_temperature_derivative),
                             FT(surface_tension_reference_temperature),
                             FT(lognormal_activation_factor),
                             FT(activated_droplet_radius),
                             FT(activation_supersaturation_threshold),
                             FT(minimum_supersaturation),
                             FT(minimum_saturation_mass_fraction))
end

Base.summary(a::AerosolActivation) = "AerosolActivation($(length(a.modes)) mode$(length(a.modes) == 1 ? "" : "s"))"

function Base.show(io::IO, a::AerosolActivation)
    print(io, summary(a))
    for (i, mode) in enumerate(a.modes)
        prefix = i < length(a.modes) ? "\n├── " : "\n└── "
        print(io, prefix, "mode $i: ", mode)
    end
end

#####
##### Activation physics (Morrison & Grabowski 2007)
#####

"""
$(TYPEDSIGNATURES)

Compute the activated number [kg⁻¹] from a single lognormal aerosol mode
at temperature `T` [K] and environmental supersaturation `S` [-].

Following [Morrison and Grabowski (2007)](@cite MorrisonGrabowski2007),
the critical supersaturation for mode activation is

```math
s_m = 2 \\left(\\frac{1}{\\beta_{\\text{act}}}\\right)^{1/2}
      \\left(\\frac{A_{\\text{act}}}{3 \\, r_m}\\right)^{3/2}
```

and the activated fraction is ``N_a / 2 \\, [1 - \\text{erf}(u)]`` where
``u = 2 \\ln(s_m / S) / (4.242 \\ln \\sigma_g)``.
"""
@inline function activated_number(mode::AerosolMode, aerosol::AerosolActivation, T, S)
    FT = typeof(T)

    # Surface tension of the condensate [N/m] (Fortran: sigvl)
    σ_v = aerosol.surface_tension_reference +
          aerosol.surface_tension_temperature_derivative *
          (T - aerosol.surface_tension_reference_temperature)

    # Kelvin parameter: Aact = 2 Mw σv / (ρᴸ R T)
    A_act = 2 * aerosol.molecular_weight_water * σ_v /
            (aerosol.liquid_water_density * aerosol.universal_gas_constant * T)

    # Critical supersaturation: sm = 2 (1/βact)^{1/2} (Aact / (3 rm))^{3/2}.
    # `sqrt(x)^3` avoids the `pow` call that `x^1.5` compiles to.
    kelvin_ratio = A_act / (3 * mode.mean_radius)
    s_m = 2 / sqrt(mode.solute_activity) * sqrt(kelvin_ratio)^3

    # Activated fraction via error function
    # Guard against S ≤ 0: argument → large positive → erf → 1 → N_act → 0
    S_safe = max(S, aerosol.minimum_supersaturation)
    erf_argument = 2 * log(s_m / S_safe) /
                   (aerosol.lognormal_activation_factor * log(mode.geometric_std))

    return mode.number_mixing_ratio * FT(0.5) * (1 - erf(erf_argument))
end

"""
$(TYPEDSIGNATURES)

Total aerosol number mixing ratio [kg⁻¹] across all modes.
"""
@inline function sum_aerosol_number(aerosol::AerosolActivation)
    N_total = zero(aerosol.activation_timescale)
    for mode in aerosol.modes
        N_total += mode.number_mixing_ratio
    end
    return N_total
end

"""
$(TYPEDSIGNATURES)

Total activated number [kg⁻¹] summed across all aerosol modes,
capped at the total aerosol number.
"""
@inline function total_activated_number(aerosol::AerosolActivation, T, S)
    N_act = zero(T)
    for mode in aerosol.modes
        N_act += activated_number(mode, aerosol, T, S)
    end
    return min(N_act, sum_aerosol_number(aerosol))
end

"""
$(TYPEDSIGNATURES)

Compute prognostic CCN activation rates from aerosol activation physics with
aerosol-pool depletion.

Returns a named tuple `(; ncnuc, qcnuc)`:
- `ncnuc`: Cloud number activation rate [kg⁻¹ s⁻¹] (also the depletion rate
  of the unactivated aerosol pool — `ρnᵃ` decreases at exactly the same rate).
- `qcnuc`: Cloud mass activation rate [kg/kg/s]

Following Morrison & Grabowski (2007) augmented with explicit aerosol-pool
tracking (matching the two-moment Seifert–Beheng convention used elsewhere in
this codebase), the equilibrium number of activated droplets at supersaturation
``S`` is ``N_{\\text{act}}(S)``, but the number that can *actually* be activated
in one step is capped by the unactivated pool ``n^a``:

```math
n_{\\text{nuc}} = \\frac{\\max(0,\\; \\min(N_{\\text{act}}(S), n^{cl} + n^a) - n^{cl})}
                       {\\tau_{\\text{act}}}.
```

This prevents the spurious re-activation that occurs when ``S`` rebounds after
autoconversion or partial cloud evaporation drains ``n^{cl}`` — without an
aerosol-pool sink, the diagnostic ``N_{\\text{act}}(S)`` keeps generating new
droplets as if the reservoir were inexhaustible. With the cap, each activated
droplet permanently removes one unit from ``n^a``.

Mass follows as ``q_{\\text{nuc}} = n_{\\text{nuc}} \\times m_{\\text{seed}}``
where ``m_{\\text{seed}} = (4\\pi/3) \\rho_w (10^{-6})^3`` is a 1 μm radius droplet.
"""
@inline function prognostic_ccn_activation_rate(aerosol::AerosolActivation, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T)
    FT = typeof(T)

    # Environmental supersaturation
    S = (qᵛ - qᵛ⁺ˡ) / max(qᵛ⁺ˡ, aerosol.minimum_saturation_mass_fraction)

    # Diagnostic equilibrium activation count from M&G2007.
    N_activated = total_activated_number(aerosol, T, S)

    # Cap by available pool: at most n^a more aerosols can ever activate.
    nᵃ_available = max(0, nᵃ)
    N_target = min(N_activated, nᶜˡ + nᵃ_available)

    # Relaxation toward the (capped) equilibrium
    ncnuc = max(0, N_target - nᶜˡ) / aerosol.activation_timescale

    # Seed droplet mass, from the activated-droplet radius (1 μm by default)
    r_seed = aerosol.activated_droplet_radius
    seed_mass = 4 * FT(π) / 3 * aerosol.liquid_water_density * r_seed^3
    qcnuc = ncnuc * seed_mass

    # Only activate when supersaturated (Fortran threshold: sup_cld > 1e-6)
    is_supersaturated = S > aerosol.activation_supersaturation_threshold
    ncnuc = ifelse(is_supersaturated, ncnuc, zero(FT))
    qcnuc = ifelse(is_supersaturated, qcnuc, zero(FT))

    return (; ncnuc, qcnuc)
end

@inline function prognostic_ccn_activation_rate(aerosol::AerosolActivation, nᶜˡ, qᵛ, qᵛ⁺ˡ, T)
    return prognostic_ccn_activation_rate(aerosol, nᶜˡ, sum_aerosol_number(aerosol), qᵛ, qᵛ⁺ˡ, T)
end
