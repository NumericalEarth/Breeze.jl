#####
##### Aerosol activation for prognostic cloud-droplet number
#####
##### Morrison and Grabowski (2007) equilibrium Kohler theory activation
##### with multi-mode lognormal aerosols and an optional prognostic reservoir.
#####

# One component of a multimodal aerosol size distribution used for CCN activation. Each
# mode represents a physically distinct particle population with a lognormal radius
# distribution and shared chemical properties; see the `AerosolMode` constructor.
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

Construct an `AerosolMode` representing one component of a multimodal aerosol
size distribution. Particles in a mode share one chemical composition and their
radii follow a lognormal distribution described by `mean_radius` and
`geometric_std`. Multiple modes can therefore represent distinct aerosol
populations, such as Aitken and accumulation particles.

The solute activity parameter ``β_{act} = ν_i ϕ_s ε_m M_w ρ_a / (M_a ρ_w)``
is precomputed at construction time from the chemistry parameters.

Default chemistry is ammonium sulfate (NH₄)₂SO₄.

# Keyword Arguments

- `number_mixing_ratio`: Aerosol number *per unit mass of air* [kg⁻¹], default 300×10⁶.
  With `prognostic=true` in [`AerosolActivation`](@ref), the reservoir `ρnᵃ` is
  initialized to air density times the total over all modes once density is available.
  `set!` repeats this initialization unless `nᵃ` [kg⁻¹] or `ρnᵃ` [m⁻³] is supplied.
- `mean_radius`: Geometric mean radius [m], default 0.05 μm
- `geometric_std`: Geometric standard deviation [-], default 2
- `vant_hoff_factor`: van't Hoff factor [-], default 3
- `osmotic_potential`: Osmotic potential [-], default 1
- `mass_fraction_soluble`: Mass fraction soluble [-], default 0.9
- `aerosol_density`: Aerosol density [kg/m³], default 1777
- `molecular_weight_aerosol`: Molecular weight of aerosol [kg/mol], default 0.132
- `thermodynamic_constants`: Constants supplying the water molecular weight and
  liquid-water density
- `molecular_weight_water`: Molecular weight of the condensate [kg/mol], default
  `thermodynamic_constants.vapor.molar_mass`

The molecular weight and `thermodynamic_constants.liquid.density` enter only
through ``β_{act}``. A condensable species other than water is configured through
`thermodynamic_constants` and the surface-tension fit in
[`AerosolActivation`](@ref).

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
function AerosolMode(FT::DataType = Oceananigans.defaults.FloatType;
                     number_mixing_ratio = 3e8,
                     mean_radius = 5e-8,
                     geometric_std = 2,
                     vant_hoff_factor = 3,
                     osmotic_potential = 1,
                     mass_fraction_soluble = 0.9,
                     aerosol_density = 1777,
                     molecular_weight_aerosol = 0.132,
                     thermodynamic_constants = ThermodynamicConstants(FT),
                     molecular_weight_water = thermodynamic_constants.vapor.molar_mass)
    liquid_water_density = thermodynamic_constants.liquid.density
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

# P selects a prognostic aerosol reservoir at compile time.
struct AerosolActivation{FT, P, M}
    modes :: M                       # Tuple of AerosolMode{FT}
    molecular_weight_water :: FT     # Mw [kg/mol]
    universal_gas_constant :: FT     # R [J/(mol·K)]
    activation_timescale :: FT       # ℂᶠᵒʳᵐ₄ [s]
    liquid_water_density :: FT       # ρᴸ [kg/m³]

    # Surface tension of the condensate, linear in temperature:
    # σ(T) = σ₀ + (dσ/dT) (T - T_ref). The defaults are the fit for water.
    surface_tension_reference :: FT              # σ₀ [N/m]
    surface_tension_temperature_derivative :: FT # dσ/dT [N/(m·K)]
    surface_tension_reference_temperature :: FT  # T_ref [K]

    lognormal_activation_factor :: FT            # denominator factor in the erf argument [-]
    activated_droplet_radius :: FT               # ℂᶠᵒʳᵐ₂ [m]
    activation_supersaturation_threshold :: FT   # ℂᶠᵒʳᵐ₃ [-]
    minimum_supersaturation :: FT                # floor on S beneath the logarithm [-]
    minimum_saturation_mass_fraction :: FT       # floor on qᵛ⁺ˡ in the supersaturation denominator [kg/kg]
end

"""
$(TYPEDSIGNATURES)

Construct an `AerosolActivation` from one or more [`AerosolMode`](@ref)s.

The activation timescale ``τ_{act}`` controls how quickly the cloud
droplet number relaxes toward the activated equilibrium. Default 1.0 s.

Passing an `AerosolActivation` to P3 makes cloud droplet number prognostic. The
`prognostic` keyword is separate: `true` carries the unactivated aerosol reservoir
``ρn^a``, which activation depletes, while the default `false` holds the population
fixed at the distribution total.

Everything else the activation physics needs is a keyword here rather than a
literal in [`activated_number`](@ref): the condensate density and molecular
weight, the linear surface-tension fit ``σ(T) = σ_0 + (dσ/dT)(T - T_{ref})``
whose defaults are water's, the ``3\\sqrt{2}`` factor of the lognormal
activation integral (`lognormal_activation_factor`, held at the conventional
rounded 4.242), the radius of a newly activated droplet, the
supersaturation above which activation proceeds, and two floors that keep the
supersaturation finite. A condensable species other than water is configured by
overriding the first group here and in [`AerosolMode`](@ref).

By default, the water molecular weight, liquid-water density, universal gas constant,
and surface-tension reference temperature come from `thermodynamic_constants`.

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

The aerosol population is fixed by default:

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: AerosolActivation, AerosolMode
summary(AerosolActivation(AerosolMode()))

# output
"AerosolActivation(1 mode, fixed reservoir)"
```
"""
function AerosolActivation(mode1::AerosolMode{FT}, rest::AerosolMode{FT}...;
                           prognostic = false,
                           thermodynamic_constants = ThermodynamicConstants(FT),
                           molecular_weight_water = thermodynamic_constants.vapor.molar_mass,
                           universal_gas_constant = thermodynamic_constants.molar_gas_constant,
                           activation_timescale = 1,
                           # Water surface tension [N/m], linear in T about 0°C
                           # This fit remains local because aerosol activation is its only consumer.
                           surface_tension_reference = 0.0761,
                           surface_tension_temperature_derivative = -1.55e-4,
                           surface_tension_reference_temperature = thermodynamic_constants.energy_reference_temperature,
                           # 3√2 = 4.24264… in the lognormal activation integral,
                           # conventionally rounded to 4.242.
                           lognormal_activation_factor = 4.242,
                           activated_droplet_radius = 1e-6,
                           activation_supersaturation_threshold = 1e-6,
                           minimum_supersaturation = 1e-20,
                           minimum_saturation_mass_fraction = 1e-20) where FT
    modes = (mode1, rest...)
    liquid_water_density = thermodynamic_constants.liquid.density
    prognostic = Bool(prognostic)
    return AerosolActivation{FT, prognostic, typeof(modes)}(
                             modes, FT(molecular_weight_water),
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

"""
$(TYPEDSIGNATURES)

Return whether the aerosol reservoir `ρnᵃ` is prognostic.
"""
@inline has_prognostic_aerosol(::AerosolActivation{<:Any, P}) where P = P

function Base.summary(a::AerosolActivation)
    n = length(a.modes)
    reservoir = has_prognostic_aerosol(a) ? "prognostic reservoir" : "fixed reservoir"
    return "AerosolActivation($n mode$(n == 1 ? "" : "s"), $reservoir)"
end

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

and the activated fraction is ``N^a / 2 \\, [1 - \\text{erf}(u)]`` where
``u = 2 \\ln(s_m / S) / (4.242 \\ln \\sigma_g)``.
"""
@inline function activated_number(mode::AerosolMode, aerosol::AerosolActivation, T, S)
    FT = typeof(T)

    # Surface tension of the condensate [N/m]
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

Compute cloud droplet activation rates from the aerosol distribution.
The supplied `nᵃ` limits activation to the remaining aerosol population;
omitting it uses the full distribution.

Returns a named tuple `(; ncnuc, qcnuc)`:
- `ncnuc`: Cloud number activation rate [kg⁻¹ s⁻¹]
- `qcnuc`: Cloud mass activation rate [kg/kg/s]

The equilibrium count ``N_{\\text{act}}(S)`` follows
[Morrison and Grabowski (2007)](@cite MorrisonGrabowski2007). Capping the target
by the available aerosol gives

```math
n_{\\text{nuc}} = \\frac{\\max(0,\\; \\min(N_{\\text{act}}(S), n^{cl} + n^a) - n^{cl})}
                       {\\mathbb{C}_{\\mathrm{form},4}}.
```

Each activated droplet consumes one aerosol from a prognostic reservoir,
giving the density tendency ``-ρ \\, n_{\\text{nuc}}``.

Mass follows as ``q_{\\text{nuc}} = n_{\\text{nuc}} \\times m_{\\text{seed}}``
where ``m_{\\text{seed}} = (4\\pi/3) \\rho_w (\\mathbb{C}_{\\mathrm{form},2})^3`` is a
droplet of the activated radius, 1 μm by default.
"""
@inline function aerosol_activation_rate(aerosol::AerosolActivation, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T)
    FT = typeof(T)
    ℂᶠᵒʳᵐ₂ = aerosol.activated_droplet_radius
    ℂᶠᵒʳᵐ₃ = aerosol.activation_supersaturation_threshold
    ℂᶠᵒʳᵐ₄ = aerosol.activation_timescale

    # Environmental supersaturation
    S = (qᵛ - qᵛ⁺ˡ) / max(qᵛ⁺ˡ, aerosol.minimum_saturation_mass_fraction)

    # Diagnostic equilibrium activation count from M&G2007.
    N_activated = total_activated_number(aerosol, T, S)

    # Cap by available pool: at most n^a more aerosols can ever activate.
    nᵃ_available = max(0, nᵃ)
    N_target = min(N_activated, nᶜˡ + nᵃ_available)

    # Relaxation toward the (capped) equilibrium
    ncnuc = max(0, N_target - nᶜˡ) / ℂᶠᵒʳᵐ₄

    # Seed droplet mass, from the activated-droplet radius (1 μm by default)
    seed_mass = 4 * FT(π) / 3 * aerosol.liquid_water_density * ℂᶠᵒʳᵐ₂^3
    qcnuc = ncnuc * seed_mass

    # Only activate when supersaturated
    is_supersaturated = S > ℂᶠᵒʳᵐ₃
    ncnuc = ifelse(is_supersaturated, ncnuc, zero(FT))
    qcnuc = ifelse(is_supersaturated, qcnuc, zero(FT))

    return (; ncnuc, qcnuc)
end

# Fixed-population form: with the whole distribution as the pool the cap cannot bind
# (`total_activated_number` is already capped there), leaving the plain relaxation toward
# the M&G2007 equilibrium.
@inline function aerosol_activation_rate(aerosol::AerosolActivation, nᶜˡ, qᵛ, qᵛ⁺ˡ, T)
    return aerosol_activation_rate(aerosol, nᶜˡ, sum_aerosol_number(aerosol), qᵛ, qᵛ⁺ˡ, T)
end
