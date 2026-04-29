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
    number_mixing_ratio :: FT        # Na [kg⁻¹]
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

- `number_mixing_ratio`: Aerosol number [kg⁻¹], default 300×10⁶
- `mean_radius`: Geometric mean radius [m], default 0.05 μm
- `geometric_std`: Geometric standard deviation [-], default 2.0
- `vant_hoff_factor`: van't Hoff factor [-], default 3.0
- `osmotic_potential`: Osmotic potential [-], default 1.0
- `mass_fraction_soluble`: Mass fraction soluble [-], default 0.9
- `aerosol_density`: Aerosol density [kg/m³], default 1777.0
- `molecular_weight_aerosol`: Molecular weight of aerosol [kg/mol], default 0.132

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
                     molecular_weight_aerosol = 0.132)
    mᵛ = FT(0.018)
    ρᴸ = FT(1000)
    solute_activity = FT(vant_hoff_factor) * FT(osmotic_potential) * FT(mass_fraction_soluble) *
                      mᵛ * FT(aerosol_density) / (FT(molecular_weight_aerosol) * ρᴸ)
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
end

"""
$(TYPEDSIGNATURES)

Construct an `AerosolActivation` from one or more [`AerosolMode`](@ref)s.

The activation timescale ``τ_{act}`` controls how quickly the cloud
droplet number relaxes toward the activated equilibrium. Default 1.0 s.

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
                           activation_timescale = 1.0) where FT
    modes = (mode1, rest...)
    return AerosolActivation(modes, FT(molecular_weight_water),
                             FT(universal_gas_constant), FT(activation_timescale))
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

    # Surface tension of water [N/m] (Fortran: sigvl)
    σ_v = FT(0.0761) - FT(1.55e-4) * (T - FT(273.15))

    # Kelvin parameter: Aact = 2 Mw σv / (ρᴸ R T)
    A_act = 2 * aerosol.molecular_weight_water * σ_v /
            (FT(1000) * aerosol.universal_gas_constant * T)

    # Critical supersaturation: sm = 2 (1/βact)^{1/2} (Aact / (3 rm))^{3/2}
    s_m = 2 / sqrt(mode.solute_activity) * (A_act / (3 * mode.mean_radius))^FT(1.5)

    # Activated fraction via error function
    # Guard against S ≤ 0: argument → large positive → erf → 1 → N_act → 0
    S_safe = max(S, FT(1e-20))
    erf_argument = 2 * log(s_m / S_safe) / (FT(4.242) * log(mode.geometric_std))

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

Compute prognostic CCN activation rates from aerosol activation physics.

Returns a named tuple `(; ncnuc, qcnuc)`:
- `ncnuc`: Cloud number activation rate [kg⁻¹ s⁻¹]
- `qcnuc`: Cloud mass activation rate [kg/kg/s]

The number rate is a relaxation toward the activated equilibrium:
``n_{\\text{nuc}} = \\max(0, N_{\\text{act}} - n^{cl}) / \\tau_{\\text{act}}``.
Mass follows as ``q_{\\text{nuc}} = n_{\\text{nuc}} \\times m_{\\text{seed}}``
where ``m_{\\text{seed}} = (4\\pi/3) \\rho_w (10^{-6})^3`` is a 1 μm radius droplet.
"""
@inline function prognostic_ccn_activation_rate(aerosol::AerosolActivation, nᶜˡ, qᵛ, qᵛ⁺ˡ, T)
    FT = typeof(T)

    # Environmental supersaturation
    S = (qᵛ - qᵛ⁺ˡ) / max(qᵛ⁺ˡ, FT(1e-20))

    # Total activated number across all modes
    N_activated = total_activated_number(aerosol, T, S)

    # Relaxation toward equilibrium
    ncnuc = max(0, N_activated - nᶜˡ) / aerosol.activation_timescale

    # Seed droplet mass (1 μm radius)
    seed_mass = FT(4π / 3) * FT(1000) * FT(1e-18)
    qcnuc = ncnuc * seed_mass

    # Only activate when supersaturated (Fortran threshold: sup_cld > 1e-6)
    is_supersaturated = S > FT(1e-6)
    ncnuc = ifelse(is_supersaturated, ncnuc, zero(FT))
    qcnuc = ifelse(is_supersaturated, qcnuc, zero(FT))

    return (; ncnuc, qcnuc)
end
