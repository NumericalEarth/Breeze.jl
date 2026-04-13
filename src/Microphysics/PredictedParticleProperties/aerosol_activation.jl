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
- `mass_fraction_soluble`: Mass fraction soluble [-], default 1.0
- `aerosol_density`: Aerosol density [kg/m³], default 1770.0
- `molecular_weight_aerosol`: Molecular weight of aerosol [kg/mol], default 0.132141

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
                     mass_fraction_soluble = 1.0,
                     aerosol_density = 1770.0,
                     molecular_weight_aerosol = 0.132141)
    Mw = FT(0.018016)
    ρw = FT(1000)
    solute_activity = FT(vant_hoff_factor) * FT(osmotic_potential) * FT(mass_fraction_soluble) *
                      Mw * FT(aerosol_density) / (FT(molecular_weight_aerosol) * ρw)
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
function AerosolActivation(modes::AerosolMode{FT}...;
                           molecular_weight_water = 0.018016,
                           universal_gas_constant = 8.3145,
                           activation_timescale = 1.0) where FT
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
