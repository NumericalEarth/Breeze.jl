#####
##### Rain Properties
#####
##### Rain particle properties and integrals for the P3 scheme.
#####

#####
##### Empirical parameter containers
#####
##### The rain fall-speed and ventilation laws are empirical fits. Their coefficients live
##### in the two small immutable containers below rather than as module constants, so a
##### calibration or sensitivity study can vary them through the public constructors and
##### have the new values reach every quadrature table and every runtime rate.
#####

"""
    RainFallSpeed{FT}

Empirical coefficients of the piecewise Gunn-Kinzer / Beard rain terminal-velocity law
evaluated by [`rain_fall_speed`](@ref),

```math
V(D) = \\begin{cases}
    \\mathbb{C}_{V,1,1} \\, \\hat{m}^{\\mathbb{C}_{V,2,1}} & D \\le \\mathbb{C}_{V,3,1} \\\\
    \\mathbb{C}_{V,1,2} \\, \\hat{m}^{\\mathbb{C}_{V,2,2}} & \\mathbb{C}_{V,3,1} < D < \\mathbb{C}_{V,3,2} \\\\
    \\mathbb{C}_{V,1,3} \\, \\hat{m}^{\\mathbb{C}_{V,2,3}} & \\mathbb{C}_{V,3,2} \\le D < \\mathbb{C}_{V,3,3} \\\\
    \\mathbb{C}_{V,4}              & D \\ge \\mathbb{C}_{V,3,3}
\\end{cases}
```

where ``\\hat{m} = m(D) / (1 \\, \\mathrm{g})`` is the dimensionless ratio of the drop mass
to one gram. The mass itself is the spherical-drop mass at the water density the published
fit was derived with (`GUNN_KINZER_WATER_DENSITY`), which belongs to the fit rather than to
the model and is therefore not exposed here.

See the constructor for the meaning, units and defaults of each coefficient.

# References

The Gunn-Kinzer / Beard fit as used by P3; see
[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).
"""
struct RainFallSpeed{FT}
    branch_velocity_scales :: NTuple{3, FT} # ℂⱽ₁, three branch scales [m/s]
    branch_mass_exponents :: NTuple{3, FT}  # ℂⱽ₂, three mass exponents [-]
    transition_diameters :: NTuple{3, FT}   # ℂⱽ₃, strictly increasing edges [m]
    plateau_velocity :: FT                  # ℂⱽ₄, large-drop plateau [m/s]
end

"""
$(TYPEDSIGNATURES)

Construct `RainFallSpeed`. The defaults reproduce the piecewise Gunn-Kinzer /
Beard law used by P3, with the published centimetre-per-second velocity scales converted
to SI.

# Keyword Arguments

- `branch_velocity_scales`: ``\\mathbb{C}_{V,1}`` [m/s], default `(4579.5, 49.62, 17.32)`
- `branch_mass_exponents`: ``\\mathbb{C}_{V,2}`` [-], default `(2/3, 1/3, 1/6)`
- `transition_diameters`: ``\\mathbb{C}_{V,3}`` [m], strictly increasing,
  default `(134.43e-6, 1511.64e-6, 3477.84e-6)`
- `plateau_velocity`: ``\\mathbb{C}_{V,4}`` [m/s], default `9.17`

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: RainFallSpeed
RainFallSpeed(Float64)

# output
RainFallSpeed(ℂⱽ₁=(4579.5, 49.62, 17.32) m/s, ℂⱽ₂=(0.667, 0.333, 0.167), ℂⱽ₃=(134.43, 1511.64, 3477.84) μm, ℂⱽ₄=9.17 m/s)
```
"""
function RainFallSpeed(FT::DataType = Oceananigans.defaults.FloatType;
                       branch_velocity_scales = (4579.5, 49.62, 17.32),
                       branch_mass_exponents = (2/3, 1/3, 1/6),
                       transition_diameters = (134.43e-6, 1511.64e-6, 3477.84e-6),
                       plateau_velocity = 9.17)

    ℂⱽ₁ = NTuple{3, FT}(branch_velocity_scales)
    ℂⱽ₂ = NTuple{3, FT}(branch_mass_exponents)
    ℂⱽ₃ = NTuple{3, FT}(transition_diameters)
    ℂⱽ₄ = FT(plateau_velocity)

    all(≥(0), ℂⱽ₁) || throw(ArgumentError("branch_velocity_scales must be nonnegative, got $ℂⱽ₁"))
    all(≥(0), ℂⱽ₂) || throw(ArgumentError("branch_mass_exponents must be nonnegative, got $ℂⱽ₂"))
    all(>(0), ℂⱽ₃) || throw(ArgumentError("transition_diameters must be positive, got $ℂⱽ₃"))
    ℂⱽ₃[1] < ℂⱽ₃[2] < ℂⱽ₃[3] ||
        throw(ArgumentError("transition_diameters must be strictly increasing, got $ℂⱽ₃"))
    ℂⱽ₄ ≥ 0 || throw(ArgumentError("plateau_velocity must be nonnegative, got $ℂⱽ₄"))

    return RainFallSpeed(ℂⱽ₁, ℂⱽ₂, ℂⱽ₃, ℂⱽ₄)
end

# Allow a container built at one precision to be reused at another, so that
# `RainDrops(Float32; fall_speed = RainFallSpeed(Float64; ...))` keeps the
# configured values instead of erroring on the field types. The identity method is also the
# tie-breaker that keeps `convert` unambiguous against `Base.convert(::Type{T}, ::T)`.
Base.convert(::Type{RainFallSpeed{FT}}, p::RainFallSpeed) where FT =
    RainFallSpeed(NTuple{3, FT}(p.branch_velocity_scales), NTuple{3, FT}(p.branch_mass_exponents),
                  NTuple{3, FT}(p.transition_diameters), FT(p.plateau_velocity))

Base.convert(::Type{RainFallSpeed{FT}}, p::RainFallSpeed{FT}) where FT = p

Base.summary(::RainFallSpeed) = "RainFallSpeed"

function Base.show(io::IO, p::RainFallSpeed)
    micrometres = map(D -> round(D * 10^6, digits=2), p.transition_diameters)
    print(io, summary(p), "(")
    print(io, "ℂⱽ₁=", p.branch_velocity_scales, " m/s, ")
    print(io, "ℂⱽ₂=", map(b -> round(b, digits=3), p.branch_mass_exponents), ", ")
    print(io, "ℂⱽ₃=", micrometres, " μm, ")
    print(io, "ℂⱽ₄=", p.plateau_velocity, " m/s)")
end

"""
    RainVentilation{FT}

Coefficients of the rain ventilation factor ``f^{ve} = \\mathbb{C}_{\\mathrm{vent},1} + \\mathbb{C}_{\\mathrm{vent},2}\\,
\\mathrm{Sc}^{1/3}\\,\\mathrm{Re}^{1/2}``, the classical form of
[Pruppacher and Klett (2010)](@cite pruppacher2010microphysics). These are P3's
traditional `f1r`/`f2r` coefficients.

The ice side uses the same form with its own pair (0.65, 0.44), folded into the lookup
tables at generation: the `*_ventilation_constant` / `*_ventilation_reynolds` fields of
[`IceDeposition`](@ref) hold the scaled integrals, not the coefficients, so the ice pair
is not configurable.

Consumed at runtime by [`rain_ventilation_integral`](@ref), which assembles the
analytical ``\\mathbb{C}_{\\mathrm{vent},1}/(λ^r)^2`` term and the Reynolds-weighted term around the tabulated
velocity-diameter integral. They deliberately do not enter that table, which stores only
``I_{VD}``.

See the constructor for the meaning and defaults of each coefficient.
"""
struct RainVentilation{FT}
    constant_coefficient :: FT # ℂᵛᵉⁿᵗ₁, the still-air term [-]
    reynolds_coefficient :: FT # ℂᵛᵉⁿᵗ₂, multiplying Sc^(1/3) Re^(1/2) [-]
end

"""
$(TYPEDSIGNATURES)

Construct `RainVentilation`.

# Keyword Arguments

- `constant_coefficient`: ``\\mathbb{C}_{\\mathrm{vent},1}`` [-], default `0.78`
- `reynolds_coefficient`: ``\\mathbb{C}_{\\mathrm{vent},2}`` [-], default `0.32`

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: RainVentilation
RainVentilation(Float64)

# output
RainVentilation(ℂᵛᵉⁿᵗ₁=0.78, ℂᵛᵉⁿᵗ₂=0.32)
```
"""
function RainVentilation(FT::DataType = Oceananigans.defaults.FloatType;
                         constant_coefficient = 0.78,
                         reynolds_coefficient = 0.32)

    ℂᵛᵉⁿᵗ₁ = constant_coefficient
    ℂᵛᵉⁿᵗ₂ = reynolds_coefficient

    ℂᵛᵉⁿᵗ₁ ≥ 0 ||
        throw(ArgumentError("constant_coefficient must be nonnegative, got $ℂᵛᵉⁿᵗ₁"))
    ℂᵛᵉⁿᵗ₂ ≥ 0 ||
        throw(ArgumentError("reynolds_coefficient must be nonnegative, got $ℂᵛᵉⁿᵗ₂"))

    return RainVentilation(FT(ℂᵛᵉⁿᵗ₁), FT(ℂᵛᵉⁿᵗ₂))
end

# See the note on `RainFallSpeed` conversion above.
Base.convert(::Type{RainVentilation{FT}}, p::RainVentilation) where FT =
    RainVentilation(FT(p.constant_coefficient), FT(p.reynolds_coefficient))

Base.convert(::Type{RainVentilation{FT}}, p::RainVentilation{FT}) where FT = p

Base.summary(::RainVentilation) = "RainVentilation"

function Base.show(io::IO, p::RainVentilation)
    print(io, summary(p), "(")
    print(io, "ℂᵛᵉⁿᵗ₁=", p.constant_coefficient, ", ")
    print(io, "ℂᵛᵉⁿᵗ₂=", p.reynolds_coefficient, ")")
end

#####
##### Rain
#####

# Rain particle size distribution, fall-speed and ventilation parameters, and the
# quadrature integrals tabulated from them; see the `RainDrops` constructor.
struct RainDrops{FT, VN, VM, EV}
    fall_speed :: RainFallSpeed{FT}
    ventilation :: RainVentilation{FT}
    velocity_number :: VN
    velocity_mass :: VM
    evaporation :: EV
end

"""
$(TYPEDSIGNATURES)

Construct `RainDrops` with empirical parameters and quadrature-based integrals.

Rain in P3 follows an exponential size distribution, the ``μ^r = 0`` special
case of the gamma distribution used for ice:

```math
N'(D) = Nʳ₀ e^{-λ^r D}
```

There is no rain shape parameter, prognostic or diagnosed: `rain_slope_parameter`
inverts the mass integral directly as ``λ^r = (π ρ^w n^r / q^r)^{1/3}``, and
`rain_quadrature.jl` integrates against the same exponential kernel.

**Terminal velocity:** the piecewise Gunn-Kinzer / Beard law of
[`rain_fall_speed`](@ref), configured by `fall_speed`. It is *not* a single power law;
the four regimes capture Stokes drag below the first transition diameter (``D ≈ 134``
μm by default) and the terminal-velocity plateau above the third (``D ≈ 3.5`` mm).

**Ventilation:** ``f^{ve} = \\mathbb{C}_{\\mathrm{vent},1} + \\mathbb{C}_{\\mathrm{vent},2}\\,\\mathrm{Sc}^{1/3}\\,\\mathrm{Re}^{1/2}``,
configured by `ventilation` and consumed by rain evaporation and by the coupled
saturation-adjustment relaxation coefficient.

**Integrals:** this is a skeleton — `velocity_number`, `velocity_mass` and `evaporation`
are `nothing` until [`tabulate_rain_from_quadrature`](@ref) materializes them from
`fall_speed`. Both parameter containers are preserved verbatim across materialization.

**Spectrum bounds:** set by `ProcessRate.minimum_rain_slope` and `maximum_rain_slope`
through [`rain_slope_parameter`](@ref), not by this container.

# Keyword Arguments

- `fall_speed`: [`RainFallSpeed`](@ref), default `RainFallSpeed(FT)`
- `ventilation`: [`RainVentilation`](@ref), default `RainVentilation(FT)`

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization),
[Milbrandt and Yau (2005)](@cite MilbrandtYau2005),
[Seifert and Beheng (2006)](@cite SeifertBeheng2006).

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: RainDrops, RainVentilation
rain = RainDrops(Float64; ventilation = RainVentilation(Float64;
                                                   constant_coefficient = 0.8))
rain.ventilation

# output
RainVentilation(ℂᵛᵉⁿᵗ₁=0.8, ℂᵛᵉⁿᵗ₂=0.32)
```
"""
function RainDrops(FT::DataType = Oceananigans.defaults.FloatType;
                   fall_speed = RainFallSpeed(FT),
                   ventilation = RainVentilation(FT))
    return RainDrops(convert(RainFallSpeed{FT}, fall_speed),
                     convert(RainVentilation{FT}, ventilation), nothing, nothing, nothing)
end

Base.summary(::RainDrops) = "RainDrops"

function Base.show(io::IO, r::RainDrops)
    print(io, summary(r), "(")
    print(io, "fall_speed=", r.fall_speed, ", ")
    print(io, "ventilation=", r.ventilation, ")")
end
