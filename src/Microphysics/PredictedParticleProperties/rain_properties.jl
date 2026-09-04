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
    a_1 \\, \\hat{m}^{b_1} & D \\le D^t_1 \\\\
    a_2 \\, \\hat{m}^{b_2} & D^t_1 < D < D^t_2 \\\\
    a_3 \\, \\hat{m}^{b_3} & D^t_2 \\le D < D^t_3 \\\\
    V_\\infty              & D \\ge D^t_3
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
    branch_velocity_scales :: NTuple{3, FT} # aᵢ of the three power-law branches [m/s]
    branch_mass_exponents :: NTuple{3, FT}  # bᵢ of the three power-law branches [-]
    transition_diameters :: NTuple{3, FT}   # Dᵗᵢ, strictly increasing branch edges [m]
    plateau_velocity :: FT                  # V∞ above the largest edge [m/s]
end

"""
$(TYPEDSIGNATURES)

Construct `RainFallSpeed`. The defaults reproduce the piecewise Gunn-Kinzer /
Beard law used by P3, with the published centimetre-per-second velocity scales converted
to SI.

# Keyword Arguments

- `branch_velocity_scales`: ``(a_1, a_2, a_3)`` [m/s], default `(4579.5, 49.62, 17.32)`
- `branch_mass_exponents`: ``(b_1, b_2, b_3)`` [-], default `(2/3, 1/3, 1/6)`
- `transition_diameters`: ``(D^t_1, D^t_2, D^t_3)`` [m], strictly increasing,
  default `(134.43e-6, 1511.64e-6, 3477.84e-6)`
- `plateau_velocity`: ``V_\\infty`` [m/s], default `9.17`

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: RainFallSpeed
RainFallSpeed(Float64)

# output
RainFallSpeed(aᵥ=(4579.5, 49.62, 17.32) m/s, bᵥ=(0.667, 0.333, 0.167), Dᵗ=(134.43, 1511.64, 3477.84) μm, V∞=9.17 m/s)
```
"""
function RainFallSpeed(FT::DataType = Oceananigans.defaults.FloatType;
                       branch_velocity_scales = (4579.5, 49.62, 17.32),
                       branch_mass_exponents = (2/3, 1/3, 1/6),
                       transition_diameters = (134.43e-6, 1511.64e-6, 3477.84e-6),
                       plateau_velocity = 9.17)

    scales = NTuple{3, FT}(branch_velocity_scales)
    exponents = NTuple{3, FT}(branch_mass_exponents)
    diameters = NTuple{3, FT}(transition_diameters)

    all(≥(0), scales) || throw(ArgumentError("branch_velocity_scales must be nonnegative, got $scales"))
    all(≥(0), exponents) || throw(ArgumentError("branch_mass_exponents must be nonnegative, got $exponents"))
    all(>(0), diameters) || throw(ArgumentError("transition_diameters must be positive, got $diameters"))
    diameters[1] < diameters[2] < diameters[3] ||
        throw(ArgumentError("transition_diameters must be strictly increasing, got $diameters"))
    plateau_velocity ≥ 0 || throw(ArgumentError("plateau_velocity must be nonnegative, got $plateau_velocity"))

    return RainFallSpeed(scales, exponents, diameters, FT(plateau_velocity))
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
    print(io, "aᵥ=", p.branch_velocity_scales, " m/s, ")
    print(io, "bᵥ=", map(b -> round(b, digits=3), p.branch_mass_exponents), ", ")
    print(io, "Dᵗ=", micrometres, " μm, ")
    print(io, "V∞=", p.plateau_velocity, " m/s)")
end

"""
    RainVentilation{FT}

Coefficients of the rain ventilation factor ``f^{ve} = f_{1r} + f_{2r}\\,
\\mathrm{Sc}^{1/3}\\,\\mathrm{Re}^{1/2}``, the classical form of
[Pruppacher and Klett (2010)](@cite pruppacher2010microphysics). The ice side carries the
same pair in the `*_ventilation_constant` / `*_ventilation_reynolds` fields of
[`IceDeposition`](@ref); these are P3's `f1r`/`f2r`.

Consumed at runtime by [`rain_ventilation_integral`](@ref), which assembles the
analytical ``f_{1r}/(λ^r)^2`` term and the Reynolds-weighted term around the tabulated
velocity-diameter integral. They deliberately do not enter that table, which stores only
``I_{VD}``.

See the constructor for the meaning and defaults of each coefficient.
"""
struct RainVentilation{FT}
    constant_coefficient :: FT # f₁ᵣ, the still-air term [-]
    reynolds_coefficient :: FT # f₂ᵣ, multiplying Sc^(1/3) Re^(1/2) [-]
end

"""
$(TYPEDSIGNATURES)

Construct `RainVentilation`.

# Keyword Arguments

- `constant_coefficient`: ``f_{1r}`` [-], default `0.78`
- `reynolds_coefficient`: ``f_{2r}`` [-], default `0.32`

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: RainVentilation
RainVentilation(Float64)

# output
RainVentilation(f₁ᵣ=0.78, f₂ᵣ=0.32)
```
"""
function RainVentilation(FT::DataType = Oceananigans.defaults.FloatType;
                         constant_coefficient = 0.78,
                         reynolds_coefficient = 0.32)

    f₁ᵣ = constant_coefficient
    f₂ᵣ = reynolds_coefficient

    f₁ᵣ ≥ 0 || throw(ArgumentError("constant_coefficient must be nonnegative, got $f₁ᵣ"))
    f₂ᵣ ≥ 0 || throw(ArgumentError("reynolds_coefficient must be nonnegative, got $f₂ᵣ"))

    return RainVentilation(FT(f₁ᵣ), FT(f₂ᵣ))
end

# See the note on `RainFallSpeed` conversion above.
Base.convert(::Type{RainVentilation{FT}}, p::RainVentilation) where FT =
    RainVentilation(FT(p.constant_coefficient), FT(p.reynolds_coefficient))

Base.convert(::Type{RainVentilation{FT}}, p::RainVentilation{FT}) where FT = p

Base.summary(::RainVentilation) = "RainVentilation"

function Base.show(io::IO, p::RainVentilation)
    print(io, summary(p), "(")
    print(io, "f₁ᵣ=", p.constant_coefficient, ", ")
    print(io, "f₂ᵣ=", p.reynolds_coefficient, ")")
end

#####
##### Rain
#####

# Rain particle size distribution, fall-speed and ventilation parameters, and the
# quadrature integrals tabulated from them; see the `RainDrops` constructor.
struct RainDrops{FT, VN, VM, EV}
    maximum_mean_diameter :: FT
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
the four regimes capture Stokes drag below ``D ≈ 100`` μm and the terminal-velocity
plateau above ``D ≈ 5`` mm.

**Ventilation:** ``f^{ve} = f_{1r} + f_{2r}\\,\\mathrm{Sc}^{1/3}\\,\\mathrm{Re}^{1/2}``,
configured by `ventilation` and consumed by rain evaporation and by the coupled
saturation-adjustment relaxation coefficient.

**Integrals:** this is a skeleton — `velocity_number`, `velocity_mass` and `evaporation`
are `nothing` until [`tabulate_rain_from_quadrature`](@ref) materializes them from
`fall_speed`. Both parameter containers are preserved verbatim across materialization.

# Keyword Arguments

- `maximum_mean_diameter`: Upper Dm limit [m], default 2×10⁻³ (2 mm). **Inactive**: no
  rate in the current source reads it, and it does not bound the rain spectrum. It is
  retained only so the field list stays stable; see the note below.
- `fall_speed`: [`RainFallSpeed`](@ref), default `RainFallSpeed(FT)`
- `ventilation`: [`RainVentilation`](@ref), default `RainVentilation(FT)`

!!! note "`maximum_mean_diameter` is inactive"
    The rain spectrum is bounded by `ProcessRate.minimum_rain_slope` and
    `maximum_rain_slope` through `rain_slope_parameter`, not by this field. It is kept
    pending a decision on removing it.

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
RainVentilation(f₁ᵣ=0.8, f₂ᵣ=0.32)
```
"""
function RainDrops(FT::DataType = Oceananigans.defaults.FloatType;
              maximum_mean_diameter = 2e-3,
              fall_speed = RainFallSpeed(FT),
              ventilation = RainVentilation(FT))
    return RainDrops(FT(maximum_mean_diameter), convert(RainFallSpeed{FT}, fall_speed),
                convert(RainVentilation{FT}, ventilation), nothing, nothing, nothing)
end

Base.summary(::RainDrops) = "RainDrops"

function Base.show(io::IO, r::RainDrops)
    print(io, summary(r), "(")
    print(io, "fall_speed=", r.fall_speed, ", ")
    print(io, "ventilation=", r.ventilation, ")")
end
