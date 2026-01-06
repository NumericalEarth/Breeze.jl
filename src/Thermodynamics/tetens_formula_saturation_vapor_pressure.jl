struct TetensFormula{FT}
    reference_saturation_vapor_pressure :: FT
    reference_temperature :: FT
    liquid_coefficient :: FT
    liquid_temperature_offset :: FT
    ice_coefficient :: FT
    ice_temperature_offset :: FT
end

function Base.summary(tf::TetensFormula{FT}) where FT
    return string("TetensFormula{", FT, "}(",
                  "pᵣ=", prettysummary(tf.reference_saturation_vapor_pressure), ", ",
                  "T₀=", prettysummary(tf.reference_temperature), ", ",
                  "aˡ=", prettysummary(tf.liquid_coefficient), ", ",
                  "bˡ=", prettysummary(tf.liquid_temperature_offset), ", ",
                  "aⁱ=", prettysummary(tf.ice_coefficient), ", ",
                  "bⁱ=", prettysummary(tf.ice_temperature_offset), ")")
end

Base.show(io::IO, tf::TetensFormula) = print(io, summary(tf))

function Adapt.adapt_structure(to, tf::TetensFormula)
    pᵛ⁺ᵣ = adapt(to, tf.reference_saturation_vapor_pressure)
    T₀ = adapt(to, tf.reference_temperature)
    aˡ = adapt(to, tf.liquid_coefficient)
    bˡ = adapt(to, tf.liquid_temperature_offset)
    aⁱ = adapt(to, tf.ice_coefficient)
    bⁱ = adapt(to, tf.ice_temperature_offset)
    FT = typeof(pᵛ⁺ᵣ)
    return TetensFormula{FT}(pᵛ⁺ᵣ, T₀, aˡ, bˡ, aⁱ, bⁱ)
end

"""
$(TYPEDSIGNATURES)

Construct a `TetensFormula` saturation vapor pressure formulation.
Tetens' formula is an empirical formula for the saturation vapor pressure,

```math
pᵛ⁺(T) = pᵛ⁺ᵣ \\exp \\left( a \\frac{T - T₀}{T - b} \\right) ,
```

where ``pᵛ⁺ᵣ`` is `reference_saturation_vapor_pressure`,
``T₀`` is `reference_temperature`, 
``a`` is an empirical coefficient, and
``b`` is a temperature offset.

Different coefficients are used for liquid water and ice surfaces. Default values
for the liquid formula are from Monteith and Unsworth (2008), and default values
for the ice formula are from Murray (1967):

**Liquid water** (T > 0°C):
- `liquid_coefficient`: 17.27
- `liquid_temperature_offset`: 35.85 K (corresponding to 237.3 K offset from 0°C)

**Ice** (T < 0°C):
- `ice_coefficient`: 21.875  
- `ice_temperature_offset`: 7.65 K (corresponding to 265.5 K offset from 0°C)

# References

- Tetens, O. (1930). Über einige meteorologische Begriffe. Zeitschrift für Geophysik, 6, 297-309.
- Monteith, J.L. and Unsworth, M.H. (2008). Principles of Environmental Physics. Academic Press.
- Murray, F.W. (1967). On the computation of saturation vapor pressure.
  Journal of Applied Meteorology, 6, 203-204.
  doi:[10.1175/1520-0450(1967)006<0203:OTCOSV>2.0.CO;2](https://doi.org/10.1175/1520-0450(1967)006<0203:OTCOSV>2.0.CO;2)
- [Wikipedia: Tetens equation](https://en.wikipedia.org/wiki/Tetens_equation)

# Example

```jldoctest
julia> using Breeze.Thermodynamics

julia> tf = TetensFormula()
TetensFormula{Float64}(pᵣ=610.0, T₀=273.15, aˡ=17.27, bˡ=35.85, aⁱ=21.875, bⁱ=7.65)
```
"""
function TetensFormula(FT = Oceananigans.defaults.FloatType;
                       reference_saturation_vapor_pressure = 610,
                       reference_temperature = 273.15,
                       liquid_coefficient = 17.27,
                       liquid_temperature_offset = 35.85,
                       ice_coefficient = 21.875,
                       ice_temperature_offset = 7.65)

    return TetensFormula{FT}(convert(FT, reference_saturation_vapor_pressure),
                             convert(FT, reference_temperature),
                             convert(FT, liquid_coefficient),
                             convert(FT, liquid_temperature_offset),
                             convert(FT, ice_coefficient),
                             convert(FT, ice_temperature_offset))
end

"""
    TetensFormulaThermodynamicConstants{FT, C, I}

Type alias for `ThermodynamicConstants` using the [Tetens formula](@ref Breeze.Thermodynamics.TetensFormula)
for saturation vapor pressure calculations.
"""
const TetensFormulaThermodynamicConstants{FT, C, I, TF<:TetensFormula} = ThermodynamicConstants{FT, C, I, TF}

"""
$(TYPEDSIGNATURES)

Compute the saturation vapor pressure over a planar liquid surface
using Tetens' empirical formula:

```math
pᵛ⁺(T) = pᵛ⁺ᵣ \\exp \\left( aˡ \\frac{T - T₀}{T - bˡ} \\right)
```
"""
@inline function saturation_vapor_pressure(T, constants::TetensFormulaThermodynamicConstants, ::PlanarLiquidSurface)
    tf = constants.saturation_vapor_pressure
    pᵛ⁺ᵣ = tf.reference_saturation_vapor_pressure
    a = tf.liquid_coefficient
    T₀ = tf.reference_temperature
    b = tf.liquid_temperature_offset
    return pᵛ⁺ᵣ * exp(a * (T - T₀) / (T - b))
end

"""
$(TYPEDSIGNATURES)

Compute the saturation vapor pressure over a planar ice surface
using Tetens' empirical formula with ice coefficients from Murray (1967):

```math
pᵛ⁺(T) = pᵛ⁺ᵣ \\exp \\left( aⁱ \\frac{T - T₀}{T - bⁱ} \\right)
```
"""
@inline function saturation_vapor_pressure(T, constants::TetensFormulaThermodynamicConstants, ::PlanarIceSurface)
    tf = constants.saturation_vapor_pressure
    pᵛ⁺ᵣ = tf.reference_saturation_vapor_pressure
    a = tf.ice_coefficient
    T₀ = tf.reference_temperature
    b = tf.ice_temperature_offset
    return pᵛ⁺ᵣ * exp(a * (T - T₀) / (T - b))
end

"""
$(TYPEDSIGNATURES)

Compute the saturation vapor pressure over a mixed-phase surface
by linearly interpolating between liquid and ice saturation vapor pressures
based on the liquid fraction.
"""
@inline function saturation_vapor_pressure(T, constants::TetensFormulaThermodynamicConstants, surface::PlanarMixedPhaseSurface)
    pᵛ⁺ˡ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
    pᵛ⁺ⁱ = saturation_vapor_pressure(T, constants, PlanarIceSurface())
    λ = surface.liquid_fraction
    return λ * pᵛ⁺ˡ + (1 - λ) * pᵛ⁺ⁱ
end

