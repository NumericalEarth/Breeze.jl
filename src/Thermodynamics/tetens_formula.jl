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
                  "Tᵣ=", prettysummary(tf.reference_temperature), ", ",
                  "aˡ=", prettysummary(tf.liquid_coefficient), ", ",
                  "δTˡ=", prettysummary(tf.liquid_temperature_offset), ", ",
                  "aⁱ=", prettysummary(tf.ice_coefficient), ", ",
                  "δTⁱ=", prettysummary(tf.ice_temperature_offset), ")")
end

Base.show(io::IO, tf::TetensFormula) = print(io, summary(tf))

function Adapt.adapt_structure(to, tf::TetensFormula)
    pᵛ⁺ᵣ = adapt(to, tf.reference_saturation_vapor_pressure)
    Tᵣ = adapt(to, tf.reference_temperature)
    aˡ = adapt(to, tf.liquid_coefficient)
    δTˡ = adapt(to, tf.liquid_temperature_offset)
    aⁱ = adapt(to, tf.ice_coefficient)
    δTⁱ = adapt(to, tf.ice_temperature_offset)
    FT = typeof(pᵛ⁺ᵣ)
    return TetensFormula{FT}(pᵛ⁺ᵣ, Tᵣ, aˡ, δTˡ, aⁱ, δTⁱ)
end

"""
$(TYPEDSIGNATURES)

Construct a `TetensFormula` saturation vapor pressure formulation.
[Tetens's (1930)](@cite Tetens1930) formula is an empirical relationship
for the saturation vapor pressure,

```math
pᵛ⁺(T) = pᵛ⁺ᵣ \\exp \\left( a \\frac{T - Tᵣ}{T - δT} \\right) ,
```

where ``pᵛ⁺ᵣ`` is `reference_saturation_vapor_pressure`,
``Tᵣ`` is `reference_temperature`,
``a`` is an empirical coefficient, and
``δT`` is a temperature offset.

See also the [wikipedia article on "Tetens equation"](https://en.wikipedia.org/wiki/Tetens_equation).
Different coefficients are used for liquid water and ice surfaces. Default values
for the liquid formula are from [Monteith and Unsworth (2014)](@cite MonteithUnsworth2014), and default values
for the ice formula are from [Murray (1967)](@cite Murray1967):

**Liquid water** (T > 0°C):
- `liquid_coefficient`: 17.27
- `liquid_temperature_offset`: 35.85 K (corresponding to 237.3 K offset from 0°C)

**Ice** (T < 0°C):
- `ice_coefficient`: 21.875
- `ice_temperature_offset`: 7.65 K (corresponding to 265.5 K offset from 0°C)

# References

* Tetens, O. (1930). Über einige meteorologische Begriffe. Zeitschrift für Geophysik 6, 297–309.
* Monteith, J. L. and Unsworth, M. H. (2014). Principles of Environmental Physics. 4th Edition (Academic Press).
* Murray, F. W. (1967). On the computation of saturation vapor pressure. Journal of Applied Meteorology 6, 203–204.
* Wikipedia: Tetens equation; https://en.wikipedia.org/wiki/Tetens_equation

# Example

```jldoctest
julia> using Breeze.Thermodynamics

julia> tf = TetensFormula()
TetensFormula{Float64}(pᵣ=610.0, Tᵣ=273.15, aˡ=17.27, δTˡ=35.85, aⁱ=21.875, δTⁱ=7.65)
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

Type alias for `ThermodynamicConstants` using the Tetens formula
for saturation vapor pressure calculations.
"""
const TetensFormulaThermodynamicConstants{FT, C, I, TF<:TetensFormula} = ThermodynamicConstants{FT, C, I, TF}

"""
$(TYPEDSIGNATURES)

Compute the saturation vapor pressure over a planar liquid surface
using Tetens' empirical formula:

```math
pᵛ⁺(T) = pᵛ⁺ᵣ \\exp \\left( aˡ \\frac{T - Tᵣ}{T - δTˡ} \\right)
```
"""
@inline function saturation_vapor_pressure(T, constants::TetensFormulaThermodynamicConstants, ::PlanarLiquidSurface)
    tf = constants.saturation_vapor_pressure
    pᵛ⁺ᵣ = tf.reference_saturation_vapor_pressure
    a = tf.liquid_coefficient
    Tᵣ = tf.reference_temperature
    δT = tf.liquid_temperature_offset
    return pᵛ⁺ᵣ * exp(a * (T - Tᵣ) / (T - δT))
end

"""
$(TYPEDSIGNATURES)

Compute the saturation vapor pressure over a planar ice surface using [Tetens'](@cite Tetens1930)
empirical formula with ice coefficients from [Murray (1967)](@cite  Murray1967):

```math
pᵛ⁺(T) = pᵛ⁺ᵣ \\exp \\left( aⁱ \\frac{T - Tᵣ}{T - δTⁱ} \\right)
```

# References
* Tetens, O. (1930). Über einige meteorologische Begriffe. Zeitschrift für Geophysik 6, 297–309.
* Murray, F. W. (1967). On the computation of saturation vapor pressure. Journal of Applied Meteorology 6, 203–204.
"""
@inline function saturation_vapor_pressure(T, constants::TetensFormulaThermodynamicConstants, ::PlanarIceSurface)
    tf = constants.saturation_vapor_pressure
    pᵛ⁺ᵣ = tf.reference_saturation_vapor_pressure
    a = tf.ice_coefficient
    Tᵣ = tf.reference_temperature
    δT = tf.ice_temperature_offset
    return pᵛ⁺ᵣ * exp(a * (T - Tᵣ) / (T - δT))
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

