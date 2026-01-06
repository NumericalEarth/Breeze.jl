"""
$(TYPEDEF)

A saturation vapor pressure formulation based on Tetens' empirical formula.

The Tetens formula approximates saturation vapor pressure as:

```math
pᵛ⁺(T) = pᵛ⁺₀ \\exp \\left( a \\frac{T - T₀}{T - b} \\right)
```

where ``pᵛ⁺₀`` is the reference saturation vapor pressure, ``a`` is an empirical coefficient,
``T₀`` is the reference temperature, and ``b`` is a temperature offset.

Default parameter values are derived from equation 2.11 in [Klemp1978](@cite):

- `reference_saturation_vapor_pressure`: 610 Pa
- `coefficient`: 17.27
- `reference_temperature`: 273 K
- `temperature_offset`: 36 K

# Reference

Klemp, J. B., & Wilhelmson, R. B. (1978). The Simulation of Three-Dimensional 
Convective Storm Dynamics. *Journal of the Atmospheric Sciences*, 35(6), 1070-1096.
[DOI: 10.1175/1520-0469(1978)035<1070:TSOTDC>2.0.CO;2](https://doi.org/10.1175/1520-0469(1978)035<1070:TSOTDC>2.0.CO;2)
"""
struct TetensFormula{FT}
    reference_saturation_vapor_pressure :: FT
    coefficient :: FT
    reference_temperature :: FT
    temperature_offset :: FT
end

function Base.summary(tf::TetensFormula{FT}) where FT
    return string("TetensFormula{", FT, "}(",
                  "p₀=", prettysummary(tf.reference_saturation_vapor_pressure), ", ",
                  "a=", prettysummary(tf.coefficient), ", ",
                  "T₀=", prettysummary(tf.reference_temperature), ", ",
                  "b=", prettysummary(tf.temperature_offset), ")")
end

Base.show(io::IO, tf::TetensFormula) = print(io, summary(tf))

function Adapt.adapt_structure(to, tf::TetensFormula)
    pᵛ⁺₀ = adapt(to, tf.reference_saturation_vapor_pressure)
    a = adapt(to, tf.coefficient)
    T₀ = adapt(to, tf.reference_temperature)
    b = adapt(to, tf.temperature_offset)
    FT = typeof(pᵛ⁺₀)
    return TetensFormula{FT}(pᵛ⁺₀, a, T₀, b)
end

"""
$(TYPEDSIGNATURES)

Construct a `TetensFormula` saturation vapor pressure formulation.

# Keyword Arguments

- `reference_saturation_vapor_pressure`: Saturation vapor pressure at the reference
  temperature (default: 610 Pa, corresponding to ~6.1 hPa at 0°C)
- `coefficient`: Empirical coefficient in the exponential (default: 17.27)
- `reference_temperature`: Reference temperature in Kelvin (default: 273 K)
- `temperature_offset`: Temperature offset in the denominator (default: 36 K)

# Example

```jldoctest
julia> using Breeze

julia> tf = TetensFormula()
TetensFormula{Float64}(p₀=610.0, a=17.27, T₀=273.0, b=36.0)
```
"""
function TetensFormula(FT = Oceananigans.defaults.FloatType;
                       reference_saturation_vapor_pressure = 610,
                       coefficient = 17.27,
                       reference_temperature = 273,
                       temperature_offset = 36)

    return TetensFormula{FT}(convert(FT, reference_saturation_vapor_pressure),
                             convert(FT, coefficient),
                             convert(FT, reference_temperature),
                             convert(FT, temperature_offset))
end

"""
    TetensFormulaThermodynamicConstants{FT, C, I}

Type alias for `ThermodynamicConstants` using the Tetens formula
for saturation vapor pressure calculations.
"""
const TetensFormulaThermodynamicConstants{FT, C, I, TF<:TetensFormula} = ThermodynamicConstants{FT, C, I, TF}

"""
$(TYPEDSIGNATURES)

Compute the saturation vapor pressure using Tetens' empirical formula:

```math
pᵛ⁺(T) = pᵛ⁺₀ \\exp \\left( a \\frac{T - T₀}{T - b} \\right)
```

This formulation ignores the `surface` argument as the Tetens formula
parameters are typically calibrated for liquid water surfaces only.
"""
@inline function saturation_vapor_pressure(T, constants::TetensFormulaThermodynamicConstants, surface)
    tf = constants.saturation_vapor_pressure
    pᵛ⁺₀ = tf.reference_saturation_vapor_pressure
    a = tf.coefficient
    T₀ = tf.reference_temperature
    b = tf.temperature_offset
    return pᵛ⁺₀ * exp(a * (T - T₀) / (T - b))
end

