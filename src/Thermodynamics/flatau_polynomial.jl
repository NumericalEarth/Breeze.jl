struct FlatauPolynomial{FT}
    liquid_coefficients :: NTuple{9, FT}
    ice_coefficients :: NTuple{9, FT}
    reference_temperature :: FT
    minimum_temperature_offset :: FT
end

function Base.summary(fp::FlatauPolynomial{FT}) where FT
    return string("FlatauPolynomial{", FT, "}(",
                  "Tᵣ=", prettysummary(fp.reference_temperature), ", ",
                  "order=", length(fp.liquid_coefficients) - 1, ")")
end

Base.show(io::IO, fp::FlatauPolynomial) = print(io, summary(fp))

function Adapt.adapt_structure(to, fp::FlatauPolynomial)
    liquid = adapt(to, fp.liquid_coefficients)
    ice = adapt(to, fp.ice_coefficients)
    Tᵣ = adapt(to, fp.reference_temperature)
    δT = adapt(to, fp.minimum_temperature_offset)
    FT = typeof(Tᵣ)
    return FlatauPolynomial{FT}(liquid, ice, Tᵣ, δT)
end

"""
$(TYPEDSIGNATURES)

Construct a `FlatauPolynomial` saturation vapor pressure formulation:
the eighth-order polynomial fits of [Flatau et al. (1992)](@cite Flatau1992)
to the saturation vapor pressure over planar liquid and ice surfaces,

```math
pᵛ⁺(T) = \\sum_{n=0}^{8} aₙ (T - Tᵣ)^n ,
```

with `reference_temperature` ``Tᵣ = 273.16`` K and the relative-error-norm coefficient
sets (their Tables 3 and 4), which are the fits in operational use in WRF-family
microphysics. The temperature argument is clamped below at
``Tᵣ -`` `minimum_temperature_offset` (80 K), the fits' stated range of validity.

Compared to the default integrated Clausius–Clapeyron formulation the polynomial agrees
to within 0.2 % (liquid, 233–313 K) while replacing a `^` and an `exp` with a
branch-free Horner chain — approximately 70× cheaper per call on CPU Float64 and free of
the FP64 transcendental penalty on GPUs. See `ClausiusClapeyron` and `TetensFormula` for
the alternative formulations.

Example
=======

```julia
using Breeze.Thermodynamics

constants = ThermodynamicConstants(; saturation_vapor_pressure = FlatauPolynomial())
```

# References

* Flatau, P. J., Walko, R. L. and Cotton, W. R. (1992). Polynomial fits to saturation
  vapor pressure. Journal of Applied Meteorology 31, 1507–1513.
"""
function FlatauPolynomial(FT = Oceananigans.defaults.FloatType;
                          # Flatau et al. (1992) relative-error-norm coefficients, converted
                          # from their hPa convention to Pa (× 100)
                          liquid_coefficients = (611.239921, 44.3987641, 1.42986287,
                                                 2.64847430e-2, 3.02950461e-4, 2.06739458e-6,
                                                 6.40689451e-9, -9.52447341e-12, -9.76195544e-14),
                          ice_coefficients = (611.147274, 50.3160820, 1.88439774,
                                              4.20895665e-2, 6.15021634e-4, 6.02588177e-6,
                                              3.85852041e-8, 1.46898966e-10, 2.52751365e-13),
                          reference_temperature = 273.16,
                          minimum_temperature_offset = 80)

    return FlatauPolynomial{FT}(map(c -> convert(FT, c), Tuple(liquid_coefficients)),
                                map(c -> convert(FT, c), Tuple(ice_coefficients)),
                                convert(FT, reference_temperature),
                                convert(FT, minimum_temperature_offset))
end

"""
    FlatauPolynomialThermodynamicConstants{FT, C, I}

Type alias for `ThermodynamicConstants` using the Flatau et al. (1992) polynomial fits
for saturation vapor pressure calculations.
"""
const FlatauPolynomialThermodynamicConstants{FT, C, I, FP<:FlatauPolynomial} = ThermodynamicConstants{FT, C, I, FP}

"""
$(TYPEDSIGNATURES)

Compute the saturation vapor pressure over a planar liquid surface from the
Flatau et al. (1992) eighth-order polynomial in ``T - Tᵣ``.
"""
@inline function saturation_vapor_pressure(T, constants::FlatauPolynomialThermodynamicConstants, ::PlanarLiquidSurface)
    fp = constants.saturation_vapor_pressure
    x = max(T - fp.reference_temperature, -fp.minimum_temperature_offset)
    return evalpoly(x, fp.liquid_coefficients)
end

"""
$(TYPEDSIGNATURES)

Compute the saturation vapor pressure over a planar ice surface from the
Flatau et al. (1992) eighth-order polynomial in ``T - Tᵣ``.
"""
@inline function saturation_vapor_pressure(T, constants::FlatauPolynomialThermodynamicConstants, ::PlanarIceSurface)
    fp = constants.saturation_vapor_pressure
    x = max(T - fp.reference_temperature, -fp.minimum_temperature_offset)
    return evalpoly(x, fp.ice_coefficients)
end
