#####
##### Shear: the squared shear S² in the Richardson number Ri = N² / (S² + S²ᵐⁱⁿ) of
##### `TKEBasedTurbulenceClosure`.
#####
##### `GridMeanShear` is the shear of the resolved wind, (∂z u)² + (∂z v)². `TurbulenceAugmentedShear`
##### adds the squared turnover rate of the subgrid eddies, ℂˢᵗ e / ℓ², so that a turbulent layer
##### without resolved shear — the interior of a convective cloud layer — has a Richardson number
##### of order N² ℓ² / e instead of N² / S²ᵐⁱⁿ. Both keep the floor S²ᵐⁱⁿ > 0 of
##### richardson_number_stability_functions.jl.
#####

"""
$(TYPEDEF)

The squared shear of the Richardson number of [`TKEBasedTurbulenceClosure`](@ref) from the
resolved wind alone, ``S² = (∂_z u)² + (∂_z v)²``, so that ``Ri = N² / (S² + S²ᵐⁱⁿ)`` with
``S²ᵐⁱⁿ`` the `minimum_shear_squared`. This is the default.

Fields
======

$(TYPEDFIELDS)
"""
struct GridMeanShear{FT}
    "floor ``S²ᵐⁱⁿ > 0`` added to the squared shear, s⁻²"
    minimum_shear_squared :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`GridMeanShear`](@ref) with the floor `minimum_shear_squared`.

```jldoctest
using Breeze

GridMeanShear()

# output
GridMeanShear(minimum_shear_squared = 1.0e-10)
```
"""
GridMeanShear(; minimum_shear_squared = 1e-10) = GridMeanShear(minimum_shear_squared)

"""
$(TYPEDEF)

The squared shear of the Richardson number of [`TKEBasedTurbulenceClosure`](@ref) from the
resolved wind and the subgrid turbulence,

```math
Ri = \\frac{N²}{S² + ℂˢᵗ e / ℓ² + S²ᵐⁱⁿ},
```

where ``S² = (∂_z u)² + (∂_z v)²``, ``\\sqrt{e} / ℓ`` is the turnover rate of the subgrid eddies,
``ℂˢᵗ ≥ 0`` the `turbulence_shear_coefficient` and ``S²ᵐⁱⁿ`` the `minimum_shear_squared`.
In a turbulent layer without resolved shear ``Ri`` is then of order ``N² ℓ² / (ℂˢᵗ e)`` rather
than ``N² / S²ᵐⁱⁿ``, so the stability functions respond to the stratification in proportion to
the turbulence that is there to feel it instead of jumping to their limits. In a neutral surface
layer ``e / ℓ²`` and ``S²`` both scale as ``u_⋆² / z²``.

``e`` is floored at the closure's `minimum_tke` and reconstructed at the cell interface, exactly
as in the diffusivity ``ℓ \\sqrt{e}``. ``ℓ`` is the stored `closure_fields.ℓᵐ`: every mixing-length
formulation depends on ``N²`` and ``e`` but not on ``Ri``, and is computed before the stability
functions in each update, so the dependence is not circular and ``ℓ`` belongs to the same state as
``S²`` and ``N²``. Where ``ℓ = 0`` (the surface, where the diffusivities vanish) the turbulence
term is zero.

The floor is kept, so that ``N² = 0`` gives ``Ri = 0`` for any ``e``, and
``ℂˢᵗ = 0``, the default, reproduces [`GridMeanShear`](@ref) bit for bit.

Fields
======

$(TYPEDFIELDS)
"""
struct TurbulenceAugmentedShear{FT}
    "coefficient ``ℂˢᵗ ≥ 0`` of the squared turnover rate ``e / ℓ²``"
    turbulence_shear_coefficient :: FT
    "floor ``S²ᵐⁱⁿ > 0`` added to the squared shear, s⁻²"
    minimum_shear_squared :: FT
end

TurbulenceAugmentedShear(ℂˢᵗ, S²ᵐⁱⁿ) = TurbulenceAugmentedShear(promote(ℂˢᵗ, S²ᵐⁱⁿ)...)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`TurbulenceAugmentedShear`](@ref) with the given `turbulence_shear_coefficient`
and floor `minimum_shear_squared`.

```jldoctest
using Breeze

TurbulenceAugmentedShear(turbulence_shear_coefficient = 0.1)

# output
TurbulenceAugmentedShear(turbulence_shear_coefficient = 0.1, minimum_shear_squared = 1.0e-10)
```
"""
TurbulenceAugmentedShear(; turbulence_shear_coefficient = 0, minimum_shear_squared = 1e-10) =
    TurbulenceAugmentedShear(turbulence_shear_coefficient, minimum_shear_squared)

# Coefficients may be traced scalars when a shared closure is differentiated.
Adapt.@adapt_structure GridMeanShear
Adapt.@adapt_structure TurbulenceAugmentedShear

@inline convert_eltype(::Type{FT}, shear::GridMeanShear) where FT =
    GridMeanShear{FT}(convert(FT, shear.minimum_shear_squared))

@inline convert_eltype(::Type{FT}, shear::TurbulenceAugmentedShear) where FT =
    TurbulenceAugmentedShear{FT}(convert(FT, shear.turbulence_shear_coefficient),
                                 convert(FT, shear.minimum_shear_squared))

# The closure's `minimum_shear_squared` keyword replaces the floor of whichever shear it is given
with_minimum_shear_squared(shear, ::Nothing) = shear
with_minimum_shear_squared(shear, S²ᵐⁱⁿ::Number) = replace_minimum_shear_squared(shear, S²ᵐⁱⁿ)

replace_minimum_shear_squared(::GridMeanShear, S²ᵐⁱⁿ) = GridMeanShear(S²ᵐⁱⁿ)
replace_minimum_shear_squared(shear::TurbulenceAugmentedShear, S²ᵐⁱⁿ) =
    TurbulenceAugmentedShear(shear.turbulence_shear_coefficient, S²ᵐⁱⁿ)

function validate_shear(shear)
    S²ᵐⁱⁿ = shear.minimum_shear_squared
    S²ᵐⁱⁿ > 0 ||
        throw(ArgumentError("minimum_shear_squared must be positive, so that the Richardson number " *
                            "N² / (S² + minimum_shear_squared) is finite, but it is $S²ᵐⁱⁿ"))
    return nothing
end

function validate_shear(shear::TurbulenceAugmentedShear)
    ℂˢᵗ = shear.turbulence_shear_coefficient
    ℂˢᵗ ≥ 0 ||
        throw(ArgumentError("turbulence_shear_coefficient must not be negative, but it is $ℂˢᵗ"))
    return validate_shear(GridMeanShear(shear.minimum_shear_squared))
end

Base.summary(::GridMeanShear) = "GridMeanShear"
Base.summary(::TurbulenceAugmentedShear) = "TurbulenceAugmentedShear"

Base.show(io::IO, shear::GridMeanShear) =
    print(io, summary(shear), "(minimum_shear_squared = ", prettysummary(shear.minimum_shear_squared), ")")

Base.show(io::IO, shear::TurbulenceAugmentedShear) =
    print(io, summary(shear), "(turbulence_shear_coefficient = ", prettysummary(shear.turbulence_shear_coefficient),
              ", minimum_shear_squared = ", prettysummary(shear.minimum_shear_squared), ")")

#####
##### The squared shear at the faces, before the floor
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The squared turnover rate of the subgrid eddies at (Center, Center, Face), ``ℂˢᵗ e / ℓ²``, with
``\\sqrt{e}`` floored and reconstructed as in the diffusivity and `ℓ` the stored mixing length.
Zero where ``ℓ = 0``.
"""
@inline function turbulence_shear_squaredᶜᶜᶠ(i, j, k, grid, closure, e, ℓ)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
    ℓᵢ = @inbounds ℓ[i, j, k]
    turbulent = ℓᵢ > 0
    # Both branches of an `ifelse` are evaluated and differentiated, so the unselected one divides by one
    ℓ⁺ = ifelse(turbulent, ℓᵢ, one(ℓᵢ))
    S²ᵗ = closure.shear.turbulence_shear_coefficient * (w★ / ℓ⁺)^2
    return ifelse(turbulent, S²ᵗ, zero(S²ᵗ))
end

@inline effective_shear_squaredᶜᶜᶠ(i, j, k, grid, closure, velocities, e, ℓ) =
    effective_shear_squaredᶜᶜᶠ(i, j, k, grid, closure.shear, closure, velocities, e, ℓ)

@inline effective_shear_squaredᶜᶜᶠ(i, j, k, grid, ::GridMeanShear, closure, velocities, e, ℓ) =
    shearᶜᶜᶠ(i, j, k, grid, velocities.u, velocities.v)

@inline effective_shear_squaredᶜᶜᶠ(i, j, k, grid, ::TurbulenceAugmentedShear, closure, velocities, e, ℓ) =
    shearᶜᶜᶠ(i, j, k, grid, velocities.u, velocities.v) + turbulence_shear_squaredᶜᶜᶠ(i, j, k, grid, closure, e, ℓ)
