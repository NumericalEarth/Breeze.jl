"""
$(TYPEDEF)

The squared shear of the Richardson number of [`TKEBasedTurbulenceClosure`](@ref) from the
resolved wind alone, ``S² = (∂_z u)² + (∂_z v)²``, so that ``Ri = N² / (S² + S²ᵐⁱⁿ)`` with
``S²ᵐⁱⁿ`` the `minimum_shear_squared`. This is the default.
"""
struct GridMeanShear{FT}
    minimum_shear_squared :: FT
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`GridMeanShear`](@ref) with the floor `minimum_shear_squared`.
"""
GridMeanShear(; minimum_shear_squared = 1e-10) = GridMeanShear(minimum_shear_squared)

Adapt.@adapt_structure GridMeanShear

@inline convert_eltype(::Type{FT}, shear::GridMeanShear) where FT =
    GridMeanShear{FT}(convert(FT, shear.minimum_shear_squared))

with_minimum_shear_squared(shear, ::Nothing) = shear
with_minimum_shear_squared(shear, S²ᵐⁱⁿ::Number) = replace_minimum_shear_squared(shear, S²ᵐⁱⁿ)

replace_minimum_shear_squared(::GridMeanShear, S²ᵐⁱⁿ) = GridMeanShear(S²ᵐⁱⁿ)

function validate_shear(shear)
    S²ᵐⁱⁿ = shear.minimum_shear_squared
    S²ᵐⁱⁿ > 0 ||
        throw(ArgumentError("minimum_shear_squared must be positive, so that the Richardson number " *
                            "N² / (S² + minimum_shear_squared) is finite, but it is $S²ᵐⁱⁿ"))
    return nothing
end

Base.summary(::GridMeanShear) = "GridMeanShear"

Base.show(io::IO, shear::GridMeanShear) =
    print(io, summary(shear), "(minimum_shear_squared = ", prettysummary(shear.minimum_shear_squared), ")")

#####
##### The squared shear at the faces, before the floor
#####

@inline effective_shear_squaredᶜᶜᶠ(i, j, k, grid, closure, velocities, e, ℓ) =
    effective_shear_squaredᶜᶜᶠ(i, j, k, grid, closure.shear, closure, velocities, e, ℓ)

@inline effective_shear_squaredᶜᶜᶠ(i, j, k, grid, ::GridMeanShear, closure, velocities, e, ℓ) =
    shearᶜᶜᶠ(i, j, k, grid, velocities.u, velocities.v)
