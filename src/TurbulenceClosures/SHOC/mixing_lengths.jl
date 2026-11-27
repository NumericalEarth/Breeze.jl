"""
$(TYPEDEF)

The mixing length of [Bogenschutz and Krueger (2013)](@cite Bogenschutz2013) as E3SM `shoc.F90` computes it
(`compute_shoc_mix_shoc_length`): three lengths blended by the sum of their inverse squares, the distance to
the surface `κ z`, an asymptotic length `ℓ∞` (the fraction `ℂˡ₄` of the height weighted by `√e` over the
column) and, in stable air, the buoyancy length `√e / N`,
"""
struct BlendedMixingLength{FT} <: AbstractMixingLength
    ℂˡ₁ :: FT
    ℂˡ₂ :: FT
    ℂˡ₃ :: FT
    ℂˡ₄ :: FT
    ℂˡ₅ :: FT
    von_karman_constant :: FT
    maximum_length :: FT
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `BlendedMixingLength{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword argument.
"""
BlendedMixingLength(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(BlendedMixingLength, FT,
                            (ℂˡ₁ = 0.5, ℂˡ₂ = 2.8284, ℂˡ₃ = 0.01, ℂˡ₄ = 0.1, ℂˡ₅ = 400,
                             von_karman_constant = 0.4, maximum_length = 20000), kw)

Adapt.@adapt_structure BlendedMixingLength

convert_eltype(::Type{FT}, ml::BlendedMixingLength) where FT =
    map_scalars(scalar -> convert(FT, scalar), ml)

Base.summary(ml::BlendedMixingLength{FT}) where FT = "BlendedMixingLength{$FT}"

Base.show(io::IO, ml::BlendedMixingLength) =
    print(io, summary(ml), " (ℂˡ₁ = ", prettysummary(ml.ℂˡ₁), ", ℂˡ₂ = ", prettysummary(ml.ℂˡ₂),
              ", ℂˡ₃ = ", prettysummary(ml.ℂˡ₃), ", ℂˡ₄ = ", prettysummary(ml.ℂˡ₄), ", ℂˡ₅ = ", prettysummary(ml.ℂˡ₅), ")")

@inline horizontal_spacing_squaredᶜᶜᵃ(i, j, k, grid) = Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid)
@inline horizontal_spacing_squaredᶜᶜᵃ(i, j, k, grid::AbstractGrid{<:Any, <:Flat}) = convert(grid_float_type(grid), Inf)
@inline horizontal_spacing_squaredᶜᶜᵃ(i, j, k, grid::AbstractGrid{<:Any, <:Any, <:Flat}) = convert(grid_float_type(grid), Inf)
@inline horizontal_spacing_squaredᶜᶜᵃ(i, j, k, grid::AbstractGrid{<:Any, <:Flat, <:Flat}) = convert(grid_float_type(grid), Inf)

@inline function blended_length(ml::BlendedMixingLength, z, u, ℓ∞, N², Δʰ²)
    κz = ml.von_karman_constant * z
    A = ml.ℂˡ₅ * u * κz
    r = ml.ℂˡ₁ / ml.ℂˡ₂
    bounds = 1 / (ml.maximum_length * ml.maximum_length) + 1 / Δʰ²
    D₀ = 1 + κz / ℓ∞ + A * bounds / (r * r)
    X = ml.ℂˡ₃ * ml.ℂˡ₅ * κz * N² / u
    X₀ = ml.ℂˡ₃ * (r * r) * D₀
    hypotenuse = sqrt(X * X + X₀ * X₀)
    X₊ = ifelse(X >= 0, (X + hypotenuse) / 2, X₀ * X₀ / (2 * (hypotenuse - X)))
    return sqrt(A / (D₀ + X₊)) / r
end

@inline function asymptotic_length(i, j, grid, ml::BlendedMixingLength, closure, e)
    FT = grid_float_type(grid)
    numerator = zero(FT)
    denominator = zero(FT)
    for k in 1:size(grid, 3)
        u = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
        uΔz = u * Δzᶜᶜᶜ(i, j, k, grid)
        numerator += uΔz * height_above_bottomᶜᶜᶜ(i, j, k, grid)
        denominator += uΔz
    end
    return ml.ℂˡ₄ * numerator / denominator
end

@inline function fill_mixing_length!(ℓ, ℓˢ, i, j, grid, ml::BlendedMixingLength, closure, e, N²)
    ℓ∞ = asymptotic_length(i, j, grid, ml, closure, e)
    for k in 1:size(grid, 3)+1
        u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
        z = height_above_bottomᶜᶜᶠ(i, j, k, grid)
        @inbounds ℓ[i, j, k] = blended_length(ml, z, u, ℓ∞, N²[i, j, k], horizontal_spacing_squaredᶜᶜᵃ(i, j, k, grid))
        @inbounds ℓˢ[i, j, k] = ℓ∞
    end
    return nothing
end

@inline function mixing_lengthᶜᶜᶜ(i, j, k, grid, ml::BlendedMixingLength, closure, e, closure_fields)
    u = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    N² = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, closure_fields.N²)
    ℓ∞ = @inbounds closure_fields.ℓˢ[i, j, k]
    z = height_above_bottomᶜᶜᶜ(i, j, k, grid)
    return blended_length(ml, z, u, ℓ∞, N², horizontal_spacing_squaredᶜᶜᵃ(i, j, k, grid))
end
