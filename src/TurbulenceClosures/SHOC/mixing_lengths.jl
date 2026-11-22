#####
##### The mixing length of SHOC as a mixing length of `TKEBasedTurbulenceClosure`
#####
##### `BlendedMixingLength` is an `AbstractMixingLength`: the closure fills it column by column into
##### `closure_fields.ℓᵐ` at the faces (`fill_mixing_length!`), and reconstructs it at the centers
##### (`mixing_lengthᶜᶜᶜ`), where the dissipation lives with `e`. Its one number of the column, the
##### asymptotic length `l∞`, is kept in `closure_fields.ℓˢ` (the field that holds what the center
##### reconstruction needs), at every face of the column.
#####

"""
$(TYPEDEF)

The mixing length of [Bogenschutz and Krueger (2013)](@cite Bogenschutz2013) as E3SM `shoc.F90` computes it
(`compute_shoc_mix_shoc_length`): three lengths blended by the sum of their inverse squares, the distance to
the surface `κ z`, an asymptotic length `l∞` (the fraction `ℂˡ₄` of the height weighted by `√e` over the
column) and, in stable air, the buoyancy length `√e / N`,

    ℓ⁻² = (ℂˡ₁ / ℂˡ₂)² [1 / (ℂˡ₅ √e κ z) + 1 / (ℂˡ₅ √e l∞) + ℂˡ₃ N²⁺ / e] + 1 / ℓᵐᵃˣ² + 1 / Δʰ²,

with `N²⁺` the stable part of the stored static stability and `√e` the turbulent velocity of the closure
(`e` floored at `minimum_tke`). `shoc.F90` bounds the length with `min(maxlen, ⋅)`, `max(minlen, ⋅)` and
`min(√(Δx Δy), ⋅)` (`check_length_scale_shoc_length`). Here the upper bounds `ℓᵐᵃˣ` (`maximum_length`) and the
horizontal grid spacing `Δʰ = √(Δx Δy)` (none along `Flat` directions) enter the same sum of inverse squares,
which changes the length by the fraction `(ℓ / ℓᵐᵃˣ)² / 2` where it is far below them, and the lower bound
`minlen` is not imposed: the length vanishes with `z` and with `√e`, and the dissipation `𝕊ᴰ e^{3/2} / ℓ`
vanishes with them.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂˡ₁` | `length_fac`     | 0.5    | divides the mixing length |
| `ℂˡ₂` | literal `2.8284` | 2.8284 | prefactor of the mixing length (`2√2`, truncated as in the Fortran) |
| `ℂˡ₃` | literal `0.01`   | 0.01   | weight of the stratification term `N² / e` |
| `ℂˡ₄` | literal `0.1`    | 0.1    | the asymptotic length `l∞` as a fraction of the TKE-weighted mean height |
| `ℂˡ₅` | `tscale`         | 400    | eddy turnover time scale, **in seconds** (the one dimensional coefficient) |

$(TYPEDFIELDS)
"""
struct BlendedMixingLength{FT} <: AbstractMixingLength
    "Mixing length divisor (`length_fac`)"
    ℂˡ₁ :: FT
    "Mixing length prefactor (literal `2.8284`)"
    ℂˡ₂ :: FT
    "Weight of the stratification term of the mixing length (literal `0.01`)"
    ℂˡ₃ :: FT
    "Asymptotic length as a fraction of the TKE-weighted mean height (literal `0.1`)"
    ℂˡ₄ :: FT
    "Eddy turnover time scale of the mixing length, s (`tscale`)"
    ℂˡ₅ :: FT
    "Von Kármán constant of the distance to the surface (`vk`)"
    von_karman_constant :: FT
    "Largest mixing length, m (`maxlen`), in the sum of inverse squares"
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

TurbulenceClosures.convert_eltype(::Type{FT}, ml::BlendedMixingLength) where FT =
    map_scalars(scalar -> convert(FT, scalar), ml)

Base.summary(ml::BlendedMixingLength{FT}) where FT = "BlendedMixingLength{$FT}"

Base.show(io::IO, ml::BlendedMixingLength) =
    print(io, summary(ml), " (ℂˡ₁ = ", prettysummary(ml.ℂˡ₁), ", ℂˡ₂ = ", prettysummary(ml.ℂˡ₂),
              ", ℂˡ₃ = ", prettysummary(ml.ℂˡ₃), ", ℂˡ₄ = ", prettysummary(ml.ℂˡ₄), ", ℂˡ₅ = ", prettysummary(ml.ℂˡ₅), ")")

# The squared horizontal grid spacing that bounds the length: none along a `Flat` direction
@inline horizontal_spacing_squared(i, j, k, grid) = Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid)
@inline horizontal_spacing_squared(i, j, k, grid::AbstractGrid{<:Any, <:Flat}) = convert(grid_float_type(grid), Inf)
@inline horizontal_spacing_squared(i, j, k, grid::AbstractGrid{<:Any, <:Any, <:Flat}) = convert(grid_float_type(grid), Inf)
@inline horizontal_spacing_squared(i, j, k, grid::AbstractGrid{<:Any, <:Flat, <:Flat}) = convert(grid_float_type(grid), Inf)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The blended length at height `z` above the surface, with turbulent velocity `u = √e`, asymptotic length
`l∞`, stable stratification `N²⁺ ≥ 0` and squared horizontal grid spacing `Δʰ²`. The sum of inverse squares
is multiplied through by `A = ℂˡ₅ u κ z`, so that the length is `√(A / D) ℂˡ₂ / ℂˡ₁` with
`D = 1 + κ z / l∞ + ℂˡ₃ ℂˡ₅ κ z N²⁺ / u + (ℂˡ₁ / ℂˡ₂)² A (1 / ℓᵐᵃˣ² + 1 / Δʰ²)`, finite and zero at `z = 0`.
"""
@inline function blended_length(ml::BlendedMixingLength, z, u, l∞, N²⁺, Δʰ²)
    κz = ml.von_karman_constant * z
    A = ml.ℂˡ₅ * u * κz
    r = ml.ℂˡ₁ / ml.ℂˡ₂
    bounds = 1 / (ml.maximum_length * ml.maximum_length) + 1 / Δʰ²
    D = 1 + κz / l∞ + ml.ℂˡ₃ * ml.ℂˡ₅ * κz * N²⁺ / u + r * r * A * bounds
    return sqrt(A / D) / r
end

# The asymptotic length l∞ = ℂˡ₄ ∫ √e z dz / ∫ √e dz over the column, from the cell centers
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

@inline function TurbulenceClosures.fill_mixing_length!(ℓ, ℓˢ, i, j, grid, ml::BlendedMixingLength, closure, e, N²)
    l∞ = asymptotic_length(i, j, grid, ml, closure, e)
    for k in 1:size(grid, 3)+1
        u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
        N²⁺ = max(@inbounds(N²[i, j, k]), zero(l∞))
        z = height_above_bottomᶜᶜᶠ(i, j, k, grid)
        @inbounds ℓ[i, j, k] = blended_length(ml, z, u, l∞, N²⁺, horizontal_spacing_squared(i, j, k, grid))
        @inbounds ℓˢ[i, j, k] = l∞
    end
    return nothing
end

# At a cell center: the blend of the center, with N² reconstructed from the two adjacent faces
@inline function TurbulenceClosures.mixing_lengthᶜᶜᶜ(i, j, k, grid, ml::BlendedMixingLength, closure, e, closure_fields)
    FT = grid_float_type(grid)
    u = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    N²⁺ = max(ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, closure_fields.N²), zero(FT))
    l∞ = @inbounds closure_fields.ℓˢ[i, j, k]
    z = height_above_bottomᶜᶜᶜ(i, j, k, grid)
    return blended_length(ml, z, u, l∞, N²⁺, horizontal_spacing_squared(i, j, k, grid))
end
