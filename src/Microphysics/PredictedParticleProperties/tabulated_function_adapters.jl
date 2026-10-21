using Oceananigans.Utils: TabulatedFunction,
                          TabulatedFunction1D,
                          TabulatedFunction4D,
                          TabulatedFunction5D,
                          interpolator

#####
##### Table wrappers carrying an indexed rime-density axis
#####
##### The ASCII lookup tables sample rime density on the non-uniform grid
##### {50, 250, 450, 650, 900} kg/m³. Rather than resample, the wrappers below
##### keep the table on its native index axis (1..5) and map a physical ρᶠ onto
##### that index with the same piecewise-linear transform the reference
##### implementation applies at runtime. This is the only thing that
##### distinguishes them from plain `TabulatedFunction4D`/`5D` objects.
#####

# Inverse of the {50, 250, 450, 650, 900} kg/m³ axis: 200 kg/m³ per index step up
# to index 4 at 650 (hence 1/200 = 0.005), then 250 kg/m³ per step (1/250 = 0.004).
@inline function rime_density_index(ρᶠ::FT) where FT
    return ifelse(ρᶠ ≤ FT(650),
                  (ρᶠ - FT(50))  * FT(0.005) + FT(1),
                  (ρᶠ - FT(650)) * FT(0.004) + FT(4))
end

struct RimeDensityIndexedTable4D{T}
    table :: T
end

@inline (f::RimeDensityIndexedTable4D)(log_m, Fᶠ, Fˡ, ρᶠ) =
    f.table(log_m, Fᶠ, Fˡ, rime_density_index(ρᶠ))

struct RimeDensityIndexedTable5D{T}
    table :: T
end

@inline (f::RimeDensityIndexedTable5D)(log_m, log_λʳ, Fᶠ, Fˡ, ρᶠ) =
    f.table(log_m, log_λʳ, Fᶠ, Fˡ, rime_density_index(ρᶠ))

# Union alias for dispatch: accept either bare or rime-density-indexed ice tables.
const P3Table4D = Union{TabulatedFunction4D, RimeDensityIndexedTable4D}

#####
##### Prepared interpolation indices
#####
##### Tables that share axes get evaluated at the same coordinate several times per
##### grid point: every P3 Table 1 entry is indexed by `(log_m, Fᶠ, Fˡ, ρᶠ)`, and both
##### rain-ice collection tables by `(log_m, log_λʳ, Fᶠ, Fˡ, ρᶠ)`. Bracketing the
##### coordinate once lets the related reads share the clamp, the fractional-index
##### arithmetic, and the boundary handling.
#####
##### The bracket is stored as the linear index of the lower corner plus the linear step to
##### the upper neighbour along each axis, rather than as per-axis indices. Every table read
##### at a bracket then costs only integer additions to locate its corners, instead of
##### re-deriving each corner's linear index from the table's runtime size: the compiler
##### cannot know that the tables sharing a bracket also share a shape, so it would repeat
##### that 64-bit index arithmetic for every table. The weights and their summation order
##### match `Oceananigans.Utils._interpolate`, so the interpolated values are unchanged.
#####
##### TODO: none of this knows anything about ice, and it duplicates the bodies of the
##### `TabulatedFunction` call operators. It belongs upstream in `Oceananigans.Utils`
##### beside `interpolator`/`_interpolate`, where a prepare/evaluate split would be
##### available to tables of any dimension.
#####

struct PreparedInterpolation{N, FT}
    "Linear index of the lower corner, counted from zero"
    offset :: Int32
    "Linear-index step from the lower to the upper neighbour along each axis; zero where the upper neighbour is clamped onto the lower one at the end of the axis"
    steps :: NTuple{N, Int32}
    "Fractional distance from the lower neighbour along each axis"
    fractions :: NTuple{N, FT}

    # The default constructor would leave `FT` unbound for `N == 0`, where `fractions` carries
    # no `FT`. Every call site supplies both parameters, so we define the inner constructor
    # and keep the ambiguous outer one from being generated.
    PreparedInterpolation{N, FT}(offset, steps, fractions) where {N, FT} =
        new{N, FT}(offset, steps, fractions)
end

# Column-major strides of an array of size `points`: (1, n₁, n₁n₂, ...).
@inline column_major_strides(::Tuple{Int}) = (1,)
@inline column_major_strides(points::Tuple) =
    (column_major_strides(Base.front(points))..., prod(Base.front(points)))

# A bracket is only valid for tables of the shape it was prepared on, the shape all tables
# sharing a set of axes have by construction.
@inline function prepare_interpolation(f::TabulatedFunction{N}, x::Vararg{Any, N}) where N
    points = size(f.table)
    strides = column_major_strides(points)
    brackets = ntuple(Val(N)) do d
        a, b = f.range[d]
        i⁻, i⁺, ξ = interpolator((clamp(x[d], a, b) - a) * f.inverse_Δ[d])
        (i⁻, min(i⁺, points[d] - 1), ξ)  # neighbours counted from zero
    end
    offset = sum(ntuple(d -> brackets[d][1] * strides[d], Val(N))) % Int32
    steps = ntuple(d -> ((brackets[d][2] - brackets[d][1]) * strides[d]) % Int32, Val(N))
    fractions = ntuple(d -> brackets[d][3], Val(N))
    return PreparedInterpolation{N, eltype(fractions)}(offset, steps, fractions)
end

# Quadrilinear interpolation of `data` about the one-based linear index `o`, with
# `_interpolate`'s weights summed in `_interpolate`'s order.
@inline function quadrilinear(data, o, (s₁, s₂, s₃, s₄), (ξ, η, ζ, θ))
    ϕ₁ = (1 - ξ) * (1 - η) * (1 - ζ)
    ϕ₂ = (1 - ξ) * (1 - η) *      ζ
    ϕ₃ = (1 - ξ) *      η  * (1 - ζ)
    ϕ₄ = (1 - ξ) *      η  *      ζ
    ϕ₅ =      ξ  * (1 - η) * (1 - ζ)
    ϕ₆ =      ξ  * (1 - η) *      ζ
    ϕ₇ =      ξ  *      η  * (1 - ζ)
    ϕ₈ =      ξ  *      η  *      ζ

    return @inbounds (
        ϕ₁ * (1 - θ) * data[o]                + ϕ₁ * θ * data[o + s₄] +
        ϕ₂ * (1 - θ) * data[o + s₃]           + ϕ₂ * θ * data[o + s₃ + s₄] +
        ϕ₃ * (1 - θ) * data[o + s₂]           + ϕ₃ * θ * data[o + s₂ + s₄] +
        ϕ₄ * (1 - θ) * data[o + s₂ + s₃]      + ϕ₄ * θ * data[o + s₂ + s₃ + s₄] +
        ϕ₅ * (1 - θ) * data[o + s₁]           + ϕ₅ * θ * data[o + s₁ + s₄] +
        ϕ₆ * (1 - θ) * data[o + s₁ + s₃]      + ϕ₆ * θ * data[o + s₁ + s₃ + s₄] +
        ϕ₇ * (1 - θ) * data[o + s₁ + s₂]      + ϕ₇ * θ * data[o + s₁ + s₂ + s₄] +
        ϕ₈ * (1 - θ) * data[o + s₁ + s₂ + s₃] + ϕ₈ * θ * data[o + s₁ + s₂ + s₃ + s₄])
end

@inline function evaluate_at(f::TabulatedFunction{4}, p::PreparedInterpolation{4})
    o = p.offset + Int32(1)
    return quadrilinear(f.table, o, p.steps, p.fractions)
end

# Two quadrilinear interpolations at the neighbouring fifth indices, as `_interpolate` does.
@inline function evaluate_at(f::TabulatedFunction{5}, p::PreparedInterpolation{5})
    o = p.offset + Int32(1)
    s₁, s₂, s₃, s₄, s₅ = p.steps
    ξ, η, ζ, θ, ψ = p.fractions
    return (1 - ψ) * quadrilinear(f.table, o,      (s₁, s₂, s₃, s₄), (ξ, η, ζ, θ)) +
                ψ  * quadrilinear(f.table, o + s₅, (s₁, s₂, s₃, s₄), (ξ, η, ζ, θ))
end

@inline prepare_interpolation(f::RimeDensityIndexedTable4D, log_m, Fᶠ, Fˡ, ρᶠ) =
    prepare_interpolation(f.table, log_m, Fᶠ, Fˡ, rime_density_index(ρᶠ))

@inline prepare_interpolation(f::RimeDensityIndexedTable5D, log_m, log_λʳ, Fᶠ, Fˡ, ρᶠ) =
    prepare_interpolation(f.table, log_m, log_λʳ, Fᶠ, Fˡ, rime_density_index(ρᶠ))

@inline evaluate_at(f::RimeDensityIndexedTable4D, p::PreparedInterpolation) =
    evaluate_at(f.table, p)

@inline evaluate_at(f::RimeDensityIndexedTable5D, p::PreparedInterpolation) =
    evaluate_at(f.table, p)
