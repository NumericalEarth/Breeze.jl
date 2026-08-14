using Oceananigans.Utils: TabulatedFunction,
                          TabulatedFunction1D,
                          TabulatedFunction4D,
                          TabulatedFunction5D,
                          interpolator,
                          _interpolate

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
##### TODO: none of this knows anything about ice, and it duplicates the bodies of the
##### `TabulatedFunction` call operators. It belongs upstream in `Oceananigans.Utils`
##### beside `interpolator`/`_interpolate`, where a prepare/evaluate split would be
##### available to tables of any dimension.
#####

struct PreparedInterpolation{N, FT}
    axes :: NTuple{N, Tuple{Int, Int, FT}}
end

@inline function prepare_interpolation(f::TabulatedFunction{N}, x::Vararg{Any, N}) where N
    points = size(f.table)
    axes = ntuple(Val(N)) do d
        a, b = f.range[d]
        i⁻, i⁺, ξ = interpolator((clamp(x[d], a, b) - a) * f.inverse_Δ[d])
        (i⁻ + 1, min(i⁺ + 1, points[d]), ξ)
    end
    return PreparedInterpolation{N, typeof(axes[1][3])}(axes)
end

@inline prepare_interpolation(f::RimeDensityIndexedTable4D, log_m, Fᶠ, Fˡ, ρᶠ) =
    prepare_interpolation(f.table, log_m, Fᶠ, Fˡ, rime_density_index(ρᶠ))

@inline prepare_interpolation(f::RimeDensityIndexedTable5D, log_m, log_λʳ, Fᶠ, Fˡ, ρᶠ) =
    prepare_interpolation(f.table, log_m, log_λʳ, Fᶠ, Fˡ, rime_density_index(ρᶠ))

@inline evaluate_at(f::TabulatedFunction, p::PreparedInterpolation) =
    _interpolate(f.table, p.axes...)

@inline evaluate_at(f::RimeDensityIndexedTable4D, p::PreparedInterpolation) =
    evaluate_at(f.table, p)

@inline evaluate_at(f::RimeDensityIndexedTable5D, p::PreparedInterpolation) =
    evaluate_at(f.table, p)
