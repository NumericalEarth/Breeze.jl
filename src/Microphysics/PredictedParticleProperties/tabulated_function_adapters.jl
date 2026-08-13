using Oceananigans.Utils: TabulatedFunction1D,
                          TabulatedFunction5D,
                          interpolator,
                          _interpolate

# Oceananigans' TabulatedFunction only supports 1D–5D, so we own the 6D variant
# outright (defining methods on TabulatedFunction{6} would be type piracy).
#
# Unlike Oceananigans' `TabulatedFunction`, this type carries no generating
# function: every 6D table in P3 is read from the ASCII lookup files by
# `make_lookup_table`, never evaluated onto a grid here.
#
# TODO: `TabulatedFunction6D` and the `prepare_*`/`evaluate_at` split below are
# scheme-agnostic multilinear-lookup machinery, not P3 physics — nothing here
# knows about ice. They live in this module only because P3 is the first consumer.
# They belong upstream in `Oceananigans.Utils` alongside `TabulatedFunction1D`–`5D`.
# Move them when a second consumer appears, rather than letting another module
# reach into `PredictedParticleProperties` for them.
struct TabulatedFunction6D{T, R, D}
    table :: T
    range :: R
    inverse_Δ :: D
end

#####
##### 6D interpolation
#####

# The clamp-and-bracket work lives in `prepare_6d`, which returns the same
# six index triplets this call operator needs; `evaluate_at` then does the
# multilinear blend. Sharing them keeps one definition of the axis bounds.
@inline (f::TabulatedFunction6D)(x₁, x₂, x₃, x₄, x₅, x₆) =
    evaluate_at(f, prepare_6d(f, x₁, x₂, x₃, x₄, x₅, x₆))

# 32-corner blend at a fixed sixth (μⁱ) index.
@inline function interpolate_6d_slice(data, ix, iy, iz, iw, iv, ni)
    i⁻, i⁺, ξ = ix
    j⁻, j⁺, η = iy
    k⁻, k⁺, ζ = iz
    l⁻, l⁺, θ = iw
    m⁻, m⁺, ψ = iv

    result = zero(eltype(data))
    @inbounds for (mi, mw) in ((m⁻, 1 - ψ), (m⁺, ψ))
        for (li, lw) in ((l⁻, 1 - θ), (l⁺, θ))
            for (ki, kw) in ((k⁻, 1 - ζ), (k⁺, ζ))
                for (ji, jw) in ((j⁻, 1 - η), (j⁺, η))
                    for (ii, iw_) in ((i⁻, 1 - ξ), (i⁺, ξ))
                        result += iw_ * jw * kw * lw * mw * data[ii, ji, ki, li, mi, ni]
                    end
                end
            end
        end
    end
    return result
end

# Same collapse as `collapse_trailing_axis`, on the 6D tables' μⁱ axis: `n⁻ == n⁺`
# means one 32-corner pass replaces two.
@inline function interpolate_6d(data, ix, iy, iz, iw, iv, iu)
    n⁻, n⁺, χ = iu
    n⁻ == n⁺ && return interpolate_6d_slice(data, ix, iy, iz, iw, iv, n⁻)
    return (1 - χ) * interpolate_6d_slice(data, ix, iy, iz, iw, iv, n⁻) +
                χ  * interpolate_6d_slice(data, ix, iy, iz, iw, iv, n⁺)
end

#####
##### Table wrappers carrying an indexed rime-density axis
#####
##### The ASCII lookup tables sample rime density on the non-uniform grid
##### {50, 250, 450, 650, 900} kg/m³. Rather than resample, the wrappers below
##### keep the table on its native index axis (1..5) and map a physical ρᶠ onto
##### that index with the same piecewise-linear transform the reference
##### implementation applies at runtime. This is the only thing that
##### distinguishes them from a plain `TabulatedFunction5D`/`6D`.
#####

# Inverse of the {50, 250, 450, 650, 900} kg/m³ axis: 200 kg/m³ per index step up
# to index 4 at 650 (hence 1/200 = 0.005), then 250 kg/m³ per step (1/250 = 0.004).
@inline function rime_density_index(ρᶠ::FT) where FT
    return ifelse(ρᶠ ≤ FT(650),
                  (ρᶠ - FT(50))  * FT(0.005) + FT(1),
                  (ρᶠ - FT(650)) * FT(0.004) + FT(4))
end

struct RimeDensityIndexedTable5D{T}
    table :: T
end

# Routed through `prepare_5d`/`evaluate_at` so this path gets `collapse_trailing_axis`;
# `f.table(...)` would reach Oceananigans' operator, which we cannot extend without piracy.
@inline function (f::RimeDensityIndexedTable5D)(log_m, Fᶠ, Fˡ, ρᶠ, μⁱ)
    return evaluate_at(f, prepare_5d(f, log_m, Fᶠ, Fˡ, ρᶠ, μⁱ))
end

struct RimeDensityIndexedTable6D{T}
    table :: T
end

@inline function (f::RimeDensityIndexedTable6D)(log_m, log_λʳ, Fᶠ, Fˡ, ρᶠ, μⁱ)
    return f.table(log_m, log_λʳ, Fᶠ, Fˡ, rime_density_index(ρᶠ), μⁱ)
end

# Union alias for dispatch: accept either a bare or a rime-density-indexed 5D table.
# The 6D tables are only ever called directly, so they need no such alias.
const P3Table5D = Union{TabulatedFunction5D, RimeDensityIndexedTable5D}

#####
##### Prepared 5D interpolation indices
#####
##### When several distinct 5D tables are queried at the *same* coordinates
##### (a common pattern in P3 — see `tabulated_z_tendency` where ~16 tables share
##### `(log_m, Fᶠ, Fˡ, ρᶠ, μⁱ)`), the per-axis clamps, fractional-index multiplies,
##### `interpolator` calls, and boundary-min checks are redundantly recomputed for
##### each table. Prepare them once and reuse across tables that share `range`,
##### `inverse_Δ`, and shape.
##### All P3 Table 1 entries share the same axes by construction, so
##### a single `Prepared5DInterpolation` is valid for any of them.
#####

struct Prepared5DInterpolation{FT}
    ix :: Tuple{Int, Int, FT}
    iy :: Tuple{Int, Int, FT}
    iz :: Tuple{Int, Int, FT}
    iw :: Tuple{Int, Int, FT}
    iv :: Tuple{Int, Int, FT}
end

@inline function prepare_5d(f::TabulatedFunction5D, x₁, x₂, x₃, x₄, x₅)
    a₁, b₁ = f.range[1]
    a₂, b₂ = f.range[2]
    a₃, b₃ = f.range[3]
    a₄, b₄ = f.range[4]
    a₅, b₅ = f.range[5]

    c₁ = clamp(x₁, a₁, b₁)
    c₂ = clamp(x₂, a₂, b₂)
    c₃ = clamp(x₃, a₃, b₃)
    c₄ = clamp(x₄, a₄, b₄)
    c₅ = clamp(x₅, a₅, b₅)

    frac_i = (c₁ - a₁) * f.inverse_Δ[1]
    frac_j = (c₂ - a₂) * f.inverse_Δ[2]
    frac_k = (c₃ - a₃) * f.inverse_Δ[3]
    frac_l = (c₄ - a₄) * f.inverse_Δ[4]
    frac_m = (c₅ - a₅) * f.inverse_Δ[5]

    i⁻, i⁺, ξ = interpolator(frac_i)
    j⁻, j⁺, η = interpolator(frac_j)
    k⁻, k⁺, ζ = interpolator(frac_k)
    l⁻, l⁺, θ = interpolator(frac_l)
    m⁻, m⁺, ψ = interpolator(frac_m)

    n₁, n₂, n₃, n₄, n₅ = size(f.table)

    return Prepared5DInterpolation{typeof(ξ)}((i⁻ + 1, min(i⁺ + 1, n₁), ξ),
                                              (j⁻ + 1, min(j⁺ + 1, n₂), η),
                                              (k⁻ + 1, min(k⁺ + 1, n₃), ζ),
                                              (l⁻ + 1, min(l⁺ + 1, n₄), θ),
                                              (m⁻ + 1, min(m⁺ + 1, n₅), ψ))
end

@inline function prepare_5d(f::RimeDensityIndexedTable5D, log_m, Fᶠ, Fˡ, ρᶠ, μⁱ)
    return prepare_5d(f.table, log_m, Fᶠ, Fˡ, rime_density_index(ρᶠ), μⁱ)
end

# Collapse the trailing axis when its bracket is degenerate: `m⁻ == m⁺` means both
# slices read the same data, so `(1-ψ) A + ψ B` is the identity and one slice will
# do. The 2-moment tables carry a single μⁱ point, so this fires at every lookup; a
# 3-moment μⁱ axis takes the blended path unchanged. The branch is on table geometry,
# not cell data, so it is uniform across a launch — and `ifelse` would evaluate both
# slices, which is the cost being removed.
@inline function collapse_trailing_axis(data, ix, iy, iz, iw, iv)
    m⁻, m⁺, ψ = iv
    m⁻ == m⁺ && return _interpolate(data, ix, iy, iz, iw, m⁻)
    return _interpolate(data, ix, iy, iz, iw, iv)
end

@inline evaluate_at(f::TabulatedFunction5D, p::Prepared5DInterpolation) =
    collapse_trailing_axis(f.table, p.ix, p.iy, p.iz, p.iw, p.iv)

@inline evaluate_at(f::RimeDensityIndexedTable5D, p::Prepared5DInterpolation) =
    evaluate_at(f.table, p)

#####
##### Prepared 6D interpolation indices
#####
##### Mirrors `Prepared5DInterpolation` for the 6-D ice-rain collection tables
##### (`mass`, `number`). Both are queried at identical
##### `(log_m, log_λʳ, Fᶠ, Fˡ, ρᶠ, μⁱ)` per cell, so prepping once and
##### reusing across the pair eliminates redundant clamps / frac-index work.
##### All P3 Table 2 entries share the same axes by construction, so
##### a single `Prepared6DInterpolation` is valid for any of them.
#####

struct Prepared6DInterpolation{FT}
    ix :: Tuple{Int, Int, FT}
    iy :: Tuple{Int, Int, FT}
    iz :: Tuple{Int, Int, FT}
    iw :: Tuple{Int, Int, FT}
    iv :: Tuple{Int, Int, FT}
    iu :: Tuple{Int, Int, FT}
end

@inline function prepare_6d(f::TabulatedFunction6D, x₁, x₂, x₃, x₄, x₅, x₆)
    a₁, b₁ = f.range[1]
    a₂, b₂ = f.range[2]
    a₃, b₃ = f.range[3]
    a₄, b₄ = f.range[4]
    a₅, b₅ = f.range[5]
    a₆, b₆ = f.range[6]

    c₁ = clamp(x₁, a₁, b₁)
    c₂ = clamp(x₂, a₂, b₂)
    c₃ = clamp(x₃, a₃, b₃)
    c₄ = clamp(x₄, a₄, b₄)
    c₅ = clamp(x₅, a₅, b₅)
    c₆ = clamp(x₆, a₆, b₆)

    frac_i = (c₁ - a₁) * f.inverse_Δ[1]
    frac_j = (c₂ - a₂) * f.inverse_Δ[2]
    frac_k = (c₃ - a₃) * f.inverse_Δ[3]
    frac_l = (c₄ - a₄) * f.inverse_Δ[4]
    frac_m = (c₅ - a₅) * f.inverse_Δ[5]
    frac_n = (c₆ - a₆) * f.inverse_Δ[6]

    i⁻, i⁺, ξ = interpolator(frac_i)
    j⁻, j⁺, η = interpolator(frac_j)
    k⁻, k⁺, ζ = interpolator(frac_k)
    l⁻, l⁺, θ = interpolator(frac_l)
    m⁻, m⁺, ψ = interpolator(frac_m)
    n⁻, n⁺, χ = interpolator(frac_n)

    n₁, n₂, n₃, n₄, n₅, n₆ = size(f.table)

    return Prepared6DInterpolation{typeof(ξ)}((i⁻ + 1, min(i⁺ + 1, n₁), ξ),
                                              (j⁻ + 1, min(j⁺ + 1, n₂), η),
                                              (k⁻ + 1, min(k⁺ + 1, n₃), ζ),
                                              (l⁻ + 1, min(l⁺ + 1, n₄), θ),
                                              (m⁻ + 1, min(m⁺ + 1, n₅), ψ),
                                              (n⁻ + 1, min(n⁺ + 1, n₆), χ))
end

@inline function prepare_6d(f::RimeDensityIndexedTable6D, log_m, log_λʳ, Fᶠ, Fˡ, ρᶠ, μⁱ)
    return prepare_6d(f.table, log_m, log_λʳ, Fᶠ, Fˡ, rime_density_index(ρᶠ), μⁱ)
end

@inline evaluate_at(f::TabulatedFunction6D, p::Prepared6DInterpolation) =
    interpolate_6d(f.table, p.ix, p.iy, p.iz, p.iw, p.iv, p.iu)

@inline evaluate_at(f::RimeDensityIndexedTable6D, p::Prepared6DInterpolation) =
    evaluate_at(f.table, p)
