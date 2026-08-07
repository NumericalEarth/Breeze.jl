using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Utils: TabulatedFunction,
                          TabulatedFunction1D,
                          TabulatedFunction2D,
                          TabulatedFunction3D,
                          TabulatedFunction4D,
                          TabulatedFunction5D,
                          interpolator,
                          _interpolate

# Oceananigans' TabulatedFunction only supports 1D–5D, so we own the 6D variant
# outright (defining methods on TabulatedFunction{6} would be type piracy).
#
# TODO: `TabulatedFunction6D` and the `prepare_*`/`evaluate_at` split below are
# scheme-agnostic multilinear-lookup machinery, not P3 physics — nothing here
# knows about ice. They live in this module only because P3 is the first consumer.
# They belong upstream in `Oceananigans.Utils` alongside `TabulatedFunction1D`–`5D`
# (which would also let the 6D case reuse the per-dimension generation there
# instead of forking it). Move them when a second consumer appears, rather than
# letting another module reach into `PredictedParticleProperties` for them.
struct TabulatedFunction6D{F, T, R, D}
    func :: F
    table :: T
    range :: R
    inverse_Δ :: D
end

# Trivial tuple constructor used by fortran_table_format.jl (6D) and lookup_table_3.jl (5D)
@inline table_range(ranges...) = ranges

#####
##### 6D table building — Oceananigans only supports up to 5D
#####

@inline function axis_coordinate(minimum, maximum, points, index, FT)
    minimum = FT(minimum)
    if points == 1
        return minimum
    end

    Δ = (FT(maximum) - minimum) / (points - 1)
    return minimum + (index - 1) * Δ
end

@inline function inverse_spacing(minimum, maximum, points, FT)
    if points == 1
        return zero(FT)
    end

    Δ = (FT(maximum) - FT(minimum)) / (points - 1)
    return 1 / Δ
end

function build_cpu_table(func, FT,
                         range::NTuple{6, <:Tuple{<:Number, <:Number}},
                         points::NTuple{6, <:Integer})
    x_range, y_range, z_range, w_range, v_range, u_range = range
    x_points, y_points, z_points, w_points, v_points, u_points = points
    table = zeros(FT, x_points, y_points, z_points, w_points, v_points, u_points)
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    w_min, w_max = w_range
    v_min, v_max = v_range
    u_min, u_max = u_range

    for n in 1:u_points
        u = axis_coordinate(u_min, u_max, u_points, n, FT)
        for m in 1:v_points
            v = axis_coordinate(v_min, v_max, v_points, m, FT)
            for l in 1:w_points
                w = axis_coordinate(w_min, w_max, w_points, l, FT)
                for k in 1:z_points
                    z = axis_coordinate(z_min, z_max, z_points, k, FT)
                    for j in 1:y_points
                        y = axis_coordinate(y_min, y_max, y_points, j, FT)
                        for i in 1:x_points
                            x = axis_coordinate(x_min, x_max, x_points, i, FT)
                            table[i, j, k, l, m, n] = func(x, y, z, w, v, u)
                        end
                    end
                end
            end
        end
    end

    return table
end

@inline function build_tabulated_function(func, arch, FT,
                                          range::NTuple{6, <:Tuple{<:Number, <:Number}},
                                          points::NTuple{6, <:Integer})
    cpu_table = build_cpu_table(func, FT, range, points)
    table = on_architecture(arch, cpu_table)
    inverse_Δ = map(range, points) do axis_range, axis_points
        inverse_spacing(axis_range[1], axis_range[2], axis_points, FT)
    end

    return TabulatedFunction6D{typeof(func), typeof(table), typeof(range), typeof(inverse_Δ)}(
        func, table, range, inverse_Δ)
end

@inline function TabulatedFunction6D(func, arch=CPU(), FT=Float64;
                                      x_range,
                                      y_range = (FT(0), FT(1)),
                                      z_range = (FT(0), FT(1)),
                                      w_range = (FT(0), FT(1)),
                                      v_range = (FT(0), FT(1)),
                                      u_range = (FT(0), FT(20)),
                                      x_points = 50,
                                      y_points = 4,
                                      z_points = 4,
                                      w_points = 4,
                                      v_points = 4,
                                      u_points = 11)
    return build_tabulated_function(func, arch, FT,
                                    table_range(x_range, y_range, z_range, w_range, v_range, u_range),
                                    (x_points, y_points, z_points, w_points, v_points, u_points))
end

#####
##### 6D interpolation
#####

# The clamp-and-bracket work lives in `prepare_6d`, which returns the same
# six index triplets this call operator needs; `evaluate_at` then does the
# multilinear blend. Sharing them keeps one definition of the axis bounds.
@inline (f::TabulatedFunction6D)(x₁, x₂, x₃, x₄, x₅, x₆) =
    evaluate_at(f, prepare_6d(f, x₁, x₂, x₃, x₄, x₅, x₆))

@inline function interpolate_6d(data, ix, iy, iz, iw, iv, iu)
    i⁻, i⁺, ξ = ix
    j⁻, j⁺, η = iy
    k⁻, k⁺, ζ = iz
    l⁻, l⁺, θ = iw
    m⁻, m⁺, ψ = iv
    n⁻, n⁺, χ = iu

    result = zero(eltype(data))
    @inbounds for (ni, nw) in ((n⁻, 1 - χ), (n⁺, χ))
        for (mi, mw) in ((m⁻, 1 - ψ), (m⁺, ψ))
            for (li, lw) in ((l⁻, 1 - θ), (l⁺, θ))
                for (ki, kw) in ((k⁻, 1 - ζ), (k⁺, ζ))
                    for (ji, jw) in ((j⁻, 1 - η), (j⁺, η))
                        for (ii, iw_) in ((i⁻, 1 - ξ), (i⁺, ξ))
                            result += iw_ * jw * kw * lw * mw * nw * data[ii, ji, ki, li, mi, ni]
                        end
                    end
                end
            end
        end
    end
    return result
end

#####
##### Helper: construct TabulatedFunction{1} from raw data (used by JLD2 loading)
#####

@inline function tabulated_function_1d(values::AbstractVector, x_min, x_max, inverse_Δx)
    FT = eltype(values)
    range = ((FT(x_min), FT(x_max)),)
    inverse_Δ = (FT(inverse_Δx),)

    return TabulatedFunction{1, Nothing, typeof(values), typeof(range), typeof(inverse_Δ)}(
        nothing, values, range, inverse_Δ)
end

#####
##### Fortran P3 table wrappers with rime-density coordinate transform
#####
##### The Fortran lookup tables use a non-uniform rime density grid
##### {50, 250, 450, 650, 900} kg/m³. These wrappers store tables in
##### Fortran-index-space (1..5) and apply the piecewise-linear
##### coordinate transform that the Fortran runtime uses.
#####

@inline function rime_density_index(ρᶠ::FT) where FT
    return ifelse(ρᶠ ≤ FT(650),
                  (ρᶠ - FT(50))  * FT(0.005) + FT(1),
                  (ρᶠ - FT(650)) * FT(0.004) + FT(4))
end

struct FortranTabulatedFunction5D{T}
    table :: T
end

@inline function (f::FortranTabulatedFunction5D)(log_m, Fᶠ, Fˡ, ρᶠ, μ)
    return f.table(log_m, Fᶠ, Fˡ, rime_density_index(ρᶠ), μ)
end

struct FortranTabulatedFunction6D{T}
    table :: T
end

@inline function (f::FortranTabulatedFunction6D)(log_m, log_λ_r, Fᶠ, Fˡ, ρᶠ, μ)
    return f.table(log_m, log_λ_r, Fᶠ, Fˡ, rime_density_index(ρᶠ), μ)
end

# Table 3 has rime density as its 2nd axis: (log_znorm, rho_index, log_qnorm, Fr, Fl)
struct FortranTabulatedFunction3{T}
    table :: T
end

@inline function (f::FortranTabulatedFunction3)(log_z, ρᶠ, log_q, Fᶠ, Fˡ)
    return f.table(log_z, rime_density_index(ρᶠ), log_q, Fᶠ, Fˡ)
end

# Union aliases for dispatch: accept either Julia-generated or Fortran-loaded tables
const P3Table5D = Union{TabulatedFunction5D, FortranTabulatedFunction5D}
const P3Table6D = Union{TabulatedFunction6D, FortranTabulatedFunction6D}

#####
##### Prepared 5D interpolation indices
#####
##### When several distinct 5D tables are queried at the *same* coordinates
##### (a common pattern in P3 — see `tabulated_z_tendency` where ~16 tables share
##### `(log_m, Fᶠ, Fˡ, ρᶠ, μ)`), the per-axis clamps, fractional-index multiplies,
##### `interpolator` calls, and boundary-min checks are redundantly recomputed for
##### each table. Prepare them once and reuse across tables that share `range`,
##### `inverse_Δ`, and shape.
##### All P3 Fortran Table 1 entries share the same axes by construction, so
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

@inline function prepare_5d(f::FortranTabulatedFunction5D, log_m, Fᶠ, Fˡ, ρᶠ, μ)
    return prepare_5d(f.table, log_m, Fᶠ, Fˡ, rime_density_index(ρᶠ), μ)
end

@inline evaluate_at(f::TabulatedFunction5D, p::Prepared5DInterpolation) =
    _interpolate(f.table, p.ix, p.iy, p.iz, p.iw, p.iv)

@inline evaluate_at(f::FortranTabulatedFunction5D, p::Prepared5DInterpolation) =
    evaluate_at(f.table, p)

#####
##### Prepared 6D interpolation indices
#####
##### Mirrors `Prepared5DInterpolation` for the 6-D ice-rain collection tables
##### (`sixth_moment`, `mass`, `number`). All three are queried at identical
##### `(log_m, log_λ_r, Fᶠ, Fˡ, ρᶠ, μ)` per cell, so prepping once and
##### reusing across the trio eliminates redundant clamps / frac-index work.
##### All P3 Fortran Table 2 entries share the same axes by construction, so
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

@inline function prepare_6d(f::FortranTabulatedFunction6D, log_m, log_λ_r, Fᶠ, Fˡ, ρᶠ, μ)
    return prepare_6d(f.table, log_m, log_λ_r, Fᶠ, Fˡ, rime_density_index(ρᶠ), μ)
end

@inline evaluate_at(f::TabulatedFunction6D, p::Prepared6DInterpolation) =
    interpolate_6d(f.table, p.ix, p.iy, p.iz, p.iw, p.iv, p.iu)

@inline evaluate_at(f::FortranTabulatedFunction6D, p::Prepared6DInterpolation) =
    evaluate_at(f.table, p)
