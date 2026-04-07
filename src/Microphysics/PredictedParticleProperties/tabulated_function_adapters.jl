using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Utils: TabulatedFunction,
                          TabulatedFunction1D,
                          TabulatedFunction2D,
                          TabulatedFunction3D,
                          TabulatedFunction4D,
                          TabulatedFunction5D,
                          interpolator

const TabulatedFunction6D = TabulatedFunction{6}

# Trivial tuple constructor used by lookup_table_2.jl (6D) and lookup_table_3.jl (5D)
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

    return TabulatedFunction{6, typeof(func), typeof(table), typeof(range), typeof(inverse_Δ)}(
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

@inline function (f::TabulatedFunction{6})(x₁, x₂, x₃, x₄, x₅, x₆)
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
    i⁻ = i⁻ + 1
    i⁺ = min(i⁺ + 1, n₁)
    j⁻ = j⁻ + 1
    j⁺ = min(j⁺ + 1, n₂)
    k⁻ = k⁻ + 1
    k⁺ = min(k⁺ + 1, n₃)
    l⁻ = l⁻ + 1
    l⁺ = min(l⁺ + 1, n₄)
    m⁻ = m⁻ + 1
    m⁺ = min(m⁺ + 1, n₅)
    n⁻ = n⁻ + 1
    n⁺ = min(n⁺ + 1, n₆)

    return _interpolate_6d(f.table,
                           (i⁻, i⁺, ξ), (j⁻, j⁺, η), (k⁻, k⁺, ζ),
                           (l⁻, l⁺, θ), (m⁻, m⁺, ψ), (n⁻, n⁺, χ))
end

@inline function _interpolate_6d(data, ix, iy, iz, iw, iv, iu)
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
