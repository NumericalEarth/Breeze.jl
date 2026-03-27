using Oceananigans.Architectures: CPU, on_architecture
import Oceananigans.Utils: TabulatedFunction,
                           TabulatedFunction1D,
                           TabulatedFunction2D,
                           TabulatedFunction3D,
                           TabulatedFunction4D,
                           TabulatedFunction5D

const P3TabulatedFunction = TabulatedFunction

@inline table_range(x_min, x_max) = (x_min, x_max)
@inline table_range(x_range, y_range, z_range) = (x_range, y_range, z_range)
@inline table_range(x_range, y_range, z_range, w_range) = (x_range, y_range, z_range, w_range)
@inline table_range(x_range, y_range, z_range, w_range, v_range) = (x_range, y_range, z_range, w_range, v_range)

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

function build_cpu_table(func, FT, range::Tuple{<:Number, <:Number}, points::Integer)
    table = zeros(FT, points)
    x_min, x_max = range

    for i in 1:points
        x = axis_coordinate(x_min, x_max, points, i, FT)
        table[i] = func(x)
    end

    return table
end

function build_cpu_table(func, FT,
                         range::NTuple{3, <:Tuple{<:Number, <:Number}},
                         points::NTuple{3, <:Integer})
    x_range, y_range, z_range = range
    x_points, y_points, z_points = points
    table = zeros(FT, x_points, y_points, z_points)
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    for k in 1:z_points
        z = axis_coordinate(z_min, z_max, z_points, k, FT)
        for j in 1:y_points
            y = axis_coordinate(y_min, y_max, y_points, j, FT)
            for i in 1:x_points
                x = axis_coordinate(x_min, x_max, x_points, i, FT)
                table[i, j, k] = func(x, y, z)
            end
        end
    end

    return table
end

function build_cpu_table(func, FT,
                         range::NTuple{4, <:Tuple{<:Number, <:Number}},
                         points::NTuple{4, <:Integer})
    x_range, y_range, z_range, w_range = range
    x_points, y_points, z_points, w_points = points
    table = zeros(FT, x_points, y_points, z_points, w_points)
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    w_min, w_max = w_range

    for l in 1:w_points
        w = axis_coordinate(w_min, w_max, w_points, l, FT)
        for k in 1:z_points
            z = axis_coordinate(z_min, z_max, z_points, k, FT)
            for j in 1:y_points
                y = axis_coordinate(y_min, y_max, y_points, j, FT)
                for i in 1:x_points
                    x = axis_coordinate(x_min, x_max, x_points, i, FT)
                    table[i, j, k, l] = func(x, y, z, w)
                end
            end
        end
    end

    return table
end

function build_cpu_table(func, FT,
                         range::NTuple{5, <:Tuple{<:Number, <:Number}},
                         points::NTuple{5, <:Integer})
    x_range, y_range, z_range, w_range, v_range = range
    x_points, y_points, z_points, w_points, v_points = points
    table = zeros(FT, x_points, y_points, z_points, w_points, v_points)
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    w_min, w_max = w_range
    v_min, v_max = v_range

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
                        table[i, j, k, l, m] = func(x, y, z, w, v)
                    end
                end
            end
        end
    end

    return table
end

@inline function build_tabulated_function(func, arch, FT, range::Tuple{<:Number, <:Number}, points::Integer)
    range_tuple = (range,)
    cpu_table = build_cpu_table(func, FT, range, points)
    table = on_architecture(arch, cpu_table)
    inverse_Δ = (inverse_spacing(range[1], range[2], points, FT),)

    return TabulatedFunction{1, typeof(func), typeof(table), typeof(range_tuple), typeof(inverse_Δ)}(
        func, table, range_tuple, inverse_Δ)
end

@inline function build_tabulated_function(func, arch, FT,
                                          range::NTuple{3, <:Tuple{<:Number, <:Number}},
                                          points::NTuple{3, <:Integer})
    cpu_table = build_cpu_table(func, FT, range, points)
    table = on_architecture(arch, cpu_table)
    inverse_Δ = map(range, points) do axis_range, axis_points
        inverse_spacing(axis_range[1], axis_range[2], axis_points, FT)
    end

    return TabulatedFunction{3, typeof(func), typeof(table), typeof(range), typeof(inverse_Δ)}(
        func, table, range, inverse_Δ)
end

@inline function build_tabulated_function(func, arch, FT,
                                          range::NTuple{4, <:Tuple{<:Number, <:Number}},
                                          points::NTuple{4, <:Integer})
    cpu_table = build_cpu_table(func, FT, range, points)
    table = on_architecture(arch, cpu_table)
    inverse_Δ = map(range, points) do axis_range, axis_points
        inverse_spacing(axis_range[1], axis_range[2], axis_points, FT)
    end

    return TabulatedFunction{4, typeof(func), typeof(table), typeof(range), typeof(inverse_Δ)}(
        func, table, range, inverse_Δ)
end

@inline function build_tabulated_function(func, arch, FT,
                                          range::NTuple{5, <:Tuple{<:Number, <:Number}},
                                          points::NTuple{5, <:Integer})
    cpu_table = build_cpu_table(func, FT, range, points)
    table = on_architecture(arch, cpu_table)
    inverse_Δ = map(range, points) do axis_range, axis_points
        inverse_spacing(axis_range[1], axis_range[2], axis_points, FT)
    end

    return TabulatedFunction{5, typeof(func), typeof(table), typeof(range), typeof(inverse_Δ)}(
        func, table, range, inverse_Δ)
end

@inline function TabulatedFunction1D(func, arch=CPU(), FT=Float64; x_range, x_points=200)
    return build_tabulated_function(func, arch, FT, x_range, x_points)
end

@inline function TabulatedFunction3D(func, arch=CPU(), FT=Float64;
                                      x_range,
                                      y_range = (FT(0), FT(1)),
                                      z_range = (FT(0), FT(1)),
                                      x_points = 50,
                                      y_points = 4,
    z_points = 4)
    return build_tabulated_function(func, arch, FT,
                                    table_range(x_range, y_range, z_range),
                                    (x_points, y_points, z_points))
end

@inline function TabulatedFunction4D(func, arch=CPU(), FT=Float64;
                                      x_range,
                                      y_range = (FT(0), FT(1)),
                                      z_range = (FT(0), FT(1)),
                                      w_range = (FT(50), FT(900)),
                                      x_points = 50,
                                      y_points = 4,
    z_points = 4,
    w_points = 5)
    return build_tabulated_function(func, arch, FT,
                                    table_range(x_range, y_range, z_range, w_range),
                                    (x_points, y_points, z_points, w_points))
end

@inline function TabulatedFunction5D(func, arch=CPU(), FT=Float64;
                                      x_range,
                                      y_range = (FT(0), FT(1)),
                                      z_range = (FT(0), FT(1)),
                                      w_range = (FT(0), FT(1)),
                                      v_range = (FT(50), FT(900)),
                                      x_points = 50,
                                      y_points = 4,
                                      z_points = 4,
                                      w_points = 4,
                                      v_points = 5)
    return build_tabulated_function(func, arch, FT,
                                    table_range(x_range, y_range, z_range, w_range, v_range),
                                    (x_points, y_points, z_points, w_points, v_points))
end

@inline function tabulated_function_1d(values::AbstractVector, x_min, x_max, inverse_Δx)
    FT = eltype(values)
    range = ((FT(x_min), FT(x_max)),)
    inverse_Δ = (FT(inverse_Δx),)

    return TabulatedFunction{1, Nothing, typeof(values), typeof(range), typeof(inverse_Δ)}(
        nothing, values, range, inverse_Δ)
end
