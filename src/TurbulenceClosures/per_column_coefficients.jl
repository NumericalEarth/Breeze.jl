"""
$(TYPEDEF)

The values of one scalar of a closure in each column of a column ensemble, an `(Nx, Ny)` array.
Built by [`per_column_closure`](@ref); a kernel reads the value of column `(i, j)` through
`getclosure(i, j, closure)`.
"""
struct PerColumnCoefficient{A}
    values :: A
end

Adapt.@adapt_structure PerColumnCoefficient

Base.summary(c::PerColumnCoefficient) = string("PerColumnCoefficient of ", summary(c.values))
Base.show(io::IO, c::PerColumnCoefficient) = print(io, summary(c))

struct ColumnSelection{I, J}
    i :: I
    j :: J
end

@inline Adapt.adapt_structure(column::ColumnSelection, c::PerColumnCoefficient) = @inbounds c.values[column.i, column.j]

@inline column_closure(i, j, closure) = adapt(ColumnSelection(i, j), closure)

@inline Oceananigans.TurbulenceClosures.getclosure(i, j, closure::TKEBasedTurbulenceClosure) = column_closure(i, j, closure)

#####
##### Construction from an array of closures, and the in-place update
#####

struct ScalarCollection{V}
    scalars :: V
end

Adapt.adapt_storage(collection::ScalarCollection, x::AbstractFloat) = (push!(collection.scalars, x); x)

struct ScalarReplacement{V}
    arrays :: V
    count :: Base.RefValue{Int}
end

function Adapt.adapt_storage(replacement::ScalarReplacement, x::AbstractFloat)
    n = replacement.count[] += 1
    return PerColumnCoefficient(replacement.arrays[n])
end

struct PerColumnCoefficientCollection{V}
    coefficients :: V
end

function Adapt.adapt_structure(collection::PerColumnCoefficientCollection, c::PerColumnCoefficient)
    push!(collection.coefficients, c)
    return c
end

function closure_scalars_in_adapt_order(closure)
    scalars = AbstractFloat[]
    adapt(ScalarCollection(scalars), closure)
    return scalars
end

function per_column_coefficients(closure)
    coefficients = PerColumnCoefficient[]
    adapt(PerColumnCoefficientCollection(coefficients), closure)
    return coefficients
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return one closure that holds, in each of its floating-point scalars, the value of that scalar in
each closure of `closures`, an `(Nx, Ny)` array of closures of one type for the `Nx × Ny` columns
of the column ensemble `grid`. Each scalar becomes a [`PerColumnCoefficient`](@ref) on the
architecture of `grid`, and column `(i, j)` of a model with this closure integrates `closures[i, j]`.
"""
function per_column_closure(grid, closures::AbstractMatrix)
    Nx, Ny = size(grid, 1), size(grid, 2)
    size(closures) == (Nx, Ny) ||
        throw(ArgumentError("per_column_closure: $(size(closures)) closures for a grid of $Nx × $Ny columns"))
    allequal(map(typeof, closures)) ||
        throw(ArgumentError("per_column_closure: the closures must have one type, but they have $(unique(map(typeof, closures)))"))

    scalars = map(closure_scalars_in_adapt_order, closures)
    template = first(closures)
    arrays = [map(column_scalars -> column_scalars[n], scalars) for n in eachindex(first(scalars))]
    arrays = map(values -> convert(Matrix{typeof(first(values))}, values), arrays)
    closure = adapt(ScalarReplacement(arrays, Ref(0)), template)

    for j in 1:Ny, i in 1:Nx
        column_closure(i, j, closure) === closures[i, j] ||
            throw(ArgumentError("per_column_closure: column ($i, $j) does not recover its closure; " *
                                "the closures differ in something other than their floating-point scalars"))
    end

    arch = Oceananigans.Architectures.architecture(grid)
    return adapt(ArchitectureTransfer(arch), closure)
end

struct ArchitectureTransfer{A}
    architecture :: A
end

Adapt.adapt_structure(transfer::ArchitectureTransfer, c::PerColumnCoefficient) =
    PerColumnCoefficient(on_architecture(transfer.architecture, c.values))
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Write into the [`PerColumnCoefficient`](@ref)s of `closure`, built by [`per_column_closure`](@ref),
the scalars of `closures`, an array of closures of the type and size it was built from. The arrays
are written in place, so a model, or a compiled time step, that holds `closure` uses the new values.
"""
function set_per_column_coefficients!(closure, closures::AbstractMatrix)
    coefficients = per_column_coefficients(closure)
    isempty(coefficients) &&
        throw(ArgumentError("set_per_column_coefficients!: the closure has no PerColumnCoefficient; build it with per_column_closure"))
    size(closures) == size(first(coefficients).values) ||
        throw(ArgumentError("set_per_column_coefficients!: $(size(closures)) closures for $(size(first(coefficients).values)) columns"))
    scalars = map(closure_scalars_in_adapt_order, closures)
    length(first(scalars)) == length(coefficients) ||
        throw(ArgumentError("set_per_column_coefficients!: the closures have $(length(first(scalars))) scalars, " *
                            "the per-column closure $(length(coefficients))"))

    for (n, coefficient) in enumerate(coefficients)
        values = map(column_scalars -> column_scalars[n], scalars)
        copyto!(coefficient.values, convert(Matrix{eltype(coefficient.values)}, values))
    end
    return closure
end
