#####
##### Closure coefficients that vary from column to column
#####
##### A closure on a column ensemble may hold one value of each scalar per column, so that one
##### model integrates an ensemble of closures with identical forcing and initial state. Every
##### floating-point scalar of such a closure is a `PerColumnCoefficient`, an `(Nx, Ny)` array, and the
##### kernels read the closure of column `(i, j)` with `getclosure(i, j, closure)`, which replaces each
##### `PerColumnCoefficient` by its value in that column. The arrays live on the architecture of the
##### grid: under Reactant they are inputs of the compiled program, so one compiled time step serves
##### every ensemble of coefficients, which `set_per_column_coefficients!` writes in place.
#####

"""
$(TYPEDEF)

The values of one scalar of a closure in each column of a column ensemble, an `(Nx, Ny)` array.
Built by [`per_column_closure`](@ref); a kernel reads the value of column `(i, j)` through
`getclosure(i, j, closure)`.

$(TYPEDFIELDS)
"""
struct PerColumnCoefficient{A}
    "the value in each column, an `(Nx, Ny)` array"
    values :: A
end

Adapt.@adapt_structure PerColumnCoefficient

Base.summary(c::PerColumnCoefficient) = string("PerColumnCoefficient of ", summary(c.values))
Base.show(io::IO, c::PerColumnCoefficient) = print(io, summary(c))

# The adaptor that selects the closure of one column
struct ColumnSelection{I, J}
    i :: I
    j :: J
end

@inline Adapt.adapt_structure(column::ColumnSelection, c::PerColumnCoefficient) = @inbounds c.values[column.i, column.j]

# The closure of column (i, j): each `PerColumnCoefficient` replaced by its value there; a closure
# without one is rebuilt unchanged.
@inline column_closure(i, j, closure) = adapt(ColumnSelection(i, j), closure)

@inline Oceananigans.TurbulenceClosures.getclosure(i, j, closure::TKEBasedTurbulenceClosure) = column_closure(i, j, closure)

#####
##### Construction from an array of closures, and the in-place update
#####

# The adaptor that collects the floating-point scalars of a closure, in the order `adapt` visits them
struct ScalarCollection{V}
    scalars :: V
end

Adapt.adapt_storage(collection::ScalarCollection, x::AbstractFloat) = (push!(collection.scalars, x); x)

# The adaptor that replaces the n-th floating-point scalar by the n-th per-column array
struct ScalarReplacement{V}
    arrays :: V
    count :: Base.RefValue{Int}
end

function Adapt.adapt_storage(replacement::ScalarReplacement, x::AbstractFloat)
    n = replacement.count[] += 1
    return PerColumnCoefficient(replacement.arrays[n])
end

# The adaptor that collects the `PerColumnCoefficient`s of a closure, in the order `adapt` visits them
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

The closure is built for a model, not changed after; [`set_per_column_coefficients!`](@ref) writes
new values in place, so that a compiled time step (under Reactant) serves any ensemble of
coefficients. Every scalar is per column, including those all the closures share.

```jldoctest
using Breeze
using Oceananigans.Grids: ColumnEnsembleSize

grid = RectilinearGrid(size=ColumnEnsembleSize(Nz=4, ensemble=(2, 1)), z=(0, 1000), topology=(Flat, Flat, Bounded))
closures = [TKEBasedTurbulenceClosure(mixing_length=GradientLimitedMixingLength(ℂˢ=1.3, ℂⁱ=ℂⁱ)) for ℂⁱ in (1.4, 1.6), _ in 1:1]
closure = per_column_closure(grid, closures)
closure.mixing_length.ℂⁱ.values

# output
2×1 Matrix{Float64}:
 1.4
 1.6
```
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

    # Every column must recover its closure exactly; a scalar that `adapt` does not reach could not vary
    for j in 1:Ny, i in 1:Nx
        column_closure(i, j, closure) === closures[i, j] ||
            throw(ArgumentError("per_column_closure: column ($i, $j) does not recover its closure; " *
                                "the closures differ in something other than their floating-point scalars"))
    end

    arch = Oceananigans.Architectures.architecture(grid)
    return adapt(ArchitectureTransfer(arch), closure)
end

# The adaptor that moves the per-column arrays to an architecture
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
