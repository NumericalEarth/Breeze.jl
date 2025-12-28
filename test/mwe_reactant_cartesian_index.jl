#=
Minimum Working Example: CartesianIndex indexing fails for OffsetArray{TracedRNumber}

This reproducer demonstrates an issue where `get_ancestor_and_indices_inner` in
Reactant's OffsetArrays extension doesn't handle CartesianIndex, only splatted indices.

The issue is triggered when using Oceananigans Fields on ReactantState architecture,
because operations like `maximum(field)` internally iterate using CartesianIndex.

Error:
  MethodError: no method matching get_ancestor_and_indices_inner(
      ::OffsetArray{TracedRNumber{Float64}, 3, TracedRArray{Float64, 3}},
      ::CartesianIndex{3})

  Closest candidates are:
    get_ancestor_and_indices_inner(
        ::OffsetArray{<:TracedRNumber, N, <:AbstractArray}, 
        ::Any...)  # <-- only handles splatted indices

Environment:
  - Julia 1.10+
  - Reactant v0.2.190
  - Oceananigans v0.102.5
  - OffsetArrays v1.17.0
=#

# Uncomment the next line to enable the BreezeReactantExt fix:
# using Breeze

using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState

println("="^70)
println("Minimum Reproducer: CartesianIndex indexing with OffsetArray<TracedRArray>")
println("="^70)
println()

# Versions
println("Versions:")
import Pkg
deps = Pkg.dependencies()
for pkg in ["Reactant", "Oceananigans", "OffsetArrays"]
    uuid = findfirst(d -> d.name == pkg, deps)
    if uuid !== nothing
        println("  $pkg: v$(deps[uuid].version)")
    end
end
println()

Reactant.set_default_backend("cpu")

# Create grid on ReactantState architecture
arch = ReactantState()
grid = RectilinearGrid(arch; size=(8, 8), extent=(1, 1), topology=(Periodic, Flat, Bounded))

println("Grid architecture: ReactantState")
println()

# Create a Field (backed by OffsetArray{Float64, 3, ConcretePJRTArray})
c = CenterField(grid)
println("Field backing array type:")
println("  ", typeof(c.data))
println()

# ============================================================================
# The Issue: maximum(field) fails during Reactant tracing
# ============================================================================

println("-"^70)
println("TEST: maximum(field) - triggers Reactant tracing internally")
println("-"^70)
println()

try
    m = maximum(c)
    println("SUCCESS: maximum(c) = $m")
catch e
    println("FAILED with error:")
    println()
    # Print the relevant part of the error
    err_str = sprint(showerror, e)
    println(err_str)
    println()
    println("-"^70)
    println("ROOT CAUSE:")
    println("-"^70)
    println("""
The `maximum` operation on an Oceananigans Field triggers Reactant compilation
internally. During this compilation, the field is traced and its backing array
becomes an OffsetArray{TracedRNumber{Float64}, 3, TracedRArray{Float64, 3}}.

The mapreduce operation iterates over the array using CartesianIndex, calling:
  offset_arr[CartesianIndex(i, j, k)]

This eventually calls get_ancestor_and_indices_inner with a CartesianIndex,
but ReactantOffsetArraysExt only defines:

  get_ancestor_and_indices_inner(arr::OffsetArray{<:TracedRNumber}, ::Any...)

which expects splatted indices, not a single CartesianIndex argument.

SUGGESTED FIX (in ReactantOffsetArraysExt.jl):
""")
    println("""
```julia
function TracedUtils.get_ancestor_and_indices_inner(
    arr::OffsetArray{T,N,AA}, idx::CartesianIndex{N}
) where {T<:TracedRNumber,N,AA<:AbstractArray{T,N}}
    # Convert CartesianIndex to splatted tuple indices
    return TracedUtils.get_ancestor_and_indices_inner(arr, Tuple(idx)...)
end
```
""")
end
