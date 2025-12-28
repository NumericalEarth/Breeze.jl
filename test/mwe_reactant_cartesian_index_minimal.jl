#=
# MWE: CartesianIndex indexing for OffsetArray during Reactant tracing

## Summary
This MWE demonstrates the CartesianIndex issue in ReactantOffsetArraysExt.
The issue is FIXED in BreezeReactantExt - run with `using Breeze` to see the fix.

## Original Error (without fix)
```
MethodError: no method matching get_ancestor_and_indices_inner(
    ::OffsetArray{TracedRNumber{Float64}, 3, TracedRArray{Float64, 3}},
    ::CartesianIndex{3})
```

## Environment
- Reactant v0.2.190
- OffsetArrays v1.17.0

## Steps to reproduce
Run this script with:
  julia --project=test test/mwe_reactant_cartesian_index_minimal.jl
=#

# Uncomment the next line to enable the BreezeReactantExt fix:
# using Breeze

using Reactant
using OffsetArrays

println("="^70)
println("MWE: CartesianIndex + OffsetArray<TracedRArray> during @compile")
println("="^70)
println()
println("Reactant: v", pkgversion(Reactant))
println("OffsetArrays: v", pkgversion(OffsetArrays))
println()

# Check if BreezeReactantExt fix is loaded
breeze_loaded = isdefined(Main, :Breeze)
if breeze_loaded
    println("BreezeReactantExt: LOADED (CartesianIndex fix enabled)")
else
    println("BreezeReactantExt: NOT LOADED (use `using Breeze` to enable fix)")
end
println()

# Create a concrete Reactant array wrapped in OffsetArray
arr = Reactant.to_rarray(rand(4, 4, 4))
offset_arr = OffsetArray(arr, 0:3, 0:3, 0:3)

println("Array type: ", typeof(offset_arr))
println()

# ============================================================================
# TEST 1: CartesianIndex on concrete array - WORKS
# ============================================================================
println("TEST 1: CartesianIndex on ConcretePJRTArray (outside @compile)")
val = offset_arr[CartesianIndex(1, 1, 1)]
println("  offset_arr[CartesianIndex(1,1,1)] = $val ✓")
println()

# ============================================================================
# TEST 2: CartesianIndex inside @compile
# ============================================================================
println("TEST 2: CartesianIndex inside @compile (during tracing)")
println()

# Simple function that uses CartesianIndex
function index_with_cartesian(arr)
    return arr[CartesianIndex(1, 1, 1)]
end

try
    compiled_fn = @compile index_with_cartesian(offset_arr)
    result = compiled_fn(offset_arr)
    println("  SUCCESS: result = $result ✓")
catch e
    println("  FAILED:")
    err_str = sprint(showerror, e)
    lines = split(err_str, "\n")
    println("  ", join(lines[1:min(5, length(lines))], "\n  "))
    println()
end

# ============================================================================
# FIX (implemented in BreezeReactantExt.jl)
# ============================================================================
println("-"^70)
println("FIX (in BreezeReactantExt.jl, should be upstreamed to Reactant.jl)")
println("-"^70)
println("""
```julia
function Reactant.TracedUtils.get_ancestor_and_indices_inner(
    x::OffsetArray{<:TracedRNumber,N}, idx::CartesianIndex{N}
) where {N}
    return Reactant.TracedUtils.get_ancestor_and_indices_inner(x, Tuple(idx)...)
end
```

This converts CartesianIndex to tuple and splats it into the existing method.
""")
