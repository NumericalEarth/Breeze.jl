#####
##### Reactant-compatible kernel launching utilities
#####
##### KEY FINDING: The Reactant `raise` pass fails when combining:
#####   1. Stencil-style offset indexing (e.g., `input[i-1, j]`)
#####   2. NEGATIVE offsets from KernelParameters (e.g., range `-2:14` gives offset `-3`)
#####
##### SOLUTION: Use POSITIVE offsets instead. The raise pass handles positive offsets fine.
#####
##### For Oceananigans Fields:
#####   - Extract parent arrays (without OffsetArray wrapper) BEFORE compilation
#####   - Reshape to true 2D for Flat z-topology
#####   - Use positive kernel_offsets and array_offsets = (0, 0)
#####

using Oceananigans.Utils: KernelParameters

# Note: These utilities are provided for reference. For Reactant-compatible
# stencil kernels, the recommended pattern is:
#
# 1. Extract parent arrays from Fields BEFORE @compile
# 2. Use positive worksize and kernel_offsets (e.g., worksize = (Nx-2, Ny-2), offsets = (1, 1))
# 3. Write kernels to apply offsets explicitly to @index results

"""
    decompose_kernel_parameters(kp::KernelParameters)

Decompose a `KernelParameters` into worksize and offsets.

Returns `(worksize::NTuple{N,Int}, offsets::Tuple)` where:
- `worksize` is the size of the kernel (tuple of integers)
- `offsets` is the tuple of offsets from the original ranges

NOTE: If offsets are negative, the Reactant raise pass may fail for stencil kernels.
Consider using positive offsets instead.

# Example

```julia
kp = KernelParameters(-2:14, -2:14)

worksize, offsets = decompose_kernel_parameters(kp)
# worksize = (17, 17)
# offsets = (-3, -3)  # WARNING: negative offsets may cause raise pass to fail!

# Better approach: use positive offsets
worksize = (15, 15)  # Iterate interior
kernel_offsets = (1, 1)  # Shift to start at index 2
array_offsets = (0, 0)  # No array offset for 1-based parent arrays
```
"""
function decompose_kernel_parameters(::KernelParameters{spec, offsets}) where {spec, offsets}
    return spec, offsets
end

"""
    launch_with_offsets!(arch, grid, kp::KernelParameters, kernel!, args...; kwargs...)

Launch a kernel with explicit offset handling for Reactant compatibility.

This function decomposes the `KernelParameters` into a worksize tuple
and appends the offsets as additional kernel arguments.

WARNING: If the KernelParameters has negative offsets (e.g., from ranges like `-2:14`),
the Reactant raise pass may fail for stencil kernels. Consider extracting parent
arrays and using positive offsets instead.

The kernel function must be written to accept offset arguments at the end
and apply them manually to indices obtained from `@index(Global, NTuple)`.

See also: [`decompose_kernel_parameters`](@ref)
"""
function launch_with_offsets!(arch, grid, kp::KernelParameters, kernel!, args...; kwargs...)
    worksize, offsets = decompose_kernel_parameters(kp)
    # Append offsets as additional kernel arguments
    all_args = (args..., offsets...)
    Oceananigans.Utils.launch!(arch, grid, worksize, kernel!, all_args...; kwargs...)
    return nothing
end

# Also support tuple worksize (passthrough for non-KernelParameters)
function launch_with_offsets!(arch, grid, worksize::Tuple, kernel!, args...; kwargs...)
    Oceananigans.Utils.launch!(arch, grid, worksize, kernel!, args...; kwargs...)
    return nothing
end
