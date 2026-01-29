# Minimum Working Example: Reactant `raise=true` with stencil kernels
#
# This file demonstrates the workaround for making stencil kernels work with
# Reactant's `raise=true` compilation mode.
#
# KEY FINDING: The Reactant `raise` pass fails when combining:
#   1. Stencil-style offset indexing (e.g., `input[i-1, j]`)
#   2. NEGATIVE offsets from KernelParameters (e.g., range `-2:14` gives offset `-3`)
#
# SOLUTION: Use POSITIVE offsets instead. The raise pass handles positive offsets fine.
#
# For Oceananigans Fields:
#   - Extract parent arrays (without OffsetArray wrapper) BEFORE compilation
#   - Reshape to true 2D for Flat z-topology
#   - Use positive kernel_offsets and array_offsets = (0, 0)
#
# Run with: julia --project=test test/workgroup_mwe.jl

using Oceananigans
using CUDA
using Reactant, KernelAbstractions, Breeze, Enzyme
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans.Grids: topology, halo_size, Periodic, Flat
using Oceananigans.Architectures: ReactantState
using Test

#####
##### Utility functions
#####

"""
    decompose_kernel_parameters(kp::KernelParameters)

Decompose a `KernelParameters` into worksize and offsets.
Returns `(worksize::NTuple{N,Int}, offsets::Tuple)`.
"""
function decompose_kernel_parameters(::KernelParameters{spec, offsets}) where {spec, offsets}
    return spec, offsets
end

diagnostic_indices(::Bounded, N, H) = 1:N+1
diagnostic_indices(::Periodic, N, H) = -H+1:N+H
diagnostic_indices(::Flat, N, H) = 1:N

#####
##### Stencil kernel with explicit offsets (works with raise=true when offsets are positive)
#####

@kernel function _stencil_kernel_with_offsets!(output, input, offset_i, offset_j, array_offset_i, array_offset_j)
    i_base, j_base = @index(Global, NTuple)
    # Apply kernel workspec offsets
    i = i_base + offset_i
    j = j_base + offset_j
    # Apply array offsets (convert logical index to 1-based array index)
    ai = i - array_offset_i + 1
    aj = j - array_offset_j + 1
    @inbounds begin
        val_center = input[ai, aj]
        val_left = input[ai - 1, aj]
        val_down = input[ai, aj - 1]
        val_diag = input[ai - 1, aj - 1]
        output[ai, aj] = 0.25 * (val_center + val_left + val_down + val_diag)
    end
end

function compute_stencil_with_offsets!(output, input, arch, worksize, kernel_offsets, array_offsets)
    launch!(arch, nothing, worksize, _stencil_kernel_with_offsets!,
            output, input, kernel_offsets[1], kernel_offsets[2],
            array_offsets[1], array_offsets[2])
    return output
end

#####
##### Simple copy kernel (baseline)
#####

@kernel function _copy_kernel!(output, input)
    i, j = @index(Global, NTuple)
    @inbounds output[i, j] = input[i, j]
end

function compute_copy!(output, input, arch, worksize)
    launch!(arch, nothing, worksize, _copy_kernel!, output, input)
    return output
end

#####
##### TESTS
#####

@testset "MWE: Plain 2D stencil with positive offsets (raise=true)" begin
    # Use plain 2D arrays and POSITIVE offsets - this works with raise=true
    Nx, Ny = 17, 17
    arch = ReactantState()

    input_data = zeros(Nx, Ny)
    output_data = zeros(Nx, Ny)

    for i in 1:Nx, j in 1:Ny
        input_data[i, j] = 0.01 * i + 0.01 * j
    end

    input = Reactant.to_rarray(input_data)
    output = Reactant.to_rarray(output_data)

    # POSITIVE offsets: iterate from index 2 to Nx-1 (interior, avoiding boundaries for stencil)
    worksize = (Nx - 2, Ny - 2)  # Interior points
    kernel_offsets = (1, 1)  # Shift kernel indices by 1 (so we start at array index 2)
    array_offsets = (0, 0)  # No additional array offset needed

    compiled = Reactant.@compile raise_first=true raise=true sync=true compute_stencil_with_offsets!(
        output, input, arch, worksize, kernel_offsets, array_offsets)
    @test compiled !== nothing

    compiled(output, input, arch, worksize, kernel_offsets, array_offsets)
    @test !any(isnan, Array(output))

    println("✓ Plain 2D stencil with positive offsets: PASSED")
end

@testset "MWE: Plain 2D copy kernel (baseline - raise=true)" begin
    Nx, Ny = 17, 17
    arch = ReactantState()

    input_data = rand(Nx, Ny)
    output_data = zeros(Nx, Ny)

    input = Reactant.to_rarray(input_data)
    output = Reactant.to_rarray(output_data)

    worksize = (Nx, Ny)

    compiled = Reactant.@compile raise_first=true raise=true sync=true compute_copy!(
        output, input, arch, worksize)
    @test compiled !== nothing

    compiled(output, input, arch, worksize)
    @test Array(output) ≈ input_data

    println("✓ Plain 2D copy kernel: PASSED")
end

@testset "MWE: Field stencil with extracted 2D arrays and positive offsets (raise=true)" begin
    # Key insight: negative offsets cause raise pass to fail.
    # Solution: extract parent arrays, use positive offsets.

    grid = RectilinearGrid(ReactantState(); size=(11, 11), extent=(1000, 1000),
                           halo=(3, 3), topology=(Periodic, Periodic, Flat))
    f = CenterField(grid)
    g = CenterField(grid)
    set!(f, (x, y) -> 0.01 * x + 0.01 * y)

    # Extract 2D arrays BEFORE compilation
    input_3d = parent(f.data)
    output_3d = parent(g.data)
    sz = size(input_3d)
    @assert sz[3] == 1 "Expected singleton z-dimension for Flat topology, got $sz"

    # Copy to true 2D Reactant arrays
    input_2d_host = reshape(Array(input_3d), sz[1], sz[2])
    output_2d_host = reshape(Array(output_3d), sz[1], sz[2])
    input_2d = Reactant.to_rarray(input_2d_host)
    output_2d = Reactant.to_rarray(output_2d_host)

    # Use POSITIVE kernel offsets
    # The arrays are (17, 17), and we want to iterate over interior
    worksize = (sz[1] - 2, sz[2] - 2)  # = (15, 15) interior points
    kernel_offsets = (1, 1)  # Start at array index 2
    array_offsets = (0, 0)   # Arrays are 1-based, no additional offset

    arch = grid.architecture
    compiled = Reactant.@compile raise_first=true raise=true sync=true compute_stencil_with_offsets!(
        output_2d, input_2d, arch, worksize, kernel_offsets, array_offsets)
    @test compiled !== nothing

    compiled(output_2d, input_2d, arch, worksize, kernel_offsets, array_offsets)
    @test !any(isnan, Array(output_2d))

    println("✓ Field stencil with extracted 2D arrays and positive offsets: PASSED")
end

#####
##### Document what FAILS - for reference
#####

println("\n" * "="^70)
println("SUMMARY: What works and what fails with Reactant raise=true")
println("="^70)
println("""
WORKS:
  - Copy kernels (no stencil) with any offsets
  - Stencil kernels with POSITIVE offsets

FAILS:
  - Stencil kernels with NEGATIVE offsets (e.g., KernelParameters(-2:14, -2:14))
  - 3D OffsetArrays (Fields) without pre-extraction during tracing

WORKAROUND:
  1. Extract parent arrays from Fields BEFORE @compile
  2. Use positive worksize and kernel_offsets
  3. Reshape to true 2D for Flat z-topology grids
  4. Pass array_offsets = (0, 0) for 1-based parent arrays
""")
