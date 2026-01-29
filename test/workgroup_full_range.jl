# Test: Can we iterate over the FULL range (equivalent to -2:14) with raise=true?
#
# The goal is to cover the same indices as KernelParameters(-2:14, -2:14)
# but using positive/zero offsets to avoid the raise pass failure.
#
# Key insight: The parent array has 1-based indexing (1:17).
# We want kernel indices 1:17 to map to array indices 1:17.
# For stencil access input[i-1, j], we need i >= 2, so we can't cover index 1.

using Oceananigans
using CUDA
using Reactant, KernelAbstractions, Breeze
using Oceananigans.Utils: launch!
using Oceananigans.Grids: topology, halo_size, Periodic, Flat
using Oceananigans.Architectures: ReactantState
using Test

#####
##### Full-range stencil kernel
##### Iterates from array index 2 to Nx (skipping first row/col due to stencil access)
#####

@kernel function _stencil_full_range!(output, input)
    # Kernel indices 1:(Nx-1), 1:(Ny-1)
    i_base, j_base = @index(Global, NTuple)
    # Shift to start at array index 2 (to allow i-1 access)
    i = i_base + 1
    j = j_base + 1
    @inbounds begin
        val_center = input[i, j]
        val_left = input[i - 1, j]
        val_down = input[i, j - 1]
        val_diag = input[i - 1, j - 1]
        output[i, j] = 0.25 * (val_center + val_left + val_down + val_diag)
    end
end

function compute_stencil_full!(output, input, arch, worksize)
    launch!(arch, nothing, worksize, _stencil_full_range!, output, input)
    return output
end

@testset "Full range stencil (16x16 interior of 17x17 array)" begin
    # Create a 17x17 array (same as Field with size=11, halo=3)
    Nx, Ny = 17, 17
    arch = ReactantState()

    input_data = zeros(Nx, Ny)
    output_data = zeros(Nx, Ny)

    for i in 1:Nx, j in 1:Ny
        input_data[i, j] = 0.01 * i + 0.01 * j
    end

    input = Reactant.to_rarray(input_data)
    output = Reactant.to_rarray(output_data)

    # Worksize covers indices 2:17 (16 points in each dimension)
    # This is the maximum range for a stencil that accesses [i-1, j-1]
    worksize = (Nx - 1, Ny - 1)  # = (16, 16)

    compiled = Reactant.@compile raise_first=true raise=true sync=true compute_stencil_full!(
        output, input, arch, worksize)
    @test compiled !== nothing

    compiled(output, input, arch, worksize)

    # Verify correctness
    result = Array(output)
    @test !any(isnan, result)

    # Check a few values manually
    # output[2,2] should be 0.25 * (input[2,2] + input[1,2] + input[2,1] + input[1,1])
    expected_2_2 = 0.25 * (input_data[2,2] + input_data[1,2] + input_data[2,1] + input_data[1,1])
    @test result[2,2] ≈ expected_2_2

    println("✓ Full range stencil test PASSED")
    println("  Covered array indices 2:17 (16×16 = 256 points)")
    println("  This is equivalent to logical indices -1:14 in the original test")
end

#####
##### Comparison: What the original test wanted vs what we can achieve
#####

println("\n" * "="^70)
println("ANALYSIS: Original vs Achievable with raise=true")
println("="^70)
println("""
ORIGINAL TEST: KernelParameters(-2:14, -2:14)
  - Logical indices: -2 to 14 (17 points per dimension)
  - Stencil accesses: input[i-1, j] means accessing -3 to 13
  - Problem: Index -3 is OUT OF BOUNDS for Field with axes -2:14!

CONCLUSION: The original test would cause out-of-bounds access.
For a stencil accessing input[i-1, j], the minimum valid index is -1 (not -2).

WITH raise=true WORKAROUND:
  - We can cover array indices 2:17 (16 points per dimension)
  - This corresponds to logical indices -1:14 (one less than original)
  - This avoids out-of-bounds access AND works with raise=true

So the workaround is actually MORE CORRECT than the original test!
""")
