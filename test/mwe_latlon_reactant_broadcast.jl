#####
##### MWE: LatitudeLongitudeGrid broadcast fails on ReactantState
#####
#
# set!(field, BinaryOperation) on a LatitudeLongitudeGrid with ReactantState
# triggers gpu__broadcast_kernel! which fails with:
#
#   InvalidIRError: unsupported dynamic function invocation (call to getindex)
#
# The grid's per-point metric arrays (λᶜᵃᵃ, φᵃᶜᵃ, Δxᶜᶜᵃ, Δxᶠᶜᵃ, …) are stored
# as OffsetVector{Float64, ConcretePJRTArray{Float64,1,1}}.  Even though the
# identity interpolation operator never indexes them, the GPU compiler rejects
# the reachable getindex method on ConcretePJRTArray inside the PTX kernel.
#
# RectilinearGrid is unaffected because its metrics are uniform scalars.

using Oceananigans
using Oceananigans.Architectures: ReactantState
using KernelAbstractions
using CUDA
using Reactant
using Test

@testset "LatitudeLongitudeGrid BinaryOperation broadcast — ReactantState" begin
    grid = LatitudeLongitudeGrid(ReactantState();
                                 size = (4, 4, 4),
                                 halo = (3, 3, 3),
                                 longitude = (0, 360),
                                 latitude = (-80, 80),
                                 z = (0, 1e3))

    a = CenterField(grid)
    b = CenterField(grid)
    c = CenterField(grid)

    set!(a, 1.0)
    set!(b, 2.0)

    # a * b  →  BinaryOperation{Center,Center,Center}
    # set!(c, a * b) falls through to the generic  u .= v  path,
    # which launches _broadcast_kernel! via KernelAbstractions on ReactantBackend.
    @test_broken set!(c, a * b) isa Field
end

#####
##### Distilled: getindex on ConcreteRArray inside a KA kernel
#####
#
# The Oceananigans test above fails because the broadcast kernel's argument
# types carry the full LatitudeLongitudeGrid, whose metric vectors are
# OffsetVector{Float64, ConcretePJRTArray{Float64,1,1}}.  The GPU compiler
# sees a reachable getindex path on ConcretePJRTArray and rejects it.
#
# Below we strip away Oceananigans entirely and check whether a bare
# KernelAbstractions kernel on ReactantBackend can index a ConcreteRArray.

# using KernelAbstractions

# @kernel function _getindex_kernel!(out, v)
#     i = @index(Global, Linear)
#     @inbounds out[i] = v[i]
# end

# @testset "ConcreteRArray getindex inside KA kernel — ReactantBackend" begin
#     N = 4
#     v   = Reactant.ConcreteRArray(collect(1.0:Float64(N)))
#     out = Reactant.ConcreteRArray(zeros(N))

#     backend = KernelAbstractions.get_backend(out)

#     @test begin
#         _getindex_kernel!(backend)(out, v; ndrange=N)
#         Array(out) ≈ collect(1.0:Float64(N))
#     end
# end
