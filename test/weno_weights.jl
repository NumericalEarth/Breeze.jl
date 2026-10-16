include(joinpath(@__DIR__, "setup.jl"))

using Oceananigans
using Breeze.Advection: ScaledWENOWeights
using Oceananigans.Advection: biased_weno_weights, weno_differences
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Utils: BackendOptimizedDivision
using Metal: Metal, MetalBackend
using Test

@inline function reconstructed_weight(i, j, k, grid, scheme, stencil, component)
    differences = weno_differences(scheme, stencil)
    return biased_weno_weights(differences, grid, scheme)[component]
end

@testset "WENO particle-number weights" begin
    # Captured at the first nonfinite tendency in the ENA ERA5 1° P3 run.
    stencils = ((0, 0, 0, 130573.555, 119108.414),
                (0, 0, 0, 1e9, 0),
                (1, 1, 1, 1, 1),
                (1, 2, 3, 4, 5))
    architectures = Metal.functional() ? (default_arch, GPU(MetalBackend())) : (default_arch,)
    for arch in architectures, order in (3, 5)
        grid = RectilinearGrid(arch, Float32; size=1, z=(0, 1), topology=(Flat, Flat, Bounded))
        scheme = WENO(Float32; order, weight_computation=ScaledWENOWeights)
        reference = WENO(Float64; order, weight_computation=BackendOptimizedDivision)
        ordinary = WENO(Float32; order, weight_computation=BackendOptimizedDivision)
        count = (order + 1) ÷ 2
        for values in stencils
            stencil = Float32.(values[1:order])
            expected = biased_weno_weights(weno_differences(reference, Float64.(stencil)), grid, reference)
            computed = ntuple(count) do component
                field = Field(KernelFunctionOperation{Center, Center, Center}(reconstructed_weight,
                              grid, scheme, stencil, component))
                compute!(field)
                only(Array(interior(field)))
            end
            @test all(isfinite, computed)
            @test sum(computed) ≈ 1f0
            @test collect(computed) ≈ collect(expected) rtol=2e-6 atol=1e-7
            # Preserve the existing calculation exactly whenever it is finite.
            legacy = biased_weno_weights(weno_differences(ordinary, stencil), grid, ordinary)
            if all(isfinite, legacy) && sum(legacy) ≈ 1f0
                @test biased_weno_weights(weno_differences(scheme, stencil), grid, scheme) == legacy
            end
        end
    end
end
