include(joinpath(@__DIR__, "setup.jl"))

using Breeze.Utils: sum_properties, cube_root
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans
using Metal: Metal, MetalBackend
using Test

@testset "Property sums [$FT]" for FT in test_float_types()
    properties = (a=FT(1), b=FT(2), c=FT(4))
    @test sum_properties(properties, (:a, :c)) === FT(5)
    @test sum_properties(properties, (:a, :b, :c)) === FT(7)

    grid = RectilinearGrid(default_arch, FT; size=2, z=(0, 1), topology=(Flat, Flat, Bounded))
    a, b, c = CenterField(grid), CenterField(grid), CenterField(grid)
    set!(a, 1)
    set!(b, 2)
    set!(c, 4)

    # A single field is returned intact; multiple fields form a lazy sum without
    # introducing a scalar zero, which field addition does not support.
    @test sum_properties((; a), (:a,)) === a
    result = Field(sum_properties((; a, b, c), (:a, :b, :c)))
    @test all(Array(interior(result)) .== 7)
end

@testset "Single-precision cube root" begin
    # Cover the Float32 exponent range against a double-precision reference.
    magnitudes = unique(vcat(Float32[0, nextfloat(0f0), floatmin(Float32), floatmax(Float32)],
                              Float32.(10.0 .^ range(-44, 38; length=1000))))
    inputs = vcat(-magnitudes, magnitudes)
    expected = Float32.(cbrt.(Float64.(inputs)))
    computed = cube_root.(inputs)
    @test eltype(computed) == Float32
    @test all(isapprox.(computed, expected; rtol=2eps(Float32), atol=0))
    @test isequal(cube_root(-0f0), -0f0)
    @test cube_root(Inf32) == Inf32
    @test cube_root(-Inf32) == -Inf32
    @test isnan(cube_root(NaN32))
    @test cube_root(8.0) === 2.0

    arches = Metal.functional() ? (default_arch, GPU(MetalBackend())) : (default_arch,)
    # Metal flushes subnormal inputs to zero. Test its normal range explicitly;
    # the host check above also covers subnormals.
    normal = iszero.(inputs) .| (abs.(inputs) .≥ floatmin(Float32))
    special = Float32[-0.0, 0.0, Inf, -Inf, NaN]
    device_inputs = vcat(inputs[normal], special)
    for arch in arches
        grid = RectilinearGrid(arch, Float32; size=length(device_inputs), z=(0, 1), topology=(Flat, Flat, Bounded))
        values = CenterField(grid)
        set!(values, reshape(device_inputs, 1, 1, :))
        evaluate(i, j, k, grid, values) = cube_root(@inbounds values[i, j, k])
        result = Field(KernelFunctionOperation{Center, Center, Center}(evaluate, grid, values))
        compute!(result)
        computed = vec(Array(interior(result)))
        @test all(isapprox.(computed[1:count(normal)], expected[normal]; rtol=8eps(Float32), atol=0))
        @test isequal(computed[end-4:end], special)
    end
end
