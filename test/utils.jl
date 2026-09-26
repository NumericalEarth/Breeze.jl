include(joinpath(@__DIR__, "setup.jl"))

using Breeze.Utils: sum_properties
using Oceananigans
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
