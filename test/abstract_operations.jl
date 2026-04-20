using Test
using Breeze
using Oceananigans
using Oceananigans.AbstractOperations: BinaryOperation

@testset "Registered binary operations [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), extent=(1, 1, 1))

    a = CenterField(grid)
    b = CenterField(grid)

    # atand(a, b) with a = b = 1 gives 45°
    set!(a, 1)
    set!(b, 1)
    op = @at (Center, Center, Center) atand(a, b)
    @test op isa BinaryOperation
    @test all(interior(compute!(Field(op))) .≈ 45)

    # atan gives the same geometry in radians
    op = @at (Center, Center, Center) atan(a, b)
    @test all(interior(compute!(Field(op))) .≈ π / 4)

    # mod wraps to principal range
    set!(a, 370)
    op = @at (Center, Center, Center) mod(a, 360)
    @test op isa BinaryOperation
    @test all(interior(compute!(Field(op))) .≈ 10)
end
