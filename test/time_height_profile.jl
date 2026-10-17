include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans: Oceananigans
using Test

@testset "Fixed-iteration profile search" begin
    for count in (2, 3, 4, 5, 8, 9, 16, 17, 33)
        coordinates = cumsum(collect(1.0:count))
        samples = vcat(first(coordinates) - 1, coordinates,
                       (coordinates[1:end-1] .+ coordinates[2:end]) ./ 2,
                       last(coordinates) + 1)
        for x in samples
            k, fraction = Breeze.Forcings.profile_bracket(coordinates, x)
            expected = clamp(searchsortedlast(coordinates, x), 1, count - 1)
            @test k == expected
            @test fraction == clamp((x - coordinates[k]) / (coordinates[k+1] - coordinates[k]), 0, 1)
        end
    end
end

@testset "Time-height interpolation [$FT]" for FT in test_float_types()
    heights = FT[0, 20, 100]
    times = FT[0, 10, 30]
    affine(z, t) = 2z + 3t + 1
    values = [affine(z, t) for z in heights, t in times]
    profile = TimeHeightProfile(heights, times, values)
    for z in FT[0, 10, 20, 70, 100], t in FT[0, 5, 10, 25, 30]
        @test profile(z, t) ≈ affine(z, t)
    end
    @test profile(FT(-1), FT(-1)) == affine(0, 0)
    @test profile(FT(101), FT(31)) == affine(100, 30)
    values .= 0
    @test profile(FT(10), FT(5)) == affine(10, 5)
    @test_throws ArgumentError TimeHeightProfile(FT[0, 0], times, zeros(FT, 2, 3))
    @test_throws ArgumentError TimeHeightProfile(heights, FT[0, NaN], zeros(FT, 3, 2))
    @test_throws ArgumentError TimeHeightProfile(heights, times, fill(FT(NaN), 3, 3))
    @test_throws DimensionMismatch TimeHeightProfile(heights, times, zeros(FT, 2, 3))

    # Exercise interpolation inside an actual field kernel on the test backend.
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    device_profile = TimeHeightProfile(heights, times, profile.values; architecture=default_arch)
    field = CenterField(grid)
    set!(field, device_profile, FT(5))
    @test minimum(field) ≈ affine(FT(12.5), FT(5))
    @test maximum(field) ≈ affine(FT(87.5), FT(5))

    # Horizontally reduced fields, such as forcing profiles, at centers and faces.
    column = Field{Nothing, Nothing, Center}(grid)
    set!(column, device_profile, FT(5))
    @test vec(Array(interior(column))) ≈ [affine(z, FT(5)) for z in FT[12.5, 37.5, 62.5, 87.5]]
    faces = Field{Nothing, Nothing, Face}(grid)
    set!(faces, device_profile, FT(5))
    @test vec(Array(interior(faces))) ≈ [affine(z, FT(5)) for z in FT[0, 25, 50, 75, 100]]
end
