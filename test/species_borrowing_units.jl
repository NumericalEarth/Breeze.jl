include(joinpath(@__DIR__, "setup.jl"))

using Test, Breeze, Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

@kernel function _test_species_borrowing!(fields, density)
    i, j, k = @index(Global, NTuple)
    Breeze.AtmosphereModels.same_level_borrow!(i, j, k, density,
                                              fields[1:4], fields[5])
end

function borrowing_oracle(values)
    mass = collect(BigFloat.(values))
    for recipient in 1:length(mass)-1
        for donor in recipient+1:length(mass)
            transfer = min(max(0, -mass[recipient]), max(0, mass[donor]))
            mass[recipient] += transfer
            mass[donor] -= transfer
        end
    end
    return mass
end

@testset "Species borrowing in partial-density units" begin
    for FT in (Float32, Float64)
        grid = RectilinearGrid(default_arch, FT; size=(2,2,2), extent=(1,1,1))
        fields = ntuple(_ -> CenterField(grid), 5)
        # Ordered reservoirs: coating, ice, rain, cloud, vapor [kg/m³]. Include
        # fully funded roundoff witnesses, multiple donors, insufficient water,
        # and a backward donor that the configured policy must leave untouched.
        cases = ((-1.7428903875682522e-9,0,0,0,0.005),
                 (-6.488883055408995e-24,0,0,0,0.005),
                 (-0.004,0.001,0.002,0,0.005),
                 (-0.009,0.001,0,0,0.002),
                 (0,0,-0.002,0,0.003),
                 (0,0.001,-0.002,0,0),
                 (0.0001,0.0002,0.0003,0.0004,0.005),
                 (0,0.001,0,0,-0.002))
        for initial in cases
            values = FT.(initial)
            expected = borrowing_oracle(values)
            scale = sum(abs, BigFloat.(values))
            reference = nothing
            for density in FT.((0.44961270689964294,0.949273943901062,
                                1.127671480178833,1e-4,1e4))
                foreach((field, value) -> set!(field, value), fields, values)
                launch!(default_arch, grid, :xyz, _test_species_borrowing!, fields, density)
                actual = map(field -> Array(interior(field))[1], fields)
                @test all(isfinite, actual)
                @test abs(sum(BigFloat.(actual)) - sum(BigFloat.(values))) <= 16eps(FT)*scale
                for n in eachindex(actual)
                    @test abs(BigFloat(actual[n]) - expected[n]) <= 16eps(FT)*scale
                    # A fully covered deficit must be exactly zero. A tolerance
                    # here would hide the negative residual rejected by RT staging.
                    iszero(expected[n]) && @test iszero(actual[n])
                    expected[n] >= 0 && @test actual[n] >= 0
                end
                reference === nothing || @test actual == reference
                reference = actual
                launch!(default_arch, grid, :xyz, _test_species_borrowing!, fields, density)
                @test map(field -> Array(interior(field))[1], fields) == actual
            end
        end
    end
end
