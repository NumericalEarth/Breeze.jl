include(joinpath(@__DIR__, "setup.jl"))

using Test, Breeze, Oceananigans
using Oceananigans.Fields: interior

using Breeze.AtmosphereModels: same_level_borrow!

# Exact transfers in the order the routine performs them: each recipient in turn draws from
# every lighter reservoir behind it. Evaluated in `BigFloat`, so it carries no roundoff of its
# own and serves as truth.
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

# `same_level_borrow!` is pointwise and architecture-independent, so it is exercised here on
# the CPU, where the reservoirs can be read back exactly. The P3 integration tests cover it
# on whichever architecture they run.
@testset "Species borrowing leaves no residual negative" begin
    for FT in all_float_types()
        grid = RectilinearGrid(CPU(), FT; size=(1, 1, 1), extent=(1, 1, 1))
        fields = ntuple(_ -> CenterField(grid), 5)
        condensates, vapor = fields[1:4], fields[5]

        # Ordered reservoirs: coating, ice, rain, cloud, vapor [kg/m³]. The first two are
        # deficits small enough that a round trip through mass-fraction units would not return
        # them to zero; the rest cover several donors, a deficit vapor cannot cover, and a
        # negative in a reservoir with no lighter donor behind it.
        cases = ((-1.7428903875682522e-9, 0, 0, 0, 0.005),
                 (-6.488883055408995e-24, 0, 0, 0, 0.005),
                 (-0.004, 0.001, 0.002, 0, 0.005),
                 (-0.009, 0.001, 0, 0, 0.002),
                 (0, 0, -0.002, 0, 0.003),
                 (0, 0.001, -0.002, 0, 0),
                 (0.0001, 0.0002, 0.0003, 0.0004, 0.005),
                 (0, 0.001, 0, 0, -0.002))

        for initial in cases
            values = FT.(initial)
            expected = borrowing_oracle(values)
            scale = sum(abs, BigFloat.(values))

            foreach((field, value) -> set!(field, value), fields, values)
            same_level_borrow!(1, 1, 1, condensates, vapor)
            actual = map(field -> interior(field)[1], fields)

            @test all(isfinite, actual)
            @test abs(sum(BigFloat.(actual)) - sum(BigFloat.(values))) <= 16eps(FT) * scale

            for n in eachindex(actual)
                @test abs(BigFloat(actual[n]) - expected[n]) <= 16eps(FT) * scale
                # A fully funded deficit has to land on zero, not near it. A tolerance here
                # would admit exactly the residual negative this routine exists to remove.
                iszero(expected[n]) && @test iszero(actual[n])
                expected[n] >= 0 && @test actual[n] >= 0
            end

            # Repairing an already-repaired state must change nothing.
            same_level_borrow!(1, 1, 1, condensates, vapor)
            @test map(field -> interior(field)[1], fields) == actual
        end
    end
end
