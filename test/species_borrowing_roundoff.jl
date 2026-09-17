include(joinpath(@__DIR__, "setup.jl"))

using Test, Breeze, Oceananigans
using Oceananigans.Fields: interior

using Breeze.AtmosphereModels: same_level_borrow!, vertical_borrow!, VerticalBorrowing

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

# The vertical phase runs after species borrowing, so it has the final say on the sign of the
# vapor density. It used to add the deficit back rather than storing zero, and
# `ρqᵛ + fl(fl(-ρqᵛ Δz) / Δz)` is not zero when Δz is not a power of two, so the level it had
# just repaired kept a residual negative.
@testset "Vertical borrowing leaves no residual negative" begin
    # Spacings and columns found by searching for cases the old formulation got wrong; with
    # uniform Δz the multiply and divide cancel exactly and nothing can be detected.
    cases = ((Float64[17.925947, 21.916277, 18.430248, 32.64371, 26.351135],
              Float64[0.00079591933, 6.736659e-5, -0.00022146673, 0.0012925179, 0.0008010718]),
             (Float64[43.71562, 28.572842, 31.05178, 25.83191, 11.608398],
              Float64[0.0012098069, 0.0011361514, 0.00025348854, 0.0004519669, -1.3999701e-5]),
             (Float64[38.30810989117188, 33.8214341458272, 19.752342365407856, 32.08863719382802, 7.366264936200736],
              Float64[0.0008740449988800256, -0.000585512649881292, 0.00038679380905242236, 0.001026332338399416, -0.0002195252140977566]),
             (Float64[5.949358174326838, 30.46975789943479, 29.736028402684596, 12.999849300764225, 41.23175844506237],
              Float64[0.0005903563460589667, 0.0004848270683791267, 0.0010217099811830152, 0.000560303893654861, -0.00032224134432356567]))

    for FT in all_float_types(), (spacing, column) in cases
        z = FT.(vcat(0, cumsum(spacing)))
        grid = RectilinearGrid(CPU(), FT; size=(1, 1, length(column)), x=(0, 1), y=(0, 1), z=z)
        ρqᵛ = CenterField(grid)

        before = FT.(column)
        Δz = FT.(spacing)
        interior(ρqᵛ)[1, 1, :] .= before
        mass_before = sum(BigFloat.(before) .* BigFloat.(Δz))

        vertical_borrow!(ρqᵛ, 1, 1, grid, VerticalBorrowing())
        after = collect(interior(ρqᵛ)[1, 1, :])

        @test all(isfinite, after)
        # Every column here holds more vapor than deficit, so the sweep can fund all of it and
        # nothing may be left negative. A tolerance would admit exactly the residual this is about.
        @test all(>=(0), after)
        # Transfers are mass per unit area, so the column integral is what is conserved.
        @test abs(sum(BigFloat.(after) .* BigFloat.(Δz)) - mass_before) <=
              32eps(FT) * sum(abs, BigFloat.(before) .* BigFloat.(Δz))
    end
end
