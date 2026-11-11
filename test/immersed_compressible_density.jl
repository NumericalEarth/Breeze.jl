include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Test

# Air at rest around a flat-faced cylinder (issue #1234)
@testset "set! keeps density inside an immersed body" begin
    underlying = RectilinearGrid(size=(32, 16, 16), x=(-8, 8), y=(-4, 4), z=(-4, 4),
                                 topology=(Periodic, Periodic, Bounded))
    body(x, y, z) = (y^2 + z^2 < 0.49) & (-3 < x < 3)
    grid = ImmersedBoundaryGrid(underlying, GridFittedBoundary(body))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)
    model = AtmosphereModel(grid; dynamics)
    set!(model; θ=300, u=0, qᵗ=0, ρ=model.dynamics.reference_state.density)

    ρ = model.dynamics.total_density
    @test minimum(interior(ρ)) > 1
    @test all(isfinite, interior(model.formulation.potential_temperature))

    time_step!(model, 0.001)
    @test all(isfinite, interior(ρ))
end
