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

# The fluid must not feel the state stored inside the body (issue #1234)
@testset "Flow past an immersed body is independent of the state inside it" begin
    underlying = RectilinearGrid(size=(16, 8, 8), x=(-4, 4), y=(-2, 2), z=(-2, 2),
                                 topology=(Periodic, Periodic, Bounded))
    body(x, y, z) = (y^2 + z^2 < 0.49) & (-1.5 < x < 1.5)
    grid = ImmersedBoundaryGrid(underlying, GridFittedBoundary(body))
    inside = [Oceananigans.ImmersedBoundaries.immersed_cell(i, j, k, grid) for i in 1:16, j in 1:8, k in 1:8]

    function flow_past_body(ρᵈ_inside, ρθ_inside)
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)
        model = AtmosphereModel(grid; dynamics)
        set!(model; θ=300, u=10, qᵗ=0, ρ=model.dynamics.reference_state.density)
        interior(model.dynamics.dry_density)[inside] .= ρᵈ_inside
        interior(model.formulation.potential_temperature_density)[inside] .= ρθ_inside
        for _ in 1:50
            time_step!(model, 0.002)
        end
        return Array(interior(model.velocities.u))
    end

    uᴮ = flow_past_body(1, 300)
    uᶜ = flow_past_body(1.2, 250)
    @test all(isfinite, uᴮ)
    @test maximum(abs, uᴮ - uᶜ) < 1e-8
end
