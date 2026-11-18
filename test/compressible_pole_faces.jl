include(joinpath(@__DIR__, "setup.jl"))

#####
##### CompressibleDynamics with acoustic substepping on a latitude-longitude grid that reaches the poles:
##### the y-faces at the poles have zero width, and the wall-normal momentum on them must not run away
#####

using Breeze
using Breeze.CompressibleEquations: is_impenetrable_wall
using Oceananigans
using Oceananigans.Units
using Test

@testset "Pole faces of a moist compressible model on a global latitude-longitude grid" begin
    grid = LatitudeLongitudeGrid(default_arch, Float64; size=(36, 18, 8), longitude=(0, 360), latitude=(-90, 90),
                                 z=(0, 10kilometers), halo=(4, 4, 3))

    θᵣ(z) = 300 + 0.004 * z
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure=1e5, reference_potential_temperature=θᵣ)
    model = AtmosphereModel(grid; dynamics, advection=WENO(order=5), coriolis=SphericalCoriolis())

    conditions = model.momentum.ρv.boundary_conditions
    @test is_impenetrable_wall(conditions.south)
    @test is_impenetrable_wall(conditions.north)

    # A moist zonal flow that is not balanced, so that the pressure-gradient force acts on the faces next to the poles
    set!(model; θ = (λ, φ, z) -> θᵣ(z) + 2 * cosd(φ)^2,
                qᵗ = (λ, φ, z) -> 5e-3 * cosd(φ)^2 * exp(-z / 3000),
                u = (λ, φ, z) -> 20 * cosd(φ),
                ρ = HydrostaticallyBalancedDensity())

    for _ in 1:12
        time_step!(model, 300)
    end

    @test all(isfinite, Array(interior(model.momentum.ρv)))
    @test all(isfinite, Array(interior(model.temperature)))
    @test maximum(abs, model.velocities.v) < 50
    @test maximum(abs, model.velocities.w) < 5
    # The south pole face holds no momentum (the north one holds what Oceananigans' polar fill writes on it)
    @test all(iszero, Array(interior(model.momentum.ρv, :, 1, :)))
end
