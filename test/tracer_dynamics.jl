include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Test

@testset "AtmosphereModel tracers advection [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(16, 8, 8), x=(0, 1_000), y=(0, 500), z=(0, 500))
    model = AtmosphereModel(grid; tracers=(:a, :b))
    set!(model; u = 1)
    # Initialize tracer with an x-gradient
    Lx = FT(1_000)
    set!(model.tracers.a, (x, y, z) -> sin(2π * x / Lx))
    set!(model.tracers.b, (x, y, z) -> cos(2π * x / Lx))

    @test try
        time_step!(model, 1)
        true
    catch
        false
    end
end

@testset "Tracer halos stay finite where the total density is zero [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000),
                           halo=(5, 5, 5), topology=(Bounded, Bounded, Bounded))
    model = AtmosphereModel(grid; tracers=(:a,))
    set!(model; u = 1)
    set!(model.tracers.a, 1)

    ρ = Array(parent(Breeze.AtmosphereModels.total_density(model.dynamics)))
    @test any(iszero, ρ)   # the outer halo a mirror fill cannot reach

    time_step!(model, 1)
    @test !any(isnan, Array(parent(model.tracers.a)))
    @test all(isfinite, Array(interior(model.tracers.a)))
end
