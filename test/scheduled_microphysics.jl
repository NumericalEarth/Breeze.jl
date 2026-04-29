using Breeze
using Oceananigans
using Oceananigans.Utils: IterationInterval
using Test

@testset "Scheduled microphysics: construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 1000))

    @testset "default (no schedule)" begin
        model = AtmosphereModel(grid)
        @test model.microphysics_schedule === nothing
        @test model.microphysics_tendencies === nothing
        @test model.microphysics_state === nothing
    end

    @testset "with IterationInterval(5)" begin
        model = AtmosphereModel(grid; microphysics_schedule = IterationInterval(5))
        @test model.microphysics_schedule isa IterationInterval
        @test model.microphysics_tendencies isa NamedTuple
        @test :ρθ in keys(model.microphysics_tendencies)
        @test :ρqᵛ in keys(model.microphysics_tendencies)
        @test model.microphysics_state isa Breeze.AtmosphereModels.MicrophysicsScheduleState{FT}
        @test model.microphysics_state.last_fire_iteration == -1
    end
end
