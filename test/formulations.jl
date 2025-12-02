using Breeze
using Oceananigans
using Test

@testset "AnelasticFormulation with different reference states [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants)

    for thermo_name in (:PotentialTemperature, :StaticEnergy)
        @testset "Time-stepping AtmosphereModel with $thermo_name thermodynamics" begin
            formulation = AnelasticFormulation(reference_state, thermodynamics=thermo_name)
            model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation)
            thermo_type = eval(Symbol(thermo_name, :Thermodynamics))
            @test model.formulation.thermodynamics isa thermo_type
        end
    end
end