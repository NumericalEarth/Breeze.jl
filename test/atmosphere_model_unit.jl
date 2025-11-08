using Breeze
using Oceananigans
using Test

@testset "AtmosphereModel" begin
    for FT in (Float32, Float64)
        grid = RectilinearGrid(FT, size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
        thermo = ThermodynamicConstants(FT)

        for p₀ in (101325, 100000)
            for θ₀ in (288, 300)
                constants = Breeze.Thermodynamics.ReferenceStateConstants(p₀, θ₀)
                formulation = AnelasticFormulation(grid, constants, thermo)
                model = AtmosphereModel(grid; formulation)

                # test set!
                ρᵣ = model.formulation.reference_density
                cᵖᵈ = model.thermodynamics.dry_air.heat_capacity
                ρeᵢ = ρᵣ * cᵖᵈ * θ₀

                set!(model; θ = θ₀)
                ρe₁ = deepcopy(model.energy)

                set!(model; ρe = ρeᵢ)
                @test model.energy == ρe₁
            end
        end
    end
end
