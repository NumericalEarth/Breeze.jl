include(joinpath(@__DIR__, "runtests_setup.jl"))

@testset "AtmosphereModel" begin
    for FT in (Float32, Float64)
        grid = RectilinearGrid(FT, size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
        thermodynamics = ThermodynamicConstants(FT)

        for p₀ in (101325, 100000)
            for θ₀ in (288, 300)
                reference_constants = ReferenceStateConstants(FT, base_pressure=p₀, potential_temperature=θ₀)
                formulation = AnelasticFormulation(grid, reference_constants, thermodynamics)
                model = AtmosphereModel(grid; formulation)

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
