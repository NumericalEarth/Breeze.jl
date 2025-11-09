using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

@testset "AtmosphereModel [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    thermo = ThermodynamicConstants(FT)

    for p₀ in (101325, 100000)
        for θ₀ in (288, 300)
            reference_state = ReferenceState(grid, thermo, base_pressure=p₀, potential_temperature=θ₀)
            formulation = AnelasticFormulation(reference_state)
            model = AtmosphereModel(grid; thermodynamics=thermo, formulation)

            # test set!
            ρᵣ = model.formulation.reference_state.density
            cᵖᵈ = model.thermodynamics.dry_air.heat_capacity
            ρeᵢ = ρᵣ * cᵖᵈ * θ₀

            set!(model; θ = θ₀)
            ρe₁ = deepcopy(model.energy_density)

            set!(model; ρe = ρeᵢ)
            @test model.energy_density ≈ ρe₁
        end
    end
end
