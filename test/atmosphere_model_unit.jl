using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Test

@testset "AtmosphereModel [$(FT)]" for FT in (Float32, Float64)
    Nx = Ny = 3
    grid = RectilinearGrid(default_arch, FT; size=(Nx, Ny, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants(FT)

    for p₀ in (101325, 100000), θ₀ in (288, 300), thermodynamics in (:LiquidIcePotentialTemperature, :StaticEnergy)
        @testset let p₀ = p₀, θ₀ = θ₀, thermodynamics = thermodynamics
            reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)

            # Check that interpolating to the first face (k=1) recovers surface values
            for i = 1:Nx, j = 1:Ny
                @test p₀ ≈ @allowscalar ℑzᵃᵃᶠ(i, j, 1, grid, reference_state.pressure)
                @test ρ₀ ≈ @allowscalar ℑzᵃᵃᶠ(i, j, 1, grid, reference_state.density)
            end

            formulation = AnelasticFormulation(reference_state; thermodynamics)
            model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation)

            # test set!
            ρᵣ = model.formulation.reference_state.density
            cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
            ρeᵢ = ρᵣ * cᵖᵈ * θ₀

            set!(model; θ = θ₀)
            ρe₁ = deepcopy(static_energy_density(model))
            θ₁ = deepcopy(liquid_ice_potential_temperature(model))

            set!(model; ρe = ρeᵢ)
            @test static_energy_density(model) ≈ ρe₁
            @test liquid_ice_potential_temperature(model) ≈ θ₁
        end
    end
end
