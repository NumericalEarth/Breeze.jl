using Test
using Breeze
using Oceananigans

@testset "Breeze.jl" begin
    @testset "Thermodynamics" begin
        thermo = AtmosphereThermodynamics()

        # Test Saturation specific humidity calculation
        T = 293.15  # 20°C
        ρ = 1.2     # kg/m³
        q★ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, thermo.liquid)
        @test q★ > 0
    end

    @testset "AtmosphereModel" begin
        for FT in (Float32, Float64)
            grid = RectilinearGrid(FT, size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
            thermo = AtmosphereThermodynamics(FT)

            for pΔcˡ in (101325, 100000)
                for θΔcˡ in (288, 300)
                    constants = Breeze.Thermodynamics.ReferenceStateConstants(pΔcˡ, θΔcˡ)
                    formulation = AnelasticFormulation(grid, constants, thermo)
                    model = AtmosphereModel(grid; formulation)

                    # test set!
                    ρᵣ = model.formulation.reference_density
                    cᵖᵈ = model.thermodynamics.dry_air.heat_capacity
                    ρeᵢ = ρᵣ * cᵖᵈ * θΔcˡ

                    set!(model; θ = θΔcˡ)
                    ρe₁ = deepcopy(model.energy)

                    set!(model; ρe = ρeᵢ)
                    @test model.energy == ρe₁
                end
            end
        end
    end

    @testset "NonhydrostaticModel with MoistAirBuoyancy" begin
        reference_constants = ReferenceStateConstants(potential_temperature=300)
        buoyancy = MoistAirBuoyancy(; reference_constants)

        grid = RectilinearGrid(size=(8, 8, 8), x=(0, 400), y=(0, 400), z=(0, 400))
        model = NonhydrostaticModel(; grid, buoyancy, tracers = (:θ, :q))

        θΔcˡ = reference_constants.reference_potential_temperature
        Δθ = 2
        Lz = grid.Lz

        θᵢ(x, y, z) = θΔcˡ + Δθ * z / Lz
        set!(model; θ = θᵢ, q = 0)

        # Can time-step
        success = try
            time_step!(model, 1e-2)
            true
        catch
            false
        end

        @test success
    end
end
