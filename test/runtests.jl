using Test
using Breeze
using Oceananigans

Ξ(x, y, z) = rand()

@testset "Breeze.jl" begin
    @testset "Thermodynamics" begin
        thermo = AtmosphereThermodynamics()

        # Test saturation specific humidity calculation
        T = 293.15  # 20°C
        ρ = 1.2     # kg/m³
        q★ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, thermo.condensation)
        @test q★ > 0
    end

    @testset "AtmosphereModel" begin
        for FT in (Float32, Float64)
            grid = RectilinearGrid(FT, size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
            thermo = AtmosphereThermodynamics(FT)

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

    @testset "NonhydrostaticModel with MoistAirBuoyancy" begin
        reference_constants = ReferenceStateConstants(potential_temperature=300.0)
        buoyancy = MoistAirBuoyancy(; reference_constants)

        grid = RectilinearGrid(size=(8, 8, 8), x=(0, 400), y=(0, 400), z=(0, 400))
        model = NonhydrostaticModel(; grid, buoyancy, tracers = (:θ, :q))

        θ₀ = reference_constants.reference_potential_temperature
        Δθ = 2.0
        Lz = grid.Lz

        θᵢ(x, y, z) = θ₀ + Δθ * z / Lz
        set!(model; θ = θᵢ, q = 0)

        T_field = TemperatureField(model)
        compute!(T_field)

        θ_data = Array(interior(model.tracers.θ))
        T_data = Array(interior(T_field))

        Rᵐ = Breeze.Thermodynamics.mixture_gas_constant(0.0, buoyancy.thermodynamics)
        cᵖᵐ = Breeze.Thermodynamics.mixture_heat_capacity(0.0, buoyancy.thermodynamics)
        p_ref = 1e5
        c_loc = Oceananigans.Grids.Center()

        for k in axes(T_data, 3)
            z = Oceananigans.Grids.znode(1, 1, k, grid, c_loc, c_loc, c_loc)
            pᵣ = Breeze.Thermodynamics.reference_pressure(z, reference_constants, buoyancy.thermodynamics)
            Π = (pᵣ / p_ref)^(Rᵐ / cᵖᵐ)
            expected = Π .* θ_data[:, :, k]
            @test isapprox(T_data[:, :, k], expected; atol=1e-6, rtol=1e-6)
        end
    end
end
