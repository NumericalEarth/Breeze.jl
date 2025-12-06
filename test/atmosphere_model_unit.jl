using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

@testset "AtmosphereModel [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants(FT)

    for p₀ in (101325, 100000), θ₀ in (288, 300), thermodynamics in (:LiquidIcePotentialTemperature, :StaticEnergy)
        @testset let p₀ = p₀, θ₀ = θ₀, thermodynamics = thermodynamics
            reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
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

@testset "liquid_ice_potential_temperature no microphysics) [$(FT)]" for FT in (Float32, Float64), thermodynamics in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state; thermodynamics)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation)

    # Initialize with potential temperature and dry air
    θᵢ = CenterField(grid)
    set!(θᵢ, (x, y, z) -> θ₀ + rand())
    set!(model; θ=θᵢ)

    θ_model = liquid_ice_potential_temperature(model) |> Field
    @test θ_model ≈ θᵢ
end

@testset "Saturation and LiquidIcePotentialTemperatureField (WarmPhase) [$(FT)]" for FT in (Float32, Float64), thermodynamics in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state; thermodynamics)
    microphysics = SaturationAdjustment()
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation, microphysics)

    # Initialize with potential temperature and dry air
    set!(model; θ=θ₀)

    # Check SaturationSpecificHumidityField matches direct thermodynamics
    qᵛ⁺ = SaturationSpecificHumidityField(model)

    # Sample mid-level cell
    _, _, Nz = size(grid)
    k = max(1, Nz ÷ 2)

    Tᵢ = @allowscalar model.temperature[1, 1, k]
    pᵣᵢ = @allowscalar model.formulation.reference_state.pressure[1, 1, k]
    q = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρᵢ = Breeze.Thermodynamics.density(pᵣᵢ, Tᵢ, q, constants)
    qᵛ⁺_expected = Breeze.Thermodynamics.saturation_specific_humidity(Tᵢ, ρᵢ, constants, constants.liquid)
    qᵛ⁺k = @allowscalar qᵛ⁺[1, 1, k]

    @test isfinite(qᵛ⁺k)
    @test qᵛ⁺k ≈ qᵛ⁺_expected rtol=FT(1e-5)
end

@testset "Advection scheme configuration [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
    potential_temperature_formulation = AnelasticFormulation(reference_state; thermodynamics=:LiquidIcePotentialTemperature)
    static_energy_formulation = AnelasticFormulation(reference_state; thermodynamics=:StaticEnergy)

    @testset "Default advection schemes" begin
        static_energy_model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation=static_energy_formulation)
        potential_temperature_model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation=potential_temperature_formulation)

        @test static_energy_model.advection.ρe isa Centered
        @test potential_temperature_model.advection.ρθ isa Centered

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa Centered
            @test model.advection.ρqᵗ isa Centered
            time_step!(model, 1)
        end
    end

    @testset "Unified advection parameter" begin
        static_energy_model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation=static_energy_formulation, advection=WENO())
        potential_temperature_model= AtmosphereModel(grid; thermodynamic_constants=constants, formulation=potential_temperature_formulation, advection=WENO())

        @test static_energy_model.advection.ρe isa WENO
        @test potential_temperature_model.advection.ρθ isa WENO

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa WENO
            @test model.advection.ρqᵗ isa WENO
            time_step!(model, 1)
        end
    end

    @testset "Separate momentum and tracer advection" begin
        kw = (thermodynamic_constants=constants, momentum_advection = WENO(), scalar_advection = Centered(order=2))
        static_energy_model = AtmosphereModel(grid; formulation=static_energy_formulation, kw...)
        potential_temperature_model = AtmosphereModel(grid; formulation=potential_temperature_formulation, kw...)

        @test static_energy_model.advection.ρe isa Centered
        @test potential_temperature_model.advection.ρθ isa Centered

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa WENO
            @test model.advection.ρqᵗ isa Centered
            time_step!(model, 1)
        end
    end

    @testset "Tracer advection with user tracers" begin
        kw = (thermodynamic_constants=constants, tracers = :c, scalar_advection = UpwindBiased(order=1))
        static_energy_model = AtmosphereModel(grid; formulation=static_energy_formulation, kw...)
        potential_temperature_model = AtmosphereModel(grid; formulation=potential_temperature_formulation, kw...)

        @test static_energy_model.advection.ρe isa UpwindBiased
        @test potential_temperature_model.advection.ρθ isa UpwindBiased

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa Centered
            @test model.advection.ρqᵗ isa UpwindBiased
            @test model.advection.c isa UpwindBiased
            time_step!(model, 1)
        end
    end

    @testset "Mixed configuration with tracers" begin
        scalar_advection = (; c=Centered(order=2), ρqᵗ=WENO())
        kw = (thermodynamic_constants=constants, tracers = :c, momentum_advection = WENO(), scalar_advection)
        static_energy_model = AtmosphereModel(grid; formulation=static_energy_formulation, kw...)
        potential_temperature_model = AtmosphereModel(grid; formulation=potential_temperature_formulation, kw...)

        @test static_energy_model.advection.ρe isa Centered
        @test potential_temperature_model.advection.ρθ isa Centered

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa WENO
            @test model.advection.ρqᵗ isa WENO
            @test model.advection.c isa Centered
            time_step!(model, 1)
        end
    end
end
