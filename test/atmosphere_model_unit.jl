using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

@testset "AtmosphereModel [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants(FT)

    for p₀ in (101325, 100000), θ₀ in (288, 300)
        @testset let p₀ = p₀, θ₀ = θ₀
            reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
            formulation = AnelasticFormulation(reference_state)
            model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation)

            # test set!
            ρᵣ = model.formulation.reference_state.density
            cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
            ρeᵢ = ρᵣ * cᵖᵈ * θ₀

            set!(model; θ = θ₀)
            ρe₁ = deepcopy(static_energy_density(model))

            set!(model; ρe = ρeᵢ)
            @test static_energy_density(model) ≈ ρe₁
        end
    end
end

@testset "LiquidIcePotentialTemperatureField (no microphysics) [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation)

    # Initialize with potential temperature and dry air
    θᵢ = CenterField(grid)
    set!(θᵢ, (x, y, z) -> θ₀ + rand())
    set!(model; θ=θᵢ)

    θ_model = potential_temperature(model) |> Field
    @test θ_model ≈ θᵢ
end

@testset "Saturation and LiquidIcePotentialTemperatureField (WarmPhase) [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
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
    formulation = AnelasticFormulation(reference_state)

    @testset "Default advection schemes" begin
        model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation)
        @test model.advection.momentum isa Centered
        @test model.advection.ρe isa Centered
        @test model.advection.ρqᵗ isa Centered

        time_step!(model, 1)
        @test true
    end

    @testset "Unified advection parameter" begin
        model_weno = AtmosphereModel(grid; thermodynamic_constants=constants, formulation, advection=WENO())
        @test model_weno.advection.momentum isa WENO
        @test model_weno.advection.ρe isa WENO
        @test model_weno.advection.ρqᵗ isa WENO
        time_step!(model_weno, 1)
        @test true

        model_centered = AtmosphereModel(grid; thermodynamic_constants=constants, formulation, advection=Centered(order=4))
        @test model_centered.advection.momentum isa Centered
        @test model_centered.advection.ρe isa Centered
        time_step!(model_centered, 1)
        @test true
    end

    @testset "Separate momentum and tracer advection" begin
        model = AtmosphereModel(grid; 
                                thermodynamic_constants = constants, 
                                formulation,
                                momentum_advection = WENO(),
                                scalar_advection = Centered(order=2))
        @test model.advection.momentum isa WENO
        @test model.advection.ρe isa Centered
        @test model.advection.ρqᵗ isa Centered
        time_step!(model, 1)
        @test true
    end

    @testset "Tracer advection with user tracers" begin
        model = AtmosphereModel(grid; 
                                thermodynamic_constants = constants, 
                                formulation,
                                tracers = :c,
                                scalar_advection = UpwindBiased(order=1))
        @test model.advection.momentum isa Centered
        @test model.advection.ρe isa UpwindBiased
        @test model.advection.ρqᵗ isa UpwindBiased
        @test model.advection.c isa UpwindBiased
        time_step!(model, 1)
        @test true
    end

    @testset "Mixed configuration with tracers" begin
        model = AtmosphereModel(grid; 
                                thermodynamic_constants = constants, 
                                formulation,
                                tracers = :c,
                                momentum_advection = WENO(),
                                scalar_advection = Centered(order=2))
        @test model.advection.momentum isa WENO
        @test model.advection.ρe isa Centered
        @test model.advection.ρqᵗ isa Centered
        @test model.advection.c isa Centered
        time_step!(model, 1)
        @test true
    end
end
