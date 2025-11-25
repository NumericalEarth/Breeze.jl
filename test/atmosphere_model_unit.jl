using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

# TODO: move this to Oceananigans
function constant_field(grid, constant) 
    field = Field{Nothing, Nothing, Nothing}(grid)
    return set!(field, constant)
end

@testset "set! AtmosphereModel [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    thermo = ThermodynamicConstants()
    @test eltype(thermo) == FT

    for p₀ in (101325, 100000), θ₀ in (288, 300), microphysics in (nothing, SaturationAdjustment())
        @testset let p₀ = p₀, θ₀ = θ₀, microphysics = microphysics
            reference_state = ReferenceState(grid, thermo, base_pressure=p₀, potential_temperature=θ₀)
            formulation = AnelasticFormulation(reference_state)
            model = AtmosphereModel(grid; thermodynamics=thermo, formulation, microphysics)
            
            set!(model; qᵗ = 1e-2)
            @test @allowscalar model.moisture_mass_fraction ≈ constant_field(grid, 1e-2)
            
            ρᵣ = model.formulation.reference_state.density
            @test @allowscalar model.moisture_density ≈ ρᵣ * 1e-2

            set!(model; u = 1, v = 2)
            @test @allowscalar model.velocities.u ≈ constant_field(grid, 1)
            @test @allowscalar model.velocities.v ≈ constant_field(grid, 2)
            @test @allowscalar model.momentum.ρu ≈ ρᵣ
            @test @allowscalar model.momentum.ρv ≈ ρᵣ * 2
            
            ρᵣ = model.formulation.reference_state.density
            @test @allowscalar model.moisture_density ≈ ρᵣ * 1e-2
            
            # test set! for a dry initial state
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

@testset "PotentialTemperatureField (no microphysics) [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    thermo = ThermodynamicConstants()

    p₀, θ₀ = 101325, 300
    reference_state = ReferenceState(grid, thermo, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    model = AtmosphereModel(grid; thermodynamics=thermo, formulation)

    # Initialize with potential temperature and dry air
    θᵢ = CenterField(grid)
    set!(θᵢ, (x, y, z) -> θ₀ + rand())
    set!(model; θ=θᵢ)

    θ_model = Breeze.AtmosphereModels.PotentialTemperatureField(model)
    @test θ_model ≈ θᵢ
end

@testset "Saturation and PotentialTemperatureField (WarmPhase) [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    thermo = ThermodynamicConstants()

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, thermo, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    microphysics = SaturationAdjustment()
    model = AtmosphereModel(grid; thermodynamics=thermo, formulation, microphysics)

    # Initialize with potential temperature and dry air
    set!(model; θ=θ₀)

    # Check SaturationSpecificHumidityField matches direct thermodynamics
    qᵛ⁺ = Breeze.AtmosphereModels.SaturationSpecificHumidityField(model)

    # Sample mid-level cell
    _, _, Nz = size(grid)
    k = max(1, Nz ÷ 2)

    Tᵢ = @allowscalar model.temperature[1, 1, k]
    pᵣᵢ = @allowscalar model.formulation.reference_state.pressure[1, 1, k]
    q = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρᵢ = Breeze.Thermodynamics.density(pᵣᵢ, Tᵢ, q, thermo)
    qᵛ⁺_expected = Breeze.Thermodynamics.saturation_specific_humidity(Tᵢ, ρᵢ, thermo, thermo.liquid)
    qᵛ⁺k = @allowscalar qᵛ⁺[1, 1, k]

    @test isfinite(qᵛ⁺k)
    @test qᵛ⁺k ≈ qᵛ⁺_expected rtol=FT(1e-5)
end
