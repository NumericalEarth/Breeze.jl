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
            @test @allowscalar model.energy_density ≈ ρe₁
        end
    end
end

@testset "PotentialTemperatureField (no microphysics) [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    thermo = ThermodynamicConstants(FT)

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, thermo, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    model = AtmosphereModel(grid; thermodynamics=thermo, formulation)

    # Initialize with potential temperature and dry air
    set!(model; θ=θ₀, qᵗ=0)

    θ_field = Breeze.AtmosphereModels.PotentialTemperatureField(model)
    compute!(θ_field)

    θ_expected = CenterField(grid)
    set!(θ_expected, θ₀)

    @test @allowscalar θ_field ≈ θ_expected
end

@testset "Saturation and PotentialTemperatureField (WarmPhase) [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    thermo = ThermodynamicConstants(FT)

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, thermo, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    microphysics = Breeze.Microphysics.WarmPhaseSaturationAdjustment(FT)
    model = AtmosphereModel(grid; thermodynamics=thermo, formulation, microphysics)

    # Initialize with potential temperature and dry air
    set!(model; θ=θ₀, qᵗ=0)

    # Check PotentialTemperatureField recovers θ₀
    θ_field = Breeze.AtmosphereModels.PotentialTemperatureField(model)
    compute!(θ_field)

    θ_expected = CenterField(grid)
    set!(θ_expected, θ₀)
    @test @allowscalar θ_field ≈ θ_expected

    # Check SaturationSpecificHumidityField matches direct thermodynamics
    q★ = Breeze.AtmosphereModels.SaturationSpecificHumidityField(model)
    compute!(q★)

    # Sample mid-level cell
    _, _, Nz = size(grid)
    k = max(1, Nz ÷ 2)

    Tᵢ = @allowscalar interior(model.temperature, 1, 1, k)[]
    pᵣᵢ = @allowscalar interior(model.formulation.reference_state.pressure, 1, 1, k)[]
    q = Breeze.Thermodynamics.MoistureMassFractions(zero(FT), zero(FT), zero(FT))
    ρᵢ = Breeze.Thermodynamics.density(pᵣᵢ, Tᵢ, q, thermo)
    q★exp = Breeze.Thermodynamics.saturation_specific_humidity(Tᵢ, ρᵢ, thermo, thermo.liquid)
    q★ᵢ = @allowscalar interior(q★, 1, 1, k)[]

    @test isfinite(q★ᵢ)
    @test q★ᵢ ≈ q★exp rtol=FT(1e-5)
end
