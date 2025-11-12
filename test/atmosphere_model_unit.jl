using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

@testset "AtmosphereModel [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    thermo = ThermodynamicConstants(FT)

    for p₀ in (101325, 100000), θ₀ in (288, 300)
        @testset let p₀ = p₀, θ₀ = θ₀
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
    θᵢ = CenterField(grid)
    set!(θᵢ, (x, y, z) -> θ₀ + rand())
    set!(model; θ=θᵢ)

    θ_model = Breeze.AtmosphereModels.PotentialTemperatureField(model)
    @test @allowscalar θ_model ≈ θᵢ
end

@testset "Saturation and PotentialTemperatureField (WarmPhase) [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    thermo = ThermodynamicConstants(FT)

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, thermo, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    microphysics = SaturationAdjustment()
    model = AtmosphereModel(grid; thermodynamics=thermo, formulation, microphysics)

    # Initialize with potential temperature and dry air
    set!(model; θ=θ₀)

    # Check SaturationSpecificHumidityField matches direct thermodynamics
    q★ = Breeze.AtmosphereModels.SaturationSpecificHumidityField(model)
    compute!(q★)

    # Sample mid-level cell
    _, _, Nz = size(grid)
    k = max(1, Nz ÷ 2)

    Tᵢ = @allowscalar interior(model.temperature, 1, 1, k)[]
    pᵣᵢ = @allowscalar interior(model.formulation.reference_state.pressure, 1, 1, k)[]
    q = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρᵢ = Breeze.Thermodynamics.density(pᵣᵢ, Tᵢ, q, thermo)
    qᵛ⁺_expected = Breeze.Thermodynamics.saturation_specific_humidity(Tᵢ, ρᵢ, thermo, thermo.liquid)
    qᵛ⁺k = @allowscalar interior(q★, 1, 1, k)[]

    @test isfinite(qᵛ⁺k)
    @test qᵛ⁺k ≈ qᵛ⁺_expected rtol=FT(1e-5)
end
