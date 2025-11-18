using Breeze
using Breeze.AtmosphereModels: compute_thermodynamic_state
using Breeze.Microphysics: MixedPhaseEquilibrium
using Breeze.Thermodynamics:
    MoistureMassFractions,
    mixture_heat_capacity
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
            @test model.energy_density ≈ ρe₁
        end
    end
end

@testset "PotentialTemperatureField (no microphysics) [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    thermo = ThermodynamicConstants()

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
    @test θ_model ≈ θᵢ
end

@testset "Saturation and PotentialTemperatureField (WarmPhase) [$(FT)]" for FT in (Float32, Float64)
    if default_arch isa GPU && FT == Float32
        # skip
    else
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
end

@testset "Thermodynamics consistency (MixedPhase) [$(FT)]" for FT in (Float32, Float64)
    if default_arch isa GPU && FT == Float32
        # skip
    else
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
        thermo = ThermodynamicConstants()

        p₀ = FT(101325)
        θ₀_ref = FT(288) 
        θ₀(z) = 245 - z/1000*6.5 # 
        qᵗ₀ = FT(0.02)

        reference_state = ReferenceState(grid, thermo, base_pressure=p₀, potential_temperature=θ₀_ref)
        formulation = AnelasticFormulation(reference_state)
        equilibrium = MixedPhaseEquilibrium(FT)
        microphysics = SaturationAdjustment(FT; equilibrium)
        model = AtmosphereModel(grid; thermodynamics=thermo, formulation, microphysics)

        θ_field = CenterField(grid)
        set!(θ_field, (x, y, z) -> θ₀(z))

        qᵗ_field = CenterField(grid)
        set!(qᵗ_field, qᵗ₀)

        q₀ = MoistureMassFractions(qᵗ₀)
        cᵖᵐ₀ = mixture_heat_capacity(q₀, thermo)
        
        # First test with moisture
        set!(model; qᵗ=qᵗ_field, θ=θ_field)

        θ_model = Breeze.AtmosphereModels.PotentialTemperatureField(model)
        compute!(θ_model)

        tol = max(sqrt(eps(FT)), FT(1e-5))
        θ_matches = Ref(true)

        @allowscalar begin
            for k in 1:size(grid, 3), j in 1:size(grid, 2), i in 1:size(grid, 1)
                θ_val = θ_model[i, j, k]
                θ_ref = θ_field[i, j, k]
                θ_matches[] &= isapprox(θ_val, θ_ref; rtol=tol, atol=tol)
            end
        end
        @test θ_matches[]

        # Now test with dry air
        set!(model; qᵗ=0, θ=θ_field)
        compute!(θ_model)

        @allowscalar begin
            for k in 1:size(grid, 3), j in 1:size(grid, 2), i in 1:size(grid, 1)
                θ_val = θ_model[i, j, k]
                θ_ref = θ_field[i, j, k]
                θ_matches[] &= isapprox(θ_val, θ_ref; rtol=tol, atol=tol)
            end
        end
        @test θ_matches[]


    end
end
