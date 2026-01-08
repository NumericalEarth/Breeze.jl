using Breeze
using Breeze.Thermodynamics: TetensFormula
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Diagnostics: erroring_NaNChecker!
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Test

# TODO: move this to Oceananigans
function constant_field(grid, constant)
    field = Field{Nothing, Nothing, Nothing}(grid)
    return set!(field, constant)
end

function run_nan_checker_test(arch; erroring)
    grid = RectilinearGrid(arch, size=(4, 2, 1), extent=(1, 1, 1))
    model = AtmosphereModel(grid)
    simulation = Simulation(model, Δt=1, stop_iteration=2)
    @allowscalar model.momentum.ρu[1, 1, 1] = NaN
    erroring && erroring_NaNChecker!(simulation)

    if erroring
        @test_throws ErrorException run!(simulation)
    else
        run!(simulation)
        @test model.clock.iteration == 1 # simulation stopped after one iteration
    end

    return nothing
end

@testset "AtmosphereModel [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    Nx = Ny = 3
    grid = RectilinearGrid(default_arch; size=(Nx, Ny, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    model = AtmosphereModel(grid)
    @test model.grid === grid

    @testset "Basic tests for set!" begin
        set!(model, time=1)
        @test model.clock.time == 1
    end

    @testset "NaN Checker" begin
        @info "  Testing NaN Checker..."
        run_nan_checker_test(default_arch, erroring=true)
        run_nan_checker_test(default_arch, erroring=false)
    end

    constants = ThermodynamicConstants()
    @test eltype(constants) == FT

    p₀, θ₀ = 101325, 300

    @testset "ReferenceState surface values" begin
        reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)

        # Check that interpolating to the first face (k=1) recovers surface values
        q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
        ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, constants)
        for i = 1:Nx, j = 1:Ny
            @test p₀ ≈ @allowscalar ℑzᵃᵃᶠ(i, j, 1, grid, reference_state.pressure)
            @test ρ₀ ≈ @allowscalar ℑzᵃᵃᶠ(i, j, 1, grid, reference_state.density)
        end
    end

    for formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
        @testset "set! and thermodynamic roundtrip [$formulation]" begin
            reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
            dynamics = AnelasticDynamics(reference_state)
            model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation)

            # Test round-trip consistency: set θ, get ρe; then set ρe, get back θ
            set!(model; θ = θ₀)
            ρe₁ = Field(static_energy_density(model))
            θ₁ = Field(liquid_ice_potential_temperature(model))

            set!(model; ρe = ρe₁)
            @test static_energy_density(model) ≈ ρe₁
            @test liquid_ice_potential_temperature(model) ≈ θ₁
        end
    end

    for microphysics in (nothing, SaturationAdjustment())
        @testset "set! moisture and momentum [microphysics=$(typeof(microphysics).name.name)]" begin
            reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
            dynamics = AnelasticDynamics(reference_state)
            model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, microphysics)
            
            set!(model; qᵗ = 1e-2)
            @test model.specific_moisture ≈ constant_field(grid, 1e-2)
            
            ρᵣ = model.dynamics.reference_state.density
            @test model.moisture_density ≈ ρᵣ * 1e-2

            set!(model; u = 1, v = 2)
            @test model.velocities.u ≈ constant_field(grid, 1)
            @test model.velocities.v ≈ constant_field(grid, 2)
            @test model.momentum.ρu ≈ ρᵣ
            @test model.momentum.ρv ≈ ρᵣ * 2
        end
    end
end

@testset "Saturation specific humidity [$(FT)]" for FT in (Float32, Float64), formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀, θ₀ = 101325, 300
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment()
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation, microphysics)

    set!(model; θ=θ₀)

    # Check SaturationSpecificHumidityField matches direct thermodynamics
    qᵛ⁺ = SaturationSpecificHumidityField(model)

    # Sample mid-level cell
    _, _, Nz = size(grid)
    k = max(1, Nz ÷ 2)

    Tᵢ = @allowscalar model.temperature[1, 1, k]
    pᵣᵢ = @allowscalar model.dynamics.reference_state.pressure[1, 1, k]
    q = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρᵢ = Breeze.Thermodynamics.density(Tᵢ, pᵣᵢ, q, constants)
    qᵛ⁺_expected = Breeze.Thermodynamics.saturation_specific_humidity(Tᵢ, ρᵢ, constants, constants.liquid)
    qᵛ⁺k = @allowscalar qᵛ⁺[1, 1, k]

    @test isfinite(qᵛ⁺k)
    @test qᵛ⁺k ≈ qᵛ⁺_expected rtol=FT(1e-5)
end

@testset "AtmosphereModel with TetensFormula [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    tetens = TetensFormula()
    constants = ThermodynamicConstants(; saturation_vapor_pressure=tetens)

    p₀, θ₀ = 101325, 300
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)

    for formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation)
        set!(model; θ=θ₀)
        time_step!(model, 1)
        @test model.clock.iteration == 1
    end
end
