using Breeze
using Breeze.Thermodynamics: TetensFormula
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Diagnostics: erroring_NaNChecker!
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Test

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

@testset "AtmosphereModel [$(FT)]" for FT in test_float_types()
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

    for p₀ in (101325, 100000), θ₀ in (288, 300), formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
        @testset let p₀ = p₀, θ₀ = θ₀, formulation = formulation
            reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)

            # Check that interpolating to the first face (k=1) approximately recovers surface values.
            # Uses rtol=1e-4 because GradientBoundaryCondition is used for discrete hydrostatic balance,
            # which introduces small interpolation errors at the surface.
            ρ₀ = surface_density(reference_state)
            for i = 1:Nx, j = 1:Ny
                @test isapprox(p₀, @allowscalar(ℑzᵃᵃᶠ(i, j, 1, grid, reference_state.pressure)), rtol=FT(1e-4))
                @test isapprox(ρ₀, @allowscalar(ℑzᵃᵃᶠ(i, j, 1, grid, reference_state.density)), rtol=FT(1e-4))
            end

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
end

@testset "Saturation and LiquidIcePotentialTemperatureField (WarmPhase) [$(FT)]" for FT in test_float_types(), formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀ = 101325
    θ₀ = 300
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment()
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation, microphysics)

    # Initialize with potential temperature and dry air
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

@testset "AtmosphereModel with TetensFormula [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    tetens = TetensFormula()
    constants = ThermodynamicConstants(; saturation_vapor_pressure=tetens)

    p₀ = 101325
    θ₀ = 300
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)

    for formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation)
        set!(model; θ=θ₀)
        time_step!(model, 1)
        @test model.clock.iteration == 1
    end
end
