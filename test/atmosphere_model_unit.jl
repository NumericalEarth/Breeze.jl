using Breeze
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

    for p₀ in (101325, 100000), θ₀ in (288, 300), thermodynamics in (:LiquidIcePotentialTemperature, :StaticEnergy)
        @testset let p₀ = p₀, θ₀ = θ₀, thermodynamics = thermodynamics
            reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)

            # Check that interpolating to the first face (k=1) recovers surface values
            q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
            ρ₀ = Breeze.Thermodynamics.density(p₀, θ₀, q₀, constants)
            for i = 1:Nx, j = 1:Ny
                @test p₀ ≈ @allowscalar ℑzᵃᵃᶠ(i, j, 1, grid, reference_state.pressure)
                @test ρ₀ ≈ @allowscalar ℑzᵃᵃᶠ(i, j, 1, grid, reference_state.density)
            end

            dynamics = AnelasticDynamics(reference_state)
            thermodynamic_formulation = thermodynamics == :StaticEnergy ? StaticEnergyFormulation(nothing, nothing) : LiquidIcePotentialTemperatureFormulation(nothing, nothing)
            model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, thermodynamic_formulation)

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

@testset "liquid_ice_potential_temperature no microphysics) [$(FT)]" for FT in (Float32, Float64), thermodynamics in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    thermodynamic_formulation = thermodynamics == :StaticEnergy ? StaticEnergyFormulation(nothing, nothing) : LiquidIcePotentialTemperatureFormulation(nothing, nothing)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, thermodynamic_formulation)

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
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    thermodynamic_formulation = thermodynamics == :StaticEnergy ? StaticEnergyFormulation(nothing, nothing) : LiquidIcePotentialTemperatureFormulation(nothing, nothing)
    microphysics = SaturationAdjustment()
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, thermodynamic_formulation, microphysics)

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
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)

    @testset "Default advection schemes" begin
        static_energy_model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics)
        potential_temperature_model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, 
                                                      thermodynamic_formulation=LiquidIcePotentialTemperatureFormulation(nothing, nothing))

        @test static_energy_model.advection.ρe isa Centered
        @test potential_temperature_model.advection.ρθ isa Centered

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa Centered
            @test model.advection.ρqᵗ isa Centered
            time_step!(model, 1)
        end
    end

    @testset "Unified advection parameter" begin
        static_energy_model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                              dynamics, advection=WENO())

        potential_temperature_model= AtmosphereModel(grid; thermodynamic_constants=constants,
                                                     dynamics, thermodynamic_formulation=LiquidIcePotentialTemperatureFormulation(nothing, nothing), advection=WENO())

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
        static_energy_model = AtmosphereModel(grid; dynamics, kw...)
        potential_temperature_model = AtmosphereModel(grid; dynamics, thermodynamic_formulation=LiquidIcePotentialTemperatureFormulation(nothing, nothing), kw...)

        @test static_energy_model.advection.ρe isa Centered
        @test potential_temperature_model.advection.ρθ isa Centered

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa WENO
            @test model.advection.ρqᵗ isa Centered
            time_step!(model, 1)
        end
    end

    @testset "FluxFormAdvection for momentum and tracers" begin
        advection = FluxFormAdvection(WENO(), WENO(), Centered(order=2))
        kw = (; thermodynamic_constants=constants, advection)
        static_energy_model = AtmosphereModel(grid; dynamics, kw...)
        potential_temperature_model = AtmosphereModel(grid; dynamics, thermodynamic_formulation=LiquidIcePotentialTemperatureFormulation(nothing, nothing), kw...)

        @test static_energy_model.advection.ρe isa FluxFormAdvection
        @test static_energy_model.advection.ρe.x isa WENO
        @test static_energy_model.advection.ρe.y isa WENO
        @test static_energy_model.advection.ρe.z isa Centered

        @test potential_temperature_model.advection.ρθ isa FluxFormAdvection
        @test potential_temperature_model.advection.ρθ.x isa WENO
        @test potential_temperature_model.advection.ρθ.y isa WENO
        @test potential_temperature_model.advection.ρθ.z isa Centered

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa FluxFormAdvection
            @test model.advection.ρqᵗ isa FluxFormAdvection
            @test model.advection.ρqᵗ.x isa WENO
            @test model.advection.ρqᵗ.y isa WENO
            @test model.advection.ρqᵗ.z isa Centered
            time_step!(model, 1)
        end
    end

    @testset "Tracer advection with user tracers" begin
        kw = (thermodynamic_constants=constants, tracers = :c, scalar_advection = UpwindBiased(order=1))
        static_energy_model = AtmosphereModel(grid; dynamics, kw...)
        potential_temperature_model = AtmosphereModel(grid; dynamics, thermodynamic_formulation=LiquidIcePotentialTemperatureFormulation(nothing, nothing), kw...)

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
        static_energy_model = AtmosphereModel(grid; dynamics, kw...)
        potential_temperature_model = AtmosphereModel(grid; dynamics, thermodynamic_formulation=LiquidIcePotentialTemperatureFormulation(nothing, nothing), kw...)

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

@testset "Setting temperature directly [$(FT), $(thermodynamics)]" for FT in (Float32, Float64), thermodynamics in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 10), x=(0, 1_000), y=(0, 1_000), z=(0, 5_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101500)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    thermodynamic_formulation = thermodynamics == :StaticEnergy ? StaticEnergyFormulation(nothing, nothing) : LiquidIcePotentialTemperatureFormulation(nothing, nothing)

    # Test with no microphysics first (no saturation adjustment effects)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, thermodynamic_formulation)

    # Set a standard lapse rate temperature profile with dry air
    T_profile(x, y, z) = FT(300) - FT(0.0065) * z

    set!(model, T=T_profile, qᵗ=FT(0))  # dry air

    # Check that temperature was set correctly (should match for dry air)
    z_nodes = Oceananigans.Grids.znodes(grid, Center())
    for k in 1:10
        T_expected = T_profile(0, 0, z_nodes[k])
        T_actual = @allowscalar model.temperature[1, 1, k]
        @test T_actual ≈ T_expected rtol=FT(1e-4)
    end

    # Check that potential temperature increases with height (stable atmosphere)
    θ = liquid_ice_potential_temperature(model) |> Field
    θ_prev = @allowscalar θ[1, 1, 1]
    for k in 2:10
        θ_k = @allowscalar θ[1, 1, k]
        @test θ_k > θ_prev  # potential temperature should increase with height
        θ_prev = θ_k
    end

    # Test round-trip consistency: set T, get θ; set θ back, get same T
    set!(model, T=FT(280), qᵗ=FT(0))
    T_after_set = @allowscalar model.temperature[2, 2, 5]
    @test T_after_set ≈ FT(280) rtol=FT(1e-4)

    # Now test with saturation adjustment
    microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())
    model_moist = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, thermodynamic_formulation, microphysics)

    # Set T with subsaturated moisture (no condensate expected)
    set!(model_moist, T=T_profile, qᵗ=FT(0.001))  # low moisture

    # Temperature should still be close to input for subsaturated air
    T_actual = @allowscalar model_moist.temperature[1, 1, 1]
    T_expected = T_profile(0, 0, z_nodes[1])
    @test T_actual ≈ T_expected rtol=FT(0.02)  # allow 2% tolerance due to moisture effects
end
