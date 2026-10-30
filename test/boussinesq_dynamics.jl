include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver
using Statistics: mean
using Test

momentum_divergence(model) = Field(KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, model.grid, model.momentum...))

@testset "BoussinesqReferenceState and BoussinesqDynamics [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
    constants = ThermodynamicConstants(FT)

    reference_state = BoussinesqReferenceState(grid, constants; pressure=9e4, temperature=280, vapor_mass_fraction=0.01)
    @test reference_state isa BoussinesqReferenceState
    @test reference_state.pressure[1, 2, 3] == FT(9e4)
    @test reference_state.temperature[4, 1, 2] == FT(280)
    @test reference_state.vapor_mass_fraction[2, 2, 2] == FT(0.01)

    qᵣ = Breeze.Thermodynamics.MoistureMassFractions(FT(0.01), zero(FT), zero(FT))
    Rᵐ = Breeze.Thermodynamics.mixture_gas_constant(qᵣ, constants)
    @test reference_state.density[3, 3, 3] ≈ FT(9e4) / (Rᵐ * FT(280))
    @test surface_density(reference_state) ≈ FT(9e4) / (Rᵐ * FT(280))

    # Hydrostatic reference states require a Bounded vertical direction
    bounded_grid = RectilinearGrid(default_arch, FT; size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    @test !(ReferenceState(bounded_grid, constants) isa BoussinesqReferenceState)

    dynamics = BoussinesqDynamics(grid, constants; temperature=280)
    @test dynamics isa BoussinesqDynamics
    @test dynamics isa AnelasticDynamics
    @test !(AnelasticDynamics(ReferenceState(bounded_grid, constants)) isa BoussinesqDynamics)
    @test startswith(summary(dynamics), "BoussinesqDynamics(pᵣ=101325.0, Tᵣ=280.0")

    # A Boussinesq reference state is not hydrostatic and cannot be recomputed as one
    @test_throws ArgumentError compute_reference_state!(reference_state, 280, 0, constants)
    @test_throws ArgumentError Breeze.Thermodynamics.compute_hydrostatic_reference!(reference_state, constants)

    model = AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid))
    @test_throws ArgumentError set_to_mean!(model.dynamics.reference_state, model)
end

@testset "BoussinesqDynamics pressure solvers [$FT]" for FT in test_float_types()
    periodic_grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), extent=(1, 1, 1),
                                    topology=(Periodic, Periodic, Periodic))

    bounded_grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), extent=(1, 1, 1),
                                   topology=(Periodic, Periodic, Bounded))

    @testset "Projection with $formulation formulation" for formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
        for (grid, Solver) in ((periodic_grid, FFTBasedPoissonSolver), (bounded_grid, FourierTridiagonalPoissonSolver))
            model = AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid), formulation)
            @test model.pressure_solver isa Solver

            ξ(x, y, z) = randn()
            set!(model, u=ξ, v=ξ, w=ξ)
            time_step!(model, 1e-3)

            δ = momentum_divergence(model)
            ρu = model.momentum.ρu
            @test maximum(abs, δ) < sqrt(eps(FT)) * maximum(abs, ρu) * size(grid, 1)
        end
    end
end

@testset "Decaying Taylor-Green vortex with BoussinesqDynamics [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=(32, 32, 4), extent=(2π, 2π, 1),
                           topology=(Periodic, Periodic, Periodic))

    ν = 0.05
    model = AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid), closure=ScalarDiffusivity(ν=ν))

    # Uniform temperature, so there is no buoyancy
    set!(model, u = (x, y, z) -> sin(x) * cos(y),
                v = (x, y, z) -> - cos(x) * sin(y))

    u₀ = maximum(abs, model.velocities.u)
    simulation = Simulation(model; Δt=0.01, stop_time=1, verbose=false)
    run!(simulation)

    # The vortex decays as exp(-2νt) without changing its shape
    t = model.clock.time
    @test maximum(abs, model.velocities.u) / u₀ ≈ exp(-2ν * t) rtol=1e-3
    @test maximum(abs, model.velocities.w) < sqrt(eps(FT))
end

@testset "BoussinesqDynamics agrees with AnelasticDynamics in a shallow domain [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=(32, 32), x=(0, 1), z=(0, 1), topology=(Periodic, Flat, Bounded))

    p₀ = 101325
    T₀ = 290
    boussinesq_dynamics = BoussinesqDynamics(grid; pressure=p₀, temperature=T₀)
    θ₀ = boussinesq_dynamics.reference_state.potential_temperature
    anelastic_dynamics = AnelasticDynamics(ReferenceState(grid; base_pressure=p₀, potential_temperature=θ₀))

    bubble(x, z) = θ₀ + exp(-((x - 0.5)^2 + (z - 0.3)^2) / 0.1^2) / 2

    vertical_velocities = map((boussinesq_dynamics, anelastic_dynamics)) do dynamics
        model = AtmosphereModel(grid; dynamics, closure=ScalarDiffusivity(ν=1e-4, κ=1e-4))
        set!(model, θ=bubble)
        simulation = Simulation(model; Δt=0.01, stop_iteration=50, verbose=false)
        run!(simulation)
        Array(interior(model.velocities.w))
    end

    wᴮ, wᴬ = vertical_velocities
    @test maximum(abs, wᴬ) > 1e-3
    @test maximum(abs, wᴮ .- wᴬ) / maximum(abs, wᴬ) < 1e-2
end

@testset "Moving reference frames [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
    constants = ThermodynamicConstants(FT)
    g = constants.gravitational_acceleration
    cᵖᵈ = constants.dry_air.heat_capacity

    @testset "Dry adiabatic motion with frame_vertical_velocity = $W" for W in (2, -2)
        dynamics = BoussinesqDynamics(grid; temperature=290, frame_vertical_velocity=W)
        @test occursin("frame_vertical_velocity=$(FT(W))", summary(dynamics))

        model = AtmosphereModel(grid; dynamics)
        frame = model.dynamics.reference_state.frame
        T₀ = mean(model.temperature)
        p₀ = frame.pressure
        ρ₀ = Breeze.Thermodynamics.reference_value(model.dynamics.reference_state.density)

        simulation = Simulation(model; Δt=0.1, stop_time=10, verbose=false)
        run!(simulation)

        # The domain cools (or warms) at the dry adiabatic lapse rate, g / cᵖᵈ, times W
        t = model.clock.time
        @test mean(model.temperature) - T₀ ≈ - g * W * t / cᵖᵈ rtol=1e-3
        @test frame.temperature ≈ mean(model.temperature)

        # The pressure changes hydrostatically
        @test frame.pressure - p₀ ≈ - ρ₀ * g * W * t rtol=2e-2
        @test frame.time ≈ t

        # Buoyancy is measured relative to the domain mean, so the domain does not accelerate vertically
        @test maximum(abs, model.velocities.w) < sqrt(eps(FT))
    end

    @testset "Condensation in a rising frame" begin
        dynamics = BoussinesqDynamics(grid; pressure=9e4, temperature=285, frame_vertical_velocity=2)
        model = AtmosphereModel(grid; dynamics, microphysics=SaturationAdjustment(FT))
        set!(model, qᵗ=0.0095)
        T₀ = mean(model.temperature)

        simulation = Simulation(model; Δt=0.1, stop_time=60, verbose=false)
        run!(simulation)

        # Latent heating slows the cooling of a saturated rising frame
        qˡ = Breeze.AtmosphereModels.liquid_mass_fraction(model)
        @test mean(qˡ) > 0
        @test mean(model.temperature) - T₀ > - g * 2 * model.clock.time / cᵖᵈ
    end

    @testset "A moving frame advances only when the model is time stepped" begin
        dynamics = BoussinesqDynamics(grid; temperature=290, frame_vertical_velocity=2)
        model = AtmosphereModel(grid; dynamics)
        frame = model.dynamics.reference_state.frame
        p₀ = frame.pressure

        # Setting the state at a later time does not move the frame
        set!(model; time=100, θ=300)
        @test frame.pressure == p₀
        @test frame.time == 100

        # Neither does changing the clock by hand before a simulation starts
        model.clock.time = 200
        Δt = FT(0.1)
        simulation = Simulation(model; Δt, stop_iteration=1, verbose=false)
        run!(simulation)

        Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
        @test frame.time ≈ 200 + Δt
        @test frame.pressure - p₀ ≈ - p₀ * g * 2 * Δt / (Rᵈ * frame.temperature) rtol=1e-2
    end

    @testset "Moving frames require the potential temperature formulation" begin
        dynamics = BoussinesqDynamics(grid; frame_vertical_velocity=2)
        @test_throws ArgumentError AtmosphereModel(grid; dynamics, formulation=:StaticEnergy)
    end

    @testset "Checkpointing moving frames" begin
        model = AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid; frame_vertical_velocity=2))
        for _ in 1:3
            time_step!(model, 0.1)
        end

        restored = AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid; frame_vertical_velocity=2))
        Oceananigans.restore_prognostic_state!(restored, Oceananigans.prognostic_state(model))

        frame = model.dynamics.reference_state.frame
        restored_frame = restored.dynamics.reference_state.frame
        @test restored_frame.pressure == frame.pressure
        @test restored_frame.time == frame.time
        @test restored_frame.temperature == frame.temperature
    end
end

@testset "A fixed frame that follows the domain mean removes the mean buoyancy [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))

    # A warm anomaly makes the domain-mean buoyancy nonzero relative to a fixed reference temperature
    warm(x, y, z) = 290 + exp(-((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2) / 0.1^2)

    mean_vertical_velocities = map((nothing, 0)) do frame_vertical_velocity
        dynamics = BoussinesqDynamics(grid; temperature=290, frame_vertical_velocity)
        model = AtmosphereModel(grid; dynamics, closure=ScalarDiffusivity(ν=1e-3, κ=1e-3))
        set!(model, θ=warm)
        simulation = Simulation(model; Δt=0.01, stop_iteration=50, verbose=false)
        run!(simulation)
        mean(model.velocities.w)
    end

    w̄_fixed_reference, w̄_mean_reference = mean_vertical_velocities
    @test abs(w̄_fixed_reference) > 1e-4
    @test abs(w̄_mean_reference) < 1e-3 * abs(w̄_fixed_reference)
end
