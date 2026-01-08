using Breeze
using Breeze: PrescribedDynamics, KinematicModel
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Fields: ZeroField
using Test

@testset "PrescribedDynamics construction [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 2000))
    reference_state = ReferenceState(grid, ThermodynamicConstants())

    # Default constructor (zero velocities)
    dynamics = PrescribedDynamics(reference_state)
    @test dynamics.reference_state === reference_state
    @test dynamics.u isa ZeroField
    @test dynamics.parameters === nothing

    # With velocity function and parameters
    w_func(x, y, z, t, p) = p.w_max * sin(π * z / p.H)
    dynamics = PrescribedDynamics(reference_state; w=w_func, parameters=(; w_max=2, H=2000))
    @test dynamics.w === w_func
    @test dynamics.parameters.w_max == 2
end

@testset "KinematicModel construction and interface [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 2000))
    reference_state = ReferenceState(grid, ThermodynamicConstants())

    for formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
        model = AtmosphereModel(grid; dynamics=PrescribedDynamics(reference_state), formulation)
        @test model isa KinematicModel
        @test !(model isa AnelasticModel)
        @test model.pressure_solver === nothing
        @test dynamics_density(model.dynamics) === reference_state.density
        @test dynamics_pressure(model.dynamics) === reference_state.pressure
    end
end

@testset "KinematicModel time stepping [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), extent=(1000, 1000, 2000))
    reference_state = ReferenceState(grid, ThermodynamicConstants())

    # Zero velocity
    model = AtmosphereModel(grid; dynamics=PrescribedDynamics(reference_state))
    set!(model, θ=300, qᵗ=0.01)
    time_step!(model, 1)
    @test model.clock.iteration == 1

    # Constant velocity
    w_const(x, y, z, t) = 2
    model = AtmosphereModel(grid; dynamics=PrescribedDynamics(reference_state; w=w_const))
    set!(model, θ=300, qᵗ=0.01)
    for _ in 1:3
        time_step!(model, 1)
    end
    @test model.clock.iteration == 3

    # Time-dependent velocity
    w_evolving(x, y, z, t) = (1 - exp(-t / 100)) * sin(π * z / 2000)
    model = AtmosphereModel(grid; dynamics=PrescribedDynamics(reference_state; w=w_evolving))
    set!(model, θ=300, qᵗ=0.01)
    time_step!(model, 10)
    @test model.clock.time ≈ 10
end

@testset "KinematicModel set! restrictions [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 2000))
    model = AtmosphereModel(grid; dynamics=PrescribedDynamics(ReferenceState(grid, ThermodynamicConstants())))

    set!(model, θ=300, qᵗ=0.01)
    @test @allowscalar(model.specific_moisture[1, 1, 4]) ≈ FT(0.01) atol=FT(1e-6)

    # Velocity/momentum setting throws
    @test_throws ArgumentError set!(model, u=1)
    @test_throws ArgumentError set!(model, ρu=1)
end

@testset "KinematicModel with microphysics [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), extent=(1000, 1000, 2000))
    reference_state = ReferenceState(grid, ThermodynamicConstants())

    model = AtmosphereModel(grid;
        dynamics = PrescribedDynamics(reference_state),
        microphysics = SaturationAdjustment())
    
    set!(model, θ=300, qᵗ=0.015)
    time_step!(model, 1)
    @test model.clock.iteration == 1
end

@testset "Gaussian advection (analytical solution) [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    # halo=3 required for WENO (size must be ≥ halo)
    Lz, Nz = 4000, 128
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Lz), halo=(3, 3, 3))
    reference_state = ReferenceState(grid, ThermodynamicConstants())

    # Constant upward velocity
    w₀ = 10
    w_const(x, y, z, t) = w₀
    
    model = AtmosphereModel(grid;
        dynamics = PrescribedDynamics(reference_state; w=w_const),
        tracers = :c,
        advection = WENO())

    # Analytical solution: Gaussian translating upward at speed w₀
    z₀, σ = 1000, 100
    c_exact(x, y, z, t) = exp(-(z - z₀ - w₀ * t)^2 / (2 * σ^2))

    c_initial(x, y, z) = c_exact(x, y, z, 0)
    set!(model, θ=300, qᵗ=0, c=c_initial)

    stop_time = 50
    simulation = Simulation(model; Δt=1, stop_time)
    run!(simulation)

    # Compare with analytical solution
    c_truth = CenterField(grid)
    c_truth_func(x, y, z) = c_exact(x, y, z, stop_time)
    set!(c_truth, c_truth_func)

    c_numerical = model.tracers.c
    error = @allowscalar maximum(abs, interior(c_numerical) .- interior(c_truth))
    @test error < FT(0.05)
end
