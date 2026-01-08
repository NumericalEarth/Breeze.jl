using Breeze
using Breeze: PrescribedDynamics, PrescribedVelocityFields, KinematicModel
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Fields: ZeroField
using Test

@testset "PrescribedDynamics construction [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 2000))
    reference_state = ReferenceState(grid, ThermodynamicConstants())

    dynamics = PrescribedDynamics(reference_state)
    @test dynamics.reference_state === reference_state
end

@testset "PrescribedVelocityFields construction" begin
    # Default (zero velocities)
    pvf = PrescribedVelocityFields()
    @test pvf.u isa ZeroField
    @test pvf.parameters === nothing

    # With function and parameters
    w_func(x, y, z, t, p) = p.w_max * sin(π * z / p.H)
    pvf = PrescribedVelocityFields(w=w_func, parameters=(; w_max=2, H=2000))
    @test pvf.w === w_func
    @test pvf.parameters.w_max == 2
end

@testset "KinematicModel with regular fields [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 2000))
    reference_state = ReferenceState(grid, ThermodynamicConstants())

    # Default velocities (regular fields, settable)
    model = AtmosphereModel(grid; dynamics=PrescribedDynamics(reference_state))
    @test model isa KinematicModel
    @test model.pressure_solver === nothing
    
    # Can set velocities
    set!(model, θ=300, qᵗ=0.01, w=1)
    @test @allowscalar(model.velocities.w[1, 1, 4]) ≈ FT(1)
    
    # Time stepping works
    time_step!(model, 1)
    @test model.clock.iteration == 1
end

@testset "KinematicModel with PrescribedVelocityFields [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), extent=(1000, 1000, 2000))
    reference_state = ReferenceState(grid, ThermodynamicConstants())

    # Time-dependent velocity function
    w_evolving(x, y, z, t) = (1 - exp(-t / 100)) * sin(π * z / 2000)
    
    model = AtmosphereModel(grid;
        dynamics = PrescribedDynamics(reference_state),
        velocities = PrescribedVelocityFields(w=w_evolving))
    
    @test model isa KinematicModel
    
    # Cannot set velocities (they're FunctionFields)
    set!(model, θ=300, qᵗ=0.01)
    @test_throws ArgumentError set!(model, w=1)
    
    # Time stepping works
    time_step!(model, 10)
    @test model.clock.time ≈ 10
end

@testset "KinematicModel momentum restriction [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 2000))
    model = AtmosphereModel(grid; dynamics=PrescribedDynamics(ReferenceState(grid, ThermodynamicConstants())))
    
    # No momentum in kinematic models
    @test_throws ArgumentError set!(model, ρu=1)
end

@testset "KinematicModel with microphysics [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), extent=(1000, 1000, 2000))
    
    model = AtmosphereModel(grid;
        dynamics = PrescribedDynamics(ReferenceState(grid, ThermodynamicConstants())),
        microphysics = SaturationAdjustment())
    
    set!(model, θ=300, qᵗ=0.015, w=2)
    time_step!(model, 1)
    @test model.clock.iteration == 1
end

@testset "Gaussian advection (analytical solution) [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    Lz, Nz, w₀ = 4000, 128, 10
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Lz), halo=(3, 3, 3))

    model = AtmosphereModel(grid;
        dynamics = PrescribedDynamics(ReferenceState(grid, ThermodynamicConstants())),
        tracers = :c,
        advection = WENO())

    # Analytical solution: Gaussian translating upward at speed w₀
    z₀, σ = 1000, 100
    c_exact(x, y, z, t) = exp(-(z - z₀ - w₀ * t)^2 / (2 * σ^2))

    set!(model, θ=300, qᵗ=0, w=w₀, c=(x, y, z) -> c_exact(x, y, z, 0))

    stop_time = 50
    simulation = Simulation(model; Δt=1, stop_time)
    run!(simulation)

    # Compare with analytical solution
    c_truth = CenterField(grid)
    set!(c_truth, (x, y, z) -> c_exact(x, y, z, stop_time))

    error = @allowscalar maximum(abs, interior(model.tracers.c) .- interior(c_truth))
    @test error < FT(0.05)
end
