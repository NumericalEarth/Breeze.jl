using Adapt: adapt
using Breeze
using Breeze: PrescribedDensity, PrescribedDynamics, KinematicModel
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, OpenBoundaryCondition
using Oceananigans.Fields: ZeroField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedVelocityFields
using Test

@testset "KinematicDriver [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 2000))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants)

    @testset "PrescribedDynamics construction" begin
        dynamics = PrescribedDynamics(reference_state)
        @test dynamics.density isa PrescribedDensity
        @test dynamics_density(dynamics) === reference_state.density

        dynamics_div = PrescribedDynamics(reference_state; divergence_correction=true)
        @test dynamics_div isa PrescribedDynamics{true}
    end

    @testset "PrescribedDensity adapt and on_architecture" begin
        pd = PrescribedDensity(reference_state.density)

        # Test adapt_structure
        adapted_pd = adapt(CPU(), pd)
        @test adapted_pd isa PrescribedDensity

        # Test on_architecture
        transferred_pd = on_architecture(CPU(), pd)
        @test transferred_pd isa PrescribedDensity
    end

    @testset "KinematicModel with prognostic density" begin
        ρ = CenterField(grid)
        set!(ρ, FT(1))
        model = AtmosphereModel(grid; dynamics=PrescribedDynamics(ρ))
        @test haskey(Oceananigans.prognostic_fields(model), :ρ)
    end

    @testset "KinematicModel with regular fields" begin
        model = AtmosphereModel(grid; dynamics=PrescribedDynamics(reference_state))
        @test model isa KinematicModel
        @test model.pressure_solver === nothing
        
        set!(model, θ=300, qᵗ=0.01, w=1)
        @test @allowscalar(model.velocities.w[1, 1, 4]) ≈ FT(1)
        
        time_step!(model, 1)
        @test model.clock.iteration == 1
    end

    @testset "KinematicModel with PrescribedVelocityFields" begin
        w_evolving(x, y, z, t) = (1 - exp(-t / 100)) * sin(π * z / 2000)
        
        model = AtmosphereModel(grid;
            dynamics = PrescribedDynamics(reference_state),
            velocities = PrescribedVelocityFields(w=w_evolving))
        
        @test model isa KinematicModel
        set!(model, θ=300, qᵗ=0.01)
        @test_throws ArgumentError set!(model, w=1)
        
        time_step!(model, 10)
        @test model.clock.time ≈ 10
    end

    @testset "Velocity boundary conditions" begin
        w_inlet(x, y, t) = FT(0.5)
        w_bcs = FieldBoundaryConditions(bottom=OpenBoundaryCondition(w_inlet))
        boundary_conditions = (; w = w_bcs)
        
        model = AtmosphereModel(grid; dynamics=PrescribedDynamics(reference_state), boundary_conditions)
        @test model isa KinematicModel
        @test model.velocities.w.boundary_conditions.bottom isa Oceananigans.BoundaryConditions.BoundaryCondition

        # AnelasticDynamics does not allow velocity boundary conditions
        @test_throws ArgumentError AtmosphereModel(grid; boundary_conditions)
    end
end

@testset "Gaussian advection (analytical solution) [Float64]" begin
    FT = Float64
    Oceananigans.defaults.FloatType = FT
    
    Lz, Nz, w₀ = 4000, 64, 10  # Reduced resolution for faster test
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Lz))
    
    model = AtmosphereModel(grid;
        dynamics = PrescribedDynamics(ReferenceState(grid, ThermodynamicConstants())),
        tracers = :c,
        advection = WENO())

    z₀, σ = 1000, 100
    c_exact(x, y, z, t) = exp(-(z - z₀ - w₀ * t)^2 / (2 * σ^2))

    set!(model, θ=300, qᵗ=0, w=w₀, c=(x, y, z) -> c_exact(x, y, z, 0))

    stop_time = 50
    simulation = Simulation(model; Δt=1, stop_time)
    run!(simulation)

    c_truth = CenterField(grid)
    set!(c_truth, (x, y, z) -> c_exact(x, y, z, stop_time))

    error = @allowscalar maximum(abs, interior(model.tracers.c) .- interior(c_truth))
    @test error < FT(0.1)  # Relaxed tolerance for reduced resolution test
end
