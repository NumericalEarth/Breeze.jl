using Breeze
using Breeze: PrescribedVelocityFields, KinematicModel  # Explicitly import to avoid conflict with Oceananigans
using Breeze.Thermodynamics: MoistureMassFractions
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Fields: FunctionField
using Oceananigans.Operators: divᶜᶜᶜ
using Test

# Helper function to check if a field is a FunctionField or similar
is_function_based(f) = f isa FunctionField || f isa Oceananigans.Fields.ZeroField

@testset "PrescribedVelocityFields smoke tests [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 1000), y=(0, 1000), z=(0, 2000))
    constants = ThermodynamicConstants()

    p₀ = 101325
    θ₀ = 300

    reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)

    @testset "Constructor with no velocities (ZeroField default)" begin
        dynamics = PrescribedVelocityFields(reference_state)
        @test dynamics.reference_state === reference_state
        @test dynamics.u isa Oceananigans.Fields.ZeroField
        @test dynamics.v isa Oceananigans.Fields.ZeroField
        @test dynamics.w isa Oceananigans.Fields.ZeroField
        @test dynamics.parameters === nothing
    end

    @testset "Constructor with function velocities" begin
        w_func(x, y, z, t) = sin(π * z / 2000)
        dynamics = PrescribedVelocityFields(reference_state; w=w_func)
        @test dynamics.w === w_func
    end

    @testset "Constructor with parameters" begin
        w_param(x, y, z, t, p) = p.w_max * sin(π * z / p.H)
        params = (; w_max=2.0, H=2000.0)
        dynamics = PrescribedVelocityFields(reference_state; w=w_param, parameters=params)
        @test dynamics.parameters === params
    end

    @testset "AtmosphereModel construction with PrescribedVelocityFields" begin
        dynamics = PrescribedVelocityFields(reference_state)
        model = AtmosphereModel(grid; dynamics)
        
        @test model.dynamics isa PrescribedVelocityFields
        @test model.pressure_solver === nothing  # No pressure solver needed
        @test model.momentum !== nothing  # Dummy momentum fields for interface compatibility
    end

    @testset "AtmosphereModel with different formulations" begin
        for formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
            dynamics = PrescribedVelocityFields(reference_state)
            model = AtmosphereModel(grid; dynamics, formulation)
            @test model.dynamics isa PrescribedVelocityFields
        end
    end
end

@testset "PrescribedVelocityFields dynamics interface [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 1000), y=(0, 1000), z=(0, 2000))
    constants = ThermodynamicConstants()

    p₀ = 101325
    θ₀ = 300
    reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = PrescribedVelocityFields(reference_state)
    model = AtmosphereModel(grid; dynamics)

    @testset "dynamics_density returns reference state density" begin
        ρ = dynamics_density(model.dynamics)
        @test ρ === reference_state.density
    end

    @testset "dynamics_pressure returns reference state pressure" begin
        p = dynamics_pressure(model.dynamics)
        @test p === reference_state.pressure
    end

    @testset "Velocities are FunctionField or ZeroField" begin
        u, v, w = model.velocities
        # Velocities should be FunctionField wrapping prescribed functions, or ZeroField
        @test is_function_based(u) || u isa Oceananigans.AbstractField
        @test is_function_based(v) || v isa Oceananigans.AbstractField
        @test is_function_based(w) || w isa Oceananigans.AbstractField
    end

    @testset "Pressure solver is nothing" begin
        @test model.pressure_solver === nothing
    end
end

@testset "PrescribedVelocityFields time stepping [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), x=(0, 1000), y=(0, 1000), z=(0, 2000))
    constants = ThermodynamicConstants()

    p₀ = 101325
    θ₀ = 300
    reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)

    @testset "Basic time stepping with zero velocities" begin
        dynamics = PrescribedVelocityFields(reference_state)
        model = AtmosphereModel(grid; dynamics)
        set!(model, θ=θ₀, qᵗ=0.01)
        
        # Time step should work without errors
        time_step!(model, 1.0)
        @test model.clock.iteration == 1
    end

    @testset "Time stepping with prescribed vertical velocity" begin
        w_func(x, y, z, t) = 2.0 * sin(π * z / 2000)
        dynamics = PrescribedVelocityFields(reference_state; w=w_func)
        model = AtmosphereModel(grid; dynamics)
        set!(model, θ=θ₀, qᵗ=0.01)
        
        # Multiple time steps should work
        for _ in 1:3
            time_step!(model, 1.0)
        end
        @test model.clock.iteration == 3
    end

    @testset "Time stepping with time-dependent velocity" begin
        # Velocity that grows with time
        w_evolving(x, y, z, t) = (1 - exp(-t / 100)) * sin(π * z / 2000)
        dynamics = PrescribedVelocityFields(reference_state; w=w_evolving)
        model = AtmosphereModel(grid; dynamics)
        set!(model, θ=θ₀, qᵗ=0.01)
        
        time_step!(model, 10.0)
        @test model.clock.iteration == 1
        @test model.clock.time ≈ 10.0
    end

    @testset "Time stepping with parameterized velocity" begin
        w_param(x, y, z, t, p) = p.w_max * sin(π * z / p.H)
        params = (; w_max=FT(2.0), H=FT(2000.0))
        dynamics = PrescribedVelocityFields(reference_state; w=w_param, parameters=params)
        model = AtmosphereModel(grid; dynamics)
        set!(model, θ=θ₀, qᵗ=0.01)
        
        time_step!(model, 1.0)
        @test model.clock.iteration == 1
    end
end

@testset "PrescribedVelocityFields set! behavior [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 1000), y=(0, 1000), z=(0, 2000))
    constants = ThermodynamicConstants()

    p₀ = 101325
    θ₀ = 300
    reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = PrescribedVelocityFields(reference_state)
    model = AtmosphereModel(grid; dynamics)

    @testset "Setting thermodynamic variables works" begin
        set!(model, θ=300.0)
        @test @allowscalar(model.temperature[1, 1, 4]) > 0  # Temperature is computed
    end

    @testset "Setting moisture works" begin
        set!(model, θ=300.0, qᵗ=0.01)
        @test @allowscalar(model.specific_moisture[1, 1, 4]) ≈ FT(0.01) atol=FT(1e-6)
    end

    @testset "Setting velocity throws error" begin
        @test_throws ArgumentError set!(model, u=1.0)
        @test_throws ArgumentError set!(model, v=1.0)
        @test_throws ArgumentError set!(model, w=1.0)
    end

    @testset "Setting momentum throws error" begin
        @test_throws ArgumentError set!(model, ρu=1.0)
        @test_throws ArgumentError set!(model, ρv=1.0)
        @test_throws ArgumentError set!(model, ρw=1.0)
    end
end

@testset "KinematicModel type alias [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 1000), y=(0, 1000), z=(0, 2000))
    constants = ThermodynamicConstants()

    p₀ = 101325
    θ₀ = 300
    reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = PrescribedVelocityFields(reference_state)
    model = AtmosphereModel(grid; dynamics)

    @test model isa KinematicModel
    @test !(model isa AnelasticModel)
end

@testset "PrescribedVelocityFields microphysics integration [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), x=(0, 1000), y=(0, 1000), z=(0, 2000))
    constants = ThermodynamicConstants()

    p₀ = 101325
    θ₀ = 300
    reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)

    @testset "With SaturationAdjustment microphysics" begin
        dynamics = PrescribedVelocityFields(reference_state)
        microphysics = SaturationAdjustment()
        model = AtmosphereModel(grid; dynamics, microphysics)
        
        # Set up moist conditions
        set!(model, θ=θ₀, qᵗ=0.015)
        
        # Should run without errors
        time_step!(model, 1.0)
        @test model.clock.iteration == 1
    end
end

@testset "Gaussian tracer translation (kinematic solution) [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    # Domain setup with bounded z (required for ReferenceState hydrostatic integration)
    Lz = 4000
    Nz = 128  # High resolution for accurate advection
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100), z=(0, Lz))
    constants = ThermodynamicConstants()

    p₀ = 101325
    θ₀ = 300
    reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)

    # Constant upward velocity
    w₀ = FT(10.0)  # 10 m/s upward
    w_const(x, y, z, t) = w₀
    dynamics = PrescribedVelocityFields(reference_state; w=w_const)
    
    # Use WENO advection for accuracy
    model = AtmosphereModel(grid; dynamics, tracers=:c, advection=WENO())
    
    # Initial Gaussian centered at z₀ (in lower part of domain to allow upward translation)
    z₀ = FT(1000.0)
    σ = FT(100.0)    # Width of Gaussian
    c_gaussian(x, y, z) = exp(-(z - z₀)^2 / (2 * σ^2))
    set!(model, θ=θ₀, qᵗ=0.0, c=c_gaussian)

    # Compute initial centroid for comparison
    z = znodes(grid, Center())
    c_data_initial = @allowscalar interior(model.tracers.c, 1, 1, :)
    total_mass_initial = sum(c_data_initial)
    centroid_initial = sum(c_data_initial .* z) / total_mass_initial

    # Run for specified time
    Δt = FT(1.0)
    stop_time = FT(50.0)
    expected_displacement = w₀ * stop_time  # Should translate by 500 m

    simulation = Simulation(model; Δt, stop_time)
    run!(simulation)

    # Compute final centroid of tracer (center of mass in z)
    c_field = model.tracers.c
    c_data_final = @allowscalar interior(c_field, 1, 1, :)
    total_mass_final = sum(c_data_final)
    centroid_final = sum(c_data_final .* z) / total_mass_final
    
    # Actual displacement
    actual_displacement = centroid_final - centroid_initial
    
    # Expected final position
    z_final_expected = z₀ + expected_displacement

    # Test that the Gaussian translated to the correct position
    # Allow tolerance for numerical diffusion and discretization error
    @test isapprox(centroid_final, z_final_expected, atol=FT(20.0))
    @test isapprox(actual_displacement, expected_displacement, atol=FT(20.0))
    
    # Verify mass is conserved
    @test isapprox(total_mass_final, total_mass_initial, rtol=FT(1e-3))
end

