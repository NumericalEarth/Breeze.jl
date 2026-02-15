#####
##### Tests for acoustic substepping in CompressibleDynamics
#####
##### These tests verify that the AcousticSSPRungeKutta3 and AcousticRungeKutta3
##### time steppers produce stable, correct results with the Exner pressure
##### acoustic substepping formulation.
#####

using Breeze
using Breeze: AcousticSubstepper
using Breeze.CompressibleEquations: ExplicitTimeStepping, SplitExplicitTimeDiscretization
using Breeze.Thermodynamics: adiabatic_hydrostatic_density
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Units
using Statistics: mean
using Test

# Note: When run through the test runner, test_float_types is defined in the init_code.
# When run directly, we need to define it.
if !@isdefined(test_float_types)
    test_float_types() = (Float64,)
end

const acoustic_test_arch = Oceananigans.Architectures.CPU()

#####
##### Test AcousticSubstepper construction
#####

@testset "AcousticSubstepper construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

    @testset "Default construction" begin
        td = SplitExplicitTimeDiscretization()
        acoustic = AcousticSubstepper(grid, td)
        @test acoustic.substeps == 8
        @test acoustic.forward_weight ≈ FT(0.6)
        @test acoustic.divergence_damping_coefficient ≈ FT(0.10)
        @test acoustic.π′ isa Oceananigans.Fields.Field
        @test acoustic.θᵥ isa Oceananigans.Fields.Field
        @test acoustic.ppterm isa Oceananigans.Fields.Field
    end

    @testset "Custom parameters" begin
        td = SplitExplicitTimeDiscretization(substeps=10,
                                              forward_weight=0.55,
                                              divergence_damping_coefficient=0.2)
        acoustic = AcousticSubstepper(grid, td)
        @test acoustic.substeps == 10
        @test acoustic.forward_weight ≈ FT(0.55)
        @test acoustic.divergence_damping_coefficient ≈ FT(0.2)
    end
end

#####
##### Test time stepper construction
#####

@testset "AcousticSSPRungeKutta3 construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

    dynamics = CompressibleDynamics()
    model = AtmosphereModel(grid;
                            dynamics,
                            timestepper=:AcousticSSPRungeKutta3)

    @test model.timestepper isa AcousticSSPRungeKutta3
    @test model.timestepper.substepper isa AcousticSubstepper
    @test model.timestepper.α¹ ≈ FT(1)
    @test model.timestepper.α² ≈ FT(1//4)
    @test model.timestepper.α³ ≈ FT(2//3)
end

@testset "AcousticRungeKutta3 construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

    dynamics = CompressibleDynamics()
    model = AtmosphereModel(grid;
                            dynamics,
                            timestepper=:AcousticRungeKutta3)

    @test model.timestepper isa AcousticRungeKutta3
    @test model.timestepper.substepper isa AcousticSubstepper
    @test model.timestepper.β₁ ≈ FT(1//3)
    @test model.timestepper.β₂ ≈ FT(1//2)
    @test model.timestepper.β₃ ≈ FT(1)
end

#####
##### Test that default time stepper for split-explicit is WS-RK3
#####

@testset "Default time stepper for SplitExplicitTimeDiscretization [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

    dynamics = CompressibleDynamics()
    model = AtmosphereModel(grid; dynamics)

    @test model.timestepper isa AcousticRungeKutta3
end

#####
##### Test that models with acoustic substepping run without NaN
#####

@testset "SSP-RK3 model runs without NaN [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(8, 8, 8), halo=(5, 5, 5),
                           x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers))

    dynamics = CompressibleDynamics(; reference_potential_temperature=300)
    model = AtmosphereModel(grid;
                            advection=WENO(),
                            dynamics,
                            timestepper=:AcousticSSPRungeKutta3)

    ref = model.dynamics.reference_state
    set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

    simulation = Simulation(model; Δt=6, stop_iteration=5)
    run!(simulation)

    @test model.clock.iteration == 5
    @test !any(isnan, parent(model.momentum.ρu))
    @test !any(isnan, parent(model.momentum.ρw))
    @test !any(isnan, parent(model.dynamics.density))
end

@testset "WS-RK3 model runs without NaN [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(8, 8, 8), halo=(5, 5, 5),
                           x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers))

    dynamics = CompressibleDynamics(; reference_potential_temperature=300)
    model = AtmosphereModel(grid;
                            advection=WENO(),
                            dynamics,
                            timestepper=:AcousticRungeKutta3)

    ref = model.dynamics.reference_state
    set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

    simulation = Simulation(model; Δt=6, stop_iteration=5)
    run!(simulation)

    @test model.clock.iteration == 5
    @test !any(isnan, parent(model.momentum.ρu))
    @test !any(isnan, parent(model.momentum.ρw))
    @test !any(isnan, parent(model.dynamics.density))
end

#####
##### SK94 inertia-gravity wave stability test
#####
##### Run the IGW benchmark for a short time with both time steppers
##### at advection-limited Δt=12 to verify the acoustic substepping is stable.
#####

function build_igw_model(; timestepper=:AcousticSSPRungeKutta3, Ns=8, kdiv=0.05)
    Nx, Ny, Nz = 100, 6, 10
    Lx, Ly, Lz = 100kilometers, 6kilometers, 10kilometers

    grid = RectilinearGrid(acoustic_test_arch; size=(Nx, Ny, Nz), halo=(5, 5, 5),
                           x=(0, Lx), y=(0, Ly), z=(0, Lz))

    p₀ = 100000
    θ₀ = 300
    U  = 20
    N² = 0.01^2

    constants = ThermodynamicConstants()
    g  = constants.gravitational_acceleration

    θᵇᵍ(z) = θ₀ * exp(N² * z / g)

    Δθ = 0.01
    a  = 5000
    x₀ = Lx / 3
    θᵢ(x, y, z) = θᵇᵍ(z) + Δθ * sin(π * z / Lz) / (1 + (x - x₀)^2 / a^2)

    td = SplitExplicitTimeDiscretization(substeps=Ns, divergence_damping_coefficient=kdiv)
    dynamics = CompressibleDynamics(; surface_pressure=p₀,
                                      reference_potential_temperature=θᵇᵍ,
                                      time_discretization=td)

    model = AtmosphereModel(grid; advection=WENO(), dynamics, timestepper)

    ref = model.dynamics.reference_state
    set!(model; θ=θᵢ, u=U, qᵗ=0, ρ=ref.density)

    return model
end

@testset "IGW stability: SSP-RK3 (Δt=12, Ns=8) [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    model = build_igw_model(timestepper=:AcousticSSPRungeKutta3, Ns=8, kdiv=0.05)

    simulation = Simulation(model; Δt=12, stop_iteration=20)
    run!(simulation)

    @test model.clock.iteration == 20
    @test !any(isnan, parent(model.dynamics.density))
    @test !any(isnan, parent(model.momentum.ρw))

    # max|w| should remain bounded (the IGW problem has max|w| ~ 0.003 at t=3000s)
    w_max = @allowscalar maximum(abs, interior(model.velocities.w))
    @test w_max < 1.0  # Should be O(0.001), definitely < 1 m/s

    # Density should remain physical
    ρ_min = @allowscalar minimum(interior(model.dynamics.density))
    @test ρ_min > 0
end

@testset "IGW stability: WS-RK3 (Δt=12, Ns=8) [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    model = build_igw_model(timestepper=:AcousticRungeKutta3, Ns=8, kdiv=0.10)

    simulation = Simulation(model; Δt=12, stop_iteration=20)
    run!(simulation)

    @test model.clock.iteration == 20
    @test !any(isnan, parent(model.dynamics.density))
    @test !any(isnan, parent(model.momentum.ρw))

    # max|w| should remain bounded
    w_max = @allowscalar maximum(abs, interior(model.velocities.w))
    @test w_max < 1.0

    # Density should remain physical
    ρ_min = @allowscalar minimum(interior(model.dynamics.density))
    @test ρ_min > 0
end

#####
##### Test balanced state stability (no perturbation → near-zero motion)
#####

@testset "Balanced state stays quiet [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nx, Ny, Nz = 16, 8, 10
    grid = RectilinearGrid(acoustic_test_arch; size=(Nx, Ny, Nz), halo=(5, 5, 5),
                           x=(0, 16kilometers), y=(0, 8kilometers), z=(0, 10kilometers))

    td = SplitExplicitTimeDiscretization(substeps=8)
    dynamics = CompressibleDynamics(; surface_pressure=100000,
                                      reference_potential_temperature=300,
                                      time_discretization=td)

    model = AtmosphereModel(grid; advection=WENO(), dynamics)

    ref = model.dynamics.reference_state
    set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

    simulation = Simulation(model; Δt=12, stop_iteration=10)
    run!(simulation)

    @test model.clock.iteration == 10

    # With no perturbation and balanced reference state, w should be near zero
    w_max = @allowscalar maximum(abs, interior(model.velocities.w))
    @test w_max < 1e-6  # Should be at machine precision level
end
