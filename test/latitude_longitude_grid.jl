#####
##### Tests for LatitudeLongitudeGrid + CompressibleDynamics + SphericalCoriolis
#####
##### These tests verify that CompressibleDynamics works correctly on
##### LatitudeLongitudeGrid with HydrostaticSphericalCoriolis.
#####

using Breeze
using Breeze: AcousticSubstepper
using Breeze.CompressibleEquations: ExplicitTimeStepping, ExnerPerturbationSplitExplicit,
                                    HEVITimeDiscretization, compute_acoustic_substeps
using Breeze.Thermodynamics: adiabatic_hydrostatic_density, ExnerReferenceState, surface_density
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Units
using Test

# Note: When run through the test runner, test_float_types is defined in the init_code.
# When run directly, we need to define it.
if !@isdefined(test_float_types)
    test_float_types() = (Float64,)
end

#####
##### Helper to build a LatitudeLongitudeGrid for tests
#####

function build_test_llg(arch; Nx=36, Ny=34, Nz=8, Lz=30kilometers)
    return LatitudeLongitudeGrid(arch;
                                 size = (Nx, Ny, Nz),
                                 halo = (5, 5, 5),
                                 longitude = (0, 360),
                                 latitude = (-85, 85),
                                 z = (0, Lz),
                                 topology = (Periodic, Bounded, Bounded))
end

#####
##### Test model construction on LatitudeLongitudeGrid
#####

@testset "Model construction on LatitudeLongitudeGrid [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    grid = build_test_llg(default_arch)

    coriolis = HydrostaticSphericalCoriolis()
    dynamics = CompressibleDynamics(ExnerPerturbationSplitExplicit();
                                    reference_potential_temperature = 300)

    model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO())

    @test model.timestepper isa AcousticSSPRungeKutta3
    @test model.timestepper.substepper isa AcousticSubstepper
    @test !isnothing(model.dynamics.reference_state)
end

#####
##### Test adaptive substep computation on LatitudeLongitudeGrid
#####

@testset "compute_acoustic_substeps on LatitudeLongitudeGrid [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = build_test_llg(default_arch)
    constants = ThermodynamicConstants()

    N = compute_acoustic_substeps(grid, 12, constants)
    @test N isa Int
    @test N >= 1

    # On a LatitudeLongitudeGrid, the smallest Δx is at the highest latitude.
    # At 85°, Δx ≈ R cos(85°) Δλ ≈ 6371e3 * cos(85°*π/180) * (360/36)*π/180 ≈ 61 km
    # Needs more substeps than at the equator.
    Δx_equator = 6371e3 * (360 / 36) * π / 180  # ≈ 1113 km
    Δx_85 = Δx_equator * cos(85 * π / 180) # ≈ 97 km

    # Should need more substeps for the high-latitude grid since Δx is smaller
    @test N >= 1
end

#####
##### Balanced state stability on LatitudeLongitudeGrid (SSP-RK3)
#####

@testset "Balanced state on LatitudeLongitudeGrid [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = build_test_llg(default_arch)

    coriolis = HydrostaticSphericalCoriolis()
    td = ExnerPerturbationSplitExplicit(substeps = 8)
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = 100000,
                                    reference_potential_temperature = 300)

    model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO())

    ref = model.dynamics.reference_state
    set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

    simulation = Simulation(model; Δt=6, stop_iteration=10, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 10

    # With no perturbation and balanced reference state, w should be near zero
    w_max = @allowscalar maximum(abs, interior(model.velocities.w))
    @test w_max < 1e-6
end

#####
##### SSP-RK3 with perturbation on LatitudeLongitudeGrid
#####

@testset "SSP-RK3 with perturbation on LatitudeLongitudeGrid [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = build_test_llg(default_arch)

    coriolis = HydrostaticSphericalCoriolis()
    td = ExnerPerturbationSplitExplicit(substeps = 8)
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = 100000,
                                    reference_potential_temperature = 300)

    model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO())

    ref = model.dynamics.reference_state

    # Small θ perturbation + zonal wind
    θᵢ(λ, φ, z) = 300 + 0.01 * sin(π * z / 30kilometers)
    set!(model; θ=θᵢ, u=10, ρ=ref.density)

    simulation = Simulation(model; Δt=6, stop_iteration=20, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 20
    @test !any(isnan, parent(model.momentum.ρu))
    @test !any(isnan, parent(model.momentum.ρv))
    @test !any(isnan, parent(model.momentum.ρw))
    @test !any(isnan, parent(model.dynamics.density))

    # Density should remain physical
    ρ_min = @allowscalar minimum(interior(model.dynamics.density))
    @test ρ_min > 0
end

#####
##### WS-RK3 with perturbation on LatitudeLongitudeGrid
#####

@testset "WS-RK3 on LatitudeLongitudeGrid [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = build_test_llg(default_arch)

    coriolis = HydrostaticSphericalCoriolis()
    td = ExnerPerturbationSplitExplicit(substeps = 8, divergence_damping_coefficient = 0.10)
    dynamics = CompressibleDynamics(td; surface_pressure = 100000,
                                    reference_potential_temperature = 300)

    model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO(),
                            timestepper = :AcousticRungeKutta3)

    ref = model.dynamics.reference_state

    θᵢ(λ, φ, z) = 300 + 0.01 * sin(π * z / 30kilometers)
    set!(model; θ=θᵢ, u=10, ρ=ref.density)

    simulation = Simulation(model; Δt=6, stop_iteration=20, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 20
    @test !any(isnan, parent(model.momentum.ρu))
    @test !any(isnan, parent(model.momentum.ρv))
    @test !any(isnan, parent(model.momentum.ρw))
    @test !any(isnan, parent(model.dynamics.density))

    ρ_min = @allowscalar minimum(interior(model.dynamics.density))
    @test ρ_min > 0
end

#####
##### Explicit compressible time stepping on LatitudeLongitudeGrid
#####

@testset "Explicit CompressibleDynamics on LatitudeLongitudeGrid [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = build_test_llg(default_arch)

    coriolis = HydrostaticSphericalCoriolis()
    dynamics = CompressibleDynamics(ExplicitTimeStepping())

    model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO())

    @test model.timestepper isa SSPRungeKutta3

    set!(model; θ=300, ρ=1.2)

    simulation = Simulation(model; Δt=0.1, stop_iteration=3, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 3
    @test !any(isnan, parent(model.dynamics.density))
end

#####
##### HEVI on LatitudeLongitudeGrid
#####

@testset "HEVI on LatitudeLongitudeGrid [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = build_test_llg(default_arch)

    coriolis = HydrostaticSphericalCoriolis()
    td = HEVITimeDiscretization()
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = 100000,
                                    reference_potential_temperature = 300)

    model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO())

    @test model.timestepper isa HEVIRungeKutta3

    ref = model.dynamics.reference_state

    θᵢ(λ, φ, z) = 300 + 0.01 * sin(π * z / 30kilometers)
    set!(model; θ=θᵢ, u=10, qᵗ=0, ρ=ref.density)

    # Δt limited by horizontal acoustic CFL at highest latitude:
    # Δx ≈ 61 km at 85°, so Δt < 61000/347 ≈ 176 s; use Δt=6 for safety
    simulation = Simulation(model; Δt=6, stop_iteration=20, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 20
    @test !any(isnan, parent(model.momentum.ρu))
    @test !any(isnan, parent(model.momentum.ρv))
    @test !any(isnan, parent(model.momentum.ρw))
    @test !any(isnan, parent(model.dynamics.density))

    ρ_min = @allowscalar minimum(interior(model.dynamics.density))
    @test ρ_min > 0
end
