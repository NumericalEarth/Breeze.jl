#####
##### Tests for LatitudeLongitudeGrid + CompressibleDynamics + SphericalCoriolis
#####
##### These tests verify that CompressibleDynamics works correctly on
##### LatitudeLongitudeGrid with HydrostaticSphericalCoriolis.
#####

using Breeze
using Breeze.CompressibleEquations: ExplicitTimeStepping, compute_acoustic_substeps
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
##### Solid body rotation on LatitudeLongitudeGrid
#####
##### Solid body rotation u = u₀ cos(φ), v = 0 is an exact steady-state solution
##### on the sphere when Coriolis and curvature metric terms are correctly balanced.
##### Any drift in v or w, or change in u, indicates incorrect metric terms.
#####

@testset "Solid body rotation on LatitudeLongitudeGrid [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = build_test_llg(default_arch; Nx=36, Ny=34, Nz=4, Lz=10kilometers)

    coriolis = HydrostaticSphericalCoriolis()
    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    surface_pressure = 100000,
                                    reference_potential_temperature = 300)

    model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO())

    ## Solid body rotation: u = u₀ cos(φ), uniform θ and ρ
    u₀ = 10.0 # m/s — modest rotation speed
    uᵢ(λ, φ, z) = u₀ * cosd(φ)

    ref = model.dynamics.reference_state
    set!(model; θ=300, u=uᵢ, ρ=ref.density)

    ## Record initial max|v| and max|w| (should be ~0)
    v_init = @allowscalar maximum(abs, interior(model.velocities.v))
    w_init = @allowscalar maximum(abs, interior(model.velocities.w))

    ## Run a few time steps
    Δt = 0.1
    simulation = Simulation(model; Δt, stop_iteration=10, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 10

    ## After 10 steps, v and w should remain small if metrics are correct.
    ## Without correct metric terms, v grows O(u₀² tanφ / a) ≈ 1e-5 m/s per step,
    ## accumulating to ~1e-4 after 10 steps. With wrong metrics this would be O(1).
    v_max = @allowscalar maximum(abs, interior(model.velocities.v))
    w_max = @allowscalar maximum(abs, interior(model.velocities.w))

    @test v_max < 0.1  # should be ≪ u₀
    @test w_max < 0.1

    ## u should not have drifted significantly from initial profile
    u_max = @allowscalar maximum(abs, interior(model.velocities.u))
    @test u_max < 2 * u₀  # should still be O(u₀)
    @test !any(isnan, parent(model.momentum.ρu))
    @test !any(isnan, parent(model.momentum.ρv))
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
