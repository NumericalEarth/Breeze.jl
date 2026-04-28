#####
##### Tests for acoustic substepping in CompressibleDynamics
#####
##### These tests verify that the AcousticRungeKutta3 (WS-RK3)
##### time steppers produce stable, correct results with the Exner pressure
##### acoustic substepping formulation.
#####

using Breeze
using Breeze: AcousticSubstepper
using Breeze.CompressibleEquations: ExplicitTimeStepping, SplitExplicitTimeDiscretization,
                                    compute_acoustic_substeps
using Breeze.AtmosphereModels: SlowTendencyMode, HorizontalSlowMode,
                               x_pressure_gradient, y_pressure_gradient, z_pressure_gradient,
                               buoyancy_forceᶜᶜᶜ, dynamics_density
using Breeze.Thermodynamics: adiabatic_hydrostatic_density, ExnerReferenceState, surface_density
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

    @testset "Default construction (adaptive substeps)" begin
        td = SplitExplicitTimeDiscretization()
        acoustic = AcousticSubstepper(grid, td)
        @test acoustic.substeps === nothing  # adaptive by default
        @test acoustic.forward_weight ≈ FT(0.65)  # off-centered CN, ε = 2ω - 1 = 0.3
        # Default damping is KlempDivergenceDamping(0.1) (Klemp/Skamarock/Ha 2018
        # with Baldauf 2010 anisotropic scaling); required for stability of the
        # WS-RK3 + substepper coupling at production Δt.
        @test acoustic.damping isa KlempDivergenceDamping
        @test acoustic.damping.coefficient ≈ FT(0.1)
        @test acoustic.outer_step_potential_temperature isa Oceananigans.Fields.Field
    end

    @testset "Custom parameters" begin
        td = SplitExplicitTimeDiscretization(substeps=10,
                                              forward_weight=0.55,
                                              damping=KlempDivergenceDamping(coefficient=0.2))
        acoustic = AcousticSubstepper(grid, td)
        @test acoustic.substeps == 10
        @test acoustic.forward_weight ≈ FT(0.55)
        @test acoustic.damping isa KlempDivergenceDamping
        @test acoustic.damping.coefficient ≈ FT(0.2)
    end
end

#####
##### Test adaptive substep computation
#####

@testset "compute_acoustic_substeps [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    constants = ThermodynamicConstants()

    @testset "1 km grid, Δt=12" begin
        grid = RectilinearGrid(acoustic_test_arch; size=(100, 6, 10), halo=(5, 5, 5),
                               x=(0, 100kilometers), y=(0, 6kilometers), z=(0, 10kilometers))
        # Δx = 1000 m, ℂᵃᶜ ≈ 347 m/s, safety factor 2.0 → acoustic CFL ≈ 0.5 (ERF/WRF target)
        # N = ceil(2.0 * 12 * 347 / 1000) = ceil(8.33) = 9
        N = compute_acoustic_substeps(grid, 12, constants)
        @test N isa Int
        @test N ≥ 1
        @test N == ceil(Int, 2.0 * 12 * sqrt(1.4 * 287.0 * 300) / 1000)
    end

    @testset "Flat y-topology" begin
        grid = RectilinearGrid(acoustic_test_arch; size=(100, 10), halo=(5, 5),
                               x=(0, 100kilometers), z=(0, 10kilometers),
                               topology=(Periodic, Flat, Bounded))
        # Should use only Δx, not Δy
        N = compute_acoustic_substeps(grid, 12, constants)
        N_expected = ceil(Int, 2.0 * 12 * sqrt(1.4 * 287.0 * 300) / 1000)
        @test N == N_expected
    end
end

#####
##### Test time stepper construction
#####

@testset "AcousticRungeKutta3 construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization())
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
##### Test that default time stepper for split-explicit is AcousticRungeKutta3 (WS-RK3)
#####

@testset "Default time stepper for SplitExplicitTimeDiscretization [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization())
    model = AtmosphereModel(grid; dynamics)

    @test model.timestepper isa AcousticRungeKutta3
end

#####
##### Test that models with acoustic substepping run without NaN
#####

@testset "WS-RK3 model runs without NaN [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(8, 8, 8), halo=(5, 5, 5),
                           x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                   reference_potential_temperature=300)
    model = AtmosphereModel(grid;
                            advection=WENO(),
                            dynamics,
                            timestepper=:AcousticRungeKutta3)

    ref = model.dynamics.reference_state
    set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

    simulation = Simulation(model; Δt=6, stop_iteration=5, verbose=false)
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

function build_igw_model(; timestepper=:AcousticRungeKutta3, Ns=8, κᵈ=0.05)
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

    td = SplitExplicitTimeDiscretization(substeps=Ns,
                                         damping=ThermodynamicDivergenceDamping(coefficient=κᵈ))
    dynamics = CompressibleDynamics(td; surface_pressure=p₀,
                                      reference_potential_temperature=θᵇᵍ)

    model = AtmosphereModel(grid; advection=WENO(), dynamics, timestepper)

    ref = model.dynamics.reference_state
    set!(model; θ=θᵢ, u=U, qᵗ=0, ρ=ref.density)

    return model
end

@testset "IGW stability: WS-RK3 (Δt=12, Ns=8) [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    model = build_igw_model(timestepper=:AcousticRungeKutta3, Ns=8, κᵈ=0.10)

    simulation = Simulation(model; Δt=12, stop_iteration=20, verbose=false)
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
##### Dry thermal bubble: split-explicit / explicit / anelastic consistency
#####
##### This is a small wiring regression, not a benchmark. The documented
##### examples cover longer physical integrations. Here we only check that
##### the split-explicit path produces the same short-time buoyant response
##### scale as explicit compressible dynamics and the anelastic model.
#####

function build_tiny_dry_bubble_model(kind)
    grid = RectilinearGrid(acoustic_test_arch;
                           size = (16, 16),
                           halo = (5, 5),
                           x = (-8kilometers, 8kilometers),
                           z = (0, 8kilometers),
                           topology = (Periodic, Flat, Bounded))

    constants = ThermodynamicConstants()
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    surface_pressure = 100000
    standard_pressure = 100000
    θ₀ = 300
    N² = 0
    θ_background(z) = θ₀ * exp(N² * z / g)
    reference_exner(z) = (surface_pressure / standard_pressure)^κ - g * z / (cᵖᵈ * θ₀)
    reference_pressure(z) = standard_pressure * reference_exner(z)^(1 / κ)

    if kind === :anelastic
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure,
                                         potential_temperature = θ_background)
        dynamics = AnelasticDynamics(reference_state)
        timestepper = :SSPRungeKutta3
    elseif kind === :explicit
        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        surface_pressure,
                                        standard_pressure,
                                        reference_potential_temperature = θ_background)
        timestepper = :SSPRungeKutta3
    elseif kind === :split_explicit
        time_discretization = SplitExplicitTimeDiscretization(; substeps = 6)
        dynamics = CompressibleDynamics(time_discretization;
                                        surface_pressure,
                                        standard_pressure,
                                        reference_potential_temperature = θ_background)
        timestepper = :AcousticRungeKutta3
    else
        error("Unknown tiny bubble model kind: $kind")
    end

    model = AtmosphereModel(grid; advection = WENO(), dynamics, timestepper)

    Δθ = 10
    radius = 2kilometers
    xᵇ = 0
    zᵇ = 3kilometers
    θ_initial(x, z) = θ_background(z) + Δθ * max(0, 1 - sqrt((x - xᵇ)^2 + (z - zᵇ)^2) / radius)
    ρ_initial(x, z) = reference_pressure(z) / (Rᵈ * θ_initial(x, z) * reference_exner(z))

    if kind === :anelastic
        set!(model; θ = θ_initial, qᵗ = 0)
    else
        set!(model; θ = θ_initial, ρ = ρ_initial, qᵗ = 0)
    end

    return model
end

function tiny_bubble_diagnostics(model)
    w = Array(interior(model.velocities.w))
    positive_w = max.(0, w)
    max_w = maximum(positive_w)
    total_positive_w = sum(positive_w)

    grid = model.grid
    z_faces = [znode(1, 1, k, grid, Center(), Center(), Face()) for k in axes(w, 3)]
    zᵂ = sum(sum(view(positive_w, :, :, k)) * z_faces[k] for k in axes(w, 3)) / total_positive_w

    return (; max_w, zᵂ)
end

@testset "Tiny dry thermal bubble consistency [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    anelastic_model = build_tiny_dry_bubble_model(:anelastic)
    explicit_model = build_tiny_dry_bubble_model(:explicit)
    split_model = build_tiny_dry_bubble_model(:split_explicit)

    simulations = (
        Simulation(anelastic_model; Δt = 0.5, stop_time = 0.5, verbose = false),
        Simulation(explicit_model; Δt = 0.25, stop_time = 0.5, verbose = false),
        Simulation(split_model; Δt = 0.5, stop_time = 0.5, verbose = false),
    )

    run!.(simulations)

    anelastic = tiny_bubble_diagnostics(anelastic_model)
    explicit = tiny_bubble_diagnostics(explicit_model)
    split = tiny_bubble_diagnostics(split_model)

    for model in (anelastic_model, explicit_model, split_model)
        @test !any(isnan, parent(model.velocities.w))
        @test !any(isinf, parent(model.velocities.w))
    end

    @test anelastic.max_w > 0
    @test explicit.max_w > 0
    @test split.max_w > 0

    @test isapprox(split.max_w, explicit.max_w; rtol = 0.25)
    # Anelastic dynamics filters acoustic adjustment, so only require the
    # same short-time buoyant response scale and centroid.
    @test isapprox(split.max_w, anelastic.max_w; rtol = 1.25)

    Δz = anelastic_model.grid.Lz / anelastic_model.grid.Nz
    @test abs(split.zᵂ - explicit.zᵂ) ≤ Δz
    @test abs(split.zᵂ - anelastic.zᵂ) ≤ 2Δz
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
    dynamics = CompressibleDynamics(td; surface_pressure=100000,
                                      reference_potential_temperature=300)

    model = AtmosphereModel(grid; advection=WENO(), dynamics)

    ref = model.dynamics.reference_state
    set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

    simulation = Simulation(model; Δt=12, stop_iteration=10, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 10

    # With no perturbation and balanced reference state, w should be near zero
    w_max = @allowscalar maximum(abs, interior(model.velocities.w))
    @test w_max < 1e-6  # Should be at machine precision level
end

#####
##### Test acoustic divergence damping (Klemp 2018)
#####

@testset "Acoustic divergence damping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(8, 8, 8), halo=(5, 5, 5),
                           x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers))

    # Exercise the divergence-damping path with the typed AcousticDampingStrategy.
    td = SplitExplicitTimeDiscretization(substeps=8,
                                         damping=ThermodynamicDivergenceDamping(coefficient=FT(0.5)))
    dynamics = CompressibleDynamics(td; reference_potential_temperature=300)
    model = AtmosphereModel(grid; advection=WENO(), dynamics,
                            timestepper=:AcousticRungeKutta3)

    ref = model.dynamics.reference_state
    set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

    simulation = Simulation(model; Δt=6, stop_iteration=3, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 3
    @test !any(isnan, parent(model.dynamics.density))
end

#####
##### Test explicit time stepping default
#####

@testset "Default time stepper for ExplicitTimeStepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

    dynamics = CompressibleDynamics(ExplicitTimeStepping())
    model = AtmosphereModel(grid; dynamics)

    @test model.timestepper isa SSPRungeKutta3
end

#####
##### CompressibleDynamics show methods
#####

@testset "CompressibleDynamics show methods" begin
    # Pre-materialization
    dynamics = CompressibleDynamics()
    s = sprint(show, dynamics)
    @test occursin("CompressibleDynamics", s)
    @test occursin("ExplicitTimeStepping", s)
    @test occursin("not materialized", s)

    # With split-explicit
    td = SplitExplicitTimeDiscretization(substeps=8)
    dynamics2 = CompressibleDynamics(td; reference_potential_temperature=300)
    s2 = sprint(show, dynamics2)
    @test occursin("SplitExplicitTimeDiscretization", s2)
end

#####
##### ExnerReferenceState construction and show
#####

@testset "ExnerReferenceState [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 10000),
                           topology=(Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(FT)

    @testset "Construction and basic properties" begin
        ref = ExnerReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
        @test ref isa ExnerReferenceState
        @test eltype(ref) == FT
        @test ref.surface_pressure == FT(101325)
        @test ref.surface_potential_temperature == FT(300)

        # Pressure should decrease monotonically
        for k in 2:grid.Nz
            pᵏ = @allowscalar ref.pressure[1, 1, k]
            pᵏ⁻¹ = @allowscalar ref.pressure[1, 1, k-1]
            @test pᵏ < pᵏ⁻¹
        end
    end

    @testset "show/summary" begin
        ref = ExnerReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
        s = sprint(show, ref)
        @test occursin("ExnerReferenceState", s)
        @test occursin("p₀", s)
    end

    @testset "surface_density" begin
        ref = ExnerReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
        ρ₀ = surface_density(ref)
        @test ρ₀ > 0
        @test ρ₀ isa FT
    end

    @testset "Function-valued θ₀" begin
        g = constants.gravitational_acceleration
        θ_func(z) = FT(300) * exp(FT(1e-4) * z / g)
        ref = ExnerReferenceState(grid, constants; surface_pressure=100000, potential_temperature=θ_func)
        @test ref isa ExnerReferenceState

        # Pressure should still decrease monotonically
        for k in 2:grid.Nz
            pᵏ = @allowscalar ref.pressure[1, 1, k]
            pᵏ⁻¹ = @allowscalar ref.pressure[1, 1, k-1]
            @test pᵏ < pᵏ⁻¹
        end
    end
end

#####
##### SlowTendencyMode and HorizontalSlowMode
#####

@testset "SlowTendencyMode and HorizontalSlowMode [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(8, 8, 8), halo=(5, 5, 5),
                           x=(0, 100), y=(0, 100), z=(0, 1000))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                   reference_potential_temperature=300)
    model = AtmosphereModel(grid; advection=WENO(), dynamics)
    ref = model.dynamics.reference_state
    set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

    @testset "SlowTendencyMode" begin
        slow = SlowTendencyMode(model.dynamics)
        @test x_pressure_gradient(1, 1, 1, grid, slow) == 0
        @test y_pressure_gradient(1, 1, 1, grid, slow) == 0
        @test z_pressure_gradient(1, 1, 1, grid, slow) == 0
        @test buoyancy_forceᶜᶜᶜ(1, 1, 1, grid, slow) == 0
        @test dynamics_density(slow) === model.dynamics.density
    end

    @testset "HorizontalSlowMode" begin
        hslow = HorizontalSlowMode(model.dynamics)
        @test z_pressure_gradient(1, 1, 1, grid, hslow) == 0
        @test buoyancy_forceᶜᶜᶜ(1, 1, 1, grid, hslow) == 0
        @test dynamics_density(hslow) === model.dynamics.density
    end
end

#####
##### CompressibleDynamics without reference state (ExplicitTimeStepping)
#####

@testset "CompressibleDynamics without reference state [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(acoustic_test_arch; size=(8, 8, 8), halo=(5, 5, 5),
                           x=(0, 4000), y=(0, 4000), z=(0, 4000))

    dynamics = CompressibleDynamics()
    model = AtmosphereModel(grid; advection=WENO(), dynamics)

    set!(model; θ=300, u=0, qᵗ=0, ρ=1.2)
    simulation = Simulation(model; Δt=0.1, stop_iteration=3, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 3
    @test !any(isnan, parent(model.dynamics.density))
    @test model.dynamics.reference_state === nothing
end
