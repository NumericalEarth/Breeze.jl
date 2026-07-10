#####
##### Tests for `balance_adiabatically!` (FV3-SHiELD `na_init`).
#####
##### - Nudge algebra: the slow-field blend is exactly (x + weight·x₀)/(1+weight),
#####   and ρw is never snapshotted/nudged.
##### - Rest-state invariance: a discrete-balanced rest state is a fixed point —
#####   slow fields return to it, ρw stays ~0, the clock resets to t=0.
##### - ρw spin-up: a seeded out-of-balance ρw shrinks across one cycle.
#####

using Breeze
using Breeze: dynamics_density
using Breeze.BoundaryConditions: EnergyFluxBoundaryCondition, EnergyFluxBoundaryConditionFunction
using Breeze.AtmosphereModels: adiabatic_balance_twin, AdiabaticBalancer, thermodynamic_density
using Breeze.AtmosphereModels: adiabatic_scalar_bcs
using Oceananigans
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.TimeSteppers: update_state!
using GPUArraysCore: @allowscalar
using Test

const T₀_AI  = 250.0
const G_AI   = 9.80665
const CPD_AI = 1005.0
θ_iso_ai(z) = T₀_AI * exp(G_AI * z / (CPD_AI * T₀_AI))

# Init-mode model: no upper sponge (the one irreversible substep ingredient),
# no microphysics. Otherwise mirrors the production split-explicit config.
function _build_adiabatic_model(arch; Nx = 8, Ny = 8, Nz = 32, Lz = 10e3, Lh = 100e3)
    grid = RectilinearGrid(arch;
                           size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (0, Lh), y = (0, Lh), z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(eltype(grid))
    td  = SplitExplicitTimeDiscretization(; sponge = nothing)
    dyn = CompressibleDynamics(td;
                               reference_potential_temperature = θ_iso_ai,
                               surface_pressure  = 1e5,
                               standard_pressure = 1e5)
    return AtmosphereModel(grid; dynamics = dyn,
                                 thermodynamic_constants = constants,
                                 microphysics = nothing)
end

# Discrete-balanced rest state (mirrors substepper_rest_state.jl::set_rest_state!).
function _set_discrete_rest!(model)
    ref = model.dynamics.reference_state
    Rᵈ  = Breeze.dry_air_gas_constant(model.thermodynamic_constants)
    parent(model.dynamics.dry_density) .= parent(ref.density)
    ρθ = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
    parent(ρθ) .= parent(ref.pressure) ./ (Rᵈ .* parent(ref.exner_function))
    fill!(parent(model.velocities.u), 0)
    fill!(parent(model.velocities.v), 0)
    fill!(parent(model.velocities.w), 0)
    update_state!(model)
    return nothing
end

@testset "balance_adiabatically!" begin

    @testset "nudge algebra (initial fields blended, ρw untouched)" begin
        model = _build_adiabatic_model(default_arch)
        _set_discrete_rest!(model)

        ρθ = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)

        # Snapshot the initial fields at value B = 2 for ρθ.
        interior(ρθ) .= 2.0
        snap = Breeze.AtmosphereModels.snapshot_initial_fields(model)

        # Move ρθ to A = 5 and seed ρw with a marker that must survive.
        interior(ρθ) .= 5.0
        interior(model.momentum.ρw) .= 7.0

        Breeze.AtmosphereModels.nudge_initial_fields!(model, snap, 2)

        # (5 + 2·2)/3 = 3 for the nudged initial field; ρw unchanged.
        @test @allowscalar(interior(ρθ)[4, 4, 16]) ≈ 3.0
        @test @allowscalar(interior(model.momentum.ρw)[4, 4, 16]) == 7.0
    end

    @testset "rest-state invariance and clock reset" begin
        model = _build_adiabatic_model(default_arch)
        _set_discrete_rest!(model)

        @test length(Breeze.AtmosphereModels.initial_fields(model)) == 5

        ρ   = dynamics_density(model.dynamics)
        ρθ  = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
        ρ₀  = Array(interior(ρ))
        ρθ₀ = Array(interior(ρθ))

        balance_adiabatically!(model; Δt = 1.0, cycles = 1)

        @test maximum(abs, Array(interior(ρ))  .- ρ₀)  <= 1e-9 * maximum(abs, ρ₀)
        @test maximum(abs, Array(interior(ρθ)) .- ρθ₀) <= 1e-9 * maximum(abs, ρθ₀)
        @test maximum(abs, Array(interior(model.momentum.ρw))) <= 1e-8
        # Fully reset, not just time/iteration: the excursion leaves stage/last_Δt dirty.
        @test model.clock.time == 0
        @test model.clock.iteration == 0
        @test model.clock.stage == 1
        @test isinf(model.clock.last_Δt)
    end

    @testset "ρw spin-up: seeded ρw shrinks across one cycle" begin
        model = _build_adiabatic_model(default_arch)
        _set_discrete_rest!(model)

        # Seed a small horizontally-uniform vertical-acoustic ρw perturbation.
        set!(model; w = (x, y, z) -> 0.01 * sin(2π * z / 2e3))
        update_state!(model)
        w_before = maximum(abs, Array(interior(model.momentum.ρw)))

        # Acoustic Δt; default forward_weight = 0.65 damps the resolved
        # acoustic branch within the symmetric excursion.
        balance_adiabatically!(model; Δt = 0.1, cycles = 1)
        w_after = maximum(abs, Array(interior(model.momentum.ρw)))

        @test w_before > 0
        @test w_after < w_before
    end

    @testset "anelastic: spin-up runs and preserves slow fields at rest" begin
        grid = RectilinearGrid(default_arch;
                               size = (8, 8, 16), halo = (3, 3, 3),
                               x = (0, 1e4), y = (0, 1e4), z = (0, 4e3),
                               topology = (Periodic, Periodic, Bounded))
        constants = ThermodynamicConstants(eltype(grid))
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure      = 1e5,
                                         potential_temperature = 300,
                                         vapor_mass_fraction   = 0)
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; dynamics,
                                formulation  = :LiquidIcePotentialTemperature,
                                microphysics = nothing)

        # Isentropic reference θᵣ = 300; θ = θᵣ with zero velocities is an
        # anelastic rest state (zero tendencies).
        set!(model; θ = (x, y, z) -> 300.0)

        # initial_fields drops ρ for anelastic → 4 entries (vs 5 for compressible).
        @test length(Breeze.AtmosphereModels.initial_fields(model)) == 4

        ρθ  = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
        ρθ₀ = Array(interior(ρθ))

        # Exercises the backward step time_step!(anelastic, -Δt).
        balance_adiabatically!(model; Δt = 1.0, cycles = 1)

        @test all(isfinite, Array(interior(model.momentum.ρu)))
        @test maximum(abs, Array(interior(ρθ)) .- ρθ₀) <= 1e-6 * maximum(abs, ρθ₀)
        @test maximum(abs, Array(interior(model.momentum.ρw))) <= 1e-6
        @test model.clock.time == 0
        @test model.clock.iteration == 0
        @test model.clock.stage == 1
        @test isinf(model.clock.last_Δt)
    end

    @testset "adiabatic twin strips diabatic surface fluxes (#842)" begin
        grid = RectilinearGrid(default_arch;
                               size = (8, 8, 32), halo = (5, 5, 5),
                               x = (0, 100e3), y = (0, 100e3), z = (0, 10e3),
                               topology = (Periodic, Periodic, Bounded))
        constants = ThermodynamicConstants(eltype(grid))
        dyn = CompressibleDynamics(SplitExplicitTimeDiscretization(; sponge = nothing);
                                   reference_potential_temperature = θ_iso_ai,
                                   surface_pressure  = 1e5,
                                   standard_pressure = 1e5)
        ρθ_bcs = FieldBoundaryConditions(bottom = EnergyFluxBoundaryCondition(100.0))
        model = AtmosphereModel(grid; dynamics = dyn,
                                      thermodynamic_constants = constants,
                                      microphysics = nothing,
                                      boundary_conditions = (; ρθ = ρθ_bcs))

        # Production carries the diabatic surface energy-flux BC...
        ρθ = thermodynamic_density(model.formulation)
        @test ρθ.boundary_conditions.bottom.condition isa EnergyFluxBoundaryConditionFunction

        # ...the adiabatic twin does not, and shares the production data so the balanced state
        # still lands in `model`.
        twin = adiabatic_balance_twin(model, AdiabaticBalancer())
        ρθ_twin = thermodynamic_density(twin.formulation)
        @test !(ρθ_twin.boundary_conditions.bottom.condition isa EnergyFluxBoundaryConditionFunction)
        @test ρθ_twin.data === ρθ.data

        # And the balancer runs to completion with the diabatic BC present on the model: from a
        # discrete-balanced rest state, ρw stays ~0 (on GPU this also exercises the flux-BC kernel
        # that #842 fails to compile).
        _set_discrete_rest!(model)
        balance_adiabatically!(model, AdiabaticBalancer())
        @test all(isfinite, Array(interior(model.momentum.ρw)))
        @test maximum(abs, Array(interior(model.momentum.ρw))) <= 1e-6
    end

    @testset "adiabatic_scalar_bcs strips every surface flux, keeps other BCs" begin
        bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.5),
                                      top    = ValueBoundaryCondition(300.0))
        stripped = adiabatic_scalar_bcs(bcs)
        # A plain (non-zero) flux BC — the case the type-specific strip used to miss — → no-flux.
        @test stripped.bottom.condition === nothing
        # Non-flux BCs (here a Dirichlet value, inert on the closure-free twin) pass through.
        @test stripped.top.condition == 300.0
    end

    @testset "adiabatic twin strips a plain ρθ surface flux" begin
        grid = RectilinearGrid(default_arch;
                               size = (8, 8, 16), halo = (5, 5, 5),
                               x = (0, 1e4), y = (0, 1e4), z = (0, 4e3),
                               topology = (Periodic, Periodic, Bounded))
        dyn = CompressibleDynamics(SplitExplicitTimeDiscretization(; sponge = nothing);
                                   reference_potential_temperature = θ_iso_ai,
                                   surface_pressure  = 1e5,
                                   standard_pressure = 1e5)
        ρθ_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1e-2))
        model = AtmosphereModel(grid; dynamics = dyn, microphysics = nothing,
                                      boundary_conditions = (; ρθ = ρθ_bcs))

        # A plain (non-wrapped) θ flux survives on production but is stripped from the twin — the
        # diabatic BC the type-specific strip used to pass through unchanged.
        ρθ = thermodynamic_density(model.formulation)
        @test ρθ.boundary_conditions.bottom.condition == 1e-2
        twin = adiabatic_balance_twin(model, AdiabaticBalancer())
        ρθ_twin = thermodynamic_density(twin.formulation)
        @test ρθ_twin.boundary_conditions.bottom.condition === nothing
        @test ρθ_twin.data === ρθ.data
    end

    @testset "adiabatic twin strips a moisture surface flux" begin
        grid = RectilinearGrid(default_arch;
                               size = (8, 8, 16), halo = (5, 5, 5),
                               x = (0, 1e4), y = (0, 1e4), z = (0, 4e3),
                               topology = (Periodic, Periodic, Bounded))
        dyn = CompressibleDynamics(SplitExplicitTimeDiscretization(; sponge = nothing);
                                   reference_potential_temperature = θ_iso_ai,
                                   surface_pressure  = 1e5,
                                   standard_pressure = 1e5)
        ρqᵛ_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1e-5))
        model = AtmosphereModel(grid; dynamics = dyn, microphysics = nothing,
                                      boundary_conditions = (; ρqᵛ = ρqᵛ_bcs))

        # The moisture density carries a surface vapor flux on production; the twin strips it too,
        # so the excursion has no surface moisture source (and never lowers a flux BC on the twin,
        # which drops the microphysics fields a bulk vapor flux would read — the #842 failure mode).
        q_prod = model.moisture_density
        @test q_prod.boundary_conditions.bottom.condition == 1e-5
        twin = adiabatic_balance_twin(model, AdiabaticBalancer())
        @test twin.moisture_density.boundary_conditions.bottom.condition === nothing
        @test twin.moisture_density.data === q_prod.data
    end

end
