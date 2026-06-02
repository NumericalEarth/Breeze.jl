#####
##### Tests for `adiabatic_initialization!` (FV3-SHiELD `na_init`).
#####
##### - Nudge algebra: the slow-field blend is exactly (x + weight·x₀)/(1+weight),
#####   and ρw is never snapshotted/nudged.
##### - Rest-state invariance: a discrete-balanced rest state is a fixed point —
#####   slow fields return to it, ρw stays ~0, the clock resets to t=0.
##### - Fast-mode damping: a seeded vertical-acoustic ρw shrinks across one cycle.
#####

using Breeze
using Breeze: dynamics_density
using Oceananigans
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
                                 microphysics = nothing,
                                 timestepper = :AcousticRungeKutta3)
end

# Discrete-balanced rest state (mirrors substepper_rest_state.jl::set_rest_state!).
function _set_discrete_rest!(model)
    ref = model.dynamics.reference_state
    Rᵈ  = Breeze.dry_air_gas_constant(model.thermodynamic_constants)
    parent(model.dynamics.density) .= parent(ref.density)
    ρθ = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
    parent(ρθ) .= parent(ref.pressure) ./ (Rᵈ .* parent(ref.exner_function))
    fill!(parent(model.velocities.u), 0)
    fill!(parent(model.velocities.v), 0)
    fill!(parent(model.velocities.w), 0)
    update_state!(model)
    return nothing
end

@testset "adiabatic_initialization!" begin

    @testset "nudge algebra (slow fields blended, ρw untouched)" begin
        model = _build_adiabatic_model(default_arch)
        _set_discrete_rest!(model)

        ρθ = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)

        # Snapshot the slow fields at value B = 2 for ρθ.
        interior(ρθ) .= 2.0
        snap = Breeze.CompressibleEquations.snapshot_slow_fields(model)

        # Move ρθ to A = 5 and seed ρw with a marker that must survive.
        interior(ρθ) .= 5.0
        interior(model.momentum.ρw) .= 7.0

        Breeze.CompressibleEquations.nudge_slow_fields!(model, snap, 2)

        # (5 + 2·2)/3 = 3 for the nudged slow field; ρw unchanged.
        @test @allowscalar(interior(ρθ)[4, 4, 16]) ≈ 3.0
        @test @allowscalar(interior(model.momentum.ρw)[4, 4, 16]) == 7.0
    end

    @testset "rest-state invariance and clock reset" begin
        model = _build_adiabatic_model(default_arch)
        _set_discrete_rest!(model)

        ρ   = dynamics_density(model.dynamics)
        ρθ  = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
        ρ₀  = Array(interior(ρ))
        ρθ₀ = Array(interior(ρθ))

        adiabatic_initialization!(model; Δt = 1.0, cycles = 1)

        @test maximum(abs, Array(interior(ρ))  .- ρ₀)  <= 1e-9 * maximum(abs, ρ₀)
        @test maximum(abs, Array(interior(ρθ)) .- ρθ₀) <= 1e-9 * maximum(abs, ρθ₀)
        @test maximum(abs, Array(interior(model.momentum.ρw))) <= 1e-8
        @test model.clock.time == 0
        @test model.clock.iteration == 0
    end

    @testset "fast-mode (acoustic) ρw shrinks across one cycle" begin
        model = _build_adiabatic_model(default_arch)
        _set_discrete_rest!(model)

        # Seed a small horizontally-uniform vertical-acoustic ρw perturbation.
        set!(model; w = (x, y, z) -> 0.01 * sin(2π * z / 2e3))
        update_state!(model)
        w_before = maximum(abs, Array(interior(model.momentum.ρw)))

        # Acoustic Δt; default forward_weight = 0.65 damps the resolved
        # acoustic branch within the symmetric excursion.
        adiabatic_initialization!(model; Δt = 0.1, cycles = 1)
        w_after = maximum(abs, Array(interior(model.momentum.ρw)))

        @test w_before > 0
        @test w_after < w_before
    end

end
