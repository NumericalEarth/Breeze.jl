#####
##### Rest-atmosphere validation tests for the acoustic substepper.
#####
##### These tests live at the heart of the pristine-substepper program
##### (validation/substepping/PRISTINE_SUBSTEPPER_PLAN.md).
##### They encode the failure-mode hypotheses from the BBI report
##### (~/BreezyBaroclinicInstability.jl/SUBSTEPPER_INSTABILITY_REPORT.md)
##### and the contract from validation/substepping/SUBSTEPPER_TEST_PLAN.md
##### (T1, T2, T3, T4, T6).
#####
##### Each test checks ONE failure mode at a time so a regression points
##### at its root cause:
#####
##### - T1 — discrete hydrostatic balance of the reference state itself
##### - T2 — pressure produced by `update_state!` agrees with `ref.pressure`
##### - T3 — slow vertical-momentum tendency is zero on a rest atmosphere
##### - T4 — `max|w|` stays at machine zero across many outer steps for a
#####        sweep of `Δt` values
##### - T6 — Bounded-y vs Periodic-y produce identical rest-atmosphere
#####        trajectories
#####
##### The pass criteria mirror SUBSTEPPER_TEST_PLAN.md exactly.
#####

using Breeze
using Breeze: dynamics_density
using Breeze.CompressibleEquations: AcousticSubstepper,
                                    freeze_outer_step_state!,
                                    assemble_slow_vertical_momentum_tendency!
using Breeze.TimeSteppers: compute_slow_momentum_tendencies!,
                           compute_slow_scalar_tendencies!

using Oceananigans
using Oceananigans.TimeSteppers: update_state!

using GPUArraysCore: @allowscalar
using Printf
using Test

#####
##### Shared setup
#####

const T₀_REST    = 250.0      # Isothermal reference temperature (K)
const Lz_REST    = 30e3       # Domain depth (m)
const Lh_REST    = 100e3      # Domain horizontal extent (m)
const NH_REST    = 16
const NZ_REST    = 64
const G_REST     = 9.80665    # Reference gravity used for analytic θ profile

# Isothermal reference: θ̄(z) = T₀ exp(g z / (cᵖᵈ T₀)). The reference state
# constructor will diagnose ρ̄, p̄, Π̄ from this θ̄ and the surface pressure.
const CPD_REST = 1005.0
θ_isothermal_ref(z) = T₀_REST * exp(G_REST * z / (CPD_REST * T₀_REST))
θ_isothermal_xyz(x, y, z) = θ_isothermal_ref(z)

function _build_rest_grid(arch; topology = (Periodic, Periodic, Bounded),
                                Nx = NH_REST, Ny = NH_REST, Nz = NZ_REST,
                                halo = (5, 5, 5),
                                Lx = Lh_REST, Ly = Lh_REST, Lz = Lz_REST)
    return RectilinearGrid(arch;
                           size = (Nx, Ny, Nz), halo = halo,
                           x = (0, Lx), y = (0, Ly), z = (0, Lz),
                           topology = topology)
end

function _build_rest_model(arch; substeps = nothing,
                                 forward_weight = nothing,
                                 damping = nothing,
                                 grid_kwargs...)
    grid = _build_rest_grid(arch; grid_kwargs...)
    constants = ThermodynamicConstants(eltype(grid))
    # Pass `nothing` to defer to the SplitExplicit defaults (forward_weight
    # = 0.6, ThermalDivergenceDamping coef = 0.1) — that is the production
    # configuration and the one Phase 4 stabilizes.
    td_kwargs = (; substeps)
    forward_weight === nothing || (td_kwargs = (; td_kwargs..., forward_weight))
    damping        === nothing || (td_kwargs = (; td_kwargs..., damping))
    td = SplitExplicitTimeDiscretization(; td_kwargs...)
    # Use isentropic θ̄(z) reference path (`_compute_exner_reference!`),
    # which the moist baroclinic wave example uses. The Phase-2 fix
    # gives discrete hydrostatic balance to ulp via Newton iteration
    # in `_compute_exner_reference!`. Align surface_pressure with
    # standard_pressure so Π_surface = 1 and tests are bit-identical
    # against an analytic isothermal reference.
    dyn = CompressibleDynamics(td;
                               reference_potential_temperature = θ_isothermal_ref,
                               surface_pressure = 1e5,
                               standard_pressure = 1e5)
    return AtmosphereModel(grid; dynamics = dyn,
                                  thermodynamic_constants = constants,
                                  timestepper = :AcousticRungeKutta3)
end

# Set the model state to the discrete-balanced reference EXACTLY.
# Bypasses `set!(model; θ, ρ)` (which uses a continuous-formula θ that
# only agrees with the discrete θ̄_ref to O((αΔz)³)). This is what
# Phase 0 tests need to validate "true rest" — the user-facing pattern
# of setting θ from a continuous profile is a separate concern (audit
# E3 / Phase 6).
function set_rest_state!(model)
    ref = model.dynamics.reference_state
    Rᵈ  = Breeze.dry_air_gas_constant(model.thermodynamic_constants)

    # ρ ← ρ_ref (the model's prognostic density field).
    parent(model.dynamics.density) .= parent(ref.density)

    # ρθ ← p_ref / (Rᵈ Π_ref). Equivalent to ρ_ref · θ̄_ref with
    # θ̄_ref = T₀/Π_ref, but avoids any continuous-formula intermediate.
    ρθ_field = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
    parent(ρθ_field) .= parent(ref.pressure) ./ (Rᵈ .* parent(ref.exner_function))

    # Zero velocities.
    fill!(parent(model.velocities.u), 0)
    fill!(parent(model.velocities.v), 0)
    fill!(parent(model.velocities.w), 0)

    update_state!(model)
    return nothing
end

# Compute the discrete hydrostatic residual ε = δz(p_ref) + g·ℑz(ρ_ref)
# at every interior z-face (k = 2..Nz). Uniform-Δz grids only — that's
# what these rest tests use. Returns a Vector across all interior cells
# / faces (host-side) so we can `maximum(abs, ...)` directly.
function discrete_hydrostatic_residual(model)
    grid = model.grid
    ref  = model.dynamics.reference_state
    g    = Float64(model.thermodynamic_constants.gravitational_acceleration)

    Δz = Float64(grid.z.Δᵃᵃᶜ)        # uniform-Δz scalar
    p  = Array(interior(ref.pressure))  # (Nx, Ny, Nz)
    ρ  = Array(interior(ref.density))

    Nx, Ny, Nz = size(p)
    ε = Vector{Float64}(undef, Nx * Ny * (Nz - 1))
    n = 0
    @inbounds for i in 1:Nx, j in 1:Ny, k in 2:Nz
        n += 1
        ε[n] = (p[i, j, k] - p[i, j, k - 1]) / Δz +
               g * (ρ[i, j, k] + ρ[i, j, k - 1]) / 2
    end
    return ε
end

#####
##### T1 — Reference-state discrete hydrostatic balance
#####
##### The reference state must satisfy the substepper's *exact* discrete
##### form `δz(p_ref) + g·ℑz(ρ_ref) = 0` at every interior face. Anything
##### above ~1 ulp of `g·ρ_max ≈ 12 N/m³` becomes a slow-tendency seed.
#####

@testset "Substepper rest-state validation" begin
@testset "T1 — Reference-state hydrostatic balance" begin
    model = _build_rest_model(default_arch)
    ε     = discrete_hydrostatic_residual(model)

    max_residual = maximum(abs, ε)
    @info @sprintf("[T1] max|δz(p_ref) + g·ℑz(ρ_ref)| = %.3e N/m³", max_residual)

    # Threshold from SUBSTEPPER_TEST_PLAN.md §T1: a few hundred ulp on a
    # `g·ρ_max ≈ 12 N/m³` scale. 1e-9 is the documented bound.
    # Phase 2 fix (PRISTINE_SUBSTEPPER_PLAN.md P2.1 / audit E2):
    # `_compute_isothermal_reference!` enforces the discrete δz / ℑz
    # relation so the residual is at machine ε × g·ρ.
    @test max_residual <= 1e-9
end

#####
##### T2 — EoS-pressure vs reference-pressure consistency
#####
##### Setting ρ ← ref.density and θ ← reference θ profile, the EoS path
##### inside `update_state!` should yield a pressure bit-identical (to
##### within rounding) to `ref.pressure`. Otherwise every outer step
##### inherits a non-physical pressure-imbalance seed (BBI report §"The
##### seed", ~3e-11 Pa).
#####

@testset "T2 — Pressure consistency between EoS and reference paths" begin
    model = _build_rest_model(default_arch)
    ref   = model.dynamics.reference_state

    set_rest_state!(model)

    Δp = maximum(abs, interior(model.dynamics.pressure) .- interior(ref.pressure))
    p_scale = maximum(abs, interior(ref.pressure))
    Δρ = maximum(abs, interior(dynamics_density(model.dynamics)) .- interior(ref.density))

    @info @sprintf("[T2] max|p - p_ref| = %.3e Pa (p_scale = %.3e); max|ρ - ρ_ref| = %.3e",
                   Δp, p_scale, Δρ)

    # SUBSTEPPER_TEST_PLAN.md §T2: Δp ≤ 100 ulp of p, Δρ exactly zero.
    @test Δp <= 100 * eps(Float64) * p_scale
    @test Δρ == 0
end

#####
##### T3 — Slow vertical-momentum tendency at rest
#####
##### `Gˢρw = Gⁿρw - ∂z(p⁰ - p_ref) - g(ρ⁰ - ρ_ref)` must be at machine
##### zero when U⁰ = reference state. A non-zero Gˢρw is the bug seed
##### before any substep loop runs (BBI report observed ~6e-14 N/m³).
#####

@testset "T3 — Slow vertical-momentum tendency on rest atmosphere" begin
    model = _build_rest_model(default_arch)
    ref   = model.dynamics.reference_state

    set_rest_state!(model)

    sub = model.timestepper.substepper
    freeze_outer_step_state!(sub, model)
    compute_slow_momentum_tendencies!(model)
    compute_slow_scalar_tendencies!(model)
    assemble_slow_vertical_momentum_tendency!(sub, model)

    max_slow_ρw = maximum(abs, interior(sub.slow_vertical_momentum_tendency))
    @info @sprintf("[T3] max|Gˢρw| = %.3e N/m³ (rest atmosphere, U⁰ = ref state)",
                   max_slow_ρw)

    # SUBSTEPPER_TEST_PLAN.md §T3: bound 1e-12.
    @test max_slow_ρw <= 1e-12
end

#####
##### T4 — Rest-atmosphere drift, Δt sweep
#####
##### A rest atmosphere should keep `max|w|` near machine epsilon × cs
##### (≈ 1e-13 m/s for Float64) over arbitrarily many outer steps. The
##### envelope must stay below 1e-10 m/s — five orders of magnitude
##### slack — for every Δt in the sweep, at the default
##### `forward_weight = 0.55` with `NoDivergenceDamping`.
#####
##### This is the canonical Phase-0 reproducer of the BBI report's
##### factor-2-per-outer-step instability: it currently FAILS at
##### Δt ≥ ~5 s on the 30 km column. We mark the failing cases with
##### `@test_broken` so the regression goes green when Phase 1 lands;
##### anyone who breaks the working Δt = 0.5 s baseline is caught
##### immediately.
#####

function _track_rest_drift(model, Δt; n_steps = 200, sample_every = 10)
    ref = model.dynamics.reference_state
    set_rest_state!(model)

    drift = Float64[]
    nans  = false
    crashed = false
    for n in 1:n_steps
        try
            time_step!(model, Δt)
        catch err
            # The rest-atmosphere instability typically runs the column
            # ρ negative within ~15 outer steps; the EoS then errors
            # with `DomainError` evaluating `(ρ θ Rᵐ / p^*)^?`. Capture
            # that as the failure mode equivalent to a NaN.
            crashed = true
            break
        end
        if (n % sample_every) == 0
            wmax = Float64(maximum(abs, interior(model.velocities.w)))
            push!(drift, wmax)
            if !isfinite(wmax)
                nans = true
                break
            end
        end
    end
    return (envelope = isempty(drift) ? 0.0 : maximum(drift),
            final = isempty(drift) ? 0.0 : last(drift),
            nans = nans || crashed, drift = drift)
end

@testset "T4 — Rest-atmosphere drift Δt sweep" begin
    # SUBSTEPPER_TEST_PLAN.md §T4: a rest atmosphere on the production
    # configuration (default forward_weight, default Klemp damping) must
    # keep `max|w|` near machine ε across many outer steps for the full
    # range of Δt up to the documented production value (20 s).
    # Phase 4 fix (`ThermalDivergenceDamping` 3-D + `forward_weight = 0.6`):
    # both Δt cases now pass at the 1e-10 m/s bound. The legacy "broken"
    # configuration `forward_weight = 0.55, NoDivergenceDamping()` still
    # blows up at Δt = 20 s — `@test_broken` documents that, so anyone
    # who claims to fix it without addressing the matrix asymmetry
    # (Phase 4 P4.5+) sees their improvement become a real `@test`.
    Δt_cases = (0.5, 20.0)

    n_steps = 200
    bound   = 1e-10

    for Δt in Δt_cases
        model = _build_rest_model(default_arch;
                                  Nx = 8, Ny = 8, Nz = 32, Lz = 10e3)
        result = _track_rest_drift(model, Δt; n_steps = n_steps,
                                              sample_every = 10)
        @info @sprintf("[T4 default] Δt = %.2f s, envelope = %.3e m/s, final = %.3e m/s, nans = %s",
                       Δt, result.envelope, result.final, result.nans)
        @test !result.nans
        @test result.envelope <= bound
    end

    # Document the legacy regression: forward_weight = 0.55 + no damping
    # (the previous default) still blows up at Δt = 20 s. The
    # @test_broken markers turn into surprise-passes if a real
    # symmetric-matrix fix lands without needing damping.
    let Δt = 20.0
        model = _build_rest_model(default_arch;
                                  Nx = 8, Ny = 8, Nz = 32, Lz = 10e3,
                                  forward_weight = 0.55,
                                  damping = NoDivergenceDamping())
        result = _track_rest_drift(model, Δt; n_steps = n_steps,
                                              sample_every = 10)
        @info @sprintf("[T4 legacy ω=0.55 noDD] Δt = %.2f s, envelope = %.3e m/s, nans = %s",
                       Δt, result.envelope, result.nans)
        @test_broken !result.nans
        @test_broken result.envelope <= bound
    end
end

#####
##### T6 — Bounded-y vs Periodic-y rest equivalence
#####
##### For a horizontally-uniform IC on a horizontally-uniform reference
##### state, both Periodic and Bounded boundaries must produce identical
##### `max|w|` trajectories — there is zero horizontal gradient to
##### distinguish them. Any divergence indicates a halo-fill bug that
##### only fires under one topology.
#####

@testset "T6 — Bounded-y vs Periodic-y rest equivalence" begin
    Δt = 0.5
    n_steps = 50
    sample_every = 5

    model_PP = _build_rest_model(default_arch;
                                 Nx = 8, Ny = 8, Nz = 32, Lz = 10e3,
                                 topology = (Periodic, Periodic, Bounded))
    model_PB = _build_rest_model(default_arch;
                                 Nx = 8, Ny = 8, Nz = 32, Lz = 10e3,
                                 topology = (Periodic, Bounded, Bounded))

    drift_PP = _track_rest_drift(model_PP, Δt; n_steps = n_steps,
                                                sample_every = sample_every).drift
    drift_PB = _track_rest_drift(model_PB, Δt; n_steps = n_steps,
                                                sample_every = sample_every).drift

    @assert length(drift_PP) == length(drift_PB)
    Δmax = isempty(drift_PP) ? 0.0 : maximum(abs.(drift_PP .- drift_PB))
    @info @sprintf("[T6] max|max|w|_PP - max|w|_PB| over %d samples = %.3e m/s",
                   length(drift_PP), Δmax)

    # SUBSTEPPER_TEST_PLAN.md §T6 bound. Generous since both cases
    # individually currently leak a tiny ~1e-13 m/s drift on Float64;
    # what we test is that they *agree* to within 1e-12 m/s, not the
    # absolute drift (T4 is responsible for that).
    @test Δmax <= 1e-12
end
end  # outer "Substepper rest-state validation"
