using KernelAbstractions: @kernel, @index

using Oceananigans.Utils: time_difference_seconds

using Oceananigans.TimeSteppers:
    AbstractTimeStepper,
    tick_stage!,
    update_state!,
    compute_flux_bc_tendencies!,
    step_lagrangian_particles!

using Breeze.AtmosphereModels: AtmosphereModel

using Breeze.CompressibleEquations:
    CompressibleDynamics,
    AcousticSubstepper,
    acoustic_rk3_substep_loop!,
    prepare_acoustic_cache!,
    freeze_outer_step_state!

"""
$(TYPEDEF)

Wicker-Skamarock third-order Runge-Kutta time stepper with acoustic
substepping for fully compressible dynamics. Implements the **CM1-style**
variant in which all three stages use the same constant substep size
``Δτ = Δt/N`` and the substep counts scale with the stage fraction:

- Stage 1: ``U^* = U^n + (Δt/3) \\, R(U^n)``       — ``N/3`` substeps
- Stage 2: ``U^{**} = U^n + (Δt/2) \\, R(U^*)``    — ``N/2`` substeps
- Stage 3: ``U^{n+1} = U^n + Δt \\, R(U^{**})``    — ``N`` substeps

Each stage evaluates the RHS at the current stage state, then resets to ``U^n``
and advances by ``β Δt``. The absence of convex combinations makes the scheme
compatible with split-explicit acoustic substepping, allowing the full pressure
gradient and buoyancy to be included in the slow tendency.

This differs from the two MPAS-A variants:

| variant | β₁  | stage-1 substeps | stage-1 Δτ |
|---------|-----|------------------|------------|
| MPAS order=2 (default) | 1/2 | ``N/2`` | ``Δt/N`` |
| MPAS order=3           | 1/3 | ``1``   | ``Δt/3`` |
| **CM1 (this)**         | 1/3 | ``N/3`` | ``Δt/N`` |

Like MPAS order=3 we are formally third-order accurate (the β fractions are
the canonical Wicker–Skamarock 1/3, 1/2, 1), but unlike MPAS order=3 the
stage-1 substep size is the same ``Δτ = Δt/N`` as the other stages — so the
horizontal acoustic CFL is set by ``Δτ``, not by the much larger ``Δt/3``
single-substep imposed by MPAS order=3.

The substepper rounds ``N`` up to a multiple of 6 so that ``N/3`` and ``N/2``
are both integers.

Fields
======

- `β₁, β₂, β₃`: Stage fractions (1/3, 1/2, 1)
- `U⁰`: Storage for state at beginning of time step
- `Gⁿ`: Tendency fields at current stage
- `implicit_solver`: Optional implicit solver for diffusion
- `substepper`: AcousticSubstepper for acoustic substepping infrastructure

References
==========

Wicker, L.J. and Skamarock, W.C. (2002). Time-Splitting Methods for Elastic Models
    Using Forward Time Schemes. Monthly Weather Review, 130, 2088-2097.

Klemp, J.B., Skamarock, W.C. and Dudhia, J. (2007). Conservative Split-Explicit
    Time Integration Methods for the Compressible Nonhydrostatic Equations.
    Monthly Weather Review, 135, 2897-2913.
"""
struct AcousticRungeKutta3{FT, U0, TG, TI, AS, SS} <: AbstractTimeStepper
    β₁ :: FT
    β₂ :: FT
    β₃ :: FT
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
    substepper :: AS
    slow_tendency_snapshot :: SS
end

"""
    AcousticRungeKutta3(grid, prognostic_fields;
                         dynamics,
                         implicit_solver = nothing,
                         Gⁿ = map(similar, prognostic_fields))

Construct an `AcousticRungeKutta3` time stepper for fully compressible dynamics.

Keyword Arguments
=================

- `dynamics`: The [`CompressibleDynamics`](@ref) object containing the `time_discretization`.
- `implicit_solver`: Optional implicit solver for diffusion. Default: `nothing`
- `Gⁿ`: Tendency fields at current stage. Default: similar to `prognostic_fields`
"""
function AcousticRungeKutta3(grid, prognostic_fields;
                              dynamics,
                              implicit_solver::TI = nothing,
                              Gⁿ::TG = map(similar, prognostic_fields)) where {TI, TG}

    FT = eltype(grid)

    # Wicker-Skamarock RK3 stage fractions, CM1-style: canonical (1/3, 1/2, 1)
    # with N/3, N/2, N substeps per stage at constant Δτ = Δt/N.
    β₁ = FT(1//3)
    β₂ = FT(1//2)
    β₃ = FT(1)

    U⁰ = map(similar, prognostic_fields)
    U0 = typeof(U⁰)

    substepper = AcousticSubstepper(grid, dynamics.time_discretization;
                                    prognostic_momentum = (ρu = prognostic_fields.ρu,
                                                            ρv = prognostic_fields.ρv,
                                                            ρw = prognostic_fields.ρw))
    AS = typeof(substepper)

    # Snapshot storage for the slow tendencies (Gⁿ.ρ, Gⁿ.ρu, Gⁿ.ρv, Gⁿ.ρw, Gⁿ.ρθ).
    # Populated at stage 1 from the outer-step-start state and restored at
    # stages 2 and 3 to mimic MPAS's freeze-tend_rho-at-rk_step==1 behavior.
    # Tracer entries (beyond the first 5) are NOT stored — those should be
    # recomputed per stage by update_state!.
    slow_tendency_snapshot = (ρ  = similar(Gⁿ.ρ),
                              ρu = similar(Gⁿ.ρu),
                              ρv = similar(Gⁿ.ρv),
                              ρw = similar(Gⁿ.ρw),
                              ρθ = similar(Gⁿ.ρθ))
    SS = typeof(slow_tendency_snapshot)

    return AcousticRungeKutta3{FT, U0, TG, TI, AS, SS}(β₁, β₂, β₃, U⁰, Gⁿ,
                                                       implicit_solver, substepper,
                                                       slow_tendency_snapshot)
end

#####
##### WS-RK3 substep with acoustic substepping
#####

"""
$(TYPEDSIGNATURES)

Apply a Wicker-Skamarock RK3 substep with acoustic substepping.

The acoustic substep loop handles momentum, density, and the thermodynamic
variable (ρθ or ρe). The substep size is constant ``Δτ = Δt/N`` across all
stages, with the substep count varying as ``Nτ = \\mathrm{round}(β N)``.
Remaining scalars (tracers) are updated with standard RK3.
"""
function acoustic_rk3_substep!(model, Δt, β; snapshot_slow_tendencies=false,
                                                restore_slow_tendencies=false)
    ts = model.timestepper
    substepper = ts.substepper
    U⁰ = ts.U⁰

    # Prepare stage-frozen reference state (needed by slow tendency computation)
    prepare_acoustic_cache!(substepper, model)

    # Compute slow momentum tendencies (advection, Coriolis, diffusion — PGF/buoyancy
    # handled by the acoustic loop via tend_w_euler in convert_slow_tendencies!).
    compute_slow_momentum_tendencies!(model)

    # Compute slow density and thermodynamic tendencies
    compute_slow_scalar_tendencies!(model)

    # MPAS computes tend_rho ONLY at rk_step==1 and freezes it (line 5433-5450 of
    # mpas_atm_time_integration.F). Similarly, tend_theta uses a perturbation-form
    # advection (rw_save × theta_save + rw_p × Δθ) that's ~0 when the wave preserves θ.
    # Both behaviors are equivalent to "compute slow tendencies once per outer step
    # and reuse them for all RK3 stages." We snapshot Gⁿ.ρ/ρθ/ρu/ρv/ρw at stage 1 and
    # restore at stages 2/3. Tracer tendencies (Gⁿ entries beyond the first 5) are
    # NOT touched, so moisture/scalar transport still picks up per-stage updates.
    Gⁿ = ts.Gⁿ
    snap = ts.slow_tendency_snapshot
    if snapshot_slow_tendencies
        # MPAS computes tend_rho ONLY at rk_step==1 (line 5433-5450 of
        # mpas_atm_time_integration.F) and freezes it. MPAS's tend_theta uses a
        # perturbation-form advection `rw × theta + (rw_save - rw) × theta_save`
        # that's ~0 when the wave preserves θ. MPAS also computes the horizontal
        # pressure gradient `tend_u_euler` ONLY at rk_step==1 (line 5467) and
        # freezes it.
        #
        # We snapshot:
        #   - Gⁿ.ρ and Gⁿ.ρθ as full slow tendency snapshots (replace at stages 2/3).
        #   - Horizontal PGF into snap.ρu/snap.ρv (these are NOT the full Gρu/v, just
        #     the −∂p/∂x and −∂p/∂y forces). Added back to Gρu/v at every stage.
        # The slow advection of u and v is recomputed at every stage by
        # compute_slow_momentum_tendencies! using SlowTendencyMode (no PGF), then
        # the frozen PGF snapshot is added on top — matching MPAS exactly.
        # Gⁿ.ρw is not snapshotted: tend_w_euler is frozen by the substepper's
        # convert_slow_tendencies! kernel, and the per-stage recomputation of
        # vertical advection of w is needed for accuracy.
        parent(snap.ρ)  .= parent(Gⁿ.ρ)
        parent(snap.ρθ) .= parent(Gⁿ.ρθ)
        snapshot_horizontal_pgf!(snap.ρu, snap.ρv, model)
    end
    if restore_slow_tendencies
        parent(Gⁿ.ρ)  .= parent(snap.ρ)
        parent(Gⁿ.ρθ) .= parent(snap.ρθ)
    end
    # Add the frozen horizontal PGF snapshot to Gρu, Gρv at every stage.
    add_horizontal_pgf!(Gⁿ.ρu, Gⁿ.ρv, snap.ρu, snap.ρv, model)

    # Execute acoustic substep loop: constant Δτ = Δt/N, varying Nτ = round(β*N)
    acoustic_rk3_substep_loop!(model, substepper, Δt, β, U⁰)

    # Update remaining scalars (tracers) using WS-RK3
    scalar_rk3_substep!(model, β * Δt)

    return nothing
end

#####
##### Scalar update with time-averaged velocities
#####

scalar_rk3_substep!(model, Δt_stage) =
    acoustic_scalar_substep!(model, _rk3_substep!, Δt_stage, Δt_stage)

@kernel function _rk3_substep!(u, u⁰, G, Δt_stage)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Wicker-Skamarock RK3: u = u⁰ + β * Δt * G
        u[i, j, k] = u⁰[i, j, k] + Δt_stage * G[i, j, k]
    end
end

#####
##### Time stepping (main entry point)
#####

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step `Δt` with Wicker-Skamarock RK3 and acoustic substepping.

The algorithm follows [Wicker and Skamarock (2002)](@cite WickerSkamarock2002),
in the CM1-style configuration:
- Outer loop: canonical 3-stage RK3 with stage fractions `Δt/3, Δt/2, Δt`
- Inner loop: Acoustic substeps for fast (pressure) tendencies, with constant
  substep size `Δτ = Δt/N` across all stages (`N/3`, `N/2`, `N` substeps in
  stages 1, 2, 3 respectively)

Each RK stage:
1. Compute slow tendencies (advection, Coriolis, diffusion only — PGF/buoyancy in acoustic loop)
2. Execute acoustic substep loop for momentum and density (full PGF + buoyancy)
3. Update scalars using standard RK update with time-averaged velocities
"""
function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:CompressibleDynamics, <:Any, <:Any, <:AcousticRungeKutta3}, Δt; callbacks=[])
    # Be paranoid and update state at iteration 0
    maybe_prepare_first_time_step!(model, callbacks)

    ts = model.timestepper
    β₁ = ts.β₁
    β₂ = ts.β₂
    β₃ = ts.β₃

    # Compute the next time step a priori
    tⁿ⁺¹ = model.clock.time + Δt

    # Store u⁰ for use in all stages
    store_initial_state!(model)

    # Snapshot pressure into the substepper's frozen_pressure field. This is the
    # linearization point for cofwz/cofwt and stays the same across all WS-RK3
    # stages of this outer step. MPAS does the equivalent: its diag%exner is only
    # recomputed at rk_step==3, so the substepper sees the same exner at every
    # stage of an outer step.
    freeze_outer_step_state!(ts.substepper, model)

    #
    # Stage 1: U* = Uⁿ + (Δt/3) R(Uⁿ)
    #
    # We compute the slow tendencies at stage 1 from the outer-step-start state and
    # SNAPSHOT them. At stages 2 and 3 the snapshot is restored after compute_slow_*_
    # tendencies! has run, so the substep loop sees the same slow tendencies as stage 1.
    # This mimics MPAS, which computes tend_rho only at rk_step==1 and uses a
    # perturbation-form advection for tend_theta that vanishes when the wave
    # preserves θ. Tracer tendencies are NOT snapshotted, so moisture/scalar transport
    # picks up per-stage updates from compute_thermodynamic_tendency!.

    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₁; snapshot_slow_tendencies = true)

    tick_stage!(model.clock, β₁ * Δt)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₁ * Δt)

    #
    # Stage 2: U** = Uⁿ + (Δt/2) R(U*)
    #

    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₂; restore_slow_tendencies = true)

    tick_stage!(model.clock, (β₂ - β₁) * Δt)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₂ * Δt)

    #
    # Stage 3: Uⁿ⁺¹ = Uⁿ + Δt R(U**)
    #

    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₃; restore_slow_tendencies = true)

    # Adjust final time-step
    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₃ * Δt)

    return nothing
end
