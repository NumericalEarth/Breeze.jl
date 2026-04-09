using KernelAbstractions: @kernel, @index

using Oceananigans.Utils: time_difference_seconds

using Oceananigans.TimeSteppers:
    AbstractTimeStepper,
    tick_stage!,
    update_state!,
    compute_flux_bc_tendencies!,
    step_lagrangian_particles!

using Oceananigans.TurbulenceClosures: step_closure_prognostics!

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

Unlike [`AcousticSSPRungeKutta3`](@ref) which uses convex combinations,
this scheme uses stage fractions ``Δt/3``, ``Δt/2``, and ``Δt``:

- Stage 1: ``U^* = U^n + (Δt/3) \\, R(U^n)``       — ``N/3`` substeps
- Stage 2: ``U^{**} = U^n + (Δt/2) \\, R(U^*)``    — ``N/2`` substeps
- Stage 3: ``U^{n+1} = U^n + Δt \\, R(U^{**})``    — ``N`` substeps

Each stage evaluates the RHS, ``R``, at the current stage state, then resets to ``U^n``
and advances by ``β Δt``. The absence of convex combinations makes this scheme
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
    physics_substeps :: Int
end

"""
    AcousticRungeKutta3(grid, prognostic_fields;
                        dynamics,
                        implicit_solver = nothing,
                        Gⁿ = map(similar, prognostic_fields),
                        physics_substeps = 1)

Construct an `AcousticRungeKutta3` time stepper for fully compressible dynamics.

Keyword Arguments
=================

- `dynamics`: The [`CompressibleDynamics`](@ref) object containing the `time_discretization`.
- `implicit_solver`: Optional implicit solver for diffusion. Default: `nothing`
- `Gⁿ`: Tendency fields at current stage. Default: similar to `prognostic_fields`
- `physics_substeps`: Number of microphysics fractional-step substeps applied
  *after* each WS-RK3 outer step (operator splitting). When `> 1`, the WS-RK3
  dynamics is computed with the microphysics tendency suppressed (so dynamics
  sees only advection + closure + forcing for the moisture and ρθ tracers),
  and microphysics is then applied in `physics_substeps` explicit forward-Euler
  substeps of size `Δt / physics_substeps`, with the EOS state updated between
  each substep. This decouples the microphysics relaxation timescale from the
  outer Δt and lets the moist BW run at the dynamics advective CFL even when
  the rain autoconversion + accretion processes have an effective timescale
  shorter than the outer Δt. Default: 1 (no operator splitting; microphysics
  is integrated as part of the WS-RK3 stages, the original behavior).
"""
function AcousticRungeKutta3(grid, prognostic_fields;
                             dynamics,
                             implicit_solver::TI = nothing,
                             Gⁿ::TG = map(similar, prognostic_fields),
                             physics_substeps::Int = 1) where {TI, TG}

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

    # Snapshot storage for the slow tendencies that are frozen across stages.
    # Populated at stage 1 from the outer-step-start state and restored at
    # stages 2 and 3 to mimic MPAS's freeze-tend_rho-at-rk_step==1 and
    # freeze-tend_u_euler-at-rk_step==1 behavior. We freeze the density
    # tendency (`ρ`) and the horizontal PGF (stored on the `ρu` / `ρv`
    # buffers, populated by `snapshot_horizontal_pgf!`). The thermodynamic
    # tendency `Gⁿ.ρθ` is intentionally **not** snapshotted — see the
    # freeze-policy comment in `acoustic_substep_helpers.jl`. Tracer entries
    # are also not stored: they are recomputed per stage by
    # `compute_thermodynamic_tendency!`.
    slow_tendency_snapshot = (ρ  = similar(Gⁿ.ρ),
                              ρu = similar(Gⁿ.ρu),
                              ρv = similar(Gⁿ.ρv))
    SS = typeof(slow_tendency_snapshot)

    return AcousticRungeKutta3{FT, U0, TG, TI, AS, SS}(β₁, β₂, β₃, U⁰, Gⁿ,
                                                       implicit_solver, substepper,
                                                       slow_tendency_snapshot,
                                                       physics_substeps)
end

#####
##### WS-RK3 substep with acoustic substepping
#####

"""
$(TYPEDSIGNATURES)

Apply a Wicker-Skamarock RK3 substep with acoustic substepping.

The acoustic substep loop handles momentum, density, and the thermodynamic
variable (``ρθ`` or ``ρe``). The substep size is constant ``Δτ = Δt / N`` across all
stages, with the substep count varying as ``N_τ = \\mathrm{round}(β N)``.
Remaining scalars (tracers) are updated with standard RK3.

The `freeze` keyword controls the [slow-tendency freeze
policy](@ref `snapshot_slow_tendencies!`):

- `:snapshot` — snapshot ``G^n.\\rho`` and the horizontal PGF at this stage
  (use at outer-step stage 1). ``G^n.\\rho\\theta`` is **not** snapshotted —
  it must be recomputed every stage so that microphysics latent heat sees
  the per-stage state.
- `:restore`  — restore the snapshotted ``G^n.\\rho`` after the per-stage
  `compute_slow_*_tendencies!` calls (use at stages 2, 3).
- `:none`     — neither snapshot nor restore.
"""
function acoustic_rk3_substep!(model, Δt, β; freeze::Symbol = :none, physics_split::Bool = false)
    ts = model.timestepper
    substepper = ts.substepper
    U⁰ = ts.U⁰
    snap = ts.slow_tendency_snapshot

    # Prepare stage-frozen reference state (needed by slow tendency computation)
    prepare_acoustic_cache!(substepper, model)

    # Compute slow momentum tendencies (advection, Coriolis, diffusion — PGF/buoyancy
    # handled by the acoustic loop via tend_w_euler in convert_slow_tendencies!).
    compute_slow_momentum_tendencies!(model)

    # Compute slow density and thermodynamic tendencies (with microphysics
    # tendency suppressed if the time stepper is operator-splitting physics).
    compute_slow_scalar_tendencies!(model; physics_split=physics_split)

    # Slow-tendency freeze policy. See `snapshot_slow_tendencies!` for the
    # MPAS-equivalence rationale.
    freeze === :snapshot && snapshot_slow_tendencies!(snap, model)
    freeze === :restore  && restore_slow_tendencies!(snap, model)
    add_horizontal_pgf!(ts.Gⁿ.ρu, ts.Gⁿ.ρv, snap.ρu, snap.ρv, model)

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
- Outer loop: canonical 3-stage RK3 with stage fractions ``Δt/3``, ``Δt/2``, ``Δt``
- Inner loop: Acoustic substeps for fast (pressure) tendencies, with constant
  substep size ``Δτ = Δt/N`` across all stages (``N/3``, ``N/2``, ``N`` substeps in
  stages 1, 2, 3 respectively)

Each RK stage:
1. Compute slow tendencies (advection, Coriolis, diffusion only — PGF/buoyancy in acoustic loop)
2. Execute acoustic substep loop for momentum and density (full PGF + buoyancy)
3. Update scalars using standard RK update with time-averaged velocities
"""
function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:CompressibleDynamics, <:Any, <:Any, <:AcousticRungeKutta3}, Δt; callbacks=[])

    # Be paranoid and prepare at iteration 0, in case run! is not used:
    maybe_prepare_first_time_step!(model, callbacks)

    ts = model.timestepper
    β₁ = ts.β₁
    β₂ = ts.β₂
    β₃ = ts.β₃

    # Operator-splitting hook: when `physics_substeps > 1`, the WS-RK3
    # dynamics is computed with the microphysics tendency suppressed in
    # `compute_slow_scalar_tendencies!` / `compute_tendencies!` (so the moist
    # tracers and ρθ see only advection, closure, and forcing during the
    # outer step) and microphysics is then applied as a `physics_substeps`-
    # subcycled fractional step at the end. This decouples the microphysics
    # relaxation timescale from the outer Δt and lets the moist BW run at the
    # dynamics advective CFL even when the rain process timescales (~10 s)
    # are shorter than the outer Δt.
    physics_split = ts.physics_substeps > 1

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
    # We compute the slow tendencies at stage 1 from the outer-step-start state
    # and SNAPSHOT Gⁿ.ρ and the horizontal PGF. At stages 2 and 3 those frozen
    # values are restored after compute_slow_*_tendencies! has run, so the
    # substep loop sees the same density and horizontal PGF tendencies as stage 1
    # (MPAS computes tend_rho and tend_u_euler only at rk_step==1 and freezes
    # them for the rest of the outer step). Gⁿ.ρθ is intentionally recomputed
    # every stage so microphysics latent heat picks up the per-stage state —
    # see the freeze-policy comment in acoustic_substep_helpers.jl. Tracer
    # tendencies are also recomputed per stage by compute_thermodynamic_tendency!.

    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₁; freeze = :snapshot, physics_split=physics_split)

    tick_stage!(model.clock, β₁ * Δt)
    update_state!(model, callbacks; compute_tendencies = true, physics_split=physics_split)
    step_lagrangian_particles!(model, β₁ * Δt)

    #
    # Stage 2: U** = Uⁿ + (Δt/2) R(U*)
    #

    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₂; freeze = :restore, physics_split=physics_split)

    tick_stage!(model.clock, (β₂ - β₁) * Δt)
    update_state!(model, callbacks; compute_tendencies = true, physics_split=physics_split)
    step_lagrangian_particles!(model, β₂ * Δt)

    #
    # Stage 3: Uⁿ⁺¹ = Uⁿ + Δt R(U**)
    #

    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₃; freeze = :restore, physics_split=physics_split)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)

    # Adjust final time-step
    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    #
    # Phase 2: physics fractional step (when `physics_substeps > 1`).
    #
    # We've now advanced the dynamics by `Δt` with microphysics suppressed.
    # Apply microphysics as `N` explicit forward-Euler substeps of size
    # `Δτ = Δt / N`, with `update_state!` (full microphysics) between each
    # substep so the EOS recovery uses the updated condensate amounts and
    # the next substep's microphysics tendency sees the post-condensation
    # T, p, qᵛ. The final `update_state!` after the loop produces the
    # correct end-of-step T, p with full microphysics, exactly as the
    # non-split path would have.
    #
    if physics_split
        N = ts.physics_substeps
        Δτ = Δt / N
        for _ in 1:N
            update_state!(model, callbacks; compute_tendencies = false)
            apply_microphysics_substep!(model, Δτ)
        end
    end

    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₃ * Δt)

    return nothing
end
