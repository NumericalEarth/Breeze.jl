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
    WickerSkamarock3,
    stage_fractions,
    acoustic_rk3_substep_loop!,
    prepare_acoustic_cache!,
    freeze_outer_step_state!

"""
$(TYPEDEF)

Wicker–Skamarock third-order Runge–Kutta time stepper with linearized
acoustic substepping for fully compressible dynamics. Stage fractions
``β = (1/3, 1/2, 1)``. Each stage:

  1. Re-evaluates slow tendencies (advection + Coriolis + closure +
     forcing only — PGF and buoyancy are handled inside the substep
     loop in linearized form).
  2. Runs an inner substep loop that evolves linearized acoustic
     perturbations from the outer-step-start state.

The acoustic substep loop is in
[`acoustic_rk3_substep_loop!`](@ref); see
[`AcousticSubstepper`](@ref) for the substepper's storage and
parameters.

Fields
======

- `β₁, β₂, β₃`: Stage fractions (1/3, 1/2, 1).
- `U⁰`: Storage for state at the beginning of the outer time-step.
- `Gⁿ`: Slow-tendency fields, recomputed each stage.
- `implicit_solver`: Optional implicit solver for diffusion.
- `substepper`: [`AcousticSubstepper`](@ref) for the linearized acoustic
  substep loop.

References
==========

Wicker, L. J. & Skamarock, W. C. (2002). *Time-splitting methods for
elastic models using forward time schemes.* MWR 130, 2088–2097.
"""
struct AcousticRungeKutta3{FT, U0, TG, TI, AS} <: AbstractTimeStepper
    β₁ :: FT
    β₂ :: FT
    β₃ :: FT
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
    substepper :: AS
end

"""
    AcousticRungeKutta3(grid, prognostic_fields;
                        dynamics,
                        implicit_solver = nothing,
                        Gⁿ = map(similar, prognostic_fields))

Construct an `AcousticRungeKutta3` time stepper for fully compressible dynamics.
"""
function AcousticRungeKutta3(grid, prognostic_fields;
                             dynamics,
                             implicit_solver::TI = nothing,
                             Gⁿ::TG = map(similar, prognostic_fields)) where {TI, TG}

    FT = eltype(grid)

    # Stage fractions come from the outer-scheme interface. Today the only
    # supported subtype is WickerSkamarock3 with canonical (1/3, 1/2, 1);
    # a future MIS outer scheme would extend AcousticOuterScheme and provide
    # its own stage_fractions method.
    β = stage_fractions(WickerSkamarock3())
    β₁ = FT(β[1])
    β₂ = FT(β[2])
    β₃ = FT(β[3])

    U⁰ = map(similar, prognostic_fields)
    U0 = typeof(U⁰)

    substepper = AcousticSubstepper(grid, dynamics.time_discretization;
                                    prognostic_momentum = (ρu = prognostic_fields.ρu,
                                                            ρv = prognostic_fields.ρv,
                                                            ρw = prognostic_fields.ρw))
    AS = typeof(substepper)

    return AcousticRungeKutta3{FT, U0, TG, TI, AS}(β₁, β₂, β₃, U⁰, Gⁿ,
                                                    implicit_solver, substepper)
end

#####
##### Per-stage substep wrapper
#####

"""
$(TYPEDSIGNATURES)

Run one Wicker–Skamarock RK3 stage: compute slow tendencies, then
execute the linearized-acoustic substep loop, then update remaining
scalars.
"""
function acoustic_rk3_substep!(model, Δt, β)
    ts = model.timestepper
    substepper = ts.substepper
    U⁰ = ts.U⁰

    # Per-stage cache prep (currently a no-op — the linearization is
    # fixed at outer-step start).
    prepare_acoustic_cache!(substepper, model)

    # Slow tendencies (advection + Coriolis + diffusion; PGF and buoyancy
    # are excluded — those are handled inside the substep loop in
    # linearized form about the outer-step-start state).
    compute_slow_momentum_tendencies!(model)
    compute_slow_scalar_tendencies!(model)

    # Linearized acoustic substep loop: Nτ substeps of size Δτ = Δt/N.
    acoustic_rk3_substep_loop!(model, substepper, Δt, β, U⁰)

    # Update remaining scalars (tracers) using WS-RK3.
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
    @inbounds u[i, j, k] = u⁰[i, j, k] + Δt_stage * G[i, j, k]
end

#####
##### Time stepping (main entry point)
#####

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step `Δt` with Wicker–Skamarock RK3 and
linearized acoustic substepping.
"""
function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:CompressibleDynamics, <:Any, <:Any, <:AcousticRungeKutta3}, Δt; callbacks=[])

    maybe_prepare_first_time_step!(model, callbacks)

    ts = model.timestepper
    β₁ = ts.β₁
    β₂ = ts.β₂
    β₃ = ts.β₃

    tⁿ⁺¹ = model.clock.time + Δt

    # Snapshot outer-step-start state into U⁰ for use in all stages,
    # and into the substepper's outer-step fields as the linearization
    # point.
    store_initial_state!(model)
    freeze_outer_step_state!(ts.substepper, model)

    # Stage 1: U* = Uⁿ + (Δt/3) R(Uⁿ)
    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₁)

    tick_stage!(model.clock, β₁ * Δt)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₁ * Δt)

    # Stage 2: U** = Uⁿ + (Δt/2) R(U*)
    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₂)

    tick_stage!(model.clock, (β₂ - β₁) * Δt)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₂ * Δt)

    # Stage 3: Uⁿ⁺¹ = Uⁿ + Δt R(U**)
    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₃)

    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)

    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₃ * Δt)

    return nothing
end
