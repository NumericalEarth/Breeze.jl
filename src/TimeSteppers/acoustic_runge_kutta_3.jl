using KernelAbstractions: @kernel, @index

using Oceananigans.Utils: time_difference_seconds

using Oceananigans.TimeSteppers:
    AbstractTimeStepper,
    tick_stage!,
    update_state!,
    compute_flux_bc_tendencies!,
    step_lagrangian_particles!

using Oceananigans.TurbulenceClosures: step_closure_prognostics!

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, microphysics_model_update!

using Breeze.CompressibleEquations:
    CompressibleDynamics,
    TerrainCompressibleDynamics,
    AcousticSubstepper,
    WickerSkamarock3,
    stage_fractions,
    acoustic_rk3_substep_loop!,
    prepare_acoustic_cache!,
    freeze_linearization_state!

"""
$(TYPEDEF)

Wicker–Skamarock third-order Runge–Kutta time stepper with linearized
acoustic substepping for fully compressible dynamics. Stage fractions
``β = (1/3, 1/2, 1)``. Each stage:

  1. Re-evaluates slow tendencies (advection + Coriolis + closure +
     forcing only — PGF and buoyancy are handled inside the substep
     loop in linearized form).
  2. Runs an inner substep loop that evolves linearized acoustic
     perturbations from the RK stage-entry state, initialized with a
     rewind term so every stage still advances from the outer-step-start
     prognostic state.

The acoustic substep loop is in
[`acoustic_rk3_substep_loop!`](@ref); see
[`AcousticSubstepper`](@ref) for the substepper's storage and
parameters.

Fields
======

- `β₁, β₂, β₃`: Stage fractions (1/3, 1/2, 1).
- `U⁰`: Storage for state at the beginning of the outer time-step.
- `Gⁿ`: Slow-tendency fields, recomputed each stage.
- `implicit_solver`: Optional solver for the vertically-implicit pieces — closure diffusion
  and the adaptive-implicit vertical-advection remainder.
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
                        Gⁿ = map(similar, prognostic_fields),
                        U⁰ = map(similar, prognostic_fields))

Construct an `AcousticRungeKutta3` time stepper for fully compressible dynamics.

`Gⁿ` and `U⁰` may be supplied to alias another stepper's tendency storage instead of
allocating fresh fields (used by the native-stepper adiabatic-balance twin).
"""
function AcousticRungeKutta3(grid, prognostic_fields;
                             dynamics,
                             implicit_solver::TI = nothing,
                             Gⁿ::TG = map(similar, prognostic_fields),
                             U⁰::U0 = map(similar, prognostic_fields)) where {TI, TG, U0}

    FT = eltype(grid)

    # Stage fractions come from the outer-scheme interface. Today the only
    # supported subtype is WickerSkamarock3 with canonical (1/3, 1/2, 1);
    # a future MIS outer scheme would extend AcousticOuterScheme and provide
    # its own stage_fractions method.
    β = stage_fractions(WickerSkamarock3())
    β₁ = FT(β[1])
    β₂ = FT(β[2])
    β₃ = FT(β[3])

    substepper = AcousticSubstepper(grid, dynamics.time_discretization;
                                    prognostic_momentum = (ρu = prognostic_fields.ρu,
                                                           ρv = prognostic_fields.ρv,
                                                           ρw = prognostic_fields.ρw))
    AS = typeof(substepper)

    return AcousticRungeKutta3{FT, U0, TG, TI, AS}(β₁, β₂, β₃, U⁰, Gⁿ,
                                                    implicit_solver, substepper)
end

#####
##### Adaptive-implicit vertical advection split time step
#####

# Stage fractions in order, and the increment each stage adds to the clock.
@inline stage_fraction(ts::AcousticRungeKutta3, ::Val{1}) = ts.β₁
@inline stage_fraction(ts::AcousticRungeKutta3, ::Val{2}) = ts.β₂
@inline stage_fraction(ts::AcousticRungeKutta3, ::Val{3}) = ts.β₃
@inline stage_increment(ts::AcousticRungeKutta3, ::Val{1}) = ts.β₁
@inline stage_increment(ts::AcousticRungeKutta3, ::Val{2}) = ts.β₂ - ts.β₁
@inline stage_increment(ts::AcousticRungeKutta3, ::Val{3}) = 1 - ts.β₂

"""
$(TYPEDSIGNATURES)

Set the adaptive-implicit split time step to the interval of the *next* Wicker–Skamarock
stage, so the explicit velocity fraction frozen into `Gⁿ` by the tendencies computed here
pairs with the implicit fraction the next stage applies to them.

Mirrors Oceananigans' `RungeKutta3TimeStepper` and `SplitRungeKuttaTimeStepper`
specializations, including recovering the outer step from the stage increment the clock was
just ticked by. Every field — the acoustic prognostics and the tracers alike — then reads one
value written at one point in the update cycle.

Stage 1 of step ``n`` is written at the end of step ``n-1``, before ``Δtₙ`` is known, so it
uses ``β₁ Δtₙ₋₁``. This lag is deliberate: `scalar_substep!` consumes tracer tendencies whose
explicit fraction was frozen against the value in force when they were built, so rewriting the
split at stage entry would scale the explicit half by ``s(Δt_old)`` and the implicit half by
``1 - s(Δt_new)`` — the desynchronization this specialization exists to remove. Consistency is
preserved; only CFL targeting suffers, and only on stage 1 of a step whose ``Δt`` changed: a
growth by factor ``r`` drives that stage's explicit vertical Courant number to ``r`` times the
requested `cfl`, capped by the unsplit Courant number.
"""
@inline function Oceananigans.Advection.update_advection_timestep!(a::AdaptiveImplicitVerticalAdvection,
                                                                   timestepper::AcousticRungeKutta3, clock)
    td = OceananigansTimeSteppers.time_discretization(a)

    # `clock.stage` names the stage about to run; `clock.last_stage_Δt` is the increment the
    # stage before it contributed. Recover the outer step from that increment, then scale by
    # the upcoming stage's fraction.
    stage = clock.stage
    completed = ifelse(stage == 1, 3, stage - 1)
    Δt = advection_outer_timestep(timestepper, completed, clock.last_stage_Δt)

    # `maybe_prepare_first_time_step!` seeds `last_stage_Δt`, so the fallback is unreachable in
    # normal use; it keeps a stage fraction rather than a whole step if a clock arrives unseeded.
    Δt_stage = advection_stage_timestep(timestepper, stage, Δt)
    Δt_last = advection_stage_timestep(timestepper, stage, clock.last_Δt)
    td.Δt[] = ifelse(isfinite(Δt_stage), Δt_stage, Δt_last)
    return nothing
end

# A `FluxFormAdvection` whose z-scheme is adaptive-implicit is itself an
# `AdaptiveImplicitVerticalAdvection` (the alias keys on the z time discretization), so this
# disambiguates against Oceananigans' `(::FluxFormAdvection, timestepper, clock)`.
Oceananigans.Advection.update_advection_timestep!(a::FluxFormAdvection, timestepper::AcousticRungeKutta3, clock) =
    Oceananigans.Advection.update_advection_timestep!(a.z, timestepper, clock)

@inline advection_outer_timestep(ts, stage, Δt_increment) =
    Δt_increment / (stage == 1 ? stage_increment(ts, Val(1)) :
                    stage == 2 ? stage_increment(ts, Val(2)) : stage_increment(ts, Val(3)))

@inline advection_stage_timestep(ts, stage, Δt) =
    Δt * (stage == 1 ? stage_fraction(ts, Val(1)) :
          stage == 2 ? stage_fraction(ts, Val(2)) : stage_fraction(ts, Val(3)))

#####
##### Per-stage substep wrapper
#####

"""
$(TYPEDSIGNATURES)

Run one Wicker–Skamarock RK3 stage: compute slow tendencies, then
execute the linearized-acoustic substep loop, then update remaining
scalars.
"""
function acoustic_rk3_substep!(model::AtmosphereModel, Δt, β)
    ts = model.timestepper
    substepper = ts.substepper
    U⁰ = ts.U⁰

    # Per-stage cache prep. The linearization coefficients are refreshed
    # to the current RK stage-entry state; the substep perturbations are
    # initialized with a rewind term inside the substep loop so the
    # WS-RK3 invariant still starts from Uⁿ.
    prepare_acoustic_cache!(substepper, model)

    # Slow tendencies (advection + Coriolis + diffusion; PGF and buoyancy
    # are excluded — those are handled inside the substep loop in
    # linearized form about the RK stage-entry state).
    compute_slow_momentum_tendencies!(model)
    compute_slow_scalar_tendencies!(model)

    # Boundary flux tendencies must be added after the stage tendencies are
    # assembled. Adding them before this function would be overwritten by
    # compute_slow_momentum_tendencies! / compute_slow_scalar_tendencies!.
    compute_flux_bc_tendencies!(model)

    # Linearized acoustic substep loop: Nτ substeps of size Δτ = Δt/N.
    acoustic_rk3_substep_loop!(model, substepper, Δt, β, U⁰)

    # Vertically-implicit solve for the acoustic prognostics (momentum and the thermodynamic
    # variable) over the stage interval β Δt: the implicit remainder of adaptive implicit
    # vertical advection combined with vertically-implicit closure diffusion. A no-op when
    # the timestepper has no implicit solver.
    implicit_substep!(model, β * Δt)

    # Update remaining scalars (tracers) using WS-RK3.
    scalar_rk3_substep!(model, β * Δt)

    return nothing
end

#####
##### Scalar update with time-averaged velocities
#####

function scalar_rk3_substep!(model::AtmosphereModel, Δt_stage)
    grid = model.grid
    Δt_FT = kernel_time_step(grid.architecture, grid, Δt_stage)
    return scalar_substep!(model, _rk3_substep!, Δt_stage, Δt_FT)
end

@kernel function _rk3_substep!(u, u⁰, G, Δt_stage)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = u⁰[i, j, k] + Δt_stage * G[i, j, k]
end

#####
##### Time stepping (main entry point)
#####

"""
$(TYPEDSIGNATURES)

Seed `clock.last_stage_Δt` before the first step so that stage 1 sees a stage interval rather
than a whole outer step. The seed is the increment a completed third stage leaves, ``(1 - β₂)
Δt``, which is exactly what `update_advection_timestep!` inverts to recover `Δt` at stage 1.

Also fires on a clock carrying a non-finite `last_stage_Δt` at nonzero iteration, which is what
a hand-assembled or externally-restored clock looks like. `PerturbationAdvection` open boundary
conditions read the same clock field, so seeding it also gives them a finite relaxation
interval on the first stage.
"""
function OceananigansTimeSteppers.maybe_prepare_first_time_step!(model::AtmosphereModel{<:CompressibleDynamics, <:Any, <:Any, <:AcousticRungeKutta3}, Δt, callbacks)
    clock = model.clock
    if clock.iteration == 0 || !isfinite(clock.last_stage_Δt)
        clock.last_Δt = Δt
        clock.last_stage_Δt = stage_increment(model.timestepper, Val(3)) * Δt
        reconcile_state!(model)
        update_state!(model, callbacks)
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step `Δt` with Wicker–Skamarock RK3 and
linearized acoustic substepping.
"""
function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:CompressibleDynamics, <:Any, <:Any, <:AcousticRungeKutta3}, Δt; callbacks=[])

    maybe_prepare_first_time_step!(model, Δt, callbacks)

    ts = model.timestepper
    β₁ = ts.β₁
    β₂ = ts.β₂
    β₃ = ts.β₃

    tⁿ⁺¹ = model.clock.time + Δt

    # Snapshot outer-step-start state into U⁰ for use by all stages.
    # The substepper also seeds its stage-1 cache and time-averaged
    # transport velocity from this state.
    store_initial_state!(model)
    freeze_linearization_state!(ts.substepper, model)

    # Stage 1: U* = Uⁿ + (Δt/3) R(Uⁿ)
    acoustic_rk3_substep!(model, Δt, β₁)

    tick_stage!(model.clock, β₁ * Δt)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₁ * Δt)

    # Stage 2: U** = Uⁿ + (Δt/2) R(U*)
    acoustic_rk3_substep!(model, Δt, β₂)

    tick_stage!(model.clock, (β₂ - β₁) * Δt)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₂ * Δt)

    # Stage 3: Uⁿ⁺¹ = Uⁿ + Δt R(U**)
    acoustic_rk3_substep!(model, Δt, β₃)

    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)

    update_state!(model, callbacks; compute_tendencies = true)

    # Apply the operator-split microphysics update exactly once per step, on the post-RK
    # state just refreshed by `update_state!`. A no-op for tendency-interface schemes.
    microphysics_model_update!(model.microphysics, model)

    step_lagrangian_particles!(model, β₃ * Δt)

    return nothing
end

#####
##### Time-averaged velocity for scalar transport (WRF/MPAS dynamics-transport split)
#####
##### Override `transport_velocities` to return the velocity averaged over
##### the previous stage's substep loop. The MAIN consumer is
##### `update_state!`'s `compute_tendencies!` between RK stages, which
##### computes Gⁿ for **moisture, tracers, chemistry, TKE** using
##### `transport_velocities(model)` — those tendencies are then applied by
##### the next stage's `scalar_rk3_substep!` (or the next outer step's stage
##### 1). This matches WRF's `rk_scalar_tend` calls with
##### `grid%ru_m, grid%rv_m, grid%ww_m` and MPAS's `ruAvg`-driven
##### tracer transport.
#####
##### Theta's slow tendency does NOT consume this — `compute_slow_scalar_tendencies!`
##### deliberately passes `model.velocities` (matching WRF's `rk_tendency`).
##### Mixing the two paths creates a feedback loop that destabilizes a rest
##### atmosphere at production Δt.
#####
##### For stage 1 (no prior substep loop in this outer step),
##### `freeze_linearization_state!` seeded the field with `model.velocities`
##### at outer-step start.
#####

function AtmosphereModels.transport_velocities(model::AtmosphereModel{<:CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                                                                      <:Any, <:Any, <:AcousticRungeKutta3})
    sub = model.timestepper.substepper
    return (u = sub.time_averaged_velocities.u,
            v = sub.time_averaged_velocities.v,
            w = sub.time_averaged_velocities.w)
end

function AtmosphereModels.transport_velocities(model::AtmosphereModel{<:TerrainCompressibleDynamics,
                                                                      <:Any, <:Any, <:AcousticRungeKutta3})
    sub = model.timestepper.substepper
    return (u = sub.time_averaged_velocities.u,
            v = sub.time_averaged_velocities.v,
            w = sub.time_averaged_velocities.w)
end
