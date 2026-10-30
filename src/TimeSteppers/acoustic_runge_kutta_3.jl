using KernelAbstractions: @kernel, @index

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: CenterField
using Oceananigans.Utils: time_difference_seconds

using Oceananigans.TimeSteppers:
    AbstractTimeStepper,
    tick_stage!,
    update_state!,
    compute_flux_bc_tendencies!,
    step_lagrangian_particles!

using Oceananigans.TurbulenceClosures: step_closure_prognostics!

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, SlowDynamics, microphysics_model_update!,
                                compute_closure_tendencies!, thermodynamic_density, add_sedimentation_tendency_parts!,
                                add_implicit_sedimentation_tendency_parts!

using Breeze.CompressibleEquations:
    CompressibleDynamics,
    TerrainCompressibleDynamics,
    AcousticSubstepper,
    WickerSkamarock3,
    stage_fractions,
    acoustic_rk3_substep_loop!,
    prepare_acoustic_cache!,
    freeze_linearization_state!,
    seed_time_averaged_velocities!

"""
$(TYPEDEF)

Wicker–Skamarock third-order Runge–Kutta time stepper with linearized
acoustic substepping for fully compressible dynamics. Stage fractions
``β = (1/3, 1/2, 1)``. Each stage:

  1. Re-evaluates slow tendencies (advection + Coriolis + closure +
     forcing only — PGF and buoyancy are handled inside the substep
     loop in linearized form).
  2. Advances moisture and tracers, their vertically-implicit solves included,
     with the acoustic-mean velocity of the previous substep loop.
  3. Runs an inner substep loop that evolves linearized acoustic
     perturbations from the RK stage-entry state, initialized with a
     rewind term so every stage still advances from the outer-step-start
     prognostic state.
  4. Adds the composition content that the sedimenting condensate moved,
     which the substep loop never sees (see `remove_sedimentation_content!`),
     then solves the vertically-implicit pieces of momentum and the
     thermodynamic variable, the content taking the variable's solve in a
     solve of its own.

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
- `sedimentation_content`: The composition content of the thermodynamic variable that
  condensate sedimentation moves in the current stage, kept out of the substep loop; `nothing`
  when no condensate sediments.
- `frozen_thermodynamic_density`: A copy of the thermodynamic variable taken after the stage's
  sedimentation content is added and before the post-loop implicit solves, from which the
  variable's solve and the content's read their coefficients; `nothing` when no condensate
  sediments or nothing about the variable is vertically implicit, in which case the stage runs
  no such solve (see `thermodynamic_implicit_substep!`).

References
==========

Wicker, L. J. & Skamarock, W. C. (2002). *Time-splitting methods for
elastic models using forward time schemes.* MWR 130, 2088–2097.
"""
struct AcousticRungeKutta3{FT, U0, TG, TI, AS, SC, FD} <: AbstractTimeStepper
    β₁ :: FT
    β₂ :: FT
    β₃ :: FT
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
    substepper :: AS
    sedimentation_content :: SC
    frozen_thermodynamic_density :: FD
end

"""
    AcousticRungeKutta3(grid, prognostic_fields;
                        dynamics,
                        implicit_solver = nothing,
                        sedimentation = NamedTuple(),
                        thermodynamic_variable_implicit = false,
                        Gⁿ = map(similar, prognostic_fields),
                        U⁰ = map(similar, prognostic_fields))

Construct an `AcousticRungeKutta3` time stepper for fully compressible dynamics.

`sedimentation` holds the model's sedimenting condensates (see `materialize_sedimentation`);
when there are any, the stepper allocates a field for the content they move each stage, and,
when `thermodynamic_variable_implicit` says something about the thermodynamic variable is
vertically implicit (its closure diffusion or a boundary flux; its adaptive-implicit advection is
applied inside the substep loop), one for the copy of the variable that the content's post-loop
solve and the variable's own read their coefficients from. That copy is what makes the stage run
those solves (see `thermodynamic_implicit_substep!`). An `implicit_solver` alone does not call for
it: the moisture and tracers may be all that is vertically implicit.
`Gⁿ` and `U⁰` may be supplied to alias another stepper's tendency storage instead of
allocating fresh fields (used by the native-stepper adiabatic-balance twin).
"""
function AcousticRungeKutta3(grid, prognostic_fields;
                             dynamics,
                             implicit_solver::TI = nothing,
                             cache_advecting_state = false,
                             sedimentation = NamedTuple(),
                             thermodynamic_variable_implicit = false,
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
                                    cache_advecting_state,
                                    prognostic_momentum = (ρu = prognostic_fields.ρu,
                                                           ρv = prognostic_fields.ρv,
                                                           ρw = prognostic_fields.ρw))
    AS = typeof(substepper)

    sedimentation_content = isempty(sedimentation) ? nothing : CenterField(grid)
    SC = typeof(sedimentation_content)

    frozen_thermodynamic_density = isempty(sedimentation) || !thermodynamic_variable_implicit ? nothing : CenterField(grid)
    FD = typeof(frozen_thermodynamic_density)

    return AcousticRungeKutta3{FT, U0, TG, TI, AS, SC, FD}(β₁, β₂, β₃, U⁰, Gⁿ,
                                                            implicit_solver, substepper,
                                                            sedimentation_content,
                                                            frozen_thermodynamic_density)
end

#####
##### Adaptive-implicit vertical advection split time step
#####

# An `AtmosphereModel` stepped by `AcousticRungeKutta3`; the free parameter is the
# architecture, so `BreezeReactantExt` can specialize on `{<:ReactantState}`.
const CompressibleAcousticModel{Arc} = AtmosphereModel{<:CompressibleDynamics, <:Any, Arc, <:AcousticRungeKutta3}

# Stage fractions, and the increment each stage adds to the clock. `ifelse` rather than
# branching: `clock.stage` is traced under Reactant.
@inline stage_fraction(ts, stage) = ifelse(stage == 1, ts.β₁, ifelse(stage == 2, ts.β₂, ts.β₃))
@inline stage_increment(ts, stage) = ifelse(stage == 1, ts.β₁, ifelse(stage == 2, ts.β₂ - ts.β₁, 1 - ts.β₂))

"""
$(TYPEDSIGNATURES)

The adaptive-implicit split time step for the *next* Wicker–Skamarock stage, so the explicit
velocity fraction frozen into `Gⁿ` pairs with the implicit fraction the next stage applies.
Stage 1 of step ``n`` is evaluated before ``Δtₙ`` is known and deliberately uses ``β₁ Δtₙ₋₁``;
the cost is confined to CFL targeting on one stage when ``Δt`` changes (see
`maybe_prepare_first_time_step!` for the cold-start seeding).
"""
@inline function Oceananigans.Advection.adaptive_advection_timestep(timestepper::AcousticRungeKutta3, clock)
    stage = clock.stage
    completed = ifelse(stage == 1, 3, stage - 1)
    Δt = clock.last_stage_Δt / stage_increment(timestepper, completed)
    Δt_stage = stage_fraction(timestepper, stage) * Δt
    Δt_last = stage_fraction(timestepper, stage) * clock.last_Δt
    return ifelse(isfinite(Δt_stage), Δt_stage, Δt_last)
end

#####
##### Per-stage substep wrapper
#####

"""
$(TYPEDSIGNATURES)

Run Wicker–Skamarock RK3 `stage` (1, 2 or 3) with stage fraction `β`: complete the slow
tendencies built by `update_state!`, update the remaining scalars, execute the linearized-acoustic
substep loop, add the stage's condensate sedimentation composition content to the thermodynamic
variable, then run the post-loop implicit solves.
"""
function acoustic_rk3_substep!(model::AtmosphereModel, Δt, β, stage)
    ts = model.timestepper
    substepper = ts.substepper
    U⁰ = ts.U⁰

    # `update_state!` built the slow tendencies (advection + Coriolis + diffusion; PGF and
    # buoyancy are excluded — those are handled inside the substep loop in linearized form
    # about the RK stage-entry state). Add the boundary flux and closure tendencies to them,
    # before the content removal below: a boundary condition or a closure may read the
    # thermodynamic variable itself, and must see the full stage-entry state.
    compute_flux_bc_tendencies!(model)
    compute_closure_tendencies!(model)

    # Take the previous stage's sedimentation composition content back out of the thermodynamic
    # variable, so that neither the linearization nor the rewind below reads it as a pressure
    # change (see `remove_sedimentation_content!`). Nothing is pending in stage 1.
    remove_sedimentation_content!(model, stage)

    # Per-stage cache prep. The linearization coefficients are refreshed
    # to the current RK stage-entry state; the substep perturbations are
    # initialized with a rewind term inside the substep loop so the
    # WS-RK3 invariant still starts from Uⁿ.
    prepare_acoustic_cache!(substepper, model)
    cache_advecting_state!(model)

    # Base-state part of the IMEX vertical-advection split's implicit half (a no-op unless
    # the thermodynamic scheme is adaptive-implicit); the perturbation part is solved per
    # substep inside the loop, dispatched on the scheme type.
    add_implicit_advection_tendency!(model)

    # This stage's sedimentation content, sensible heat into the slow tendency and composition
    # content into the stage's own field: the explicit part now, from the tracers' frozen
    # transport velocity, and the implicit remainder's in `scalar_rk3_substep!`.
    add_explicit_sedimentation_content!(model)

    # Update remaining scalars (tracers) using WS-RK3, their implicit solves included. Their
    # tendencies and the transport velocity their solves split were frozen before this stage, so
    # they need nothing from the substep loop, and they run before it so that the content of the
    # mass their solves sediment reaches the loop's slow tendency and the stage's own field in time.
    # They follow the removal and the cache prep because the linearization's mixture coefficients
    # read the stage-entry moisture, which they change; a tracer closure coefficient that reads
    # the thermodynamic variable itself therefore sees it with the content removed.
    scalar_rk3_substep!(model, β * Δt)

    # Linearized acoustic substep loop: Nτ substeps of size Δτ = Δt/N.
    θ_advection = field_advection_scheme(model.advection, thermodynamic_density_name(model.formulation))
    acoustic_rk3_substep_loop!(model, substepper, Δt, β, U⁰, θ_advection)

    # Add the stage's sedimentation composition content over the stage interval, so that the
    # solves below see the full state. The variable's implicit vertical transport was applied
    # inside the substep loop, without it.
    apply_sedimentation_content!(model, β * Δt)

    # Vertically-implicit solve for the acoustic prognostics (momentum and the thermodynamic
    # variable) over the stage interval β Δt: the implicit remainder of adaptive implicit
    # vertical advection combined with vertically-implicit closure diffusion. The stage's
    # sedimentation composition content takes the thermodynamic variable's solve too, in a solve
    # of its own, so that the next stage takes out exactly what the variable received
    # (`thermodynamic_implicit_substep!`). A no-op when the timestepper has no implicit solver.
    implicit_substep!(model, β * Δt)

    return nothing
end

#####
##### Scalar update with time-averaged velocities
#####

function scalar_rk3_substep!(model::AtmosphereModel, Δt_stage)
    grid = model.grid
    kernel_Δt = kernel_time_step(grid.architecture, grid, Δt_stage)
    return scalar_substep!(model, _rk3_substep!, Δt_stage, kernel_Δt)
end

@kernel function _rk3_substep!(u, u⁰, G, Δt_stage)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = u⁰[i, j, k] + Δt_stage * G[i, j, k]
end

#####
##### Condensate sedimentation content, kept out of the acoustic substep loop
#####
##### The loop forms its pressure perturbation from the thermodynamic perturbation alone,
##### p′ = γᵐ Rᵐ Πᴸ (ρθ)′, so it reads any change of ρθ as a change of pressure. The content of
##### ρθˡⁱ that sedimenting condensate moves has two parts (see `sedimentation_tendency`). The
##### sensible heat the arriving mass brings changes the temperature and the pressure, like any
##### heating, so it joins the slow tendency the loop integrates. The composition content (mostly
##### the latent deficit −ℒ q / (cᵖᵐ Π); see `condensate_content`) changes ρθ at fixed
##### temperature, and the gas pressure (ρᵈ Rᵈ + ρᵛ Rᵛ) T does not change. Fed to the loop, it
##### reads as a pressure perturbation of about γᵐ Rᵐ ℒ / cᵖᵐ per unit condensate density. Every
##### stage after the first starts its perturbations from the rewind Uⁿ − Uᴸ_stage, which carries
##### the previous stage's content with the opposite sign: with explicit sedimentation a stage
##### opens at −β_{s−1} Δt times the content tendency and its own slow tendency brings that to
##### (β_s − β_{s−1}) Δt by the stage's end, so the stage-mean spurious pressure is small. Under
##### adaptive implicit vertical advection the remainder's content arrived after the loop, only
##### the rewind carried it, the whole −β_{s−1} Δt persisted through the stage, and the spurious
##### pressure where the split engages drove motion and adiabatic heating and cooling of several
##### kelvin, concentrated where the fall Courant number crosses the split's CFL target.
#####
##### So the composition content bypasses the loop. Each stage:
#####   - takes out the composition content the previous stage added, halos included, so the
#####     rewind and the linearization base θᴸ measure only pressure-changing differences from Uⁿ.
#####     The base is thus a hybrid: Πᴸ, pᴸ, γᵐRᵐᴸ and the total density are those of the full
#####     state `update_state!` diagnosed, while θᴸ is the pressure-relevant ρθ over ρ. The content
#####     is O(Δt), so the Θ-flux the loop forms from θᴸ differs by a second-order term;
#####   - collects its own, explicit and implicit, as a tendency;
#####   - after the loop, scales that tendency to the stage interval and adds the increment to the
#####     variable, then takes the increment through the variable's post-loop solve in a solve of
#####     its own, with the variable's boundary conditions, both solves reading their coefficients
#####     from a copy of the full variable frozen between the addition and the solves, so both see
#####     one operator; the field keeps the solved increment for the next stage to take out exactly.
##### Uⁿ includes the content of every completed step, so stage 1 has none to take out.
#####
##### The bypass covers sedimentation only. Condensate that the resolved flow advects moves its
##### content with it, so that part of the thermodynamic advection tendency changes ρθˡⁱ at nearly
##### fixed pressure too, and the loop reads it as pressure all the same, in the slow tendency and
##### in the rewind alike. Nothing here corrects that.
#####

"""
$(TYPEDSIGNATURES)

Take the sedimentation composition content that the previous stage added out of the
thermodynamic variable, and clear the stepper's `sedimentation_content` for this stage to collect
its own. The substep loop's linearization and rewind then read only the differences from the
outer-step-start state that change the pressure: the composition content moved with sedimenting
condensate changes `ρθˡⁱ` at fixed temperature, and so at fixed pressure. A no-op when no
condensate sediments. In `stage` 1 there is nothing to take out, the entry state `Uⁿ` including
the content of every completed step, and the field is only cleared.

The content comes out of the halos too, extended into them with no gradient at a wall and
periodically, so that the reduced variable's halos are the full variable's less the content. The
variable's own boundary conditions hold for the full variable: refilling the reduced variable's
halos with them would hold a wall value against a reduced interior, and the substep loop's
open-boundary relaxation reads its target from the halo (see `apply_open_boundary_relaxation!`).
"""
remove_sedimentation_content!(model, stage) =
    remove_sedimentation_content!(model.timestepper.sedimentation_content, model, stage)

remove_sedimentation_content!(::Nothing, model, stage) = nothing

function remove_sedimentation_content!(content, model, stage)
    if stage > 1
        ρᵡ = thermodynamic_density(model.formulation)
        fill_halo_regions!(content)
        parent(ρᵡ) .-= parent(content)
    end
    fill!(content, 0)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Add the explicit condensate sedimentation tendency of the thermodynamic variable in its two parts:
the sensible heat to the slow tendency the substep loop integrates, and the composition content
to the stepper's `sedimentation_content`. Both are formed from the state `update_state!` diagnosed
and the frozen transport velocity the moisture and tracer tendencies were built with, so their
fluxes pair with the mass fluxes those tendencies apply (see `add_sedimentation_tendency_parts!`).
"""
function add_explicit_sedimentation_content!(model)
    ts = model.timestepper
    G = getproperty(ts.Gⁿ, thermodynamic_density_name(model.formulation))
    return add_sedimentation_tendency_parts!(G, ts.sedimentation_content, model, tendency_transport_velocities(model).w)
end

"""
$(TYPEDSIGNATURES)

The implicit counterpart of [`add_explicit_sedimentation_content!`](@ref): the content of the mass
that the tracers' adaptive implicit solves have just sedimented, from the state those solves
produced and the `velocities` they split, in the same two parts
(see [`add_implicit_sedimentation_tendency_parts!`](@ref)).
"""
function add_implicit_sedimentation_content!(model, velocities)
    ts = model.timestepper
    G = getproperty(ts.Gⁿ, thermodynamic_density_name(model.formulation))
    return add_implicit_sedimentation_tendency_parts!(G, ts.sedimentation_content, model, velocities)
end

"""
$(TYPEDSIGNATURES)

Turn the sedimentation composition content tendency the stage collected into its increment over
the stage interval `Δt_stage`, and add that to the thermodynamic variable, ahead of the variable's
post-loop implicit solve. `implicit_substep!` then takes the increment through that solve too,
with the variable's own boundary conditions, both solves reading their coefficients from a copy
of the full variable frozen between this addition and the solves. The solve is linear in the field
it solves for, so the stepper's `sedimentation_content` keeps exactly the increment the variable
received, which the next stage takes back out (see [`remove_sedimentation_content!`](@ref)).

The variable's halos are left to `update_state!`, which refills them after the stage: the solves
read the interior, as Oceananigans' own time steppers have them do before their halos are refilled.
"""
apply_sedimentation_content!(model, Δt_stage) =
    apply_sedimentation_content!(model.timestepper.sedimentation_content, model, Δt_stage)

apply_sedimentation_content!(::Nothing, model, Δt_stage) = nothing

function apply_sedimentation_content!(content, model, Δt_stage)
    ρᵡ = thermodynamic_density(model.formulation)
    parent(content) .*= Δt_stage
    parent(ρᵡ) .+= parent(content)
    return nothing
end

# This stepper forms the sedimentation content itself at the start of each stage and splits it,
# so the tendencies `update_state!` builds exclude it.
AtmosphereModels.add_sedimentation_tendency!(::AcousticRungeKutta3, G, model, tracer_transport_velocity) = nothing

#####
##### Time stepping (main entry point)
#####

"""
$(TYPEDSIGNATURES)

Seed `clock.last_stage_Δt` before the first step (or for a clock carrying a non-finite value)
with the increment a completed third stage leaves, ``(1 - β₂) Δt`` — what
`adaptive_advection_timestep` inverts at stage 1. `PerturbationAdvection` open boundaries read
the same field.

Also seed the substepper's time-averaged transport velocity before the first tendencies are
built, then freeze the copy that the first stage's implicit remainder splits.
"""
function OceananigansTimeSteppers.maybe_prepare_first_time_step!(model::CompressibleAcousticModel, Δt, callbacks)
    clock = model.clock
    if clock.iteration == 0 || !isfinite(clock.last_stage_Δt)
        clock.last_Δt = Δt
        clock.last_stage_Δt = stage_increment(model.timestepper, 3) * Δt
        reconcile_state!(model)

        # Seed the transport velocity *before* the tendencies below: `set!` has already
        # diagnosed `model.velocities` from the initial momentum, and no acoustic loop has run
        # to average over. Without this, stage 1 of step 1 would build its scalar tendencies
        # from the constructor's zeros and transport no tracers vertically.
        seed_time_averaged_velocities!(model.timestepper.substepper, model)
        update_state!(model, callbacks)
        cache_transport_velocity!(model)
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step `Δt` with Wicker–Skamarock RK3 and
linearized acoustic substepping.
"""
function OceananigansTimeSteppers.time_step!(model::CompressibleAcousticModel, Δt; callbacks=[])

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
    acoustic_rk3_substep!(model, Δt, β₁, 1)

    tick_stage!(model.clock, β₁ * Δt)
    update_state!(model, callbacks; compute_tendencies = true)
    # Freeze the transport velocity the scalar tendencies were just built with; the next
    # stage's acoustic loop overwrites the live field before `scalar_substep!` reads it.
    cache_transport_velocity!(model)

    # Stage 2: U** = Uⁿ + (Δt/2) R(U*)
    acoustic_rk3_substep!(model, Δt, β₂, 2)

    tick_stage!(model.clock, (β₂ - β₁) * Δt)
    update_state!(model, callbacks; compute_tendencies = true)
    cache_transport_velocity!(model)

    # Stage 3: Uⁿ⁺¹ = Uⁿ + Δt R(U**)
    acoustic_rk3_substep!(model, Δt, β₃, 3)

    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)

    update_state!(model, callbacks; compute_tendencies = true)
    # These tendencies are consumed by stage 1 of the next step, after
    # `freeze_linearization_state!` has reseeded the live field.
    cache_transport_velocity!(model)

    # Apply the operator-split microphysics update exactly once per step, on the post-RK
    # state just refreshed by `update_state!`. A no-op for tendency-interface schemes.
    microphysics_model_update!(model.microphysics, model)

    # Advect particles once per step, over the full Δt, with the velocity of the state
    # just refreshed to tⁿ⁺¹: Xⁿ⁺¹ = Xⁿ + Δt u(Xⁿ, tⁿ⁺¹) — consistent, but first order,
    # and so lower order than the dycore. A stage-wise update is possible in principle
    # (X obeys dX/dt = u like any prognostic), but would need Xⁿ stored alongside the
    # current position, since every Wicker–Skamarock stage restarts from Uⁿ. Pushing
    # with the stage fractions alone would be wrong: β₁ + β₂ + β₃ = 11/6, not 1.
    step_lagrangian_particles!(model, Δt)

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
##### 1). Like WRF's `rk_scalar_tend` (called with `grid%ru_m, grid%rv_m,
##### grid%ww_m`) and MPAS's `ruAvg`-driven tracer transport, scalars are
##### advected by the acoustic-mean velocity rather than the RK predictor.
##### The stage alignment differs, though: WRF builds these tendencies inside
##### the stage, after its own substep loop, while Breeze builds them between
##### stages and so advects with the *preceding* loop's average.
#####
##### Theta's slow tendency does NOT consume this — `thermodynamic_transport_velocities`
##### routes it to `model.velocities` (matching WRF's `rk_tendency`; see
##### `slow_thermodynamic_velocities`). Mixing the two paths creates a feedback
##### loop that destabilizes a rest atmosphere at production Δt. Its condensate
##### sedimentation content is not built with that tendency on this stepper: the stage
##### forms it from the tracers' frozen transport velocity, pairing its content fluxes
##### with the mass fluxes the tracer tendencies apply, adds its sensible heat to that
##### tendency, and adds its composition content after the loop.
#####
##### The implicit remainder of a scalar update must split the SAME velocity
##### its explicit fraction was scaled by, so `scalar_substep!` reads a frozen
##### copy (`cache_transport_velocity!`) instead of this live field — by then
##### the stage's own acoustic loop has already rebuilt it. At outer-step start
##### `freeze_linearization_state!` seeds the live field with `model.velocities`
##### for whatever reads it before the first loop finalizes, and
##### `maybe_prepare_first_time_step!` seeds it before the very first tendency
##### computation so stage 1 of step 1 splits a physical velocity.
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

# `update_state!` builds the slow tendencies the stages apply: momentum without the pressure-gradient
# force and buoyancy, which the substep loop integrates, and the thermodynamic variable advected by
# the RK predictor velocity rather than the time-averaged transport velocity.
AtmosphereModels.slow_dynamics(::AcousticRungeKutta3, dynamics) = SlowDynamics(dynamics)
AtmosphereModels.thermodynamic_transport_velocities(model::CompressibleAcousticModel) = slow_thermodynamic_velocities(model)

Oceananigans.prognostic_state(timestepper::AcousticRungeKutta3) =
    (substepper = Oceananigans.prognostic_state(timestepper.substepper),)

function Oceananigans.restore_prognostic_state!(restored::AcousticRungeKutta3, from)
    Oceananigans.restore_prognostic_state!(restored.substepper, from.substepper)
    return restored
end

Oceananigans.restore_prognostic_state!(timestepper::AcousticRungeKutta3, ::Nothing) = timestepper
