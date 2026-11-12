using KernelAbstractions: @kernel, @index

using Oceananigans: TendencyCallsite
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models: boundary_condition_args
using Oceananigans.Utils: time_difference_seconds

using Oceananigans.TimeSteppers:
    AbstractTimeStepper,
    tick_stage!,
    update_state!,
    compute_flux_bc_tendencies!,
    step_lagrangian_particles!

using Oceananigans.TurbulenceClosures: step_closure_prognostics!

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, SlowDynamics, microphysics_model_update!,
                                compute_closure_tendencies!, refreshes_state_after_update, invalid_time_step,
                                thermodynamic_density, add_sedimentation_tendency_parts!,
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
    seed_time_averaged_velocities!,
    collect_composition_content!

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
     prognostic state. The loop integrates the thermodynamic perturbation
     less its composition content, the part of a change of the variable
     that a change of moisture makes without changing the pressure (see
     the section comment on the composition content bypass).
  4. Adds the composition content of the moisture the stage ends with,
     then solves the vertically-implicit pieces of momentum and the
     thermodynamic variable.

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
  substep loop; it also holds the `CompositionBypass` through which the
  composition content of the thermodynamic variable skips the loop's pressure.
- `rounding`: Round-off treatment of the final stage's update, [`RoundToNearest`](@ref)
  or [`StochasticRounding`](@ref Breeze.RoundingTreatments.StochasticRounding).

References
==========

Wicker, L. J. & Skamarock, W. C. (2002). *Time-splitting methods for
elastic models using forward time schemes.* MWR 130, 2088–2097.
"""
struct AcousticRungeKutta3{FT, U0, TG, TI, AS, R} <: AbstractTimeStepper
    β₁ :: FT
    β₂ :: FT
    β₃ :: FT
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
    substepper :: AS
    rounding :: R
end

"""
$(TYPEDSIGNATURES)

Construct an `AcousticRungeKutta3` time stepper for fully compressible dynamics.

With `composition_bypass = true` the substepper allocates the `CompositionBypass` through which
the composition content of the thermodynamic variable skips the substep loop's pressure (see the
section comment on the bypass). `AtmosphereModel` asks for it when the model has a microphysics
scheme, and with it condensate; without one the stepper is what it is without the bypass.
`Gⁿ` and `U⁰` may be supplied to alias another stepper's tendency storage instead of
allocating fresh fields (used by the native-stepper adiabatic-balance twin).

`rounding` sets the round-off treatment of the step's final update (default:
[`RoundToNearest`](@ref)). Every Wicker–Skamarock stage restarts from ``Uⁿ``, so only the third
stage's rounding reaches ``Uⁿ⁺¹``: with [`StochasticRounding`](@ref Breeze.RoundingTreatments.StochasticRounding), the recovery of the acoustic
prognostics from their stage-entry values plus the accumulated perturbations, and the update
``u⁰ + Δt G`` of the other scalars, are rounded stochastically in the third stage. The
perturbations themselves, the first two stages, and the vertically-implicit solves stay
round-to-nearest.
"""
function AcousticRungeKutta3(grid, prognostic_fields;
                             dynamics,
                             implicit_solver::TI = nothing,
                             cache_advecting_state = false,
                             composition_bypass = false,
                             Gⁿ::TG = map(similar, prognostic_fields),
                             U⁰::U0 = map(similar, prognostic_fields),
                             rounding = RoundToNearest()) where {TI, TG, U0}

    FT = eltype(grid)
    rounding = materialize_rounding(rounding, grid.architecture)
    R = typeof(rounding)

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
                                    composition_bypass,
                                    prognostic_momentum = (ρu = prognostic_fields.ρu,
                                                           ρv = prognostic_fields.ρv,
                                                           ρw = prognostic_fields.ρw))
    AS = typeof(substepper)

    return AcousticRungeKutta3{FT, U0, TG, TI, AS, R}(β₁, β₂, β₃, U⁰, Gⁿ, implicit_solver, substepper, rounding)
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
substep loop on the thermodynamic perturbation less its composition content, add the content of
the moisture the stage ends with, then run the post-loop implicit solves. The stage's updates are
rounded with the round-off treatment `rounding`.
"""
function acoustic_rk3_substep!(model::AtmosphereModel, Δt, β, stage, rounding = RoundToNearest())
    ts = model.timestepper
    substepper = ts.substepper
    U⁰ = ts.U⁰
    Δt_stage = β * Δt

    # `update_state!` built the slow tendencies (advection + Coriolis + diffusion; PGF and
    # buoyancy are excluded — those are handled inside the substep loop in linearized form
    # about the RK stage-entry state). Add the boundary flux and closure tendencies.
    compute_flux_bc_tendencies!(model)
    compute_closure_tendencies!(model)

    # Per-stage cache prep, from the stage-entry state `update_state!` diagnosed. The substep
    # perturbations are initialized with a rewind term inside the substep loop so the WS-RK3
    # invariant still starts from Uⁿ.
    prepare_acoustic_cache!(substepper, model)
    cache_advecting_state!(model)

    # Base-state part of the IMEX vertical-advection split's implicit half (a no-op unless the
    # thermodynamic scheme is adaptive-implicit); the perturbation part is solved per substep
    # inside the loop.
    add_implicit_advection_tendency!(model)

    # The explicit condensate sedimentation tendency of the thermodynamic variable, from the
    # tracers' frozen transport velocity; the implicit remainder's follows in
    # `scalar_rk3_substep!`, after the tracer solves.
    add_explicit_sedimentation_tendency!(model)

    # Update remaining scalars (tracers) using WS-RK3, their implicit solves included. Their
    # tendencies and the transport velocity their solves split were frozen before this stage, so
    # they need nothing from the substep loop, and they run before it because the loop's slow
    # tendency takes the content of the moisture they update (`collect_composition_content!`,
    # between the explicit updates and the solves). They follow the cache prep because the
    # linearization's mixture coefficients read the stage-entry moisture.
    scalar_rk3_substep!(model, Δt_stage, rounding; first_stage = stage == 1,
                        extrapolation = stage_extrapolation(ts, β, stage))

    # Linearized acoustic substep loop: Nτ substeps of size Δτ = Δt/N.
    θ_advection = field_advection_scheme(model.advection, thermodynamic_density_name(model.formulation))
    acoustic_rk3_substep_loop!(model, substepper, Δt, β, U⁰, θ_advection; rounding)

    # Add the content of the moisture the stage ends with, so that the solves below see the full
    # state, halos included.
    apply_composition_content!(model, rounding)

    # Vertically-implicit solve for the acoustic prognostics (momentum and the thermodynamic
    # variable) over the stage interval: the implicit remainder of adaptive implicit vertical
    # advection combined with vertically-implicit closure diffusion. A no-op when the timestepper
    # has no implicit solver.
    implicit_substep!(model, Δt_stage)

    return nothing
end

#####
##### Scalar update with time-averaged velocities
#####

function scalar_rk3_substep!(model::AtmosphereModel, Δt_stage, rounding = RoundToNearest();
                             first_stage = model.clock.stage == 1, extrapolation = 1)
    grid = model.grid
    kernel_Δt = kernel_time_step(grid.architecture, grid, Δt_stage)
    return scalar_substep!(model, Δt_stage, kernel_Δt, rounding, first_stage, extrapolation)
end

# The ratio of the stage's interval β Δt to the interval the stage-entry state was advanced over
# from the outer step start, β₁ Δt for stage 2 and β₂ Δt for stage 3: the factor that extrapolates
# a change known at stage entry to the stage's end. The first stage starts from the outer step
# start itself and has nothing to extrapolate.
stage_extrapolation(ts, β, stage) = stage == 1 ? one(β) : β / ifelse(stage == 2, ts.β₁, ts.β₂)

# Round-to-nearest keeps the original kernel, so the default is bitwise unchanged.
launch_rk3_substep!(::RoundToNearest, arch, grid, u, u⁰, G, Δt_stage, key) =
    launch!(arch, grid, :xyz, _rk3_substep!, u, u⁰, G, Δt_stage)

launch_rk3_substep!(rounding, arch, grid, u, u⁰, G, Δt_stage, key) =
    launch!(arch, grid, :xyz, _rounded_rk3_substep!, u, u⁰, G, Δt_stage, rounding, key)

@kernel function _rk3_substep!(u, u⁰, G, Δt_stage)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = u⁰[i, j, k] + Δt_stage * G[i, j, k]
end

@kernel function _rounded_rk3_substep!(u, u⁰, G, Δt_stage, rounding, key)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = rounded_add(rounding, u⁰[i, j, k], Δt_stage * G[i, j, k], i, j, k, key)
end

#####
##### Composition content of the thermodynamic variable, kept out of the substep loop's pressure
#####
##### The loop forms its pressure perturbation from the thermodynamic perturbation alone,
##### p′ = Cᴸ (ρθ)′, with the moisture frozen at the stage-entry state, so it reads any change of
##### ρθ as a change of pressure. Part of a change of ρθˡⁱ is not one: its composition content.
##### Condensate that arrives in a cell at the cell's temperature lowers θˡⁱ by about
##### ℒ Δq / (cᵖᵐ Π) while the gas pressure (ρᵈ Rᵈ + ρᵛ Rᵛ) T stays what it was. Whatever moves
##### moisture moves that content with it: sedimentation, advection by the resolved flow, closure
##### diffusion. Read as pressure it is about Cᴸ ℒ / (cᵖᵐ Π) per unit condensate density, some
##### 10⁶ Pa per kg m⁻³.
#####
##### The content reaches the loop by two routes. What a slow tendency carries ramps up over the
##### stage: a stage opens at −β_{s−1} Δt times the content tendency, the rewind Uⁿ − Uᴸ_stage
##### carrying the previous stage's content with the opposite sign, and ends at (β_s − β_{s−1}) Δt,
##### so the spurious pressure is the content tendency over a fraction of a stage. What an
##### operator adds after the loop (the thermodynamic variable's vertically-implicit closure
##### diffusion, the implicit remainder of condensate sedimentation) only the rewind carries: the
##### whole −β_{s−1} Δt persists through the stage, and the spurious pressure drives motion and
##### adiabatic heating and cooling.
#####
##### The pressure of the equation of state also depends on the moisture, and that dependence is
##### what the loop leaves out: with it, the composition content cancels. So the stage measures
##### the content from the equation of state rather than from each process that moves moisture.
##### To first order, at fixed dry density,
#####
#####   p′ = Cᴸ [(ρθ)′ − χ] ,   χ = Λᵛ (ρᵛ)′ + Λˡ (ρˡ)′ + Λⁱ (ρⁱ)′ ,
#####
##### with (ρˣ)′ the differences of the partial densities of vapor, liquid and ice from the
##### stage-entry state and Λˣ the content of ρθ per unit partial density
##### (`composition_content_coefficients`): for condensate, the content χˣ = ∂θ/∂qˣ at fixed
##### temperature and pressure that the sedimentation tendency moves with the falling mass
##### (`condensate_content`), so the two cancel to roundoff. The loop integrates (ρθ)′ − χ. Each
##### stage, through the substepper's `CompositionBypass`:
#####   - updates the moisture with its explicit tendencies and collects the contents
#####     (`collect_composition_content!`): χ⁰ of the outer-step-start moisture relative to the
#####     stage-entry moisture, which the loop takes out of the rewind, so that a stage starts
#####     from the pressure of Uⁿ whatever moved the moisture since, before or after a loop; and
#####     χ¹ of the updated moisture, whose ramp (χ¹ − χ⁰) / (β Δt) comes out of the slow
#####     tendency the loop integrates. That tendency carries the content the same
#####     processes move, the explicit condensate sedimentation content included
#####     (`add_explicit_sedimentation_tendency!`), so what is left is their sensible heat;
#####   - solves the moisture's vertically-implicit pieces. Their increments stay out of the ramp:
#####     the thermodynamic variable receives the matching content after the loop, from its own
#####     solve and, for the implicit remainder of condensate sedimentation, from the mass fluxes
#####     the solves applied (`add_implicit_sedimentation_content!`), whose sensible heat joins the
#####     slow tendency;
#####   - after the loop, adds χ¹ and the implicit sedimentation content to the variable, rounded
#####     like the loop's recovery of the full state (`apply_composition_content!`), before the
#####     variable's post-loop solve. The next stage measures χ⁰ afresh from the state these leave.
##### The live thermodynamic variable stays the full stage-entry variable until the loop recovers
##### the full state, so whatever reads it in between (the base flux of the adaptive implicit
##### split, the coefficients of the moisture and tracer solves, the open-boundary relaxation)
##### reads the full variable.
#####
##### χ⁰ takes back the content the previous stage added to first order only: that stage added
##### it with its own coefficients, or from the fluxes of an implicit solve. The loop starts from
##### the remainder, second order in the moisture difference. The open-boundary relaxation, which
##### holds a boundary cell without a thermodynamic wall value on its stage-entry base, takes the
##### remainder off its target (the bypass's `rewind_remainder`), so that the content passes
##### through the cell without accumulating across stages.
#####
##### The partial densities are those of the model's own moisture partition. With prognostic
##### condensate they follow the prognostic densities, and χ¹ − χ⁰ is the content the tendencies
##### move. A saturation-adjustment scheme diagnoses its cloud: the partition at the outer step
##### start and at stage entry is the adjusted one `update_state!` left, and what the adjustment
##### has moved between vapor and cloud by stage entry is extrapolated to the stage's end, with
##### no further adjustment. The first stage has no such drift to extrapolate, so the content of
##### cloud a saturation adjustment condenses or moves within it reaches its loop as before.
#####
##### A model without a microphysics scheme has no condensate and no bypass: the stepper is then
##### what it is without one, and the pressure a change of vapor makes through the gas constant
##### reaches the loop with the next stage's linearization, as it always has.
#####
##### The bypass also carries what is not composition content but belongs to the same dependence:
##### the pressure a change of vapor makes through the gas constant, and the heating of a phase
##### change in a scheme that advances condensate by tendencies, which reach the loop as a ramp
##### over the stage rather than with the next stage's linearization.
#####
##### What remains:
#####   - The content is first order in the moisture difference, like the content the tendencies
#####     move: neither reconstructs the thermal energy of a finite increment.
#####   - Under adaptive implicit vertical advection, the loop applies the implicit remainder of
#####     the variable's own vertical transport, while the moisture's implicit remainder stays out
#####     of the ramp. Where the resolved vertical Courant number engages the split in cloud, the
#####     content that remainder carries is read as pressure within the stage, and the next
#####     stage's rewind removes it.
#####   - The loop transports the thermodynamic variable with the perturbation mass flux and
#####     θᴸ, content included, and leaves the moisture where it is: second order, the
#####     perturbation flux times the content gradient.
#####   - The moisture loading in the loop's gravity term is still the stage-entry one.
#####

# The substepper's `CompositionBypass`, `nothing` when it is switched off.
composition_bypass(model) = model.timestepper.substepper.composition_bypass

"""
$(TYPEDSIGNATURES)

Add the explicit condensate sedimentation tendency of the thermodynamic variable, sensible heat
and composition content together, to the slow tendency the substep loop integrates. It is formed
from the state `update_state!` diagnosed and the frozen transport velocity the moisture and tracer
tendencies were built with, so its fluxes pair with the mass fluxes those tendencies apply (see
`add_sedimentation_tendency_parts!`). The composition content it carries comes out again with the
ramp of the stage's content (`collect_composition_content!`). A no-op when no condensate sediments.
"""
function add_explicit_sedimentation_tendency!(model)
    G = model.timestepper.Gⁿ[thermodynamic_density_name(model.formulation)]
    return add_sedimentation_tendency_parts!(G, G, model, tendency_transport_velocities(model).w)
end

"""
$(TYPEDSIGNATURES)

Move the content of the mass that the tracers' adaptive implicit solves have just sedimented, from
the solved mass fluxes and the `velocities` they split, with the content per unit mass of the
stage-entry state (see `add_implicit_sedimentation_tendency_parts!`): its sensible heat into the
slow tendency the substep loop integrates, its composition content, scaled by the stage interval
`kernel_Δt`, into the bypass's content, which the stage adds to the thermodynamic variable after
the loop. The solves' increments of the moisture are not in the ramp of the stage's content, so
the composition content they move must not reach the loop either. Without a bypass both parts
join the slow tendency.
"""
add_implicit_sedimentation_content!(model, velocities, kernel_Δt) =
    add_implicit_sedimentation_content!(composition_bypass(model), model, velocities, kernel_Δt)

function add_implicit_sedimentation_content!(::Nothing, model, velocities, kernel_Δt)
    G = model.timestepper.Gⁿ[thermodynamic_density_name(model.formulation)]
    return add_implicit_sedimentation_tendency_parts!(G, G, model, velocities)
end

function add_implicit_sedimentation_content!(bypass, model, velocities, kernel_Δt)
    G = model.timestepper.Gⁿ[thermodynamic_density_name(model.formulation)]
    return add_implicit_sedimentation_tendency_parts!(G, bypass.content, model, velocities; composition_factor = kernel_Δt)
end

"""
$(TYPEDSIGNATURES)

Add the composition content of the moisture the stage ends with to the thermodynamic variable
after the substep loop, rounded with the round-off treatment `rounding` like the loop's recovery
of the full state, with a key of its own, and refill the variable's halos with its boundary
conditions. `implicit_substep!` then takes the full variable through its post-loop solve. A no-op
without a bypass.
"""
apply_composition_content!(model, rounding = RoundToNearest()) =
    apply_composition_content!(composition_bypass(model), model, rounding)

apply_composition_content!(::Nothing, model, rounding) = nothing

function apply_composition_content!(bypass, model, rounding)
    grid = model.grid
    ρᵡ = thermodynamic_density(model.formulation)
    # A field number past those of the prognostic fields, which key the scalar updates, and the
    # zero that keys the loop's recovery of the full state.
    field_number = length(prognostic_fields(model)) + 1
    key = rounding_key(rounding, model.clock.iteration, model.clock.stage, field_number)
    # The stage update u = u⁰ + Δt G, with the variable as u⁰, the content as G and Δt = 1.
    launch_rk3_substep!(rounding, grid.architecture, grid, ρᵡ, ρᵡ, bypass.content, one(eltype(grid)), key)
    fill_halo_regions!(ρᵡ, boundary_condition_args(model)...)
    return nothing
end

# This stepper forms the sedimentation tendency itself at the start of each stage, from the frozen
# transport velocity, so the tendencies `update_state!` builds exclude it.
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

    # Stage 3: Uⁿ⁺¹ = Uⁿ + Δt R(U**). Every stage restarts from Uⁿ, so only this stage's
    # rounding reaches Uⁿ⁺¹, and only this stage takes the time stepper's round-off treatment.
    acoustic_rk3_substep!(model, Δt, β₃, 3, ts.rounding)

    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)

    # Kessler reads diagnostics, then rebuilds tendencies after updating its prognostics.
    # Keep the preceding pass when Kessler will return early (the same `invalid_time_step`
    # test it applies), or when a tendency callback is present, so that the callback still runs
    # once per stage.
    # TODO: Kessler's internal update_state! does not forward the time stepper's callbacks, so
    # whatever a tendency callback adds to Gⁿ here is overwritten before stage 1 of the next step.
    # Forwarding the callbacks through microphysics_model_update! would fix this and let the pass
    # always be skipped.
    skip_tendencies = skip_pre_microphysics_tendencies(model.microphysics, Δt, callbacks)
    update_state!(model, callbacks; compute_tendencies = !skip_tendencies)
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

skip_pre_microphysics_tendencies(microphysics, Δt, callbacks) =
    refreshes_state_after_update(microphysics) && !invalid_time_step(Δt) &&
    !any(callback -> callback.callsite isa TendencyCallsite, callbacks)

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
##### sedimentation tendency is not built with that tendency on this stepper: the stage
##### forms it from the tracers' frozen transport velocity, pairing its content fluxes
##### with the mass fluxes the tracer tendencies apply.
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
