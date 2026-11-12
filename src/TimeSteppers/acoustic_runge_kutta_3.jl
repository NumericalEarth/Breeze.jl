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
     which the substep loop never sees (see the section comment on the
     sedimentation content bypass), then solves the vertically-implicit
     pieces of momentum and the thermodynamic variable, the content taking
     the variable's solve in a solve of its own.

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
  substep loop; it also holds the `SedimentationBypass` through which the
  condensate sedimentation content of the thermodynamic variable skips the loop.
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

`sedimentation` holds the model's sedimenting condensates (see `materialize_sedimentation`);
when there are any, the substepper allocates the `SedimentationBypass` through which the content
they move each stage skips the substep loop (see the section comment on the bypass).
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
                             sedimentation = NamedTuple(),
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
                                    sedimentation_bypass = !isempty(sedimentation),
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
substep loop, add the stage's condensate sedimentation composition content to the thermodynamic
variable, then run the post-loop implicit solves. The stage's updates are rounded with the
round-off treatment `rounding`.
"""
function acoustic_rk3_substep!(model::AtmosphereModel, Δt, β, stage, rounding = RoundToNearest())
    ts = model.timestepper
    substepper = ts.substepper
    U⁰ = ts.U⁰
    Δt_stage = β * Δt

    # `update_state!` built the slow tendencies (advection + Coriolis + diffusion; PGF and
    # buoyancy are excluded — those are handled inside the substep loop in linearized form
    # about the RK stage-entry state). Add the boundary flux and closure tendencies, which may
    # read the thermodynamic variable and must see the full stage-entry state, before the
    # content removal below.
    compute_flux_bc_tendencies!(model)
    compute_closure_tendencies!(model)

    # Per-stage cache prep, from the full stage-entry state `update_state!` diagnosed and before
    # the content removal below: the temperature the linearization's slope is differentiated at
    # must match the moisture the variable carries (see the section comment on the sedimentation
    # content bypass). The substep perturbations are initialized with a rewind term inside the
    # substep loop so the WS-RK3 invariant still starts from Uⁿ.
    prepare_acoustic_cache!(substepper, model)
    cache_advecting_state!(model)

    # Keep the full stage-entry thermodynamic variable and take the previous stage's sedimentation
    # composition content out of the live field, so that the rewind below does not read it as a
    # pressure change. Nothing is pending in stage 1.
    remove_sedimentation_content!(model, stage)

    # Base-state part of the IMEX vertical-advection split's implicit half (a no-op unless the
    # thermodynamic scheme is adaptive-implicit); the perturbation part is solved per substep
    # inside the loop. It reads the live variable, now without the previous stage's content.
    add_implicit_advection_tendency!(model)

    # This stage's sedimentation content: the explicit part now, from the tracers' frozen
    # transport velocity; the implicit remainder's in `scalar_rk3_substep!`, after the tracer solves.
    add_explicit_sedimentation_content!(model, Δt_stage)

    # Update remaining scalars (tracers) using WS-RK3, their implicit solves included. Their
    # tendencies and the transport velocity their solves split were frozen before this stage, so
    # they need nothing from the substep loop, and they run before it so that the content of the
    # mass their solves sediment reaches the loop's slow tendency in time. They follow the cache
    # prep because the linearization's mixture coefficients read the stage-entry moisture.
    scalar_rk3_substep!(model, Δt_stage, rounding)

    # Linearized acoustic substep loop: Nτ substeps of size Δτ = Δt/N.
    θ_advection = field_advection_scheme(model.advection, thermodynamic_density_name(model.formulation))
    acoustic_rk3_substep_loop!(model, substepper, Δt, β, U⁰, θ_advection; rounding)

    # Add the stage's sedimentation composition content, so that the solves below see the full
    # state, halos included.
    apply_sedimentation_content!(model, rounding)

    # Vertically-implicit solve for the acoustic prognostics (momentum and the thermodynamic
    # variable) over the stage interval: the implicit remainder of adaptive implicit vertical
    # advection combined with vertically-implicit closure diffusion, the content taking the
    # variable's solve in a solve of its own. A no-op when the timestepper has no implicit solver.
    implicit_substep!(model, Δt_stage)

    return nothing
end

#####
##### Scalar update with time-averaged velocities
#####

function scalar_rk3_substep!(model::AtmosphereModel, Δt_stage, rounding = RoundToNearest())
    grid = model.grid
    kernel_Δt = kernel_time_step(grid.architecture, grid, Δt_stage)
    return scalar_substep!(model, Δt_stage, kernel_Δt, rounding)
end

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
##### Condensate sedimentation content, kept out of the acoustic substep loop
#####
##### The loop forms its pressure perturbation from the thermodynamic perturbation alone,
##### p′ = Cᴸ (ρθ)′, so it reads any change of ρθ as a change of pressure. The content of
##### ρθˡⁱ that sedimenting condensate moves has two parts (see `sedimentation_tendency`). The
##### sensible heat the arriving mass brings changes the temperature and the pressure, like any
##### heating, so it joins the slow tendency the loop integrates. The composition content (mostly
##### the latent deficit −ℒ q / (cᵖᵐ Π); see `condensate_content`) changes ρθ at fixed
##### temperature, and the gas pressure (ρᵈ Rᵈ + ρᵛ Rᵛ) T does not change. Fed to the loop, it
##### reads as a pressure perturbation of about Cᴸ ℒ / (cᵖᵐ Π) per unit condensate density. Every
##### stage after the first starts its perturbations from the rewind Uⁿ − Uᴸ_stage, which carries
##### the previous stage's content with the opposite sign: with explicit sedimentation a stage
##### opens at −β_{s−1} Δt times the content tendency and its own slow tendency brings that to
##### (β_s − β_{s−1}) Δt by the stage's end, so the stage-mean spurious pressure is small. Under
##### adaptive implicit vertical advection the remainder's content arrived after the loop, only
##### the rewind carried it, the whole −β_{s−1} Δt persisted through the stage, and the spurious
##### pressure where the split engages drove motion and adiabatic heating and cooling of several
##### kelvin, concentrated where the fall Courant number crosses the split's CFL target.
#####
##### So the composition content bypasses the loop, through the substepper's
##### `SedimentationBypass`: a content field and a copy of the thermodynamic variable. Each stage:
#####   - refreshes the loop's linearization, θᴸ and the slope Cᴸ, from the full stage-entry state
#####     `update_state!` diagnosed, whose temperature matches its moisture
#####     (`prepare_acoustic_cache!`). Differentiated at the variable less a stage's content, with
#####     the full moisture, the slope would sit at a temperature off by about ℒ Δq / cᵖ, tens of
#####     kelvin in a cell a rain front crossed during the previous stage at a large time step, and
#####     the loop's sound speed off by several percent there;
#####   - copies the full stage-entry variable, then takes out the composition content the
#####     previous stage added, halos included, so that the rewind, and the base the loop recovers
#####     the full state from, measure only pressure-changing differences from Uⁿ
#####     (`remove_sedimentation_content!`). What must see the full variable meanwhile reads the
#####     copy: the coefficients of the moisture and tracer solves (`frozen_thermodynamic_fields`),
#####     and the loop's open-boundary relaxation where a value condition prescribes the full
#####     variable at the wall (`apply_open_boundary_relaxation!`);
#####   - collects its own content, explicit and implicit, scaled to the stage's increment
#####     (`add_explicit_sedimentation_content!`, `add_implicit_sedimentation_content!`);
#####   - after the loop, adds the increment to the variable, rounded like the loop's recovery of
#####     the full state (`apply_sedimentation_content!`), and takes it through the variable's
#####     post-loop solve in a solve of its own, both solves reading their coefficients from the
#####     copy refreshed between the addition and the solves, so that the content field keeps
#####     exactly what the variable received, for the next stage to take out
#####     (`thermodynamic_implicit_substep!`).
##### Uⁿ includes the content of every completed step, so stage 1 has none to take out.
#####
##### Two first-order splitting differences remain in the variable's own vertical transport
##### under an adaptive implicit scheme. The content never takes the implicit transport the loop
##### applies to the rest of the variable, only the post-loop solve. And the base flux of the
##### implicit half (`add_implicit_advection_tendency!`) is formed from the live variable after the
##### removal, so it carries the variable less the previous stage's content, while the explicit
##### half in the slow tendency was formed from the full variable. Both differences are the
##### content, O(Δt), transported over a stage.
#####
##### The bypass covers sedimentation only. Condensate that the resolved flow advects moves its
##### content with it, so that part of the thermodynamic advection tendency changes ρθˡⁱ at nearly
##### fixed pressure too, and the loop reads it as pressure all the same, in the slow tendency and
##### in the rewind alike. Nothing here corrects that.
#####

# The substepper's `SedimentationBypass`, `nothing` when no condensate sediments, and its content.
sedimentation_bypass(model) = model.timestepper.substepper.sedimentation_bypass
sedimentation_content(::Nothing) = nothing
sedimentation_content(bypass) = bypass.content

"""
$(TYPEDSIGNATURES)

Open a stage of the sedimentation content bypass (see the section comment above): copy the full
thermodynamic variable, halos included, into the bypass's `frozen_thermodynamic_density`, take the
composition content the previous stage added out of the live variable, halos included, and clear
the content field for this stage to collect its own. In `stage` 1 there is nothing to take out,
the entry state `Uⁿ` including the content of every completed step. A no-op when no condensate
sediments.

The content comes out of the halos too, extended into them with no gradient at a wall and
periodically, so that the reduced variable's halos are the full variable's less the content and
the vertical operators the stage applies to the reduced variable see no step at the boundaries.
The variable's own boundary conditions hold for the full variable, which the copy keeps.
"""
remove_sedimentation_content!(model, stage) = remove_sedimentation_content!(sedimentation_bypass(model), model, stage)

remove_sedimentation_content!(::Nothing, model, stage) = nothing

function remove_sedimentation_content!(bypass, model, stage)
    grid = model.grid
    ρᵡ = thermodynamic_density(model.formulation)
    content = bypass.content
    stage > 1 && fill_halo_regions!(content)
    params = KernelParameters(size(parent(ρᵡ)), (0, 0, 0))
    launch!(grid.architecture, grid, params, _remove_sedimentation_content!,
            parent(bypass.frozen_thermodynamic_density), parent(ρᵡ), parent(content), stage > 1)
    return nothing
end

# One pass over the parent arrays: the copy, the removal (a no-op in stage 1), and the clearing.
@kernel function _remove_sedimentation_content!(frozen, ρᵡ, content, remove)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        frozen[i, j, k] = ρᵡ[i, j, k]
        ρᵡ[i, j, k] -= remove * content[i, j, k]
        content[i, j, k] = 0
    end
end

"""
$(TYPEDSIGNATURES)

Collect the explicit condensate sedimentation content of the thermodynamic variable in its two
parts: the sensible heat into the slow tendency the substep loop integrates, and the composition
content, scaled by the stage interval `Δt_stage`, into the bypass's content field. Both are formed
from the state `update_state!` diagnosed and the frozen transport velocity the moisture and tracer
tendencies were built with, so their fluxes pair with the mass fluxes those tendencies apply
(see `add_sedimentation_tendency_parts!`). A no-op when no condensate sediments.
"""
function add_explicit_sedimentation_content!(model, Δt_stage)
    grid = model.grid
    G, C = sedimentation_parts_destinations(model)
    return add_sedimentation_tendency_parts!(G, C, model, tendency_transport_velocities(model).w;
                                             composition_factor = kernel_time_step(grid.architecture, grid, Δt_stage))
end

"""
$(TYPEDSIGNATURES)

The implicit counterpart of `add_explicit_sedimentation_content!`: the content of the mass that
the tracers' adaptive implicit solves have just sedimented, from the solved mass fluxes and the
`velocities` they split, with the content per unit mass of the stage-entry state
(see `add_implicit_sedimentation_tendency_parts!`). `kernel_Δt` is the stage interval as the
kernels take it.
"""
function add_implicit_sedimentation_content!(model, velocities, kernel_Δt)
    G, C = sedimentation_parts_destinations(model)
    return add_implicit_sedimentation_tendency_parts!(G, C, model, velocities; composition_factor = kernel_Δt)
end

# Where the stage puts the two parts: the slow tendency of the thermodynamic variable, and the
# bypass's content field (`nothing` when no condensate sediments, and nothing is added).
sedimentation_parts_destinations(model) =
    (model.timestepper.Gⁿ[thermodynamic_density_name(model.formulation)],
     sedimentation_content(sedimentation_bypass(model)))

"""
$(TYPEDSIGNATURES)

Add the stage's sedimentation composition content increment to the thermodynamic variable after
the substep loop, rounded with the round-off treatment `rounding` like the loop's recovery of the
full state, with a key of its own, and refill the variable's halos with its boundary conditions,
which hold for the full variable. `implicit_substep!` then takes the increment through the
variable's post-loop solve too (see `thermodynamic_implicit_substep!`). A no-op when no condensate
sediments.
"""
apply_sedimentation_content!(model, rounding = RoundToNearest()) =
    apply_sedimentation_content!(sedimentation_bypass(model), model, rounding)

apply_sedimentation_content!(::Nothing, model, rounding) = nothing

function apply_sedimentation_content!(bypass, model, rounding)
    grid = model.grid
    ρᵡ = thermodynamic_density(model.formulation)
    # A field number past those of the prognostic fields, which key the scalar updates, and the
    # zero that keys the loop's recovery of the full state.
    field_number = length(prognostic_fields(model)) + 1
    key = rounding_key(rounding, model.clock.iteration, model.clock.stage, field_number)
    # The stage update u = u⁰ + Δt G, with the variable as u⁰, the increment as G and Δt = 1.
    launch_rk3_substep!(rounding, grid.architecture, grid, ρᵡ, ρᵡ, bypass.content, one(eltype(grid)), key)
    fill_halo_regions!(ρᵡ, boundary_condition_args(model)...)
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
