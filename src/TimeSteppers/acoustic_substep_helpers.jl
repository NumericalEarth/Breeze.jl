using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields, architecture, TendencyCallsite
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection, vertical_scheme,
                              implicit_vertical_velocityᶜᶜᶠ
using Oceananigans.Operators: Azᶜᶜᶠ, δzᵃᵃᶜ, V⁻¹ᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.Utils: launch!, KernelParameters

using Breeze.AtmosphereModels:
    AtmosphereModel,
    advecting_vertical_velocity,
    continuity_density,
    total_density,
    thermodynamic_density_name,
    transport_velocities,
    field_advection_scheme,
    closure_scalar_index,
    dynamics_prognostic_fields,
    implicit_advection_velocities,
    implicit_step_scheme,
    implicit_sedimentation_step!,
    compute_scalar_tendencies!,
    add_scalar_tendency!,
    tracer_density_to_specific!,
    tracer_specific_to_density!

using Breeze.CompressibleEquations: CompressibleDynamics, implicit_steps_in_place!, implicit_step_scratch
using Breeze.TerrainFollowingDiscretization: TerrainMetrics

const TerrainCompressibleAcousticModel =
    AtmosphereModel{<:CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:TerrainMetrics}}

#####
##### Thermodynamic-variable transport velocity
#####
##### The thermodynamic variable is advected by the current RK predictor velocity
##### (`model.velocities`), matching WRF (`rk_tendency` in `solve_em.F`, called with
##### `grid%ru, grid%rv, grid%ww`) and MPAS. Routing the substepper's time-averaged
##### velocity here creates a closed feedback loop (Gⁿ.ρθ → ρθ′ → PGF → (ρu)′ →
##### time-averaged velocity → next stage's Gⁿ.ρθ) that destabilizes the rest
##### atmosphere; T4 blows up at production Δt. For nonflat terrain, the same current
##### predictor is used horizontally while vertical transport uses the current
##### terrain-following `w̃`. The dynamics-transport split applies only to **moisture,
##### tracers, chemistry, TKE**, which `transport_velocities(model)` routes to the
##### substepper's time-averaged velocity.
#####

slow_thermodynamic_velocities(model) = model.velocities

function slow_thermodynamic_velocities(model::TerrainCompressibleAcousticModel)
    u = model.velocities.u
    v = model.velocities.v
    w̃ = model.dynamics.contravariant_vertical_velocity
    return (; u, v, w=w̃)
end

#####
##### Scalar (tracer / moisture) update with time-averaged velocities
#####

# The state the current stage's acoustic integration starts from and its scalars update from:
# the stored outer-step-start state, unless the stepper's stages start elsewhere.
stage_start_state(timestepper) = timestepper.U⁰

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Freeze the time-averaged transport velocity that `update_state!` just built the moisture and
tracer tendencies from. The next acoustic loop resets and rebuilds `time_averaged_velocities`
(and `freeze_linearization_state!` reseeds it at outer-step start), so the stage cannot read it
live. The implicit remainder in `scalar_substep!` pairs fluxes with those tendencies: it has to
split the same velocity the explicit fraction in `Gⁿ` was scaled by (invariant: ⟨w⟩ = wᵉ + wⁱ),
as does `implicit_sedimentation_step!`. (The condensate sedimentation term of the thermodynamic
tendency is built in `update_state!` with the same live velocity.) Called after every tendency
computation the stepper issues, once per stage.
"""
function cache_transport_velocity!(model)
    w_cache = model.timestepper.substepper.time_averaged_vertical_velocity_cache
    copyto!(parent(w_cache), parent(transport_velocities(model).w))
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The transport velocities the moisture and tracer tendencies in `Gⁿ` were built with: the live
horizontal components and the frozen vertical one. Only `w` is frozen — under adaptive implicit
vertical advection the horizontal fluxes stay fully explicit, so the implicit solve reads no
horizontal velocity, and condensate sediments vertically.
"""
tendency_transport_velocities(model) =
    merge(transport_velocities(model), (; w = model.timestepper.substepper.time_averaged_vertical_velocity_cache))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Update non-acoustic scalar fields (moisture, microphysics, tracers) with the
Wicker–Skamarock stage update ``u = u⁰ + Δt G``, rounded with the round-off
treatment `rounding`, followed by their implicit solves over `Δt_stage`. Iterates over prognostic
fields, skipping the ones the acoustic substep loop advances (see `acoustic_prognostic_names`).
`kernel_Δt` is `Δt_stage` as the kernels take it (see `kernel_time_step`).
"""
function scalar_substep!(model, Δt_stage, kernel_Δt, rounding)
    grid = model.grid
    arch = grid.architecture
    U⁰ = stage_start_state(model.timestepper)
    Gⁿ = model.timestepper.Gⁿ
    prognostic = prognostic_fields(model)
    names = keys(prognostic)
    acoustic_names = acoustic_prognostic_names(model)

    # Water species and tracers advect as mass fractions of the total density ρ = ρᵈ + Σρˣ
    # (see `scalar_tendency`), so the implicit solve is weighted with the same density; only
    # `update_state!` refreshes `total_density`, so it still holds the value the tendencies were
    # built with. The vertical velocity is frozen for the same reason (`cache_transport_velocity!`):
    # the acoustic loop has already overwritten the live time-averaged field with this stage's
    # average, while `Gⁿ` was scaled by the previous one.
    ρ = total_density(model.dynamics)
    velocities = tendency_transport_velocities(model)
    implicit_solver = model.timestepper.implicit_solver
    implicit_requests = []

    for (field_number, (name, u, u⁰, G)) in enumerate(zip(names, prognostic, U⁰, Gⁿ))
        name ∈ acoustic_names && continue

        key = rounding_key(rounding, model.clock.iteration, model.clock.stage, field_number)
        launch_rk3_substep!(rounding, arch, grid, u, u⁰, G, kernel_Δt, key)

        field_index = closure_scalar_index(model, name)
        advection = field_advection_scheme(model.advection, name)

        # Guarded on the solver rather than on `needs_implicit_solver(advection)`; see the note in
        # ssp_runge_kutta_3.jl for why that predicate would drop the mass-flux weighting.
        if !isnothing(implicit_solver)
            # The explicit tendency advected this species with the full transport velocity —
            # dynamical plus microphysical (terminal) — so the implicit half must split the
            # same combined velocity, or precipitating species lose the withheld fraction of
            # their sedimentation flux wherever the split engages (issue #914);
            # `implicit_advection_velocities` forms that sum and lets the implicit remainder
            # carry sedimenting condensate out through the bottom.
            w = implicit_advection_velocities(model.dynamics, velocities, name,
                                              model.microphysics, model.microphysical_fields)
            push!(implicit_requests, (u, field_index, implicit_step_scheme(advection), w))
        end
    end

    # The scalars' implicit solves, several fields per column thread, once every scalar has taken
    # its explicit update: no scalar's implicit coefficients depend on another scalar.
    if !isnothing(implicit_solver)
        scratch = implicit_step_scratch(model.timestepper.substepper, implicit_solver)
        implicit_steps_in_place!(implicit_requests, implicit_solver, scratch, model.closure, model.closure_fields,
                                 model.clock, fields(model), kernel_Δt, ρ)
    end

    # The tracers' solves have just moved sedimenting condensate implicitly; move its latent
    # content with it, from the state the solves produced and with the same frozen velocity
    # (see `implicit_sedimentation_step!`). The thermodynamic variable's post-loop solve follows
    # in `implicit_substep!`, so the moved content takes the same closure diffusion as the rest
    # of the field; its implicit vertical transport ran inside the substep loop, before this
    # step (a first-order splitting difference).
    isnothing(model.timestepper.implicit_solver) || implicit_sedimentation_step!(model, Δt_stage, velocities)

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The prognostic fields the acoustic substep loop advances — the dynamics-specific prognostics
(the compressible dry density), momentum and the thermodynamic variable — and which
`scalar_substep!` therefore skips.
"""
acoustic_prognostic_names(model) = tuple(keys(dynamics_prognostic_fields(model.dynamics))...,
                                         keys(model.momentum)...,
                                         thermodynamic_density_name(model.formulation))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The prognostic fields `scalar_substep!` advances — moisture, the microphysical prognostics and
the tracers: every prognostic the acoustic substep loop does not.
"""
scalar_prognostic_names(model) = filter(name -> name ∉ acoustic_prognostic_names(model),
                                        keys(prognostic_fields(model)))

#####
##### When the moisture and tracer tendencies are built: before or after the stage's acoustic loop
#####
##### By default (`PreLoopScalarTendencies`) `update_state!` builds them between stages, so a stage
##### advects its scalars with the mean velocity of the *preceding* loop (stage 1 with the
##### outer-step-start velocity), while continuity applies the stage's own loop mean: the
##### moisture mass flux is then not q/(1 − q) times the dry mass flux, and a uniform q does not
##### stay uniform (issue #1089). `PostLoopScalarTendencies` rebuilds them right after the stage's
##### own loop, as WRF (`rk_scalar_tend` with `ru_m, rv_m, ww_m`) and MPAS do, from that loop's
##### time-averaged mass flux. Both densities of the flux (ρ/ρᵈ)⟨ρᵈu⟩q are then stage-entry values
##### — `total_density` is refreshed only by `update_state!`, and the average divides by the
##### stage-entry ρᵈ — so their ratio is the uniform 1/(1 − qᵗ) and the flux is exactly
##### q/(1 − q) times the one continuity applied.
#####
##### What moves and what stays:
#####   - Advection, closure diffusion, forcing and the microphysical sources of the scalars are
#####     rebuilt after the loop (`compute_scalar_tendencies!` with the live transport velocity);
#####     `update_state!` skips them (`scalar_tendencies_in_update_state`). The stepper picks the
#####     build (`post_loop_scalar_tendency_kernel`): the full tendency added to `Gⁿ`, or, for a
#####     multirate infinitesimal step, the advection kept apart from the sources it treats as slow.
#####   - The flux boundary conditions and the closure's own scalar sources (TKE production) are
#####     still evaluated before the loop, at the stage-entry state, onto zeroed scalar
#####     tendencies, and the post-loop build adds to them. Evaluating the closure sources after
#####     the loop would pair stage-end velocities and θ with stage-entry TKE and diffusivities.
#####   - `TendencyCallsite` callbacks fire a second time after the post-loop build, so their
#####     scalar edits are not overwritten. Their momentum edits in that second firing are inert:
#####     the loop has consumed the momentum tendencies, which the next `update_state!` rebuilds.
#####   - The thermodynamic variable's condensate sedimentation term feeds the acoustic loop, so
#####     it is still built before the loop with the preceding loop's transport velocity, while
#####     the condensate mass it accompanies now falls with the current loop's. The mismatch is a
#####     first-order splitting difference, of the same order as the existing split between the
#####     explicit sedimentation term and `implicit_sedimentation_step!`, and only arises with
#####     sedimenting condensate.
#####   - The transport velocity is frozen (`cache_transport_velocity!`) after the post-loop
#####     build, so the implicit remainder of the scalar update splits the velocity the explicit
#####     fraction was scaled by.
#####

"""
$(TYPEDEF)

Build the moisture and tracer tendencies in `update_state!`, between stages, with the
preceding acoustic loop's time-averaged transport velocity (the default).
"""
struct PreLoopScalarTendencies end

"""
$(TYPEDEF)

Rebuild the moisture and tracer tendencies after each stage's own acoustic loop, with that
loop's time-averaged transport velocity, so their mass fluxes are consistent with the dry-mass
flux continuity applied (issue #1089). Select it with
`timestepper_kwargs = (; post_loop_scalar_tendencies = true)`.
"""
struct PostLoopScalarTendencies end

scalar_tendency_timing(post_loop_scalar_tendencies::Bool) =
    post_loop_scalar_tendencies ? PostLoopScalarTendencies() : PreLoopScalarTendencies()

# The `AcousticTimeStepper` method is in acoustic_runge_kutta_3.jl, with the abstract type.
scalar_tendencies_in_update_state(::PreLoopScalarTendencies) = true
scalar_tendencies_in_update_state(::PostLoopScalarTendencies) = false

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Prepare the scalar tendencies for the stage-entry sources added before the acoustic loop. With
`PostLoopScalarTendencies`, zero them: `update_state!` left no tendency there, and the flux
boundary conditions and closure sources about to be added are what the post-loop build adds to.
"""
prepare_scalar_tendencies!(::PreLoopScalarTendencies, model) = nothing

function prepare_scalar_tendencies!(::PostLoopScalarTendencies, model)
    # TODO: the acoustic loop projects the stage's moisture tendencies onto the pressure before
    # the loop (`project_moisture_onto_pressure!`), but post-loop scalar tendencies are only built
    # after it, so the projection would read zeroed moisture tendencies. Until the projection has a
    # defined post-loop counterpart, the combination is rejected.
    throw(ArgumentError("post_loop_scalar_tendencies = true is not supported with the acoustic loop's " *
                        "moisture pressure projection: the projection runs before the loop and would read " *
                        "moisture tendencies that are only built after it. Use post_loop_scalar_tendencies = false."))
    Gⁿ = model.timestepper.Gⁿ
    for name in scalar_prognostic_names(model)
        fill!(parent(Gⁿ[name]), 0)
    end
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

With `PostLoopScalarTendencies`, build the moisture and tracer tendencies after the stage's
acoustic loop with its time-averaged transport velocity, adding them to the stage-entry sources
(flux boundary conditions and closure sources) built before the loop; fire the `TendencyCallsite`
callbacks again on the result; assemble the scalars' stage forcing from it
(`assemble_post_loop_stage_forcing!`, a no-op for Wicker–Skamarock RK3); and freeze the transport
velocity for the implicit remainder of the scalar update. The tracers are converted to specific
values around the build, as in `update_state!`, with the stage-entry total density that the
moisture flux is weighted by.
"""
compute_post_loop_scalar_tendencies!(::PreLoopScalarTendencies, model, Δt, callbacks) = nothing

function compute_post_loop_scalar_tendencies!(::PostLoopScalarTendencies, model, Δt, callbacks)
    tracer_density_to_specific!(model)
    kernel = post_loop_scalar_tendency_kernel(model.timestepper)
    compute_scalar_tendencies!(model, fields(model), transport_velocities(model), kernel)

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    assemble_post_loop_stage_forcing!(model.timestepper, model, Δt)
    tracer_specific_to_density!(model)
    cache_transport_velocity!(model)
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The `compute_scalar_tendencies!` kernel of the post-loop scalar tendency build: by default
`add_scalar_tendency!`, the full tendency added to the stage-entry sources in `Gⁿ`. A stepper that
treats a scalar's advection and its sources differently (`MultirateInfinitesimalStep`) builds the
two apart instead.
"""
post_loop_scalar_tendency_kernel(timestepper) = add_scalar_tendency!

#####
##### Implicit vertical solve for the acoustic prognostics
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Freeze the stage-entry advecting velocity and the density that weights it so `implicit_substep!` sizes
the withheld remainder from the state whose fluxes the slow tendencies split (invariant:
wᴸ = wᵉ + wⁱ). The full wᴸ is cached, not the clipped wᵉ, which loses the remainder in
saturated cells. A no-op when the substepper carries no cache.
"""
cache_advecting_state!(model) =
    cache_advecting_state!(model.timestepper.substepper.vertical_velocity_cache,
                           model.timestepper.substepper.density_cache, model)

cache_advecting_state!(::Nothing, ::Nothing, model) = nothing

# One launch for both copies, sized to the w array; ρ has one fewer z level, so its index
# is clamped and the top ρ value is redundantly (but harmlessly) rewritten.
@kernel function _cache_advecting_state!(w_cache, ρ_cache, w, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds w_cache[i, j, k] = w[i, j, k]
    k′ = min(k, size(ρ_cache, 3))
    @inbounds ρ_cache[i, j, k′] = ρ[i, j, k′]
end

function cache_advecting_state!(w_cache, ρ_cache, model)
    w = advecting_vertical_velocity(model.dynamics, model.velocities)
    ρ = continuity_density(model.dynamics)
    params = KernelParameters(size(parent(w_cache)), (0, 0, 0))
    launch!(architecture(model.grid), model.grid, params, _cache_advecting_state!,
            parent(w_cache), parent(ρ_cache), parent(w), parent(ρ))
    return nothing
end

# Frozen stage-entry state when the cache exists; live fields otherwise (closure-only
# solves keep their current behavior).
advecting_state(model) =
    advecting_state(model.timestepper.substepper.vertical_velocity_cache,
                    model.timestepper.substepper.density_cache, model)

advecting_state(::Nothing, ::Nothing, model) = (advecting_vertical_velocity(model.dynamics, model.velocities), continuity_density(model.dynamics))

advecting_state(w_cache, ρ_cache, model) = (w_cache, ρ_cache)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Apply the vertically-implicit tridiagonal solve to the prognostics that the acoustic substep
loop advances: momentum and the thermodynamic variable. Dispatch on the timestepper's
`implicit_solver` selects the method: `nothing` means nothing in the model is vertically
implicit and the substep is a no-op.

Each field's solve combines every implicit vertical piece into a single tridiagonal system:
the first-order-upwind remainder of adaptive implicit vertical advection (whose CFL-limited
explicit flux the slow tendencies carry through the advection dispatch), plus vertically-implicit
closure diffusion. Explicit advection schemes contribute no advection coefficients and explicit
closures no diffusion coefficients, so each combination reduces to the right system. The solve
runs once per RK stage after the substep loop and after the scalar update, whose
`implicit_sedimentation_step!` adds to the thermodynamic variable content that takes this
solve's diffusion too, over the stage interval — the operator split WRF and CM1 use for their
implicit vertical pieces. Under an adaptive-implicit thermodynamic scheme the advection half of
this solve is empty (`postloop_thermodynamic_scheme`): it was applied inside the loop. Continuity takes no implicit solve: the continuity-density tendency is the
acoustic mass-flux divergence itself, not scalar advection.

The advecting velocity passed to each solve must be the one its slow tendency was built with,
so the explicit/implicit velocity split is consistent: the RK stage-entry predictor velocities
(see `slow_thermodynamic_velocities`), not the
substepper's time-averaged transport velocities that moisture and tracers use. The one exception,
on both sides of the split, is the condensate sedimentation term, which pairs with the tracers'
mass fluxes and reads their velocity.
"""
# First-order upwind flux of the stage-entry thermodynamic state carried by the implicit
# half's velocity wⁱ = (1 - s) w, density-weighted like the implicit Center-field
# coefficients so base + perturbation sum to the full-field operator.
@inline function implicit_advective_base_flux(i, j, k, grid, scheme, td, W, ρθ, ρᵈ)
    wⁱ = implicit_vertical_velocityᶜᶜᶠ(i, j, k, grid, scheme, td, W)
    ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᵈ)
    θ⁻ = @inbounds ρθ[i, j, k-1] / ρᵈ[i, j, k-1]
    θ⁺ = @inbounds ρθ[i, j, k]   / ρᵈ[i, j, k]
    return Azᶜᶜᶠ(i, j, k, grid) * ρᶠ * (max(wⁱ, 0) * θ⁻ + min(wⁱ, 0) * θ⁺)
end

@kernel function _implicit_advection_base_tendency!(Gρθ, grid, scheme, td, W, ρθ, ρᵈ)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρθ[i, j, k] -= V⁻¹ᶜᶜᶜ(i, j, k, grid) *
        δzᵃᵃᶜ(i, j, k, grid, implicit_advective_base_flux, scheme, td, W, ρθ, ρᵈ)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Fold the base-state part of the IMEX vertical-advection split's implicit half into the
slow tendency: `Gˢρθ` gains the first-order upwind flux divergence of the frozen stage-entry
(ρθ, ρᵈ) carried by wⁱ = (1 - s) w. The predictors then apply it per substep with the same
Crank-Nicolson factors as the rest of the slow tendency, so the acoustic pressure adjusts to
the implicit-half transport inside the loop (issue #897). The perturbation part is handled
per substep by `implicit_advection_substep!` inside the loop. A no-op unless the scheme's
vertical discretization is adaptive-implicit (dispatch below).
"""
add_implicit_advection_tendency!(model) =
    add_implicit_advection_tendency!(model,
        field_advection_scheme(model.advection, thermodynamic_density_name(model.formulation)))

add_implicit_advection_tendency!(model, advection) = nothing

function add_implicit_advection_tendency!(model, advection::AdaptiveImplicitVerticalAdvection)
    grid = model.grid
    scheme = vertical_scheme(advection)
    td = OceananigansTimeSteppers.time_discretization(scheme)
    w, ρᵈ = advecting_state(model)
    θ_name = thermodynamic_density_name(model.formulation)
    ρθ = prognostic_fields(model)[θ_name]
    launch!(architecture(grid), grid, :xyz, _implicit_advection_base_tendency!,
            model.timestepper.Gⁿ[θ_name], grid, scheme, td, w, ρθ, ρᵈ)
    return nothing
end

# The implicit half of the IMEX thermodynamic split is applied inside the acoustic loop
# (`implicit_advection_substep!`), so post-loop the thermodynamic variable keeps only
# density-weighted closure diffusion under an adaptive-implicit scheme.
postloop_thermodynamic_scheme(advection, ρ) = implicit_step_scheme(advection, ρ)
postloop_thermodynamic_scheme(::AdaptiveImplicitVerticalAdvection, ρ) = implicit_step_scheme(nothing, ρ)

implicit_substep!(model, Δt_stage) =
    implicit_substep!(model, model.timestepper.implicit_solver, Δt_stage)

# No implicit solver ⇒ nothing in the model is vertically implicit.
implicit_substep!(model, ::Nothing, Δt_stage) = nothing

function implicit_substep!(model, implicit_solver, Δt_stage)
    # Momentum and the thermodynamic variable are continuity-density-weighted (ρu = ρᵈ u, ρθ = ρᵈ θ).
    # Frozen stage-entry (w, ρᵈ), so the explicit and implicit halves partition one transport.
    w, ρᵈ = advecting_state(model)

    # The diffusion half of each row is weighted with the *live* continuity density instead: it
    # reconstructs u and θ from the prognostic the acoustic loop has just advanced, and unlike the
    # advective split it has no explicit fraction to pair with the frozen state.
    diffusion_density = continuity_density(model.dynamics)

    # The step enters every coefficient of the column sweeps: in the float type of the grid.
    grid = model.grid
    Δt = kernel_time_step(architecture(grid), grid, Δt_stage)

    prognostic = prognostic_fields(model)
    momentum_scheme = implicit_step_scheme(model.advection.momentum, diffusion_density)

    θ_name = thermodynamic_density_name(model.formulation)
    θ_advection = field_advection_scheme(model.advection, θ_name)
    θ_scheme = postloop_thermodynamic_scheme(θ_advection, diffusion_density)
    θ_velocities = merge(slow_thermodynamic_velocities(model), (; w))

    # One request per field: (field, closure tracer index, scheme, velocities).
    requests = [(prognostic.ρu, nothing, momentum_scheme, (; w)),
                (prognostic.ρv, nothing, momentum_scheme, (; w)),
                (prognostic.ρw, nothing, momentum_scheme, (; w)),
                (prognostic[θ_name], closure_scalar_index(model, θ_name), θ_scheme, θ_velocities)]

    scratch = implicit_step_scratch(model.timestepper.substepper, implicit_solver)
    implicit_steps_in_place!(requests, implicit_solver, scratch, model.closure, model.closure_fields,
                             model.clock, fields(model), Δt, ρᵈ)

    return nothing
end
