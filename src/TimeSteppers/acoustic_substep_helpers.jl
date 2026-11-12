using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields, architecture
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection, vertical_scheme,
                              implicit_vertical_velocityᶜᶜᶠ
using Oceananigans.Operators: Azᶜᶜᶠ, δzᵃᵃᶜ, V⁻¹ᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.Utils: launch!, KernelParameters

using Oceananigans.TimeSteppers: implicit_step!

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
    thermodynamic_density

using Breeze.CompressibleEquations: CompressibleDynamics
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

"""
$(TYPEDSIGNATURES)

Freeze the time-averaged transport velocity that `update_state!` just built the moisture and
tracer tendencies from. The next acoustic loop resets and rebuilds `time_averaged_velocities`
(and `freeze_linearization_state!` reseeds it at outer-step start), so the stage cannot read it
live. The implicit remainder in `scalar_substep!` pairs fluxes with those tendencies: it has to
split the same velocity the explicit fraction in `Gⁿ` was scaled by (invariant: ⟨w⟩ = wᵉ + wⁱ),
as does the content of the mass those solves sediment. (The stage forms the explicit condensate
sedimentation tendency from this copy too; see `add_explicit_sedimentation_tendency!`.) Called
after every tendency computation the stepper issues, once per stage.
"""
function cache_transport_velocity!(model)
    w_cache = model.timestepper.substepper.time_averaged_vertical_velocity_cache
    copyto!(parent(w_cache), parent(transport_velocities(model).w))
    return nothing
end

"""
$(TYPEDSIGNATURES)

The transport velocities the moisture and tracer tendencies in `Gⁿ` were built with: the live
horizontal components and the frozen vertical one. Only `w` is frozen — under adaptive implicit
vertical advection the horizontal fluxes stay fully explicit, so the implicit solve reads no
horizontal velocity, and condensate sediments vertically.
"""
tendency_transport_velocities(model) =
    merge(transport_velocities(model), (; w = model.timestepper.substepper.time_averaged_vertical_velocity_cache))

"""
$(TYPEDSIGNATURES)

Update non-acoustic scalar fields (moisture, microphysics, tracers) with the
Wicker–Skamarock stage update ``u = u⁰ + Δt G``, rounded with the round-off
treatment `rounding`, followed by their implicit solves over `Δt_stage`. Iterates over prognostic
fields, skipping the ones the acoustic substep loop advances (see `acoustic_prognostic_names`).
`kernel_Δt` is `Δt_stage` as the kernels take it (see `kernel_time_step`), `first_stage` says
whether the stage starts from the outer-step-start state, and `extrapolation` is the ratio of the
stage's interval to the one its stage-entry state was advanced over (see `stage_extrapolation`).

The explicit updates of every field come first, then the composition content of the moisture they
give (`collect_composition_content!`), then the solves: the substep loop integrates the
thermodynamic variable with the slow tendencies that pair with the explicit updates, and the
variable receives what pairs with the solves after the loop (see the section comment on the
composition content bypass in `acoustic_runge_kutta_3.jl`).
"""
function scalar_substep!(model, Δt_stage, kernel_Δt, rounding, first_stage, extrapolation)
    grid = model.grid
    arch = grid.architecture
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ
    prognostic = prognostic_fields(model)
    names = keys(prognostic)
    acoustic_names = acoustic_prognostic_names(model)

    for (field_number, (name, u, u⁰, G)) in enumerate(zip(names, prognostic, U⁰, Gⁿ))
        name ∈ acoustic_names && continue
        key = rounding_key(rounding, model.clock.iteration, model.clock.stage, field_number)
        launch_rk3_substep!(rounding, arch, grid, u, u⁰, G, kernel_Δt, key)
    end

    # The contents of the outer-step-start moisture and of the moisture the explicit tendencies
    # give, and the ramp between them out of the slow tendency of the thermodynamic variable.
    collect_composition_content!(model.timestepper.substepper, model, U⁰, Gⁿ, kernel_Δt, first_stage, extrapolation)

    isnothing(model.timestepper.implicit_solver) && return nothing

    # Water species and tracers advect as mass fractions of the total density ρ = ρᵈ + Σρˣ
    # (see `scalar_tendency`), so the implicit solve is weighted with the same density; only
    # `update_state!` refreshes `total_density`, so it still holds the value the tendencies were
    # built with. The vertical velocity is the frozen copy `Gⁿ` was scaled by
    # (`cache_transport_velocity!`), which every acoustic loop leaves intact while it rebuilds
    # the live time-averaged field.
    ρ = total_density(model.dynamics)
    velocities = tendency_transport_velocities(model)

    for (name, u) in zip(names, prognostic)
        name ∈ acoustic_names && continue

        field_index = closure_scalar_index(model, name)
        advection = field_advection_scheme(model.advection, name)

        # Guarded on the solver rather than on `needs_implicit_solver(advection)`; see the note in
        # ssp_runge_kutta_3.jl for why that predicate would drop the mass-flux weighting.
        #
        # The explicit tendency advected this species with the full transport velocity —
        # dynamical plus microphysical (terminal) — so the implicit half must split the
        # same combined velocity, or precipitating species lose the withheld fraction of
        # their sedimentation flux wherever the split engages (issue #914);
        # `implicit_advection_velocities` forms that sum and lets the implicit remainder
        # carry sedimenting condensate out through the bottom.
        implicit_step!(u,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       field_index,
                       model.clock,
                       fields(model),
                       Δt_stage,
                       implicit_step_scheme(advection),
                       implicit_advection_velocities(model.dynamics, velocities, name,
                                                     model.microphysics, model.microphysical_fields),
                       ρ)
    end

    # The tracers' solves have just moved sedimenting condensate implicitly. Move the content of
    # that mass in its two parts (see `add_sedimentation_tendency_parts!`): the mass fluxes from
    # the solved densities and the same frozen velocity, the content per unit mass from the
    # stage-entry state.
    add_implicit_sedimentation_content!(model, velocities, kernel_Δt)

    return nothing
end

"""
$(TYPEDSIGNATURES)

The prognostic fields the acoustic substep loop advances — the dynamics-specific prognostics
(the compressible dry density), momentum and the thermodynamic variable — and which
`scalar_substep!` therefore skips.
"""
acoustic_prognostic_names(model) = tuple(keys(dynamics_prognostic_fields(model.dynamics))...,
                                         keys(model.momentum)...,
                                         thermodynamic_density_name(model.formulation))

#####
##### Implicit vertical solve for the acoustic prognostics
#####

"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

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

"""
$(TYPEDSIGNATURES)

Apply the vertically-implicit tridiagonal solve to the prognostics that the acoustic substep
loop advances: momentum and the thermodynamic variable. Dispatch on the timestepper's
`implicit_solver` selects the method: `nothing` means nothing in the model is vertically
implicit and the substep is a no-op.

Each field's solve combines every implicit vertical piece into a single tridiagonal system:
the first-order-upwind remainder of adaptive implicit vertical advection (whose CFL-limited
explicit flux the slow tendencies carry through the advection dispatch), plus vertically-implicit
closure diffusion. Explicit advection schemes contribute no advection coefficients and explicit
closures no diffusion coefficients, so each combination reduces to the right system. The solve
runs once per RK stage after the substep loop, over the stage interval — the operator split WRF
and CM1 use for their implicit vertical pieces. The thermodynamic variable enters its solve with
the stage's composition content added (`apply_composition_content!`). Under an
adaptive-implicit thermodynamic scheme the advection half of the variable's solve is empty
(`postloop_thermodynamic_scheme`): it was applied inside the loop. Continuity takes no implicit
solve: the coupling-density tendency is the acoustic mass-flux divergence itself, not scalar
advection.

The advecting velocity passed to each solve must be the one its slow tendency was built with,
so the explicit/implicit velocity split is consistent: the RK stage-entry predictor velocities
(see `slow_thermodynamic_velocities`), not the
substepper's time-averaged transport velocities that moisture and tracers use. The one exception,
on both sides of the split, is the condensate sedimentation content, which pairs with the
tracers' mass fluxes and reads their velocity.
"""
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

    prognostic = prognostic_fields(model)
    momentum_advection = model.advection.momentum
    for name in (:ρu, :ρv, :ρw)
        implicit_step!(prognostic[name],
                       implicit_solver,
                       model.closure,
                       model.closure_fields,
                       nothing,
                       model.clock,
                       fields(model),
                       Δt_stage,
                       implicit_step_scheme(momentum_advection, diffusion_density),
                       (; w),
                       ρᵈ)
    end

    thermodynamic_implicit_step!(model, implicit_solver, Δt_stage)
    return nothing
end

# The advective half of the variable's post-loop operator is weighted with the frozen stage-entry
# (w, ρᵈ) and the diffusive half with the live continuity density (see `implicit_substep!`).
postloop_thermodynamic_scheme(model) =
    postloop_thermodynamic_scheme(field_advection_scheme(model.advection, thermodynamic_density_name(model.formulation)),
                                  continuity_density(model.dynamics))

# Oceananigans' solve of the variable, reading the live model fields.
function thermodynamic_implicit_step!(model, implicit_solver, Δt_stage)
    w, ρᵈ = advecting_state(model)
    θ_name = thermodynamic_density_name(model.formulation)
    implicit_step!(thermodynamic_density(model.formulation), implicit_solver, model.closure, model.closure_fields,
                   closure_scalar_index(model, θ_name), model.clock, fields(model), Δt_stage,
                   postloop_thermodynamic_scheme(model), (; w), ρᵈ)
    return nothing
end
