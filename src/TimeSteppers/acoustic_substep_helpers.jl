using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields, architecture
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection, vertical_scheme,
                              implicit_vertical_velocityᶜᶜᶠ
using Oceananigans.Operators: Azᶜᶜᶠ, δzᵃᵃᶜ, V⁻¹ᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.Utils: launch!, KernelParameters

using Oceananigans.TimeSteppers: implicit_step!

using Breeze.AtmosphereModels:
    AtmosphereModels,
    AtmosphereModel,
    SlowTendencyMode,
    advecting_momentum,
    advecting_vertical_velocity,
    dynamics_density,
    total_density,
    thermodynamic_density_name,
    transport_velocities,
    field_advection_scheme,
    closure_scalar_index,
    dynamics_prognostic_fields,
    implicit_advection_velocities,
    implicit_step_scheme,
    implicit_sedimentation_step!,
    compute_x_momentum_tendency!,
    compute_y_momentum_tendency!,
    compute_z_momentum_tendency!,
    compute_dynamics_tendency!,
    specific_prognostic_moisture

using Breeze.CompressibleEquations: CompressibleDynamics
using Breeze.TerrainFollowingDiscretization: TerrainMetrics

const TerrainCompressibleAcousticModel =
    AtmosphereModel{<:CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:TerrainMetrics}}

#####
##### Slow momentum tendencies
#####
##### `SlowTendencyMode` zeros the pressure-gradient force and buoyancy in the
##### momentum tendency assembly. The PGF and buoyancy are handled in
##### linearized form inside the substep loop, so the slow tendency carries
##### only advection, Coriolis, closure, and forcing.
#####

slow_momentum_advection_momentum(model) = model.momentum

function slow_momentum_advection_momentum(model::TerrainCompressibleAcousticModel)
    return advecting_momentum(model)
end

"""
$(TYPEDSIGNATURES)

Compute slow momentum tendencies (advection, Coriolis, closure, forcing).
The pressure-gradient force and buoyancy are excluded; they are handled
in linearized form inside the acoustic substep loop.
"""
function compute_slow_momentum_tendencies!(model)
    grid = model.grid
    arch = architecture(grid)

    slow_dynamics = SlowTendencyMode(model.dynamics)

    model_fields = fields(model)

    momentum_args = (
        dynamics_density(model.dynamics),
        model.advection.momentum,
        model.velocities,
        model.closure,
        model.closure_fields,
        slow_momentum_advection_momentum(model),
        model.coriolis,
        model.clock,
        model_fields)

    u_args = tuple(momentum_args..., model.forcing.ρu, slow_dynamics)
    v_args = tuple(momentum_args..., model.forcing.ρv, slow_dynamics)

    w_args = tuple(momentum_args..., model.forcing.ρw,
                   slow_dynamics,
                   model.formulation,
                   model.temperature,
                   specific_prognostic_moisture(model),
                   model.microphysics,
                   model.microphysical_fields,
                   model.thermodynamic_constants)

    Gⁿ = model.timestepper.Gⁿ

    launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gⁿ.ρu, grid, u_args)
    launch!(arch, grid, :xyz, compute_y_momentum_tendency!, Gⁿ.ρv, grid, v_args)
    launch!(arch, grid, :xyz, compute_z_momentum_tendency!, Gⁿ.ρw, grid, w_args)

    return nothing
end

#####
##### Slow scalar tendencies (density and thermodynamic variable)
#####

slow_thermodynamic_velocities(model) = model.velocities

function slow_thermodynamic_velocities(model::TerrainCompressibleAcousticModel)
    u = model.velocities.u
    v = model.velocities.v
    w̃ = model.dynamics.contravariant_vertical_velocity
    return (; u, v, w=w̃)
end

"""
$(TYPEDSIGNATURES)

Compute slow tendencies for density and the thermodynamic variable:

  - ``Gˢ_ρᵈ = -∇·m``: full dry-density tendency (continuity equation),
    written into `model.timestepper.Gⁿ.ρᵈ`.
  - ``Gˢ_ρᵡ``: full thermodynamic-density tendency (advection + physics).
"""
function compute_slow_scalar_tendencies!(model)
    compute_dynamics_tendency!(model)

    # Theta's slow tendency uses the current RK predictor velocity
    # (`model.velocities`), matching WRF (`rk_tendency` in `solve_em.F`,
    # called with `grid%ru, grid%rv, grid%ww`) and MPAS. Routing the
    # substepper's time-averaged velocity here creates a closed feedback
    # loop (Gⁿ.ρθ → ρθ′ → PGF → (ρu)′ → time-averaged velocity →
    # next stage's Gⁿ.ρθ) that destabilizes the rest atmosphere; T4
    # blows up at production Δt. For nonflat terrain, the same current
    # predictor is used horizontally while vertical scalar transport uses
    # the current terrain-following `w̃`. The dynamics-transport split applies
    # only to **moisture, tracers, chemistry, TKE** — those tendencies are
    # computed in `update_state!`'s `compute_tendencies!` via
    # `transport_velocities(model)`, which the `AcousticRungeKutta3` override
    # routes to the substepper's time-averaged velocity.
    #
    # The condensate sedimentation term is the exception: it pairs its content fluxes with the
    # tracer tendencies' mass fluxes, which recombine only at the velocity they were formed at
    # (see `sedimentation_tendency`), so it reads the frozen copy
    # (`tendency_transport_velocities`), as `implicit_sedimentation_step!` does for the implicit
    # remainder. Being a difference of those fluxes, zero wherever nothing sediments, it forms no
    # part of the feedback loop above.
    common_args = (
        model.dynamics,
        model.formulation,
        model.thermodynamic_constants,
        specific_prognostic_moisture(model),
        slow_thermodynamic_velocities(model),
        model.microphysics,
        model.microphysical_fields,
        model.closure,
        model.closure_fields,
        model.clock,
        fields(model))

    AtmosphereModels.compute_thermodynamic_tendency!(model, common_args, tendency_transport_velocities(model).w)

    return nothing
end

#####
##### Scalar (tracer / moisture) update with time-averaged velocities
#####

"""
$(TYPEDSIGNATURES)

Freeze the time-averaged transport velocity that `update_state!` just built the moisture and
tracer tendencies from. The next acoustic loop resets and rebuilds `time_averaged_velocities`
(and `freeze_linearization_state!` reseeds it at outer-step start), so the stage cannot read it
live. Two readers pair fluxes with those tendencies: the implicit remainder in `scalar_substep!`,
which has to split the same velocity the explicit fraction in `Gⁿ` was scaled by (invariant:
⟨w⟩ = wᵉ + wⁱ), and the condensate sedimentation term of the thermodynamic tendency (see
`compute_slow_scalar_tendencies!`). Called after every tendency computation the stepper issues,
once per stage.
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

Update non-acoustic scalar fields (moisture, microphysics, tracers) using the given kernel.
Iterates over prognostic fields, skipping the ones the acoustic substep loop advances
(see `acoustic_prognostic_names`).
"""
function scalar_substep!(model, kernel!, Δt_implicit, kernel_args...)
    grid = model.grid
    arch = grid.architecture
    U⁰ = model.timestepper.U⁰
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

    for (name, u, u⁰, G) in zip(names, prognostic, U⁰, Gⁿ)
        name ∈ acoustic_names && continue

        launch!(arch, grid, :xyz, kernel!, u, u⁰, G, kernel_args...)

        field_index = closure_scalar_index(model, name)
        advection = field_advection_scheme(model.advection, name)

        # Guarded on the solver rather than on `needs_implicit_solver(advection)`; see the note in
        # ssp_runge_kutta_3.jl for why that predicate would drop the mass-flux weighting.
        if !isnothing(model.timestepper.implicit_solver)
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
                           Δt_implicit,
                           implicit_step_scheme(advection),
                           implicit_advection_velocities(model.dynamics, velocities, name,
                                                         model.microphysics, model.microphysical_fields),
                           ρ)
        end
    end

    # The tracers' solves have just moved sedimenting condensate implicitly; move its latent
    # content with it, from the state the solves produced and with the same frozen velocity
    # (see `implicit_sedimentation_step!`). The thermodynamic variable's post-loop solve follows
    # in `implicit_substep!`, so the moved content takes the same closure diffusion as the rest
    # of the field; its implicit vertical transport ran inside the substep loop, before this
    # step (a first-order splitting difference).
    isnothing(model.timestepper.implicit_solver) || implicit_sedimentation_step!(model, Δt_implicit, velocities)

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

Freeze the stage-entry advecting velocity and carrier density so `implicit_substep!` sizes
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
    ρ = dynamics_density(model.dynamics)
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

advecting_state(::Nothing, ::Nothing, model) = (advecting_vertical_velocity(model.dynamics, model.velocities), dynamics_density(model.dynamics))

advecting_state(w_cache, ρ_cache, model) = (w_cache, ρ_cache)

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
runs once per RK stage after the substep loop and after the scalar update, whose
`implicit_sedimentation_step!` adds to the thermodynamic variable content that takes this
solve's diffusion too, over the stage interval — the operator split WRF and CM1 use for their
implicit vertical pieces. Under an adaptive-implicit thermodynamic scheme the advection half of
this solve is empty (`postloop_thermodynamic_scheme`): it was applied inside the loop. Continuity takes no implicit solve: the coupling-density tendency is the
acoustic mass-flux divergence itself, not scalar advection.

The advecting velocity passed to each solve must be the one its slow tendency was built with,
so the explicit/implicit velocity split is consistent: the RK stage-entry predictor velocities
(see `compute_slow_momentum_tendencies!` and `compute_slow_scalar_tendencies!`), not the
substepper's time-averaged transport velocities that moisture and tracers use. The one exception,
on both sides of the split, is the condensate sedimentation term, which pairs with the tracers'
mass fluxes and reads their velocity (see `compute_slow_scalar_tendencies!`).
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

implicit_substep!(model, Δt_stage) =
    implicit_substep!(model, model.timestepper.implicit_solver, Δt_stage)

# No implicit solver ⇒ nothing in the model is vertically implicit.
implicit_substep!(model, ::Nothing, Δt_stage) = nothing

function implicit_substep!(model, implicit_solver, Δt_stage)
    # Momentum and the thermodynamic variable are coupling-density-weighted (ρu = ρᵈ u, ρθ = ρᵈ θ).
    # Frozen stage-entry (w, ρᵈ), so the explicit and implicit halves partition one transport.
    w, ρᵈ = advecting_state(model)

    # The diffusion half of each row is weighted with the *live* coupling density instead: it
    # reconstructs u and θ from the prognostic the acoustic loop has just advanced, and unlike the
    # advective split it has no explicit fraction to pair with the frozen state.
    diffusion_density = dynamics_density(model.dynamics)

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

    θ_name = thermodynamic_density_name(model.formulation)
    θ_advection = field_advection_scheme(model.advection, θ_name)
    implicit_step!(prognostic[θ_name],
                   implicit_solver,
                   model.closure,
                   model.closure_fields,
                   closure_scalar_index(model, θ_name),
                   model.clock,
                   fields(model),
                   Δt_stage,
                   postloop_thermodynamic_scheme(θ_advection, diffusion_density),
                   merge(slow_thermodynamic_velocities(model), (; w)),
                   ρᵈ)

    return nothing
end
