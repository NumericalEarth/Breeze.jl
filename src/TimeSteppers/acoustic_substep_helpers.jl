using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields, architecture
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
    implicit_step_scheme,
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

    AtmosphereModels.compute_thermodynamic_tendency!(model, common_args)

    return nothing
end

#####
##### Scalar (tracer / moisture) update with time-averaged velocities
#####

"""
$(TYPEDSIGNATURES)

Freeze the time-averaged transport velocity that `update_state!` just built the moisture and
tracer tendencies from. The next acoustic loop resets and rebuilds `time_averaged_velocities`,
so `scalar_substep!` cannot read it live: the implicit remainder has to split the same velocity
the explicit fraction in `Gⁿ` was scaled by (invariant: ⟨w⟩ = wᵉ + wⁱ). Called after every
tendency computation the stepper issues, once per stage. A no-op when the substepper carries no
cache — without adaptive-implicit advection there is no split to pair.
"""
cache_transport_velocity!(model) =
    cache_transport_velocity!(model.timestepper.substepper.time_averaged_vertical_velocity_cache, model)

cache_transport_velocity!(::Nothing, model) = nothing

function cache_transport_velocity!(w_cache, model)
    copyto!(parent(w_cache), parent(transport_velocities(model).w))
    return nothing
end

"""
$(TYPEDSIGNATURES)

The transport velocities the scalar tendencies in `Gⁿ` were built with: the frozen vertical
component when the cache exists, the live field otherwise. Only `w` is frozen — under adaptive
implicit vertical advection the horizontal fluxes stay fully explicit, so the implicit solve
reads no horizontal velocity.
"""
tendency_transport_velocities(model) =
    tendency_transport_velocities(model.timestepper.substepper.time_averaged_vertical_velocity_cache, model)

tendency_transport_velocities(::Nothing, model) = transport_velocities(model)

tendency_transport_velocities(w_cache, model) = merge(transport_velocities(model), (; w = w_cache))

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
            implicit_step!(u,
                           model.timestepper.implicit_solver,
                           model.closure,
                           model.closure_fields,
                           field_index,
                           model.clock,
                           fields(model),
                           Δt_implicit,
                           implicit_step_scheme(advection),
                           velocities,
                           ρ)
        end
    end

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
runs once per RK stage after the substep loop, over the stage interval — the operator split WRF
and CM1 use for their implicit vertical pieces. Continuity takes no implicit solve: the
coupling-density tendency is the acoustic mass-flux divergence itself, not scalar advection.

The advecting velocity passed to each solve must be the one its slow tendency was built with,
so the explicit/implicit velocity split is consistent: the RK stage-entry predictor velocities
(see `compute_slow_momentum_tendencies!` and `compute_slow_scalar_tendencies!`), not the
substepper's time-averaged transport velocities that moisture and tracers use.
"""
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
                   implicit_step_scheme(θ_advection, diffusion_density),
                   merge(slow_thermodynamic_velocities(model), (; w)),
                   ρᵈ)

    return nothing
end
