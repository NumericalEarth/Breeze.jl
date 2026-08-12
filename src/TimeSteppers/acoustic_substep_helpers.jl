using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields, architecture
using Oceananigans.Advection: needs_implicit_solver, AdaptiveImplicitVerticalAdvection,
                              vertical_scheme, explicit_velocity_scaleᶜᶜᶠ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: Azᶜᶜᶠ, δzᵃᵃᶜ, V⁻¹ᶜᶜᶜ
using Oceananigans.Utils: launch!

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
    implicit_step_advection,
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

"""
$(TYPEDSIGNATURES)

Set every adaptive-implicit advection scheme's split time step to `Δt_stage`. The
Wicker–Skamarock stages act over β Δt, so the CFL-scaled explicit/implicit velocity split
must be computed from the stage interval — not from `clock.last_Δt`, which the generic
`update_advection_timestep!` fallback would use (stale by one outer step, and `Inf` before
the first).
"""
set_advection_timestep!(advection, Δt_stage) = nothing

function set_advection_timestep!(a::AdaptiveImplicitVerticalAdvection, Δt_stage)
    td = OceananigansTimeSteppers.time_discretization(a)
    td.Δt[] = Δt_stage
    return nothing
end

set_advection_timestep!(a::FluxFormAdvection, Δt_stage) = set_advection_timestep!(a.z, Δt_stage)

function set_advection_timestep!(a::NamedTuple, Δt_stage)
    for scheme in values(a)
        set_advection_timestep!(scheme, Δt_stage)
    end
    return nothing
end

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

Update non-acoustic scalar fields (moisture, tracers) using the given
kernel. Iterates over prognostic fields, skipping the first 5
(``ρ, ρu, ρv, ρw, ρθ``) which are handled by the acoustic substep loop.
"""
function scalar_substep!(model, kernel!, Δt_implicit, kernel_args...)
    grid = model.grid
    arch = grid.architecture
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ
    prognostic = prognostic_fields(model)
    names = keys(prognostic)
    n_acoustic = 5  # ρ, ρu, ρv, ρw, ρθ are advanced inside the substep loop

    # Water species and tracers advect as mass fractions of the total density ρ = ρᵈ + Σρˣ
    # (see `scalar_tendency`), so the implicit solve is weighted with the same density. The
    # velocities are the time-averaged transport velocities that the explicit scalar tendencies
    # (Gⁿ) were built with — adaptive implicit vertical advection must use the same `w` so the
    # explicit/implicit velocity split is consistent.
    ρ = total_density(model.dynamics)
    velocities = transport_velocities(model)

    for (i, (u, u⁰, G)) in enumerate(zip(prognostic, U⁰, Gⁿ))
        i <= n_acoustic && continue

        launch!(arch, grid, :xyz, kernel!, u, u⁰, G, kernel_args...)

        field_index = Val(i - n_acoustic)
        advection = field_advection_scheme(model.advection, names[i])

        if needs_implicit_solver(advection)
            implicit_step!(u,
                           model.timestepper.implicit_solver,
                           model.closure,
                           model.closure_fields,
                           field_index,
                           model.clock,
                           fields(model),
                           Δt_implicit,
                           advection,
                           velocities,
                           ρ)
        else
            implicit_step!(u,
                           model.timestepper.implicit_solver,
                           model.closure,
                           model.closure_fields,
                           field_index,
                           model.clock,
                           fields(model),
                           Δt_implicit)
        end
    end

    return nothing
end

#####
##### Implicit vertical solve for the acoustic prognostics
#####

# The vertical momentum whose divergence drives the continuity equation: the prognostic
# ρw on height grids, the contravariant ρw̃ on terrain-following grids (matching the
# density-tendency kernels).
continuity_vertical_momentum(model) = model.momentum.ρw
continuity_vertical_momentum(model::TerrainCompressibleAcousticModel) =
    model.dynamics.contravariant_vertical_momentum

@inline function residual_vertical_mass_flux(i, j, k, grid, scheme, td, W, ρw)
    s = explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, td, W)
    return @inbounds Azᶜᶜᶠ(i, j, k, grid) * (1 - s) * ρw[i, j, k]
end

@kernel function _remove_residual_vertical_mass_flux!(Gρ, grid, scheme, td, W, ρw)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρ[i, j, k] += V⁻¹ᶜᶜᶜ(i, j, k, grid) *
        δzᵃᵃᶜ(i, j, k, grid, residual_vertical_mass_flux, scheme, td, W, ρw)
end

"""
$(TYPEDSIGNATURES)

Partition the continuity equation like the thermodynamic variable: remove the CFL-withheld
residual of the vertical mass-flux divergence from the slow density tendency (the in-loop
implicit solve applies it over each acoustic substep instead). Without this, a stage-frozen
full-strength vertical mass flux drains terrain wall cells faster than the acoustic
adjustment responds once α = wΔt/Δz exceeds one, and the dry density goes negative even
when every advected prognostic is treated implicitly. A no-op unless the thermodynamic
scheme is adaptive-implicit.
"""
split_slow_continuity_tendency!(model) =
    split_slow_continuity_tendency!(model, vertical_scheme(field_advection_scheme(model.advection,
                                    thermodynamic_density_name(model.formulation))))

split_slow_continuity_tendency!(model, scheme) = nothing

function split_slow_continuity_tendency!(model, scheme::AdaptiveImplicitVerticalAdvection)
    grid = model.grid
    td = OceananigansTimeSteppers.time_discretization(scheme)
    W = slow_thermodynamic_velocities(model).w
    ρw = continuity_vertical_momentum(model)
    launch!(architecture(grid), grid, :xyz, _remove_residual_vertical_mass_flux!,
            model.timestepper.Gⁿ.ρᵈ, grid, scheme, td, W, ρw)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Build the per-substep adaptive-implicit vertical-advection applicator for the acoustic
substep loop, or return `nothing` when no advection scheme is adaptive-implicit. The
returned closure applies the CFL-withheld remainder of vertical advection to each AIVA
prognostic over one acoustic substep Δτ, operating on the reconstructed full field
(stage-start base `U⁰` plus the loop's perturbation) so the transport acts on the full
stratification. Solving inside the loop lets the acoustic pressure adjust to the residual
transport substep by substep; the once-per-stage post-loop placement is unstable at
terrain walls (issue #897).
"""
function in_loop_implicit_advection(model)
    solver = model.timestepper.implicit_solver
    solver === nothing && return nothing

    momentum_advection = model.advection.momentum
    θ_name = thermodynamic_density_name(model.formulation)
    θ_advection = field_advection_scheme(model.advection, θ_name)
    needs_implicit_solver(momentum_advection) || needs_implicit_solver(θ_advection) || return nothing

    substepper = model.timestepper.substepper
    U⁰ = model.timestepper.U⁰
    ρᵈ = dynamics_density(model.dynamics)
    w = advecting_vertical_velocity(model.dynamics, model.velocities)
    θ_velocities = slow_thermodynamic_velocities(model)
    perturbations = (; ρu = substepper.momentum_perturbation.u,
                       ρv = substepper.momentum_perturbation.v,
                       ρw = substepper.momentum_perturbation.w,
                       NamedTuple{(θ_name,)}((substepper.density_potential_temperature_perturbation,))...)

    function implicit_advection!(Δτ)
        for name in (:ρu, :ρv, :ρw)
            needs_implicit_solver(momentum_advection) || continue
            implicit_advection_substep!(perturbations[name], U⁰[name], solver, model, Δτ,
                                        implicit_step_advection(momentum_advection, name), (; w), ρᵈ)
        end
        if needs_implicit_solver(θ_advection)
            implicit_advection_substep!(perturbations[θ_name], U⁰[θ_name], solver, model, Δτ,
                                        θ_advection, θ_velocities, ρᵈ)
            # Continuity carries the matching residual (see `split_slow_continuity_tendency!`);
            # `density = nothing` advects the density field itself.
            implicit_advection_substep!(substepper.density_perturbation, U⁰.ρᵈ, solver, model, Δτ,
                                        θ_advection, θ_velocities, nothing)
        end
        return nothing
    end

    return implicit_advection!
end

# One field's per-substep residual-advection solve: reconstruct the full field from the
# stage base and the perturbation, solve (I - Δτ Lⁱ) in place, restore the perturbation.
function implicit_advection_substep!(perturbation, base, solver, model, Δτ, advection, velocities, ρ)
    parent(perturbation) .+= parent(base)
    implicit_step!(perturbation, solver, nothing, nothing, nothing,
                   model.clock, fields(model), Δτ, advection, velocities, ρ)
    parent(perturbation) .-= parent(base)
    fill_halo_regions!(perturbation)
    return nothing
end

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
    ρᵈ = dynamics_density(model.dynamics)
    prognostic = prognostic_fields(model)

    # Momentum advects with the (possibly contravariant) advecting vertical velocity — the
    # same velocity the slow momentum flux divergence splits.
    w = advecting_vertical_velocity(model.dynamics, model.velocities)
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
                       nothing,   # AIVA advection is applied inside the substep loop
                       (; w),
                       ρᵈ)
    end

    θ_name = thermodynamic_density_name(model.formulation)
    θ_advection = field_advection_scheme(model.advection, θ_name)
    implicit_step!(prognostic[θ_name],
                   implicit_solver,
                   model.closure,
                   model.closure_fields,
                   Val(1),   # the thermodynamic variable leads the closure's scalar names (see `with_tracers`)
                   model.clock,
                   fields(model),
                   Δt_stage,
                   nothing,   # AIVA advection is applied inside the substep loop
                   slow_thermodynamic_velocities(model),
                   ρᵈ)

    return nothing
end
