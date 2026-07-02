using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields, architecture
using Oceananigans.Advection: needs_implicit_solver
using Oceananigans.Utils: launch!

using Oceananigans.TimeSteppers: implicit_step!

using Breeze.AtmosphereModels:
    AtmosphereModels,
    AtmosphereModel,
    SlowTendencyMode,
    advecting_momentum,
    dynamics_density,
    transport_velocities,
    field_advection_scheme,
    breeze_implicit_step!,
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

    # Reference/prognostic density and the time-averaged transport velocity that the explicit
    # scalar tendencies (Gⁿ) were built with — adaptive implicit vertical advection must use the
    # same `w` so the explicit/implicit velocity split is consistent.
    ρ = dynamics_density(model.dynamics)
    velocities = transport_velocities(model)

    for (i, (u, u⁰, G)) in enumerate(zip(prognostic, U⁰, Gⁿ))
        i <= n_acoustic && continue

        launch!(arch, grid, :xyz, kernel!, u, u⁰, G, kernel_args...)

        field_index = Val(i - n_acoustic)
        advection = field_advection_scheme(model.advection, names[i])

        if needs_implicit_solver(advection)
            breeze_implicit_step!(u,
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
