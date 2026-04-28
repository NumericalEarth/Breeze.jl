using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields, architecture
using Oceananigans.Utils: launch!

using Oceananigans.TimeSteppers: implicit_step!

using Breeze.AtmosphereModels:
    AtmosphereModels,
    SlowTendencyMode,
    dynamics_density,
    compute_x_momentum_tendency!,
    compute_y_momentum_tendency!,
    compute_z_momentum_tendency!,
    compute_dynamics_tendency!,
    specific_prognostic_moisture

#####
##### Slow momentum tendencies
#####
##### `SlowTendencyMode` zeros the pressure-gradient force and buoyancy in the
##### momentum tendency assembly. The PGF and buoyancy are handled in
##### linearized form inside the substep loop, so the slow tendency carries
##### only advection, Coriolis, closure, and forcing.
#####

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
        model.momentum,
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

Compute slow tendencies for density and the thermodynamic variable:

  - ``Gˢ_ρ = -∇·m``: full density tendency (continuity equation),
    written into `model.timestepper.Gⁿ.ρ`.
  - ``Gˢ_χ``: full thermodynamic tendency (advection + physics).
"""
function compute_slow_scalar_tendencies!(model)
    compute_dynamics_tendency!(model)

    common_args = (
        model.dynamics,
        model.formulation,
        model.thermodynamic_constants,
        specific_prognostic_moisture(model),
        model.velocities,
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
function acoustic_scalar_substep!(model, kernel!, Δt_implicit, kernel_args...)
    grid = model.grid
    arch = grid.architecture
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ
    prognostic = prognostic_fields(model)
    n_acoustic = 5  # ρ, ρu, ρv, ρw, ρθ are advanced inside the substep loop

    for (i, (u, u⁰, G)) in enumerate(zip(prognostic, U⁰, Gⁿ))
        i <= n_acoustic && continue

        launch!(arch, grid, :xyz, kernel!, u, u⁰, G, kernel_args...)

        field_index = Val(i - n_acoustic)
        implicit_step!(u,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       field_index,
                       model.clock,
                       fields(model),
                       Δt_implicit)
    end

    return nothing
end
