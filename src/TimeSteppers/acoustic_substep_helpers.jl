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
    specific_prognostic_moisture,
    x_pressure_gradient,
    y_pressure_gradient

#####
##### Stage-frozen tendency computation
#####

"""
$(TYPEDSIGNATURES)

Compute slow momentum tendencies (advection, Coriolis, turbulence, forcing).

The pressure gradient and buoyancy are excluded using [`SlowTendencyMode`](@ref).
These "fast" terms are handled by the acoustic substep loop, which resolves
the acoustic CFL through substepping with constant ``Δτ = Δt/N``.

The horizontal pressure gradient is frozen at outer-step start and added
separately by [`AcousticRungeKutta3.acoustic_rk3_substep!`](@ref) via the
`slow_tendency_snapshot.ρu`/`.ρv` buffers, matching MPAS's `tend_u_euler`
which is computed at `rk_step == 1` and frozen across the substeps.
"""
function compute_slow_momentum_tendencies!(model)
    grid = model.grid
    arch = architecture(grid)

    # Wrap dynamics in SlowTendencyMode: zero pressure gradient and buoyancy.
    # The vertical PGF/buoyancy are handled by the substepper's column kernel.
    # The horizontal PGF is added by the caller from a frozen-at-stage-1 snapshot.
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

"""
$(TYPEDSIGNATURES)

Compute the horizontal pressure gradient force `(-∂p/∂x, -∂p/∂y)` and write
it into the snapshot buffers `pgf_u`, `pgf_v`. This is called once per outer
step (at WS-RK3 stage 1) so that the slow horizontal PGF can be added to the
slow momentum tendencies at every stage with a frozen value, matching MPAS's
`tend_u_euler` which is computed at `rk_step == 1` and frozen for the rest of
the outer step.
"""
function snapshot_horizontal_pgf!(pgf_u, pgf_v, model)
    grid = model.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _compute_horizontal_pgf!, pgf_u, pgf_v, grid, model.dynamics)
    return nothing
end

@kernel function _compute_horizontal_pgf!(pgf_u, pgf_v, grid, dynamics)
    i, j, k = @index(Global, NTuple)
    @inbounds pgf_u[i, j, k] = -x_pressure_gradient(i, j, k, grid, dynamics)
    @inbounds pgf_v[i, j, k] = -y_pressure_gradient(i, j, k, grid, dynamics)
end

"""
$(TYPEDSIGNATURES)

Add the snapshotted horizontal PGF (negative of `∂p/∂x`, `∂p/∂y`) to the slow
momentum tendencies `Gρu`, `Gρv`. The buffers `pgf_u`, `pgf_v` should have been
populated by `snapshot_horizontal_pgf!` at WS-RK3 stage 1.
"""
function add_horizontal_pgf!(Gρu, Gρv, pgf_u, pgf_v, model)
    grid = model.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _add_horizontal_pgf!, Gρu, Gρv, pgf_u, pgf_v)
    return nothing
end

@kernel function _add_horizontal_pgf!(Gρu, Gρv, pgf_u, pgf_v)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρu[i, j, k] += pgf_u[i, j, k]
    @inbounds Gρv[i, j, k] += pgf_v[i, j, k]
end

#####
##### Slow-tendency freeze policy
#####
##### MPAS computes `tend_rho` and `tend_u_euler` (the horizontal PGF) at
##### `rk_step == 1` and freezes them across the remaining stages. MPAS's
##### `tend_theta` is computed every stage in perturbation form
##### `(rw × θ + (rw_save - rw) × θ_save)` and additionally picks up physics
##### tendencies (microphysics latent heat, radiation) every stage. We mimic
##### this by:
#####
#####   1. snapshotting `Gⁿ.ρ` and the horizontal PGF at stage 1, and
#####   2. restoring `Gⁿ.ρ` at stages 2 and 3 after the per-stage
#####      `compute_slow_*_tendencies!` calls.
#####
##### **`Gⁿ.ρθ` is NOT snapshotted.** It is recomputed every stage by
##### `compute_thermodynamic_tendency!`, which is essential for moist runs:
##### the microphysics latent heat in `scalar_tendency` is a strong function
##### of the (per-stage) state, and freezing it at stage 1 causes condensate
##### sources to drift wildly out of step with the thermodynamic state and
##### blow up within a few outer steps. The dry-case cost is small: the
##### advective contribution to `Gⁿ.ρθ` only changes by O(Δt × Gρθ̇), which is
##### well within the WS-RK3 truncation envelope.
#####
##### `Gⁿ.ρu` / `Gⁿ.ρv` are recomputed every stage with `SlowTendencyMode`
##### (which zeroes the PGF), and the frozen horizontal PGF snapshot is added
##### back on top via `add_horizontal_pgf!`. `Gⁿ.ρw` is not snapshotted: its
##### `tend_w_euler` is frozen by the substepper's `convert_slow_tendencies!`
##### kernel, and the per-stage recomputation of `w` advection is needed for
##### accuracy. Tracer entries (beyond ρ, ρu, ρv, ρw, ρθ) are not touched, so
##### moisture/scalar transport picks up per-stage updates.

"""
$(TYPEDSIGNATURES)

Snapshot the slow tendencies that the WS-RK3 freeze policy holds constant
across stages: `Gⁿ.ρ` and the horizontal pressure gradient force. Called
once per outer step at WS-RK3 stage 1. `Gⁿ.ρθ` is intentionally not
snapshotted — see the freeze-policy comment above for the moist-physics
rationale.
"""
function snapshot_slow_tendencies!(snap, model)
    Gⁿ = model.timestepper.Gⁿ
    parent(snap.ρ) .= parent(Gⁿ.ρ)
    snapshot_horizontal_pgf!(snap.ρu, snap.ρv, model)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Restore the snapshotted scalar slow tendency (`Gⁿ.ρ`) onto the model
timestepper. Called at WS-RK3 stages 2 and 3 after
`compute_slow_scalar_tendencies!` so the substep loop sees the same density
tendency as stage 1. `Gⁿ.ρθ` is *not* restored — the per-stage
recomputation is what makes microphysics latent heat work.
"""
function restore_slow_tendencies!(snap, model)
    Gⁿ = model.timestepper.Gⁿ
    parent(Gⁿ.ρ) .= parent(snap.ρ)
    return nothing
end

#####
##### Slow density and thermodynamic tendencies
#####

"""
$(TYPEDSIGNATURES)

Compute slow tendencies for density and thermodynamic variable.

In the perturbation-variable approach, the slow tendencies are simply the full
RHS ``R^t`` evaluated at the stage-level state. No correction is needed because
the acoustic loop advances perturbation variables, not full fields.

- ``G^s_ρ = -\\boldsymbol{∇·m}^t``: full density tendency (continuity equation)
- ``G^s_χ``: full thermodynamic tendency (advection + physics)
"""
function compute_slow_scalar_tendencies!(model)
    # Compute Gˢρ = -∇·m^t (full density tendency at stage start)
    # Writes directly to model.timestepper.Gⁿ.ρ
    compute_dynamics_tendency!(model)

    # Compute Gˢχ = full thermodynamic tendency (no correction needed)
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
##### Scalar update with time-averaged velocities
#####

"""
$(TYPEDSIGNATURES)

Update non-acoustic scalar fields (moisture, tracers) using the given kernel.

Iterates over prognostic fields, skipping the first 5 (ρ, ρu, ρv, ρw, ρθ)
which are handled by the acoustic substep loop. For each remaining field,
launches `kernel!` with the provided `kernel_args` and applies the implicit
diffusion step.
"""
function acoustic_scalar_substep!(model, kernel!, Δt_implicit, kernel_args...)
    grid = model.grid
    arch = grid.architecture
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ
    prognostic = prognostic_fields(model)
    n_acoustic = 5  # ρ, ρu, ρv, ρw, ρθ (handled by acoustic loop)

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
