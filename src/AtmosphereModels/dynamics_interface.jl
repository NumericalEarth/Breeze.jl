#####
##### Dynamics Interface
#####
##### This file defines the interface that all dynamics implementations must provide.
##### These functions are called by the AtmosphereModel constructor and
##### must be extended by specific dynamics implementations (AnelasticEquations, CompressibleEquations).
#####

#####
##### Construction interface
#####

"""
    default_dynamics(grid, constants)

Return the default dynamics for the given grid and thermodynamic constants.
"""
function default_dynamics end

"""
    materialize_dynamics(dynamics_stub, grid, boundary_conditions, thermodynamic_constants, microphysics=nothing)

Materialize a dynamics stub into a complete dynamics object with all required fields.

The `microphysics` argument is optional and used by dynamics types that need to know
the microphysics scheme to create appropriate prognostic state (e.g., `ParcelDynamics`).
"""
function materialize_dynamics end

# Default: ignore microphysics argument (for backward compatibility)
materialize_dynamics(d, grid, bcs, constants, microphysics) = materialize_dynamics(d, grid, bcs, constants)

"""
    materialize_momentum_and_velocities(dynamics, grid, boundary_conditions)

Create momentum and velocity fields for the given dynamics.
"""
function materialize_momentum_and_velocities end

"""
    materialize_velocities(velocities, grid)

Create velocity fields from a velocity specification (e.g., `PrescribedVelocityFields`).
"""
function materialize_velocities end

"""
    update_dynamics_with_velocities(dynamics, velocities)

Update dynamics with velocity specification. Default is a no-op.
For `PrescribedDynamics`, stores the `PrescribedVelocityFields` for dispatch.
"""
update_dynamics_with_velocities(dynamics, velocities) = dynamics

"""
    dynamics_pressure_solver(dynamics, grid)

Create the pressure solver for the given dynamics.
Returns `nothing` for dynamics that do not require a pressure solver (e.g., compressible).
"""
function dynamics_pressure_solver end

"""
$(TYPEDSIGNATURES)

Return the default timestepper symbol for the given dynamics.

For anelastic dynamics or compressible dynamics with explicit timestepping, returns `:SSPRungeKutta3`.
For compressible dynamics with acoustic substepping, returns `:AcousticRungeKutta3`.
"""
default_timestepper(dynamics) = :SSPRungeKutta3

#####
##### Pressure correction interface
#####

"""
$(TYPEDSIGNATURES)

Compute the pressure correction for the given model. Default: no-op.
For anelastic dynamics, solves the pressure Poisson equation.
"""
compute_pressure_correction!(model, Δt) = nothing

"""
$(TYPEDSIGNATURES)

Apply the pressure correction to the momentum fields. Default: no-op.
For anelastic dynamics, projects momentum to enforce the divergence constraint.
"""
make_pressure_correction!(model, Δt) = nothing

#####
##### Pressure interface
#####

"""
    reference_pressure(dynamics)

Total thermodynamic pressure (Pa) to feed radiation. A compressible atmosphere carries the
true total (its diagnosed pressure); an anelastic atmosphere has no thermodynamic pressure
other than its reference profile, so that is its total. NOT the dynamics' pressure-gradient
reference state — that must never reach radiation.
"""
reference_pressure(dynamics) = dynamics_pressure(dynamics)

"""
    reference_density(dynamics)

Companion to [`reference_pressure`](@ref): the total (moist) air density to feed radiation.
Delegates to [`total_density`](@ref) — never [`dynamics_density`](@ref), which is the coupling
density (the dry-air density on the compressible core) and would understate the absorber and
condensate mass seen by radiation.
"""
reference_density(dynamics) = total_density(dynamics)

"""
    mean_pressure(dynamics)

Return the mean (background/reference) pressure field in Pa.
"""
function mean_pressure end

"""
    pressure_anomaly(dynamics)

Return the pressure anomaly (deviation from mean) in Pa.
"""
function pressure_anomaly end

"""
    total_pressure(dynamics)

Return the total pressure (mean + anomaly) in Pa.
"""
function total_pressure end

#####
##### Density and pressure access interface
#####

"""
    dynamics_density(dynamics)

Return the *coupling density* — the density weighting the momentum (`ρu = ρᵈ u`) and the
thermodynamic flux variable (`ρθ = ρᵈ θ`), and the divisor for diagnosing velocity (`u = ρu/ρᵈ`)
and potential temperature (`θ = ρθ/ρᵈ`). It is the prognostic mass variable advanced by continuity.

- `AnelasticDynamics`: the time-independent reference density `ρᵣ`.
- `CompressibleDynamics`: the prognostic dry-air density `ρᵈ`.

The *total* air density `ρ = ρᵈ + Σ ρˣ` (dry air plus every water species) is a separate, diagnosed
quantity — see [`total_density`](@ref) — used wherever total mass enters the physics: the
moisture **mass-fraction** recovery (`qˣ = ρˣ/ρ`, so the thermodynamics stays in mass fractions),
scalar and water advection, the equation of state, and buoyancy. The water densities (`ρqᵛ`, `ρqˡ`,
…) are stored as partial densities (mass per volume), *not* coupling-weighted. On the anelastic core
the two densities coincide (`total_density === dynamics_density`).
"""
function dynamics_density end

"""
    total_density(dynamics)

Return the total air density ρ = ρᵈ + Σρˣ used by the thermodynamics, scalar advection,
equation of state, and buoyancy. Defaults to [`dynamics_density`](@ref) — correct for
formulations with a single density (e.g. the anelastic reference density). `CompressibleDynamics`
overrides it with a diagnosed total-density field, distinct from the coupling density ρᵈ.
"""
total_density(dynamics) = dynamics_density(dynamics)

"""
    advecting_vertical_velocity(dynamics, velocities)

Return the vertical velocity that advects momentum through the grid's coordinate surfaces:
the Cartesian `velocities.w` on height-coordinate grids, and the contravariant vertical
velocity `w̃` on terrain-following grids (mirroring [`advecting_momentum`](@ref), whose
vertical component is the contravariant momentum). The adaptive-implicit vertical-advection
split must partition this velocity on both the explicit (flux-scaling) and implicit
(tridiagonal) sides, so it stays consistent with the momentum flux divergence.
"""
@inline advecting_vertical_velocity(dynamics, velocities) = velocities.w

"""
    dynamics_pressure(dynamics)

Return the pressure field appropriate to the dynamical formulation.

For anelastic dynamics, returns the reference pressure (hydrostatic background state).
For compressible dynamics, returns the prognostic pressure field.
"""
function dynamics_pressure end

#####
##### Buoyancy interface
#####

"""
    buoyancy_forceᶜᶜᶜ(i, j, k, grid, dynamics, temperature,
                      specific_prognostic_moisture, microphysics, microphysical_fields, constants)

Compute the buoyancy force density ``ρ b`` at cell center `(i, j, k)`.

This function is used in the vertical momentum equation to compute the gravitational
forcing term.

"""
function buoyancy_forceᶜᶜᶜ end

#####
##### Boundary condition interface
#####

"""
    validate_velocity_boundary_conditions(dynamics, user_boundary_conditions)

Validate that velocity boundary conditions are only provided for dynamics that support them.

By default, throws an error if the user provides boundary conditions for `:u`, `:v`, or `:w`,
since velocity is a diagnostic field for most dynamics (e.g., anelastic, compressible).

For `PrescribedDynamics`, velocity boundary conditions are allowed since velocities are
regular fields that can be set directly.
"""
function validate_velocity_boundary_conditions(dynamics, user_boundary_conditions)
    velocity_names = (:u, :v, :w)
    user_bc_names = keys(user_boundary_conditions)
    provided_velocity_bcs = filter(name -> name ∈ user_bc_names, velocity_names)

    if !isempty(provided_velocity_bcs)
        throw(ArgumentError(
            "Boundary conditions for velocity components $(provided_velocity_bcs) are not supported " *
            "for $(summary(dynamics)). Velocity boundary conditions are only valid for PrescribedDynamics " *
            "(kinematic models) where velocities are regular fields. For prognostic dynamics, " *
            "set boundary conditions on momentum fields (:ρu, :ρv, :ρw) instead."
        ))
    end
    return nothing
end

"""
    surface_pressure(dynamics)

Return the surface pressure used for boundary condition regularization.
For anelastic dynamics, this is the reference state surface pressure.
For compressible dynamics, this may be a constant or computed value.
"""
function surface_pressure end

"""
    boundary_conditions_reference_state(dynamics, grid, thermodynamic_constants)

Return a reference state with `pressure`, `density`, and `standard_pressure` fields
suitable for constructing boundary-condition diagnostics (e.g. virtual potential
temperature for stability-dependent bulk fluxes).

Boundary conditions are materialized before `materialize_dynamics` runs, so this
hook lets each dynamics type decide what to expose at that point. The default
returns `dynamics.reference_state`, which works for dynamics where the user
constructs a fully-built reference state up front (e.g. `AnelasticDynamics`).
"""
boundary_conditions_reference_state(dynamics, grid, thermodynamic_constants) = dynamics.reference_state

"""
    dynamics_reference_state(dynamics)

Return the dynamics' reference state (an anelastic `ReferenceState` or a split-explicit
`ExnerReferenceState`), or `nothing` if the dynamics carries none. Dispatched so that
`reset_reference_state!` needn't reach into fields by name.
"""
dynamics_reference_state(dynamics) = nothing

"""
    standard_pressure(dynamics)

Return the standard pressure used for potential temperature calculations.
Default is 100000 Pa (1000 hPa).
"""
function standard_pressure end

"""
    default_drag_surface_temperature(dynamics, grid, thermodynamic_constants)

Return a default surface temperature for `BulkDrag` when the user has not supplied
one. Dispatched on the dynamics type because the notion of a "default" surface
temperature depends on what reference structure the dynamics carries: anelastic
has a full reference profile whose surface value is well-defined; compressible
has no equivalent up-front surface temperature and therefore requires the user
to provide one explicitly.

The default (no method) throws an informative error. Each dynamics type extends
this hook.
"""
default_drag_surface_temperature(dynamics, grid, thermodynamic_constants) =
    throw(ArgumentError(
        "BulkDrag was constructed without `surface_temperature`, and no default is " *
        "defined for dynamics of type `$(typeof(dynamics))`. Please pass " *
        "`surface_temperature` explicitly when constructing `BulkDrag`."))

"""
    initialize_model_thermodynamics!(model)

Initialize the thermodynamic state for a newly constructed model.
For anelastic dynamics, sets initial θ to the reference potential temperature.
For compressible dynamics, no default initialization is performed.
"""
initialize_model_thermodynamics!(model) = nothing  # default: do nothing

#####
##### Prognostic fields interface
#####

"""
    prognostic_momentum_field_names(dynamics)

Return a tuple of prognostic momentum field names.

For prognostic dynamics (anelastic, compressible), returns `(:ρu, :ρv, :ρw)`.
For kinematic dynamics (prescribed velocities), returns an empty tuple.
"""
prognostic_momentum_field_names(::Any) = (:ρu, :ρv, :ρw)

"""
    prognostic_dynamics_field_names(dynamics)

Return a tuple of prognostic field names specific to the dynamics formulation.

For anelastic dynamics, returns an empty tuple (no prognostic density).
For compressible dynamics, returns `(:ρᵈ,)` for prognostic density.
"""
prognostic_dynamics_field_names(::Any) = ()

"""
    additional_dynamics_field_names(dynamics)

Return a tuple of additional (diagnostic) field names for the dynamics.
"""
additional_dynamics_field_names(::Any) = ()

"""
    velocity_boundary_condition_names(dynamics)

Return a tuple of velocity field names that need default boundary conditions.

For most dynamics (anelastic, compressible), velocities are diagnostic and their boundary
conditions are created internally. Returns an empty tuple.

For `PrescribedDynamics`, velocities are regular fields that can have user-provided
boundary conditions, so this returns `(:u, :v, :w)`.
"""
velocity_boundary_condition_names(::Any) = ()

"""
    dynamics_prognostic_fields(dynamics)

Return a NamedTuple of prognostic fields specific to the dynamics formulation.

For anelastic dynamics, returns an empty NamedTuple.
For compressible dynamics, returns `(ρ=density_field,)`.
"""
dynamics_prognostic_fields(dynamics) = NamedTuple()

#####
##### Pressure gradient interface
#####

"""
    x_pressure_gradient(i, j, k, grid, dynamics)

Return the x-component of the pressure gradient force at (Face, Center, Center).

For anelastic dynamics, returns zero (pressure is handled via projection).
For compressible dynamics, returns `-∂p/∂x`.
"""
@inline x_pressure_gradient(i, j, k, grid, dynamics) = zero(grid)

"""
    y_pressure_gradient(i, j, k, grid, dynamics)

Return the y-component of the pressure gradient force at (Center, Face, Center).

For anelastic dynamics, returns zero (pressure is handled via projection).
For compressible dynamics, returns `-∂p/∂y`.
"""
@inline y_pressure_gradient(i, j, k, grid, dynamics) = zero(grid)

"""
    z_pressure_gradient(i, j, k, grid, dynamics)

Return the z-component of the pressure gradient force at (Center, Center, Face).

For anelastic dynamics, returns zero (pressure is handled via projection).
For compressible dynamics, returns `-∂p/∂z`.
"""
@inline z_pressure_gradient(i, j, k, grid, dynamics) = zero(grid)

#####
##### Slow tendency mode for split-explicit time-stepping
#####

"""
$(TYPEDEF)

Wrapper type indicating that only "slow" tendencies should be computed.

When computing momentum tendencies with a `SlowTendencyMode`-wrapped dynamics,
the "fast" terms (pressure gradient and buoyancy) return zero. This is used
for split-explicit time-stepping where fast terms are handled separately
in an acoustic substep loop.

See also [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization).
"""
struct SlowTendencyMode{D}
    dynamics :: D
end

Adapt.adapt_structure(to, s::SlowTendencyMode) = SlowTendencyMode(adapt(to, s.dynamics))

# Forward dynamics_density to the wrapped dynamics
@inline dynamics_density(s::SlowTendencyMode) = dynamics_density(s.dynamics)

# Fast terms return zero in slow tendency mode
@inline x_pressure_gradient(i, j, k, grid, ::SlowTendencyMode) = zero(grid)
@inline y_pressure_gradient(i, j, k, grid, ::SlowTendencyMode) = zero(grid)
@inline z_pressure_gradient(i, j, k, grid, ::SlowTendencyMode) = zero(grid)

@inline buoyancy_forceᶜᶜᶜ(i, j, k, grid, ::SlowTendencyMode, args...) = zero(grid)

"""
$(TYPEDEF)

Wrapper type indicating that vertical "fast" terms should be excluded from tendencies.

When computing momentum tendencies with a `HorizontalSlowMode`-wrapped dynamics,
the horizontal pressure gradient is computed normally, but the vertical pressure
gradient and buoyancy return zero. These vertical fast terms are handled by the
acoustic substep loop through perturbation variables ``-ψ ∂ρ''/∂z - g ρ''``.

Including the full vertical PG and buoyancy in the slow tendency introduces a
hydrostatic truncation error ``O(Δz^2)`` that drives spurious acoustic modes.
The horizontal PG does not suffer from this issue and can safely be included.
"""
struct HorizontalSlowMode{D}
    dynamics :: D
end

Adapt.adapt_structure(to, s::HorizontalSlowMode) = HorizontalSlowMode(adapt(to, s.dynamics))

# Forward dynamics_density to the wrapped dynamics
@inline dynamics_density(s::HorizontalSlowMode) = dynamics_density(s.dynamics)

# Horizontal PG: forward to the wrapped dynamics
@inline x_pressure_gradient(i, j, k, grid, s::HorizontalSlowMode) =
    x_pressure_gradient(i, j, k, grid, s.dynamics)
@inline y_pressure_gradient(i, j, k, grid, s::HorizontalSlowMode) =
    y_pressure_gradient(i, j, k, grid, s.dynamics)

# Vertical PG and buoyancy return zero (handled by acoustic loop)
@inline z_pressure_gradient(i, j, k, grid, ::HorizontalSlowMode) = zero(grid)
@inline buoyancy_forceᶜᶜᶜ(i, j, k, grid, ::HorizontalSlowMode, args...) = zero(grid)

#####
##### Tendency computation interface
#####

"""
$(TYPEDSIGNATURES)

Compute tendencies for dynamics-specific prognostic fields.

For anelastic dynamics, this is a no-op (no prognostic density).
For compressible dynamics, this computes the density tendency from the continuity equation:

```math
\\partial_t \\rho = -\\boldsymbol{\\nabla \\cdot \\,} (\\rho \\boldsymbol{u})
```
"""
compute_dynamics_tendency!(model) = nothing  # default: no dynamics-specific tendencies

#####
##### Transport velocity/momentum interface (for terrain-following coordinates)
#####

"""
$(TYPEDSIGNATURES)

Return the velocity tuple used for scalar advection transport.

For standard (non-terrain) models, this is `model.velocities`.
For terrain-following coordinates, the vertical component is replaced
by the contravariant vertical velocity ``\\tilde{w}``.
"""
transport_velocities(model) = model.velocities

"""
$(TYPEDSIGNATURES)

Return the momentum tuple used for momentum advection transport
and the continuity equation divergence.

For standard (non-terrain) models, this is `model.momentum`.
For terrain-following coordinates, the vertical component `ρw` is
replaced by the contravariant vertical momentum ``\\rho \\tilde{w}``.
"""
advecting_momentum(model) = model.momentum

#####
##### Auxiliary dynamics variables interface
#####

"""
$(TYPEDSIGNATURES)

Compute auxiliary (diagnostic) variables specific to the dynamics formulation.

For anelastic dynamics, this is a no-op (pressure is computed during time-stepping
via the pressure Poisson equation).

For compressible dynamics, this computes the pressure field from the equation of state:

```math
p = ρ R^m T
```

where ``R^m`` is the mixture gas constant.
"""
compute_auxiliary_dynamics_variables!(model) = nothing  # default: no-op
