#####
##### Dynamics Interface
#####
##### This file defines the interface that all dynamics implementations must provide.
##### These functions are called by the AtmosphereModel constructor and
##### must be extended by specific dynamics implementations.
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
    materialize_dynamics(dynamics_stub, grid, boundary_conditions)

Materialize a dynamics stub into a complete dynamics object with all required fields.
"""
function materialize_dynamics end

"""
    materialize_momentum_and_velocities(dynamics, grid, boundary_conditions)

Create momentum and velocity fields for the given dynamics.
"""
function materialize_momentum_and_velocities end

"""
    dynamics_pressure_solver(dynamics, grid)

Create the pressure solver for the given dynamics.
Returns `nothing` for dynamics that do not require a pressure solver (e.g., compressible).
"""
function dynamics_pressure_solver end

#####
##### Pressure interface
#####

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

Return the density field appropriate to the dynamical formulation.

For anelastic dynamics, returns the reference density (time-independent background state).
For compressible dynamics, returns the prognostic density field.
"""
function dynamics_density end

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
    buoyancy_forceᶜᶜᶜ(i, j, k, grid, dynamics, formulation, temperature,
                      specific_moisture, microphysics, microphysical_fields, constants)

Compute the buoyancy force density `ρ b` at cell center `(i, j, k)`.

For anelastic dynamics, this returns `-g (ρ - ρᵣ)` where `ρᵣ` is the reference density.
For compressible dynamics, this returns `-g ρ` directly.

This function is used in the vertical momentum equation to compute the gravitational
forcing term.
"""
function buoyancy_forceᶜᶜᶜ end

#####
##### Boundary condition interface
#####

"""
    dynamics_surface_pressure(dynamics)

Return the surface pressure used for boundary condition regularization.
For anelastic dynamics, this is the reference state surface pressure.
For compressible dynamics, this may be a constant or computed value.
"""
function dynamics_surface_pressure end

"""
    standard_pressure(dynamics)

Return the standard pressure used for potential temperature calculations.
Default is 100000 Pa (1000 hPa).
"""
standard_pressure(dynamics) = 100000.0  # Pa

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
    prognostic_dynamics_field_names(dynamics)

Return a tuple of prognostic field names specific to the dynamics formulation.

For anelastic dynamics, returns an empty tuple (no prognostic density).
For compressible dynamics, returns `(:ρ,)` for prognostic density.
"""
prognostic_dynamics_field_names(::Any) = ()

"""
    additional_dynamics_field_names(dynamics)

Return a tuple of additional (diagnostic) field names for the dynamics.
"""
additional_dynamics_field_names(::Any) = ()

"""
    dynamics_prognostic_fields(dynamics)

Return a NamedTuple of prognostic fields specific to the dynamics formulation.

For anelastic dynamics, returns an empty NamedTuple.
For compressible dynamics, returns `(ρ=density_field,)`.
"""
dynamics_prognostic_fields(dynamics) = NamedTuple()

