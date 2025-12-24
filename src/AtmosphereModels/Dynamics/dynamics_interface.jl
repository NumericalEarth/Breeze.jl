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

