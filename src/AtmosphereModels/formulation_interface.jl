#####
##### Formulation Interface
#####
##### This file defines the interface that all formulations must implement.
##### These functions are called by the AtmosphereModel constructor and
##### must be extended by specific formulation implementations.
#####

#####
##### Construction interface
#####

"""
    default_formulation(grid, constants)

Return the default formulation for the given grid and thermodynamic constants.
"""
function default_formulation end

"""
    materialize_formulation(stub, grid, boundary_conditions)

Materialize a formulation stub into a complete formulation with all required fields.
"""
function materialize_formulation end

"""
    materialize_thermodynamics(thermodynamics_type, grid, boundary_conditions)

Create thermodynamic fields for the given thermodynamics type (e.g., `Val(:StaticEnergy)`).
"""
function materialize_thermodynamics end

"""
    materialize_momentum_and_velocities(formulation, grid, boundary_conditions)

Create momentum and velocity fields for the given formulation.
"""
function materialize_momentum_and_velocities end

"""
    formulation_pressure_solver(formulation, grid)

Create the pressure solver for the given formulation.
"""
function formulation_pressure_solver end

"""
    prognostic_field_names(formulation)

Return a tuple of prognostic field names for the given formulation.
"""
function prognostic_field_names end

"""
    additional_field_names(formulation)

Return a tuple of additional (diagnostic) field names for the given formulation.
"""
function additional_field_names end

#####
##### Pressure interface
#####

"""
    mean_pressure(formulation)

Return the mean (background/reference) pressure field in Pa.
"""
function mean_pressure end

"""
    pressure_anomaly(formulation)

Return the pressure anomaly (deviation from mean) in Pa.
"""
function pressure_anomaly end

"""
    total_pressure(formulation)

Return the total pressure (mean + anomaly) in Pa.
"""
function total_pressure end

#####
##### Density interface
#####

"""
    formulation_density(formulation)

Return the density field appropriate to the dynamical formulation.

For anelastic formulations, returns the reference density (time-independent background state).
For compressible formulations, returns the prognostic density field.
"""
function formulation_density end
