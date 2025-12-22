#####
##### Thermodynamic Formulation Interface
#####
##### This file defines the interface that all thermodynamic formulation implementations must provide.
##### These functions are called by the AtmosphereModel constructor and update_state! pipeline.
#####

#####
##### Construction interface
#####

"""
    default_thermodynamic_formulation()

Return the default thermodynamic formulation (StaticEnergyFormulation).
"""
function default_thermodynamic_formulation end

"""
    materialize_thermodynamic_formulation(formulation_stub, dynamics, grid, boundary_conditions)

Materialize a thermodynamic formulation stub into a complete formulation with all required fields.
"""
function materialize_thermodynamic_formulation end

#####
##### Field naming interface
#####

"""
    prognostic_thermodynamic_field_names(formulation)

Return a tuple of prognostic field names for the given thermodynamic formulation.
"""
function prognostic_thermodynamic_field_names end

"""
    additional_thermodynamic_field_names(formulation)

Return a tuple of additional (diagnostic) field names for the given thermodynamic formulation.
"""
function additional_thermodynamic_field_names end

"""
    thermodynamic_density_name(formulation)

Return the name of the thermodynamic density field (e.g., `:ρθ`, `:ρe`, `:ρE`).
"""
function thermodynamic_density_name end

"""
    thermodynamic_density(formulation)

Return the thermodynamic density field for the given formulation.
"""
function thermodynamic_density end

#####
##### Prognostic field collection
#####

"""
    collect_prognostic_fields(formulation, dynamics, momentum, moisture_density, microphysical_fields, tracers)

Collect all prognostic fields into a single NamedTuple.
"""
function collect_prognostic_fields end

#####
##### State computation interface
#####

"""
    compute_auxiliary_thermodynamic_variables!(formulation, dynamics, i, j, k, grid)

Compute auxiliary thermodynamic variables from prognostic fields at grid point (i, j, k).
"""
function compute_auxiliary_thermodynamic_variables! end

"""
    diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)

Diagnose the thermodynamic state at grid point `(i, j, k)` from the given `formulation`, 
`dynamics`, and pre-computed moisture mass fractions `q`.

Note: This function does NOT compute moisture fractions internally to avoid circular dependencies.
The caller is responsible for computing `q = compute_moisture_fractions(...)` before calling.
"""
function diagnose_thermodynamic_state end

#####
##### Tendency computation interface
#####

"""
    compute_thermodynamic_tendency!(model, common_args)

Compute the thermodynamic tendency. Dispatches on the thermodynamic formulation type.
"""
function compute_thermodynamic_tendency! end

