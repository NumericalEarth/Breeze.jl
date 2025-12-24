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
    materialize_formulation(formulation, dynamics, grid, boundary_conditions)

Materialize a thermodynamic formulation from a Symbol (or formulation struct) into a 
complete formulation with all required fields.

Valid symbols:
- `:LiquidIcePotentialTemperature`, `:θ`, `:ρθ`, `:PotentialTemperature` → `LiquidIcePotentialTemperatureFormulation`
- `:StaticEnergy`, `:e`, `:ρe` → `StaticEnergyFormulation`
"""
materialize_formulation(formulation_name::Symbol, args...) =
    materialize_formulation(Val(formulation_name), args...)

#####
##### Field naming interface
#####

"""
    prognostic_thermodynamic_field_names(formulation)

Return a tuple of prognostic field names for the given thermodynamic formulation.
Accepts a Symbol, Val(Symbol), or formulation struct.
"""
prognostic_thermodynamic_field_names(formulation_name::Symbol) =
    prognostic_thermodynamic_field_names(Val(formulation_name))

"""
    additional_thermodynamic_field_names(formulation)

Return a tuple of additional (diagnostic) field names for the given thermodynamic formulation.
Accepts a Symbol, Val(Symbol), or formulation struct.
"""
additional_thermodynamic_field_names(formulation_name::Symbol) =
    additional_thermodynamic_field_names(Val(formulation_name))

"""
    thermodynamic_density_name(formulation)

Return the name of the thermodynamic density field (e.g., `:ρθ`, `:ρe`, `:ρE`).
Accepts a Symbol, Val(Symbol), or formulation struct.
"""
thermodynamic_density_name(formulation::Symbol) =
    thermodynamic_density_name(Val(formulation))

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

