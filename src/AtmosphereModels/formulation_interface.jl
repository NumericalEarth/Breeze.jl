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

Materialize a thermodynamic formulation from a `Symbol` (or formulation struct) into a
complete formulation with all required fields.

Valid symbols:
- `:LiquidIcePotentialTemperature`, `:θ`, `:ρθ`, `:PotentialTemperature` → `LiquidIcePotentialTemperatureFormulation`
- `:StaticEnergy`, `:e`, `:ρe` → `StaticEnergyFormulation`
"""
function materialize_formulation end

materialize_formulation(formulation_name::Symbol, args...) =
    materialize_formulation(Val(formulation_name), args...)

#####
##### Field naming interface
#####

"""
    prognostic_thermodynamic_field_names(formulation)

Return a tuple of prognostic field names for the given thermodynamic formulation.
Accepts a `Symbol`, `Val(Symbol)`, or formulation struct.
"""
function prognostic_thermodynamic_field_names end

prognostic_thermodynamic_field_names(formulation_name::Symbol) =
    prognostic_thermodynamic_field_names(Val(formulation_name))

"""
    additional_thermodynamic_field_names(formulation)

Return a tuple of additional (diagnostic) field names for the given thermodynamic formulation.
Accepts a `Symbol`, `Val(Symbol)`, or formulation struct.
"""
function additional_thermodynamic_field_names end

additional_thermodynamic_field_names(formulation_name::Symbol) =
    additional_thermodynamic_field_names(Val(formulation_name))

"""
    thermodynamic_density_name(formulation)

Return the name of the thermodynamic density field (e.g., `:ρθ`, `:ρe`, `:ρE`).
Accepts a `Symbol`, `Val(Symbol)`, or formulation struct.
"""
function thermodynamic_density_name end

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
The caller is responsible for computing [`q = grid_moisture_fractions(...)`](@ref grid_moisture_fractions) before passing `q` to this function.
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

#####
##### Set thermodynamic variable interface
#####

"""
    set_thermodynamic_variable!(model, variable_name, value)

Set a thermodynamic variable (e.g., `:θ`, `:T`, `:e`, `:ρθ`, `:ρe`) from the given value.
Dispatches on the thermodynamic formulation type and variable name.
"""
function set_thermodynamic_variable! end

#####
##### Helper accessor functions
#####

"""
    static_energy(model)

Return the specific static energy field for the given model.
"""
function static_energy end

"""
    static_energy_density(model)

Return the static energy density field for the given model.

For `LiquidIcePotentialTemperatureFormulation`, returns a `Field` with boundary conditions
that convert potential temperature fluxes to energy fluxes. This allows users to use
`BoundaryConditionOperation` to extract energy flux values from the model.

For `StaticEnergyFormulation`, returns the prognostic energy density field directly.
"""
function static_energy_density end

"""
    liquid_ice_potential_temperature(model)

Return the liquid-ice potential temperature field for the given model.
"""
function liquid_ice_potential_temperature end

"""
    liquid_ice_potential_temperature_density(model)

Return the liquid-ice potential temperature density field for the given model.
"""
function liquid_ice_potential_temperature_density end
