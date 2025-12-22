"""
    ThermodynamicFormulations

Submodule defining the thermodynamic formulations for atmosphere models.

A thermodynamic formulation specifies the prognostic thermodynamic variable used to
evolve the atmospheric state. Currently supports:

- `StaticEnergyFormulation`: uses moist static energy density `ρe` as prognostic variable.
- `LiquidIcePotentialTemperatureFormulation`: uses liquid-ice potential temperature density `ρθ` as prognostic variable.

Future implementations may include:
- `TotalEnergyFormulation`: uses total energy density `ρE` (for compressible dynamics).
"""
module ThermodynamicFormulations

export
    # Types
    StaticEnergyFormulation,
    LiquidIcePotentialTemperatureFormulation,
    # Interface functions
    default_thermodynamic_formulation,
    materialize_thermodynamic_formulation,
    prognostic_thermodynamic_field_names,
    additional_thermodynamic_field_names,
    thermodynamic_density_name,
    thermodynamic_density,
    collect_prognostic_fields,
    compute_auxiliary_thermodynamic_variables!,
    diagnose_thermodynamic_state,
    compute_thermodynamic_tendency!

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, Center, znode, fields, prognostic_fields
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.Utils: prettysummary, launch!

# The lowercase c is a singleton instance of Center
const c = Center()

# Import dynamics interface functions from sibling module
using ..Dynamics: dynamics_density, dynamics_pressure

# Note: compute_moisture_fractions is NOT imported here because it creates a circular dependency.
# Instead, diagnose_thermodynamic_state accepts already-computed moisture fractions `q`.

include("formulation_interface.jl")
include("static_energy_formulation.jl")
include("potential_temperature_formulation.jl")

end # module

