"""
    StaticEnergyFormulations

Submodule defining the static energy thermodynamic formulation for atmosphere models.

`StaticEnergyFormulation` uses moist static energy density `ρe` as the prognostic thermodynamic variable.
Moist static energy is a conserved quantity in adiabatic, frictionless flow that combines
sensible heat, gravitational potential energy, and latent heat.
"""
module StaticEnergyFormulations

export StaticEnergyFormulation

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, Center, znode
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.Operators: ℑzᵃᵃᶜ
using Oceananigans.Utils: prettysummary, launch!

using Breeze.Thermodynamics: StaticEnergyState, LiquidIcePotentialTemperatureState, with_temperature

# The lowercase c is a singleton instance of Center
const c = Center()

# Import interface functions from parent module
import Breeze.AtmosphereModels: materialize_formulation,
                                prognostic_thermodynamic_field_names,
                                additional_thermodynamic_field_names,
                                thermodynamic_density_name,
                                thermodynamic_density,
                                collect_prognostic_fields,
                                compute_auxiliary_thermodynamic_variables!,
                                diagnose_thermodynamic_state,
                                compute_thermodynamic_tendency!,
                                set_thermodynamic_variable!,
                                static_energy,
                                static_energy_density,
                                liquid_ice_potential_temperature,
                                liquid_ice_potential_temperature_density,
                                dynamics_density,
                                dynamics_pressure,
                                standard_pressure,
                                dynamics_prognostic_fields,
                                compute_moisture_fractions,
                                maybe_adjust_thermodynamic_state,
                                div_ρUc,
                                ∇_dot_Jᶜ,
                                w_buoyancy_forceᶜᶜᶠ,
                                AtmosphereModelBuoyancy,
                                microphysical_tendency

include("static_energy_formulation.jl")
include("static_energy_tendency.jl")

# Kernel wrapper for launching static_energy_tendency
# (needs to be defined after static_energy_tendency is defined)
@kernel function compute_static_energy_tendency!(Gρe, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρe[i, j, k] = static_energy_tendency(i, j, k, grid, args...)
end

end # module

