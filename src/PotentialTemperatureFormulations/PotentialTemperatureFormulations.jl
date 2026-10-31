"""
    PotentialTemperatureFormulations

Submodule defining the potential temperature thermodynamic formulations for atmosphere models.

`PotentialTemperatureFormulation` uses a potential temperature density `ρθ` as the prognostic
thermodynamic variable: the liquid-ice potential temperature (`LiquidIcePotentialTemperatureFormulation`)
or the potential temperature of cloud models like CM1 (`DryPotentialTemperatureFormulation`).
"""
module PotentialTemperatureFormulations

export PotentialTemperatureFormulation, LiquidIcePotentialTemperatureFormulation, DryPotentialTemperatureFormulation

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, Center, znode, fields
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.Operators: ℑzᵃᵃᶜ, divᶜᶜᶜ
using Oceananigans.Utils: prettysummary, launch!

using Breeze.AtmosphereModels: AtmosphereModels, diagnose_thermodynamic_state,
    set_thermodynamic_variable!, dynamics_density, total_density, dynamics_pressure, standard_pressure,
    pressure_from_density_temperature,
    dynamics_prognostic_fields, grid_moisture_fractions, maybe_adjust_thermodynamic_state,
    div_ρUc, c_div_ρU, ∇_dot_Jᶜ, AtmosphereModelBuoyancy,
    add_sedimentation_tendency!, sedimentation_composition_increment,
    dry_air_increment, enthalpy_increment,
    radiation_flux_divergence, materialize_formulation,
    DefaultTemperatureSolver, default_temperature_solver,
    PrecipitationDragHeating, FixedVolumePhaseChange, SubgridDissipativeHeating,
    materialize_phase_change_source, subgrid_dissipation,
    moisture_prognostic_name, condensate_field_names, condensate_liquid_fraction,
    phase_change_source_applies
using Breeze.Solvers: materialize_solver
using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, StaticEnergyState, with_temperature,
    mixture_heat_capacity, mixture_gas_constant, dry_air_gas_constant, vapor_gas_constant,
    vapor_equivalent_heat_capacity, vapor_equivalent_gas_constant, condensate_latent_energy,
    exner_exponent, AbstractExnerExponent, VaporEquivalentExponent, DryExponent, absolute_zero_latent_heat,
    DryDensityState

# The lowercase c is a singleton instance of Center
const c = Center()

include("potential_temperature_formulation.jl")
include("potential_temperature_tendency.jl")
include("dry_potential_temperature.jl")

# Kernel wrapper for launching potential_temperature_tendency
# (needs to be defined after potential_temperature_tendency is defined)
@kernel function compute_potential_temperature_tendency!(Gρθ, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρθ[i, j, k] = potential_temperature_tendency(i, j, k, grid, args...)
end

end # module
