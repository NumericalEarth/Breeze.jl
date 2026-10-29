"""
    MoistEntropyFormulations

Submodule defining the moist entropy thermodynamic formulation for atmosphere models.

`MoistEntropyFormulation` uses the moist entropy density `ρη` as the prognostic thermodynamic
variable. It is supported with `CompressibleDynamics`.
"""
module MoistEntropyFormulations

export MoistEntropyFormulation

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, Center, Face, Field, znode
using Oceananigans.BoundaryConditions: BoundaryConditions as OceananigansBC, fill_halo_regions!,
                                       BoundaryCondition, Flux, FieldBoundaryConditions, Bottom
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ℑzᵃᵃᶠ, V⁻¹ᶜᶜᶜ
using Oceananigans.Utils: prettysummary, launch!

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, set_thermodynamic_variable!,
    dynamics_density, total_density, standard_pressure, dynamics_prognostic_fields,
    grid_moisture_fractions, div_ρUc, c_div_ρU, ∇_dot_Jᶜ, AtmosphereModelBuoyancy,
    add_sedimentation_tendency!, explicit_condensate_mass_fluxes, radiation_flux_divergence,
    materialize_formulation, specific_prognostic_moisture, moisture_prognostic_name,
    prognostic_field_names, grid_microphysical_state, microphysical_tendencies,
    condensate_liquid_fraction
using Breeze.AtmosphereModels.Diagnostics: Diagnostics
using Breeze.Thermodynamics: MoistEntropyState, LiquidIceDensityState, MoistureMassFractions,
    moist_entropy, temperature, vapor_entropy, liquid_entropy, ice_entropy,
    vapor_liquid_gibbs_difference, ice_liquid_gibbs_difference, VaporEquivalentExponent,
    mixture_heat_capacity, total_specific_moisture
using Breeze.Solvers: NewtonSolver
using Breeze.BoundaryConditions: BoundaryConditions as AtmosphereBoundaryConditions,
    BulkSensibleHeatFluxFunction, BulkSensibleHeatFluxBoundaryCondition,
    BulkVaporFluxBoundaryCondition, wall_moisture_fractions, wall_height, near_wall_indices,
    materialize_atmosphere_boundary_condition, map_field_boundary_conditions, boundary_sides

using Oceananigans.Fields: set!

# The lowercase c is a singleton instance of Center
const c = Center()
const f = Face()

include("moist_entropy_formulation.jl")
include("moist_entropy_tendency.jl")
include("moist_entropy_boundary_conditions.jl")

# Kernel wrapper for launching moist_entropy_tendency
@kernel function compute_moist_entropy_tendency!(Gρη, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρη[i, j, k] = moist_entropy_tendency(i, j, k, grid, args...)
end

end # module
