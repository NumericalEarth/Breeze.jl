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

using Breeze.AtmosphereModels: AtmosphereModels, diagnose_thermodynamic_state,
    dynamics_density, dynamics_pressure, standard_pressure, dynamics_prognostic_fields,
    compute_moisture_fractions, maybe_adjust_thermodynamic_state, div_ρUc,
    c_div_ρU, ∇_dot_Jᶜ, w_buoyancy_forceᶜᶜᶠ,
    AtmosphereModelBuoyancy, microphysical_tendency

using Breeze.Thermodynamics: StaticEnergyState, LiquidIcePotentialTemperatureState, with_temperature

# The lowercase c is a singleton instance of Center
const c = Center()

include("static_energy_formulation.jl")
include("static_energy_tendency.jl")

# Kernel wrapper for launching static_energy_tendency
# (needs to be defined after static_energy_tendency is defined)
@kernel function compute_static_energy_tendency!(Gρe, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρe[i, j, k] = static_energy_tendency(i, j, k, grid, args...)
end

end # module
