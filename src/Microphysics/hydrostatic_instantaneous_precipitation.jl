#####
##### Instantaneous precipitation (zero-moment) on moist HydrostaticDynamics
#####
#
# The scheme keeps no condensate between time steps, so the hydrostatic core carries vapor alone
# (the moisture density ρqᵛ) and diagnoses the state with all of its water as vapor. Its generic
# once-per-step update (`microphysics_model_update!`) condenses the supersaturation of every cell
# at its pressure with θˡⁱ fixed and removes the condensate in place. `precipitation_mass_change!`
# of the dynamics then takes the weight of that water off the surface pressure, with the energy
# bookkeeping of a removal at fixed density (plvd_logs/3c_0M_ENERGY_DERIVATION.txt). The scheme
# runs on a PressureLevelGrid only. 2M and other schemes are not adapted to HydrostaticDynamics.

using Breeze.HydrostaticEquations: HydrostaticEquations, MoistHydrostaticModel
using Oceananigans.Fields: ZeroField

HydrostaticEquations.supports_hydrostatic_dynamics(::IP) = true

# No liquid between time steps and no rain field: the fall speed is not computed (`wʳ = nothing`)
function HydrostaticEquations.hydrostatic_water_fields(::IP, μ)
    zero_field = ZeroField(eltype(μ.qᵛ))
    return (; qᵛ = μ.qᵛ, qˡ = zero_field, qⁱ = zero_field, qᶜˡ = zero_field, ρqʳ = zero_field, ρqˢ = zero_field, wʳ = nothing)
end

# The generic update of the scheme acts on HydrostaticDynamics as on any dynamics: it condenses
# in place, and `precipitation_mass_change!` takes the mass of the water. What the hydrostatic
# core adds at the end of its time step is the check of its state.
function AtmosphereModels.microphysics_model_update!(ip::IP, model::MoistHydrostaticModel)
    invoke(AtmosphereModels.microphysics_model_update!, Tuple{IP, Any}, ip, model)
    HydrostaticEquations.validate_time_step_state(model)
    return nothing
end

# The rates of the removal are densities of the volume of the coordinate: carried with the density
# of a PressureLevelGrid, they keep the mass per unit area of the water that left
HydrostaticEquations.carried_microphysical_densities(::IP, μ) = (μ.precipitation_rate, μ.ice_precipitation_rate)

# On HydrostaticDynamics the generic update needs a grid whose Δz is a physical depth
HydrostaticEquations.requires_pressure_level_grid(::IP) = true
