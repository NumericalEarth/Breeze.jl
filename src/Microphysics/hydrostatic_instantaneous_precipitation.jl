#####
##### Instantaneous precipitation (zero-moment) on moist HydrostaticDynamics
#####
#
# The scheme keeps no condensate between time steps, so the hydrostatic core carries vapor alone
# (the moisture density ρqᵛ) and diagnoses the state with all of its water as vapor. Once per time
# step, after the dynamics, the column physics condenses the supersaturation of every layer at its
# pressure, with the liquid-ice potential temperature fixed, and the condensate falls to the ground
# within the step (`HydrostaticEquations.rain_out_column_physics!`), with the enthalpy, pressure
# and ledger bookkeeping of the sedimentation of rain. The saturation adjustment is that of the
# scheme (`saturation_adjustment`), isobaric at the layer pressure because the hydrostatic state is
# closed on pressure. 2M and other schemes are not adapted to HydrostaticDynamics.

using Breeze.HydrostaticEquations: HydrostaticEquations, MoistHydrostaticModel
using Oceananigans.Fields: ZeroField

HydrostaticEquations.supports_hydrostatic_dynamics(::IP) = true

# No liquid between time steps and no rain field: the fall speed is not computed (`wʳ = nothing`)
HydrostaticEquations.hydrostatic_water_fields(::IP, μ) =
    (; qᵛ = μ.qᵛ, qˡ = ZeroField(eltype(μ.qᵛ)), qᶜˡ = ZeroField(eltype(μ.qᵛ)), ρqʳ = ZeroField(eltype(μ.qᵛ)), wʳ = nothing)

HydrostaticEquations.apply_column_physics!(ip::IP, model) = HydrostaticEquations.rain_out_column_physics!(ip, model)

# The time steppers apply the column physics once per step through this
AtmosphereModels.microphysics_model_update!(ip::IP, model::MoistHydrostaticModel) =
    HydrostaticEquations.time_step_column_physics!(ip, model)

# The condensate of layer k: the supersaturation that the saturation adjustment of the scheme
# condenses at the layer pressure with θ fixed, as a density of the measure of ρqᵛ, and the
# temperature after condensation (which the condensate carries away)
@inline function HydrostaticEquations.rainout_condensate(ip::IP, i, j, k, grid, formulation, dynamics, constants, ρqᵛ)
    ρ = @inbounds total_density(dynamics)[i, j, k]
    qᵗ = @inbounds ρqᵛ[i, j, k] / ρ
    𝒰₀ = diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, MoistureMassFractions(qᵗ))
    𝒰₁ = adjust_thermodynamic_state(𝒰₀, ip.saturation_adjustment, constants)
    qᵛ⁺ = 𝒰₁.moisture_mass_fractions.vapor
    return ρ * max(0, qᵗ - qᵛ⁺), temperature(𝒰₁, constants)
end
