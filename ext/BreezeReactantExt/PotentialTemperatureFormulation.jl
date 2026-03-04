using Oceananigans: ReactantState
using Oceananigans.Grids: LatitudeLongitudeGrid
using Oceananigans.Fields: set!
using Breeze.AtmosphereModels: AtmosphereModels, dynamics_density
using Breeze.PotentialTemperatureFormulations: LiquidIcePotentialTemperatureFormulation

# Workaround: on LatitudeLongitudeGrid + ReactantState, the BinaryOperation broadcast
# set!(ρθ, ρ * θ) triggers _broadcast_kernel! whose type closure carries the grid's
# ConcretePJRTArray metric vectors — the GPU compiler rejects getindex on them.
# All three fields are CenterField, so parent-array broadcasting is equivalent.
# See https://github.com/NumericalEarth/Breeze.jl/issues/543

const ReactantLatLonPTModel = AtmosphereModel{<:Any,
                                              <:LiquidIcePotentialTemperatureFormulation,
                                              <:ReactantState,
                                              <:Any,
                                              <:LatitudeLongitudeGrid}

function AtmosphereModels.set_thermodynamic_variable!(model::ReactantLatLonPTModel, ::Union{Val{:θ}, Val{:θˡⁱ}}, value)
    set!(model.formulation.potential_temperature, value)
    ρ   = dynamics_density(model.dynamics)
    θˡⁱ = model.formulation.potential_temperature
    ρθ  = model.formulation.potential_temperature_density
    parent(ρθ) .= parent(ρ) .* parent(θˡⁱ)
    return nothing
end
