using Reactant: @jit
using Oceananigans: ReactantState
using Oceananigans.Grids: LatitudeLongitudeGrid
using Oceananigans.Fields: set!
using Breeze.AtmosphereModels: AtmosphereModels, dynamics_density
using Breeze.PotentialTemperatureFormulations: LiquidIcePotentialTemperatureFormulation

# Workaround: on LatitudeLongitudeGrid + ReactantState, eager KA kernel compilation
# rejects getindex on the grid's ConcretePJRTArray metric vectors.
# For CCC × CCC (thermodynamic set!), parent-array broadcasting is exact.
# For CCC × FCC (velocity set!), we @jit to trace through MLIR where all
# arrays are TracedRArray and interpolation operators work correctly.
# See https://github.com/NumericalEarth/Breeze.jl/issues/543

const ReactantLatLonModel = AtmosphereModel{<:Any, <:Any, <:ReactantState, <:Any, <:LatitudeLongitudeGrid}
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

_set_field_product!(dest, a, b) = set!(dest, a * b)

function AtmosphereModels.set_velocity!(model::ReactantLatLonModel, name::Symbol, value)
    u = model.velocities[name]
    set!(u, value)
    ρ = dynamics_density(model.dynamics)
    ϕ = model.momentum[Symbol(:ρ, name)]
    @jit _set_field_product!(ϕ, ρ, u)
    return nothing
end
