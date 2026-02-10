module AtmosphereModels

using Reactant
using Oceananigans: ReactantState
using Oceananigans.Models.NonhydrostaticModels: compute_pressure_correction!, make_pressure_correction!
using Oceananigans.TimeSteppers: update_state!
using Breeze: Breeze, AtmosphereModel

function Breeze.AtmosphereModels.enforce_mass_conservation!(model::AtmosphereModel{<:Any, <:Any, <:ReactantState})
    FT = eltype(model.grid)
    Δt = one(FT)
    Reactant.@jit compute_pressure_correction!(model, Δt)
    Reactant.@jit make_pressure_correction!(model, Δt)
    Reactant.@jit update_state!(model, compute_tendencies=false)
    return nothing
end

end # module
