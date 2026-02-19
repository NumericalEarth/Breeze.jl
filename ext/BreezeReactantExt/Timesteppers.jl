using Reactant: Reactant

using Oceananigans: AbstractModel, ReactantState
using Oceananigans.TimeSteppers: update_state!, tick!, step_lagrangian_particles!, compute_flux_bc_tendencies!
using Breeze.AtmosphereModels: compute_pressure_correction!, make_pressure_correction!
using Breeze.TimeSteppers: SSPRungeKutta3, store_initial_state!, ssp_rk3_substep!

using Oceananigans: initialize!
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, time_step!

# TODO: move maybe_initialize_state! to Oceananigans
# (see https://github.com/CliMA/Oceananigans.jl/issues/5300)

# Reactant handles initialization via first_time_step!, so this is a no-op.
Breeze.TimeSteppers.maybe_initialize_state!(::AbstractModel{<:Any, <:ReactantState}, callbacks) = nothing

function OceananigansTimeSteppers.first_time_step!(model::AbstractModel{<:SSPRungeKutta3, <:ReactantState}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end
