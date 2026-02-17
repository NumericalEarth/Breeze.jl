using Reactant: Reactant

using Oceananigans: AbstractModel, ReactantState
using Oceananigans.TimeSteppers: update_state!, tick!, step_lagrangian_particles!, compute_flux_bc_tendencies!
using Breeze.AtmosphereModels: compute_pressure_correction!, make_pressure_correction!
using Breeze.TimeSteppers: SSPRungeKutta3, store_initial_state!, ssp_rk3_substep!,
                           maybe_initialize_state!

using Oceananigans: initialize!
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, time_step!

# TODO: move maybe_initialize_state! to Oceananigans
# (see https://github.com/CliMA/Oceananigans.jl/issues/5300)

# Reactant handles initialization via first_time_step!, so this is a no-op.
Breeze.TimeSteppers.maybe_initialize_state!(::AbstractModel{<:Any, <:ReactantState}, callbacks) = nothing

#####
##### SSPRungeKutta3 time stepping for Reactant
#####

function OceananigansTimeSteppers.time_step!(model::AbstractModel{<:SSPRungeKutta3{FT}, <:ReactantState}, Δt; callbacks=[]) where FT
    ts = model.timestepper
    α¹, α², α³ = ts.α¹, ts.α², ts.α³

    store_initial_state!(model)

    # Stage 1: u⁽¹⁾ = u⁽⁰⁾ + Δt * L(u⁽⁰⁾)
    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α¹)
    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)
    tick!(model.clock, Δt; stage=true)
    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, Δt)

    # Stage 2: u⁽²⁾ = 3/4 u⁽⁰⁾ + 1/4 (u⁽¹⁾ + Δt * L(u⁽¹⁾))
    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α²)
    compute_pressure_correction!(model, α² * Δt)
    make_pressure_correction!(model, α² * Δt)
    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, α² * Δt)

    # Stage 3: u⁽³⁾ = 1/3 u⁽⁰⁾ + 2/3 (u⁽²⁾ + Δt * L(u⁽²⁾))
    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α³)
    compute_pressure_correction!(model, α³ * Δt)
    make_pressure_correction!(model, α³ * Δt)
    tick!(model.clock, Δt)

    # Only update Float64 clock fields if Δt is not traced
    if !(Δt isa Reactant.TracedRNumber)
        model.clock.last_stage_Δt = Δt
        model.clock.last_Δt = Δt
    end

    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, α³ * Δt)

    return nothing
end

function OceananigansTimeSteppers.first_time_step!(model::AbstractModel{<:SSPRungeKutta3, <:ReactantState}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end
