module TimeSteppers

using Reactant
using Oceananigans
using Breeze

using Oceananigans: AbstractModel, ReactantState

import Oceananigans: initialize!
import Oceananigans.TimeSteppers: time_step!, first_time_step!

using Oceananigans.TimeSteppers:
    update_state!,
    tick!,
    step_lagrangian_particles!,
    compute_flux_bc_tendencies!

using Oceananigans.Models.NonhydrostaticModels:
    compute_pressure_correction!,
    make_pressure_correction!

using Breeze.TimeSteppers: SSPRungeKutta3, store_initial_state!, ssp_rk3_substep!


#####
##### SSPRungeKutta3 time stepping for Reactant
#####

"""
Step forward `model` one time step `Δt` with the SSP RK3 method for Reactant-traced models.

This version removes the `iteration == 0` check since that causes issues with
TracedRNumber in boolean contexts. The check is handled by `first_time_step!` instead.

Note: When Δt is traced (a function argument that Reactant traces), we skip
updating last_Δt and last_stage_Δt since they are Float64 fields that can't
accept TracedRNumber values.
"""
function time_step!(model::AbstractModel{<:SSPRungeKutta3{FT}, <:ReactantState}, Δt; callbacks=[]) where FT
    ts = model.timestepper
    α¹ = ts.α¹
    α² = ts.α²
    α³ = ts.α³

    # Store u^(0) for use in stages 2 and 3
    store_initial_state!(model)

    #
    # First stage: u^(1) = u^(0) + Δt * L(u^(0))
    #

    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α¹)

    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)

    tick!(model.clock, Δt; stage=true)
    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, Δt)

    #
    # Second stage: u^(2) = 3/4 u^(0) + 1/4 (u^(1) + Δt * L(u^(1)))
    #

    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α²)

    compute_pressure_correction!(model, α² * Δt)
    make_pressure_correction!(model, α² * Δt)

    # Don't tick - still at t + Δt for time-dependent forcing
    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, α² * Δt)

    #
    # Third stage: u^(3) = 1/3 u^(0) + 2/3 (u^(2) + Δt * L(u^(2)))
    #

    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α³)

    compute_pressure_correction!(model, α³ * Δt)
    make_pressure_correction!(model, α³ * Δt)

    # Final tick
    tick!(model.clock, Δt)

    # Update last_stage_Δt and last_Δt only if Δt is NOT traced
    # (Float64 fields can't accept TracedRNumber values)
    if !(Δt isa Reactant.TracedRNumber)
        model.clock.last_stage_Δt = Δt
        model.clock.last_Δt = Δt
    end

    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, α³ * Δt)

    return nothing
end

function first_time_step!(model::AbstractModel{<:SSPRungeKutta3, <:ReactantState}, Δt)
    initialize!(model)
    # The first update_state is conditionally gated from within time_step! normally, but not Reactant
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

end # module