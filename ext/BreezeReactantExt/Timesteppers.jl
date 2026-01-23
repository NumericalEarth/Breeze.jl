module TimeSteppers

using Reactant
using Oceananigans
using Breeze

using Oceananigans: AbstractModel, Distributed, ReactantState
using Oceananigans.Grids: AbstractGrid
using Oceananigans.TimeSteppers:
    update_state!,
    tick!,
    step_lagrangian_particles!,
    QuasiAdamsBashforth2TimeStepper,
    cache_previous_tendencies!,
    compute_flux_bc_tendencies!

using Oceananigans.Models.NonhydrostaticModels:
    compute_pressure_correction!,
    make_pressure_correction!

using Breeze.TimeSteppers: SSPRungeKutta3, store_initial_state!, ssp_rk3_substep!

import Oceananigans.TimeSteppers: Clock, first_time_step!, time_step!, ab2_step!
import Oceananigans: initialize!

#####
##### Type aliases
#####

const ReactantGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:ReactantState}

const ReactantModel{TS} = Union{
    AbstractModel{TS, <:ReactantState},
    AbstractModel{TS, <:Distributed{<:ReactantState}}
}

#####
##### Clock constructor for ReactantGrid
#####

function Clock(::ReactantGrid)
    FT = Oceananigans.defaults.FloatType
    t = ConcreteRNumber(zero(FT))
    iter = ConcreteRNumber(0)
    stage = 0
    last_Δt = convert(FT, Inf)
    last_stage_Δt = convert(FT, Inf)
    return Clock(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)
end

#####
##### QuasiAdamsBashforth2 time stepping (from Oceananigans)
#####

function time_step!(model::ReactantModel{<:QuasiAdamsBashforth2TimeStepper{FT}}, Δt;
                    callbacks=[], euler=false) where FT

    # Only update last_Δt if Δt is not traced
    if !(Δt isa Reactant.TracedRNumber)
        model.clock.last_Δt = Δt
    end

    # If euler, then set χ = -0.5
    minus_point_five = convert(FT, -0.5)
    ab2_timestepper = model.timestepper
    χ = ifelse(euler, minus_point_five, ab2_timestepper.χ)
    χ₀ = ab2_timestepper.χ
    ab2_timestepper.χ = χ

    ab2_step!(model, Δt, callbacks)
    cache_previous_tendencies!(model)

    tick!(model.clock, Δt)

    # Only update if Δt is not traced (Float64 fields can't accept TracedRNumber)
    if !(Δt isa Reactant.TracedRNumber)
        model.clock.last_Δt = Δt
        model.clock.last_stage_Δt = Δt
    end

    update_state!(model, callbacks)
    step_lagrangian_particles!(model, Δt)

    ab2_timestepper.χ = χ₀

    return nothing
end

function first_time_step!(model::ReactantModel, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

function first_time_step!(model::ReactantModel{<:QuasiAdamsBashforth2TimeStepper}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt, euler=true)
    return nothing
end

#####
##### SSPRungeKutta3 time stepping (Breeze-specific)
#####

function time_step!(model::ReactantModel{<:SSPRungeKutta3{FT}}, Δt; callbacks=[]) where FT
    ts = model.timestepper
    α¹ = ts.α¹
    α² = ts.α²
    α³ = ts.α³

    store_initial_state!(model)

    # Stage 1: u^(1) = u^(0) + Δt * L(u^(0))
    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α¹)
    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)
    tick!(model.clock, Δt; stage=true)
    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, Δt)

    # Stage 2: u^(2) = 3/4 u^(0) + 1/4 (u^(1) + Δt * L(u^(1)))
    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α²)
    compute_pressure_correction!(model, α² * Δt)
    make_pressure_correction!(model, α² * Δt)
    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, α² * Δt)

    # Stage 3: u^(3) = 1/3 u^(0) + 2/3 (u^(2) + Δt * L(u^(2)))
    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α³)
    compute_pressure_correction!(model, α³ * Δt)
    make_pressure_correction!(model, α³ * Δt)
    tick!(model.clock, Δt)

    # Only update if Δt is not traced (Float64 fields can't accept TracedRNumber)
    if !(Δt isa Reactant.TracedRNumber)
        model.clock.last_stage_Δt = Δt
        model.clock.last_Δt = Δt
    end

    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, α³ * Δt)

    return nothing
end

function first_time_step!(model::ReactantModel{<:SSPRungeKutta3}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

end # module
