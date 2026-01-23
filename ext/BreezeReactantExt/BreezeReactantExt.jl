module BreezeReactantExt

using Reactant
using Oceananigans
using Breeze

include("Timesteppers.jl")
using .TimeSteppers

#####
##### Clock tick functions for TracedRNumber
#####

function Oceananigans.TimeSteppers.tick_time!(clock::Oceananigans.TimeSteppers.Clock{<:Reactant.TracedRNumber}, Δt)
    nt = Oceananigans.TimeSteppers.next_time(clock, Δt)
    clock.time.mlir_data = nt.mlir_data
    return nt
end

function Oceananigans.TimeSteppers.tick!(clock::Oceananigans.TimeSteppers.Clock{<:Any, <:Any, <:Reactant.TracedRNumber}, Δt; stage=false)
    Oceananigans.TimeSteppers.tick_time!(clock, Δt)

    if stage
        clock.stage += 1
        # Only update if Δt is not traced (Float64 field can't accept TracedRNumber)
        if !(Δt isa Reactant.TracedRNumber)
            clock.last_stage_Δt = Δt
        end
    else
        clock.iteration.mlir_data = (clock.iteration + 1).mlir_data
        clock.stage = 1
    end

    return nothing
end

end # module
