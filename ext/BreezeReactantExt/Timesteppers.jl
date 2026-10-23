using Oceananigans: ReactantState, initialize!
using Reactant: TracedRNumber
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, update_state!
using Breeze.AtmosphereModels: AtmosphereModels
using Breeze.Microphysics: Microphysics, DCMIP2016KesslerMicrophysics
using Breeze.TimeSteppers: SSPRungeKutta3, CompressibleAcousticModel

function OceananigansTimeSteppers.first_time_step!(model::AtmosphereModel{<:Any, <:Any, <:ReactantState, <:SSPRungeKutta3}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

# Reactant tracing breaks if Δt is materialized via `convert` outside the kernel;
# pass it through unchanged and let in-kernel arithmetic see the traced value.
@inline AtmosphereModels.kernel_time_step(::ReactantState, grid, Δt) = Δt

# A traced Δt is a time step: traced stepping sets `last_Δt` before the first step, and the
# eager guard cannot branch on a traced value.
@inline AtmosphereModels.invalid_time_step(::TracedRNumber) = false

# Reactant initializes through `first_time_step!`, and the eager guard branches on traced
# clock fields; mirrors OceananigansReactantExt's no-ops for the upstream steppers.
OceananigansTimeSteppers.maybe_prepare_first_time_step!(model::CompressibleAcousticModel{<:ReactantState}, Δt, callbacks) = nothing

# Reverse mode cannot differentiate the data-dependent number of Kessler sedimentation substeps:
# loop a static `maximum_sedimentation_substeps` (see `sedimentation_substep_limit`)
Microphysics.sedimentation_substep_limit(::ReactantState, microphysics::DCMIP2016KesslerMicrophysics) =
    Val(microphysics.maximum_sedimentation_substeps)

function AtmosphereModels.validate_microphysics_architecture(microphysics::DCMIP2016KesslerMicrophysics, ::ReactantState)
    microphysics.maximum_sedimentation_substeps ≥ 1 && return nothing
    throw(ArgumentError("DCMIP2016KesslerMicrophysics on ReactantState needs `maximum_sedimentation_substeps ≥ 1`: " *
                        "Reactant cannot differentiate a data-dependent number of sedimentation substeps, so " *
                        "it loops a static number, and a step that needs more is rejected (its update is NaN). " *
                        "Choose it from eager control runs of the same case, where " *
                        "`model.microphysical_fields.sedimentation_substeps` records the most substeps each " *
                        "column needed."))
end
