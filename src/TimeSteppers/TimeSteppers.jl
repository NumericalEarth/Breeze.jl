"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including:
- `SSPRungeKutta3`: Standard SSP RK3 scheme for explicit time stepping
- `AcousticRungeKutta3`: Wicker-Skamarock RK3 with acoustic substepping for compressible dynamics
"""
module TimeSteppers

export SSPRungeKutta3, AcousticRungeKutta3,
       store_initial_state!,
       ssp_rk3_substep!,
       maybe_prepare_first_time_step!

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans: Oceananigans
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers,
                                 update_state!, maybe_prepare_first_time_step!, reconcile_state!
using Oceananigans.Fields: Field, instantiated_location
using Breeze.AtmosphereModels: kernel_time_step

# Storage for the state at the start of a time step, with the float type of each prognostic field.
# TODO: Oceananigans' `similar(::Field)` allocates data with `eltype(grid)` rather than `eltype(field)`,
# which silently rounds prognostic fields whose float type differs from the grid's — for example the
# double-precision thermodynamic fields of `AtmosphereModel(grid; thermodynamic_float_type = Float64)`
# on a single-precision grid — every time the state is restored from `U⁰`. Once `similar` preserves
# the float type in Oceananigans, this can be replaced by `similar`. (Tendency storage `Gⁿ` may keep
# the grid float type: a tendency needs only relative precision.)
initial_state_storage(field::Field) = Field(instantiated_location(field), field.grid, eltype(field))
initial_state_storage(field) = similar(field)

include("ssp_runge_kutta_3.jl")
include("acoustic_substep_helpers.jl")
include("acoustic_runge_kutta_3.jl")
include("lagrangian_particle_stages.jl")

# Extend TimeStepper to support time steppers via Symbol
OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta3}, args...; kwargs...) =
    SSPRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:AcousticRungeKutta3}, args...; kwargs...) =
    AcousticRungeKutta3(args...; kwargs...)

end # module
