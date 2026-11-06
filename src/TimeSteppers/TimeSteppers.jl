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
       maybe_prepare_first_time_step!,
       prepare_first_time_step!

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans: Oceananigans
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers,
                                 update_state!, maybe_prepare_first_time_step!, reconcile_state!
using Breeze.AtmosphereModels: kernel_time_step

"""
$(TYPEDSIGNATURES)

Prepare `model`, at iteration 0, for a first time step of `Δt`: set `clock.last_Δt`, reconcile
the state and `update_state!`, which at initialization also computes the initial radiative
fluxes. `time_step!` does this itself at iteration 0 (`maybe_prepare_first_time_step!`), so
eager stepping needs no call.

Under Reactant the iteration is traced, so `time_step!` cannot tell that it starts the run and
skips the preparation. A traced loop of `time_step!` (a calibration window, say) then calls
`prepare_first_time_step!` before the loop; `BreezeReactantExt` computes the initial radiative
fluxes explicitly there, and refreshes the tendencies with them. Without it the first step would
use the diffusivities, tendencies and radiative fluxes the model held before the trace.

```jldoctest
using Breeze
using Breeze.TimeSteppers: prepare_first_time_step!

grid = RectilinearGrid(size=4, z=(0, 1000), topology=(Flat, Flat, Bounded))
model = AtmosphereModel(grid)
prepare_first_time_step!(model, 10)
model.clock.last_Δt

# output
10.0
```
"""
prepare_first_time_step!(model, Δt) = maybe_prepare_first_time_step!(model, Δt, [])

include("ssp_runge_kutta_3.jl")
include("acoustic_substep_helpers.jl")
include("acoustic_runge_kutta_3.jl")

# Extend TimeStepper to support time steppers via Symbol
OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta3}, args...; kwargs...) =
    SSPRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:AcousticRungeKutta3}, args...; kwargs...) =
    AcousticRungeKutta3(args...; kwargs...)

end # module
