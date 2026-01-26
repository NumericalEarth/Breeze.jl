"""
    CompressibleEquations

Module implementing fully compressible dynamics for atmosphere models.

The compressible formulation directly time-steps density as a prognostic variable
and computes pressure from the ideal gas law. This formulation does not filter
acoustic waves, so explicit time-stepping with small time steps (or acoustic
substepping) is required.

The fully compressible Euler equations in conservation form are:

```math
\\begin{aligned}
&\\text{Mass:} && \\partial_t \\rho + \\boldsymbol{\\nabla \\cdot} (\\rho \\boldsymbol{u}) = 0 \\\\
&\\text{Momentum:} && \\partial_t (\\rho \\boldsymbol{u}) + \\boldsymbol{\\nabla \\cdot} (\\rho \\boldsymbol{u} \\boldsymbol{u}) + \\boldsymbol{\\nabla} p = -\\rho g \\hat{\\boldsymbol{z}} + \\rho \\boldsymbol{f} + \\boldsymbol{\\nabla \\cdot \\mathcal{T}}
\\end{aligned}
```

Pressure is computed from the ideal gas law:
```math
p = \\rho R^m T
```
where ``R^m`` is the mixture gas constant.
"""
module CompressibleEquations

export
    CompressibleDynamics,
    CompressibleModel,
    AcousticSubstepper

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, XFaceField, YFaceField, ZFaceField, prognostic_fields
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions, fill_halo_regions!
using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Utils: prettysummary, launch!

using Breeze.Thermodynamics: mixture_gas_constant, mixture_heat_capacity

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, grid_moisture_fractions, dynamics_density, standard_pressure
using Breeze.PotentialTemperatureFormulations: LiquidIcePotentialTemperatureFormulation

include("compressible_dynamics.jl")
include("compressible_buoyancy.jl")

# Define type alias after CompressibleDynamics is defined
const CompressibleModel = AtmosphereModel{<:CompressibleDynamics}

include("compressible_density_tendency.jl")
include("compressible_time_stepping.jl")
include("acoustic_substepping.jl")

end # module
