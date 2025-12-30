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
&\\text{Mass:} && \\partial_t \\rho + \\nabla \\cdot (\\rho \\mathbf{u}) = 0 \\\\
&\\text{Momentum:} && \\partial_t (\\rho \\mathbf{u}) + \\nabla \\cdot (\\rho \\mathbf{u} \\mathbf{u}) + \\nabla p = -\\rho g \\hat{\\mathbf{z}} + \\rho \\mathbf{f} + \\nabla \\cdot \\boldsymbol{\\mathcal{T}}
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
    CompressibleModel

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, XFaceField, YFaceField, ZFaceField
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions, fill_halo_regions!
using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Utils: prettysummary, launch!

using Breeze.Thermodynamics: mixture_gas_constant

using Breeze.AtmosphereModels: AtmosphereModel, compute_moisture_fractions

# Import interface functions to extend
import Breeze.AtmosphereModels:
    materialize_dynamics,
    materialize_momentum_and_velocities,
    dynamics_pressure_solver,
    dynamics_density,
    dynamics_pressure,
    surface_pressure,
    standard_pressure,
    mean_pressure,
    pressure_anomaly,
    total_pressure,
    buoyancy_forceᶜᶜᶜ,
    prognostic_dynamics_field_names,
    additional_dynamics_field_names,
    dynamics_prognostic_fields,
    initialize_model_thermodynamics!,
    compute_dynamics_tendency!,
    compute_auxiliary_dynamics_variables!

include("compressible_dynamics.jl")
include("compressible_buoyancy.jl")

# Define type alias after CompressibleDynamics is defined
const CompressibleModel = AtmosphereModel{<:CompressibleDynamics}

include("compressible_density_tendency.jl")
include("compressible_time_stepping.jl")

end # module

