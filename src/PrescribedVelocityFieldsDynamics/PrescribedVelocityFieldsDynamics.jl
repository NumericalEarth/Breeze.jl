"""
    PrescribedVelocityFieldsDynamics

Module implementing kinematic dynamics for atmosphere models.

Kinematic dynamics prescribes the velocity field rather than solving for it,
enabling isolated testing of microphysics, thermodynamics, and other physics
without the complexity of solving the momentum equations.
"""
module PrescribedVelocityFieldsDynamics

export
    PrescribedDynamics,
    KinematicModel

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt

using Oceananigans: Oceananigans, CenterField, XFaceField, YFaceField, ZFaceField, fields
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: FunctionField, ZeroField, field
using Oceananigans.Grids: Face, Center
using Oceananigans.TimeSteppers: Clock, TimeSteppers
using Oceananigans.Utils: prettysummary

using Breeze.Thermodynamics: ReferenceState
using Breeze.AtmosphereModels:
    AtmosphereModels,
    AtmosphereModel,
    compute_velocities!,
    compute_momentum_tendencies!,
    set_velocity!,
    set_momentum!

include("prescribed_velocity_fields.jl")

# Type alias for kinematic models
const KinematicModel = AtmosphereModel{<:PrescribedDynamics}

include("prescribed_velocity_time_stepping.jl")

end # module
