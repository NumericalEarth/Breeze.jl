"""
    PrescribedVelocityFieldsDynamics

Module implementing kinematic dynamics for atmosphere models.

Kinematic dynamics prescribes the velocity field rather than solving for it,
enabling isolated testing of microphysics, thermodynamics, and other physics
without the complexity of solving the momentum equations.

This is analogous to the `kin1d` driver in P3-microphysics.
"""
module PrescribedVelocityFieldsDynamics

export
    PrescribedVelocityFields,
    KinematicModel

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt

using Oceananigans: Oceananigans, CenterField, XFaceField, YFaceField, ZFaceField, fields
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions, fill_halo_regions!
using Oceananigans.Fields: FunctionField, ZeroField, field
using Oceananigans.Grids: Face, Center
using Oceananigans.TimeSteppers: Clock, TimeSteppers
using Oceananigans.Utils: prettysummary, launch!

using Breeze.Thermodynamics: ReferenceState
using Breeze.AtmosphereModels:
    AtmosphereModels,
    AtmosphereModel,
    has_prescribed_velocities

include("prescribed_velocity_fields.jl")

# Define type alias after PrescribedVelocityFields is defined
const KinematicModel = AtmosphereModel{<:PrescribedVelocityFields}

include("prescribed_velocity_time_stepping.jl")

end # module

