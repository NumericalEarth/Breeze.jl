"""
    KinematicDriver

Module implementing kinematic dynamics for atmosphere models.

Kinematic dynamics prescribes the velocity field rather than solving for it,
enabling isolated testing of microphysics, thermodynamics, and other physics
without the complexity of solving the momentum equations.

This is analogous to the `kin1d` driver in P3-microphysics.
"""
module KinematicDriver

export
    PrescribedDensity,
    PrescribedDynamics,
    KinematicModel

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Adapt: Adapt, adapt

using Oceananigans: Oceananigans, CenterField, XFaceField, YFaceField, ZFaceField
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, ValueBoundaryCondition, fill_halo_regions!
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: AbstractField, FunctionField, ZeroField, field, set!
using Oceananigans.Grids: Face, Center
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.TimeSteppers: Clock, TimeSteppers
using Oceananigans.Utils: launch!, prettysummary

using KernelAbstractions: @kernel, @index

# Import PrescribedVelocityFields from Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedVelocityFields

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, dynamics_density
using Breeze.Thermodynamics: ReferenceState

include("prescribed_dynamics.jl")

# Type alias for kinematic models
const KinematicModel = AtmosphereModel{<:PrescribedDynamics}

include("kinematic_driver_time_stepping.jl")

end # module
