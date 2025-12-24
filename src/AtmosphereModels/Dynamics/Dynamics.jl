"""
    Dynamics

Submodule defining the dynamical formulations for atmosphere models.

Currently supports:
- `AnelasticDynamics`: filters acoustic waves by assuming density and pressure
  are small perturbations from a dry, hydrostatic, adiabatic reference state.

Future implementations may include:
- `CompressibleDynamics`: fully compressible dynamics with prognostic density and pressure.
"""
module Dynamics

export
    # Types
    AnelasticDynamics,
    # Interface functions
    default_dynamics,
    materialize_dynamics,
    materialize_momentum_and_velocities,
    dynamics_pressure_solver,
    dynamics_density,
    dynamics_pressure,
    mean_pressure,
    pressure_anomaly,
    total_pressure,
    # Solver function
    solve_for_anelastic_pressure!

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, XFaceField, YFaceField, ZFaceField
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans.Grids: ZDirection, inactive_cell
using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ, divᶜᶜᶜ, Δzᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.Solvers: Solvers, solve!, FourierTridiagonalPoissonSolver, AbstractHomogeneousNeumannFormulation
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Utils: prettysummary, launch!

using Breeze.Thermodynamics: ReferenceState

include("dynamics_interface.jl")
include("anelastic_dynamics.jl")
include("anelastic_pressure_solver.jl")

end # module
