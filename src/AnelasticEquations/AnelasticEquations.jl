"""
    AnelasticEquations

Module implementing anelastic dynamics for atmosphere models.

The anelastic approximation filters acoustic waves by assuming density and pressure
are small perturbations from a dry, hydrostatic, adiabatic reference state.
The key constraint is that mass flux divergence vanishes: `∇⋅(ρᵣ u) = 0`.
"""
module AnelasticEquations

export
    AnelasticDynamics,
    AnelasticModel,
    solve_for_anelastic_pressure!

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, XFaceField, YFaceField, ZFaceField, fields
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions, fill_halo_regions!
using Oceananigans.Fields: set!
using Oceananigans.Grids: ZDirection, inactive_cell
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ, divᶜᶜᶜ, Δzᶜᶜᶜ, ℑzᵃᵃᶠ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ
using Oceananigans.Solvers: Solvers, solve!, FourierTridiagonalPoissonSolver, AbstractHomogeneousNeumannFormulation
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Utils: prettysummary, launch!

using Breeze.Thermodynamics: ReferenceState, mixture_gas_constant

using Breeze.AtmosphereModels: AtmosphereModel

# Import interface functions to extend
import Breeze.AtmosphereModels:
    default_dynamics,
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
    initialize_model_thermodynamics!

# Import microphysics interface for buoyancy computation
import Breeze.AtmosphereModels: compute_moisture_fractions

include("anelastic_dynamics.jl")
include("anelastic_pressure_solver.jl")
include("anelastic_buoyancy.jl")

# Define type alias after AnelasticDynamics is defined
const AnelasticModel = AtmosphereModel{<:AnelasticDynamics}

include("anelastic_time_stepping.jl")

end # module

