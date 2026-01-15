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
using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, mean_pressure, pressure_anomaly

# Import microphysics interface for buoyancy computation
using Breeze.AtmosphereModels: grid_compute_moisture_fractions

include("anelastic_dynamics.jl")
include("anelastic_pressure_solver.jl")
include("anelastic_buoyancy.jl")

# Define type alias after AnelasticDynamics is defined
const AnelasticModel = AtmosphereModel{<:AnelasticDynamics}

include("anelastic_time_stepping.jl")

end # module
