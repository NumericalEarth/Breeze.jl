"""
    AnelasticFormulations

Submodule defining the anelastic dynamical formulation for atmosphere models.

The anelastic formulation filters acoustic waves by assuming density and pressure
are small perturbations from a dry, hydrostatic, adiabatic reference state.
"""
module AnelasticFormulations

export
    # Types
    AnelasticFormulation,
    AnelasticModel,
    # Solver function
    solve_for_anelastic_pressure!

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, XFaceField, YFaceField, ZFaceField, fields
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions, fill_halo_regions!
using Oceananigans.Grids: ZDirection, inactive_cell
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ, divᶜᶜᶜ, Δzᶜᶜᶜ, ℑzᵃᵃᶠ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ
using Oceananigans.Solvers: Solvers, solve!, FourierTridiagonalPoissonSolver, AbstractHomogeneousNeumannFormulation
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Utils: prettysummary, launch!

using Breeze.Thermodynamics: ReferenceState

# Import interface functions from parent module to extend them
import Breeze.AtmosphereModels:
    default_formulation,
    materialize_formulation,
    materialize_thermodynamics,
    materialize_momentum_and_velocities,
    formulation_pressure_solver,
    prognostic_field_names,
    additional_field_names,
    mean_pressure,
    pressure_anomaly,
    total_pressure,
    formulation_density

# Import AtmosphereModel for type alias
using Breeze.AtmosphereModels: AtmosphereModel

include("anelastic_formulation.jl")
include("anelastic_pressure_solver.jl")

# Define type alias for AnelasticModel
const AnelasticModel = AtmosphereModel{<:AnelasticFormulation}

include("anelastic_time_stepping.jl")

end # module
