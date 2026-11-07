"""
    AnelasticEquations

Module implementing anelastic dynamics for atmosphere models.

The anelastic approximation filters acoustic waves by assuming density and pressure
are small perturbations from a dry, hydrostatic, adiabatic reference state.
The key constraint is that mass flux divergence vanishes:

```math
\\boldsymbol{\\nabla} ⋅ (ρᵣ \\boldsymbol{u}) = 0
```
"""
module AnelasticEquations

export
    AnelasticDynamics,
    AnelasticModel,
    solve_for_anelastic_pressure!

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, XFaceField, YFaceField, ZFaceField
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: set!
using Oceananigans.Grids: AbstractGrid, ZDirection, inactive_cell, with_number_type
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Models: boundary_condition_args, enforce_net_zero_transport!,
                           initialize_boundary_transport
using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ, divᶜᶜᶜ, Δzᶜᶜᶜ, ℑzᵃᵃᶠ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ
using Oceananigans.Solvers: Solvers, solve!, FourierTridiagonalPoissonSolver, AbstractHomogeneousNeumannFormulation
using Oceananigans.Utils: prettysummary, launch!

using Breeze.Thermodynamics: ReferenceState, MoistureMassFractions, mixture_gas_constant,
                             surface_temperature_value
using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, dynamics_pressure, fixed_pressure_state_density,
                                pressure_anomaly,
                                kernel_time_step

# Import microphysics interface for buoyancy computation
using Breeze.AtmosphereModels: grid_moisture_fractions, enforce_lateral_mass_balance!

include("anelastic_dynamics.jl")
include("anelastic_pressure_solver.jl")
include("anelastic_buoyancy.jl")
include("lateral_mass_balance.jl")

# Define type alias after AnelasticDynamics is defined
const AnelasticModel = AtmosphereModel{<:AnelasticDynamics}

include("anelastic_time_stepping.jl")

end # module
