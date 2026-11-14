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
    BoussinesqDynamics,
    AdiabaticExpansion,
    solve_for_anelastic_pressure!

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using Statistics: mean

using Oceananigans: Oceananigans, CenterField, XFaceField, YFaceField, ZFaceField
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: set!
using Oceananigans.DistributedComputations: Distributed, DistributedFourierTridiagonalPoissonSolver,
                                            child_architecture, partition_coordinate, reconstruct_global_grid
using Oceananigans.Grids: AbstractGrid, Periodic, ZDirection, inactive_cell, topology, with_number_type
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Models: boundary_condition_args
using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ, divᶜᶜᶜ, Δzᶜᶜᶜ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ
using Oceananigans.Solvers: Solvers, solve!, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver,
                            AbstractHomogeneousNeumannFormulation, poisson_eigenvalues
using Oceananigans.Utils: prettysummary, launch!

using Breeze.Thermodynamics: ReferenceState, BoussinesqReferenceState, ThermodynamicConstants,
                             MoistureMassFractions, mixture_gas_constant, surface_temperature_value,
                             MovingFrame, advance_frame!, synchronize_frame!, frame_summary, reference_value,
                             PlanarLiquidSurface, potential_temperature_from_temperature, supersaturation,
                             mixture_heat_capacity, liquid_latent_heat
using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, dynamics_pressure, fixed_pressure_state_density,
                               pressure_anomaly, kernel_time_step, specific_humidity, liquid_mass_fraction,
                               ice_mass_fraction, grid_moisture_fractions
using Breeze.PotentialTemperatureFormulations: LiquidIcePotentialTemperatureFormulation
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS

include("anelastic_dynamics.jl")
include("anelastic_pressure_solver.jl")
include("anelastic_buoyancy.jl")
include("adiabatic_expansion.jl")
include("boussinesq_dynamics.jl")

# Define type alias after AnelasticDynamics is defined
const AnelasticModel = AtmosphereModel{<:AnelasticDynamics}

include("anelastic_time_stepping.jl")

end # module
