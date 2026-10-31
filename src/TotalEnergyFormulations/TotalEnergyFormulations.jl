"""
    TotalEnergyFormulations

Submodule defining the total-energy thermodynamic formulation for compressible atmosphere models.

`TotalEnergyFormulation` evolves the total energy density `ρE`: internal, kinetic and potential
energy per unit volume, advanced in conservation form so that the energy budget closes
structurally (Satoh 2003).
"""
module TotalEnergyFormulations

export TotalEnergyFormulation

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, Center, Face, znode, prognostic_fields
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.Fields: Field, set!
using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, Axᶠᶜᶜ, Ayᶜᶠᶜ, Azᶜᶜᶠ
using Oceananigans.Utils: prettysummary, launch!
using Oceananigans.TimeSteppers: ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization, time_discretization
using Oceananigans.TurbulenceClosures: _diffusive_flux_z

using Breeze.Thermodynamics: InternalEnergyDensityState, LiquidIceDensityState,
    VaporEquivalentExponent, temperature, with_temperature,
    mixture_heat_capacity, mixture_gas_constant, mixture_isochoric_heat_capacity,
    dry_air_gas_constant, vapor_gas_constant,
    liquid_referenced_latent_energy, absolute_zero_latent_heat,
    saturation_specific_humidity

using Breeze.Solvers: NewtonSolver

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, AtmosphereModelBuoyancy,
    dynamics_density, total_density, standard_pressure, dynamics_prognostic_fields,
    grid_moisture_fractions, specific_prognostic_moisture, div_ρUc, ∇_dot_Jᶜ,
    radiation_flux_divergence, materialize_formulation,
    explicit_condensate_mass_fluxes, implicit_condensate_mass_fluxes, implicit_sedimentation,
    kernel_time_step, compute_velocities!, transport_velocities, moisture_prognostic_name,
    condensate_field_names, condensate_liquid_fraction, specific_field_name, field_advection_scheme,
    closure_scalar_index

using Breeze.Advection: tracer_mass_flux_x, tracer_mass_flux_y, tracer_mass_flux_z

using Breeze.TurbulenceClosures: Jᶜx, Jᶜy, Jᶜz

using Breeze.AtmosphereModels.Diagnostics: Diagnostics

using Breeze.BoundaryConditions: energy_boundary_conditions

using Breeze.CompressibleEquations: CompressibleEquations, CompressibleDynamics

using Breeze.Microphysics: Microphysics

# The lowercase c is a singleton instance of Center
const c = Center()

include("total_energy_formulation.jl")
include("total_energy_tendency.jl")
include("total_energy_compressible.jl")

end # module
