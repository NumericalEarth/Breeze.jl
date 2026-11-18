"""
    HydrostaticEquations

Module implementing dry hydrostatic dynamics for atmosphere models on a hybrid sigma-pressure
coordinate.

The vertical coordinate of the grid labels surfaces of constant ``η``, on which the pressure is

```math
p = A(η) + B(η) pˢ ,
```

with ``pˢ`` the surface pressure. Writing ``ρ = - g⁻¹ ∂p/∂z`` for the mass per unit area and unit
of the grid's vertical coordinate ``z``, and ``w = \\dot z`` for the velocity through the coordinate
surfaces, the equations are

```math
\\begin{aligned}
&\\text{Mass:} && ∂_t ρ + \\boldsymbol{∇ ⋅} (ρ \\boldsymbol{u}) + ∂_z (ρ w) = 0 \\\\
&\\text{Potential temperature:} && ∂_t (ρ θ) + \\boldsymbol{∇ ⋅} (ρ \\boldsymbol{u} θ) + ∂_z (ρ w θ) = 0 \\\\
&\\text{Momentum:} && ∂_t \\boldsymbol{u} + (ζ + f) \\boldsymbol{\\hat z × u} + \\boldsymbol{∇} K + w ∂_z \\boldsymbol{u} = - \\boldsymbol{∇} Φ - cᵖᵈ θ \\boldsymbol{∇} Π \\\\
&\\text{Hydrostatic balance:} && ∂Φ / ∂Π = - cᵖᵈ θ
\\end{aligned}
```

where ``\\boldsymbol{u}`` is the horizontal velocity, ``\\boldsymbol{∇}`` acts along coordinate
surfaces, ``K`` is the kinetic energy, ``Φ`` the geopotential and ``Π = (p / pˢᵗ)^κ`` the Exner
function. Because ``ρ`` is a known function of ``pˢ``, the column integral of the mass equation
is the only prognostic mass equation, and ``ρ w`` is diagnosed from the rest.
"""
module HydrostaticEquations

export
    HydrostaticDynamics,
    HydrostaticModel

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, Center, Face, CenterField, XFaceField, YFaceField, ZFaceField, Field,
                    instantiated_location
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Advection: VectorInvariantKEGradientEnergyConserving, Centered,
                              horizontal_advection_U, horizontal_advection_V,
                              bernoulli_head_U, bernoulli_head_V
using Oceananigans.Architectures: architecture, on_architecture, CPU
using Oceananigans.BoundaryConditions: BoundaryCondition, MultiRegionCommunication, fill_halo_regions!
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, FPlane, BetaPlane, HydrostaticSphericalCoriolis
using Oceananigans.Fields: Fields, ZeroField, set!, interior, compute!
using Oceananigans.Grids: AbstractGrid, Bounded, Flat, topology, znodes, minimum_xspacing, minimum_yspacing
using Oceananigans.Models: boundary_condition_args
using Oceananigans.MultiRegion: MultiRegion
using Oceananigans.Operators: Δzᶜᶜᶜ, Axᶠᶜᶜ, Ayᶜᶠᶜ, Azᶜᶜᶠ, Az⁻¹ᶜᶜᶜ, Az_qᶜᶜᶠ, V⁻¹ᶜᶜᶜ, Vᶜᶜᶜ, Vᶠᶜᶜ, Vᶜᶠᶜ,
                              δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ,
                              ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Utils: prettysummary, launch!, @apply_regionally

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel,
                               fixed_pressure_state_density, fill_horizontal_velocity_halos!,
                               horizontal_div_ρUc, div_ρUc, supports_prognostic_velocity,
                               copy_parent!, kernel_time_step, specific_prognostic_scalar_fields, flux_divergence,
                               owned_integral, owned_integral_value, thermodynamic_density_name,
                               radiation_flux_divergence, uses_level_pressure
using Breeze.PotentialTemperatureFormulations: LiquidIcePotentialTemperatureFormulation
using Breeze.Thermodynamics: dry_air_gas_constant
using Breeze.Utils: @regional, TYPEDSIGNATURES_WITH_DEFAULTS, TYPEDSIGNATURES_WITH_DEFAULTS_NO_RETURN

using DocStringExtensions: TYPEDEF

include("grid_scale_hyperdiffusion.jl")
include("vertical_diffusion.jl")
include("hydrostatic_dynamics.jl")

# Define type aliases after HydrostaticDynamics is defined
const HydrostaticModel = AtmosphereModel{<:HydrostaticDynamics}
const HydrostaticPotentialTemperatureModel =
    AtmosphereModel{<:HydrostaticDynamics, <:LiquidIcePotentialTemperatureFormulation}

include("hydrostatic_column_diagnostics.jl")
include("hydrostatic_tendencies.jl")
include("vertical_diffusion_step.jl")
include("hydrostatic_time_stepping.jl")
include("gravity_wave_substepping.jl")
include("hydrostatic_state_checks.jl")
include("hydrostatic_conservation_fixer.jl")

end # module
