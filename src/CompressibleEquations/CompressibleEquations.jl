"""
    CompressibleEquations

Module implementing fully compressible dynamics for atmosphere models.

The compressible formulation directly time-steps density as a prognostic variable
and computes pressure from the ideal gas law. This formulation does not filter
acoustic waves, so explicit time-stepping with small time steps (or acoustic
substepping) is required.

The fully compressible Euler equations in conservation form are:

```math
\\begin{aligned}
&\\text{Mass:} && \\partial_t \\rho + \\boldsymbol{\\nabla \\cdot} (\\rho \\boldsymbol{u}) = 0 \\\\
&\\text{Momentum:} && \\partial_t (\\rho \\boldsymbol{u}) + \\boldsymbol{\\nabla \\cdot} (\\rho \\boldsymbol{u} \\boldsymbol{u}) + \\boldsymbol{\\nabla} p = -\\rho g \\hat{\\boldsymbol{z}} + \\rho \\boldsymbol{f} + \\boldsymbol{\\nabla \\cdot \\mathcal{T}}
\\end{aligned}
```

Pressure is computed from the ideal gas law:
```math
p = \\rho R^m T
```
where ``R^m`` is the mixture gas constant.
"""
module CompressibleEquations

export
    CompressibleDynamics,
    CompressibleModel,
    AcousticSubstepper,
    SplitExplicitTimeDiscretization,
    AcousticPressureCoefficient,
    FrozenCompositionPressureCoefficient,
    PhaseEquilibriumPressureCoefficient,
    AcousticOuterScheme,
    WickerSkamarock3,
    stage_fractions,
    AcousticSubstepDistribution,
    ProportionalSubsteps,
    ConstantSubstepSize,
    MonolithicFirstStage,
    AcousticDampingStrategy,
    NoDivergenceDamping,
    ThermalDivergenceDamping,
    DirectDivergenceDamping,
    UpperSponge,
    AbstractRamp,
    LinearRamp,
    CubicRamp,
    Sin2Ramp,
    GaussianRamp,
    ExplicitTimeStepping,
    prepare_acoustic_cache!,
    freeze_linearization_state!,
    acoustic_rk3_substep_loop!

using DocStringExtensions: TYPEDEF
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, Center, Face, CenterField, XFaceField, YFaceField, ZFaceField, prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: rnode, znode
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ,
                              ∂zᶜᶜᶜ, ∂zᶜᶜᶠ, divᶜᶜᶜ
using Oceananigans.Utils: prettysummary, launch!, KernelParameters

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, grid_moisture_fractions,
                               base_pressure, standard_pressure, thermodynamic_density,
                               thermodynamic_density_name, specific_prognostic_moisture
using Breeze.PotentialTemperatureFormulations: PotentialTemperatureFormulations, LiquidIcePotentialTemperatureFormulation
using Breeze.Solvers: NewtonSolver
using Breeze.RoundingTreatments: RoundToNearest, rounded_add, rounding_key
using Breeze.Thermodynamics: absolute_zero_latent_heat, mixture_heat_capacity, mixture_gas_constant, dry_air_gas_constant, MoistureMassFractions,
                             vapor_equivalent_heat_capacity, condensate_latent_energy, exner_exponent,
                             reject_renamed_surface_pressure, specific_heat_difference,
                             saturation_specific_humidity, equilibrated_surface,
                             vapor_gas_constant, ExnerReferenceState, temperature, temperature_and_sensitivity,
                             LiquidIceDensityState

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, grid_moisture_fractions,
                               base_pressure, standard_pressure, thermodynamic_density,
                               thermodynamic_density_name, specific_prognostic_moisture,
                               precipitating_mass_fraction
using Breeze.AtmosphereModels.Diagnostics: microphysics_phase_equilibrium
using Breeze.PotentialTemperatureFormulations: PotentialTemperatureFormulations, LiquidIcePotentialTemperatureFormulation
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS

using ReactantCore: @trace

include("time_discretizations.jl")
include("compressible_dynamics.jl")
include("compressible_buoyancy.jl")

# Define type alias after CompressibleDynamics is defined
const CompressibleModel = AtmosphereModel{<:CompressibleDynamics}

include("compressible_density_tendency.jl")
include("compressible_time_stepping.jl")
include("stored_tridiagonal_factors.jl")
include("acoustic_substepping.jl")
include("terrain_compressible_physics.jl")

end # module
