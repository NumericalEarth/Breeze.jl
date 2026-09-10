"""
    CompressibleEquations

Module implementing fully compressible dynamics for atmosphere models.

The compressible formulation directly time-steps the dry-air density as a prognostic
variable and computes pressure from the ideal gas law. This formulation does not filter
acoustic waves, so explicit time-stepping with small time steps (or acoustic
substepping) is required.

The fully compressible Euler equations in conservation form, with dry air as the mass
and momentum carrier, are:

```math
\\begin{aligned}
&\\text{Dry mass:} && \\partial_t \\rho^d + \\boldsymbol{\\nabla \\cdot} (\\rho^d \\boldsymbol{u}) = 0 \\\\
&\\text{Momentum:} && \\partial_t (\\rho^d \\boldsymbol{u}) + \\boldsymbol{\\nabla \\cdot} (\\rho^d \\boldsymbol{u} \\boldsymbol{u}) = - q^d \\boldsymbol{\\nabla} p - \\rho^d g \\hat{\\boldsymbol{z}} + \\rho^d \\boldsymbol{f} + \\boldsymbol{\\nabla \\cdot \\mathcal{T}}
\\end{aligned}
```

The mixture momentum balance contains the full pressure gradient ``-∇p`` and the total
gravitational force per unit volume ``-ρ g \\hat{\\boldsymbol{z}}``, where ``ρ = ρ^d + ρ^t`` is
the total mixture density. Multiplying this balance by the dry-air mass fraction
``q^d = ρ^d / ρ = 1 - q^t`` and using dry-air continuity gives the conservative equation above
for ``ρ^d \\boldsymbol{u}``. Its pressure and gravitational contributions are therefore
``-q^d ∇p`` and ``-ρ^d g \\hat{\\boldsymbol{z}}``.

At each momentum face, ``q^d`` is computed as the ratio of the interpolated dry-air and total
densities by `AtmosphereModels.dynamics_mass_fractionᶠᶜᶜ` and its counterparts.

Pressure is computed from the ideal gas law:
```math
p = \\rho R^m T
```
where ``R^m`` is the mixture gas constant and ``\\rho`` is the total density.
"""
module CompressibleEquations

export
    CompressibleDynamics,
    CompressibleModel,
    AcousticSubstepper,
    SplitExplicitTimeDiscretization,
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
    ExplicitTimeStepping,
    prepare_acoustic_cache!,
    freeze_linearization_state!,
    acoustic_rk3_substep_loop!

using DocStringExtensions: TYPEDEF, TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, Center, Face, CenterField, XFaceField, YFaceField, ZFaceField, prognostic_fields
using Oceananigans.Grids: rnode, znode
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ,
                                ∂zᶜᶜᶜ, ∂zᶜᶜᶠ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.Utils: prettysummary, launch!, KernelParameters

using Breeze.Solvers: NewtonSolver
using Breeze.Utils: safe_divide
using Breeze.Thermodynamics: mixture_gas_constant, dry_air_gas_constant,
                             vapor_gas_constant, ExnerReferenceState, temperature, LiquidIceDensityState

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, grid_moisture_fractions,
                               surface_pressure, standard_pressure, thermodynamic_density,
                               thermodynamic_density_name, specific_prognostic_moisture
using Breeze.PotentialTemperatureFormulations: LiquidIcePotentialTemperatureFormulation

include("time_discretizations.jl")
include("compressible_dynamics.jl")
include("compressible_buoyancy.jl")

# Define type alias after CompressibleDynamics is defined
const CompressibleModel = AtmosphereModel{<:CompressibleDynamics}

include("compressible_density_tendency.jl")
include("compressible_time_stepping.jl")
include("acoustic_substepping.jl")
include("terrain_compressible_physics.jl")

end # module
