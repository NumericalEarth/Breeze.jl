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
using Breeze.Thermodynamics: absolute_zero_latent_heat, mixture_heat_capacity, mixture_gas_constant, dry_air_gas_constant, MoistureMassFractions,
                             reject_renamed_surface_pressure,
                             vapor_gas_constant, ExnerReferenceState, temperature, LiquidIceDensityState, exner_function_at_temperature

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel, grid_moisture_fractions,
                               base_pressure, standard_pressure, thermodynamic_density,
                               thermodynamic_density_name, specific_prognostic_moisture
using Breeze.PotentialTemperatureFormulations: PotentialTemperatureFormulations, LiquidIcePotentialTemperatureFormulation

using ReactantCore: @trace

include("time_discretizations.jl")
include("compressible_dynamics.jl")
include("compressible_buoyancy.jl")

# Define type alias after CompressibleDynamics is defined
const CompressibleModel = AtmosphereModel{<:CompressibleDynamics}

include("compressible_density_tendency.jl")
include("compressible_time_stepping.jl")
include("acoustic_substepping.jl")
include("terrain_compressible_physics.jl")

# Switches read from the environment at load (not frozen at precompile time).
switch_from_env(name, default) = lowercase(get(ENV, name, string(default))) != "false"

function __init__()
    STORED_ENERGY_DENOMINATOR[] = lowercase(get(ENV, "BREEZE_STORED_ENERGY_DENOMINATOR", "true")) != "false"
    TERRAIN_FUSED_AUXILIARY[] = lowercase(get(ENV, "BREEZE_TERRAIN_FUSED_AUXILIARY", "true")) != "false"
    STORED_TERRAIN_PRESSURE_GRADIENTS[] = switch_from_env("BREEZE_STORED_TERRAIN_PRESSURE_GRADIENTS", true)
    STORED_TERRAIN_PRESSURE_GRADIENTS[] || @info "Stored terrain pressure gradients off (environment)"
    CARRIED_VERTICAL_RHS[] = switch_from_env("BREEZE_CARRIED_VERTICAL_RHS", true)
    CARRIED_VERTICAL_RHS[] || @info "Carried vertical rhs off (environment)"
    LOAD_STAGE_CONTRAVARIANT_MOMENTUM[] = switch_from_env("BREEZE_LOAD_STAGE_CONTRAVARIANT_MOMENTUM", true)
    LOAD_STAGE_CONTRAVARIANT_MOMENTUM[] || @info "Stage contravariant momentum load off (environment)"
    FOLDED_LAST_SUBSTEP_RECOVERY[] = switch_from_env("BREEZE_FOLDED_LAST_SUBSTEP_RECOVERY", true)
    FOLDED_LAST_SUBSTEP_RECOVERY[] || @info "Folded last-substep recovery off (environment)"
    NO_STEP_LINEARIZATION_REFRESH[] = switch_from_env("BREEZE_NO_STEP_LINEARIZATION_REFRESH", true)
    NO_STEP_LINEARIZATION_REFRESH[] || @info "Outer-step linearization refresh on (environment)"
    FOLDED_LINEARIZATION[] = switch_from_env("BREEZE_FOLDED_LINEARIZATION", true)
    FOLDED_LINEARIZATION[] || @info "Linearization folded into the auxiliary pass off (environment)"
    return nothing
end

end # module
