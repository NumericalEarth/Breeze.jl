module Thermodynamics

export ThermodynamicConstants, ReferenceState, ExnerReferenceState, compute_reference_state!, compute_hydrostatic_reference!, IdealGas,
       CondensedPhase,
       ClausiusClapeyron, ClausiusClapeyronThermodynamicConstants,
       TetensFormula, TetensFormulaThermodynamicConstants,
       MoistureMassFractions, MoistureMixingRatio,
       vapor_gas_constant, dry_air_gas_constant,
       mixture_gas_constant, mixture_heat_capacity,
       total_mixing_ratio, total_specific_moisture,
       liquid_latent_heat, ice_latent_heat,
       saturation_vapor_pressure, saturation_specific_humidity, supersaturation,
       equilibrium_saturation_specific_humidity, adjustment_saturation_specific_humidity,
       dewpoint_temperature,
       vapor_pressure, relative_humidity,
       adiabatic_hydrostatic_pressure, adiabatic_hydrostatic_density, surface_density,
       temperature_from_potential_temperature, temperature, with_temperature, with_moisture,
       PlanarLiquidSurface, PlanarIceSurface, PlanarMixedPhaseSurface,
       # Phase equilibrium types
       AbstractPhaseEquilibrium, WarmPhaseEquilibrium, MixedPhaseEquilibrium,
       equilibrated_surface

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans: Oceananigans

include("thermodynamics_constants.jl")
include("vapor_saturation.jl")
include("clausius_clapeyron.jl")
include("tetens_formula.jl")
include("reference_states.jl")
include("dynamic_states.jl")

end # module
