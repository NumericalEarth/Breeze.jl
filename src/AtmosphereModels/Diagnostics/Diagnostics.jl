module Diagnostics

export
    DryPotentialTemperature,
    VirtualPotentialTemperature,
    EquivalentPotentialTemperature,
    LiquidIcePotentialTemperature,
    StaticEnergy

using Breeze.Thermodynamics:
    Thermodynamics,
    MoistureMassFractions,
    dry_air_mass_fraction,
    total_specific_moisture,
    vapor_gas_constant,
    dry_air_gas_constant,
    liquid_latent_heat,
    saturation_vapor_pressure,
    PlanarLiquidSurface

using Breeze.AtmosphereModels: AtmosphereModel, compute_moisture_fractions

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using Oceananigans: Center, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: znode

# Flavor types for specific vs density-weighted diagnostics
struct Specific end
struct Density end

# Location aliases
const c = Center()

include("dry_potential_temperature.jl")
include("moist_potential_temperatures.jl")
include("static_energy.jl")

end # module

