module Diagnostics

export
    DryPotentialTemperature,
    VirtualPotentialTemperature,
    EquivalentPotentialTemperature,
    LiquidIcePotentialTemperature,
    StaticEnergy

using Breeze.Thermodynamics:
    Thermodynamics,
    vapor_gas_constant,
    dry_air_gas_constant,
    liquid_latent_heat

using Breeze.AtmosphereModels: AtmosphereModel, compute_moisture_fractions

using Adapt: Adapt, adapt
using Oceananigans: Center
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

