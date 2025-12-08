module Diagnostics

export
    VirtualPotentialTemperature,
    EquivalentPotentialTemperature,
    StabilityEquivalentPotentialTemperature,
    LiquidIcePotentialTemperature,
    StaticEnergy

using Breeze.Thermodynamics:
    Thermodynamics,
    vapor_gas_constant,
    dry_air_gas_constant,
    liquid_latent_heat,
    mixture_gas_constant,
    mixture_heat_capacity,
    relative_humidity,
    PlanarLiquidSurface

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

include("potential_temperatures.jl")
include("static_energy.jl")

end # module

