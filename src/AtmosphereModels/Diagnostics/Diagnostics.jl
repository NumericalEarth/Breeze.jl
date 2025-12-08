module Diagnostics

export
    DryPotentialTemperature,
    VirtualPotentialTemperature,
    EquivalentPotentialTemperature,
    LiquidIcePotentialTemperature,
    StaticEnergy

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using Oceananigans: Center, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: znode

using ...Thermodynamics:
    Thermodynamics,
    MoistureMassFractions,
    dry_air_mass_fraction,
    total_specific_moisture,
    vapor_gas_constant,
    dry_air_gas_constant,
    saturation_vapor_pressure,
    PlanarLiquidSurface

# Import from parent module
using ..AtmosphereModels: AtmosphereModel, compute_moisture_fractions

# Flavor types for specific vs density-weighted diagnostics
struct Specific end
struct Density end

# Location aliases
const c = Center()

include("dry_potential_temperature.jl")
include("virtual_potential_temperature.jl")
include("equivalent_potential_temperature.jl")
include("liquid_ice_potential_temperature.jl")
include("static_energy.jl")

end # module

