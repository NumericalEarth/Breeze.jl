module Diagnostics

export
    PotentialTemperature,
    VirtualPotentialTemperature,
    EquivalentPotentialTemperature,
    StabilityEquivalentPotentialTemperature,
    LiquidIcePotentialTemperature,
    StaticEnergy,
    SaturationSpecificHumidity,
    SaturationSpecificHumidityField,
    DewpointTemperature,
    DewpointTemperatureField,
    equilibrium_saturation_specific_humidity,
    # Interface functions extended by Microphysics
    microphysics_phase_equilibrium

using DocStringExtensions: TYPEDSIGNATURES

using Breeze.Thermodynamics:
    Thermodynamics,
    vapor_gas_constant,
    dry_air_gas_constant,
    liquid_latent_heat,
    mixture_gas_constant,
    mixture_heat_capacity,
    relative_humidity,
    saturation_specific_humidity,
    saturation_vapor_pressure,
    equilibrium_saturation_specific_humidity,
    density,
    PlanarLiquidSurface,
    # Phase equilibrium types
    WarmPhaseEquilibrium,
    equilibrated_surface

using Breeze.AtmosphereModels: AtmosphereModel, grid_moisture_fractions, specific_prognostic_moisture

using Adapt: Adapt, adapt
using Oceananigans: Oceananigans, Center
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field
using Oceananigans.Grids: znode

# Flavor types for specific vs density-weighted diagnostics
struct Specific end
struct Density end

# Location aliases
const c = Center()

include("potential_temperatures.jl")
include("static_energy.jl")
include("saturation_specific_humidity.jl")
include("dewpoint_temperature.jl")

end # module
