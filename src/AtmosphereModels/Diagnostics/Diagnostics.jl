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
    azimuthal_mean,
    azimuthal_mean!,
    # Interface functions extended by Microphysics
    microphysics_phase_equilibrium

using DocStringExtensions: TYPEDSIGNATURES

using Breeze.Solvers: SecantSolver

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
    PlanarLiquidSurface,
    # Phase equilibrium types
    WarmPhaseEquilibrium,
    equilibrated_surface

using Breeze.AtmosphereModels: AtmosphereModel, dynamics_pressure, humidity_density, total_density,
                               grid_moisture_fractions, specific_prognostic_moisture

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans, Center
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Field, CenterField
using Oceananigans.Grids: znode, znodes, xnode, ynode, RectilinearGrid, Bounded, Flat, Face
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ
using Oceananigans.Utils: launch!

# Flavor types for specific vs density-weighted diagnostics
struct Specific end
struct Density end

# Location aliases
const c = Center()

include("potential_temperatures.jl")
include("static_energy.jl")
include("saturation_specific_humidity.jl")
include("dewpoint_temperature.jl")
include("azimuthal_mean.jl")

end # module
