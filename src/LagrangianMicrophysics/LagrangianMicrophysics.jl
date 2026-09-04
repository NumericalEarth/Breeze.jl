module LagrangianMicrophysics

export Droplet,
       DropletDynamics,
       droplet_tracked_fields,
       interpolate_to_droplets!,
       equilibrium_supersaturation,
       critical_diameter,
       critical_supersaturation,
       equilibrium_diameter,
       growth_coefficient,
       implicit_growth_step,
       ambient_supersaturation,
       activated,
       activated_fraction

using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, PlanarLiquidSurface,
                             saturation_vapor_pressure, liquid_latent_heat, dry_air_gas_constant,
                             density, supersaturation
using Breeze.AtmosphereModels: specific_prognostic_moisture, dynamics_pressure

using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: interpolate, location
using Oceananigans.Grids: Center, Bounded, XFlatGrid, YFlatGrid, topology, xnode, ynode, rnode
using Oceananigans.Models.LagrangianParticleTracking: flattened_node
using Oceananigans.Utils: launch!, KernelParameters, instantiate

using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index

include("kappa_kohler.jl")
include("droplets.jl")

end # module
