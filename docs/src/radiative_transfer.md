# Radiative Transfer

Breeze.jl integrates with [RRTMGP.jl](https://github.com/NumericalEarth/RRTMGP.jl) to provide radiative transfer capabilities for atmospheric simulations. The radiative transfer model computes longwave and shortwave radiative fluxes and heating rates, which are then incorporated into the moist static energy tendency equation.

## Overview

The radiative transfer implementation in Breeze uses a column-based approach, where each horizontal column of the 3D grid is treated independently. This allows efficient computation of radiative fluxes using RRTMGP's optimized column radiation solver.

## Basic Usage

To use radiative transfer in a Breeze simulation, create a `RadiativeTransferModel` and pass it to the `AtmosphereModel` constructor:

```julia
using Breeze
using Oceananigans

# Create grid
grid = RectilinearGrid(size=(32, 32, 64), x=(0, 10_000), y=(0, 10_000), z=(0, 20_000))

# Create radiative transfer model
rtm = RadiativeTransferModel(
    grid;
    surface_emissivity = 0.98,
    surface_albedo_direct = 0.1,
    surface_albedo_diffuse = 0.1,
    cos_zenith = 0.5,
    toa_solar_flux = 1360.0,
    toa_longwave_flux = 0.0
)

# Create atmosphere model with radiative transfer
model = AtmosphereModel(
    grid;
    radiative_transfer = rtm
)
```

## Column Model Example

Here we demonstrate a simple column model calculation of radiative fluxes. This example sets up a single-column atmosphere and computes radiative heating rates.

```julia
using Breeze
using Oceananigans
using CairoMakie

# Create a single-column grid with 64 vertical levels
grid = RectilinearGrid(; size=64, z=(0, 20_000), topology = (Flat, Flat, Bounded))

# Thermodynamic constants
thermo = ThermodynamicConstants()

# Reference state
reference_state = ReferenceState(grid, thermo, 
    base_pressure = 101325.0,
    potential_temperature = 288.0
)

formulation = AnelasticFormulation(reference_state)

# Create radiative transfer model
rtm = RadiativeTransferModel(
    grid;
    surface_emissivity = 0.98,
    surface_albedo_direct = 0.1,
    surface_albedo_diffuse = 0.1,
    cos_zenith = 0.5,  # Solar zenith angle cosine
    toa_solar_flux = 1360.0,  # Top-of-atmosphere solar flux [W/m²]
    toa_longwave_flux = 0.0
)

# Create atmosphere model
model = AtmosphereModel(
    grid;
    thermodynamics = thermo,
    formulation = formulation,
    radiative_transfer = rtm
)

# Initialize with a temperature profile
# Simple linear temperature profile decreasing with height
set!(model.temperature, (x, y, z) -> 288.0 - 0.0065 * z)

# Update model state
update_state!(model)

# Update radiative fluxes
Breeze.AtmosphereModels._update_radiative_fluxes!(rtm, model)

# Extract heating rates
nz = size(grid, 3)
heating_rate = zeros(Float64, nz)

using Oceananigans: interior
using Oceananigans.Grids: znode
using GPUArraysCore: @allowscalar

@allowscalar begin
    ρᵣ = model.formulation.reference_state.density
    for k in 1:nz
        hr = Breeze.AtmosphereModels._radiative_heating_rate(
            1, 1, k, grid, rtm, 
            ρᵣ,
            thermo
        )
        # Convert from energy density tendency to heating rate per unit mass
        heating_rate[k] = hr / ρᵣ[1, 1, k]
    end
end

# Extract height levels for plotting
z_lev = [znode(1, 1, k, grid, Center(), Center(), Center()) for k in 1:nz]

# Plot heating rate profile
using CairoMakie

fig = Figure(resolution = (600, 400))
ax = Axis(fig[1, 1], 
    xlabel = "Heating Rate [K/day]",
    ylabel = "Height [m]",
    title = "Radiative Heating Rate Profile"
)

# Convert to K/day
cp = thermo.dry_air.heat_capacity
seconds_per_day = 86400.0
heating_rate_k_per_day = heating_rate .* seconds_per_day ./ cp

lines!(ax, heating_rate_k_per_day, z_lev, linewidth = 2)
fig
```

## Surface Properties

The `RadiativeTransferModel` requires several surface properties that are not part of the `AtmosphereModel`:

- **Surface temperature**: Temperature at the surface (can be extracted from model or specified)
- **Surface emissivity**: Longwave emissivity of the surface (typically 0.95-0.99)
- **Surface albedo (direct)**: Albedo for direct solar radiation
- **Surface albedo (diffuse)**: Albedo for diffuse solar radiation
- **Cosine of solar zenith angle**: Determines solar insolation
- **TOA solar flux**: Top-of-atmosphere solar flux [W/m²]
- **TOA longwave flux**: Top-of-atmosphere longwave flux [W/m²] (usually 0)

These properties are stored in the `RadiativeTransferModel` and can be updated as needed during the simulation.

## Reference Pressure

Breeze uses the **reference pressure** from the anelastic formulation for radiative transfer calculations, not the total pressure. This is consistent with the anelastic approximation where pressure perturbations are small compared to the reference state.

## Integration with Dynamics

Radiative heating is automatically added to the moist static energy tendency equation when a `RadiativeTransferModel` is provided. The heating rate is computed from flux differences:

```math
\frac{\partial (\rho e)}{\partial t} = \ldots + \rho_r \frac{g}{c_p} \frac{F_{k+1} - F_k}{\Delta p}
```

where ``F_k`` is the net radiative flux at level ``k``, ``\Delta p`` is the pressure difference across the layer, ``g`` is gravitational acceleration, ``c_p`` is specific heat capacity, and ``\rho_r`` is the reference density.

## Gray Atmosphere Model

The current implementation uses a gray atmosphere radiation model, which treats the atmosphere as having a single effective absorption coefficient. This is suitable for initial testing and development. Future versions will support full spectral radiation using RRTMGP's band-by-band calculations.

## Architecture Support

The radiative transfer implementation supports both CPU and GPU architectures. The grid conversion utilities automatically handle the conversion between Oceananigans' 3D grid format and RRTMGP's column-based format, including proper handling of CPU and GPU arrays.

