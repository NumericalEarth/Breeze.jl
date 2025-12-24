# Radiative Transfer

Breeze.jl integrates with [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl) to provide radiative transfer capabilities for atmospheric simulations. The radiative transfer model computes longwave and shortwave radiative fluxes, which can be incorporated into energy tendency equations.

## Gray Atmosphere Radiation

The simplest radiative transfer option is gray atmosphere radiation, which uses the optical thickness parameterization from [Schneider2004](@citet) and [OGormanSchneider2008](@citet). This approximation treats the atmosphere as having a single effective absorption coefficient rather than computing full spectral radiation.

### Basic Usage

To use gray radiation in a Breeze simulation, create a [`RadiativeTransferModel`](@ref) model with the `:gray` optics flavor and pass it to the [`AtmosphereModel`](@ref) constructor:

```@example
using Breeze
using Oceananigans.Units
using Dates
using RRTMGP

Nz = 64
λ, φ = -70.9, 42.5  # longitude, latitude
grid = RectilinearGrid(size=Nz, x=λ, y=φ, z=(0, 20kilometers),
                       topology=(Flat, Flat, Bounded))

# Thermodynamic setup
surface_temperature = 300
constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 101325,
                                 potential_temperature = surface_temperature)

formulation = AnelasticFormulation(reference_state)

# Create gray radiation model
radiation = RadiativeTransferModel(grid, :gray, constants;
                                   surface_temperature,
                                   surface_emissivity = 0.98,
                                   surface_albedo = 0.1,
                                   solar_constant = 1361) # W/m²

# Create atmosphere model with DateTime clock for solar position
clock = Clock(time=DateTime(2024, 9, 27, 16, 0, 0))
model = AtmosphereModel(grid; clock, formulation, radiation)
```

When a `DateTime` clock is used, the solar zenith angle is computed automatically from the time and grid location (longitude and latitude).

### Gray Radiation Model

The [`RadiativeTransferModel`](@ref) model computes:

- **Longwave radiation**: Both upwelling and downwelling thermal radiation using RRTMGP's two-stream solver
- **Shortwave radiation**: Direct beam solar radiation

The gray atmosphere optical thickness for longwave follows the parameterization by [OGormanSchneider2008](@citet)š

```math
τ_{lw} = α \frac{Δp}{p_0} \left[ f_l + 4 (1 - f_l) \left(\frac{p}{p_0}\right)^3 \right] \left[ τ_e + (τ_p - τ_e) \sin^2 φ \right]
```

where ``φ`` is latitude and ``α``, ``f_l``, ``τ_e``, ``τ_p`` are empirical parameters.

For shortwave:
```math
τ_{sw} = 2 τ_0 \frac{Δp}{p_0} \frac{p}{p_0}
```

where ``τ_0 = 0.22`` is the shortwave optical depth parameter.

The above two expressions are identical to those in the [RRTMGP documentation](https://clima.github.io/RRTMGP.jl/latest/Optics/#Gray-atmosphere-optics).

### Radiative Fluxes

After running [`set!`](@ref), the radiative fluxes are available from the radiation model:

```julia
# Longwave fluxes (ZFaceFields)
ℐ_lw_up = radiation.upwelling_longwave_flux
ℐ_lw_dn = radiation.downwelling_longwave_flux

# Shortwave flux (direct beam only for non-scattering solver)
ℐ_sw = radiation.downwelling_shortwave_flux
```

!!! note "Shortwave Radiation"
    The gray atmosphere uses a non-scattering shortwave approximation, so only
    the direct beam flux is computed. There is no diffuse shortwave or upwelling
    shortwave in this model.

### Solar Zenith Angle

When using a `DateTime` clock, the solar zenith angle is computed from:
- Grid location (longitude from `x`, latitude from `y` for single-column grids)
- Date and time from `model.clock.time`

The calculation accounts for:
- Day of year (for solar declination)
- Hour angle (based on solar time)
- Latitude (for observer position)

## Clear-sky Full-spectrum Radiation

For more accurate radiative transfer calculations, use the `:clear_sky` optics flavor which computes full-spectrum gas optics using RRTMGP's lookup tables:

```julia
using NCDatasets  # Required for RRTMGP lookup tables

radiation = RadiativeTransferModel(grid, :clear_sky, constants;
                                   surface_temperature,
                                   surface_emissivity = 0.98,
                                   surface_albedo = 0.1,
                                   background_atmosphere = BackgroundAtmosphere(CO₂ = 400e-6))
```

The [`BackgroundAtmosphere`](@ref) struct specifies volume mixing ratios for radiatively active gases (CO₂, CH₄, N₂O, O₃, etc.). Water vapor is computed from the model's prognostic moisture field.

## Surface Properties

The [`RadiativeTransferModel`](@ref) model requires surface properties:

| Property | Description | Typical Values |
|----------|-------------|----------------|
| `surface_temperature` | Temperature at the surface [K] | 280-310 |
| `surface_emissivity` | Longwave emissivity (0-1) | 0.95-0.99 |
| `surface_albedo` | Shortwave albedo (0-1) | 0.1-0.3 |
| `solar_constant` | TOA solar flux [W/m²] | 1361 |

## Integration with dynamics

Radiative fluxes can be used to compute heating rates for the energy equation. The radiative heating rate is computed from flux divergence:

```math
F_{\mathscr{I}} = -\frac{1}{\rho cᵖᵐ} \frac{\partial \mathscr{I}_{net}}{\partial z}
```

where ``\mathscr{I}_{net}`` is the net radiative flux (upwelling minus downwelling), ``cᵖᵐ`` is the mixture heat capacity, and ``F_{\mathscr{I}}`` is the radiative flux divergence (heating rate).

## Architecture Support

The radiative transfer implementation supports both CPU and GPU architectures. The column-based RRTMGP solver is called from Oceananigans' field data arrays with appropriate data layout conversions.
