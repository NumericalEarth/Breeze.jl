# Radiative Transfer

Breeze.jl integrates with [RRTMGP.jl](https://github.com/NumericalEarth/RRTMGP.jl) to provide radiative transfer capabilities for atmospheric simulations. The radiative transfer model computes longwave and shortwave radiative fluxes, which can be incorporated into energy tendency equations.

## Gray Atmosphere Radiation

The simplest radiative transfer option is gray atmosphere radiation, which uses the optical thickness parameterization from [OGormanSchneider2008](@cite) and [Schneider2004](@cite). This approximation treats the atmosphere as having a single effective absorption coefficient rather than computing full spectral radiation.

### Basic Usage

To use gray radiation in a Breeze simulation, create a [`GrayRadiativeTransferModel`](@ref) model and pass it to the [`AtmosphereModel`](@ref) constructor:

```julia
using Breeze
using Oceananigans.Units
using Dates

# Create grid (single column at Beverly, MA)
Nz = 64
λ, φ = -70.9, 42.5  # longitude, latitude
grid = RectilinearGrid(size=Nz, x=λ, y=φ, z=(0, 20kilometers),
                       topology=(Flat, Flat, Bounded))

# Thermodynamic setup
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 101325,
                                 potential_temperature = 300)
formulation = AnelasticFormulation(reference_state,
                                   thermodynamics = :LiquidIcePotentialTemperature)

# Create gray radiation model
radiation = GrayRadiativeTransferModel(grid;
                          surface_temperature = 300,    # K
                          surface_emissivity = 0.98,
                          surface_albedo = 0.1,
                          solar_constant = 1361)        # W/m²

# Create atmosphere model with DateTime clock for solar position
clock = Clock(time=DateTime(2024, 9, 27, 16, 0, 0))
model = AtmosphereModel(grid; clock, formulation, radiation)
```

When a `DateTime` clock is used, the solar zenith angle is computed automatically from the time and grid location (longitude and latitude).

### Gray Radiation Model

The [`GrayRadiativeTransferModel`](@ref) model computes:

- **Longwave radiation**: Both upwelling and downwelling thermal radiation using RRTMGP's two-stream solver
- **Shortwave radiation**: Direct beam solar radiation (no scattering) using the O'Gorman optical thickness

The gray atmosphere optical thickness follows the parameterization in [OGormanSchneider2008](@cite):

```math
τ_{lw} = α \frac{Δp}{p} \left( f_l σ + (1-f_l) 4σ^4 \right) \left( τ_e + (τ_p - τ_e) \sin^2 φ \right)
```

where ``σ = p/p_0`` is the normalized pressure, ``φ`` is latitude, and ``α``, ``f_l``, ``τ_e``, ``τ_p`` are empirical parameters.

For shortwave:
```math
τ_{sw} = 2 τ_0 \frac{p}{p_0} \frac{Δp}{p_0}
```

where ``τ_0 = 0.22`` is the shortwave optical depth parameter.

### Radiative Fluxes

After running [`set!`](@ref) or [`update_state!`](@ref), the radiative fluxes are available from the radiation model:

```julia
# Longwave fluxes (ZFaceFields)
F_lw_up = radiation.upwelling_longwave_flux
F_lw_dn = radiation.downwelling_longwave_flux

# Shortwave flux (direct beam only for non-scattering solver)
F_sw = radiation.downwelling_shortwave_flux
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

## Surface Properties

The [`GrayRadiativeTransferModel`](@ref) model requires surface properties:

| Property | Description | Typical Values |
|----------|-------------|----------------|
| `surface_temperature` | Temperature at the surface [K] | 280-310 |
| `surface_emissivity` | Longwave emissivity (0-1) | 0.95-0.99 |
| `surface_albedo` | Shortwave albedo (0-1) | 0.1-0.3 |
| `solar_constant` | TOA solar flux [W/m²] | 1361 |

## Integration with Dynamics

Radiative fluxes can be used to compute heating rates for the energy equation. The radiative heating rate is computed from flux divergence:

```math
\dot{Q}_{rad} = -\frac{1}{\rho c_p} \frac{\partial F_{net}}{\partial z}
```

where ``F_{net}`` is the net radiative flux (upwelling minus downwelling) and ``c_p`` is the specific heat capacity.

## Architecture Support

The radiative transfer implementation supports both CPU and GPU architectures. The column-based RRTMGP solver is called from Oceananigans' field data arrays with appropriate data layout conversions.

## References

- [OGormanSchneider2008](@cite): Gray atmosphere optical thickness parameterization
- [Schneider2004](@cite): Idealized dry atmosphere radiation
