# Radiative Transfer

Breeze.jl integrates with [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl) to provide radiative transfer capabilities for atmospheric simulations. The radiative transfer model computes longwave and shortwave radiative fluxes, which can be incorporated into energy tendency equations.

## Gray Atmosphere Radiation

The simplest radiative transfer option is gray atmosphere radiation, which uses the optical thickness parameterization from [Schneider2004](@citet) and [OGormanSchneider2008](@citet). This approximation treats the atmosphere as having a single effective absorption coefficient rather than computing full spectral radiation.

### Basic Usage

To use gray radiation in a Breeze simulation, create a [`RadiativeTransferModel`](@ref) model with the [`GrayOptics`](@ref) optics flavor and pass it to the [`AtmosphereModel`](@ref) constructor:

```@example
using Breeze
using Breeze.AtmosphereModels
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

dynamics = AnelasticDynamics(reference_state)

# Create gray radiation model
radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                   surface_temperature,
                                   surface_emissivity = 0.98,
                                   surface_albedo = 0.1,
                                   solar_constant = 1361) # W/m²

# Create atmosphere model with DateTime clock for solar position
clock = Clock(time=DateTime(2024, 9, 27, 16, 0, 0))
model = AtmosphereModel(grid; clock, dynamics, radiation)
```

When a `DateTime` clock is used, the cosine of the solar zenith angle is computed automatically from the time and grid location (longitude and latitude). See [Solar zenith angle](@ref) below for the full set of options.

### Gray Radiation Model

The [`RadiativeTransferModel`](@ref) model computes:

- **Longwave radiation**: Both upwelling and downwelling thermal radiation using RRTMGP's two-stream solver
- **Shortwave radiation**: Direct beam solar radiation

The gray atmosphere optical thickness for longwave follows the parameterization by [OGormanSchneider2008](@citet),

```math
τ_{lw} = α \frac{Δp}{p_0} \left[ f_l + 4 (1 - f_l) \left(\frac{p}{p_0}\right)^3 \right] \left[ τ_e + (τ_p - τ_e) \sin^2 φ \right]
```

where ``φ`` is latitude and ``α``, ``f_l``, ``τ_e``, and ``τ_p`` are empirical parameters.

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

## Solar zenith angle

The cosine of the solar zenith angle ``\cos(θ_z)`` controls the magnitude of the
top-of-atmosphere shortwave flux and the slant-path optical depth through the
column. Breeze provides one keyword — `solar_position` — that selects how
``\cos(θ_z)`` is determined on each radiation update. It takes any subtype of
[`AbstractSolarPosition`](@ref). The two concrete subtypes cover the common
cases:

| Subtype | Behavior | Typical use |
|---|---|---|
| [`ApparentSolarPosition`](@ref) (default) | Time-varying. ``\cos(θ_z)`` is recomputed each update from the model clock and observer ``(λ, φ)`` via [`Breeze.CelestialMechanics.cos_solar_zenith_angle`](@ref). | Real-world simulations, diurnal cycles, seasonal forcing. |
| [`FixedCosineZenith`](@ref) | Constant ``\cos(θ_z)``. The clock has no effect on the sun position. | Idealized radiative-convective equilibrium (RCE), forcing-shape studies. |

### Time-varying apparent sun

[`ApparentSolarPosition`](@ref) accepts two optional keyword arguments:

- `coordinate`: an explicit `(longitude, latitude)` tuple in degrees, or
  `nothing` (the default) to read ``(λ, φ)`` from the grid's coordinates.
  For single-column grids the grid's `(x, y)` is interpreted as `(λ, φ)`.
- `epoch`: a `DateTime` anchor for floating-point model clocks. The model time
  in seconds is added to `epoch` to obtain the absolute `DateTime`. With a
  `DateTime` clock, `epoch` is ignored.

```julia
# Today's default: DateTime clock, λ/φ inferred from the grid.
solar_position = ApparentSolarPosition()

# Float clock + epoch — useful on lat–lon / curvilinear grids where clock
# precision matters but you want full per-column zenith.
solar_position = ApparentSolarPosition(epoch = DateTime(2024, 1, 1))

# Pin a 3D simulation to a specific observer (overrides per-column lat/lon).
solar_position = ApparentSolarPosition(coordinate = (-70.9, 42.5),
                                       epoch      = DateTime(2024, 1, 1))
```

!!! warning "Numeric clock + `epoch = nothing`"
    `ApparentSolarPosition(epoch = nothing)` with a floating-point model clock
    cannot resolve a `DateTime`. The radiation update will throw an
    `ArgumentError` with instructions: switch to a `DateTime` clock, supply an
    `epoch`, or use [`FixedCosineZenith`](@ref).

### Constant cos(θ_z)

For idealized studies where you want a single, time-independent solar geometry
— typical in RCE intercomparisons — use [`FixedCosineZenith`](@ref):

```julia
radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                   surface_temperature  = 300,
                                   surface_albedo       = 0.1,
                                   solar_constant       = 1361,    # W/m²
                                   solar_position       = FixedCosineZenith(0.5))

# A floating-point clock works fine — no epoch is required.
clock = Clock(time = 0.0)
model = AtmosphereModel(grid; clock, dynamics, radiation)
```

The cosine of the zenith angle is written into the RRTMGP boundary-condition
array once at construction and never recomputed; the per-step
`update_solar_zenith_angle!` call becomes a no-op.

#### Choosing a value

Common choices for ``\cos(θ_z)`` in idealized work:

| Setup | ``\cos(θ_z)`` | Notes |
|---|---|---|
| Diurnal mean at the equator | ``\approx 0.5`` | A common RCE default. |
| Global annual mean | ``\approx 0.41`` | Matches the planet's spherical insolation when paired with ``S_0 / 4``. |
| Overhead sun | ``1`` | No slant-path effect. |

#### Interaction with `solar_constant`

The top-of-atmosphere downward shortwave flux is `solar_constant * cos_zenith`,
so `solar_constant` and `FixedCosineZenith` together control both:

- the **TOA SW magnitude** (their product), and
- the **slant-path absorption** (which depends on ``\cos(θ_z)`` *alone*, through
  ``\exp(-τ/\cos(θ_z))``).

Note that scaling `solar_constant` and scaling `cos_zenith` are **not**
equivalent for the shortwave heating profile. The TOA flux changes the same
way, but the shape of the absorption with height does not. If your study cares
about the vertical structure of SW heating (it usually does), pick
``\cos(θ_z)`` to match the slant path you actually want, then choose
`solar_constant` to set the magnitude.

For example, to model diurnal-mean conditions with the full solar constant
spread over the day's path:

```julia
solar_position = FixedCosineZenith(0.5)
solar_constant = 1361 / 2      # because TOA SW = solar_constant * cos_zenith
```

#### Latitude for gray optics

The gray-optics longitudinal-mean optical thickness depends on latitude through
``τ_e`` and ``τ_p``. With [`FixedCosineZenith`](@ref) you specify only
``\cos(θ_z)`` — Breeze still reads the latitude needed for the gray τ from the
grid's coordinates (or from `coordinate`, in 3D setups where you pass an
explicit position via `ApparentSolarPosition`). This means it's fine to combine
a fixed zenith with a single-column grid located at a particular latitude:
`τ_e/τ_p` will reflect that latitude, while the zenith stays pinned.

## Clear-sky Full-spectrum Radiation

For more accurate radiative transfer calculations, use the [`ClearSkyOptics`](@ref) optics flavor which computes full-spectrum gas optics using RRTMGP's lookup tables:

```@example
using Breeze, Oceananigans.Units
using RRTMGP, NCDatasets # Required for RRTMGP lookup tables

grid = RectilinearGrid(; size=16, x=0, y=45, z=(0, 10kilometers),
                       topology=(Flat, Flat, Bounded))
constants = ThermodynamicConstants()
radiation = RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                   surface_temperature = 300,
                                   surface_emissivity = 0.98,
                                   surface_albedo = 0.1,
                                   background_atmosphere = BackgroundAtmosphere(CO₂ = 400e-6))
```

The [`BackgroundAtmosphere`](@ref) struct specifies volume mixing ratios for radiatively active gases (CO₂, CH₄, N₂O, O₃, etc.). Water vapor is computed from the model's prognostic moisture field.

Clear-sky and all-sky models accept the same `solar_position` keyword as
gray-optics; the three optics flavors share the solar-position machinery.

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
