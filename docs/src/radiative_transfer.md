# Radiative Transfer

Breeze.jl computes longwave and shortwave radiative fluxes with a [`RadiativeTransferModel`](@ref), whose flux divergence enters the energy tendency of an [`AtmosphereModel`](@ref). Two radiation backends are available as package extensions, selected by the optics passed to the constructor:

- [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl) solves gray-atmosphere ([`GrayOptics`](@ref)), clear-sky ([`ClearSkyOptics`](@ref)), and all-sky ([`AllSkyOptics`](@ref)) radiation with its correlated-``k`` lookup tables; load it with `using RRTMGP, ClimaComms, NCDatasets`.
- [NumericalRadiation.jl](https://github.com/NumericalEarth/NumericalRadiation.jl) solves clear-sky and all-sky radiation with the ecCKD gas optics ([`EcCKDOptics`](@ref)) in one kernel per column, on CPUs and GPUs; load it with `using NumericalRadiation: NumericalRadiation` and `using NCDatasets` (NumericalRadiation exports its own `ThermodynamicConstants`, so it is loaded qualified next to Breeze). See [ecCKD radiation with NumericalRadiation.jl](@ref) below.

Both backends fill the same four flux fields and the same flux divergence, so a simulation switches between them by changing the optics alone.

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
                                 base_pressure = 101325,
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

After running [`set!`](@ref), the radiative fluxes are available from the radiation
model as four `ZFaceField`s, one per stream. All three optics flavors expose the
same four fields, and all of them follow the sign convention **positive upward**,
so downwelling fluxes are stored as negative numbers:

```julia
# Longwave
ℐ_lw_up = radiation.upwelling_longwave_flux     # ≥ 0
ℐ_lw_dn = radiation.downwelling_longwave_flux   # ≤ 0

# Shortwave
ℐ_sw_up = radiation.upwelling_shortwave_flux    # ≥ 0
ℐ_sw_dn = radiation.downwelling_shortwave_flux  # ≤ 0
```

For clear-sky and all-sky optics, `downwelling_shortwave_flux` is the *total*
downwelling shortwave (direct beam plus diffuse), and `upwelling_shortwave_flux`
carries the sunlight scattered back upward by air and clouds together with the
fraction reflected by the surface.

!!! note "Shortwave radiation in the gray atmosphere"
    The gray atmosphere uses RRTMGP's non-scattering shortwave solver, so
    `downwelling_shortwave_flux` is the direct beam alone (no diffuse component).
    The surface reflects the fraction `direct_surface_albedo` of the direct beam
    diffusely, and `upwelling_shortwave_flux` is that reflected beam, attenuated by
    ``\exp(-D τ_{sw})`` across each layer on its way up, where ``D ≈ 1.64`` is the
    diffusivity of the gray longwave solver's default one-angle quadrature. The atmosphere absorbs what the
    reflected beam loses, and the rest leaves through the top of the atmosphere.
    The `diffuse_surface_albedo` has no effect, since no diffuse shortwave reaches
    the surface.

## Solar zenith angle

The cosine of the solar zenith angle ``\cos(θ_z)`` controls the magnitude of the
top-of-atmosphere shortwave flux and the slant-path optical depth through the
column. Breeze provides one keyword — `solar_position` — that selects how
``\cos(θ_z)`` is determined on each radiation update. It takes any subtype of
[`AbstractSolarPosition`](@ref). The two concrete subtypes cover the common
cases:

| Subtype | Behavior | Typical use |
|---|---|---|
| [`ApparentSolarPosition`](@ref) (default) | Real-Earth time-varying. ``\cos(θ_z)`` is recomputed each update from the model clock and observer ``(λ, φ)`` via [`Breeze.CelestialMechanics.cos_solar_zenith_angle`](@ref) — includes orbital declination, equation of time, etc. | Real-world simulations driven by a calendar date. |
| [`DiurnalSolarPosition`](@ref) | Idealized diurnal cycle at a fixed latitude and declination. ``\cos(θ_z)`` follows the analytical hour-angle formula without any calendar dependence. Requires a numeric clock. | Idealized diurnal-cycle studies, perpetual equinox or solstice runs, non-Earth rotators. |
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

### Idealized diurnal cycle

For idealized studies where you want a clean 24-hour cycle (or any rotation
period) without orbital mechanics, calendar dates, or equation-of-time
corrections, use [`DiurnalSolarPosition`](@ref):

```julia
# Perpetual equinox at 30°N — 24-hour day, noon at t = 0
solar_position = DiurnalSolarPosition(latitude = 30)

# Perpetual June solstice analog at 45°N
solar_position = DiurnalSolarPosition(latitude = 45, declination = 23.5)

# Non-Earth rotator with a 10-hour day; the simulation starts at sunrise
solar_position = DiurnalSolarPosition(latitude = 0,
                                      day_length  = 10 * 3600,
                                      noon_offset = 5  * 3600)
```

The cosine of the solar zenith angle is computed analytically each radiation
update from the model clock:

```math
\cos(θ_z) = \sin(φ) \sin(δ) + \cos(φ) \cos(δ) \cos(ω),
\qquad
ω = \frac{2π}{T_d} (t - t_{\text{noon}}),
```

where ``φ`` is the (fixed) latitude, ``δ`` is the (fixed) declination,
``T_d`` is the day length, and ``t_{\text{noon}}`` is the simulation time
at which local noon occurs. There is no annual cycle, no equation of time,
and no longitude — by design.

#### Required: numeric clock

`DiurnalSolarPosition` reads `model.clock.time` as "seconds since the start
of the simulation" and computes the hour angle from it. A `DateTime` clock
is rejected with an actionable error, because pairing one with this
idealized cycle would be semantically ambiguous (the calendar would be
ignored). Use `Clock(time = 0.0)` (or any numeric `Clock`) when constructing
the `AtmosphereModel`.

#### Choosing `noon_offset`

The default `noon_offset = 0` places local noon at `t = 0` — the sun is
overhead at the start of the simulation. To start at sunrise instead, set
`noon_offset = day_length / 4`; to start at midnight, set
`noon_offset = day_length / 2`. The cosine function is naturally periodic,
so no mod is required and the cycle simply repeats every `day_length`
seconds.

#### Gray-optics latitude

Unlike [`FixedCosineZenith`](@ref) — where latitude for the gray τ comes
from the grid — `DiurnalSolarPosition` carries its own `latitude` and uses
it for *both* the diurnal cos(θ_z) calculation *and* the gray-optics
latitude-dependent τ. This keeps the two consistent in idealized
single-column setups where the grid's `y` might be a placeholder.

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
`update_cos_zenith!` call becomes a no-op.

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
gray-optics; all optics flavors share the solar-position machinery.

## ecCKD radiation with NumericalRadiation.jl

[`EcCKDOptics`](@ref) selects the ecCKD correlated-``k`` gas optics of
[HoganMatricardi2022](@citet), solved by
[NumericalRadiation.jl](https://github.com/NumericalEarth/NumericalRadiation.jl).
The gas optics tables ship with the ecRad data that NumericalRadiation downloads as an
artifact, and are read from netCDF files, so both `NumericalRadiation` and `NCDatasets`
must be loaded. NumericalRadiation exports its own `ThermodynamicConstants`; loading it
qualified avoids the clash with Breeze's:

```@example ecckd
using Breeze, Oceananigans.Units
using NCDatasets
using NumericalRadiation: NumericalRadiation

grid = RectilinearGrid(; size=16, x=0, y=45, z=(0, 10kilometers),
                       topology=(Flat, Flat, Bounded))
constants = ThermodynamicConstants()

radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                   surface_temperature = 300,
                                   surface_emissivity = 0.98,
                                   surface_albedo = 0.1,
                                   background_atmosphere = BackgroundAtmosphere(CO₂ = 400e-6),
                                   overlying_atmosphere = OverlyingAtmosphereColumn(top = 65kilometers, layers = 40))
```

The keyword arguments are those of the RRTMGP clear-sky and all-sky constructors
(surface properties, `solar_position`, `solar_constant`, `schedule`,
`background_atmosphere`, effective radii) plus `overlying_atmosphere`. The default
`EcCKDOptics()` uses the `:climate_32x32` reference model with 32 longwave and 32
shortwave g-points; `EcCKDOptics(:climate_64x64)` doubles the spectral resolution, and
a `(longwave = path, shortwave = path)` pair selects ecCKD definition files of your own.
On a `GPU()` grid the tables and the solve live on the device.

### Overlying atmosphere

A limited-area grid rarely reaches the top of the atmosphere. Rather than treating the
grid top as the top of the atmosphere, an ecCKD model continues the radiation column
above it through a [`OverlyingAtmosphereColumn`](@ref): by default 40 geometrically stretched
layers from the grid top to 65 km, whose temperature follows the U.S. Standard
Atmosphere ([`standard_atmosphere_temperature`](@ref)) anchored to the temperature on the
grid's top face and relaxing to the standard profile over a 1 km `blending_height`.
Pressure is hydrostatic above the grid, the specific humidity follows an idealized profile
floored at the stratospheric value, and the ozone mole fraction follows
`BackgroundAtmosphere.O₃` — a number or a function of height, evaluated above the grid as
well — unless `ozone_mole_fraction` overrides it. Extension layers are clear.

The extension is what gives the downwelling longwave flux on the grid's top face its
stratospheric contribution (a few W m⁻² for a 20 km grid) and lets the stratospheric
ozone absorb ultraviolet sunlight before it reaches the grid, instead of depositing that
absorption in the grid's top cells. Pass `overlying_atmosphere = nothing` to stop the column
at the grid top, which is the right choice for comparisons with the RRTMGP models (whose
column ends there) or for a grid that already reaches the mesosphere.

```julia
# Stop at the grid top
radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                   surface_temperature = 300, surface_albedo = 0.1,
                                   overlying_atmosphere = nothing)

# A taller, finer extension with a constant stratospheric ozone mole fraction
extension = OverlyingAtmosphereColumn(top = 80kilometers, layers = 60, ozone_mole_fraction = 2e-6)
```

### Gases

The ecCKD reference models tabulate eight gases. Two vary in space: water vapor, taken
from the model's prognostic moisture, and ozone, from `BackgroundAtmosphere.O₃`. Five are
well-mixed and read from the [`BackgroundAtmosphere`](@ref): CO₂, CH₄, N₂O, CFC₁₁ and
CFC₁₂. Nitrogen and oxygen form the dry air, whose absorption the tables fold in. Gas amounts follow the "dry" column convention the ecCKD tables were derived with:
the dry-air amount of a layer is its total mass over the dry molar mass, `ρ Δz / mᵈ`
(`Δp / (g mᵈ)` for a hydrostatic layer), the water vapor `ρ qᵛ Δz / mᵛ` is counted on top of
it, and the well-mixed gases are their mole fractions times the dry-air amount. The
molar masses `mᵈ` and `mᵛ` and the gravitational acceleration `g` are those of the
`ThermodynamicConstants` passed to `RadiativeTransferModel`; the Stefan–Boltzmann constant
of the gray Planck source (used by g-points without a Planck source table) is carried by
NumericalRadiation's gas optics model.
The remaining gases of `BackgroundAtmosphere` (CO, NO₂, CFC₂₂, CCl₄, CF₄ and the
HFCs) are not tabulated and must be left at zero; a nonzero value throws an `ArgumentError`
rather than being silently ignored.

### Clouds

All-sky ecCKD radiation adds the cloud liquid and ice diagnosed by the model's
microphysics to the layer optics through single-scattering tables mapped onto the same
g-points as the gas optics, selected with [`CloudScatteringTables`](@ref):

```julia
optics = EcCKDOptics(clouds = CloudScatteringTables(liquid = :mie_droplet,
                                                    ice = :baum_general_habit_mixture))

radiation = RadiativeTransferModel(grid, optics, constants;
                                   surface_temperature = 300, surface_albedo = 0.1,
                                   liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                   ice_effective_radius = ConstantRadiusParticles(30e-6))
```

The two selectors name the Mie droplet and Baum general-habit-mixture ice tables shipped
with the ecRad data; a path to a netCDF file in the same format works in their place. The
tables are evaluated once, at construction, at the constant effective radii of
`liquid_effective_radius` and `ice_effective_radius` (only
[`ConstantRadiusParticles`](@ref) for now). The shortwave folds the cloud scattering into
the two-stream solution with delta-Eddington scaling; the longwave adds the cloud
absorption only and neglects longwave cloud scattering (ecRad's
`do_lw_cloud_scattering = false` setting, as RRTMGP's no-scattering longwave solver does).
With
`clouds = nothing` (the default) the sky is clear whatever the microphysics holds.

### Sign and units conventions

NumericalRadiation solves each column top-down with longwave and shortwave fluxes positive
in their own direction of travel. Breeze converts on the way back to the grid, so an ecCKD
model exposes the same four `ZFaceField`s as the RRTMGP models, in W m⁻², all
**positive upward** with downwelling fluxes stored as negative numbers, and the same
`flux_divergence` in W m⁻³ (see [Integration with dynamics](@ref)). On the grid's top face
the downwelling shortwave equals `-solar_constant * cos_zenith` only without a column
extension; with one it is reduced by the absorption above the grid. The downwelling
longwave on the top face is zero without an extension and the emission of the atmosphere
above with one.

### Coexistence with RRTMGP

The two extensions load side by side (`using RRTMGP, ClimaComms, NCDatasets` together
with `using NumericalRadiation: NumericalRadiation`), and [`GrayOptics`](@ref),
[`ClearSkyOptics`](@ref) and [`AllSkyOptics`](@ref) remain RRTMGP models. Because both
backends fill the same fields, one script can build an RRTMGP model and an ecCKD model on
two `AtmosphereModel`s with identical initial conditions and difference their fluxes; the
[single column radiation example](literated/single_column_radiation.md) does exactly that.
On the same 20 km column, with `overlying_atmosphere = nothing` so that both stop at the grid
top, the two agree to 2 W m⁻² in the clear-sky boundary fluxes, to 0.1 W m⁻² in the 2×CO₂
forcing and to 0.2 K day⁻¹ (root mean square) in the clear-sky heating rates, the spread
expected between two independent correlated-``k`` models; they part at cloud top, where the
cloud optics differ in kind.

## Surface Properties

The [`RadiativeTransferModel`](@ref) model requires surface properties:

| Property | Description | Typical Values |
|----------|-------------|----------------|
| `surface_temperature` | Temperature at the surface [K] | 280-310 |
| `surface_emissivity` | Longwave emissivity (0-1) | 0.95-0.99 |
| `surface_albedo` | Shortwave albedo (0-1) | 0.1-0.3 |
| `solar_constant` | TOA solar flux [W/m²] | 1361 |

## Integration with dynamics

Radiative fluxes provide a volumetric energy source from flux convergence,

```math
Q_{\mathscr{I}} = -\frac{\partial \mathscr{I}_{net}}{\partial z},
```

where ``Q_{\mathscr{I}}`` has units W/m³. For
[`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics), holding
moisture phase composition fixed, the corresponding temperature tendency is

```math
\left. \frac{\partial T}{\partial t} \right|_{rad} = \frac{Q_{\mathscr{I}}}{ρᵣ cᵖᵐ}.
```

Here ``ρᵣ`` is the reference density, ``cᵖᵐ`` is the mixture heat capacity, and
``\mathscr{I}_{net}`` is the net upward radiative flux. Because Breeze stores
downwelling components as negative values, the four signed streams are added:

```math
\mathscr{I}_{net} = \mathscr{I}_{lw}^{↑} + \mathscr{I}_{lw}^{↓}
                  + \mathscr{I}_{sw}^{↑} + \mathscr{I}_{sw}^{↓} .
```

Despite its historical name, `radiation.flux_divergence` stores ``Q_{\mathscr{I}}``:
minus the vertical derivative of the signed flux sum, evaluated at cell centers
from the two bounding cell faces. The static-energy formulation adds this W/m³
source directly to its energy-density tendency. The liquid-ice potential-temperature
formulation adds ``Q_{\mathscr{I}} / (\hat{c}ᵖ \hat{Π})`` to its potential-temperature-density
tendency, with the vapor-equivalent heat capacity ``\hat{c}ᵖ`` and Exner function ``\hat{Π}``
(see [Liquid-ice potential temperature](@ref liquid-ice-potential-temperature)). Omitting the upwelling shortwave would bias either tendency wherever
shortwave radiation scatters or reflects.

## Radiation between solves

A `schedule` other than every iteration solves the radiative transfer equations only now and then.
Between two solves the surface warms or cools and the sun moves, and every state update corrects the
fluxes for both, following [HoganBozzo2015](@citet):

- **Longwave.** Each solve repeats the longwave solve with the surface 1 K warmer, which gives the
  response of the fluxes on every face to the fourth power of the surface temperature ``T₀``. Each
  Runge–Kutta stage then shifts the saved fluxes of that solve by the response times the change
  in ``T₀⁴`` since the solve. Using this fixed reference lets small Float32 corrections accumulate
  without being rounded away at each stage. The column absorbs its share of a change in surface
  emission, and the surface and the atmosphere
  exchange the same longwave flux at every step. With a constant `surface_temperature` there is no
  second solve. The full-spectrum RRTMGP solve clamps the surface temperature into its Planck
  tables, 160 K to 355 K: within 1 K of the top the second solve cools the surface by 1 K instead,
  and above 355 K the longwave fluxes no longer follow the surface temperature, as in a solve.
  This clamp applies only to RRTMGP; ecCKD retains its own extrapolation of surface emission.
- **Shortwave.** Each solve computes the slant path of the direct beam with ``\bar{μ}₀``, the
  cosine of the solar zenith angle averaged over the sunlit part of the radiation window
  ([HoganHirahara2015](@citet)). For `DiurnalSolarPosition` and `ApparentSolarPosition`, this is a
  time integral clipped at sunrise and sunset, divided by the actual sunlit duration. The apparent
  solar position is integrated separately on each calendar day. Each state update
  then scales the solved shortwave fluxes by ``μ₀ / \bar{μ}₀``, which keeps the column's
  transmittance and reflectance and brings the flux entering the top of the atmosphere to the
  current sun, including a sun that rises between solves.

Corrections also run at the end of the step, so the fluxes match the completed model clock.
Restoring an older checkpoint without radiation fluxes or a required approximate-update cache
triggers a fresh solve at the next state update, including with the default every-iteration schedule.

The radiation window covers every model time at which a state update scales the fluxes of a solve:
from the start of the time step that solves, since later Runge–Kutta stages can return to times
before the first (SSP RK3 evaluates its third stage at ``tⁿ + Δt/2``, after its second at
``tⁿ + Δt``), to the start of the time step that solves next. For a `TimeInterval` or
`SpecifiedTimes` schedule, this is the first step whose first stage reaches the next scheduled
time, estimated using the last time step and the time stepper's stage abscissae. The interval
ends at that step's start, before its first-stage solve. With unknown stage timing it ends at
the scheduled time. An `IterationInterval` uses its next actuating iteration and the last time step.
When the interval is not known, past the last explicit time, for an infinite interval, or for other schedules,
each solve uses the current sun to compute the slant path. Shortwave fluxes still scale with the
current solar cosine, so they vanish at night. Custom solar-position types also use the current
angle unless they implement interval averaging.

The update cache stores six extra face fields: the two longwave responses and the four fluxes
of the last solve. These reference fluxes and the solved surface temperature are checkpointed;
checkpoints from the earlier incremental-update format trigger a fresh solve on restore.

Over its window the sun stays within four times ``\bar{μ}₀``, a bound approached only by a sun
skimming the horizon on both sides of a short night. A state update that finds ``μ₀ > 4 \bar{μ}₀``,
sunlight the last solve did not average over, solves again without advancing the schedule, since
the slant path of ``\bar{μ}₀`` would deposit the beam far too high in the column. That happens
when daylight follows a night-time solve with no known interval, when the sun climbs well above
the sun of a solve with no known interval, and when time steps grow past the interval an
`IterationInterval` estimated. Under Reactant the fluxes are not updated between solves.

## Architecture Support

The radiative transfer implementation supports both CPU and GPU architectures. The column-based RRTMGP solver is called from Oceananigans' field data arrays with appropriate data layout conversions. The ecCKD backend runs on the grid's architecture without leaving the device. It can solve the columns in batches of whole grid rows (`column_batches`, as for RRTMGP; one solve of the whole grid by default): per batch, a layer pass reads the model's fields directly and stores each layer's gas-optics stencil, gas amounts, cloud water paths and interface temperatures, one work item per layer, and a column kernel streams the g points through them and accumulates the fluxes directly into the radiation model's flux fields. Only one batch of per-column storage is allocated, so setting `column_batches` caps the radiation's memory on large grids, at some cost in speed on a GPU.
