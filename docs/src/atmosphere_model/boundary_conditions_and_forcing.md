# Boundary conditions and forcing

[`AtmosphereModel`](@ref) accepts `boundary_conditions` and `forcing` as `NamedTuple`s whose
keys name the variable each applies to:

```julia
model = AtmosphereModel(grid; boundary_conditions = (; ρθ = ρθ_bcs),
                              forcing = (; ρθ = ρθ_forcing))
```

Both are validated when the model is built. A key that names nothing the model can apply it to
raises an `ArgumentError`, rather than being accepted and then quietly ignored:

```@example bcs
using Breeze
using Oceananigans

grid = RectilinearGrid(size=(8, 8), x=(0, 1e3), z=(0, 1e3), topology=(Periodic, Flat, Bounded))
bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(100))

try
    AtmosphereModel(grid; boundary_conditions=(; ρe=bcs))
catch err
    println(err.msg)
end
```

## Prognostic names depend on the model

Two of Breeze's prognostic variables are named by the choices you make elsewhere in the model.
The thermodynamic variable follows the `formulation`:

| `formulation` | thermodynamic prognostic |
|---|---|
| `:LiquidIcePotentialTemperature` (default) | ``ρθ`` |
| `:StaticEnergy` | ``ρs`` |

and the moisture variable follows the `microphysics`:

| `cloud_formation` / scheme | moisture prognostic |
|---|---|
| `SaturationAdjustment` | ``ρqᵉ``, equilibrium moisture |
| `NonEquilibriumCloudFormation`, `DCMIP2016KesslerMicrophysics`, `nothing` | ``ρqᵛ``, vapor |

Keying a surface flux by one of those names ties the setup to one formulation or one
microphysics scheme. Changing `cloud_formation` would then silently change which key you are
supposed to use — a detail of how condensation is parameterized, leaking into the specification
of an evaporative flux.

## [Interface keys: `ρE` and `ρqᵗ`](@id interface-keys)

To avoid that, an energy or water input is supplied under a key that names the *physical
input* rather than the variable that carries it:

| Key | Input | Specific alias (forcings) | Applied to |
|---|---|---|---|
| ``ρE`` | Energy: W m⁻² at a boundary, W m⁻³ in the interior | ``E`` | the thermodynamic prognostic |
| ``ρqᵗ`` | Water: kg m⁻² s⁻¹ at a boundary, kg m⁻³ s⁻¹ in the interior | ``qᵗ`` | the moisture prognostic |

``ρE`` is a *total energy* density and ``ρqᵗ`` a *total moisture* density (see the
[notation appendix](@ref "Notation and conventions")). Neither is itself prognostic, so both
are unambiguous, and a setup written with them runs unchanged under any formulation and any
microphysics scheme:

```@example bcs
Q = 100    # sensible heat flux, W m⁻²
E = 1e-5   # evaporative mass flux, kg m⁻² s⁻¹

ρE_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Q))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(E))
boundary_conditions = (; ρE=ρE_bcs, ρqᵗ=ρqᵗ_bcs)

θ_model = AtmosphereModel(grid; boundary_conditions)
s_model = AtmosphereModel(grid; formulation=:StaticEnergy, boundary_conditions)
nothing # hide
```

Breeze converts the input as the receiving variable requires. Static energy is itself an energy
per unit mass, so an energy flux reaches ``ρs`` unchanged; for ``ρθ`` it is divided by the local
vapor-equivalent heat capacity ``\hat{c}ᵖ`` and Exner function ``\hat{Π}``, because
``\hat{c}ᵖ \hat{Π} θˡⁱ`` is the enthalpy ``H`` (see
[Liquid-ice potential temperature](@ref liquid-ice-potential-temperature)), so that heating at
fixed composition changes ``θˡⁱ`` by ``1 / (\hat{c}ᵖ \hat{Π})`` per joule. A flux and a forcing take
the same conversion: at fixed moisture ``cᵖᵐ δT = \hat{c}ᵖ \hat{Π} δθ``, so with
``𝒬 = ρ cᵖᵐ \overline{w'T'}`` the flux that reaches ``ρθ`` is ``𝒬 / (\hat{c}ᵖ \hat{Π})``.
Water needs no conversion under either scheme: water
added to the prognostic moisture is water added to ``qᵗ``.

The interface key is *not* a field, and does not appear in the model:

```@example bcs
keys(θ_model.timestepper.Gⁿ)
```

`ρE` and `ρqᵗ` are the recommended keys. The specific names remain available where they are
genuinely prognostic — `ρs` under `formulation = :StaticEnergy`, `ρqᵉ` under saturation
adjustment — which is useful when you mean a flux of that variable specifically rather than an
energy or water input. Supplying both an interface key and the variable it routes onto is an
error, since the two would be summed into a single flux on the same field.

## Surface fluxes from bulk formulae

A constant or spatially varying flux can be given directly, as above. Fluxes that depend on the
evolving surface state are instead supplied as bulk formulae, which Breeze evaluates against
the model state each time step:

```@example bcs
Cᴰ = 1.2e-3  # drag coefficient
Cᵀ = 1.2e-3  # transfer coefficient for heat and moisture
T₀ = 300     # sea surface temperature, K

ρE_bcs = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=T₀))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=Cᵀ, surface_temperature=T₀))
ρu_bcs = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=Cᴰ))  # `BulkDrag` is also an Oceananigans name

model = AtmosphereModel(grid; boundary_conditions=(; ρE=ρE_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs))
nothing # hide
```

[`BulkSensibleHeatFlux`](@ref Breeze.BoundaryConditions.BulkSensibleHeatFlux) forms its surface
difference in whichever thermodynamic variable the model evolves — ``Δθ`` for a potential temperature
model, ``Δs`` for a static energy model — so the same specification is correct for both. See
[Wall fluxes](@ref "Wall fluxes") for placement on any of the six boundaries, the forms the wall state
may take, and stability-corrected transfer coefficients.

## Forcing

Interior sources are supplied under `forcing`, using the same keys. Each also accepts a
*specific* alias — the name without its leading ``ρ`` — which Breeze multiplies by the density
at kernel time. A forcing keyed `ρθ` is a source of ``ρθ``; one keyed `θ` is a source of ``θ``:

```@example bcs
∂t_θ = 1 / 86400   # radiative cooling rate, K s⁻¹
model = AtmosphereModel(grid; forcing=(; θ=Returns(-∂t_θ)))
nothing # hide
```

The energy and water interface keys work the same way, with specific aliases ``E`` and ``qᵗ``:

```@example bcs
∂t_E = -2.0   # radiative cooling, W m⁻³
model = AtmosphereModel(grid; forcing=(; ρE=Returns(∂t_E)))
nothing # hide
```

As with boundary conditions, an unrecognized forcing key is an error, and only one key per
variable may be supplied.

### [Held–Suarez forcing](@id held_suarez_notation)

[`held_suarez_forcings`](@ref Breeze.Forcings.held_suarez_forcings) returns the Rayleigh friction and
Newtonian temperature relaxation of [HeldSuarez1994](@citet) as `(; ρu, ρv, ρE)`, to pass as `forcing`
or to merge with other forcings. The relaxation is supplied under the energy key `ρE`, so it is converted to
whichever thermodynamic variable the formulation evolves:

```@example bcs
sphere = LatitudeLongitudeGrid(size=(36, 16, 8), longitude=(0, 360), latitude=(-80, 80), z=(0, 30e3))
dynamics = CompressibleDynamics(reference_potential_temperature = z -> 250 * exp(9.81 * z / (1005 * 250)))
model = AtmosphereModel(sphere; dynamics, forcing=held_suarez_forcings())
model.forcing.ρE
```

Momentum is damped and temperature relaxes toward ``Tᵉ``,

```math
∂_t (ρu) = ⋯ - kᵘ ρu , \qquad ∂_t (ρv) = ⋯ - kᵘ ρv , \qquad ∂_t T = ⋯ - kᵀ (T - Tᵉ) ,
```

with

```math
\begin{aligned}
Tᵉ &= \max \left \{ Tᵐⁱⁿ, \left [ Tᵉ₀ - ΔTʸ \sin^2 φ - Δθᶻ \log \left ( \frac{p}{p₀} \right ) \cos^2 φ \right ] \left ( \frac{p}{p₀} \right )^κ \right \} , \\
kᵀ &= kᵃ + (kˢ - kᵃ) \max \left ( 0, \frac{σ - σᵇ}{1 - σᵇ} \right ) \cos^4 φ , \\
kᵘ &= kᶠ \max \left ( 0, \frac{σ - σᵇ}{1 - σᵇ} \right ) .
\end{aligned}
```

The surface pressure ``pˢ`` in ``σ = p / pˢ`` is extrapolated from the lowest cell center to the bottom face
assuming hydrostatic balance at fixed temperature. The latitude ``φ`` is the geographic latitude of each cell,
so the forcing is unchanged on rotated grids.

The notation of this section is scoped to it. The keywords of `held_suarez_forcings` and their defaults are

| symbol | keyword | default | meaning |
| ------ | ------- | ------- | ------- |
| ``σᵇ`` | `boundary_layer_top` | 0.7 | top of the frictional boundary layer in ``σ`` |
| ``kᶠ`` | `friction_rate` | 1 / day | Rayleigh friction rate at the surface |
| ``kᵃ`` | `atmosphere_relaxation_rate` | 1 / 40 days | relaxation rate of the free atmosphere |
| ``kˢ`` | `surface_relaxation_rate` | 1 / 4 days | relaxation rate at the equatorial surface |
| ``ΔTʸ`` | `equator_to_pole_temperature_difference` | 60 K | equator-to-pole difference of ``Tᵉ`` |
| ``Δθᶻ`` | `vertical_potential_temperature_difference` | 10 K | static stability of ``Tᵉ`` |
| ``Tᵉ₀`` | `equatorial_surface_temperature` | 315 K | ``Tᵉ`` at the equatorial surface |
| ``Tᵐⁱⁿ`` | `minimum_equilibrium_temperature` | 200 K | stratospheric floor of ``Tᵉ`` |
| ``p₀`` | `reference_pressure` | 10⁵ Pa | reference pressure of ``Tᵉ`` |
| ``κ`` | `exner_exponent` | 2/7 | exponent of ``p / p₀`` in ``Tᵉ`` (not the von Kármán constant) |

``kᵘ`` is ``k_v`` in the paper, because ``v`` is reserved for vapor.
