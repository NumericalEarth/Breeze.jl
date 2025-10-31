# Warm-phase saturation adjustment

```@setup microphysics
using CairoMakie
CairoMakie.activate!(type = "svg")
```

Warm-phase saturation adjustment is a model for water droplet nucleation that assumes that water vapor in excess of the saturation specific humidity is instantaneously converted to liquid water.
Mixed-phase saturation adjustment is described by [Chammas2023](@citet).
Saturation adjustment may be formulated as a nonlinear algebraic equation that relates temperature, potential temperature, and total specific humidity, derived from the definition of liquid potential temperature,

```math
θ = \frac{T}{Π} \left \{1 - \frac{ℒᵥ₀}{cᵖᵐ T} \max \left [0, qᵗ - qᵛ⁺(T) \right ] \right \} ,
```

where ``Π`` is the Exner function, ``θ`` is potential temperature, ``T`` is temperature,
``ℒᵥ₀`` is the reference latent heat of vaporization, ``qᵗ`` is the total specific humidity,
``qᵛ⁺`` is the saturation specific humidity, and ``cᵖᵐ`` is the moist air specific heat.
The condensate specific humidity is ``qˡ = \max(0, qᵗ - qᵛ⁺)``: ``qˡ = 0`` if the air is undersaturated with ``qᵗ < qᵛ⁺``.
Both ``Π`` and ``cᵖᵐ`` depend on the dry and vapor mass fractions ``qᵈ = 1 - qᵗ`` and
``qᵛ = qᵗ - qˡ``, and ``qᵛ⁺`` is an increasing function of temperature ``T``.
Rewriting the potential temperature relation, saturation adjustment requires solving ``r(T) = 0``,

```math
r(T) = T - θ Π - \frac{ℒᵥ₀}{cᵖᵐ} \max[0, qᵗ - qᵛ⁺(T)] .
```

We use a secant method after checking for ``θ = 0`` and ``qˡ = 0`` given the guess ``T₁ = θ Π(qᵗ)``.
If ``qᵗ > qᵛ⁺(T₁)``, then we are guaranteed that ``T > T₁`` because ``qᵛ⁺`` is an increasing function of ``T``.
We initialize the secant iteration with a second guess ``T₂ = Θ Π - [qᵗ - qᵛ⁺(T₁)] ℒᵥ₀ / cᵖᵐ``.


As an example, we consider an air parcel at sea-level and with potential temperature of ``θ = 290``ᵒK, within a reference state with base pressure of 101325 Pa and a reference potential temperature ``288``ᵒK.
The saturation specific humidity is then

```@example microphysics
using Breeze
using Breeze.MoistAirBuoyancies: saturation_specific_humidity, HeightReferenceThermodynamicState

thermo = AtmosphereThermodynamics()
ref = ReferenceStateConstants(base_pressure=101325, potential_temperature=288)

z = 0.0    # [m] height
θ = 290.0  # [ᵒK] potential temperature
qᵛ⁺₀ = saturation_specific_humidity(θ, z, ref, thermo, thermo.liquid)
```

Recall that the specific humidity is unitless, or has units "``kg / kg``": kg of water vapor
per total kg of air.
We then perform a non-trivial saturation adjustment by computing temperature
given a total specific humidity slightly above saturation:

```@example microphysics
using Breeze.MoistAirBuoyancies: temperature

qᵗ = 0.012   # [kg kg⁻¹] total specific humidity
U = HeightReferenceThermodynamicState(θ, qᵗ, z)
T = temperature(U, ref, thermo)
```

Finally, we recover the amount of liquid condensate by subtracting the saturation
specific humidity from the total:

```@example microphysics
qᵛ⁺ = saturation_specific_humidity(T, z, ref, thermo, thermo.liquid)
qˡ = qᵗ - qᵛ⁺
```

### Saturation adjustment with varying total specific humidity

As a second example, we examine the dependence of temperature on total specific humidity
when the potential temperature is constant:

```@example microphysics
qᵗ = 0:1e-4:0.02 # [kg kg⁻¹] total specific humidity
U = [HeightReferenceThermodynamicState(θ, qᵗⁱ, z) for qᵗⁱ in qᵗ]
T = [temperature(Uⁱ, ref, thermo) for Uⁱ in U]

## Compare with a simple piecewise linear model
ℒᵥ₀ = thermo.liquid.latent_heat
cᵖᵈ = thermo.dry_air.heat_capacity
T̃ = [290 + ℒᵥ₀ / cᵖᵈ * max(0, qᵗⁱ - qᵛ⁺₀) for qᵗⁱ in qᵗ]

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Total specific humidity (kg kg⁻¹)", ylabel="Temperature (ᵒK)")
lines!(ax, qᵗ, T, label="Temperature from saturation adjustment")
lines!(ax, qᵗ, T̃, label="Temperature from linearized formula")
axislegend(ax)
fig
```

### Saturation adjustment with varying height

For a third example, we consider a state with constant potential temperature and total specific humidity,
but at varying heights:

```@example microphysics
qᵗ = 0.005
z = 0:100:10e3
T = [temperature(HeightReferenceThermodynamicState(θ, qᵗ, zᵏ), ref, thermo) for zᵏ in z]
qᵛ⁺ = [saturation_specific_humidity(T[k], z[k], ref, thermo, thermo.liquid) for k = 1:length(z)]
qˡ = [max(0, qᵗ - qᵛ⁺ᵏ) for qᵛ⁺ᵏ in qᵛ⁺]

fig = Figure()

yticks = 0:2e3:10e3
axT = Axis(fig[1, 1]; xlabel="Temperature (ᵒK)", ylabel="Height (m)", yticks)
axq⁺ = Axis(fig[1, 2]; xlabel="Saturation \n specific humidity \n (kg kg⁻¹)",
                       yticks, yticklabelsvisible=false)
axqˡ = Axis(fig[1, 3]; xlabel="Liquid \n specific humidity \n (kg kg⁻¹)",
                       yticks, yticklabelsvisible=false)

lines!(axT, T, z)
lines!(axq⁺, qᵛ⁺, z)
lines!(axqˡ, qˡ, z)

fig
```
