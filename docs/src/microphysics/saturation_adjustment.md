# Warm-phase saturation adjustment {#microphysics-saturation-adjustment}

```@setup microphysics_sat
using CairoMakie
CairoMakie.activate!(type = "svg")
```

Warm-phase saturation adjustment is a model for water droplet nucleation that assumes that water vapor in excess of the saturation specific humidity is instantaneously converted to liquid water.
Mixed-phase saturation adjustment is described by [Chammas et al 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003619).
Saturation adjustment may be formulated as a nonlinear algebraic equation that relates temperature, potential temperature, and total specific humidity, derived from the definition of liquid potential temperature,

```math
θ = \frac{T}{Π} \left (1 - \frac{ℒᵥ₀ qˡ}{cᵖᵐ T} \right ) ,
```

where ``Π`` is the Exner function, ``θ`` is potential temperature, ``T`` is temperature,
``ℒᵥ₀`` is the reference latent heat of vaporization, ``qᵛ⁺`` is the saturation specific humidity,
``qˡ`` is the liquid water specific humidity, and ``cᵖᵐ`` is the moist air specific heat.
``Π`` and ``cᵖᵐ`` depend on the dry and vapor mass fraction, and ``ℒᵥ`` and ``qᵛ⁺`` depend
on temperature ``T``.
Rewriting the potential temperature relation, saturation adjustment requires solving ``r(T) = 0``,

```math
r(T) = T - θ Π - \frac{ℒᵥ₀ qˡ}{cᵖᵐ T}
```

We use a secant method.
As an example, we consider an air parcel at sea-level and with potential temperature of ``θ = 290ᵒ``K, within a reference state with base pressure of 101325 Pa and a reference potential temperature ``288ᵒ``K.
The saturation specific humidity is then

```@example microphysics
using Breeze
using Breeze.MoistAirBuoyancies: saturation_specific_humidity, HeightReferenceThermodynamicState

thermo = AtmosphereThermodynamics()
ref = ReferenceStateConstants(base_pressure=101325, potential_temperature=288)

z = 0.0    # [m] height
θ = 290.0  # [ᵒK] potential temperature
saturation_specific_humidity(θ, z, ref, thermo, thermo.liquid)
```

We can then perform a non-trivial saturation adjustment by computing temperature
given a total specific humidity slightly above saturation:

```@example microphysics
using Breeze.MoistAirBuoyancies: temperature

qᵗ = 0.012   # [kg kg⁻¹] total specific humidity 
U = HeightReferenceThermodynamicState(θ, qᵗ, z)
T = temperature(U, ref, thermo)
```

This performed saturation adjustment.
Finally, we recover the amount of liquid condensate by subtracting the saturation
specific humidity from the total:

```@example microphysics_set
qᵛ⁺ = saturation_specific_humidity(T, z, ref, thermo, thermo.liquid)
qˡ = qᵗ - qᵛ⁺
```

### Saturation adjustment with varying total specific humidity

As a second example, we examine the dependence of temperature on total specific humidity
when the potential temperature is constant:

```@example microphysics
using CairoMakie

qᵗ = collect(0:1e-4:0.02) # [kg kg⁻¹] total specific humidity 
U = [HeightReferenceThermodynamicState(θ, qᵗⁱ, z) for qᵗⁱ in qᵗ]
T = [temperature(Uⁱ, ref, thermo) for Uⁱ in U]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Total specific humidity (kg kg⁻¹)", ylabel="Temperature (K)")
lines!(ax, qᵗ, T)
fig
```

### Saturation adjustment with varying height

For a third example, we consider a state with constant potential temperature and total specific humidity,
but at varying heights:

```@example microphysics_set
using CairoMakie

qᵗ = 0.005
z = collect(0:100:10e3)
T = [temperature(HeightReferenceThermodynamicState(θ, qᵗ, zᵏ), ref, thermo) for zᵏ in z]
qᵛ⁺ = [saturation_specific_humidity(T[k], z[k], ref, thermo, thermo.liquid) for k = 1:length(z)]
qˡ = [max(0, qᵗ - qᵛ⁺ᵏ) for qᵛ⁺ᵏ in qᵛ⁺]

fig = Figure()

axT = Axis(fig[1, 1], xlabel="Temperature (K)", ylabel="Height (m)")
axq⁺ = Axis(fig[1, 2], xlabel="Saturation \n specific humidity \n (kg kg⁻¹)", ylabel="Height (m)")
axqˡ = Axis(fig[1, 3], xlabel="Liquid \n specific humidity \n (kg kg⁻¹)", ylabel="Height (m)")

lines!(axT, T, z)
lines!(axq⁺, qᵛ⁺, z)
lines!(axqˡ, qˡ, z)

fig
```
