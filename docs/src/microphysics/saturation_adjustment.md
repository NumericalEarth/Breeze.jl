# Warm-phase saturation adjustment

Warm-phase saturation adjustment is a model for water droplet nucleation that assumes that water vapor in excess of the saturation specific humidity is instantaneously converted to liquid water.
Mixed-phase saturation adjustment is described by [Pressel2015](@citet).
Saturation adjustment may be formulated as a nonlinear algebraic equation that relates temperature, potential temperature, and total specific humidity, derived from the definition of liquid potential temperature,

```math
θ = \frac{1}{Π} \left \{T - \frac{ℒᵥ₀}{cᵖᵐ} \max \left [0, qᵗ - qᵛ⁺(T) \right ] \right \} ,
```

where ``Π`` is the Exner function, ``θ`` is potential temperature, ``T`` is temperature,
``ℒᵥ₀`` is the reference latent heat of vaporization, ``qᵗ`` is the total specific humidity,
``qᵛ⁺`` is the saturation specific humidity, and ``cᵖᵐ`` is the moist air specific heat.
The condensate specific humidity is ``qˡ = \max(0, qᵗ - qᵛ⁺)``: ``qˡ = 0`` if the air is undersaturated with ``qᵗ < qᵛ⁺``.
Both ``Π`` and ``cᵖᵐ`` depend on the dry and vapor mass fractions ``qᵈ = 1 - qᵗ`` and
``qᵛ = qᵗ - qˡ``, and the saturation specific humidity``qᵛ⁺`` is an increasing function of temperature ``T``.

Rewriting the potential temperature relation above, saturation adjustment requires solving ``r(T) = 0``, where

```math
r(T) ≡ T - θ Π - \frac{ℒᵥ₀}{cᵖᵐ} \max[0, qᵗ - qᵛ⁺(T)] .
```

We use a secant method after checking for ``θ = 0`` and ``qˡ = 0`` given the guess ``T₁ = θ Π(qᵗ)``.
If ``qᵗ > qᵛ⁺(T₁)``, then we are guaranteed that ``T > T₁`` because ``qᵛ⁺`` is an increasing function of ``T``.
We initialize the secant iteration with a second guess ``T₂ = θ Π - [qᵗ - qᵛ⁺(T₁)] ℒᵥ₀ / cᵖᵐ``.
See [`temperature`](@ref Breeze.MoistAirBuoyancies.temperature) for more details.

As an example, we consider an air parcel at sea-level and with potential temperature of ``θ = 290``ᵒK,
within a reference state with base pressure of 101325 Pa and a reference potential temperature ``288``ᵒK.
The saturation specific humidity is then

```@example microphysics
using Breeze
using Breeze.Thermodynamics: saturation_specific_humidity

thermo = ThermodynamicConstants()

p₀ = 101325.0
θ₀ = 288.0
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermo)
ρ₀ = p₀ / (Rᵈ * θ₀)
qᵛ⁺₀ = saturation_specific_humidity(θ₀, ρ₀, thermo, thermo.liquid)
```

Recall that the specific humidity is unitless, or has units "``kg / kg``": kg of water vapor
per total kg of air.
We then perform a non-trivial saturation adjustment by computing temperature
given a total specific humidity slightly above saturation:

```@example microphysics
using Breeze.MoistAirBuoyancies: temperature
using Breeze.Thermodynamics: PotentialTemperatureState, MoistureMassFractions

z = 0.0
qᵗ = 0.012   # [kg kg⁻¹] total specific humidity
q = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
U = PotentialTemperatureState(θ₀, q, z, p₀, p₀, ρ₀)
T = temperature(U, thermo)
```

Finally, we recover the amount of liquid condensate by subtracting the saturation
specific humidity from the total:

```@example microphysics
qᵛ⁺ = saturation_specific_humidity(T, ρ₀, thermo, thermo.liquid)
qˡ = qᵗ - qᵛ⁺
```

### Saturation adjustment with varying total specific humidity

As a second example, we examine the dependence of temperature on total specific humidity
when the potential temperature is constant:

```@example microphysics
qᵗ = 0:1e-4:0.035 # [kg kg⁻¹] total specific humidity
q = [MoistureMassFractions(qᵗⁱ, 0.0, 0.0) for qᵗⁱ in qᵗ]
U = [PotentialTemperatureState(θ₀, qⁱ, z, p₀, p₀, ρ₀) for qⁱ in q]
T = [temperature(Uⁱ, thermo) for Uⁱ in U]

## Compare with a simple piecewise linear model
ℒᵥ₀ = thermo.liquid.reference_latent_heat
cᵖᵈ = thermo.dry_air.heat_capacity
T̃ = [288 + ℒᵥ₀ / cᵖᵈ * max(0, qᵗⁱ - qᵛ⁺₀) for qᵗⁱ in qᵗ]

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Total specific humidity (kg kg⁻¹)", ylabel="Temperature (ᵒK)")
lines!(ax, qᵗ, T, label="Temperature from saturation adjustment")
lines!(ax, qᵗ, T̃, label="Temperature from linearized formula", linewidth=2)
axislegend(ax, position=:lt)
fig
```

### Saturation adjustment with varying height

For a third example, we consider a state with constant potential temperature and total specific humidity,
but at varying heights:

```@example microphysics
grid = RectilinearGrid(size=100, z=(0, 1e4), topology=(Flat, Flat, Bounded))
thermo = ThermodynamicConstants()
reference_state = ReferenceState(grid, thermo)

θᵣ = reference_state.potential_temperature
p₀ = reference_state.base_pressure
qᵗ = 0.005
q = MoistureMassFractions(qᵗ, 0.0, 0.0)

z = znodes(grid, Center())
T = zeros(grid.Nz)
qᵛ⁺ = zeros(grid.Nz)
qˡ = zeros(grid.Nz)
rh = zeros(grid.Nz)

for k = 1:grid.Nz
    ρᵣ = reference_state.density[1, 1, k]
    pᵣ = reference_state.pressure[1, 1, k]

    U = PotentialTemperatureState(θᵣ, q, z[k], p₀, pᵣ, ρᵣ)
    T[k] = temperature(U, thermo)

    qᵛ⁺[k] = saturation_specific_humidity(T[k], ρᵣ, thermo, thermo.liquid)
    qˡ[k] = max(0, qᵗ - qᵛ⁺[k])
    rh[k] = 100 * min(qᵗ, qᵛ⁺[k]) / qᵛ⁺[k]
end

cᵖᵈ = thermo.dry_air.heat_capacity
g = thermo.gravitational_acceleration

fig = Figure(size=(700, 350))

yticks = 0:2e3:10e3

axT = Axis(fig[1, 1:2]; xlabel = "Temperature (ᵒK)", ylabel = "Height (m)", yticks)
axq⁺ = Axis(fig[1, 3]; xlabel = "Saturation\n specific humidity\n (kg kg⁻¹)",
                       yticks, yticklabelsvisible = false)
axqˡ = Axis(fig[1, 4]; xlabel = "Liquid\n specific humidity\n (kg kg⁻¹)",
                       yticks, yticklabelsvisible = false)

axrh = Axis(fig[1, 5]; xlabel = "Relative\n humidity (%)",
                       xticks = 0:20:100,
                       yticks, yticklabelsvisible = false)

lines!(axT, T, z)
lines!(axT, T[1] .- g * z / cᵖᵈ, z, linestyle = :dash, color = :orange, linewidth = 2)
lines!(axq⁺, qᵛ⁺, z)
lines!(axqˡ, qˡ, z)
lines!(axrh, rh, z)

fig
```
