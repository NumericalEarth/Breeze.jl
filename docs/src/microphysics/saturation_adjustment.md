# [Warm-phase saturation adjustment](@id saturation_adjustment-section)

Warm-phase saturation adjustment is a model for water droplet nucleation that assumes that water vapor in excess of the saturation specific humidity is instantaneously converted to liquid water.
Mixed-phase saturation adjustment is described by [Pressel2015](@citet).

In Breeze, it is convenient to express the adjustment in terms of a moist static energy variable rather than potential temperature. Following the anelastic formulation of [Pauluis2008](@citet) used elsewhere in the docs, we define the energy variable

```math
e \equiv c^{p m} \, T + g z - ℒˡᵣ qˡ ,
```

where ``cᵖᵐ`` is the mixture heat capacity, ``T`` is temperature, ``g`` is gravitational acceleration,
``z`` is height, ``ℒˡᵣ`` is the latent heat at the energy reference temperature, and ``qˡ`` is the liquid mass fraction.

```math
r(T) \equiv T - \left ( e - g z + ℒˡᵣ qˡ \right ) / cᵖᵐ ,
```

with ``q^{v+}(T)`` the saturation specific humidity. We use a secant method, typically starting from the clear‑air guess ``T_1 = (e - g z)/c^{p m}`` and adjusting toward saturation when ``q^{t} > q^{v+}(T_1)``.

As an example, we consider an air parcel at sea level within a reference state with base pressure of 101325 Pa and a surface temperature ``T₀ = 288``ᵒK.
The saturation specific humidity is then

```@example microphysics
using Breeze
using Breeze.Thermodynamics: saturation_specific_humidity

thermo = ThermodynamicConstants()

p₀ = 101325.0
T₀ = 288.0
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermo)
ρ₀ = p₀ / (Rᵈ * T₀)
qᵛ⁺₀ = saturation_specific_humidity(T₀, ρ₀, thermo, thermo.liquid)
```

Recall that the specific humidity is unitless, or has units "``kg / kg``": kg of water vapor
per total kg of air.
We then perform a non-trivial saturation adjustment by computing temperature
given a total specific humidity slightly above saturation:

```@example microphysics
using Breeze.Thermodynamics: MoistureMassFractions, mixture_heat_capacity, saturation_vapor_pressure
using Breeze.Thermodynamics: MoistStaticEnergyState
using Breeze.AtmosphereModels: compute_temperature
using Breeze.Microphysics: WarmPhaseSaturationAdjustment

z = 0.0
qᵗ = 0.012   # [kg kg⁻¹] total specific humidity
q = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))

# e = cᵖᵐ T + g z − ℒᵥ₀ qˡ with T ≈ T₀ and qˡ = 0 at the surface
cᵖᵐ = mixture_heat_capacity(q, thermo)
g = thermo.gravitational_acceleration
ℒᵥ₀ = thermo.liquid.reference_latent_heat
e = cᵖᵐ * T₀ + g * z - ℒᵥ₀ * 0

U = MoistStaticEnergyState(e, q, z, p₀)
μ = WarmPhaseSaturationAdjustment()
T = compute_temperature(U, μ, thermo)
```

Finally, we recover the amount of liquid condensate by subtracting the saturation
specific humidity from the total:

```@example microphysics
# Compute qᵛ⁺ from T and p₀ using the microphysics adjustment formula
pᵛ⁺ = saturation_vapor_pressure(T, thermo, thermo.liquid)
ϵᵈᵛ = Breeze.Thermodynamics.dry_air_gas_constant(thermo) / Breeze.Thermodynamics.vapor_gas_constant(thermo)
qᵛ⁺ = ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (p₀ - pᵛ⁺)
qˡ = qᵗ - qᵛ⁺
```

### Saturation adjustment with varying total specific humidity

As a second example, we examine the dependence of temperature on total specific humidity
when the moist static energy is held fixed (equivalently, ``θ`` is held fixed for this reference state):

```@example microphysics
using Breeze.Thermodynamics: MoistureMassFractions, MoistStaticEnergyState, mixture_heat_capacity
using Breeze.Microphysics: WarmPhaseSaturationAdjustment
using Breeze.AtmosphereModels: compute_temperature

qᵗ = 0:1e-4:0.035 # [kg kg⁻¹] total specific humidity
z = 0.0

# Hold e approximately constant by fixing it at a baseline value
q_base = MoistureMassFractions(qᵛ⁺₀, 0.0, 0.0)
e₀ = mixture_heat_capacity(q_base, thermo) * T₀

T = similar(collect(qᵗ))
μ = WarmPhaseSaturationAdjustment()
for (i, qᵗⁱ) in enumerate(qᵗ)
    qⁱ = MoistureMassFractions(qᵗⁱ, 0.0, 0.0)
    Uⁱ = MoistStaticEnergyState(e₀, qⁱ, z, p₀)
    T[i] = compute_temperature(Uⁱ, μ, thermo)
end

## Compare with a simple piecewise linear model (approximation)
ℒᵥ₀ = thermo.liquid.reference_latent_heat
cᵖᵈ = thermo.dry_air.heat_capacity
T̃ = [T₀ + ℒᵥ₀ / cᵖᵈ * max(0, qᵗⁱ - qᵛ⁺₀) for qᵗⁱ in qᵗ]

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Total specific humidity (kg kg⁻¹)", ylabel="Temperature (ᵒK)")
lines!(ax, qᵗ, T, label="Temperature from saturation adjustment")
lines!(ax, qᵗ, T̃, label="Temperature from linearized formula", linewidth=2)
axislegend(ax, position=:lt)
fig
```

### Saturation adjustment with varying height

For a third example, we consider a state with constant moist static energy and total specific humidity
(equivalently, a constant ``θ`` in this reference state),
but at varying heights:

```@example microphysics
using Breeze.AtmosphereModels: compute_temperature
grid = RectilinearGrid(size=100, z=(0, 1e4), topology=(Flat, Flat, Bounded))
thermo = ThermodynamicConstants()
reference_state = ReferenceState(grid, thermo)

θ₀ = reference_state.potential_temperature
p₀ = reference_state.base_pressure
qᵗ = 0.005
q = MoistureMassFractions(qᵗ, 0.0, 0.0)

z = znodes(grid, Center())
T = zeros(grid.Nz)
qᵛ⁺ = zeros(grid.Nz)
qˡ = zeros(grid.Nz)
rh = zeros(grid.Nz)

# Set a constant moist static energy referenced to z = 0, clear air
cᵖᵐ = mixture_heat_capacity(q, thermo)
g = thermo.gravitational_acceleration
ℒᵥ₀ = thermo.liquid.reference_latent_heat
e₀ = cᵖᵐ * θ₀ + g * 0.0 - ℒᵥ₀ * 0.0

μ = WarmPhaseSaturationAdjustment()
for k = 1:grid.Nz
    pᵣ = reference_state.pressure[1, 1, k]
    U = MoistStaticEnergyState(e₀, q, z[k], pᵣ)
    T[k] = compute_temperature(U, μ, thermo)

    # Saturation specific humidity via adjustment formula using T[k], pᵣ, and qᵗ
    pᵛ⁺ = saturation_vapor_pressure(T[k], thermo, thermo.liquid)
    ϵᵈᵛ = Breeze.Thermodynamics.dry_air_gas_constant(thermo) / Breeze.Thermodynamics.vapor_gas_constant(thermo)
    qᵛ⁺[k] = ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
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
