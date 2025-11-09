# [Warm-phase saturation adjustment](@id saturation_adjustment-section)

Warm-phase saturation adjustment is a model for water droplet nucleation that assumes that water vapor in excess of the saturation specific humidity is instantaneously converted to liquid water.
Mixed-phase saturation adjustment is described by [Pressel2015](@citet).

## Equilirbium expressions for moist static energy and saturation specific humidity

`Breeze.AtmosphereModel` uses moist static energy density ``ρ e`` as a prognostic variable.
With warm-phase microphysics, the moist static energy ``e`` is defined

```math
e \equiv cᵖᵐ \, T + g z - ℒˡᵣ qˡ ,
```

where ``cᵖᵐ`` is the mixture heat capacity, ``T`` is temperature, ``g`` is gravitational acceleration,
``z`` is height, ``ℒˡᵣ`` is the latent heat at the energy reference temperature, and ``qˡ`` is the liquid mass fraction.

Saturation adjustment microphysics assumes that temperature and the moisture mass fractions
instantaneously adjust to an equilibrium in which the specific humidity is equal to
or less than the saturation specific humidity. This condition implies that the 
liquid mass fraction ``qˡ`` is

```math
qˡ = max(0, qᵗ - qᵛ⁺)
```

where ``qᵗ`` is the total moisture mass fraction, and ``qᵛ⁺`` is the saturation
specific humidity at the temperature ``T``. The saturation specific humidity is
defined as

```math
qᵛ⁺ = \frac{ρᵛ⁺}{ρ},
```

where ``ρᵛ⁺ = pᵛ⁺ / Rᵛ T`` is the density associated with the saturation vapor pressure ``pᵛ⁺``
and ``Rᵛ`` is the vapor gas constant. Note that the air density ``ρ`` itself depends
on the specific humidity, since

```math
ρ = \frac{pᵣ}{Rᵐ T} = \frac{pᵣ}{\left (qᵈ Rᵈ + qᵛ Rᵛ) T} ,
```

where ``qᵈ = 1 - qᵗ`` is the dry air mass fraction, ``qᵛ`` is the specific humidity,
``Rᵈ`` is the dry air gas constant, and ``Rᵛ`` is the vapor gas constant.
The density is expressed in terms of ``pᵣ`` under the anelastic approximation.

In saturated conditions, we have ``qᵛ ≡ qᵛ⁺`` by definition, which leads to the expression 

```math
qᵛ⁺ = \frac{ρᵛ⁺}{ρ} = \frac{Rᵐ}{Rᵛ} \frac{pᵛ⁺}{pᵣ} = ϵ \left ( 1 - qᵗ \right ) \frac{pᵛ⁺}{pᵣ} + qᵛ⁺ \frac{pᵛ⁺}{pᵣ}
```

where ``ϵ ≡ Rᵈ / Rᵛ``. Rearranging, we find a new expression for the saturation specific humidity which is
_valid only in saturated conditions and under the assumptions of saturation adjustment_,

```math
qᵛ⁺ = \frac{ϵ \left ( 1 - qᵗ \right ) pᵛ⁺}{pᵣ - pᵛ⁺} .
```

## Saturation adjustment algorithm

To compute temperature during saturation adjustment, we solve the nonlinear
algebraic ewquation

```math
0 = r(T) \equiv T - \frac{1}{cᵖᵐ} \left [ e - g z + ℒˡᵣ max(0, qᵗ - qᵛ⁺) \right ] \,
```

where ``r`` is the "residual", using a secant method.

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
