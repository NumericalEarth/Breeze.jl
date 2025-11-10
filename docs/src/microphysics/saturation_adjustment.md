# [Warm-phase saturation adjustment](@id saturation_adjustment-section)

Warm-phase saturation adjustment is a model for water droplet nucleation that assumes that water vapor
in excess of the saturation specific humidity is instantaneously converted to liquid water.
Mixed-phase saturation adjustment is described by [Pressel2015](@citet).

## Moist static energy and total moisture mass fraction

The saturation adjustment solver (specific to our anelastic formulation) takes four inputs:
    * moist static energy ``e``
    * total moisture mass fraction ``qᵗ``
    * height ``z``
    * reference pressure ``pᵣ``

Note that moist static energy density ``ρᵣ e`` and moisture density ``ρᵣ qᵗ``
are prognostic variables for `Breeze.AtmosphereModel` when using `AnelasticFormulation`,
where ``ρᵣ`` is the reference density.
With warm-phase microphysics, the moist static energy ``e`` is related to temperature ``T``,
height ``z``, and liquid mass fraction ``qˡ`` by

```math
e \equiv cᵖᵐ \, T + g z - ℒˡᵣ qˡ ,
```

where ``cᵖᵐ`` is the mixture heat capacity, ``g`` is gravitational acceleration,
and ``ℒˡᵣ`` is the latent heat at the energy reference temperature.

## Equilibrium expressions for moist static energy and saturation specific humidity

Saturation adjustment microphysics assumes that temperature and the moisture mass fractions
instantaneously adjust to an equilibrium in which the specific humidity is equal to
or less than the saturation specific humidity. This condition implies that the 
liquid mass fraction ``qˡ`` is

```math
qˡ = \max(0, qᵗ - qᵛ⁺)
```

where ``qᵗ`` is the total moisture mass fraction, and ``qᵛ⁺`` is the saturation
specific humidity at the temperature ``T``. The saturation specific humidity is
defined as

```math
qᵛ⁺ = \frac{ρᵛ⁺}{ρ},
```

where ``ρᵛ⁺ = pᵛ⁺ / Rᵛ T`` is the density associated with the saturation vapor pressure ``pᵛ⁺``
and ``Rᵛ`` is the vapor gas constant. Note that the air density ``ρ`` itself depends
on the specific humidity, since according to the ideal gas law,

```math
ρ = \frac{pᵣ}{Rᵐ T} = \frac{pᵣ}{\left (qᵈ Rᵈ + qᵛ Rᵛ \right ) T} ,
```

where ``qᵈ = 1 - qᵗ`` is the dry air mass fraction, ``qᵛ`` is the specific humidity,
``Rᵈ`` is the dry air gas constant, and ``Rᵛ`` is the vapor gas constant.
The density is expressed in terms of ``pᵣ`` under the anelastic approximation.

In saturated conditions, we have ``qᵛ ≡ qᵛ⁺`` by definition, which leads to the expression 

```math
qᵛ⁺ = \frac{ρᵛ⁺}{ρ} = \frac{Rᵐ}{Rᵛ} \frac{pᵛ⁺}{pᵣ} = \frac{Rᵈ}{Rᵛ} \left ( 1 - qᵗ \right ) \frac{pᵛ⁺}{pᵣ} + qᵛ⁺ \frac{pᵛ⁺}{pᵣ} .
```

Rearranging, we find a new expression for the saturation specific humidity which is
_valid only in saturated conditions and under the assumptions of saturation adjustment_,

```math
qᵛ⁺ = \frac{Rᵈ}{Rᵛ} \left ( 1 - qᵗ \right ) \frac{pᵛ⁺}{pᵣ - pᵛ⁺} .
```

## Saturation adjustment algorithm

We compute the saturation adjustment temperature by solving the nonlinear algebraic equation

```math
0 = r(T) \equiv T - \frac{1}{cᵖᵐ} \left [ e - g z + ℒˡᵣ \max(0, qᵗ - qᵛ⁺) \right ] \,
```

where ``r`` is the "residual", using a secant method.

As an example, we consider an air parcel at sea level within a reference state with base pressure of 101325 Pa and a surface temperature ``T₀ = 288``ᵒK.
We first compute the saturation specific humidity assuming a dry-air density,

```@example microphysics
using Breeze
using Breeze.Thermodynamics: saturation_specific_humidity

thermo = ThermodynamicConstants()

p = 101325.0
T = 314.0
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermo)
ρ = p / (Rᵈ * T)
qᵛ⁺₀ = saturation_specific_humidity(T, ρ, thermo, thermo.liquid)
```

Next, we compute the saturation specific humidity for moist air with
a carefully chosen moist air mass fraction,

```@example microphysics
using Breeze.Microphysics: adjustment_saturation_specific_humidity

# qᵗ = 0.012   # [kg kg⁻¹] total specific humidity
qᵗ = 0.05   # [kg kg⁻¹] total specific humidity
qᵛ⁺ = Breeze.Microphysics.adjustment_saturation_specific_humidity(T, p, qᵗ, thermo)
```

We have thus identified a situation in which ``qᵗ > qᵛ⁺``. Note that the saturation specific humidity
in moist air is higher than in dry air at the same temperature and pressure. This is because moist air
is less dense than dry air.

In equilibrium (and thus under the assumptions of saturation adjustment), the specific humidity is
``qᵛ = qᵛ⁺``, while the liquid mass fraction is

```@example microphysics
qˡ = qᵗ - qᵛ⁺ 
```

We can then compute moist static energy,

```@example microphysics
using Breeze.Thermodynamics: MoistureMassFractions

q = MoistureMassFractions(qᵛ⁺, qˡ, zero(qᵗ))
cᵖᵐ = mixture_heat_capacity(q, thermo)
g = thermo.gravitational_acceleration
z = 0.0
ℒˡᵣ = thermo.liquid.reference_latent_heat
e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ
```

We can use the saturation adjustment solver to recover the input temperature,
passing it an "unadjusted" moisture mass fraction,

```@example microphysics
using Breeze.Microphysics: WarmPhaseSaturationAdjustment, compute_temperature
microphysics = WarmPhaseSaturationAdjustment()

q₀ = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
𝒰 = Breeze.Thermodynamics.MoistStaticEnergyState(e, q₀, z, p)
T★, r₂ = compute_temperature(𝒰, microphysics, thermo)
```

```@example microphysics
using Breeze.Microphysics: saturation_adjustment_residual
T★ = compute_temperature(𝒰, microphysics, thermo)
saturation_adjustment_residual(T★, 𝒰, thermo)
```

The saturation adjustment solver is initialized with a guess corresponding
to the temperature in unsaturated conditions,

```@example microphysics
cᵖᵐ₁ = mixture_heat_capacity(q₀, thermo)
T₁ = (e - g * z) / cᵖᵐ₁
```

The difference between ``T₁`` and the solution ``T_\mathrm{eq}`` is
``T_\mathrm{eq} - T₁ = ℒˡᵣ qˡ / cᵖᵐ`` and is therefore strictly positive.
In other words, ``T₁`` represents a lower bound.

To generate a second guess for the secant solver, we start by estimating
the liquid mass fraction using the guess ``T = T₁``,

```@example  microphysics
qᵛ⁺₂ = adjustment_saturation_specific_humidity(T₁, p, qᵗ, thermo)
qˡ₁ = qᵗ - qᵛ⁺₂
```

In general, this represents an _overestimate_ of the liquid mass fraction,
because ``qᵛ⁺₂`` is underestimated by the too-low temperature ``T₁``.
We thus increment the first guess by half of the difference implied by the
estimate ``qˡ₁``,

```@example  microphysics
q₂ = MoistureMassFractions(qᵛ⁺₂, qˡ₁, zero(qᵗ))
cᵖᵐ₂ = mixture_heat_capacity(q₂, thermo)
ΔT = ℒˡᵣ * qˡ₁ / cᵖᵐ₂
T₂ = T₁ + ΔT / 2
```

The residual looks like

```@example microphysics
using Breeze.Microphysics: saturation_adjustment_residual
using CairoMakie

# T = 230:0.5:320
#T = 280:0.01:330
T = 310:0.01:320
r = [saturation_adjustment_residual(Tʲ, 𝒰, thermo) for Tʲ in T]
qᵛ⁺ = [adjustment_saturation_specific_humidity(Tʲ, p, qᵗ, thermo) for Tʲ in T]

fig = Figure()
axr = Axis(fig[1, 1], xlabel="Temperature (K)", ylabel="Saturation adjustment residual (K)")
axq = Axis(fig[2, 1], xlabel="Temperature (K)", ylabel="Estimated liquid fraction")
lines!(axr, T, r)
# scatter!(axr, 288, 0, marker=:star5, markersize=30, color=:tomato)
scatter!(axr, 314, 0, marker=:star5, markersize=30, color=:tomato)

lines!(axq, T, max.(0, qᵗ .- qᵛ⁺))

fig
```

There is a kink at the temperature wherein the estimated liquid mass fraction bottoms out.