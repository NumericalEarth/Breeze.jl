# [Atmosphere Thermodynamics](@id Thermodynamics-section)

```@setup thermo
using Breeze
thermo = ThermodynamicConstants()
```

Breeze implements thermodynamic relations for moist atmospheres.
By "moist", we mean that the atmosphere is a binary mixture
of _(i)_ "dry" air, and _(ii)_ "vapor".
The presence of moisture makes life interesting, because vapor can _condense_ or _solidify_
(and liquid can _freeze_) into droplets and particles of so many shapes and sizes.

On Earth, dry air is itself a mixture of gases, the vapor component is ``\mathrm{H_2 O}``,
also known as "water".
Liquid droplets, which include almost-suspended tiny droplets as well as big raindrops,
and ice particles such as snow, graupel, and hail, are called "condensates".
Breeze models dry air as having a fixed composition with
constant [molar mass](https://en.wikipedia.org/wiki/Molar_mass).
Dry air on Earth's is mostly nitrogen, oxygen, and argon, whose combination produces the typical
(and Breeze's default) dry air molar mass

```@example thermo
using Breeze
thermo = ThermodynamicConstants()
thermo.dry_air.molar_mass
```

Water vapor, which in Breeze has the default molar mass

```@example thermo
thermo.vapor.molar_mass
```

is lighter than dry air.
As a result, moist, humid air is _lighter_ than dry air.

## Two laws for ideal gases

Both dry air and vapor are modeled as ideal gases, which means that
the [ideal gas law](https://en.wikipedia.org/wiki/Ideal_gas_law) relates
pressure ``p``, temperature ``T``, and density ``ρ``,

```math
p = ρ R T .
```

Above, ``R ≡ ℛ / m`` is the specific gas constant given the
[molar or "universal" gas constant](https://en.wikipedia.org/wiki/Gas_constant)
``ℛ ≈ 8.31 \; \mathrm{J} \, \mathrm{K}^{-1} \, \mathrm{mol}^{-1}`` and molar mass ``m`` of the gas species under consideration.

The [first law of thermodynamics](https://en.wikipedia.org/wiki/First_law_of_thermodynamics),
aka "conservation of energy", states that infinitesimal changes in
"heat content"[^1] ``\mathrm{d} \mathcal{H}`` are related to infinitesimal changes
in temperature ``\mathrm{d} T`` and pressure ``\mathrm{d} p`` according to:[^2]

```math
\mathrm{d} \mathcal{H} = cᵖ \mathrm{d} T - \frac{\mathrm{d} p}{ρ} ,
```

[^1]: ``\mathcal{H}`` is called [_enthalpy_](https://en.wikipedia.org/wiki/Enthalpy)

[^2]: The conservation of energy states that any external heat input into the gas must equal the sum
      of the change of the gas's internal energy and the work done by the gas, ``p \, \mathrm{d} V``.
      For atmospheric flows it's convenient to express everything per unit mass. Assuming the mass of
      the fluid is conserved, we have that the work done per unit mass is ``p \, \mathrm{d}(\rho^{-1})``
      and the internal energy per unit mass is ``\mathcal{I} ≡ cᵛ \mathrm{d} T``.
      Therefore, if ``\mathrm{d} \mathcal{H}`` is the change in heat content per unit mass,
      we have:

    ```math
    \mathrm{d} \mathcal{H} = cᵛ \mathrm{d} T + p \, \mathrm{d}(ρ^{-1}) .
    ```

    By utilizing the identity ``\mathrm{d}(p / ρ) = p \, \mathrm{d}(ρ^{-1}) + ρ^{-1} \mathrm{d}p`` and using
    the ideal gas, we can rewrite the above conservation law as:

    ```math
    \mathrm{d} \mathcal{H} = (cᵛ + R) \mathrm{d} T - ρ^{-1} \mathrm{d}p ,
    ```

    which is the expression in the main text after noting that the specific heat capacities under
    constant pressure and under constant volume are related via ``cᵖ ≡ cᵛ + R``.


where ``cᵖ`` is the specific heat capacity at constant pressure of the gas in question.

For example, to represent dry air typical for Earth, with molar mass ``m = 0.029 \; \mathrm{kg} \, \mathrm{mol}^{-1}`` and constant-pressure heat capacity ``c^p = 1005 \; \mathrm{J} \, \mathrm{kg}^{-1} \, \mathrm{K}^{-1}``,
we write

```@example thermo
using Breeze.Thermodynamics: IdealGas
dry_air = IdealGas(molar_mass=0.029, heat_capacity=1005)
```

We can also change the properties of dry air by specifying new values
when constructing `ThermodynamicConstants`,

```@example thermo
weird_thermo = ThermodynamicConstants(dry_air_molar_mass=0.042, dry_air_heat_capacity=420)
weird_thermo.dry_air
```

### Potential temperature and "adiabatic" transformations

Within adiabatic transformations, ``\mathrm{d} \mathcal{H} = 0``.
Then, combining the ideal gas law with conservation of energy yields

```math
\frac{\mathrm{d} T}{T} = \frac{R}{cᵖ} \frac{\mathrm{d} p}{p} ,
```

which implies that ``T ∼ ( p / p₀ )^{R / cᵖ}``,
where ``p₀`` is some reference pressure value.

As a result, the _potential temperature_, ``θ``, defined as

```math
θ ≡ T \big / \left ( \frac{p}{p₀} \right )^{R / cᵖ} = \frac{T}{Π} ,
```

remains constant under adiabatic transformations.
Notice that above, we also defined the Exner function, ``Π ≡ ( p / p₀ )^{R / cᵖ}``.

!!! note "About subscripts"
    The subscript "0" typically indicates some quantity evaluated at the surface ``z=0``.
    By convention, we tend to invoke constants that represent profiles evaluated at ``z=0``: i.e., ``p₀ = p(z=0)``, ``T₀ = T(z=0)``, etc.
    This implies that the potential temperature under adiabatic transformation is ``θ(z) = θ₀ = T₀``.

### Hydrostatic balance

Next we consider a reference state that does not exchange energy with its environment
(i.e., ``\mathrm{d} \mathcal{H} = 0``) and thus has constant potential temperature

```math
θ₀ = Tᵣ \left ( \frac{p₀}{pᵣ} \right )^{R / cᵖ} .
```

!!! note "Reference states"
    Subscripts ``r`` indicate a _reference_ state.
    The adiabatic, hydrostatically-balanced reference state in the process
    of elucidation presently has a ``z`` dependent reference pressure ``pᵣ(z)``,
    density ``ρᵣ(z)``, and temperature ``Tᵣ(z)``.
    This reference state also has a _constant_ potential temperature
    ``θ₀``, which we attempt to clarify by writing ``θ₀`` (since it's constant,
    it has the same value at ``z=0`` as at any height).
    We apologize that our notation differs from the usual in which
    ``0`` subscripts indicate "reference" (🤔) and ``00`` (🫣) means ``z=0``.

Hydrostatic balance requires

```math
∂_z pᵣ = - ρᵣ g ,
```

where ``g`` is gravitational acceleration, naturally by default

```@example thermo
thermo.gravitational_acceleration
```

By combining the hydrostatic balance with the ideal gas law and the definition of potential
temperature we get

```math
\frac{pᵣ}{p₀} = \left (1 - \frac{g z}{cᵖ θ₀} \right )^{cᵖ / R} .
```

Thus

```math
Tᵣ(z) = θ₀ \left ( \frac{pᵣ}{p₀} \right )^{R / cᵖ} = θ₀ \left ( 1 - \frac{g z}{cᵖ θ₀} \right ) ,
```

and

```math
ρᵣ(z) = \frac{p₀}{Rᵈ θ₀} \left ( 1 - \frac{g z}{cᵖ θ₀} \right )^{cᵖ / R - 1} .
```

## An example of a dry reference state in Breeze

We can visualise a hydrostatic reference profile evaluating Breeze's reference-state
utilities (which assume a dry reference state) on a one-dimensional `RectilinearGrid`.
In the following code, the superscript ``d`` denotes dry air, e.g., an ideal gas
with ``Rᵈ = 286.71 \; \mathrm{J} \, \mathrm{K}^{-1}``:

```@example reference_state
using Breeze
using CairoMakie

grid = RectilinearGrid(size=160, z=(0, 12_000), topology=(Flat, Flat, Bounded))
thermo = ThermodynamicConstants()
reference_state = ReferenceState(grid, thermo, base_pressure=101325, potential_temperature=288)

pᵣ = reference_state.pressure
ρᵣ = reference_state.density

Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermo)
cᵖᵈ = thermo.dry_air.heat_capacity
p₀ = reference_state.base_pressure
θ₀ = reference_state.potential_temperature
g = thermo.gravitational_acceleration

# Verify that Tᵣ = θ₀ (1 - g z / (cᵖᵈ θ₀))
z = KernelFunctionOperation{Center, Center, Center}(znode, grid, Center(), Center(), Center())
Tᵣ₁ = Field(θ₀ * (pᵣ / p₀)^(Rᵈ / cᵖᵈ))
Tᵣ₂ = Field(θ₀ * (1 - g * z / (cᵖᵈ * θ₀)))

fig = Figure()

axT = Axis(fig[1, 1]; xlabel = "Temperature (ᵒK)", ylabel = "Height (m)")
lines!(axT, Tᵣ₁)
lines!(axT, Tᵣ₂, linestyle = :dash, color = :orange, linewidth = 2)

axp = Axis(fig[1, 2]; xlabel = "Pressure (10⁵ Pa)", yticklabelsvisible = false)
lines!(axp, pᵣ / 1e5)

axρ = Axis(fig[1, 3]; xlabel = "Density (kg m⁻³)", yticklabelsvisible = false)
lines!(axρ, ρᵣ)

fig
```

## Thermodynamic relations for gaseous mixtures

"Moist air" is conceived to be a mixture of two gas phases: "dry air" (itself a mixture of gases)
and water vapor, as well as a collection of liquid droplet and solid ice particle "condensates".
We assume that the volume of the condensates is negligible, such that the total
pressure is the sum of partial pressures of vapor and dry air,

```math
p = pᵈ + pᵛ .
```

(Superscripts ``d`` and ``v`` denote dry air and vapor respectively.)

The partial pressure of the dry air and vapor components are related to the component densities
``ρᵈ`` and ``ρᵛ`` through the ideal gas law,

```math
pᵈ = ρᵈ Rᵈ T \qquad \text{and} \qquad pᵛ = ρᵛ Rᵛ T ,
```

where ``T`` is temperature, ``Rⁱ = ℛ / m^β`` is the specific gas constant for component ``β``,
``ℛ``  is the [molar or "universal" gas constant](https://en.wikipedia.org/wiki/Gas_constant),
and ``m^β`` is the molar mass of component ``β``.

Central to Breeze's implementation of moist thermodynamics is a struct that
holds parameters like the molar gas constant and molar masses,

```@example thermo
thermo = ThermodynamicConstants()
```

The default parameter evince basic facts about water vapor air typical to Earth's atmosphere:
for example, the molar masses of dry air (itself a mixture of mostly nitrogen, oxygen, and argon),
and water vapor are ``mᵈ = 0.029 \; \mathrm{kg} \, \mathrm{mol}^{-1}`` and ``mᵛ = 0.018 \; \mathrm{kg} \, \mathrm{mol}^{-1}``.

To write the effective gas law for moist air, we introduce the mass ratios, e.g., specific humidity and specific hydrometeor contents,

```math
qᵈ ≡ \frac{ρᵈ}{ρ} \qquad \text{and} \qquad qᵛ ≡ \frac{ρᵛ}{ρ} ,
```

where ``ρ`` is total density of the fluid including dry air, vapor, and condensates,
``ρᵈ`` is the density of dry air, and ``ρᵛ`` is the density of vapor.
It's then convenient to introduce the "mixture" gas constant ``Rᵐ(qᵛ)`` such that

```math
p = ρ Rᵐ T, \qquad \text{where} \qquad Rᵐ ≡ qᵈ Rᵈ + qᵛ Rᵛ .
```

In "clear" (not cloudy) air, we have that ``qᵈ = 1 - qᵛ``.
More generally, ``qᵈ = 1 - qᵛ - qᶜ``, where ``qᶜ`` is the total mass
ratio of condensed species. In most situations on Earth, ``qᶜ ≪ qᵛ``.

```@example thermo
using Breeze.Thermodynamics: MoistureMassFractions

# Compute mixture properties for air with 0.01 specific humidity
qᵗ = 0.01 # 1% water vapor by mass
q = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
Rᵐ = mixture_gas_constant(q, thermo)
```

We likewise define a mixture heat capacity via ``cᵖᵐ = qᵈ cᵖᵈ + qᵛ cᵖᵛ``,

```@example thermo
cᵖᵐ = mixture_heat_capacity(q, thermo)
```

## Liquid-ice potential temperature

## The Clausius--Clapeyron relation and saturation specific humidity

The [Clausius--Clapeyron relation](https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation)
for an ideal gas

```math
\frac{\mathrm{d} pᵛ⁺}{\mathrm{d} T} = \frac{pᵛ⁺ ℒ^β(T)}{Rᵛ T^2} ,
```

where ``pᵛ⁺`` is saturation vapor pressure, ``T`` is temperature, ``Rᵛ`` is the specific
gas constant for vapor, ``ℒ^β(T)`` is the latent heat of the transition from vapor to the
``β`` phase (e.g., ``β = l`` for vapor → liquid and ``β = i`` for vapor to ice).

For a thermodynamic formulation that uses constant (i.e. temperature-independent) specific
heats, the latent heat of a phase transition is linear in temperature.
For example, for phase change from vapor to liquid,

```math
ℒˡ(T) = ℒˡ(T=0) + \big ( \underbrace{cᵖᵛ - cˡ}_{≡Δcˡ} \big ) T ,
```

where ``ℒˡ(T=0)`` is the latent heat at absolute zero, ``T = 0 \; \mathrm{K}``.
By integrating from the triple-point temperature ``Tᵗʳ`` for which ``p(Tᵗʳ) = pᵗʳ``, we get

```math
pᵛ⁺(T) = pᵗʳ \left ( \frac{T}{Tᵗʳ} \right )^{Δcˡ / Rᵛ} \exp \left [ \frac{ℒˡ(T=0)}{Rᵛ} \left (\frac{1}{Tᵗʳ} - \frac{1}{T} \right ) \right ] .
```

Consider parameters for liquid water,

```@example thermo
using Breeze.Thermodynamics: CondensedPhase
liquid_water = CondensedPhase(reference_latent_heat=2500800, heat_capacity=4181)
```

or water ice,

```@example thermo
water_ice = CondensedPhase(reference_latent_heat=2834000, heat_capacity=2108)
```

The saturation vapor pressure is

```@example
using Breeze
using Breeze.Thermodynamics: saturation_vapor_pressure

thermo = ThermodynamicConstants()

T = collect(200:0.1:320)
pᵛˡ⁺ = [saturation_vapor_pressure(Tⁱ, thermo, thermo.liquid) for Tⁱ in T]
pᵛⁱ⁺ = [saturation_vapor_pressure(Tⁱ, thermo, thermo.solid) for Tⁱ in T]
pᵛⁱ⁺[T .> thermo.triple_point_temperature] .= NaN

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Temperature (ᵒK)", ylabel="Saturation vapor pressure pᵛ⁺ (Pa)", yscale = log10, xticks=200:20:320)
lines!(ax, T, pᵛˡ⁺, label="vapor pressure over liquid")
lines!(ax, T, pᵛⁱ⁺, linestyle=:dash, label="vapor pressure over ice")
axislegend(ax, position=:rb)
fig
```

The saturation specific humidity is

```math
qᵛ⁺ ≡ \frac{ρᵛ⁺}{ρ} = \frac{pᵛ⁺}{ρ Rᵐ T} .
```

and this is what it looks like:

```@example
using Breeze
using Breeze.Thermodynamics: saturation_specific_humidity

thermo = ThermodynamicConstants()

p₀ = 101325
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermo)
T = collect(273.2:0.1:313.2)
qᵛ⁺ = zeros(length(T))

for i = 1:length(T)
    ρ = p₀ / (Rᵈ * T[i])
    qᵛ⁺[i] = saturation_specific_humidity(T[i], ρ, thermo, thermo.liquid)
end

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Temperature (ᵒK)", ylabel="Saturation specific humidity qᵛ⁺ (kg kg⁻¹)")
lines!(ax, T, qᵛ⁺)
fig
```
