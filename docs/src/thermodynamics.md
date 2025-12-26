# [Atmosphere Thermodynamics](@id Thermodynamics-section)

```@setup thermo
using Breeze
thermo = ThermodynamicConstants()
```

Breeze implements thermodynamic relations for moist atmospheres.
By "moist", we mean that the atmosphere is a mixture of four components:
two gas phases _(i)_ "dry" air and _(ii)_ "vapor", and two "condensed" phases _(iii)_ "liquid", and _(iv)_ "ice".
Moisture makes _life_ interesting, because vapor can _condense_ or _solidify_
(and liquid can _freeze_) into liquid droplets and ice particles - with major consequences.

On Earth, dry air is itself a mixture of gases with fixed composition
constant [molar mass](https://en.wikipedia.org/wiki/Molar_mass).
Dry air on Earth's is mostly nitrogen, oxygen, and argon, whose combination produces the typical
(and Breeze's default) dry air molar mass

```julia
using Breeze
thermo = ThermodynamicConstants()
thermo.dry_air.molar_mass
```

The vapor, liquid, and ice components are ``\mathrm{H_2 O}``, also known as "water".
Water vapor, which in Breeze has the default molar mass

```julia
thermo.vapor.molar_mass
```

is lighter than dry air. As a result, moist, humid air is _lighter_ than dry air.

Liquid in Earth's atmosphere consists of falling droplets that range from tiny, nearly-suspended mist particles
to careening fat rain drops.
Ice in Earth's atmosphere consists of crystals, graupel, sleet, hail and, snow.

## "Moist" thermodynamic relations for a four-component mixture

What does it mean that moist air is a mixture of four components?
It means that the total mass ``\mathcal{M}`` of air per volume, or _density_, ``ρ``,
can be expressed as the sum of the masses of the individual components over the total volume ``V``,

```math
ρ = \frac{\mathcal{M}}{V} = \frac{\mathcal{M}ᵈ + \mathcal{M}ᵛ + \mathcal{M}ˡ + \mathcal{M}ⁱ}{V} = ρᵈ + ρᵛ + ρˡ + ρⁱ
```

where ``\mathcal{M}ᵈ``, ``\mathcal{M}ᵛ``, ``\mathcal{M}ˡ``, and ``\mathcal{M}ⁱ`` are the masses of dry air, vapor, liquid, and ice, respectively, while ``ρᵈ``, ``ρᵛ``, ``ρˡ``, and ``ρⁱ`` denote their _fractional densities_.
We likewise define the _mass fractions_ of each component,

```math
qᵈ ≡ \frac{\mathcal{M}ᵈ}{\mathcal{M}} = \frac{ρᵈ}{ρ} , \qquad qᵛ ≡ \frac{\mathcal{M}ᵛ}{\mathcal{M}} = \frac{ρᵛ}{ρ} , \qquad
qˡ ≡ \frac{\mathcal{M}ˡ}{\mathcal{M}} = \frac{ρˡ}{ρ}, \qquad \text{and} \qquad qⁱ ≡ \frac{ρⁱ}{ρ} = \frac{\mathcal{M}ⁱ}{\mathcal{M}} .
```

!!! note "The significance of certain superscripts"
    Throughout this documentation, superscripts are used to distinguish the components of moist air:
    - ``d`` denotes "dry air"
    - ``v`` denotes "vapor"
    - ``l`` denotes "liquid"
    - ``i`` denotes "ice"

    A fifth super script ``t`` is used to denote "total".
    For example, ``qᵈ`` is the mass fraction of dry air, ``qᵛ`` is the mass fraction of vapor,
    ``qˡ`` is the mass fraction of liquid, ``qⁱ`` is the mass fraction of ice,
    and

    ```math
    qᵗ = qᵛ + qˡ + qⁱ
    ```

    is the "total" mass fraction of the moisture components.

The liquid and ice components are not always present.
For example, a model with warm-phase microphysics does not have ice.
With no microphysics at all, there is no liquid _or_ ice.

By definition, all of the mass fractions sum up to unity,

```math
1 = qᵈ + qᵛ + qˡ + qⁱ ,
```

so that, using ``qᵗ = qᵛ + qˡ + qⁱ``, the dry air mass fraction can be diagnosed with ``qᵈ = 1 - qᵗ``.
The sometimes tedious bookkeeping required to correctly diagnose the effective mixture properties
of moist air are facilitated by Breeze's handy [`MoistureMassFractions`](@ref Breeze.Thermodynamics.MoistureMassFractions) abstraction.
For example,

```julia
q = Breeze.Thermodynamics.MoistureMassFractions(0.01, 0.002, 1e-5)
```

from which we can compute the total moisture mass fraction,

```julia
qᵗ = Breeze.Thermodynamics.total_specific_moisture(q)
```

And the dry as well,

```julia
qᵈ = Breeze.Thermodynamics.dry_air_mass_fraction(q)
```

To be sure,

```julia
qᵈ + qᵗ
```

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

```julia
using Breeze.Thermodynamics: IdealGas
dry_air = IdealGas(molar_mass=0.029, heat_capacity=1005)
```

We can also change the properties of dry air by specifying new values
when constructing `ThermodynamicConstants`,

```julia
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

```julia
thermo.gravitational_acceleration
```

By combining the hydrostatic balance with the ideal gas law and the definition of potential
temperature we get

```math
\frac{pᵣ}{p₀} = \left (1 - \frac{g z}{cᵖ θ₀} \right )^{cᵖ / R} .
```

Thus

```math
\begin{align*}
Tᵣ(z) & = θ₀ \left ( \frac{pᵣ}{p₀} \right )^{R / cᵖ} \\
      & = θ₀ - \frac{g}{cᵖ} z,
\end{align*}
```

and

```math
ρᵣ(z) = \frac{p₀}{Rᵈ θ₀} \left ( 1 - \frac{g z}{cᵖ θ₀} \right )^{cᵖ / R - 1} .
```

The quantity ``g / cᵖ ≈ 9.76 \;\mathrm{K}\,\mathrm{km}^{-1}`` that appears above is also referred to as
the "[dry adiabatic lapse rate](https://en.wikipedia.org/wiki/Lapse_rate)".

### An example of a dry reference state in Breeze

We can visualise a hydrostatic reference profile evaluating Breeze's reference-state
utilities (which assume a dry reference state) on a one-dimensional [`RectilinearGrid`](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Grids.RectilinearGrid).
In the following code, the superscript ``d`` denotes dry air, e.g., an ideal gas
with ``Rᵈ = 286.71 \; \mathrm{J} \, \mathrm{K}^{-1}``:

```julia
using Breeze
using CairoMakie

grid = RectilinearGrid(size=160, z=(0, 12_000), topology=(Flat, Flat, Bounded))
thermo = ThermodynamicConstants()
reference_state = ReferenceState(grid, thermo, surface_pressure=101325, potential_temperature=288)

pᵣ = reference_state.pressure
ρᵣ = reference_state.density

Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermo)
cᵖᵈ = thermo.dry_air.heat_capacity
p₀ = reference_state.surface_pressure
θ₀ = reference_state.potential_temperature
g = thermo.gravitational_acceleration

# Verify that Tᵣ = θ₀ - (g / cᵖᵈ) * z
z = KernelFunctionOperation{Center, Center, Center}(znode, grid, Center(), Center(), Center())
Tᵣ₁ = Field(θ₀ * (pᵣ / p₀)^(Rᵈ / cᵖᵈ))
Tᵣ₂ = Field(θ₀ - (g / cᵖᵈ) * z)

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

## The gaseous nature of moist air

To define the gaseous nature of moist air - that is, the equation of state relating density and pressure,
we assume that the _volume_ of liquid and ice components are negligible.
As a result, moist air pressure is the sum of partial pressures of vapor and dry air with no
contribution from liquid or ice phases,

```math
p = pᵈ + pᵛ .
```

Because the dry air and vapor components are ideal gases, their densities are related to pressure through the
ideal gas law,

```math
pᵈ = ρᵈ Rᵈ T \qquad \text{and} \qquad pᵛ = ρᵛ Rᵛ T ,
```

where ``T`` is temperature, ``Rⁱ = ℛ / m^β`` is the specific gas constant for component ``β``,
``m^β`` is the molar mass of component ``β``, and
``ℛ``  is the [molar or "universal" gas constant](https://en.wikipedia.org/wiki/Gas_constant),

```julia
thermo = ThermodynamicConstants()
thermo.molar_gas_constant
```

[`ThermodynamicConstants`](@ref), which is central to Breeze's implementation of moist thermodynamics.
holds constants like the molar gas constant and molar masses, latent heats, gravitational acceleration, and more,

```julia
thermo
```

These default values evince basic facts about water vapor air typical to Earth's atmosphere:
for example, the molar masses of dry air (itself a mixture of mostly nitrogen, oxygen, and argon),
and water vapor are ``mᵈ = 0.029 \; \mathrm{kg} \, \mathrm{mol}^{-1}`` and ``mᵛ = 0.018 \; \mathrm{kg} \, \mathrm{mol}^{-1}``.
And even more interesting, the triple point temperature and pressure of water vapor are

```julia
thermo.triple_point_temperature, thermo.triple_point_pressure
```

not so far from the typical conditions we experience on Earth's surface - one of the reasons that things are
so interesting down here. Also, that temperature is not a typo: the triple point temperature really is just
``+0.01^\circ``C.


It's then convenient to introduce the "mixture" gas constant ``Rᵐ(qᵛ)`` such that

```math
p = ρ Rᵐ T, \qquad \text{where} \qquad Rᵐ ≡ qᵈ Rᵈ + qᵛ Rᵛ .
```

To illustrate, let's compute the mixture gas constant ``Rᵐ`` for air with a small amount of water vapor.
The contribution of vapor increases ``Rᵐ`` above the dry air value:

```julia
q = Breeze.Thermodynamics.MoistureMassFractions(0.01, 0.0, 0.0) # 1% vapor by mass
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermo)
Rᵐ = Breeze.Thermodynamics.mixture_gas_constant(q, thermo)
Rᵐ - Rᵈ # shows the uplift from the vapor component
```

A small increase in specific humidity increases the effective gas constant of air.

## The thermal properties of moist air

Though we neglect the volume of liquid and ice, we do not neglect their _mass_ or _energy_.
The heat capacity of moist air thus includes contributions from all four components,

```math
cᵖᵐ = qᵈ cᵖᵈ + qᵛ cᵖᵛ + qˡ cˡ + qⁱ cⁱ,
```

where the ``cᵖᵝ`` denote the specific heat capacity at constant pressure of
constituent ``β``, and we have neglected the superscript ``p`` for liquid
and ice because they are assumed incompressible (their specific heats and constant
pressure or volume are the same).
We call ``cᵖᵐ`` the "mixture heat capacity", and because with default parameters the
heat capacity of dry air is the smallest of either vapor, liquid, or ice,
any moisture at all tends to increase the mixture heat capacity,

```julia
q = Breeze.Thermodynamics.MoistureMassFractions(0.01, 0.0, 0.0)
cᵖᵈ = thermo.dry_air.heat_capacity
cᵖᵐ = Breeze.Thermodynamics.mixture_heat_capacity(q, thermo)
cᵖᵐ - cᵖᵈ
```


## The Clausius--Clapeyron relation and saturation vapor pressure

The [Clausius--Clapeyron relation](https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation)
for an ideal gas describes how saturation vapor pressure changes with temperature:

```math
\frac{\mathrm{d} pᵛ⁺}{\mathrm{d} T} = \frac{pᵛ⁺ ℒ^β(T)}{Rᵛ T^2} ,
```

where ``pᵛ⁺`` is saturation vapor pressure over a surface of condensed phase ``β``,
``T`` is temperature, ``Rᵛ`` is the specific gas constant for vapor, and
``ℒ^β(T)`` is the latent heat of the phase transition from vapor to phase ``β``.
For atmospheric moist air, the relevant condensed phases are liquid water (``β = l``)
and ice (``β = i``).

### Temperature-dependent latent heat

For a thermodynamic formulation that uses constant (i.e. temperature-independent) specific
heats, the latent heat of a phase transition is linear in temperature:

```math
ℒ^β(T) = ℒ^β_0 + \Delta c^β \, T ,
```

where ``ℒ^β_0 ≡ ℒ^β(T=0)`` is the latent heat at absolute zero and
``\Delta c^β ≡ c_p^v - c^β`` is the constant difference between the vapor specific heat
capacity at constant pressure and the specific heat capacity of the condensed phase ``β``.

Note that we typically parameterize the latent heat in terms of a reference
temperature ``T_r`` that is well above absolute zero. In that case,
the latent heat is written

```math
ℒ^β(T) = ℒ^β_r + \Delta c^β (T - T_r) \qquad \text{and} \qquad
ℒ^β_0 = ℒ^β_r - \Delta c^β T_r ,
```

where ``ℒ^β_r`` is the latent heat at the reference temperature ``T_r``.

### Integration of the Clausius-Clapeyron relation

To find the saturation vapor pressure as a function of temperature, we integrate
the Clausius-Clapeyron relation with the temperature-linear latent heat model
from the triple point pressure and temperature ``(p^{tr}, T^{tr})`` to a generic
pressure ``pᵛ⁺`` and temperature ``T``:

```math
\int_{p^{tr}}^{pᵛ⁺} \frac{\mathrm{d} p}{p} = \int_{T^{tr}}^{T} \frac{ℒ^β_0 + \Delta c^β T'}{Rᵛ T'^2} \, \mathrm{d} T' .
```

Evaluating the integrals yields

```math
\log\left(\frac{pᵛ⁺}{p^{tr}}\right) = -\frac{ℒ^β_0}{Rᵛ T} + \frac{ℒ^β_0}{Rᵛ T^{tr}} + \frac{\Delta c^β}{Rᵛ} \log\left(\frac{T}{T^{tr}}\right) .
```

Exponentiating both sides gives the closed-form solution:

```math
pᵛ⁺(T) = p^{tr} \left ( \frac{T}{T^{tr}} \right )^{\Delta c^β / Rᵛ} \exp \left [ \frac{ℒ^β_0}{Rᵛ} \left (\frac{1}{T^{tr}} - \frac{1}{T} \right ) \right ] .
```

### Example: liquid water and ice parameters

Consider parameters for liquid water,

```julia
using Breeze.Thermodynamics: CondensedPhase
liquid_water = CondensedPhase(reference_latent_heat=2500800, heat_capacity=4181)
```

and water ice,

```julia
water_ice = CondensedPhase(reference_latent_heat=2834000, heat_capacity=2108)
```

These represent the latent heat of vaporization at the reference temperature and
the specific heat capacity of each condensed phase. We can compute the specific heat
difference ``\Delta c^β`` for liquid water:

```julia
using Breeze.Thermodynamics: vapor_gas_constant
cᵖᵛ = thermo.vapor.heat_capacity
cˡ = thermo.liquid.heat_capacity
Δcˡ = cᵖᵛ - cˡ
```

This difference ``\Delta c^l`` above is negative because water vapor has a lower heat
capacity than liquid water.

### Mixed-phase saturation vapor pressure

In atmospheric conditions near the freezing point, condensate may exist as a mixture of
liquid and ice. Following [Pressel2015](@citet), we model the saturation vapor pressure
over a mixed-phase surface using a liquid fraction ``λ`` that varies smoothly between
0 (pure ice) and 1 (pure liquid). The effective latent heat and specific heat difference
for the mixture are computed as weighted averages:

```math
ℒ^{li}_0 = λ \, ℒ^l_0 + (1 - λ) \, ℒ^i_0 ,
```

```math
\Delta c^{li} = λ \, \Delta c^l + (1 - λ) \, \Delta c^i .
```

These effective properties are then used in the Clausius-Clapeyron formula to compute
the saturation vapor pressure over the mixed-phase surface. This approach ensures
thermodynamic consistency and smooth transitions between pure liquid and pure ice states.

We can illustrate this by computing the mixed-phase specific heat difference for a
50/50 mixture:

```julia
Δcⁱ = thermo.vapor.heat_capacity - thermo.ice.heat_capacity
λ = 0.5
Δcˡⁱ = λ * Δcˡ + (1 - λ) * Δcⁱ
```

### Visualizing saturation vapor pressure

The saturation vapor pressure over liquid, ice, and mixed-phase surfaces can be computed
and visualized:

```julia
using Breeze
using Breeze.Thermodynamics: saturation_vapor_pressure, PlanarMixedPhaseSurface

thermo = ThermodynamicConstants()

T = collect(200:0.1:320)
pᵛˡ⁺ = [saturation_vapor_pressure(Tⁱ, thermo, thermo.liquid) for Tⁱ in T]
pᵛⁱ⁺ = [saturation_vapor_pressure(Tⁱ, thermo, thermo.ice) for Tⁱ in T]
pᵛⁱ⁺[T .> thermo.triple_point_temperature] .= NaN

# Mixed-phase surface with 50% liquid, 50% ice
mixed_surface = PlanarMixedPhaseSurface(0.5)
pᵛᵐ⁺ = [saturation_vapor_pressure(Tⁱ, thermo, mixed_surface) for Tⁱ in T]

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Temperature (ᵒK)", ylabel="Saturation vapor pressure pᵛ⁺ (Pa)",
          yscale = log10, xticks=200:20:320)
lines!(ax, T, pᵛˡ⁺, label="liquid", linewidth=2)
lines!(ax, T, pᵛⁱ⁺, label="ice", linestyle=:dash, linewidth=2)
lines!(ax, T, pᵛᵐ⁺, label="mixed (λ=0.5)", linestyle=:dot, linewidth=2, color=:purple)
axislegend(ax, position=:rb)
fig
```

The mixed-phase saturation vapor pressure lies between the liquid and ice curves,
providing a smooth interpolation between the two pure phases.

## Saturation specific humidity

The saturation specific humidity ``qᵛ⁺`` is the maximum amount of water vapor that
can exist in equilibrium with a condensed phase at a given temperature and density.
It is related to the saturation vapor pressure by:

```math
qᵛ⁺ ≡ \frac{ρᵛ⁺}{ρ} = \frac{pᵛ⁺}{ρ Rᵛ T} ,
```

where ``ρᵛ⁺`` is the vapor density at saturation, ``ρ`` is the total air density,
and ``Rᵛ`` is the specific gas constant for water vapor.

### Visualizing saturation vapor pressure and specific humidity

We can visualize how both saturation vapor pressure and saturation specific humidity
vary with temperature for different liquid fractions, demonstrating the smooth
interpolation provided by the mixed-phase model:

```julia
using Breeze
using Breeze.Thermodynamics: saturation_vapor_pressure, saturation_specific_humidity, PlanarMixedPhaseSurface

thermo = ThermodynamicConstants()

# Temperature range covering typical atmospheric conditions
T = collect(250:0.1:320)
p₀ = 101325  # Surface pressure (Pa)
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermo)

# Liquid fractions to visualize
λ_values = [0.0, 0.25, 0.5, 0.75, 1.0]
labels = ["ice (λ=0)", "λ=0.25", "λ=0.5", "λ=0.75", "liquid (λ=1)"]
colors = [:blue, :cyan, :purple, :orange, :red]
linestyles = [:solid, :dash, :dot, :dashdot, :solid]

using CairoMakie

fig = Figure(size=(1000, 400))

# Panel 1: Saturation vapor pressure
ax1 = Axis(fig[1, 1], xlabel="Temperature (K)", ylabel="Saturation vapor pressure (Pa)",
           yscale=log10, title="Saturation vapor pressure")

for (i, λ) in enumerate(λ_values)
    surface = PlanarMixedPhaseSurface(λ)
    pᵛ⁺ = [saturation_vapor_pressure(Tⁱ, thermo, surface) for Tⁱ in T]
    lines!(ax1, T, pᵛ⁺, label=labels[i], color=colors[i], linestyle=linestyles[i], linewidth=2)
end

axislegend(ax1, position=:lt)

# Panel 2: Saturation specific humidity
ax2 = Axis(fig[1, 2], xlabel="Temperature (K)", ylabel="Saturation specific humidity (kg/kg)",
           title="Saturation specific humidity")

for (i, λ) in enumerate(λ_values)
    surface = PlanarMixedPhaseSurface(λ)
    qᵛ⁺ = zeros(length(T))
    for (j, Tⁱ) in enumerate(T)
        ρ = p₀ / (Rᵈ * Tⁱ)  # Approximate density using dry air
        qᵛ⁺[j] = saturation_specific_humidity(Tⁱ, ρ, thermo, surface)
    end
    lines!(ax2, T, qᵛ⁺, label=labels[i], color=colors[i], linestyle=linestyles[i], linewidth=2)
end

fig
```

This figure shows how the liquid fraction ``λ`` smoothly interpolates between pure ice
(``λ = 0``) and pure liquid (``λ = 1``). At lower temperatures, the differences between
phases are more pronounced. The mixed-phase model allows for realistic representation of
conditions near the freezing point where both liquid and ice may coexist.

## Moist static energy

For moist air, a convenient thermodynamic invariant that couples temperature, composition, and height is the moist static energy (MSE),

```math
e ≡ cᵖᵐ \, T + g z - Lˡᵣ \, qˡ - Lⁱᵣ qⁱ .
```

!!! note "The alternative 'frozen moist static energy' variable"

    An alternative, physically equivalent, definition of moist static energy used in atmospheric
    models such as the Global System for Atmospheric Modeling (GSAM) [Khairoutdinov2022](@cite) is

    ```math
    ẽ ≡ cᵖᵐ \, T + g z + Lˡᵣ \, qᵛ - Lᶠᵣ qⁱ .
    ```

    ``e`` and ``ẽ`` are not the same, but they obey the same conservation equation provided
    that total moisture fraction is conserved, or that ``\mathrm{D}qᵗ / \mathrm{D}t = 0``.

## Liquid-ice potential temperature
