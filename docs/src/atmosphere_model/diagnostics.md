# Diagnostics

Breeze.jl provides a variety of diagnostic fields for analyzing atmospheric simulations.
These diagnostics can be computed as `KernelFunctionOperation`s,
which allows them to be evaluated lazily or wrapped in `Field`s for storage and output.

## Naming conventions

Diagnostic functions follow a naming convention that indicates their return type:

- **`TitleCase` names** (e.g., `VirtualPotentialTemperature`, `StaticEnergy`): These functions
  *always* return a `KernelFunctionOperation`.
  These are "pure diagnostics" that are computed on-the-fly from the model state.

- **`snake_case` names** (e.g., `temperature`, `density`): These functions may return either
  a `Field` or a `KernelFunctionOperation`, depending on
  the model formulation. When a quantity is directly stored or computed by a particular
  [`AtmosphereModel`](@ref) formulation, it is returned as a `Field`. Otherwise, it is
  computed on-the-fly as a `KernelFunctionOperation`.

This convention helps users understand whether a diagnostic is a stored model variable
or a derived quantity.

## Potential temperatures

Potential temperatures are conserved quantities that are useful for diagnosing atmospheric stability
and identifying air masses. See the [notation appendix](@ref "Notation and conventions") for
the symbols used (``θ``, ``θᵛ``, ``θᵉ``, ``θˡⁱ``, ``θᵇ``).

### Potential temperature (mixture)

```@docs
Breeze.AtmosphereModels.PotentialTemperature
```

### Virtual potential temperature

```@docs
Breeze.AtmosphereModels.VirtualPotentialTemperature
```

### Equivalent potential temperature

```@docs
Breeze.AtmosphereModels.EquivalentPotentialTemperature
```

### Liquid-ice potential temperature

```@docs
Breeze.AtmosphereModels.LiquidIcePotentialTemperature
```

### Stability-equivalent potential temperature

```@docs
Breeze.AtmosphereModels.StabilityEquivalentPotentialTemperature
```

## Static energy

Static energy is another conserved thermodynamic variable used in atmospheric models.
It combines sensible heat, gravitational potential energy, and latent heat contributions.

```@docs
Breeze.AtmosphereModels.StaticEnergy
```

## Saturation specific humidity

The saturation specific humidity is a key diagnostic for understanding moisture in the atmosphere.
It represents the maximum amount of water vapor that air can hold at a given temperature and pressure.

```@docs
Breeze.Microphysics.SaturationSpecificHumidity
```
