# Diagnostics

Breeze.jl provides a variety of diagnostic fields for analyzing atmospheric simulations.
These diagnostics can be computed as `KernelFunctionOperation`s, which allows them to be
evaluated lazily or wrapped in `Field`s for storage and output.

## Potential temperatures

Potential temperatures are conserved quantities that are useful for diagnosing atmospheric stability
and identifying air masses.

### Dry potential temperature

```@docs
Breeze.AtmosphereModels.DryPotentialTemperature
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
Breeze.AtmosphereModels.LiquidIcePotentialTemperatureField
```

## Static energy

Static energy is another conserved thermodynamic variable used in atmospheric models.

```@docs
Breeze.AtmosphereModels.StaticEnergy
Breeze.AtmosphereModels.StaticEnergyField
```

## Saturation specific humidity

The saturation specific humidity is a key diagnostic for understanding moisture in the atmosphere.

```@docs
Breeze.Microphysics.SaturationSpecificHumidity
```

