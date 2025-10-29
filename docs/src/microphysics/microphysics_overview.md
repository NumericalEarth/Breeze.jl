# Microphysics

```@setup microphysics
using Breeze
```

Breeze provides a collection of microphysics utilities that close the thermodynamics of moist
atmospheres whenever condensation is present.
These routines complement the large-scale dynamics
implemented in [`MoistAirBuoyancy`](@ref) and [`AtmosphereModel`](@ref Breeze.AtmosphereModels.AtmosphereModel) by
diagnosing temperatures, condensate loadings, and derived thermodynamic coefficients that respect
moist thermodynamic balances.

We currently only support phase changes via warm-phase saturation adjustment.
Future developments will expand to mixed-phase saturation adjustment and explicit rate-based
phase change.
