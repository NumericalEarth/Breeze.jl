# [Mixed-phase saturation adjustment](@id section:mixed-phase-saturation-adjustment)

Mixed-phase saturation adjustment extends the warm-phase model to temperatures between the homogeneous
ice nucleation temperature and the freezing point, where condensate may exist as a mixture of liquid and ice.

We assume instantaneous adjustment to an equilibrium in which vapor is at or below the saturation specific
humidity over a temperature-dependent mixed surface. The mixed surface is parameterized by a liquid fraction
λ that varies linearly with temperature between the homogeneous ice nucleation temperature `Tʰ` and the
freezing temperature `Tᶠ`.

## Equilibrium surface model

Let `T` be temperature. Define the clamped temperature

```math
T' = \mathrm{clamp}(T, T^h, T^f) ,
```

where ``\mathrm{clamp}(x, x_{\min}, x_{\max})`` limits values of ``x`` within range ``[x_{\min}, x_{\max}]``,
and the liquid fraction

```math
\lambda(T) = \frac{T' - T^f}{T^h - T^f} , \qquad \lambda \in [0, 1] .
```

A value of ``λ = 1`` corresponds to a pure liquid surface (warm side); ``λ = 0`` corresponds to a pure
ice surface (cold side). This model yields a mixed-phase surface with effective latent heat and heat-capacity
difference that are linear blends of the pure-phase values, consistent with [Pressel2015](@citet).

In Breeze, the equilibrium surface is constructed internally via `PlanarMixedPhaseSurface(λ)` using
[`MixedPhaseEquilibrium`](@ref Breeze.Microphysics.MixedPhaseEquilibrium), and is accessed by

```@example mixed_phase
using Breeze
using Breeze.Microphysics: MixedPhaseEquilibrium

FT = Float64
thermo = ThermodynamicConstants(FT)

# Define equilibrium with default temperatures (Tᶠ = 273.15 K, Tʰ = 233.15 K)
eq = MixedPhaseEquilibrium(FT)
```

## Saturation specific humidity across the mixed-phase range

We can compute the saturation specific humidity across the entire range from homogeneous
ice nucleation up to freezing using the equilibrium model above:

```@example mixed_phase
using Breeze: equilibrium_saturation_specific_humidity

p = 101325.0
qᵗ = 0.012

Tᶠ = eq.freezing_temperature
Tʰ = eq.homogeneous_ice_nucleation_temperature
T = range(Tʰ - 10, Tᶠ + 10; length=81) # slightly beyond the mixed-phase range

qᵛ⁺ = [equilibrium_saturation_specific_humidity(Tʲ, p, qᵗ, thermo, eq) for Tʲ in T]
```

Optionally, we can visualize how `qᵛ⁺` varies with temperature:

```@example mixed_phase
using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Temperature (K)", ylabel="qᵛ⁺ (kg kg⁻¹)")
lines!(ax, T, qᵛ⁺)
fig
```

## Liquid/ice partitioning of condensate

Under mixed-phase saturation adjustment, excess total moisture is partitioned between liquid and ice
according to the temperature-dependent liquid fraction ``λ(T)`` implied by the equilibrium surface.
Given total moisture `qᵗ` and saturation specific humidity `qᵛ⁺(T)`, the condensate amount is

```math
q^{\mathrm{cond}} = \max(0, q^t - q^{v+}) , \qquad
q^l = \lambda(T) q^{\mathrm{cond}} , \quad
q^i = [1 - \lambda(T)] q^{\mathrm{cond}} .
```

```@example mixed_phase
# Compute partitioning across temperature range
T′ = clamp.(T, Tʰ, Tᶠ)
λ = @. (T′ - Tᶠ) / (Tʰ - Tᶠ)
qcond = @. max(0, qᵗ - qᵛ⁺)

qˡ = λ .* qcond
qⁱ = (1 .- λ) .* qcond

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Temperature (K)", ylabel="Specific humidity (kg kg⁻¹)")
lines!(ax, T, qˡ, label="liquid")
lines!(ax, T, qⁱ, label="ice")
axislegend(ax, position=:lc)
fig
```

## Notes

- Warm-phase saturation adjustment is recovered for `T ≥ Tᶠ` (``λ = 1``, all liquid) and ice-phase saturation for
  `T ≤ Tʰ` (``λ = 0``, all ice).
- The moist static energy in Breeze includes both liquid and ice latent heats in mixed-phase conditions:
  ```math
  e = cᵖᵐ T + g z - ℒˡᵣ qˡ - ℒⁱᵣ qⁱ .
  ```

For details about the saturation adjustment algorithm itself and moist static energy,
see the [Warm-phase saturation adjustement](@ref sec:warm-saturation-adjustment).
