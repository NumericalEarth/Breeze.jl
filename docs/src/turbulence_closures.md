# Turbulence closures

A turbulence closure supplies the subgrid fluxes of momentum and of every tracer — the transport
carried by motions the grid does not resolve. Breeze's closures are Oceananigans
`AbstractScalarDiffusivity`s, so they compose with the rest of the model through the same
diffusivity interface as Oceananigans' own. A closure may be either three-dimensional, as a
subgrid-scale (or subfilter-scale) model for large-eddy simulation; or vertical only, commonly
referred to as a planetary boundary-layer scheme.

!!! tip "Other available turbulence closures"

    In addition to the [`TKEBasedTurbulenceClosure`](@ref) described here, Breeze also inherits
    [Oceananigans' own turbulence closures](https://clima.github.io/OceananigansDocumentation/stable/physics/turbulence_closures).

## Prognostic-TKE eddy diffusivity

[`TKEBasedTurbulenceClosure`](@ref) is a vertical eddy-diffusivity closure with one prognostic
equation, for the subgrid turbulent kinetic energy ``e``, in the spirit of CATKE
([Wagner et al. 2025](@cite Wagner25catke)). It is an empirical closure: the eddy diffusivities
of momentum, scalars and turbulent kinetic energy are the products of a turbulent velocity
``\sqrt{e}`` and a mixing length for each,

```math
K^u = ℓ^u \sqrt{e}, \qquad K^c = ℓ^c \sqrt{e}, \qquad K^e = ℓ^e \sqrt{e},
```

where each mixing length is a *stability function* times one primary length ``ℓ``,

```math
ℓ^u = S^u ℓ, \qquad ℓ^c = S^c ℓ, \qquad ℓ^e = S^e ℓ,
```

and the dissipation of turbulent kinetic energy follows the same pattern with a dissipation length
``ℓ^D = ℓ / S^D``,

```math
ε = \frac{e^{3/2}}{ℓ^D} = S^D \frac{e^{3/2}}{ℓ}.
```

The turbulent kinetic energy obeys

```math
∂_t (ρ e) + ∇ ⋅ (ρ 𝐮 e) = ∂_z (ρ K^e ∂_z e) + ρ (P + B - ε), \qquad P = K^u S², \qquad B = -K^c N²,
```

with shear production ``P`` from the squared vertical shear ``S² = (∂_z u)² + (∂_z v)²``, the
buoyancy flux ``B`` from the squared buoyancy frequency ``N² = g \, ∂_z \ln θᵥ``, the dissipation
``ε``, and transport. The density ``ρ e`` is the tracer `ρe`, which the closure adds to the model's
tracers; it is advected and vertically diffused like every other scalar, and the closure applies
the local terms ``P + B - ε``.

See the [single-column boundary layer example](literated/single_column_tke_boundary_layer.md) for
the closure in stable, neutral and convective boundary layers.

### The mixing length

The primary mixing length ([`TKEMixingLength`](@ref)) is the smaller of the height above the
surface and the stratification length,

```math
ℓ = \min(z, \, Cᴺ \sqrt{e} / N),
```

the distance to the wall and the distance a parcel with kinetic energy ``e`` travels against a
stable stratification of buoyancy frequency ``N``. The stratification length is infinite in neutral
and unstable air, where ``ℓ = z``. The height above the surface carries no coefficient of its own
— the stability functions set the scale of every diffusivity — and ``Cᴺ = 0.76`` by default, after
[Deardorff (1980)](@cite Deardorff1980).

### Stability functions

In this first version the stability functions are constants ([`ConstantStabilityFunctions`](@ref)),
``S^u = Cᵘ``, ``S^c = Cᶜ``, ``S^e = Cᵉ``, ``S^D = Cᴰ``. Three consequences are worth stating,
because they are what the constants mean:

- the turbulent Prandtl number is ``Pr = K^u / K^c = Cᵘ / Cᶜ``, and the TKE Schmidt number
  ``K^u / K^e = Cᵘ / Cᵉ``;
- in a neutral constant-stress layer, where ``ℓ = z``, the closure is Prandtl's mixing-length
  model: production balances dissipation at ``e / u_\star² = 1 / \sqrt{Cᵘ Cᴰ}``, and the wind
  profile is logarithmic with von Kármán constant ``κ = (Cᵘ³ / Cᴰ)^{1/4}``;
- in a stably stratified layer far from the surface, where ``ℓ = Cᴺ \sqrt{e} / N``, turbulent
  kinetic energy grows below and decays above the gradient Richardson number
  ``Ri^\dagger = Cᵘ Cᴺ² / (Cᶜ Cᴺ² + Cᴰ)``.

The defaults, ``Cᵘ = 0.196``, ``Cᶜ = 0.265``, ``Cᵉ = 0.392``, ``Cᴰ = 0.295``, are the
Mellor–Yamada coefficients of [Nakanishi and Niino (2009)](@cite NakanishiNiino2009) with the von
Kármán constant absorbed; they give ``κ = 0.40``, ``e / u_\star² = 4.2``, ``Pr = 0.74`` and
``Ri^\dagger = 0.25``. They are placeholders for calibration. Richardson-number-dependent stability
functions, as in CATKE, a convective length scale driven by the surface buoyancy flux, a surface
flux of turbulent kinetic energy, and a non-local flux are natural extensions.

### Numerics

The diffusivities are computed at the cell interfaces where the fluxes live, from ``\sqrt{e}``
reconstructed from the cell centers and floored at `minimum_tke`. The numerics of the TKE equation
follow CATKE. The sinks — dissipation and the negative part of the buoyancy flux, as the rate
``-Lᵉ = Sᴰ \sqrt{e} / ℓ + |B⁻| / e``, and the damping of ``e`` that advection drives negative, at
the rate ``1/τ`` — enter the vertically implicit tridiagonal solve of every time-step stage together
with the vertical diffusion of ``e``, so that ``e`` stays positive for any time step. The sources —
shear production and the positive part of the buoyancy flux — enter the tendency of the same stage.
Under an explicit time discretization the sinks enter the tendency as well.
