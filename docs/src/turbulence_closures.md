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

The primary mixing length ([`TKEMixingLength`](@ref)) is the smaller of a wall length and the
stratification length,

```math
ℓ = \min(Cˢ z, \, \sqrt{e} / N),
```

``Cˢ`` times the distance to the wall, and the distance a parcel with kinetic energy ``e`` travels
against a stable stratification of buoyancy frequency ``N``. The stratification length is infinite
in neutral and unstable air, where ``ℓ = Cˢ z``. It carries no coefficient of its own — the
stability functions set the scale of every diffusivity — so ``Cˢ`` alone sets the ratio of the two
lengths. The default ``Cˢ = 1.316`` is the reciprocal of Deardorff's coefficient ``0.76`` of the
stratification length ([Deardorff (1980)](@cite Deardorff1980)), which the equivalent normalization
``ℓ = \min(z, 0.76 \sqrt{e} / N)`` carries on the stratification length instead.

### Static stability

The squared buoyancy frequency ``N²`` enters the closure three times — in the stratification length,
in the buoyancy flux ``B = -K^c N²``, and (for Richardson-number-dependent stability functions)
in ``Ri``. It is diagnosed once per time-step stage at the cell interfaces and stored with the
closure fields as `closure_fields.N²`, so that every term sees the same value. Which ``N²`` is
diagnosed is the closure's `static_stability`. [`DryStaticStability`](@ref) is the gradient
``∂_z b = g \, ∂_z \ln θᵨ`` of the buoyancy the dynamics uses, where ``θᵨ`` is the density
potential temperature, so that liquid and ice water load the air by their mass.

[`MoistStaticStability`](@ref), the default, accounts for saturation. Where the air is subsaturated
it is the dry gradient; where it is saturated — the nonprecipitating water exceeds the saturation
specific humidity at the interface — it is the buoyancy frequency of a saturated displacement of
[Durran and Klemp (1982)](@cite DurranKlemp1982),

```math
N²_s = g \left[ \frac{1 + ℒ rˢ / (Rᵈ T)}{1 + ϵ ℒ² rˢ / (cᵖᵈ Rᵈ T²)}
               \left( ∂_z \ln θ + \frac{ℒ}{cᵖᵈ T} ∂_z rˢ \right) - ∂_z rʷ \right],
```

in which a rising parcel condenses and its latent heating offsets part of the stratification of
the dry potential temperature ``θ``; ``rˢ`` and ``rʷ`` are the saturation and nonprecipitating-water
mixing ratios, ``ϵ = Rᵈ / Rᵛ`` and ``ℒ`` the latent heat. The phase equilibrium of the microphysics
supplies the saturation test and the latent heating, so over a mixed-phase surface the liquid and
ice branches interpolate with its liquid fraction. On a saturated adiabat — uniform liquid-water
potential temperature and total water — ``N²_s`` vanishes to within the approximations of the
expression, while the dry ``N²`` is strongly positive; inside stratocumulus this is the difference
between a stratification length that shuts the mixing off and one that lets it through.

### Stability functions

The stability functions are either constants ([`ConstantStabilityFunctions`](@ref), the default),
``S^u = Cᵘ``, ``S^c = Cᶜ``, ``S^e = Cᵉ``, ``S^D = Cᴰ``, or functions of the gradient Richardson
number ([`RiDependentStabilityFunctions`](@ref)). Three consequences of the constants are worth
stating, because they are what the constants mean:

- the turbulent Prandtl number is ``Pr = K^u / K^c = Cᵘ / Cᶜ``, and the TKE Schmidt number
  ``K^u / K^e = Cᵘ / Cᵉ``;
- in a neutral constant-stress layer, where ``ℓ = Cˢ z``, the closure is Prandtl's mixing-length
  model: production balances dissipation at ``e / u_\star² = 1 / \sqrt{Cᵘ Cᴰ}``, and the wind
  profile is logarithmic with von Kármán constant ``κ = Cˢ (Cᵘ³ / Cᴰ)^{1/4}``;
- in a stably stratified layer far from the surface, where ``ℓ = \sqrt{e} / N``, turbulent
  kinetic energy grows below and decays above the gradient Richardson number
  ``Ri^\dagger = Cᵘ / (Cᶜ + Cᴰ)``.

The defaults, ``Cᵘ = 0.149``, ``Cᶜ = 0.201``, ``Cᵉ = 0.298``, ``Cᴰ = 0.388``, are the
Mellor–Yamada coefficients of [Nakanishi and Niino (2009)](@cite NakanishiNiino2009) re-expressed
for this normalization of the mixing length; they give ``κ = 0.40``, ``e / u_\star² = 4.2``,
``Pr = 0.74`` and ``Ri^\dagger = 0.25``. They are placeholders for calibration.

[`RiDependentStabilityFunctions`](@ref) follow CATKE ([Wagner et al. 2025](@cite Wagner25catke)):
each of ``S^u, S^c, S^e, S^D`` is a constant ``C⁻`` in unstable stratification, its neutral value
``C⁰`` from ``Ri = 0`` to the onset ``Ri⁰`` of the stable transition, and a linear ramp over the
width ``Riᵟ`` to a stable asymptote ``C⁺``,

```math
S(Ri) = \begin{cases}
C⁻ & Ri < 0 \\
C⁰ + (C⁺ - C⁰) \, \mathrm{clamp}\left( \frac{Ri - Ri⁰}{Riᵟ}, 0, 1 \right) & Ri ≥ 0.
\end{cases}
```

``Ri = N² / S²`` is formed at each interface from the stored ``N²`` and the vertical shear, zero
where ``N² = 0`` and ``±∞`` where only the shear vanishes, which the ramp maps to an endpoint; the
dissipation function forms ``Ri`` at the cell center from ``N²`` and ``S²`` reconstructed there.
The twelve endpoints, the onset
and the width default to CATKE's values, calibrated against ocean large-eddy simulations and frozen
here, and CATKE's wall coefficient ``Cˢ = 1.131`` goes with them: `catke_parameters()` returns both
as keyword arguments,

```jldoctest
using Breeze

closure = TKEBasedTurbulenceClosure(; catke_parameters()...)
closure.stability_functions

# output
RiDependentStabilityFunctions{Float64}
├── Ri < 0 (Cᵘ⁻, Cᶜ⁻, Cᵉ⁻, Cᴰ⁻): 0.37, 0.572, 1.447, 0.923
├── Ri = 0 (Cᵘ⁰, Cᶜ⁰, Cᵉ⁰, Cᴰ⁰): 0.361, 0.369, 7.863, 1.604
├── Ri → ∞ (Cᵘ⁺, Cᶜ⁺, Cᵉ⁺, Cᴰ⁺): 0.242, 0.098, 0.548, 0.579
└── stable transition: Ri⁰ = 0.254, Riᵟ = 1.02
```

Two things CATKE's calibration relied on are not part of this closure — the convective and
entrainment length scales driven by the surface buoyancy flux, which dominate CATKE's mixing and
lengthen its dissipation length in convecting layers, and the surface flux of turbulent kinetic
energy — so weaker mixing and stronger dissipation in convective boundary layers than in CATKE are
to be expected. In the neutral surface layer the two coefficient sets imply

| | ``κ`` | ``e / u_\star²`` | ``Pr`` | ``K^e / K^u`` | ``Ri^\dagger`` |
|:--|:--|:--|:--|:--|:--|
| `ConstantStabilityFunctions` (Nakanishi–Niino) | 0.40 | 4.2 | 0.74 | 2.0 | 0.25 |
| `RiDependentStabilityFunctions` (CATKE, ``Ri = 0``) | 0.47 | 1.3 | 0.98 | 21.8 | 0.18 |

with ``κ = Cˢ (C^{u}{}^3 / C^D)^{1/4}``, ``e / u_\star² = 1 / \sqrt{C^u C^D}``, ``Pr = C^u / C^c``
and ``Ri^\dagger = C^u / (C^c + C^D)`` evaluated at the neutral values. The atmospheric surface
layer constrains the first three well, which is why the constants remain the default; in stable
stratification CATKE's Prandtl number rises to ``C^{u+} / C^{c+} = 2.5``. A convective length
scale, a surface flux of turbulent kinetic energy and a non-local flux are natural extensions.

### Numerics

The diffusivities are computed at the cell interfaces where the fluxes live, from ``\sqrt{e}``
reconstructed from the cell centers and floored at `minimum_tke`. The numerics of the TKE equation
follow CATKE. The sinks — dissipation and the negative part of the buoyancy flux, as the rate
``-Lᵉ = Sᴰ \sqrt{e} / ℓ + |B⁻| / e``, and the damping of ``e`` that advection drives negative, at
the rate ``1/τ`` — enter the vertically implicit tridiagonal solve of every time-step stage together
with the vertical diffusion of ``e``, so that ``e`` stays positive for any time step. The sources —
shear production and the positive part of the buoyancy flux — enter the tendency of the same stage.
Under an explicit time discretization the sinks enter the tendency as well.
