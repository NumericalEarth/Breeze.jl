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
ℓ^u = \mathbb{S}^u ℓ, \qquad ℓ^c = \mathbb{S}^c ℓ, \qquad ℓ^e = \mathbb{S}^e ℓ,
```

and the dissipation of turbulent kinetic energy follows the same pattern with a dissipation length
``ℓ^D = ℓ / \mathbb{S}^D``,

```math
ε = \frac{e^{3/2}}{ℓ^D} = \mathbb{S}^D \frac{e^{3/2}}{ℓ}.
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

The primary mixing length ([`AbstractMixingLength`](@ref)) is built from the wall length ``ℂˢ z``
and the buoyancy penetration depth ``ℓᵇ = \sqrt{e} / N``, which is infinite where ``N² ≤ 0``.
Every formulation gives ``ℓ = ℂˢ z`` in a neutral surface layer and ``ℓ = ℓᵇ`` in uniform
stratification; they differ elsewhere:

- [`LocalMinimumMixingLength`](@ref): ``ℓ = \min(ℂˢ z, ℓᵇ)``, after Deardorff (1980). In an elevated
  layer with ``N² ≤ 0``, such as a saturated cloud layer, only the distance to the ground bounds ``ℓ``,
  so the diffusivity can jump by orders of magnitude.
- [`GradientLimitedMixingLength`](@ref) (default): ``ℓ(z) = \min(ℂˢ z, \min_{z′}[ℂˡᵇ ℓᵇ(z′) + ℂⁱ |z - z′|])``,
  which also bounds ``ℓ`` by the distance to nearby stratification, growing at slope ``ℂⁱ``. In stable
  air ``ℓ → ℂˡᵇ \sqrt{e} / N``; with ``ℂˡᵇ > 1`` (1 by default) the stability functions carry the
  stratification limit beyond ``α_N = 1``.
  The envelope before capping at the surface length is stored in `closure_fields.ℓˢ`, and at the cell
  centers, where the dissipation reads it, in `closure_fields.ℓˢᶜ`: the upward sweep of the face below
  with the downward sweep alone of the face above, each carried half a cell, which hold different bounds.
- [`IntegralMixingLength`](@ref): the parcel length of
  [Bougeault and Lacarrère (1989)](@cite BougeaultLacarrere1989), ``ℓ = ℂˢ \min(ℓ↑, ℓ↓)``, where
  ``ℓ↑`` and ``ℓ↓`` are the distances a parcel with energy ``e / (2 ℂˢ²)`` travels up and down before
  its accumulated buoyancy deficit consumes that energy. It penetrates inversions more than the
  gradient-limited length, at the cost of a state-dependent walk per face.

The mixing length is stored in `closure_fields.ℓᵐ`. The standalone constructors default to
``ℂˢ = 1.316`` (`GradientLimitedMixingLength(ℂˢ=c)` also sets ``ℂⁱ = c``); the default closure uses the
calibrated ``ℂˢ ≈ 0.788`` and ``ℂⁱ ≈ 1.78``.

### Static stability

``N²`` enters the stratification length, the buoyancy flux, and ``Ri``. It is diagnosed once per
stage at cell interfaces and stored in `closure_fields.N²`, according to the closure's
`static_stability`. [`DryStaticStability`](@ref) uses ``g \, ∂_z \ln θᵨ``, with ``θᵨ`` the density
potential temperature. [`MoistStaticStability`](@ref) uses the dry value where
subsaturated and, where saturated, the saturated buoyancy frequency of
[Durran and Klemp (1982)](@cite DurranKlemp1982),

```math
N²_s = g \left[ \frac{1 + ℒ rˢ / (Rᵈ T)}{1 + ϵ ℒ² rˢ / (cᵖᵈ Rᵈ T²)}
               \left( ∂_z \ln θ + \frac{ℒ}{cᵖᵈ T} ∂_z rˢ \right) - ∂_z rᵗ \right],
```

where ``rˢ`` and ``rᵗ`` are the saturation and total-water mixing ratios, ``ϵ = Rᵈ / Rᵛ``,
and ``ℒ`` is the latent heat, taken from the microphysics' phase equilibrium. Saturation is tested
using the local moist density and excludes precipitation, while the loading term includes all
condensate. ``N²_s`` nearly vanishes
on a saturated adiabat, so mixing is not shut off inside stratocumulus.

With a `subgrid_distribution` ([`DoubleGaussian`](@ref), the default) the static stability is
[`SaturationWeightedStaticStability`](@ref)`(`[`CondensateFluxFraction`](@ref)`())`,
``N² = (1 - aᴺ) N²_d + aᴺ N²_s`` with the saturated fraction ``aᴺ = w′qᶜ′ / w′s′`` of the distribution
([Cuijpers and Duynkerke 1993](@cite CuijpersDuynkerke1993)), the stratification that the buoyancy production of
the TKE carries ([`LatentStratification`](@ref)).

### Stability functions

The default is [`TurbulentRichardsonStabilityFunction`](@ref). Its argument is the turbulent
Richardson number, formed from the stratification number ``α_N = τ² N²`` and the shear number
``α_S = τ² (S² + S²ᵐⁱⁿ)`` of turbulence with the time scale ``τ = ℓ / \sqrt{e}``:

```math
Riᵗ = Riᵗ⁻ + (Riᵗ⁺ - Riᵗ⁻) \frac{1 + s}{2}, \qquad
Riᵗ^± = \frac{α_N}{ℂᴿⁱ^± α_S + ℂᵅ}, \qquad
s = \frac{x}{\sqrt{1 + x²}}, \qquad x = \frac{α_N}{δ \min(ℂᴿⁱ^± α_S + ℂᵅ)},
```

a blend, with ``δ = 1/4``, of a number for the unstable side and one for the stable side by a smooth
sign of ``α_N``. It is ``Ri / ℂᴿⁱ⁻`` (unstable) or ``Ri / ℂᴿⁱ⁺`` (stable) where the shear is strong,
with ``Ri = N² / (S² + S²ᵐⁱⁿ)``, exactly ``α_N / ℂᵅ`` where the shear vanishes, increasing in ``α_N``
and smooth through ``N² = 0``, where its slope is the mean of the slopes of the two sides. Each of the four functions takes its strongly unstable,
neutral and strongly stable values ``ℂ⁻, ℂ⁰, ℂ⁺`` at ``v = -1, 0, 1``, with
``v = Riᵗ / \sqrt{1 + Riᵗ²}``, through a quadratic in ``v`` of its logarithm:

```math
\ln 𝕊 = \ln ℂ⁰ + \tfrac{1}{2} \ln(ℂ⁺ / ℂ⁻) \, v + \left[\tfrac{1}{2} \ln(ℂ⁺ ℂ⁻) - \ln ℂ⁰\right] v².
```

The functions are positive, smooth and without branches for any positive coefficients. The squared
shear ``S²`` and its floor ``S²ᵐⁱⁿ = 10⁻¹⁰ \, \mathrm{s⁻²}`` come from the closure's `shear`,
[`GridMeanShear`](@ref).

The fifteen coefficients are what [`closure_parameters`](@ref) lists. For a calibration,
`Breeze.TurbulenceClosures.stability_coefficient_bounds` gives each an independent interval,
``[0.05, 20]`` by default, except ``ℂᵅ ≥ 4.37``: at any shear the logarithmic slope of a function in
``α_N`` is at most ``0.699 × 1.044 \ln(20 / 0.05) / ℂᵅ`` (1.044 is the largest slope of the blend), and
this bound keeps it below one, the width condition of the stability functions. The defaults are hand-tuned starting values for calibration
(``e / u★² = 3.75`` and ``κ = 0.4`` in the neutral surface layer, mixing decreasing with stability):

| Function | Strongly unstable | Neutral | Strongly stable |
|:--|--:|--:|--:|
| momentum | 0.361 | 0.159 | 0.135 |
| scalars | 0.45 | 0.199 | 0.09 |
| TKE | 0.361 | 0.159 | 0.135 |
| dissipation | 0.447 | 0.447 | 0.447 |

with ``ℂᴿⁱ⁻ = 16``, ``ℂᴿⁱ⁺ = 0.33`` and ``ℂᵅ = 4.4``. Under strong shear ``ℂᴿⁱ⁺ = 0.33`` puts the
stable functions halfway through their transition, ``v = 1/2``, at ``Ri = 0.20``, where turbulence
observations show marginal stability (``v = 0.1`` at ``Ri = 0.045``, ``v = 0.9`` at ``Ri = 0.69``);
``ℂᴿⁱ⁻ = 16`` puts the unstable halfway point at ``Ri = -9.2``. The mixing length's wall coefficient is ``ℂˢ = 0.4 (ℂᴰ⁰ / ℂᵘ⁰³)^{1/4} ≈ 1.30`` and its
interior slope ``ℂⁱ ≈ 1.56``.

[`RationalStabilityFunction`](@ref) is
``S(Ri) = C⁰ + (C^± - C⁰) x / (1 + x)`` with ``x = |Ri| / Ri^±``, which is continuous but kinked at
neutral. The floor ``S²ᵐⁱⁿ`` keeps ``Ri`` finite where the shear vanishes, but there the functions
pass from neutral to their limits over ``|N²| ∼ Ri^± S²ᵐⁱⁿ``, a step in ``N²``.
Its calibrated coefficients (rounded) are

| Function | Unstable limit | Neutral | Stable limit |
|:--|--:|--:|--:|
| momentum | 1.20 | 0.154 | 1.01 |
| scalars | 2.08 | 0.193 | 0.277 |
| TKE | 312 | 0.0735 | 16.4 |
| dissipation | 2.59 | 0.0554 | 0.0507 |

with ``Ri⁻ ≈ 6.56`` and ``Ri⁺ ≈ 15.1``, paired with ``ℂˢ ≈ 0.788`` (``= 0.4 (ℂᴰ⁰ / ℂᵘ⁰³)^{1/4}``)
and ``ℂⁱ ≈ 1.78``. These coefficients are calibrated to the
[Shen et al. (2022)](@cite Shen2022) LES library of low clouds using stochastic gradient descent.

[`ConstantStabilityFunctions`](@ref) give ``Pr = ℂᵘ / ℂᶜ``, a log layer with
``κ = ℂˢ (ℂᵘ³ / ℂᴰ)^{1/4}`` and ``e / u_\star² = 1 / \sqrt{ℂᵘ ℂᴰ}``, and a critical Richardson
number ``Ri^\dagger = ℂᵘ / (ℂᶜ + ℂᴰ)``. Their defaults, ``ℂᵘ = 0.149``, ``ℂᶜ = 0.201``,
``ℂᵉ = 0.298``, ``ℂᴰ = 0.388``, are the coefficients of
[Nakanishi and Niino (2009)](@cite NakanishiNiino2009), giving ``κ = 0.40``, ``Pr = 0.74`` and
``Ri^\dagger = 0.25``.

### Numerics

The diffusivities are computed at the cell interfaces where the fluxes live, from the stored mixing
length and ``\sqrt{e}`` reconstructed from the cell centers and floored at `minimum_tke`.
The numerics of the TKE equation follow CATKE. The sinks — dissipation and the negative part of the buoyancy flux, as the rate
``-Lᵉ = 𝕊ᴰ \sqrt{e} / ℓ + |B⁻| / \max(e, eᵐⁱⁿ)`` with ``eᵐⁱⁿ`` the `minimum_tke` (so that the buoyancy
sink is continuous at ``e = eᵐⁱⁿ`` and falls linearly with ``e`` below it), and the damping of ``e``
that advection drives negative, at the rate ``1/τ`` — enter the vertically implicit tridiagonal solve of every time-step stage together
with the vertical diffusion of ``e``, so that ``e`` stays positive for any time step. The sources —
shear production and the positive part of the buoyancy flux — enter the tendency of the same stage.
Shear production uses the current stage's velocities in both factors of ``Kᵘ S²``; its spatial
reconstruction retains CATKE's grid-spacing weights.
Under an explicit time discretization the sinks enter the tendency as well.
