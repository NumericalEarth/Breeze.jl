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

Every formulation of the primary mixing length ([`AbstractMixingLength`](@ref)) is built from two
lengths: the wall length ``Cˢ z``, ``Cˢ`` times the height above the surface, and the *buoyancy
penetration depth*

```math
ℓᵇ = \\frac{\\sqrt{e}}{N},
```

how far an eddy with kinetic energy ``e`` penetrates air of buoyancy frequency ``N`` before its
kinetic energy is spent against buoyancy; ``ℓᵇ`` is infinite where ``N² ≤ 0``. In a neutral surface
layer every formulation gives ``ℓ = Cˢ z`` and in uniformly stratified air every formulation gives
``ℓ = ℓᵇ``; they differ in what else bounds ``ℓ``:

- [`LocalMinimumMixingLength`](@ref) is the smaller of the two at the eddy's own level,
  ``ℓ = \\min(Cˢ z, ℓᵇ)``, the form of Deardorff (1980) and of Nakanishi and Niino's closure without
  its turbulent-layer depth scale. It knows two obstacles, the ground and the stratification at the
  level itself, so in an elevated layer whose static stability is not positive — a saturated cloud
  layer under [`MoistStaticStability`](@ref), a neutral layer — nothing bounds ``ℓ`` but the distance to
  the ground, kilometers. The diffusivity ``Cᵘ ℓ \\sqrt{e}`` then jumps by two to five orders of
  magnitude whenever such a layer forms, mixes it away within a time step and collapses, or, when the
  layer persists, mixes the whole column.
- [`GradientLimitedMixingLength`](@ref), the default, separates the wall coefficient ``Cˢ``
  from the interior envelope slope ``Cⁱ``:
  ``ℓ(z) = \min(Cˢ z, \min_{z′ > 0}[ℓᵇ(z′) + Cⁱ |z-z′|])``.
  Two sweeps compute the stratification envelope, then the wall cap is applied separately.
  Its uncapped field `closure_fields.ℓˢ` is retained for center reconstruction.
  Equal slopes recover the original gradient-limited formulation.
- [`IntegralMixingLength`](@ref) is the parcel formulation of
  [Bougeault and Lacarrère (1989)](@cite BougeaultLacarrere1989): a parcel released at ``z`` with
  kinetic energy ``e★ = e / (2 Cˢ²)`` rises and sinks until the buoyancy deficit it accumulates,
  ``∫ [b(z) - b(z ± s)] \\, ds``, has consumed its energy, and ``ℓ = Cˢ \\min(ℓ↑, ℓ↓)``. The energy
  ``e / (2 Cˢ²)`` makes it ``Cˢ z`` against the ground and ``ℓᵇ`` in uniform stratification like the
  others. It carries the parcel's own energy into the air it penetrates and credits the energy gained
  crossing unstable air, so it penetrates inversions and mixes convective layers more than the
  gradient-limited length, whose penetration depth is that of the obstacle; its cost is a walk per
  face whose length depends on the state.

The standalone mixing-length constructors retain the legacy coefficient `Cˢ = 1.316`;
`GradientLimitedMixingLength(Cˢ=c)` sets `Cⁱ=c` as well. The default closure instead supplies
the calibrated pair `Cˢ = 0.6565692899592015`, `Cⁱ = 1.5609936948461662`.
The mixing length is stored in `closure_fields.ℓ`.

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

The default stability functions are [`RationalStabilityFunction`](@ref), described below.
[`ConstantStabilityFunctions`](@ref) and [`PiecewiseStabilityFunction`](@ref) remain available
explicitly; `RiDependentStabilityFunctions` is a compatibility alias for the latter.
For constant stability functions, three useful relations are:

- the turbulent Prandtl number is ``Pr = K^u / K^c = Cᵘ / Cᶜ``, and the TKE Schmidt number
  ``K^u / K^e = Cᵘ / Cᵉ``;
- in a neutral constant-stress layer, where ``ℓ = Cˢ z``, the closure is Prandtl's mixing-length
  model: production balances dissipation at ``e / u_\star² = 1 / \sqrt{Cᵘ Cᴰ}``, and the wind
  profile is logarithmic with von Kármán constant ``κ = Cˢ (Cᵘ³ / Cᴰ)^{1/4}``;
- in a stably stratified layer far from the surface, where ``ℓ = \sqrt{e} / N``, turbulent
  kinetic energy grows below and decays above the gradient Richardson number
  ``Ri^\dagger = Cᵘ / (Cᶜ + Cᴰ)``.

The legacy constant coefficients, ``Cᵘ = 0.149``, ``Cᶜ = 0.201``, ``Cᵉ = 0.298``, ``Cᴰ = 0.388``, are the
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
PiecewiseStabilityFunction{Float64}
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
and ``Ri^\dagger = C^u / (C^c + C^D)`` evaluated at the neutral values. The tabulated critical-Richardson formula is for constant coefficients, not a stability threshold
for the calibrated rational family.

### Calibrated rational defaults and provenance

[`RationalStabilityFunction`](@ref) uses
``S(Ri) = C⁰ + (C^± - C⁰) x^p / (1+x^p)``, with ``x = |Ri|/Ri^±``.
Both exponents are one. The function is continuous at neutral but generally has a kink there;
this release does not claim validated automatic differentiation through the full model.

The default coefficients are the evaluated member 324 of generation 11 of
`P60-west-v3-T1-N800-s3-PrON` (800 members, seed 3, evaluated pseudotime
0.1783616628408866). Its data RMS is 1.7068703929788749 and neutral
``Pr₀ = 0.8001428358397753``. It was the lowest data loss among 374 exported
generation-mean/data-best vectors across 19 arms available to the selection, with or without
the neutral-Prandtl filter. This is a finite-progress training fit, **not** a converged
optimum, a held-out validation result, or an exhaustive search of unexported checkpoints.
The provisional-60 suite excludes the unresolved native-case-61 parity case.
The stable transition scale and friction-velocity flux coefficient are near their prior bounds.

| Function | Unstable limit | Neutral | Stable limit |
|:--|--:|--:|--:|
| momentum | 0.36113018152281917 | 0.15900569445804447 | 0.010956324667755535 |
| scalars | 0.08170851244564359 | 0.19872163735761372 | 0.532550451554028 |
| TKE | 1.9176239082347815 | 0.00731292096489522 | 0.02299192764300173 |
| dissipation | 0.005407068485856925 | 0.029182354585815418 | 0.8022301180336915 |

The scales are `Ri⁻ = 12.669701017862883` and `Ri⁺ = 0.0501147186437052`.
The wall coefficient is derived, not independently fitted:
``Cˢ = 0.4 (Cᴰ⁰ / Cᵘ⁰³)^{1/4} = 0.6565692899592015``.
The independent interior slope is `Cⁱ = 1.5609936948461662`.

Reproducing the calibrated setup **also requires** the surface TKE-density flux
``ρ₀(Cᵂu★ u★³ + Cᵂʷ wΔ³)``, with
`Cᵂu★ = 0.01002182909933591` and `Cᵂʷ = 0.013669219822302736`.
[`tke_surface_flux`](@ref) implements this scalar calculation; the closure does not
silently add a boundary condition. Use each case's lowest-center reference density, friction
velocity cubed, and convective velocity cubed
``wΔ³ = Δz₁ \max(Jᵇ, 0)``, where ``Δz₁`` is the bottom-cell thickness.
For a constant-input example, construct the bottom condition explicitly:

```jldoctest
using Breeze, Oceananigans

flux = tke_surface_flux(1.2, 0.3^3, 20 * 1e-3);
tke_bc = FieldBoundaryConditions(bottom=FluxBoundaryCondition(flux));
boundary_conditions = (; ρe=tke_bc);
TKEBasedTurbulenceClosure().mixing_length.Cⁱ

# output
1.5609936948461662
```

For time-varying cases, use a discrete boundary callback that evaluates
`tke_surface_flux` using the prescribed series at `clock.time`.
The checkpoint SHA-256 is
`bcb1b9c03ce5068f5deb81efc5f0617e8be74d777fd6b47f24d76e9b1f7f6140`;
the calibration suite fingerprint is
`b371f23d633f27207c0ada2404c7d46cb999d49c490b69e338b2960b732d7cb2`.
No calibration drivers or checkpoint files are included in this PR.

### Numerics

The diffusivities are computed at the cell interfaces where the fluxes live, from the stored mixing
length and ``\sqrt{e}`` reconstructed from the cell centers and floored at `minimum_tke`; the
dissipation reconstructs the uncapped stratification envelope at the center using `Cⁱ`, then
bounds it by the center's local penetration depth and wall length `Cˢ z`.
The numerics of the TKE equation
follow CATKE. The sinks — dissipation and the negative part of the buoyancy flux, as the rate
``-Lᵉ = Sᴰ \sqrt{e} / ℓ + |B⁻| / e``, and the damping of ``e`` that advection drives negative, at
the rate ``1/τ`` — enter the vertically implicit tridiagonal solve of every time-step stage together
with the vertical diffusion of ``e``, so that ``e`` stays positive for any time step. The sources —
shear production and the positive part of the buoyancy flux — enter the tendency of the same stage.
Under an explicit time discretization the sinks enter the tendency as well.
