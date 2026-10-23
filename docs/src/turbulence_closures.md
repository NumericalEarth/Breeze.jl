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

The primary mixing length ([`AbstractMixingLength`](@ref)) is built from the wall length ``ℂˢ z``
and the buoyancy penetration depth ``ℓᵇ = \sqrt{e} / N``, which is infinite where ``N² ≤ 0``.
Every formulation gives ``ℓ = ℂˢ z`` in a neutral surface layer and ``ℓ = ℓᵇ`` in uniform
stratification; they differ elsewhere:

- [`LocalMinimumMixingLength`](@ref): ``ℓ = \min(ℂˢ z, ℓᵇ)``, after Deardorff (1980). In an elevated
  layer with ``N² ≤ 0``, such as a saturated cloud layer, only the distance to the ground bounds ``ℓ``,
  so the diffusivity can jump by orders of magnitude.
- [`GradientLimitedMixingLength`](@ref) (default): ``ℓ(z) = \min(ℂˢ z, \min_{z′}[ℓᵇ(z′) + ℂⁱ |z - z′|])``,
  which also bounds ``ℓ`` by the distance to nearby stratification, growing at slope ``ℂⁱ``.
  The envelope before the wall cap is stored in `closure_fields.ℓˢ`.
- [`IntegralMixingLength`](@ref): the parcel length of
  [Bougeault and Lacarrère (1989)](@cite BougeaultLacarrere1989), ``ℓ = ℂˢ \min(ℓ↑, ℓ↓)``, where
  ``ℓ↑`` and ``ℓ↓`` are the distances a parcel with energy ``e / (2 ℂˢ²)`` travels up and down before
  its accumulated buoyancy deficit consumes that energy. It penetrates inversions more than the
  gradient-limited length, at the cost of a state-dependent walk per face.

The mixing length is stored in `closure_fields.ℓᵐ`. The standalone constructors default to
``ℂˢ = 1.316`` (`GradientLimitedMixingLength(ℂˢ=c)` also sets ``ℂⁱ = c``); the default closure uses the
calibrated ``ℂˢ ≈ 0.657`` and ``ℂⁱ ≈ 1.56``.

### Static stability

``N²`` enters the stratification length, the buoyancy flux, and ``Ri``. It is diagnosed once per
stage at cell interfaces and stored in `closure_fields.N²`, according to the closure's
`static_stability`. [`DryStaticStability`](@ref) uses ``g \, ∂_z \ln θᵨ``, with ``θᵨ`` the density
potential temperature. [`MoistStaticStability`](@ref), the default, uses the dry value where
subsaturated and, where saturated, the saturated buoyancy frequency of
[Durran and Klemp (1982)](@cite DurranKlemp1982),

```math
N²_s = g \left[ \frac{1 + ℒ rˢ / (Rᵈ T)}{1 + ϵ ℒ² rˢ / (cᵖᵈ Rᵈ T²)}
               \left( ∂_z \ln θ + \frac{ℒ}{cᵖᵈ T} ∂_z rˢ \right) - ∂_z rʷ \right],
```

where ``rˢ`` and ``rʷ`` are the saturation and nonprecipitating-water mixing ratios, ``ϵ = Rᵈ / Rᵛ``,
and ``ℒ`` is the latent heat, taken from the microphysics' phase equilibrium. ``N²_s`` nearly vanishes
on a saturated adiabat, so mixing is not shut off inside stratocumulus.

### Stability functions

The default is [`RationalStabilityFunction`](@ref),
``S(Ri) = C⁰ + (C^± - C⁰) x / (1 + x)`` with ``x = |Ri| / Ri^±``, which is continuous but kinked at
neutral. Its calibrated coefficients (rounded) are

| Function | Unstable limit | Neutral | Stable limit |
|:--|--:|--:|--:|
| momentum | 0.361 | 0.159 | 0.0110 |
| scalars | 0.0817 | 0.199 | 0.533 |
| TKE | 1.92 | 0.00731 | 0.0230 |
| dissipation | 0.00541 | 0.0292 | 0.802 |

with ``Ri⁻ ≈ 12.7`` and ``Ri⁺ ≈ 0.0501``. The wall coefficient follows from
``ℂˢ = 0.4 (ℂᴰ⁰ / ℂᵘ⁰³)^{1/4}``. These come from an unconverged calibration fit
and are not independently validated.

[`ConstantStabilityFunctions`](@ref) give ``Pr = ℂᵘ / ℂᶜ``, a log layer with
``κ = ℂˢ (ℂᵘ³ / ℂᴰ)^{1/4}`` and ``e / u_\star² = 1 / \sqrt{ℂᵘ ℂᴰ}``, and a critical Richardson
number ``Ri^\dagger = ℂᵘ / (ℂᶜ + ℂᴰ)``. Their defaults, ``ℂᵘ = 0.149``, ``ℂᶜ = 0.201``,
``ℂᵉ = 0.298``, ``ℂᴰ = 0.388``, are the coefficients of
[Nakanishi and Niino (2009)](@cite NakanishiNiino2009), giving ``κ = 0.40``, ``Pr = 0.74`` and
``Ri^\dagger = 0.25``.

### Surface TKE flux

The calibrated closure also assumes a surface flux of TKE density,
``ρ₀ (ℂᵂu★ u★³ + ℂᵂʷ wΔ³)`` with ``wΔ³ = Δz₁ \max(Jᵇ, 0)``, which the closure does not add by itself.
[`tke_surface_flux`](@ref) computes it for use in a boundary condition:

```jldoctest
using Breeze, Oceananigans

flux = tke_surface_flux(1.2, 0.3^3, 20 * 1e-3);
tke_bc = FieldBoundaryConditions(bottom=FluxBoundaryCondition(flux));
boundary_conditions = (; ρe=tke_bc);
TKEBasedTurbulenceClosure().mixing_length.ℂⁱ

# output
1.5609936948461662
```

### Numerics

The diffusivities are computed at the cell interfaces where the fluxes live, from the stored mixing
length and ``\sqrt{e}`` reconstructed from the cell centers and floored at `minimum_tke`.
The numerics of the TKE equation follow CATKE. The sinks — dissipation and the negative part of the buoyancy flux, as the rate
``-Lᵉ = Sᴰ \sqrt{e} / ℓ + |B⁻| / e``, and the damping of ``e`` that advection drives negative, at
the rate ``1/τ`` — enter the vertically implicit tridiagonal solve of every time-step stage together
with the vertical diffusion of ``e``, so that ``e`` stays positive for any time step. The sources —
shear production and the positive part of the buoyancy flux — enter the tendency of the same stage.
Under an explicit time discretization the sinks enter the tendency as well.
