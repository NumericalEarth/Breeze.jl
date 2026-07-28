# Turbulence closures

A turbulence closure supplies the subgrid fluxes of momentum and of every tracer — the transport
carried by motions the grid does not resolve. Breeze's closures are Oceananigans
`AbstractScalarDiffusivity`s, so they compose with the rest of the model through the same
diffusivity interface as Oceananigans' own. A closure may be either three-dimensional, as a
subgrid-scale (or subfilter-scale) model for large-eddy simulation; or vertical only, commonly
referred to as a boundary-layer scheme.

In addition to the [`TKEBasedTurbulenceClosure`](@ref) described here, Breeze also inherits
[Oceananigans' own turbulence closures](https://clima.github.io/OceananigansDocumentation/stable/physics/turbulence_closures).

## Prognostic-TKE eddy diffusivity

[`TKEBasedTurbulenceClosure`](@ref) carries one prognostic equation, for the subgrid turbulent
kinetic energy ``e``, and closes the vertical eddy viscosity and diffusivity on it:

```math
ν = Cᴷ ℓ \sqrt{e}, \qquad K = ν / \mathrm{Pr}, \qquad ε = Cᵋ e^{3/2} / ℓ,
```

```math
∂e/∂t = P + B - ε + \text{transport}, \qquad P = ν S², \qquad B = -K N².
```

The source terms on the right-hand side include: shear production ``P``, buoyancy ``B``,
dissipation ``ε``, and transport, which is the ordinary scalar machinery acting on a `:ρtke` tracer.
The turbulent Prandtl number ``Pr`` grows with the gradient Richardson number, so that heat mixes
less readily than momentum as the column stabilizes.

See the [single-column boundary layer example](literated/single_column_tke_boundary_layer.md) for
an example of closure performance with stable, neutral and convective boundary layers.

### The mixing length

The length scale ``ℓ`` is supplied by a dispatched component, by default
[`MesoscaleLengthScale`](@ref). Structurally this is the three-branch harmonic blend of MYNN
([Nakanishi and Niino 2009](@cite NakanishiNiino2009), their Eqs. 52–55), which blends the branches
so that the smallest wins smoothly:

```math
\frac{1}{ℓ} = \frac{1}{ℓᵍ} + \frac{1}{ℓᵗ} + \frac{1}{ℓᵇ},
\qquad ℓᵍ = κ (z + ℓʳ), \qquad ℓᵗ = Cᵗ \frac{∫ q z \, dz}{∫ q \, dz}, \qquad ℓᵇ = Cᵇ q / N,
```

with ``q = \sqrt{2e}``. The branches are, in order, the distance to the surface offset by a
roughness length, the depth over which the column is turbulent, and the distance a parcel would
travel against stable stratification. ``ℓᵗ`` is the only non-local contribution.

### Coefficients

Eliminating ``ℓ`` between the viscosity and the dissipation gives the ``k``–``ε`` eddy-viscosity
relation, whose coefficient is the product of the other two:

```math
ν = Cᴷ ℓ \sqrt{e} = (Cᴷ Cᵋ) \frac{e²}{ε} ≡ Cμ \frac{e²}{ε}, \qquad Cμ ≡ Cᴷ Cᵋ.
```

So the closure stores ``Cᴷ`` and ``Cμ`` rather than ``Cᴷ`` and ``Cᵋ``, and the dissipation
coefficient ``Cᵋ = Cμ/Cᴷ``, the surface turbulence level and the stress coefficient all derive from
the stored pair.

Using ``Cμ`` has the advantage of being defined without reference to a master length scale
(``Cμ = ν ε / e²``) and therefore may be compared with families that carry no ``ℓ`` at all:
0.058 here (MYNN) against 0.090 for standard ``k``–``ε``, 0.094 (MY82), 0.148 (MYJ), and 0.200
(SHOC, which is uniquely written with a timescale rather than a length). ``Cμ`` alone fixes the
turbulence level in the neutral surface layer: a constant-stress layer at local equilibrium
``P = ε`` settles to ``e/u_\star² = (Cμ)^{-1/2}``, regardless of ``Cᴷ``. Higher up, ``e`` is set by
the full budget rather than by any coefficient.

The surface layer value is also imposed as a floor on ``e`` in the first cell. Since it is where the
local budget already balances, the floor is inert in a spun-up constant-stress layer; what
it does is get a column started, where ``e`` would otherwise begin too small for ``P = ν S²`` to
bootstrap any turbulence, and hold the surface consistent with the applied stress in a column that
has run down.

``Cᴷ`` alone carries consistency with the neutral logarithmic wind profile: in a constant-stress
layer with ``ℓ = κ z`` the closure collapses onto Prandtl's mixing-length model, but with an
effective length ``Cˢ κ z`` rather than ``κ z``, where

```math
Cˢ ≡ Cᴷ / (Cμ)^{1/4}
```

is the stress coefficient. The model's own velocity gradient is then ``u_\star / (Cˢ κ z)``, so
regressing ``U/u_\star`` on ``\ln(z/ℓʳ)`` over the constant-stress layer  returns ``Cˢ κ`` rather than
``κ``. The log law is recovered exactly on ``Cˢ = 1`` or, equivalently, on the locus ``Cμ = Cᴷ⁴``.
The defaults are MYNN's ([Nakanishi and Niino 2009](@cite NakanishiNiino2009)), ``Cᴷ = 0.4903``
and ``Cμ = 0.0578``, which satisfy it exactly.

!!! note "Note about Subgrid Coefficients"

    Subgrid coefficients from large-eddy models (e.g., ``Cˢ``) do not apply here, because there ``ℓ``
    is the filter width rather than an equilibrium mixing length.
