# Compressible equations and vertically implicit splitting

## Governing equations

Breeze solves the fully compressible Euler equations in conservation form
on a rotating sphere. The prognostic variables are density ρ, momentum
(ρu, ρv, ρw), and potential temperature density ρθ.

### Continuity

```
∂ρ/∂t + ∇·(ρv) = 0
```

### Momentum

```
∂(ρu)/∂t + ∇·(ρu ⊗ v) + metric terms = -∂p/∂x + f ρv
∂(ρv)/∂t + ∇·(ρv ⊗ v) + metric terms = -∂p/∂y - f ρu
∂(ρw)/∂t + ∇·(ρw ⊗ v)                = -∂p/∂z - ρg
```

The curvature metric terms arise from the Christoffel symbols on the
latitude-longitude grid and are handled by `U_dot_∇u_metric` etc.

### Thermodynamic equation

```
∂(ρθ)/∂t + ∇·(ρθ v) = 0
```

(adiabatic, no sources)

### Equation of state

Pressure is diagnosed from ρ and θ:

```
p = p₀ (ρ Rᵈ θ / p₀)^γ
```

where γ = cₚ/cᵥ. Equivalently, via the Exner function Π = (p/p₀)^κ:

```
p = ρ Rᵈ T = ρ Rᵈ θ Π
Π = (ρ Rᵈ θ / p₀)^(κ/(1-κ))
```

## Acoustic modes

The fast acoustic modes arise from the coupling between ρw and p (or equivalently ρθ)
in the vertical momentum and thermodynamic equations. Linearizing around a
hydrostatic state:

```
∂(ρw)'/∂t = -∂p'/∂z - ρ'g ≈ -(ℂ²/θ) ∂(ρθ)'/∂z - ρ'g
∂(ρθ)'/∂t = -∂(θ ρw')/∂z  (vertical advective flux only)
```

where ℂ² = γ Rᵐ T is the squared acoustic speed. These two equations
form a wave equation with phase speed ℂ ≈ 340 m/s. With Δz = 1 km,
the acoustic CFL gives Δt < Δz/ℂ ≈ 3s.

The key linearization is:

```
∂p/∂(ρθ) = ℂ²/θ
```

which comes from p = p₀(ρRᵈθ/p₀)^γ, so ∂p/∂(ρθ) = γp/(ρθ) = γRᵈT/θ = ℂ²/θ.

## IMEX splitting

The vertically implicit time stepping (VITS) treats the vertical acoustic
coupling implicitly while keeping everything else explicit.

### What is subtracted from the explicit tendencies

For each RK stage, the explicit tendencies are computed as usual, then the
vertical fast terms are **subtracted** (they will be handled implicitly):

**ρw tendency correction** — subtract vertical PGF + buoyancy:

```
Gρw_explicit -= (ℂ²/θ)ᶠ ∂(ρθ)/∂z + ρg
```

This removes the vertically propagating acoustic wave source. In hydrostatic
balance, `(ℂ²/θ) ∂(ρθ)/∂z + ρg = 0` exactly, so the correction is zero for
the balanced state. This is the key property that makes the method stable
without a reference state.

**ρθ tendency correction** — subtract vertical ρθ flux:

```
Gρθ_explicit -= ∂(θ w ρθ)/∂z
```

This removes the vertical advection of ρθ by w (which couples to the acoustic mode).

### After the explicit substep

After the explicit RK substep advances the state using the corrected
(slow-only) tendencies, the implicit solve restores the vertical acoustic
coupling:

**Step 1: Helmholtz solve for δ(ρθ)**

Combining the linearized ρw and ρθ equations and eliminating ρw gives a
Helmholtz equation for the perturbation δ(ρθ):

```
[I - (αΔt)² ∂z(ℂ² ∂z)] δ(ρθ) = -αΔt ∂z(θ ρw*)
```

where:
- αΔt is the RK stage coefficient times the time step
- ρw* is the explicitly updated momentum (before the implicit correction)
- ℂ² = γ Rᵐ T is evaluated at cell centers
- The operator `∂z(ℂ² ∂z)` is a second-order vertical diffusion operator (positive definite)

This is a tridiagonal system in each (i,j) column, solved with a batched
Thomas algorithm. The operator `I - (αΔt)² ∂z(ℂ² ∂z)` has eigenvalues
≥ 1 (since `-∂z(ℂ² ∂z)` is positive semi-definite), so the system is
**unconditionally stable** for any Δt.

**Step 2: Update ρθ**

```
(ρθ)⁺ = (ρθ)* + δ(ρθ)
```

**Step 3: Back-solve for ρw**

Using the change in ρθ, update ρw:

```
(ρw)⁺ = (ρw)* - αΔt (ℂ²/θ)ᶠ ∂(δρθ)/∂z
```

This uses only δ(ρθ), not the full (ρθ)⁺, which preserves hydrostatic
balance: if δ(ρθ) = 0, then (ρw)⁺ = (ρw)*.

## Tridiagonal structure

For a column with Nz cells, the Helmholtz system has the form:

```
| 1+Q₁ᵗ   -Q₁ᵗ                          | | δρθ₁ |   | r₁ |
| -Q₂ᵇ   1+Q₂ᵇ+Q₂ᵗ  -Q₂ᵗ               | | δρθ₂ | = | r₂ |
|         -Q₃ᵇ   1+Q₃ᵇ+Q₃ᵗ  -Q₃ᵗ        | | δρθ₃ |   | r₃ |
|                  ⋱       ⋱      ⋱       | |  ⋮   |   |  ⋮ |
|                       -Qₙᵇ   1+Qₙᵇ     | | δρθₙ |   | rₙ |
```

where:
- `Qₖᵇ = (αΔt)² ℂ²(k-½) / (Δzₖ₋½ Δzₖ)` — coupling to cell below
- `Qₖᵗ = (αΔt)² ℂ²(k+½) / (Δzₖ₊½ Δzₖ)` — coupling to cell above
- `rₖ = -αΔt/Δzₖ [θ(k+½) ρw*(k+½) - θ(k-½) ρw*(k-½)]`
- Boundary conditions: Q₁ᵇ = 0 (solid bottom), Qₙᵗ = 0 (solid top)

The matrix is symmetric positive definite with diagonal dominance ≥ 1.

## What MPAS does

For comparison, MPAS (Skamarock et al. 2012) uses a similar split-explicit
approach but with acoustic substepping:

1. Compute slow tendencies (advection, Coriolis, etc.) once per outer step
2. Subcycle acoustic modes with small Δτ = Δt/N_substeps
3. Each acoustic substep updates (u, w, θ', π') using the linearized
   acoustic equations
4. Vertical acoustics are treated **implicitly** within each substep

The MPAS vertical implicit solve is essentially the same Helmholtz problem,
but applied within the acoustic substep loop rather than as a correction
to the full RK stage.

## Time step limits with VITS

With vertical acoustics implicit, the remaining CFL constraints are:

1. **Horizontal acoustic**: Δt < Δx/ℂ. At 2° and 85° latitude,
   Δx ≈ 19 km → Δt < 56s. With polar filter, Δt ~ 200s.

2. **Advective**: Δt < Δx/U. With U ≈ 50 m/s and Δx ≈ 19 km → Δt < 380s.
   Usually not the bottleneck.

3. **Gravity wave**: Δt < Δx/N·H/f. Much less restrictive than acoustic.
