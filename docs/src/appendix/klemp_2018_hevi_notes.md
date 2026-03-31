# Notes from Klemp, Skamarock & Ha (2018)

"Damping Acoustic Modes in Compressible HEVI and Split-Explicit Time Integration Schemes"
MWR 146, 1911-1923.

## Key insight: divergence damping as a final adjustment

The paper proposes applying divergence damping as a **final adjustment** to the
horizontal velocity after completing the vertically implicit step. This is critical
because it ensures the divergence in the damping term has the **exact same numerical
form** as the divergence in the θ equation (or equivalently, the pressure equation).

## The HEVI algorithm from Klemp (eqs 21-25)

The linearized equations with forward-backward horizontal and implicit vertical:

```
U*^{t+Δt} = U^t - c² Δt Q^t_x                      (21) explicit horizontal PGF
W^{t+Δt}  = W^t - c² Δt [Q̄^t_z - Q̄^t/(2H) + g/c² r^t]  (22) implicit vertical PGF+buoyancy
r^{t+Δt}  = r^t - Δt [U*^{t+Δt}_x + W̄^t_z - W̄^t/(2H)]   (23) density: horiz uses new U*, vert uses implicit W̄
Q^{t+Δt}  = Q^t - Δt [U*^{t+Δt}_x + W̄^t_z - ηW̄^t]       (24) θ equation: same structure as (23)
U^{t+Δt}  = U*^{t+Δt} + γ_h Δt D^{t+Δt}_x               (25) divergence damping adjustment
```

where `Q̄^t = (1+σ)/2 * Q^{t+Δt} + (1-σ)/2 * Q^t` is the off-centered implicit average,
`D = U_x + W_z - ηW` is the full 3D divergence, and σ is the off-centering parameter.

## Critical observations for Breeze VITS

### 1. The density equation (23) uses the **new U*** horizontally and **implicit W̄** vertically

This is exactly what Gardner et al. (2018) splitting C requires. In Breeze:
- Horizontal divergence uses the explicitly updated momentum → already in the explicit step
- Vertical divergence uses the implicitly corrected ρw → must be applied after the Helmholtz solve

### 2. Off-centering (σ > 0) provides vertical acoustic damping

The implicit vertical solve uses a Crank-Nicolson-like average:
`Q̄ = (1+σ)/2 Q^{t+Δt} + (1-σ)/2 Q^t`

With σ = 0 (centered), there's no numerical diffusion — acoustic modes bounce forever.
With σ > 0, vertical acoustic modes are damped. The damping rate is ~ σ Δt l_z² / (1 + (1+σ)²l_z²).

**Our VITS uses σ = 0 (no off-centering).** This is why vertical acoustic noise persists.
Adding σ ≈ 0.1-0.2 would damp vertical acoustic noise without significantly affecting
gravity waves.

### 3. Divergence damping handles horizontal acoustic modes

The horizontal divergence damping (eq 25) provides:
- Stability for `α_h < (1 - λ_x²)/2`
- Acoustic damping rate independent of vertical wavenumber for small l_z
- Negligible effect on gravity wave frequencies

The dimensionless coefficient `α_h = γ_h Δt / Δx²` should be O(0.1-0.3).

### 4. Combined off-centering + divergence damping covers all modes

From eq (31): `|A|² = 1 - 4(α_h S² + σ l_z²) / [1 + (1+σ)² l_z²]`

- At small l_z: horizontal divergence damping dominates
- At large l_z: vertical off-centering dominates
- Together: effective damping at all wavenumbers

### 5. The divergence should be D = ∇·(ρ̄ v), not D = ∇·v

The paper emphasizes that for compressible (non-Boussinesq) atmospheres,
the correct divergence for acoustic filtering is `D = ∇·(ρ̄ v)` (mass-weighted),
not `D = ∇·v`. Using plain velocity divergence creates artificial effects on
gravity wave frequencies.

## Action items for Breeze VITS

1. **Add implicit vertical off-centering** (σ ≈ 0.1) to damp vertical acoustic modes
2. **Add horizontal divergence damping** as a final adjustment after the implicit step
3. **Add implicit density update** from the corrected ρw vertical divergence
4. The divergence in the damping term should use the same numerical form as
   the divergence in the ρθ equation: `D = -∂(ρθ)/∂t / (ρθ) ≈ -(Q^{t+Δt} - Q^t) / (Δt Q)`

## Relation to our current problems

Our VITS is unstable because:
- No off-centering (σ = 0): vertical acoustic modes are not damped
- No divergence damping: horizontal acoustic modes are not damped
- No density update: the acoustic coupling w ↔ p ↔ ρ is incomplete

With all three additions, VITS should be unconditionally stable for the vertical CFL
and conditionally stable for the horizontal CFL (Δt < Δx/c), matching MPAS behavior.
