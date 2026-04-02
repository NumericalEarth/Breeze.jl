# Schur complement for δρw: acoustic + gravity tridiagonal

## Coupled implicit system

At each implicit stage, given predictor (ρw*, ρθ*, ρ*), solve:

```
ρw_i = ρw* - τ (ℂ²/θ) ∂ρθ_i/∂z - τ ρ_i g          (1: momentum)
ρθ_i = ρθ* - τ θ₀ ∂(ρw_i)/∂z                         (2: compressibility)
ρ_i  = ρ*  - τ ∂(ρw_i)/∂z                              (3: continuity)
```

where τ = γ h, frozen coefficients ℂ²/θ and θ₀ from predictor.

## Perturbation equations

Define δρw = ρw_i - ρw*, δρθ = ρθ_i - ρθ*, δρ = ρ_i - ρ*.

From (2): δρθ = -τ θ₀ ∂ρw*/∂z - τ θ₀ ∂(δρw)/∂z
From (3): δρ  = -τ ∂ρw*/∂z   - τ ∂(δρw)/∂z

Substituting into (1):

δρw + τ²ℂ² ∂²(δρw)/∂z² + τ²g ∂(δρw)/∂z
= -τ[(ℂ²/θ)∂ρθ*/∂z + ρ*g] + τ²ℂ² ∂²ρw*/∂z² + τ²g ∂ρw*/∂z

## Tridiagonal for δρw

```
[I - τ²ℂ² ∂²/∂z² - τ²g ∂/∂z] δρw = RHS
```

**Operator**: acoustic (second derivative, symmetric) + gravity (first derivative, skew)

**RHS** at face k:
```
RHS[k] = -τ [(ℂ²/θ)_k ∂ρθ*/∂z|_k + ρ*_k g]       (mean PGF+gravity)
         + τ²ℂ²_k ∂²ρw*/∂z²|_k                      (acoustic on ρw*)
         + τ²g ∂ρw*/∂z|_k                             (gravity on ρw*)
```

For balanced state (ρw*=0): RHS = -τ · O(Δz²) (truncation error).
The operator's non-identity terms act on the z-structure of the error,
providing partial cancellation through the acoustic+gravity feedback.

## Discrete stencil

At face k (between centers k-1 and k):

**Acoustic** (already in current Helmholtz):
```
Q_bot = τ² Az_bot ℂ²_bot / (Δzᶠ_bot V_below)     (coupling to face k-1)
Q_top = τ² Az_top ℂ²_top / (Δzᶠ_top V_above)     (coupling to face k+1)
```

Wait — this needs to be reformulated. The current Helmholtz is for δρθ at centers.
The Schur complement is for δρw at FACES. The stencil is different.

**Second derivative at face k**: involves ρw at faces k-1, k, k+1.
The ∂²(ρw)/∂z² at face k uses the second-order centered stencil through
the two neighboring centers (k-1 and k).

This requires: the ρθ equation maps faces → centers (divergence ∂ρw/∂z),
and the PGF maps centers → faces (gradient ∂ρθ/∂z). The composition
∂/∂z(ℂ² ∂/∂z) maps faces → centers → faces, giving the stencil:

At face k:
```
(ℂ²∂²ρw/∂z²)|_k = ℂ²[k] (ρw[k+1]-ρw[k])/Δz_center[k] - ℂ²[k-1] (ρw[k]-ρw[k-1])/Δz_center[k-1]
                   ──────────────────────────────────────────────────────────────────────────────────
                                              Δzᶠ[k]
```

Note: ℂ² is at CENTERS (not faces), and Δz_center is the cell width.
This gives the acoustic tridiagonal entries:
```
a_k = -τ² ℂ²[k-1] / (Δz_center[k-1] Δzᶠ[k])
c_k = -τ² ℂ²[k]   / (Δz_center[k]   Δzᶠ[k])
b_k = 1 - a_k - c_k
```

**First derivative at face k** (gravity): involves ρw at faces through δρ.
δρ[k] = -τ (ρw[k+1] - ρw[k])/Δz_center[k]. Interpolated to face k:
δρ_face[k] = (δρ[k-1] + δρ[k])/2

The gravity contribution: -τ δρ_face g
= -τ g [(δρ[k-1] + δρ[k])/2]
= -τ g [(-τ(ρw[k]-ρw[k-1])/Δz[k-1]) + (-τ(ρw[k+1]-ρw[k])/Δz[k])] / 2
= τ²g/2 [(ρw[k]-ρw[k-1])/Δz[k-1] + (ρw[k+1]-ρw[k])/Δz[k]]

Collecting terms:
```
gravity_lower = +τ²g / (2 Δz_center[k-1])     (from δρw[k-1])
gravity_diag  = -τ²g / (2 Δz_center[k-1]) + τ²g / (2 Δz_center[k])
gravity_upper = -τ²g / (2 Δz_center[k])       (from δρw[k+1])
```

For uniform Δz: gravity_lower = +τ²g/(2Δz), gravity_upper = -τ²g/(2Δz),
gravity_diag = 0. This is the standard centered first-derivative stencil.

## Combined tridiagonal

```
a_k = a_acoustic + gravity_lower
b_k = b_acoustic + gravity_diag
c_k = c_acoustic + gravity_upper
```

The matrix is NOT symmetric (gravity adds skew-symmetric terms).
But it IS diagonally dominant as long as the acoustic terms dominate
the gravity terms: τ²ℂ²/Δz² >> τ²g/Δz, i.e., ℂ²/Δz >> g, i.e.,
Δz << ℂ²/g ≈ 120000/10 = 12000 m. For typical atmospheric grids
(Δz ≤ 2000 m), this is satisfied.

## Back-substitution

After solving for δρw:

```
δρθ[k] = -τ θ₀[k] (ρw_i[k+1] - ρw_i[k]) / Δz_center[k]    (from eq 2)
        = -τ θ₀[k] (ρw*[k+1]+δρw[k+1] - ρw*[k]-δρw[k]) / Δz_center[k]

δρ[k]  = -τ (ρw_i[k+1] - ρw_i[k]) / Δz_center[k]            (from eq 3)
```

Update fields:
```
ρθ_i = ρθ* + δρθ
ρ_i  = ρ*  + δρ
ρw_i = ρw* + δρw
```

## Stored fᴵ

The residual gives the full linearized implicit function:
```
fᴵ_ρw = δρw / τ     (includes acoustic perturbation PGF + mean PGF + gravity)
fᴵ_ρθ = δρθ / τ     (compressibility flux response)
fᴵ_ρ  = δρ  / τ     (continuity response)
```

## Key properties

1. For balanced state (ρw*=0): RHS ≈ -τ·O(Δz²). The solve produces
   δρw ≈ O(Δz²·τ) — small but nonzero from truncation error.

2. The gravity coupling IS in the operator (not just the RHS), so the
   tridiagonal inversion handles the PGF-gravity balance internally.

3. The tridiagonal is for δρw at FACES (Nz+1 values, with δρw=0 at
   boundaries). This is different from the current Helmholtz which is
   for δρθ at CENTERS (Nz values).

4. Back-substitution gives δρθ and δρ from the solved δρw. Both ρθ and
   ρ are updated consistently from the SAME δρw — no ρ/ρθ inconsistency.
