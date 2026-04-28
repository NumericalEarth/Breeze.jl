# Phase 1 derivation — linearized acoustic-buoyancy column system

This document is the canonical derivation that the substepper code in
`src/CompressibleEquations/acoustic_substepping.jl` must match. Each
algebraic step is intended to be verifiable line-by-line against the
implementation; cross-references to the current code are noted at the
end.

The derivation establishes prime notation `ρ′`, `(ρθ)′`, `(ρu)′` etc.
(Standard S13). The symbols `σ` and `η` previously used in the file
are *retired* — they collide with the sigma vertical coordinate and the
hybrid coordinate / surface elevation respectively in standard
atmospheric-science notation.

## 1. Notation

Subscripts and superscripts:

- `0` (superscript) — frozen at the **outer-step start** linearization
  point `U⁰ = (ρ⁰, ρθ⁰, ρu⁰, ρv⁰, ρw⁰)` for the outer Δt. Π⁰, p⁰, θ⁰
  are diagnosed from the EoS at `U⁰`.
- `_ref` — reference state (function of z only) such that `δz(p_ref) +
  g·ℑz(ρ_ref) = 0` to ulp at every face. The reference state is the
  vertically stratified hydrostatic background; *the substepper assumes
  this discrete identity holds* (Standard S3).
- `′` (prime) — perturbation about `U⁰`. So `ρ′ = ρ − ρ⁰`, `(ρθ)′ = ρθ −
  ρθ⁰`, `(ρu)′ = ρu − ρu⁰`, etc. *These are the substepper's prognostics.*
- `n` (subscript), `o` (subscript) — quantities at the new and old
  substep times within a single substep `[τ, τ + Δτ]`. So `ρ′_n = ρ′(τ
  + Δτ)`, `ρ′_o = ρ′(τ)`.

Off-centering parameter:

- `ω_n ∈ [1/2, 1]` — the new-step weight in the Crank-Nicolson average.
- `ω_o = 1 − ω_n` — the old-step weight. ω_o ≥ 0.
- ε ≡ 2ω_n − 1 ∈ [0, 1] — the off-centering. ε = 0 is centered CN.
- `δτ_n ≡ ω_n · Δτ` and `δτ_o ≡ ω_o · Δτ`. These are the central
  algebraic quantities; the matrix is parameterized by `δτ_n`.

Discrete spatial operators on a vertically stretched, horizontally
staggered C-grid. We use Oceananigans's notation: `δz` is a difference
operator and `ℑz` is the average operator, with the location subscript
indicating where the operator *lives* (i.e., where its result sits):

- `δz_f(Q)[k_f] = Q_c[k_f] − Q_c[k_f − 1]` — center → face difference.
  Here `k_c = k_f − 1` is the cell *below* face `k_f`. (Substepper's
  PGF needs this on `(ρθ)′`.)
- `ℑz_f(Q)[k_f] = (Q_c[k_f] + Q_c[k_f − 1]) / 2` — center → face mean.
  (Substepper's buoyancy needs this on `ρ′`.)
- `δz_c(M)[k_c] = M_f[k_c + 1] − M_f[k_c]` — face → center difference.
  (Substepper's mass continuity and θ-flux divergence both use this.)
- `ℑz_c(M)[k_c] = (M_f[k_c + 1] + M_f[k_c]) / 2` — face → center mean.
  (Used in the buoyancy back-substitution.)

Vertical spacings: `Δz_c[k_c] ≡ z_f[k_c + 1] − z_f[k_c]` is the
cell-center spacing; `Δz_f[k_f] ≡ z_c[k_f] − z_c[k_f − 1]` is the
face-to-face spacing as seen by a face-centered field. They are
*independent* on a stretched grid, equal only on uniform-Δz grids.

Throughout: `f` subscripts refer to **z-faces** (where μw, ρw live);
`c` subscripts refer to **cell centers** (where ρ, ρθ, ρ′, (ρθ)′
live). `μu, μv` live at x-faces and y-faces respectively (Arakawa C
staggering); their vertical structure is on cell centers.

## 2. Continuous governing equations

The conservation form of the compressible Euler equations, in the
absence of subgrid closures and forcing (which add into `Gˢ`):

```
∂ρ/∂t  + ∇·m              = Gˢρ
∂ρθ/∂t + ∇·(θ m)          = Gˢρθ
∂m/∂t  + ∇·(m ⊗ u) + ∇p
        + g ρ ẑ           = Gˢm
```

with `m ≡ ρu = (ρu, ρv, ρw)`, `u = m/ρ`, and the equation of state

```
p = (R ρθ)^γ / pˢᵗ^(γ−1).            (*)
```

Here `R` is the *moist* mixture gas constant (Standard S2: not `Rᵈ`)
and `γ = cᵖ/cᵛ` is the *moist* mixture ratio. For dry-only flow, `R =
Rᵈ`, `γ = γᵈ`, and (*) reduces to the dry ideal-gas form.

Differentiating (*) at fixed `ρ` and combining `Π = (p/pˢᵗ)^κ` (κ ≡
R/cᵖ):

```
∂p / ∂(ρθ) |_(ρ) = γ R Π                            (**)
```

This is the linearized PGF coefficient. Note that *the only role* `R`
and `γ` play in the substepper is through (**) — they enter no
geometric or kinematic expression.

## 3. Linearization at `U⁰`

Decompose `U = U⁰ + U′`. The linearized continuous perturbation
equations, dropping `O(U′²)`:

```
∂t ρ′      + ∇·m′                                       = Gˢρ
∂t (ρθ)′   + ∇·(θ⁰ m′)                                  = Gˢρθ
∂t (ρu)′   +  γ R Π⁰ ∂x((ρθ)′)                          = Gˢρu
∂t (ρv)′   +  γ R Π⁰ ∂y((ρθ)′)                          = Gˢρv
∂t (ρw)′   +  γ R Π⁰ ∂z((ρθ)′)  +  g ρ′                 = Gˢρw          (***)
```

- *No* `θ⁰` flux of `m′` in the (ρu)′ and (ρv)′ equations: the
  thermodynamic flux only appears in the (ρθ)′ equation.
- The buoyancy term `g ρ′` appears only in the *vertical* momentum
  equation. *This is dry buoyancy*: at this order the moist correction
  (virtual ρ′) is bookkept as a separate term and treated in Phase 3
  (Standard S5; audit B2). In the dry-only derivation that follows we
  retain `g ρ′` as written.
- `γ R Π⁰` is the moist-aware PGF coefficient. The current code uses
  `γᵈ Rᵈ Π⁰` everywhere — fix this in Phase 3 (Standard S2; audit A3,
  B1).

## 4. Reference-state decomposition of the slow tendency

The slow tendency `Gˢ` is computed by the WS-RK3 dynamics kernels in
`SlowTendencyMode`, with **the PGF and buoyancy zeroed out** of `Gⁿρu,
Gⁿρv, Gⁿρw`. We must reinstate them at `U⁰` so that the *full*
nonlinear vertical-momentum equation reads

```
∂t (ρw) = -∇·(ρw u) - ∂z p - g ρ + Gⁿρw |_advection-only.
```

Decompose the total state as `U = U_ref + U⁰ − U_ref + U′`. With the
reference satisfying `∂z p_ref + g ρ_ref = 0` exactly *in the
substepper's discrete operators*, the right-hand side splits:

```
- ∂z p - g ρ  =  - ∂z(p − p_ref) - g (ρ − ρ_ref)
              =  - ∂z(p⁰ − p_ref)        - γ R Π⁰ ∂z((ρθ)′)            ← linearized
                 - g (ρ⁰ − ρ_ref)        - g ρ′                         ← linearized
```

The U⁰-only pieces `−∂z(p⁰ − p_ref) − g (ρ⁰ − ρ_ref)` are *frozen* over
the substep and become part of the slow tendency:

```
Gˢρw[k_f] := Gⁿρw[k_f] − δz_f(p⁰ − p_ref)[k_f] / Δz_f[k_f]
                       − g · ℑz_f(ρ⁰ − ρ_ref)[k_f].                    (1)
```

The perturbation-only pieces remain inside the substep loop and become
the linearized sound and buoyancy forces on `(ρw)′`. This split is
what `assemble_slow_vertical_momentum_tendency!` implements.

The horizontal counterpart for `(ρu)′` is structurally identical:

```
Gˢρu := Gⁿρu − δx_f(p⁰ − p_ref) / Δx_f.                                (2)
```

(p_ref depends only on z for ExnerReferenceState, so the horizontal
reference-pressure gradient is zero. We retain the form (2) for
generality.) The substep loop then adds the linearized `−γ R Π⁰
∂x((ρθ)′)` perturbation force on top of `Gˢρu`.

## 5. Discrete spatial operators

The discrete linearized perturbation system on the staggered C-grid:

```
∂t ρ′[k_c]    + δz_c((ρw)′)[k_c] / Δz_c[k_c]                 + 𝒟ₕ[k_c]    = Gˢρ[k_c]
∂t (ρθ)′[k_c] + δz_c(θ⁰_f · (ρw)′)[k_c] / Δz_c[k_c]          + 𝒟ₕ_θ[k_c]  = Gˢρθ[k_c]
∂t (ρu)′[ix_f]+ γ R Π⁰_x[ix_f] · δx_f((ρθ)′)[ix_f] / Δx_f    = Gˢρu[ix_f]
∂t (ρv)′[iy_f]+ γ R Π⁰_y[iy_f] · δy_f((ρθ)′)[iy_f] / Δy_f    = Gˢρv[iy_f]
∂t (ρw)′[k_f] + γ R Π⁰_z[k_f]  · δz_f((ρθ)′)[k_f]  / Δz_f[k_f]
              + g · ℑz_f(ρ′)[k_f]                                          = Gˢρw[k_f]
                                                                               (3)
```

with face-evaluated background quantities:

```
θ⁰_f[k_f]  ≡ ℑz_f(θ⁰)[k_f]   = (θ⁰[k_f] + θ⁰[k_f − 1]) / 2
Π⁰_z[k_f]  ≡ ℑz_f(Π⁰)[k_f]
Π⁰_x[ix_f] ≡ ℑx_f(Π⁰)[ix_f]
Π⁰_y[iy_f] ≡ ℑy_f(Π⁰)[iy_f]
```

`𝒟ₕ` and `𝒟ₕ_θ` are the horizontal flux divergences `∇ₕ·(ρu)′` and
`∇ₕ·(θ⁰ (ρu)′)`. They are evaluated explicitly (forward Euler) — the
implicit half is *vertical only*.

## 6. Off-centered Crank-Nicolson time discretization

Apply Crank-Nicolson to the *vertical* implicit half of (3) with weights
ω_n, ω_o = 1 − ω_n. The horizontal half of the momentum equations is
forward-Euler. Multiplying through by Δτ:

**Density (3a):**

```
ρ′_n[k_c] = ρ′_o[k_c] + Δτ Gˢρ[k_c] − Δτ 𝒟ₕ[k_c]
            − ω_n Δτ · δz_c((ρw)′_n)[k_c] / Δz_c[k_c]
            − ω_o Δτ · δz_c((ρw)′_o)[k_c] / Δz_c[k_c]
                                                                       (4)
```

Group everything that does *not* depend on `(ρw)′_n` into the
**density predictor** `ρ̃[k_c]`:

```
ρ̃[k_c] := ρ′_o[k_c] + Δτ Gˢρ[k_c] − Δτ 𝒟ₕ[k_c]
          − δτ_o · δz_c((ρw)′_o)[k_c] / Δz_c[k_c].                     (5)
```

Then (4) reduces to

```
ρ′_n[k_c] = ρ̃[k_c] − δτ_n · δz_c((ρw)′_n)[k_c] / Δz_c[k_c].           (6)
```

**Density-weighted potential temperature (3b):** identical structure
with the θ-flux:

```
(ρθ)̃[k_c] := (ρθ)′_o[k_c] + Δτ Gˢρθ[k_c] − Δτ 𝒟ₕ_θ[k_c]
             − δτ_o · δz_c(θ⁰_f · (ρw)′_o)[k_c] / Δz_c[k_c]            (7)

(ρθ)′_n[k_c] = (ρθ)̃[k_c] − δτ_n · δz_c(θ⁰_f · (ρw)′_n)[k_c] / Δz_c[k_c].
                                                                       (8)
```

**Vertical momentum (3e):**

```
(ρw)′_n[k_f] − (ρw)′_o[k_f]
  = Δτ · Gˢρw[k_f]
    − ω_n Δτ · γ R Π⁰_z[k_f] / Δz_f[k_f] · δz_f((ρθ)′_n)[k_f]
    − ω_o Δτ · γ R Π⁰_z[k_f] / Δz_f[k_f] · δz_f((ρθ)′_o)[k_f]
    − ω_n Δτ · g · ℑz_f(ρ′_n)[k_f]
    − ω_o Δτ · g · ℑz_f(ρ′_o)[k_f].                                    (9)
```

## 7. The implicit vertical solve

Substitute (6) and (8) into (9). For the PGF term:

```
δz_f((ρθ)′_n)[k_f] = δz_f((ρθ)̃)[k_f]
                    − δτ_n · {δz_c(θ⁰_f (ρw)′_n)[k_f] / Δz_c[k_f]
                              − δz_c(θ⁰_f (ρw)′_n)[k_f − 1] / Δz_c[k_f − 1]}.
                                                                       (10)
```

For the buoyancy term:

```
ℑz_f(ρ′_n)[k_f] = ℑz_f(ρ̃)[k_f]
                  − δτ_n / 2 · {δz_c((ρw)′_n)[k_f] / Δz_c[k_f]
                                + δz_c((ρw)′_n)[k_f − 1] / Δz_c[k_f − 1]}.
                                                                       (11)
```

Inserting (10) and (11) back into (9), and using `ω_n Δτ · δτ_n =
δτ_n²` (since `δτ_n = ω_n Δτ`), the new-step `(ρw)′_n` collects on the
LHS:

```
(ρw)′_n[k_f] · 1
  + δτ_n² · γ R Π⁰_z[k_f] / Δz_f[k_f]
            · {δz_c(θ⁰_f (ρw)′_n)[k_f] / Δz_c[k_f]
               − δz_c(θ⁰_f (ρw)′_n)[k_f − 1] / Δz_c[k_f − 1]}
  + δτ_n² · g / 2
            · {δz_c((ρw)′_n)[k_f] / Δz_c[k_f]
               + δz_c((ρw)′_n)[k_f − 1] / Δz_c[k_f − 1]}
                                                                       (12)
  =   (ρw)′_o[k_f]                                                     ← old-step μw
    + Δτ · Gˢρw[k_f]                                                   ← slow tendency
    − ω_n Δτ · γ R Π⁰_z[k_f] / Δz_f[k_f] · δz_f((ρθ)̃)[k_f]            ← predictor PGF
    − δτ_o · γ R Π⁰_z[k_f] / Δz_f[k_f] · δz_f((ρθ)′_o)[k_f]            ← old-step PGF
    − δτ_n · g · ℑz_f(ρ̃)[k_f]                                          ← predictor buoyancy
    − δτ_o · g · ℑz_f(ρ′_o)[k_f].                                       ← old-step buoyancy
                                                                       (13)
```

Rearranging the LHS into tridiagonal form: `δz_c((ρw)′_n)[k_c] =
(ρw)′_n[k_c + 1] − (ρw)′_n[k_c]`, so each `δz_c` term contributes to
two distinct face-row coefficients.

**Tridiagonal LHS coefficients** for face row `k_f ∈ [2, Nz]` (the
boundary rows `k_f = 1` and `k_f = Nz + 1` are trivial — see §7.4):

```
A[k_f, k_f − 1] :=  − δτ_n² · γ R Π⁰_z[k_f] · θ⁰_f[k_f − 1] · rdz_c[k_f − 1] / Δz_f[k_f]
                    + δτ_n² · g / 2                          · rdz_c[k_f − 1]

A[k_f, k_f]     :=  1
                    + δτ_n² · γ R Π⁰_z[k_f] · θ⁰_f[k_f] · (rdz_c[k_f] + rdz_c[k_f − 1]) / Δz_f[k_f]
                    + δτ_n² · g / 2                    · (rdz_c[k_f] − rdz_c[k_f − 1])

A[k_f, k_f + 1] :=  − δτ_n² · γ R Π⁰_z[k_f] · θ⁰_f[k_f + 1] · rdz_c[k_f] / Δz_f[k_f]
                    − δτ_n² · g / 2                          · rdz_c[k_f]
                                                                       (14)
```

with `rdz_c[k_c] ≡ 1 / Δz_c[k_c]`. **These are the entries that the
column tridiag must use.**

Note the *sign* asymmetry of the buoyancy in (14):

- Sub-diagonal (k_f, k_f − 1): buoyancy contribution **+δτ_n² g rdz_c[k_f − 1] / 2**
- Diagonal: **+δτ_n² g (rdz_c[k_f] − rdz_c[k_f − 1]) / 2** (zero on uniform Δz)
- Super-diagonal (k_f, k_f + 1): buoyancy contribution **−δτ_n² g rdz_c[k_f] / 2**

This sign pattern is what (11) imposes. It is *not* a discretization
choice; it is the algebraic consequence of substituting the (I')-style
post-solve relation into the buoyancy term.

**Tridiagonal RHS** (the right-hand side of (13)):

```
b[k_f] := (ρw)′_o[k_f] + Δτ · Gˢρw[k_f]
          − γ R Π⁰_z[k_f] / Δz_f[k_f] · {δτ_n · δz_f((ρθ)̃)[k_f]
                                         + δτ_o · δz_f((ρθ)′_o)[k_f]}
          − g · {δτ_n · ℑz_f(ρ̃)[k_f] + δτ_o · ℑz_f(ρ′_o)[k_f]}.        (15)
```

### 7.1 Predictors `ρ̃, (ρθ)̃` revisited

Definitions (5) and (7) repeat. The crucial point: **the predictor
uses `δτ_o` for the old-step vertical-flux contribution**, not `δτ_n`.

For the centered case ω_n = 1/2, `δτ_o = δτ_n` and the distinction is
moot. For the off-centered ω_n > 1/2 (e.g. 0.55), `δτ_o = (1 − ω_n) Δτ
< δτ_n`, and using the wrong weight introduces an algebraic mismatch
of size `(2 ω_n − 1) Δτ × δz_c(θ⁰ (ρw)′_o) / Δz_c`. *This is the BBI
report's hypothesis 1 (predictor / matrix off-centering mismatch).*

### 7.2 Post-solve recovery

From (6) and (8):

```
ρ′_n[k_c]    = ρ̃[k_c]    − δτ_n · δz_c((ρw)′_n)[k_c] / Δz_c[k_c]      (16)
(ρθ)′_n[k_c] = (ρθ)̃[k_c] − δτ_n · δz_c(θ⁰_f (ρw)′_n)[k_c] / Δz_c[k_c]. (17)
```

Both use `δτ_n` (not `δτ_o`); this is correct, since they reconstruct
the new-step value from the new-step `(ρw)′_n`. The factor that appears
in front of `δz_c((ρw)′_n)` is exactly the new-step weight applied to
the implicit half.

### 7.3 Horizontal momentum

Forward-Euler (entirely explicit):

```
(ρu)′_n[ix_f] = (ρu)′_o[ix_f] + Δτ · Gˢρu[ix_f]
              − Δτ · γ R Π⁰_x[ix_f] · δx_f((ρθ)′_o)[ix_f] / Δx_f.      (18)
```

There is no implicit term here, because horizontal acoustic-CFL is the
limiter we set the substep count `N` to satisfy. Forward-Euler is
stable for `cs · Δτ / Δx ≤ 1`. **Important:** the `(ρθ)′` in (18) is
the *current* substep's old value. (No CN average — that would couple
horizontal columns through θ⁰_f, which we explicitly choose not to do.)

The reference-pressure-imbalance contribution `−δx_f(p⁰ − p_ref) /
Δx_f` already lives inside `Gˢρu` per (2). In code this manifests as
`pressure_imbalance` being added to the horizontal explicit step
(audit confirmed: substepper_horizontal_pgf memory entry).

### 7.4 Boundary conditions

`(ρw)′[1] = 0` and `(ρw)′[Nz + 1] = 0` by impenetrability. These
translate to trivial boundary rows in the tridiagonal:

```
A[1, 1] = 1, A[1, 2] = 0, b[1] = 0   ⟹   (ρw)′_n[1] = 0.
A[Nz + 1, Nz] = 0, A[Nz + 1, Nz + 1] = 1, b[Nz + 1] = 0
                                       ⟹   (ρw)′_n[Nz + 1] = 0.
```

In the current implementation the bottom row is part of the solver
(rows `k = 1..Nz`) and the top face `Nz + 1` is set outside the solver.

## 8. Algebraic closure verification

### 8.1 Round-trip identity

Starting from `(ρw)′_n` from the solve, plug into (16)-(17) to get
`ρ′_n, (ρθ)′_n`. Then substitute back into the *original* discrete
equation (9) and the predictor formulas (5), (7). The result must
*identically* (algebraically, not just numerically) reproduce (9). This
is the closure check — the predictor and matrix must be exact algebraic
complements.

The verification is by direct substitution; it succeeds *if and only
if* the predictor uses `δτ_o`, not `δτ_n`, for the old-step
vertical-flux contributions. This is the closure proof.

### 8.1.1 Empirical Phase 4 finding: drift accumulates non-hydrostatic noise

After Phase 1 (algebra closure) and Phase 2 (discrete-balance reference),
the substepper passes T1, T2, T3, T6, S1-S7 at machine ε. The column-M
spectrum is positive-definite (`eigvals(M) ∈ [+1.7e-3, +1.84]`). Yet the
rest atmosphere at default ω = 0.55, Δt = 20 s **still amplifies at
1.77× / outer step**.

Per-substep diagnostic (`test/substepper_structural.jl::S8`,
`::S9`): all of `(ρ′, (ρθ)′, (ρw)′, Gˢρw, |p − p_ref|, hydro_residual)`
grow in lockstep at 1.77×/step. The hydrostatic-balance residual of the
drift `δz(p − p_ref)/Δz_face + g·ℑ_z(ρ − ρ_ref)` starts at 3e-14 N/m³
and reaches 4e-8 N/m³ over 30 outer steps — a factor of 10⁶.

**Mechanism.** The substep loop integrates `(ρ′, (ρθ)′, (ρw)′)` with
discrete operators that do *not* preserve the discrete hydrostatic
constraint `γRᵐ Π · δz_f((ρθ)′)/Δz_f + g · ℑ_f(ρ′) = 0` for
perturbations. After one outer step, the recovered drift has a small
*non-hydrostatic* component. At the next outer step's
`freeze_outer_step_state!`, the drift's hydrostatic residual becomes
the slow-tendency seed `Gˢρw`. The substep loop's DC response to a
unit Gˢρw seed has gain ≈ 1.83 (limited from above by `1/(δτ_n · λ_min)`).
Combined with the 6% per-outer-step CN damping, the closed-loop
amplification is ≈ 1.83 × 0.94 = 1.72 — matching observation.

**Two structural experiments tried, BOTH worse:**

1. **Per-stage refresh** of `outer_step_*` (audit B3, S10): envelope at
   Δt = 20 s, ω = 0.55 in 50 outer steps grows from 1.79 m/s (no
   refresh) to 41 m/s (with refresh).
2. **Current-state slow-tendency reference**: replacing
   `outer_step_pressure / outer_step_density` with `model.dynamics.pressure
   / model.dynamics.density` in `_assemble_slow_vertical_momentum_tendency!`
   — same 41 m/s in 50 steps as full refresh.

The frozen-at-outer-step-start design is empirically the LEAST BAD of
the three options, despite the WS-RK3 inconsistency it introduces.

**Conclusion.** Phase 4 is a non-local structural redesign. The fix is
not a single sign / index correction. Atmospheric models that handle
this cleanly use either:
  - A specific discrete operator pair `(δz_f, ℑ_f, δz_c)` chosen to
    *exactly* preserve the discrete hydrostatic-balance constraint
    (energy-conserving discretization).
  - An explicit hydrostatic-balance projection step at the end of each
    outer step (or each stage).
  - A different time-integration architecture (e.g., IMEX-RK with no
    substepping inside stages).

The current Breeze design satisfies neither.

### 8.2 Centered CN (ω_n = 1/2) is neutrally stable on the rest atmosphere

At rest with `U⁰ = U_ref`, the slow tendency `Gˢ` and the perturbation
state `(ρ′, (ρθ)′, (ρu)′, ...)` are all zero. The substep loop must
preserve this: any seeded numerical noise should not amplify.

The amplification operator for the linearized rest-atmosphere column
is `G(δτ_n) ≡ A(δτ_n)⁻¹ · B(δτ_n)`, where `A` is the LHS matrix from
(14) and `B` carries the old-step contributions from (13). For
centered CN (ω_n = 1/2):

- `δτ_n = δτ_o = Δτ / 2`.
- The old-step buoyancy contribution to (13) has coefficient `−δτ_o ·
  g = −(Δτ/2) g`; the new-step LHS contribution has coefficient `+δτ_n²
  g / 2 = (Δτ/2)² g / 2`. The two combine into a CN-symmetric form
  `(I − (Δτ/2)² M) μw_n = (I + ...) μw_o + ...`.
- For *any* matrix `M` whose discrete spectrum lies on the real axis
  (acoustic-buoyancy is energy-conserving in the continuous limit, so
  the spatial discretization must give a real spectrum), `G = (I + (Δτ/2)²
  M)⁻¹ (I − (Δτ/2)² M)` has eigenvalues `(1 − μ²)/(1 + μ²)` with `μ²
  = (Δτ/2)² · λ`, and `|G| ≤ 1` always.

**Therefore: a correctly-implemented centered CN must keep |G| ≤ 1 on
the rest atmosphere for any Δτ.** This is Standard S8.

The current code violates S8 (the eigenvalue scan in
`test/substepper_eigenvalue_scan.jl` shows `λ_min ≈ −8.79e−3` even on
uniform Δz). The reason is one of:

  (a) The predictor uses `δτ_n` (current code) instead of `δτ_o`,
      breaking the round-trip identity.
  (b) The reference state is not in discrete hydrostatic balance,
      seeding `(ρ′, (ρθ)′)` perturbations at every outer step (Phase 2).
  (c) The buoyancy off-diagonals (14) are sign-asymmetric (always),
      which combined with stratification of θ⁰ gives the operator a
      negative eigenvalue.

The Phase-1 derivation eliminates (a). Phase 2 eliminates (b). (c)
remains in (14) but is *bounded*: the asymmetry is `O(g · rdz_c)` in
absolute terms; the row-sum on a perfectly stratified isothermal-T
column produces a small constant offset that must be absorbed into the
reference-state seed. The Phase-2 reference-state fix exactly removes
this offset.

## 9. Cross-check against current code

### 9.1 Predictor formula (lines 742-749 of acoustic_substepping.jl)

Current code uses `δτ_new = ω_n · Δτ` for the old-step flux:

```julia
σ_pred[i, j, k] = σ[i, j, k] + Δτ * Gˢρ[i, j, k] - Δτ * div_h_M -
                  (δτ_new / Δz_c) * (μw_above - μw_here)         # ← δτ_n, but should be δτ_o
```

**Correct form (from (5)):**

```julia
σ_pred[i, j, k] = σ[i, j, k] + Δτ * Gˢρ[i, j, k] - Δτ * div_h_M -
                  (δτ_old / Δz_c) * (μw_above - μw_here)
```

with `δτ_old = (1 - forward_weight) * Δτ`. Same fix applies to η_pred.

### 9.2 Vertical RHS (lines 752-769 of acoustic_substepping.jl)

Current code (lines 760-768):

```julia
sound_force = γRᵈ * Π_face / Δz_face * δτ_new * (∂z_η_old + ∂z_η_pred)
buoy_force  = g * δτ_new * (σ_face_old + σ_face_pred)
μw_rhs[i, j, k] = μw[i, j, k] + Δτ * Gˢρw[i, j, k] - sound_force - buoy_force
```

This sums `δτ_new × (X_old + X_pred)`. **Correct form (from (15))**:

```julia
sound_force = γR_face / Δz_face * (δτ_old * ∂z_(ρθ)′_o + δτ_n * ∂z_(ρθ)̃)
buoy_force  = g * (δτ_old * ℑz(ρ′_o) + δτ_n * ℑz(ρ̃))
```

For ω_n = 1/2 the two are equal; for ω_n = 0.55 the difference is `(δτ_n
− δτ_o) × X_old = (2ω_n − 1) Δτ × X_old = 0.1 Δτ × X_old`, which feeds
back into the vertical RHS at every substep.

### 9.3 PGF coefficient `γᵈ Rᵈ` (audit A3, B1)

Current code uses dry-air mixture constants throughout the substep.
Replace `γᵈ Rᵈ` with `γᵐ Rᵐ` evaluated from the moist basic state
(Phase 3).

### 9.4 Sound speed in damping (audit A1)

Current code (lines 845-851) hardcodes `cs² = γᵈ Rᵈ × 300`. Phase 5
replaces with locally-evaluated `cs(i, j, k)`.

## 10. Naming conventions in the rewritten code (S13)

Replace throughout:

```
σ                   →   ρp        (or `density_perturbation_value`
                                    when used as a kernel argument
                                    description; struct field already
                                    named `density_perturbation`)
η                   →   ρθp       (or `density_potential_temperature_perturbation_value`)
σ̃                   →   ρp_predictor    (or `density_predictor`, matching the struct field)
η̃                   →   ρθp_predictor   (or `density_potential_temperature_predictor`)
σ_face              →   ρp_face
gσ_face             →   gρp_face
μu, μv, μw          →   ρup, ρvp, ρwp   (or as kernel args with descriptive names)
forward_weight      →   ω_new   (in derivation comments only;
                                  the public API name stays
                                  `forward_weight` for backward
                                  compatibility with the user-facing
                                  TimeDiscretization constructor)
```

In math: prime notation `ρ′`, `(ρθ)′`, `(ρu)′` etc. is the canonical
form; ASCII `ρp`, `ρθp`, `ρup` is the kernel-argument analogue.

## Summary of bugs identified by this derivation

1. **Predictor weight mismatch** (line 743, 748): the predictor uses
   `δτ_new` for the old-step vertical-flux contribution but should use
   `δτ_old`. Fix: introduce `δτ_old = (1 − ω) Δτ` as a kernel
   parameter, replace `δτ_new` with `δτ_old` in lines 743 and 748.
   Single-character change × 2 lines.

2. **Vertical-RHS weight mismatch** (lines 760-765): the sound and
   buoyancy forces use `δτ_new × (X_old + X_pred)`, but (15) says the
   correct form is `δτ_old × X_old + δτ_new × X_pred`. Fix: split the
   weights in the RHS assembly.

3. **PGF coefficient is dry** (lines 519, 539, 558, 685, 690, 760):
   `γᵈ Rᵈ` should be `γᵐ Rᵐ` evaluated locally. Phase 3.

4. **Buoyancy is dry** (matrix line 540, RHS line 765, slow-tendency
   line 622): `g · ρ′` should be `g · ρ′_v` (virtual-density
   perturbation) for moist flow. Phase 3.

5. **Sound speed in damping is hardcoded** (lines 845-851): should be
   `cs(i, j, k)` from the basic state. Phase 5.

6. **Reference state is not in discrete hydrostatic balance with the
   substepper's operators** (Phase 2; quantitative finding `5.13e-3
   N/m³`).

7. **σ, η notation collides with vertical-coordinate symbols** (S13;
   replace globally as part of this Phase 1 rewrite).

The Phase-1 patch addresses (1), (2), and (7). Phase 2 addresses (6).
Phase 3 addresses (3) and (4). Phase 5 addresses (5).

## How this derivation is intended to be used

1. **Read it cold.** Anyone touching `acoustic_substepping.jl` should be
   able to read this derivation without prior project context and
   verify each line of the implementation against it.

2. **Patch in the order above.** Fix (1) and (2) first — they are the
   smallest-blast-radius changes and they are *necessary* for the
   centered-CN neutrally-stable property (S8) to hold.

3. **Re-run Phase-0 tests.** After (1) and (2), `test/substepper_eigenvalue_scan.jl`
   should show `re_min ≥ 0`. After Phase 2, T1 should pass. Together,
   T4-failing should pass at any reasonable Δt.

4. **Promote to file header.** Once the algebra is verified end-to-end,
   move the body of this document into the head of
   `src/CompressibleEquations/acoustic_substepping.jl` (replacing
   lines 1-41 plus 483-507) so it lives next to the code it
   constrains.
