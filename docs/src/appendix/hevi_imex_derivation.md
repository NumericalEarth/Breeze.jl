# HEVI IMEX derivation for compressible dynamics

## 1. Full continuous equations

The prognostic variables are density ρ, momentum (ρu, ρv, ρw), and
potential temperature density ρθ. The vertical momentum and thermodynamic
equations are:

```
∂(ρw)/∂t = -∇·(ρw ⊗ v) - ∂p/∂z - ρg + [Coriolis, metric terms]
∂(ρθ)/∂t = -∇·(ρθ v)
∂ρ/∂t    = -∇·(ρv)
```

Pressure is diagnosed from the equation of state:

```
p = p₀ (Rᵈ ρθ / p₀)^γ
```

where γ = cₚ/cᵥ.

## 2. Base state + acoustic perturbation decomposition

At each implicit stage, we have a predictor state `(ρw⁻, ρθ⁻, ρ⁻)`
computed from previous explicit and implicit tendency contributions.
The pressure at this state is:

```
p⁻ = p₀ (Rᵈ ρθ⁻ / p₀)^γ
```

We seek the corrected state `(ρw⁺, ρθ⁺, ρ⁺)` where:

```
ρw⁺ = ρw⁻ + δρw
ρθ⁺ = ρθ⁻ + δρθ
ρ⁺  = ρ⁻  + δρ
p⁺  = p⁻  + δp
```

## 3. Linearization of the perturbation pressure

The pressure perturbation to first order in δρθ:

```
δp = (∂p/∂(ρθ))|₋ δρθ = (ℂ²/θ)⁻ δρθ
```

where `ℂ² = γ Rᵐ T = γ p⁻ / ρ⁻` is the squared acoustic speed and
θ⁻ = ρθ⁻/ρ⁻. This linearization is accurate when δρθ is small compared
to ρθ⁻.

## 4. Linearized implicit equations for the perturbation

The perturbation evolves over implicit time scale τ = γᵢ h (SDIRK diagonal
times Δt):

**Vertical momentum perturbation:**
```
δρw / τ = -(ℂ²/θ)⁻ ∂(δρθ)/∂z - δρ g
```

This is the linearized pressure gradient of the perturbation plus the
buoyancy perturbation. Note: the full `∂p⁻/∂z + ρ⁻g` is NOT here — it
was already applied by the explicit step.

**ρθ perturbation** (vertical flux coupling):
```
δρθ / τ = -V⁻¹ δz(Az θ⁻ δρw)
```

Treating θ as frozen at the predictor value θ⁻. This represents the change
in ρθ due to vertical mass flux convergence/divergence.

**Density perturbation:**
```
δρ / τ = -V⁻¹ δz(Az δρw)
```

## 5. Helmholtz equation (eliminating δρw)

Substituting the δρw equation into the δρθ equation (and dropping the
small δρ g term):

```
δρθ = -τ V⁻¹ δz(Az θ⁻ [ρw⁻ + δρw])
    ≈ -τ V⁻¹ δz(Az θ⁻ ρw⁻) + τ² V⁻¹ δz(Az ℂ² ∂(δρθ)/∂z / Δz)
```

Wait — more carefully. The Helmholtz RHS uses the full `ρw⁻` (the predictor
value), not `δρw`. This is because the implicit equation for ρθ is:

```
ρθ⁺ = ρθ⁻ + τ fᴵ_ρθ(ρw⁺)
```

where `fᴵ_ρθ(ρw⁺) = -V⁻¹ δz(Az θ⁻ ρw⁺)`. This is the vertical flux
of ρθ at the **corrected** ρw. Substituting `ρw⁺ = ρw⁻ - τ(ℂ²/θ)∂(δρθ)/∂z`:

```
δρθ = ρθ⁺ - ρθ⁻ = -τ V⁻¹ δz(Az θ⁻ ρw⁻) + τ² V⁻¹ δz(Az ℂ² ∂(δρθ)/∂z / Δz)
```

Rearranging:

```
[I - τ² L] δρθ = -τ V⁻¹ δz(Az θ⁻ ρw⁻)
```

where `L = V⁻¹ δz(Az ℂ² δz(·)/Δz)` is the positive-definite vertical
diffusion operator. This is the tridiagonal Helmholtz equation solved in
the code.

### Important: the RHS contains ρw⁻, not δρw

The RHS is `-τ V⁻¹ δz(Az θ⁻ ρw⁻)` where `ρw⁻` is the predictor value
of ρw. If the explicit step correctly applies the full PGF+buoyancy, then
`ρw⁻` is approximately in balance and the RHS is small (acoustic noise).
If the explicit step has no PGF (Option F), then `ρw⁻` is wildly unbalanced
and the RHS is enormous, breaking the linearization.

## 6. Back-solve

Given δρθ from the Helmholtz solve:

```
ρw⁺ = ρw⁻ - τ (ℂ²/θ)⁻ ∂(δρθ)/∂z
```

The code implements this using `δρθ = ρθ⁺ - ρθ⁻` (stored as the difference
between the post-solve and pre-solve ρθ).

## 7. Density update

```
ρ⁺ = ρ⁻ - τ V⁻¹ δz(Az ρw⁺)
```

This applies the vertical divergence of the corrected ρw to update ρ.

## 8. Time discretization: explicit + implicit

The IMEX-RK stage predictor assembles:

```
z⁻ᵢ = yₙ + h Σⱼ [aᴱᵢⱼ fᴱ(zⱼ) + aᴵᵢⱼ fᴵ(zⱼ)]
```

### What goes in fᴱ (explicit tendency)

```
fᴱ_ρu = full (advection, horizontal PGF, Coriolis, metric terms)
fᴱ_ρv = full
fᴱ_ρw = full (advection + vertical PGF ∂p/∂z + buoyancy ρg + Coriolis + metric)
fᴱ_ρθ = full 3D advection -∇·(ρθ v)
fᴱ_ρ  = full 3D divergence -∇·(ρv)
```

The explicit tendency uses the FULL vertical PGF and buoyancy computed from
the nonlinear equation of state at the current stage value. This keeps the
state approximately in hydrostatic balance between stages.

### What goes in fᴵ (implicit tendency from the solve)

```
fᴵ_ρw = (ρw⁺ - ρw⁻) / τ = -(ℂ²/θ) ∂(δρθ)/∂z
fᴵ_ρθ = (ρθ⁺ - ρθ⁻) / τ = δρθ / τ
fᴵ_ρ  = -V⁻¹ δz(Az ρw⁺)
fᴵ_ρu = 0
fᴵ_ρv = 0
```

Note: fᴵ_ρw is the **linearized perturbation** PGF response to the ρθ change.
It is NOT the full `∂p/∂z`. It does not duplicate the explicit PGF.

### Why there is no double-counting

The explicit tendency contributes `∂p⁻/∂z + ρ⁻g` (exact, nonlinear, at
the predictor state).

The implicit tendency contributes `(ℂ²/θ) ∂(δρθ)/∂z` (linearized
perturbation correction).

Together, the effective PGF at the corrected state is:

```
∂p⁻/∂z + (ℂ²/θ) ∂(δρθ)/∂z ≈ ∂p⁺/∂z
```

This is a first-order Taylor expansion of the PGF at the corrected state.
The two terms are complementary, not duplicative.

### Why removing the explicit PGF breaks things

If fᴱ_ρw has no vertical PGF (set to zero), then the stage predictor
ρw⁻ has no vertical restoring force. It is far from hydrostatic balance.
The Helmholtz RHS `-τ V⁻¹ δz(Az θ⁻ ρw⁻)` is then enormous because ρw⁻
is large (unbalanced). The solve must produce a large δρθ to compensate.
But the linearization `δp = (ℂ²/θ) δρθ` is only accurate for small δρθ.
The result is an inaccurate correction that makes things worse.

## 9. Connection to ERF / Klemp (2007) substepping

In the acoustic substepping approach (ERF, MPAS), the same decomposition
is used but applied at each **substep** within an RK stage. The slow
tendencies (advection, Coriolis) are frozen during the substeps. Each
substep advances the perturbation variables (δu, δw, δρθ, δρ) with the
linearized acoustic equations. The implicit solve within each substep
handles only the vertical column.

The key difference: with substepping, the perturbation variables are
explicitly defined as deviations from the frozen RK-stage state. With
our single-correction IMEX-RK approach, the perturbation is implicit
in the Helmholtz solve residual.

## 10. Summary

The correct IMEX discretization for the linearized Helmholtz approach:

1. **Explicit step**: compute full tendencies including vertical PGF+buoyancy
   from the nonlinear EOS. This keeps the state approximately balanced.

2. **Implicit solve**: find the acoustic correction (δρθ, δρw, δρ) that
   accounts for the O(Δt) error in the explicit vertical acoustic treatment.
   The correction is small, so the linearization is accurate.

3. **Do NOT remove** the vertical PGF from the explicit tendency. The
   linearized solver cannot provide the full PGF — only a perturbation
   correction.

## References

- Ascher, Ruuth, Spiteri (1997). Implicit-explicit Runge-Kutta methods.
- Gardner et al. (2018, GMD). IMEX Runge-Kutta methods for non-hydrostatic
  atmospheric models.
- Klemp et al. (2007, MWR). Conservative split-explicit time integration.
- Klemp et al. (2018, MWR). Damping acoustic modes in HEVI and split-explicit
  schemes.
- Lattanzi et al. (2025, JAMES). ERF: Energy Research and Forecasting Model.
