# WRF vs MPAS acoustic substepping: comparative analysis

## Overview

Both WRF and MPAS use split-explicit time integration with the Wicker-Skamarock RK3
outer scheme and forward-backward acoustic substeps. They share the same algorithmic
lineage (Klemp, Skamarock, and collaborators) but differ in grid structure and some
implementation details. This document compares them to guide Breeze's implementation.

## Outer RK3 scheme (identical in both)

Wicker-Skamarock (2002) low-storage RK3:

| Stage | Interval | Substeps |
|-------|----------|----------|
| 1 | Δt/3 | ~Nₛ/3 |
| 2 | Δt/2 | ~Nₛ/2 |
| 3 | Δt | Nₛ |

Each stage:
1. Computes slow tendencies from current state
2. Runs acoustic substeps over the stage interval
3. Uses time-averaged velocities for scalar advection

The acoustic substep size `Δτ = Δt/Nₛ` is approximately constant across stages.
Importantly, each stage resets to the saved state at time level t, not to the
previous stage output.

## Acoustic substep: forward-backward structure (both models)

Within each substep n:

```
Step 1 (FORWARD):  u^{n+1} = u^n + Δτ (Gˢu - cₚ θᵥ ∂Π̃'/∂x)    [old pressure]
Step 2 (BACKWARD): Π'^{n+1} from div(u^{n+1}, v^{n+1}, w^{n+1})  [new velocity]
Step 3:            w^{n+1} from vertically implicit solve
Step 4:            Divergence damping on Π' or u
```

## Key differences

### 1. Prognostic variables in the acoustic loop

| | WRF | MPAS |
|---|---|---|
| Horizontal momentum | u, v (velocity) | ru_p (ρd·u·J perturbation) |
| Vertical momentum | w (velocity) | rw_p (ρd·ω·J perturbation) |
| Pressure | π' (Exner perturbation) | rtheta_pp (ρθ double-perturbation) |
| Density | diagnostic (from π' via EOS) | rho_pp (ρ double-perturbation) |
| Coordinate | height (z) or terrain-following | terrain-following height |

**Key insight**: WRF advances the Exner pressure π' directly. MPAS advances ρθ (potential
temperature density) and derives pressure diagnostically. Both are equivalent formulations
of the pressure equation — one in Exner form, one in conservation form.

### 2. Perturbation definitions

**WRF**: Single perturbation from fixed 1D base state
```
π = π₀(z) + π'       (π₀ fixed, π' evolved by acoustic loop)
θ = θ₀(z) + θ'       (θ₀ fixed, only used for linearization)
```

**MPAS**: Double perturbation
```
ρθ = ρθ_base(z) + ρθ_p_save + ρθ_pp
                  ─────────   ───────
                  slow pert   acoustic pert (reset to 0 each RK stage)
```
The `_pp` variables measure the acoustic response within one RK stage.
The `_p_save` captures the slow evolution up to the start of the stage.

### 3. Vertically implicit solve

Both solve a tridiagonal system for w (or rw_p) that couples vertical velocity
to the pressure/theta equation through vertical flux divergence.

**WRF tridiagonal** (for w):
- Eliminates π' by substituting the backward Exner update into the w equation
- Coefficients involve `mm = Δτ · rdz · cₚ · θᵥ_face` (PGF coefficient)
- Off-centering parameter α (default 0.5-0.6): `β = 1 - α`
- Explicit part uses β-weighted old-time vertical terms
- Implicit part uses α-weighted new-time vertical terms

**MPAS tridiagonal** (for rw_p):
- Eliminates both rtheta_pp AND rho_pp by substituting their updates
- Three physical couplings in the tridiagonal:
  - `cofwz`: Exner PGF from θ perturbation (acoustic, like WRF)
  - `cofwr`: buoyancy from ρ perturbation (gravity-density coupling)
  - `cofwt`: buoyancy correction from θ perturbation (linearized EOS feedback)
- Off-centering: `epssm` (default 0.1), with `resm = (1-epssm)/(1+epssm)`

**The cofwt term**: This is absent in WRF's tridiagonal but present in MPAS.
It captures how θ perturbations modify the pressure through the nonlinear EOS,
providing additional gravity wave coupling in the implicit solve. In height
coordinates (WRF), this effect is partly captured by the Exner pressure equation.

### 4. Divergence damping

**WRF**: Forward extrapolation of Exner pressure
```
Π̃' = Π' + κᵈⁱ (Π'^{n+1} - Π'^n)
```
Applied to the pressure used in the NEXT substep's horizontal PGF.
Coefficient κᵈⁱ ≈ 0.10. Damps horizontal acoustic modes.

**MPAS**: Klemp, Skamarock & Ha (2018) 3D divergence damping
```
ru_p += coef · grad_h(-(rtheta_pp - rtheta_pp_old) / θₘ)
```
Applied as a velocity correction AFTER each substep. Uses the time tendency of ρθ
as a proxy for 3D divergence (since ∂(ρθ)/∂t ∝ -div(ρθ·v)). This ensures the
damping uses the SAME discrete divergence as the pressure equation — no spurious
modification of gravity wave frequencies.

**Breeze currently implements** the WRF-style forward extrapolation (κᵈⁱ in
`SplitExplicitTimeDiscretization`). The MPAS-style 3D filter could be added later.

### 5. Slow tendency computation

Both compute slow tendencies ONCE per RK stage and freeze them during substeps.

**WRF w-tendency includes**:
```
wten = advection + Coriolis + buoyancy - cₚ θᵥ ∂π'/∂z
```
The full vertical PGF (perturbation Exner gradient + buoyancy) is in the slow tendency.

**MPAS w-tendency includes** (line 5905-5907):
```
tend_w = advection + Coriolis - cqw · (rdzu · (pp[k]-pp[k-1]) - fzm·dpdz[k] - fzp·dpdz[k-1])
```
where:
- `pp` = perturbation pressure = `zz · Rᵈ · (Π · ρθ_p + ρθ_base · (Π - Π_base))`
- `dpdz` = buoyancy = `-g · (ρ_base · q_total + ρ_p_save · (1 + q_total))`

The buoyancy `dpdz` is computed from the BASE state density + SAVED perturbation density.
It uses a 1D base state even for baroclinic flows.

### 6. Density handling

**WRF**: Density is NOT an independent prognostic in the acoustic loop. The pressure
equation (Exner form) implicitly advances density through the EOS. After the acoustic
loop, density is recovered diagnostically: `ρ = p/(Rᵈ·T)`.

**MPAS**: Density IS advanced in the acoustic loop as `rho_pp`:
```
rho_pp = rs - cofrz · (rw_p[k+1] - rw_p[k])
```
where `rs` includes horizontal flux divergence from `ru_p`, slow tendency, and old-time
vertical divergence. The new-time vertical divergence uses the just-solved `rw_p`.

### 7. Time averaging for scalar transport

Both time-average the acoustic velocities for consistency with scalar advection.

**WRF**: Simple arithmetic mean `ūᵢ = (1/Nₛ) Σ uⁿ` for horizontal; off-centered
average for w: final substep weighted by α.

**MPAS**: Same for horizontal (`ruAvg = ru_save + ruAvg/ns`). For vertical,
off-centered average matching the implicit solver:
```
wwAvg += 0.5·(1-epssm)·rw_p_old + 0.5·(1+epssm)·rw_p_new
```

## WRF v4 acoustic step equations (from NCAR/TN-556+STR)

WRF uses a mass (eta) coordinate with prognostic geopotential. The acoustic
perturbation variables (double-prime) are deviations from the current RK predictor:

**Forward step for U** (Eq. 3.7):
```
δτ(U'') + (mx/my)(α/αd)^{t*} [μd^{t*} (αd d_x(p'') + α''d d_x(p̄) + d_x(φ''))
          + d_x(φ)(d_η(p'') - μ''d)]^τ = R_U^{t*}
```

**Potential temperature** (Eq. 3.10, forward using new U,V,Ω):
```
δτ(Θ''m) + mx my [d_x(U'' θm^{t*}) + d_y(V'' θm^{t*})]^{τ+Δτ}
         + my d_η(Ω''^{τ+Δτ} θm^{t*}) = R_{Θm}^{t*}
```

**Vertically implicit W-φ coupling** (Eqs. 3.11-3.12, off-centered):
```
δτ(W'') - g/my (α/αd)^{t*} {d_η(C d_η(φ'')) + d_η(cs²/αd · Θ''m/Θm) - μ''d}^{τ̄} = R_W^{t*}
δτ(φ'') + (1/μd^{t*}) [my Ω'' d_η(φ^{t*}) - my g W''^{τ̄}] = R_φ^{t*}
```
where `{·}^{τ̄}` denotes off-centered average: `((1+β)/2)·new + ((1-β)/2)·old`

**Linearized EOS** (Eq. 3.5):
```
p'' = (cs²/αd^{t*}) (Θ''m/Θm^{t*} - α''d/αd^{t*} - μ''d/μd^{t*})
```

**Divergence damping** (Eq. 4.11):
```
p^{*,τ} = p^τ + γd (p^τ - p^{τ-Δτ})        γd ≈ 0.1
```

### WRF vs MPAS: coordinate system implications

| | WRF (mass coordinate) | MPAS (height coordinate) | Breeze |
|---|---|---|---|
| Vertical coord | η (terrain-following pressure) | ζ (terrain-following height) | z (height) |
| Implicit couple | W'' ↔ φ'' (geopotential) | rw_p ↔ rtheta_pp + rho_pp | ρw ↔ ρθ |
| Column mass | μ_d (prognostic, integrated) | Not needed (height coord) | Not needed |
| EOS linearization | p'' from (Θ''m, α''d, μ''d) | Implicit through cofwz, cofwr, cofwt | Through ℂᵃᶜ² |
| Geopotential | Prognostic φ'' | Diagnostic (gz) | Diagnostic (gz) |

**For Breeze**: MPAS's height-coordinate formulation maps directly. WRF's mass-coordinate
adds complexity (prognostic μ_d, φ'') that we don't need.

## Summary: what Breeze should implement

For global baroclinic wave simulations, the acoustic substepping approach (WRF/MPAS)
is more robust than IMEX-ARK because:
1. The outer RK3 applies ALL slow tendencies with uniform Butcher weights
2. The acoustic substeps handle fast modes through forward-backward iteration (not a single linearized Helmholtz)
3. No fᴱ/fᴵ splitting with different weights → no cancellation amplification

Breeze already has acoustic substepping infrastructure (`SplitExplicitTimeDiscretization`).
The key improvements needed (informed by this analysis):

1. **Per-RK-stage perturbation variables** (MPAS style): reset acoustic perturbations
   to zero at the start of each RK stage, not relative to a fixed reference
2. **3D divergence damping** (Klemp et al. 2018): use ∂(ρθ)/∂t as the divergence proxy
3. **cofwt buoyancy correction** in the vertically implicit tridiagonal (MPAS has it, WRF doesn't)
4. **Consistent time averaging** with off-centering for scalar transport
