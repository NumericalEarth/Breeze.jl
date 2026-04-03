# ERF vs MPAS acoustic substepping comparison

## ERF overview

ERF (Energy Research and Forecasting) is a C++/AMReX compressible atmospheric model
following Klemp, Skamarock & Dudhia (2007) in height coordinates. Published in JAMES:
Lattanzi et al. (2025), DOI: 10.1029/2024MS004884.

Like MPAS, ERF uses height coordinates (not WRF's mass coordinate). This makes the
ERF-MPAS comparison most relevant for Breeze.

## Shared algorithm structure

Both ERF and MPAS use:
- Wicker-Skamarock RK3 outer scheme (stages at Δt/3, Δt/2, Δt)
- Forward-backward acoustic substeps within each RK stage
- Vertically implicit tridiagonal for the w-θ-ρ coupling
- Off-centering parameter (β_s = 0.1 in ERF, ε = 0.1 in MPAS)
- 1D hydrostatically-balanced reference state
- Time-averaged velocities for scalar transport
- Forward extrapolation of θ for horizontal PGF damping

## Key differences

### 1. Prognostic variables in the acoustic loop

| | ERF | MPAS |
|---|---|---|
| Horizontal momentum | ρu, ρv (full, conservation form) | ru_p (perturbation from RK predictor) |
| Vertical momentum | ρw (full) → tridiagonal solves for δ(ρw) | rw_p (perturbation) → tridiagonal |
| Thermodynamic | ρθ (full, conservation form) | rtheta_pp (double-perturbation) |
| Density | ρ (full, prognostic) | rho_pp (double-perturbation) |

**ERF works with full conservation-form variables.** The acoustic step computes
perturbations internally (e.g., `new_drho_u = cur_xmom - stage_xmom`) but stores
the full state.

**MPAS works with perturbation variables.** The double-perturbation `_pp` variables
are reset to zero at each RK stage, measuring only the acoustic response within that
stage. The full state is recovered at the end.

**Implication for Breeze:** MPAS's perturbation approach is more elegant and keeps the
acoustic variables small. ERF's approach is simpler to implement but operates on full-
magnitude fields.

### 2. Pressure gradient formulation

**ERF** uses the Exner function formulation (Klemp et al. 2007, Eq. 11-12):
```
horizontal PGF: -γ R_d π^t ∂(Θ''_m)/∂x
vertical PGF:   -γ R_d π^t ∂(ζ_z Θ''_m)/∂z + g ρ̃_d (π'/π̄) - g ρ̃'_m
```

**MPAS** uses an equivalent formulation (tech note Eq. 3.34):
```
horizontal PGF: -(ρ^t_d/ρ^t_m) γ R_d π^t ∇(Θ''^{τ̄}_m)
vertical PGF:   cofwz · ∂(Θ''_m)/∂z - cofwt · Θ''_m + cofwr · ρ̃''_d
```

Both are mathematically equivalent: the pressure gradient `∇p = γ R_d π ∇(Θ_m)`.
The difference is organizational — ERF writes it explicitly while MPAS absorbs the
coefficients into named variables (cofwz, cofwt, cofwr).

### 3. Tridiagonal system

**ERF** (from `ERF_MakeFastCoeffs.cpp`):
```
coeffA(k) = D (-0.5g + coeff_Q · θ_face_below)
coeffB(k) = 1 + D (coeff_Q - coeff_P) · θ_face_mid
coeffC(k) = D ( 0.5g - coeff_P · θ_face_above)
```
where `D = Δτ² β₂² / Δz` and:
```
coeff_P = (-γ R_d / Δz + 0.5g R_d ρ̄/(c_v π̄ Θ^t_m)) π_face
coeff_Q = ( γ R_d / Δz + 0.5g R_d ρ̄/(c_v π̄ Θ^t_m)) π_face
```

These combine the acoustic PGF (first term in P/Q) with the EOS buoyancy correction
(second term) — analogous to MPAS's cofwz + cofwt.

**MPAS** (tech note Appendix A):
```
C_- = -C_Ωz C_Θ/Δz + C_Ωρ C_ρ - C_Ωθ C_Θ/Δz
C₀  = 1 + C_Ωz (C_Θ/Δz + C_Θ/Δz) - C_Θ (C_Ωθ/Δz - C_Ωθ/Δz) + C_Ωρ (C_ρ - C_ρ)
C_+ = -C_Ωz C_Θ/Δz - C_Ωρ C_ρ + C_Ωθ C_Θ/Δz
```

Three distinct couplings:
- `C_Ωz · C_Θ`: acoustic PGF from θ flux (same physics as ERF's first P/Q term)
- `C_Ωρ · C_ρ`: gravity-density coupling (similar to ERF's `0.5g` terms)
- `C_Ωθ · C_Θ`: EOS buoyancy correction (similar to ERF's second P/Q term)

**The tridiagonal systems are physically equivalent.** ERF combines the couplings into
two composite coefficients (P, Q), while MPAS keeps them as three separate terms.

### 4. Divergence damping

**ERF: Forward extrapolation only** (β_d = 0.1)
```
θ_extrap = Θ''_prev + β_d (Θ''_prev - Θ''_lagged)
```
Applied to the horizontal PGF's θ argument. No separate velocity filter.
ERF does NOT implement the Klemp et al. (2018) 3D divergence damping.

**MPAS: Klemp et al. (2018) 3D filter** (γ_D = 0.1)
```
V''_h += (γ_D Δx / Θ^t_m) δ_τ Θ''_m
```
Applied as a velocity correction after each substep. Uses the time tendency of ρθ
as a 3D divergence proxy, ensuring consistency with the pressure equation.

**MPAS also uses** forward extrapolation in the horizontal PGF (same as ERF), PLUS
the 3D velocity filter. The combination damps all acoustic modes (horizontal and
vertical) uniformly.

**For Breeze:** Start with forward extrapolation (already implemented). Add the
MPAS-style 3D filter later for robustness on non-uniform grids.

### 5. Density update strategy

**ERF:** Full density ρ is prognostic and updated every substep:
```
ρ^{n+1} = ρ^n + Δτ (slow_rhs_ρ - div_h(pert_momentum) - div_z(off-centered ρw))
```
Uses full 3D divergence of PERTURBATION momenta (new - stage values).

**MPAS:** Double-perturbation ρ'' is updated using the new rw_p:
```
ρ''(k) = rs(k) - cofrz(k) · (rw_p(k+1) - rw_p(k))
```
where `rs` includes horizontal flux divergence + slow tendency + old vertical divergence.

**Key insight:** Both models update density using the FULL 3D divergence in the acoustic
step. There is no separate horizontal/vertical split with different weights (unlike the
IMEX-ARK approach that caused instability in our HEVI implementation).

### 6. Source term re-evaluation

**ERF:** Calls `make_sources` and `make_mom_sources` every acoustic substep, but with
`is_slow_step=false` only a SUBSET is recomputed: Rayleigh damping and immersed boundary
forcing (if enabled). The main physics (subsidence, nudging, radiation) are only computed
on the slow step. Momentum sources (Coriolis, geostrophic wind) are also re-evaluated
per substep since they depend on the current velocity.

**MPAS:** Slow tendencies are computed ONCE per RK stage and frozen during all substeps.
No source re-evaluation within the acoustic loop.

### 7. Code structure

**ERF** (C++/AMReX):
- `ERF_Substep_NS.cpp`: single-file acoustic substep (~600 lines)
- `ERF_MakeFastCoeffs.cpp`: tridiagonal coefficients
- `ERF_MRI.H`: multirate integrator (RK3 outer loop)
- GPU via AMReX abstraction (CUDA, HIP, SYCL)

**MPAS** (Fortran):
- `mpas_atm_time_integration.F`: monolithic (~7000 lines)
- Subroutines: `atm_advance_acoustic_step_work`, `atm_compute_vert_imp_coefs_work`
- MPI parallelism only (no GPU)

## Summary: ERF vs MPAS for Breeze implementation

| Aspect | ERF approach | MPAS approach | Recommendation for Breeze |
|--------|-------------|---------------|--------------------------|
| Variable form | Full conservation | Double-perturbation | **MPAS**: smaller variables, better numerics |
| PGF formulation | Direct from code | Coefficient-based | Either works; MPAS's named coefficients are clearer |
| Tridiagonal | Combined P/Q coeffs | Three separate couplings | **MPAS**: clearer physics, matches tech note |
| Divergence damping | Forward extrap only | Forward extrap + 3D filter | **MPAS**: more complete damping |
| Density | Full prognostic | Perturbation | **MPAS**: consistent with perturbation framework |
| Source re-eval | Subset per substep | Once per RK stage | **MPAS**: simpler, standard practice |
| Code complexity | Simpler (one approach) | More complex (many options) | ERF is easier to read; MPAS is more complete |

**Bottom line:** ERF and MPAS implement the same algorithm (Klemp et al. 2007) with
minor variations. ERF is simpler and well-documented but lacks the 3D divergence
damping and double-perturbation framework. MPAS is more mature and has the complete
feature set. For Breeze, follow MPAS's double-perturbation approach with the Klemp
et al. (2018) 3D divergence filter.
