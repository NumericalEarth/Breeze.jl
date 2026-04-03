# MPAS-A acoustic solver equations

Source: Skamarock, Duda, Klemp, Fowler (2025), "A Description of MPAS-A Version 8"
(NCAR Tech Note, draft). https://www2.mmm.ucar.edu/projects/mpas/mpas_website_linked_files/MPAS-A_tech_note.pdf

Equations numbered as in the tech note. Simplified here for no terrain (ζ_z = 1, z_H = 0).

## Prognostic variables (Eqs. 3.1-3.5)

```
ρ̃_d = ρ_d              (dry air density, no terrain Jacobian)
V_H  = ρ̃_d v_H         (horizontal momentum density)
W    = ρ̃_d w            (vertical momentum density)
Θ_m  = ρ̃_d θ_m         (moist potential temperature density, θ_m = θ(1 + R_v/R_d q_v))
Q_j  = ρ̃_d q_j         (moisture densities)
```

EOS (Eq. 3.11): `p = p₀ (R_d Θ_m / p₀)^(c_p/c_v)`

## Base state (Section 8.1.1)

1D isothermal reference at T₀ = 250K:
```
p̄ = p₀ exp(-gz/(R_d T₀))
ρ̄_d = p̄/(R_d T₀)
θ̄ = T₀ (p̄/p₀)^(-R_d/c_p)
```

Perturbation: `ρ_d = ρ̄_d + ρ'_d`, `Θ_m = ρ̄_d θ̄ + Θ'_m`, `p = p̄ + p'`

## Double-prime acoustic perturbations (Eq. 3.23)

Reset each RK stage:
```
V''_h = V_h - V^t_h          (horizontal momentum pert from RK predictor)
Ω''  = Ω - Ω^t              (vertical mass flux pert, Ω = ρ̃_d w for no terrain)
Θ''_m = Θ'_m - Θ'^t_m        (double-pert potential temperature density)
ρ̃''_d = ρ̃'_d - ρ̃'^t_d       (double-pert density)
```

## Acoustic step equations (Eqs. 3.25-3.28, no terrain)

Off-centered time average (Eq. 3.29): `φ̄^τ = ½(1+ε)φ^{τ+Δτ} + ½(1-ε)φ^τ`

**Step 1 — Horizontal momentum (forward, Eq. 3.25):**
```
δ_τ V''_h = -(ρ^t_d/ρ^t_m) γR_d π^t ∇(Θ''^{τ̄}_m) + R^t_{V_h}
```

**Step 2 — Vertical mass flux (implicit, Eq. 3.26):**
```
δ_τ Ω'' + ½(1+ε) {
    (ρ^t_d/ρ^t_m) [γR_d π^t ∂(Θ''^{τ+Δτ}_m)/∂z
                   - g ρ̃^t_m (R_d/c_v) Θ''^{τ+Δτ}_m / Θ^t_m]
    + g ρ̃''^{τ+Δτ}_d
} = R^t_Ω
```

**Step 3 — Potential temperature density (backward, Eq. 3.27):**
```
δ_τ Θ''_m + ∇·(V''^{τ+Δτ}_h θ^t_m) + ∂(Ω''^{τ̄} θ^t_m)/∂z = R^t_{Θ_m}
```

**Step 4 — Density (backward, Eq. 3.28):**
```
δ_τ ρ̃''_d + ∇·V''^{τ+Δτ}_h + ∂Ω''^{τ̄}/∂z = R^t_{ρ_d}
```

## Rearranged for the tridiagonal (Eqs. 3.37-3.42, no terrain)

Isolate new-time unknowns. Define α = ½Δτ(1+ε):

**Θ update** (Eq. 3.38):
```
Θ''^{τ+Δτ}_m(k) = -α/Δz_w(k) [θ^t_m(k+½) Ω''(k+1) - θ^t_m(k-½) Ω''(k)] + Δτ F_{Θ}(k)
```

**ρ update** (Eq. 3.39):
```
ρ̃''^{τ+Δτ}_d(k) = -α/Δz_w(k) [Ω''(k+1) - Ω''(k)] + Δτ F_{ρ}(k)
```

**Ω equation** (Eq. 3.37, substitute Θ and ρ updates):
Tridiagonal in Ω''(k) at interfaces.

## Appendix A coefficients (Eqs. A.4-A.8, no terrain)

```
C_{Ωz}(k)  = α (c_p/c_v)(ρ^t_d/ρ^t_m)^{k-face} π^{t,k-face} / Δz(k)     [interface]  (A.4)
C_{Ωθ}(k)  = α (R_d/c_v)(g/2) ρ̃^t_d(k) / Θ^t_m(k)                        [level]      (A.5)
C_{Ωρ}(k)  = α g/2                                                          [interface]  (A.6)
C_Θ(k)     = α θ^{t,k-face}_m                                               [interface]  (A.7)
C_ρ(k)     = α / Δz_w(k)                                                    [level]      (A.8)
```

where `α = ½Δτ(1+ε)`, `k-face` denotes interpolation to the interface from
bounding levels, `Δz(k)` is the spacing between level centers k-1 and k,
and `Δz_w(k)` is the spacing between interfaces k and k+1 (= layer thickness).

## Tridiagonal coefficients (Eq. A.10)

```
C_- Ω''(k-1) + C₀ Ω''(k) + C_+ Ω''(k+1) = R(k)
```

**Sub-diagonal** (coupling to k-1):
```
C_- = -C_{Ωz}(k) C_Θ(k-1) / Δz_w(k-1)
      + C_{Ωρ}(k) C_ρ(k-1)
      - C_{Ωθ}(k-1) C_Θ(k-1) / Δz_w(k-1)
```

**Diagonal**:
```
C₀ = 1 + C_{Ωz}(k) [C_Θ(k)/Δz_w(k) + C_Θ(k)/Δz_w(k-1)]
       - C_Θ(k) [C_{Ωθ}(k)/Δz_w(k) - C_{Ωθ}(k-1)/Δz_w(k-1)]
       + C_{Ωρ}(k) [C_ρ(k) - C_ρ(k-1)]
```

**Super-diagonal** (coupling to k+1):
```
C_+ = -C_{Ωz}(k) C_Θ(k+1) / Δz_w(k)
      - C_{Ωρ}(k) C_ρ(k)
      + C_{Ωθ}(k) C_Θ(k+1) / Δz_w(k)
```

**Physical interpretation of each coupling:**
- `C_{Ωz} C_Θ` terms: acoustic coupling (vertical PGF from θ flux → Ω)
- `C_{Ωρ} C_ρ` terms: gravity-density coupling (vertical ρ divergence → buoyancy → Ω)
- `C_{Ωθ} C_Θ` terms: EOS buoyancy correction (θ perturbation → pressure via EOS → Ω)

**RHS** (Eq. A.9):
```
R(k) = Δτ [F_Ω(k)
          - (C_{Ωz}(k) - C_{Ωθ}(k)) F_Θ(k)
          + (C_{Ωz}(k) + C_{Ωθ}(k-1)) F_Θ(k-1)
          - C_{Ωρ}(k) (F_ρ(k) + F_ρ(k-1))]
```

where F_Ω, F_Θ, F_ρ contain the old-time terms and slow tendencies
(Eqs. 3.40-3.42).

## Divergence damping (Eq. 3.45)

Applied after each acoustic substep to V''_h:
```
V''^{τ+Δτ}_h += (γ_D Δx / Θ^t_m) δ_τ Θ''_m
```

where `δ_τ Θ''_m = (Θ''^{τ+Δτ}_m - Θ''^τ_m)/Δτ` is the acoustic θ tendency,
which is proportional to `-∇·(ρθv)` (the 3D divergence). Default `γ_D = 0.1`.

From Klemp, Skamarock & Ha (2018): using `δ_τ Θ''_m` as the divergence ensures
numerical consistency with the discrete pressure equation, avoiding corruption
of gravity wave frequencies.

## Gravity-wave absorbing layer (Eq. 3.44)

Implicit Rayleigh damping after tridiagonal solve:
```
Ω''^{τ+Δτ} = (Ω''^{τ*} - Δτ R_Ω Ω^t) / (1 + Δτ R_Ω)
```

where `R_Ω = ν sin²[π/2 (z-z_d)/(z_t-z_d)]` for `z ≥ z_d`, zero below.

## Correspondence to Breeze source code

| Tech note symbol | MPAS Fortran variable | Breeze equivalent |
|------------------|-----------------------|-------------------|
| C_{Ωz} | `cofwz(k,iCell)` | acoustic PGF coeff |
| C_{Ωθ} | `cofwt(k,iCell)` | EOS buoyancy correction |
| C_{Ωρ} | `cofwr(k,iCell)` | gravity-density coupling |
| C_Θ | `coftz(k,iCell)` | θ flux from Ω |
| C_ρ | `cofrz(k)` | ρ divergence from Ω |
| ε | `epssm` | off-centering (default 0.1) |
| α = ½Δτ(1+ε) | `dtseps` | implicit time weight |
| (1-ε)/(1+ε) | `resm` | old-time weight |
| γR_d = c_p²/c_v | `c2` | `cₚᵈ * Rᵈ / cᵥᵈ` |
| Ω'' | `rw_p` | vertical mass flux perturbation |
| Θ''_m | `rtheta_pp` | double-pert θ density |
| ρ̃''_d | `rho_pp` | double-pert density |
| V''_h | `ru_p` | horizontal momentum perturbation |
