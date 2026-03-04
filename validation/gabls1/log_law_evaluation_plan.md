# Plan: Evaluating the log-law mismatch

## Goal

Quantify the near-surface wind speed overshoot in LES as a function of horizontal
resolution (grid aspect ratio) and demonstrate that it is an inherent problem by
comparing against reference data and MOST predictions.

## Background

Standard LES overpredicts mean wind shear near the surface because SGS models are too
dissipative there — resolved fluctuations are suppressed and the mean flow carries too much
momentum flux. This "log-law mismatch" is well-documented (Mason & Thomson 1992, Sullivan
et al. 1994, Brasseur & Wei 2010) and is worst in stable conditions where turbulent length
scales are small relative to the grid.

Brasseur & Wei (2010) showed that the overshoot depends critically on the grid aspect
ratio Δx/Δz — the ratio of horizontal to vertical grid spacing controls how well the
energy-containing eddies near the surface are resolved. This motivates a systematic study
varying Δx at fixed Δz.

## Numerics

All simulations use `WENO(order=9, minimum_upwind_bias_order=1)` for advection. No
explicit SGS closure (ILES). The high-order WENO with minimum upwind bias provides low
numerical dissipation away from sharp gradients while maintaining stability.

## Horizontal resolution study

Fix the vertical resolution Δz and domain height. Vary the horizontal grid spacing
Δx = Δy systematically to study how the grid aspect ratio Δx/Δz affects the log-law
mismatch.

### Neutral ABL (primary)

Based on the Mirocha et al. (2018) SWiFT benchmark:

| Parameter | Value |
|-----------|-------|
| Domain height | 2000 m |
| Δz | 5 m (Nz = 400) |
| Geostrophic wind | Ug = 6.5 m/s, Vg = 0 |
| Coriolis | latitude 33.5° (f ≈ 8.05e-5 s⁻¹) |
| Surface roughness z₀ | 0.05 m |
| Initial θ | 300 K uniform below 500 m, +0.01 K/m inversion above |
| Surface BC | no heat flux (neutral) |
| Duration | ~15 hours (analyze 2-hour window around first wind maximum at 80 m) |
| Perturbations | ±0.25 K below 500 m |
| Advection | WENO(order=9, minimum_upwind_bias_order=1) |

Horizontal resolution sweep (Lx = Ly = 2400 m):

| Run | Δx = Δy (m) | Nx = Ny | Δx/Δz |
|-----|-------------|---------|-------|
| N1  | 30          | 80      | 6     |
| N2  | 15          | 160     | 3     |
| N3  | 10          | 240     | 2     |
| N4  | 7.5         | 320     | 1.5   |
| N5  | 5           | 480     | 1     |

The neutral case cleanly isolates the log-law mismatch: MOST predicts φ_m = 1, so any
deviation is unambiguously an LES artifact.

### GABLS1 stable ABL (secondary)

Same Δz sweep approach applied to GABLS1 (Δz = 3.125 m, Lx = Ly = 400 m):

| Run | Δx = Δy (m) | Nx = Ny | Δx/Δz |
|-----|-------------|---------|-------|
| G1  | 18.75       | ~21     | 6     |
| G2  | 9.375       | ~43     | 3     |
| G3  | 6.25        | 64      | 2     |
| G4  | 3.125       | 128     | 1     |

GABLS1 is stably stratified, so the expected φ_m includes a stability correction
(Beljaars & Holtslag 1991). Less clean for diagnosing the mismatch, but useful because
the stable case is where the problem is worst and there is intercomparison data from
8 LES groups.

## Diagnostics

From time-averaged profiles:

- **Non-dimensional wind shear** φ_m(z) = (κz / u★) dU/dz.
  Neutral: expect φ_m = 1. Stable: expect φ_m = 1 + β(z/L).

- **Wind speed ratio** U_LES(z₁) / U_MOST(z₁) at the first grid point.

- **Friction velocity** u★ = (⟨u'w'⟩² + ⟨v'w'⟩²)^(1/4) from surface flux.

- **Obukhov length** (GABLS1 only) L = -u★³ θ̄ / (κ g ⟨w'θ'⟩₀).

## Key plots

1. **φ_m(z) vs z for each Δx/Δz** — does the overshoot decrease with finer Δx?
2. **φ_m at first grid point vs Δx/Δz** — summary curve across all runs
3. **Wind speed vs log(z)** — deviation from log-law slope at each resolution
4. **GABLS1 profiles vs intercomparison envelope** — at Δx/Δz = 1 (standard case)

## Expected outcome

- Overshoot decreases as Δx/Δz → 1 (isotropic grid) but does not disappear.
- Consistent with Brasseur & Wei (2010): a "high accuracy zone" exists but the
  first grid point remains contaminated.
- Motivates implementing Sullivan et al. (1994) `SurfaceLayerDiffusivity` and
  eventually the Kosovic (1997) NBA closure.

## References

- Mirocha et al. (2018) Wind Energ. Sci. 3, 589-613
- Lattanzi et al. (2025) JAMES (ERF paper)
- Beare et al. (2006) BLM 128, 117-137 (GABLS1)
- Mason & Thomson (1992) JFM 242
- Sullivan, McWilliams & Moeng (1994) BLM 71
- Brasseur & Wei (2010) Phys. Fluids 22
