# Plan: Evaluating the log-law mismatch

## Goal

Quantify the near-surface wind speed overshoot in LES and demonstrate that it is an
inherent problem (not a Breeze bug) by comparing against reference data and MOST predictions.

## Background

Standard LES overpredicts mean wind shear near the surface because SGS models are too
dissipative there — resolved fluctuations are suppressed and the mean flow carries too much
momentum flux. This "log-law mismatch" is well-documented (Mason & Thomson 1992, Sullivan
et al. 1994, Brasseur & Wei 2010) and is worst in stable conditions where turbulent length
scales are small relative to the grid.

## Two test cases

### Neutral ABL (primary log-law diagnostic)

The neutral case cleanly isolates the log-law mismatch because MOST predicts φ_m = 1
throughout the surface layer — any deviation is unambiguously an LES artifact. No stability
function is needed in the comparison.

Based on the Mirocha et al. (2018) SWiFT benchmark, also used in the ERF JAMES paper
(Lattanzi et al. 2025):

| Parameter | Value |
|-----------|-------|
| Domain | 2400 × 2400 × 2000 m |
| Grid | 160 × 160 × 400 (Δx = Δy = 15 m, Δz = 5 m) |
| Geostrophic wind | Ug = 6.5 m/s, Vg = 0 |
| Coriolis | latitude 33.5° (f ≈ 8.05e-5 s⁻¹) |
| Surface roughness z₀ | 0.05 m |
| Initial θ | 300 K uniform below 500 m, +0.01 K/m inversion above |
| Surface BC | no heat flux (neutral) |
| Duration | ~15 hours (analyze 2-hour window around first wind speed maximum at 80 m) |
| Perturbations | ±0.25 K below 500 m |

Reference data: WRF, SOWFA, HiGrad results from Mirocha et al. (2018).

### GABLS1 stable ABL (secondary)

GABLS1 is stably stratified, so the expected surface layer profile includes a stability
correction: φ_m = 1 + β(z/L) with the Beljaars & Holtslag (1991) stable functions.
This makes the overshoot diagnostic less clean — errors in the diagnosed Obukhov length L
propagate into the "expected" profile. However, GABLS1 is useful because:

- It has well-established intercomparison data from 8 LES groups
- The stable case is where the problem is worst (small turbulent length scales)
- It motivates the need for improved closures in realistic conditions

Parameters: see `gabls1.jl`.

## Evaluation steps

### 1. Run the neutral ABL case (ILES)

Create `neutral_abl.jl` following the Mirocha et al. (2018) setup. No explicit SGS closure.

### 2. Diagnose the log-law mismatch (neutral case)

From the 2-hour averaged profiles around the wind speed maximum:

- **Non-dimensional wind shear** φ_m(z) = (κz / u★) dU/dz.
  MOST predicts φ_m = 1. The overshoot shows up as φ_m >> 1 at the first few grid points.

- **Wind speed ratio** U_LES(z₁) / U_log(z₁), where U_log = (u★/κ) ln(z/z₀).
  Values > 1 indicate overshoot.

- **Friction velocity** u★ = (⟨u'w'⟩² + ⟨v'w'⟩²)^(1/4) from surface momentum flux.

### 3. Run the GABLS1 stable case (ILES)

Run `gabls1.jl` as-is. Compare against intercomparison data.

### 4. Diagnose the mismatch (GABLS1)

Same diagnostics but accounting for stability:

- Compute Obukhov length L = -u★³ θ̄ / (κ g ⟨w'θ'⟩₀)
- Expected φ_m = 1 + β(z/L) with Beljaars & Holtslag (1991) coefficients
- Compare LES φ_m against stable MOST prediction

### 5. Run both cases with SmagorinskyLilly (optional)

Show that the explicit Smagorinsky SGS model makes the overshoot worse.

### 6. Resolution sensitivity (optional)

Run the neutral case at 2× and 0.5× resolution to confirm the mismatch persists.

## Expected outcome

- Both cases show φ_m overshoot near the surface.
- The neutral case demonstrates the problem cleanly (φ_m >> 1 vs expected φ_m = 1).
- Breeze GABLS1 profiles fall within the intercomparison spread.
- This motivates implementing Sullivan et al. (1994) `SurfaceLayerDiffusivity`
  and eventually the Kosovic (1997) NBA closure.

## Key plots

1. **Neutral ABL**: Wind speed vs log(z) — deviation from straight log-law line
2. **Neutral ABL**: φ_m(z) vs z — overshoot at first grid points vs φ_m = 1
3. **GABLS1**: Wind speed and θ profiles vs intercomparison envelope
4. **GABLS1**: φ_m(z) vs stable MOST prediction
5. **Both cases**: φ_m comparison showing neutral is cleaner diagnostic

## References

- Mirocha et al. (2018) Wind Energ. Sci. 3, 589-613
- Lattanzi et al. (2025) JAMES (ERF paper)
- Beare et al. (2006) BLM 128, 117-137 (GABLS1)
- Mason & Thomson (1992) JFM 242
- Sullivan, McWilliams & Moeng (1994) BLM 71
- Brasseur & Wei (2010) Phys. Fluids 22
