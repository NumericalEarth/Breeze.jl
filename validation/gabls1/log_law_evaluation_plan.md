# Plan: Evaluating the log-law mismatch in GABLS1

## Goal

Quantify the near-surface wind speed overshoot in the GABLS1 stable boundary layer case
and demonstrate that it is an inherent LES problem (not a Breeze bug) by comparing against
intercomparison data and MOST predictions.

## Background

Standard LES overpredicts mean wind shear near the surface because SGS models are too
dissipative there — resolved fluctuations are suppressed and the mean flow carries too much
momentum flux. This "log-law mismatch" is well-documented (Mason & Thomson 1992, Sullivan
et al. 1994, Brasseur & Wei 2010) and is worst in stable conditions where turbulent length
scales are small relative to the grid.

## Evaluation steps

### 1. Run the baseline GABLS1 case (ILES)

Run `gabls1.jl` as-is (no explicit SGS closure). This is the control case that should
exhibit the overshoot.

### 2. Diagnose the mismatch

From the hour 8-9 averaged profiles, compute:

- **Non-dimensional wind shear** φ_m(z) = (κz / u★) dU/dz, where u★ is diagnosed from
  the surface momentum flux: u★ = sqrt(sqrt(⟨u'w'⟩² + ⟨v'w'⟩²)) at z = 0.
  MOST predicts φ_m = 1 in neutral conditions and φ_m = 1 + β(z/L) in stable conditions
  (Beljaars & Holtslag 1991). The overshoot shows up as φ_m >> 1 at the first few grid points.

- **Wind speed ratio** U_LES(z₁) / U_MOST(z₁), where U_MOST is the log-law prediction
  at the first grid point. Values > 1 indicate overshoot.

- **Obukhov length** L = -u★³ θ̄ / (κ g ⟨w'θ'⟩₀) from surface fluxes.

### 3. Compare against intercomparison data

Overlay the Breeze φ_m profile against the envelope from the 8 intercomparison groups.
The key plot is φ_m vs z/L (or just vs z) in the lowest 50 m.

### 4. Run with SmagorinskyLilly closure

Re-run GABLS1 with `closure = SmagorinskyLilly()` to show that the explicit SGS model
makes the overshoot worse (Smagorinsky is known to be overly dissipative near the surface).

### 5. Resolution sensitivity (optional)

Run at 64³ (6.25 m) and 256³ (1.5625 m) to confirm the mismatch persists or worsens
with resolution, consistent with Bou-Zeid et al. (2025).

## Expected outcome

- The ILES and Smagorinsky runs both overshoot the log law near the surface.
- Breeze profiles fall within the intercomparison spread (the other groups have
  the same problem).
- This motivates implementing Sullivan et al. (1994) `SurfaceLayerDiffusivity`
  and eventually the Kosovic (1997) NBA closure in Breeze.

## Key plots

1. Wind speed vs z (linear) — Breeze vs intercomparison envelope
2. Wind speed vs z (log scale) — shows deviation from log-law slope
3. φ_m(z) vs z — non-dimensional shear showing overshoot at first grid points
4. φ_m(z) for ILES vs Smagorinsky — demonstrating closure dependence
