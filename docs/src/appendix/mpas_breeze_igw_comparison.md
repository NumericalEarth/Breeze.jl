# MPAS vs Breeze IGW Acoustic Substepping Comparison

## Purpose

Compare MPAS and Breeze substep-by-substep on the Skamarock & Klemp (1994)
inertia-gravity wave (IGW) test case. This is a 2D (x, z) case with nonzero
background wind and a horizontally varying θ perturbation — the simplest case
that exercises the horizontal pressure gradient in the acoustic substep loop.

## Setup

### Physical parameters
- Domain: 300 km × 10 km, periodic in x, bounded in z
- Δx = 1000 m, Δz = 1000 m, 10 vertical levels
- Background: θ₀ = 300 K, N² = 10⁻⁴ s⁻², u = 20 m/s, w = 0
- Perturbation: Δθ = 0.01 K × sin(πz/H) / (1 + (x − 100 km)²/a²), a = 5 km
- θ(z) = θ₀ exp(N²z/g), ρ from hydrostatic balance, p₀ = 100000 Pa

### Constants (MPAS defaults)
- Rᵈ = 287.0 J/(kg·K), cₚ = 1003.0 J/(kg·K), g = 9.80616 m/s²
- κ = Rᵈ/cₚ = 0.28614158, c² = cₚ Rᵈ/cᵥ = 402.04, R/cᵥ = 0.40084

### Time stepping
- Δt = 40 s, N = 24 substeps → Δτ = Δt/N = 1.667 s
- epssm = 0.1, smdiv = 0.1, len_disp = 1000 m
- WS-RK3 stages: stage 1 = 1 substep (Δτ = Δt/3 = 13.33 s), stage 2 = 12 substeps, stage 3 = 24 substeps
- dtseps = 0.5 × Δτ × (1 + epssm) = 0.9167
- resm = (1 − epssm)/(1 + epssm) = 0.8182
- cofwr = dtseps × g/2 = 4.4945

### MPAS stability limit
Δt_max ≈ 40 s (with N chosen to satisfy acoustic CFL). At Δt = 50 s the run blows up (advective CFL limit Δx/U = 50 s). N must be ≥ 2.

### MPAS binary
`/tmp/mpas_igw/atmosphere_model` (includes debug prints at line 2976 of `mpas_atm_time_integration.F`).

### Files
- `/tmp/mpas_igw/` — MPAS run directory
- `/tmp/mpas_igw/init.nc` — initial conditions (IGW profile)
- `/tmp/mpas_igw/grid_igw.nc` — 300×4 planar hex mesh (dc = 1000 m)
- `/tmp/mpas_igw/output.nc` — output after 1 step (dt = 40 s)
- `/tmp/mpas_igw/mpas_debug_igw.txt` — substep debug output
- `/tmp/mpas_igw/set_igw_ics.py` — script to overwrite init.nc with IGW ICs
- `/tmp/mpas_igw/mpas_igw_step1.npz` — extracted cross-section data

## MPAS reference data (cell 0, x = 0.5 km)

### Initial state

Cell 0 is far from the perturbation center (x₀ = 100 km), so θ ≈ θ_base, ρ ≈ ρ_base.

| k | z (m) | θ_base (K) | ρ_base (kg/m³) |
|---|-------|------------|----------------|
| 1 |  500  | 301.5336   | 1.109363       |
| 2 | 1500  | 304.6242   | 1.010465       |
| 3 | 2500  | 307.7466   | 0.918563       |
| 4 | 3500  | 310.9009   | 0.833268       |
| 5 | 4500  | 314.0876   | 0.754211       |
| 6 | 5500  | 317.3070   | 0.681037       |
| 7 | 6500  | 320.5593   | 0.613412       |
| 8 | 7500  | 323.8450   | 0.551014       |
| 9 | 8500  | 327.1644   | 0.493537       |
| 10| 9500  | 330.5178   | 0.440690       |

### Implicit coefficients (constant across all stages)

| k | cofwz | cofwr |
|---|-------|-------|
| 2 | 0.35643 | 4.49449 |
| 3 | 0.34463 | 4.49449 |
| 4 | 0.33295 | 4.49449 |
| 5 | 0.32139 | 4.49449 |
| 6 | 0.30995 | 4.49449 |
| 7 | 0.29862 | 4.49449 |
| 8 | 0.28740 | 4.49449 |
| 9 | 0.27630 | 4.49449 |
| 10| 0.26531 | 4.49449 |

### Stage 1, substep 1 (Δτ = Δt/3 = 13.33 s, single substep)

Cell 0 is nearly unperturbed, so tend_rw = 0 at boundaries (k = 2, 10)
and O(10⁻⁵) elsewhere. Column headers below use Breeze names with the
corresponding MPAS names in parentheses.

| k | ρw″ (`rw_p`) | ρθ″ (`rtheta_pp`) | ρ″ (`rho_pp`) | tend_rw |
|---|---|---|---|---|
| 2 | -7.679e-07 | 2.306e-06 | 7.524e-09 | 0.0 |
| 3 | -8.975e-06 | -2.130e-05 | -6.659e-08 | -1.036e-05 |
| 4 | 8.837e-07 | -1.660e-05 | -5.099e-08 | 1.284e-06 |
| 5 | 1.512e-06 | -1.457e-05 | -4.423e-08 | 1.150e-06 |
| 6 | 1.731e-06 | -1.391e-05 | -4.188e-08 | 1.027e-06 |
| 7 | 5.611e-06 | 5.991e-06 | 1.928e-08 | 6.898e-06 |
| 8 | -3.332e-06 | -1.225e-05 | -3.630e-08 | -5.170e-06 |
| 9 | 4.936e-06 | 1.374e-06 | 4.225e-09 | 5.170e-06 |
| 10| 3.263e-07 | 9.835e-08 | 2.991e-10 | 0.0 |

### Stage 1, substep 2 (same stage, accumulated perturbations)

| k | ρw″ (`rw_p`) | ρθ″ (`rtheta_pp`) | ρ″ (`rho_pp`) |
|---|---|---|---|
| 2 | -3.076e-06 | 5.768e-06 | 1.879e-08 |
| 3 | -9.222e-06 | -2.355e-05 | -7.131e-08 |
| 4 | -1.307e-07 | -1.626e-05 | -4.747e-08 |
| 5 | 2.699e-06 | -1.432e-05 | -4.125e-08 |
| 6 | 3.712e-06 | -1.446e-05 | -4.165e-08 |
| 7 | 4.391e-06 | 1.200e-05 | 3.862e-08 |
| 8 | 3.067e-07 | -1.557e-05 | -4.499e-08 |
| 9 | 5.393e-06 | 3.382e-06 | 1.041e-08 |
| 10| 1.312e-06 | 5.510e-07 | 1.676e-09 |

### Stage 2, substep 1 (slow tendencies recomputed)

Note: tend_rw is now slightly nonzero at k = 2 (−4.1e-08) — from the
state change during stage 1.

| k | ρw″ (`rw_p`) | ρθ″ (`rtheta_pp`) | ρ″ (`rho_pp`) | tend_rw |
|---|---|---|---|---|
| 2 | -8.492e-07 | 2.332e-06 | 7.593e-09 | -4.101e-08 |
| 3 | -9.290e-06 | -2.133e-05 | -6.647e-08 | -1.056e-05 |
| 4 | 4.377e-07 | -1.666e-05 | -5.098e-08 | 1.012e-06 |
| 5 | 1.088e-06 | -1.463e-05 | -4.426e-08 | 8.947e-07 |
| 6 | 1.389e-06 | -1.396e-05 | -4.191e-08 | 8.226e-07 |
| 7 | 5.411e-06 | 6.107e-06 | 1.910e-08 | 6.808e-06 |
| 8 | -3.397e-06 | -1.226e-05 | -3.619e-08 | -5.231e-06 |
| 9 | 4.870e-06 | 1.417e-06 | 4.345e-09 | 5.139e-06 |
| 10| 3.098e-07 | 1.067e-07 | 3.216e-10 | -1.517e-08 |

### State after 1 full time step (Δt = 40 s)

Cell near perturbation center (cell 399, x = 100 km):

| k | z (m) | w (m/s) | Δθ (K) | Δρ (kg/m³) |
|---|-------|---------|--------|------------|
| 1 |  500  | -6.64e-04 | -5.0e-05 | -6.4e-06 |
| 2 | 1500  | -1.03e-03 | -1.7e-04 | -1.6e-05 |
| 3 | 2500  | -8.49e-04 | -2.8e-04 | -2.2e-05 |
| 4 | 3500  | -4.80e-04 | -3.9e-04 | -2.4e-05 |
| 5 | 4500  |  4.67e-04 | -4.9e-04 | -2.2e-05 |
| 6 | 5500  |  1.68e-03 | -5.1e-04 | -1.8e-05 |
| 7 | 6500  |  2.58e-03 | -5.2e-04 | -1.4e-05 |
| 8 | 7500  |  3.41e-03 | -4.5e-04 | -7.1e-06 |
| 9 | 8500  |  2.92e-03 | -3.3e-04 |  9.8e-07 |
| 10| 9500  |  0.00e+00 | -1.2e-04 |  7.6e-06 |

### t = 3000 s results

| Model | max\|w\| (m/s) | max\|θ′\| (K) |
|-------|----------------|---------------|
| MPAS (Δt = 40 s, N = 24) | 0.002591 | 2.77e-03 |
| Breeze anelastic (Δt = 40 s) | 0.002578 | 2.84e-03 |

## Breeze comparison target

Reproduce the stage 1, substep 1 values above (`ρw″`/`rw_p`, `ρθ″`/`rtheta_pp`,
`ρ″`/`rho_pp`, `cofwz`, `cofwr`, `tend_rw`) at cell (1, 1) of a matching
RectilinearGrid. See `mpas_breeze_naming.md` for the full Breeze ↔ MPAS
variable mapping.

The grid differences (hex vs rectilinear, 1200 cells vs 300 cells) mean
exact agreement is not expected for horizontal operators, but the vertical
column dynamics at cell 0 (far from the perturbation) should match closely
since the horizontal fluxes are nearly zero there.

## Known issues in Breeze acoustic substepper

1. **Missing horizontal PGF in slow tendency**: `SlowTendencyMode` zeros the
   horizontal pressure gradient. MPAS includes it in `tend_u_euler`. The
   acoustic substep only provides the ρθ″ correction, so
   c² Π ∂(ρθ★)/∂x is never applied.

2. **Halo bug** (fixed): `rtheta_pp` halos were stale when divergence damping
   read them, creating spurious gradients at periodic boundaries.

3. **forward_weight default**: Code has 0.6, docstring says 0.55. Gives
   epssm = 0.2 instead of MPAS's 0.1.

4. **Missing `_ssp_rk3_substep!`**: The `AcousticSSPRungeKutta3` time stepper
   is broken because the scalar substep kernel was never defined.
