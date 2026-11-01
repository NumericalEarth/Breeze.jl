# moist-bubble: Bryan & Fritsch (2002) moist benchmark, θ / E / η compressible and θ anelastic

## Verdict

- **Why it was inconclusive (root cause found, fixed).** Four setup defects, not formulation physics:
  1. `set!(T, qᵗ)` under saturation adjustment converted T as if the air were dry. #1173 fixes θ and s. **E (`TotalEnergyFormulation`) had the same bug and #1173 does not cover it.** I fixed it on this branch with the same hook, plus a test.
  2. The non-equilibrium runs settled cloud droplets through the open bottom, about 3.6e5 J/m² in 1000 s. Fixed in the driver by setting the Stokes fall speed to zero.
  3. The environment was built from Breeze's η instead of BF02's θₑ = 320 K. The anelastic reference was a dry column 2 % off in p.
  4. There was no CM1 reference to compare against.
  After the fixes all 8 null runs stay at rest: max|w| ≤ 3e-5 m/s, against up to 10 m/s before.
- **Pass/fail at Δ = 100 m, against criteria committed before any result** (`validation/moist_bubble/CRITERIA.md`, commit a8c26d90):

| formulation (core, microphysics) | result |
|---|---|
| θ, anelastic, saturation adjustment | **PASS** (all of A1–A7, B1–B8, C) |
| θ, compressible, saturation adjustment | **FAIL B5 only**: θₑ′ structure L2 vs CM1 = 0.279 (tolerance 0.25) |
| E (T₀ = 273.16 and 0), compressible, saturation adjustment | **FAIL B5 only**: L2 = 0.275 |
| η, compressible, non-equilibrium τ = 2 s (η rejects saturation adjustment) | **PASS** at 100 m, but see the robustness caveat below |
| θ, E, anelastic θ with the same non-equilibrium scheme | PASS at 100 m |

- **The B5 failure is not formulation-specific.** It belongs to the compressible core (plus moisture), not to θ or E. I did not find its mechanism and did not fix it.

## Evidence (100 m, t = 1000 s; CM1 r21.1 eqtset = 2 at 100 m: w max 15.75, θₑ′ [−0.103, 3.879] K, top 8418 m)

- **Initial state.** The environment's θₑ = 320 K to 1e-6 K. T after `set!` is within 7e-5 K of what was set (A2), and p within 2e-7 (A4). The environment T and qᵛ are within 0.013 K and 1.1e-5 of CM1's.
- **Scalar metrics.**
  - Every Breeze run: w max 14.5–15.3 (−3 to −8 % vs CM1), top within 85 m, θₑ′max 3.38–3.68 (−5 to −13 %), θₑ′min ≥ −0.07.
  - Every run is symmetric about x = 0 to round-off.
  - The paper reports w = 15.71 / −9.93 m/s.
- **Structure L2 of θₑ′ vs CM1.**
  - Anelastic with saturation adjustment: 0.12. Compressible with saturation adjustment: 0.275–0.31 (θ, E0, E273). Compressible non-equilibrium: 0.15–0.17.
  - For scale, CM1 against itself: 50 m vs 100 m, 0.156; a different vertical WENO setting, 0.053; Δt halved, 0.024.
  - The error is in the bubble cap (z = 7–9 km holds about 70 % of it). Breeze's cap is thinner, with lower θₑ′max, and sits 50–85 m higher.
- **What the compressible gap is not.** It does not change with:
  - Δt (1, 0.5, 0.25 s);
  - the acoustic CFL (0.25; L2 0.29);
  - τ.
  - Non-equilibrium E and η converge to the saturation-adjustment E solution as τ → 0: L2 to E-SA is 0.37 / 0.19 / 0.10 / 0.05 at τ = 2 / 1 / 0.5 / 0.25 s. At τ = 0.25 their L2 vs CM1 is 0.234 (E) and 0.244 (η), just under tolerance.
  - So the non-equilibrium "pass" at τ = 2 is helped by the condensation lag.
  - At 50 m the compressible-SA gap stays: 0.34–0.36, against 0.19 for anelastic.
  - Hypothesis, untested: an isochoric (fixed-ρ) saturation adjustment on the compressible core, or condensation frozen during the acoustic substeps (CM1 eqtset = 2 applies it on the small steps). Either would change the latent heating in the cap.
- **Energy.**
  - E: |Δℰ|/max KE ≤ 4e-10 in every run (structural).
  - θ compressible: −1.5e-2. η: −5e-3. Anelastic θ: −2.8e-2. All pass the 5 % tolerance.
  - Total water: ≤ 1e-13.
  - CM1's own diagnosed energy rises by 0.29 × max KE at 100 m (0.55 at 50 m). Its far field warms about 3e-3 K and its null run moves (w 7e-4 m/s). So CM1 is a reference for dynamics, not for energy.
- **Convergence** (θₑ′max, Δ = 250 / 200 / 100 / 50 m):

| run | 250 m | 200 m | 100 m | 50 m |
|---|---|---|---|---|
| CM1 | 3.29 | 3.40 | 3.88 | 3.82 |
| A-θ SA | 2.42 | 2.68 | 3.38 | 3.78 |
| C-E273 SA | 2.69 | 2.95 | 3.46 | 3.78 |
| C-θ SA | 2.14 | 2.46 | 3.52 | 4.04 |

  - Breeze converges toward CM1 more slowly in θₑ′max. That fails B2 at 200–250 m, where B2 was graded only as a convergence report.
  - w max is within 3 % of CM1 at every Δ.
  - The full table is in `validation/moist_bubble/RESULTS_TABLE.md`.

## Problems found, out of scope, not fixed

1. **Compressible non-equilibrium runs are not robust.**
   - η at 50 m: from about 800 s, a cold supersaturated patch at the lid (RH 1.57, θₑ′ −6.2 K).
   - E273 at 250 m: from about 200 s, a cold patch near the surface (RH 1.14, θₑ′ −3.1 K).
   - θ at 50 m: blew up at 750 s.
   - Anelastic and compressible saturation-adjustment runs never did this.
   - It matches the earlier C-E0 200 m anomaly. It matters for RCE, which uses non-equilibrium microphysics.
2. Breeze with thermal damping 0 or forward weight 0.55 blows up within 100–560 s. CM1 with kdiv = 0 or alph = 0.5 degrades badly as well. These probes were uninformative.
3. My B7 "θₑ conservation" metric (Δ∫ρᵈθₑ′) is ill-posed: CM1 itself shows +3 %. Breeze runs show −0.4 to +1.1 %.

## Branch, commits, tests

- Branch `glw/fix-moist-bubble`, pushed to origin.
- Commits:
  - f17b2cee: #1173 cherry-picked.
  - a8c26d90: E `set!` fix and test, and the pre-registered criteria.
  - Up to b3418307 plus the report commit: the driver and analysis (`validation/moist_bubble/{bubble.jl,analyze.py,run.sbatch}`).
- Tests (job 7003, GPU, commit b3418307): `set_temperature_saturation_adjustment` (new E cases, T₀ = 0 and 273.16, warm and mixed phase, Float32 and Float64, plus an E rest state), `total_energy_formulation`, `set_atmosphere_model`, `compressible_saturation_adjustment` and `quality_assurance`. Result: 830 pass, 1 broken (pre-existing), 0 fail.
- Runs: CM1 jobs 6597, 6903 and 6991 (`~/rce-energy/moist-bubble-work/cm1/`). Breeze jobs 6806, 6904, 6905 and 6992.

## Open

- The mechanism of the compressible-core cap discrepancy.
- The compressible non-equilibrium cold-patch instability.
- E's `set!(θ)` path and η's `set!(T)` path still use `grid_moisture_fractions`. They are correct for non-equilibrium schemes; η cannot use saturation adjustment anyway.
