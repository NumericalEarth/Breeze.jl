# Recorded campaign: 17 September 2026

**Conclusion: these tests falsify an unconditional energy-conservation claim.**
They support two isolated compressible corrections, but neither a full model
energy law nor a production-ready finite-step fix. Production code is unchanged.

Production base: `dae9e46d543720f4f1c4f3a57e8d7c7817e90f24` (PR #959).
Campaign: `18f0880968ddad57f40b41fbfdabb26657a0f207`.
Oceananigans: **0.111.0**, pinned in the dedicated manifest, not the manuscript's
0.112 baseline. Local Julia 1.12.7; Apple M5 Max; Metal.jl 1.11.0.
Each TOML contains source and script/environment hashes. See the parent
[README](../README.md) for commands, equations, tolerances and case definitions.

## Completed local runs

| Backend | Precision | Isolated cases / harness assertions | Coupled cases / finite-state assertions |
|---|---|---|---|
| CPU | Float64 | 85 / 201 pass | 8 / 242 pass |
| CPU | Float32 | 85 / 201 pass | NOT RUN |
| Metal, Apple M5 Max | Float32 | 85 / 201 pass | 8 / 242 pass |

These assertion counts validate the harness and controls, **not energy
conservation**. All 12 original isothermal composition-contrast cases fail the
physical invariant on each tested backend/precision. Raw data are the adjacent
`cpu-*` and `metal-*` TOML files; temperature fields/tendencies were computed on
the selected backend, with host Float64 diagnostic analysis.

## Isothermal compressible defect: demonstrated

Receiving-cell temperature tendency, two 100 m cells, first-order flux, 1 m/s
fall speed; expected zero (K/s):

| Phase | Original CPU64 | Original Metal32 | Phase-enthalpy-only Metal32 |
|---|---:|---:|---:|
| Liquid | 0.00200308030 | 0.00200313434 | 5.04e-8 |
| Ice | 0.00314619184 | 0.00314619107 | 7.36e-9 |

Changing only the diagnostic donor quantity from `hx-h_mixture` to `hx`
removes this defect. Across all 36 two-cell isothermal/control cases the
phase-only maximum residual is 2.41e-16 K/s (CPU64), 7.09e-8 (CPU32) and
1.01e-7 (Metal32). Equal-composition and no-fall controls pass; donor reversal
also passes the phase-enthalpy isothermal check.

## Nonisothermal compressible reference: isolated correction supported

Closed-bottom, fixed gas partial densities, no phase change or mechanical energy
exchange. The reference uses standard mixture internal energy. For a 275 K
donor entering a 280 K receiver (CPU64, K/s):

| Phase | Original | Phase enthalpy only | Phase enthalpy + beta_cv | Reference |
|---|---:|---:|---:|---:|
| Liquid | -0.000166741108 | -0.002913521763 | -0.002923720156 | -0.002923720156 |
| Ice | +0.002415123405 | -0.001472456059 | -0.001478325689 | -0.001478325689 |

For both 275 and 285 K donors, phase-only/reference ratios are 0.9965118439
(liquid) and 0.9960295419 (ice). With `beta_cv`, CPU64 absolute rate error is
at most 1.23e-16 K/s; CPU32/Metal32 errors are at most 8.15e-8 K/s. Metal32
ratios range from 0.99999975 to 1.00005446. These are **diagnostic substitutions**,
not implemented corrections or full compressible conservation validation.

The closed bottom matters: the receiving tracer tendency equals incoming mass
rate only here. For an open bottom, outgoing mass at receiver temperature must
not be subtracted from the sensible-heating inflow rate. The general reference
uses separate face enthalpy fluxes minus receiver enthalpy times net mass gain.

## Interior, active bounded limiter: version-specific mass defect

The 4×4×32 test verifies the actual bounded dispatch and an interior limited
reconstruction change of 1.44e-5. The interior high-order versus donor-cell
reconstruction difference reaches 4.39e-4. This is not merely a two-cell routing
test. With closed boundaries:

| Scheme | CPU64 column mass rate (kg/m²/s) |
|---|---:|
| Upwind | 5.42e-19 |
| Ordinary WENO3 | -1.63e-18 |
| Bounded WENO3, Oceananigans 0.111 | -1.80578365e-5 |

Metal32 bounded residual is -1.80576717e-5 kg/m²/s. The normalized CPU residual
`sum(r_dot)/sum(abs(r_dot))` is -7.2107e-4. Independently, phase-only local
isothermal consistency is 8.52e-16 K/s (CPU64), despite this mass defect.

This is the **0.111 non-shared-face limiter behavior**, not a general nonlinear
aggregation effect and not evidence against 0.112. Source inspection confirms
0.112/0.113 use all four donor-indexed, prelimited states; those versions were
NOT RUN here. PR `bounded_face_reconstructions` must be migrated with the
upstream API when rebasing: its copied 0.111 formulas are a rebase hazard.

## Finite implicit and coupled observations

The supplementary anelastic two-cell liquid test at fall CFL 4 gives
temperature errors (-0.501229, +0.451009) K on CPU64 and
(-0.501222, +0.451019) K on Metal32 (open bottom). The independent finite
reference uses the PR's **dry-air replacement convention**, not a unique
physical anelastic energy law. Static-energy controls agree within 1.14e-13 K
(CPU64), 3.24e-5 K (Float32). The Float32 floor is
`eps(Float32)*280 = 3.34e-5 K`; those residuals are numerical noise.

The open CPU64 theta test has thermal-energy residual -1259.45 J/m². The
corresponding closed test has near-zero column residual despite cell errors
(-0.441954, +0.451009) K. **Global balance alone misses the local failure.**

Closed-column acoustic runs, original callback, maximum temperature change at
2 s (K):

| Path | dt (s) | CPU64 | Metal32 |
|---|---:|---:|---:|
| Explicit sedimentation | 0.1 | 0.214902 | 0.214966 |
| Explicit sedimentation | 0.025 | 0.088154 | 0.088226 |
| AIVA | 0.1 | 0.228246 | 0.228241 |
| AIVA | 2 | 2.862031 | 2.862030 |

No-fall controls stay at roundoff. These are coupled observations, not a total
energy budget, an asymptotic convergence study, or attribution of every error
to the donor callback. Corrected-callback coupled runs are NOT RUN.

## Remaining limits

NOT RUN: compressible StaticEnergy (unsupported constructor/diagnostic path),
P3 mixed coating/rime and differential-speed species, phase change, gravity-on
full energy budgets, corrected coupled dynamics, exact finite compressible
energy reconstruction, concentration-scaling of implicit error, and newer
Oceananigans bounded operators. No pressure-work, forcing, latent-reference or
main-branch acoustic/AIVA correction is included. In particular, no gravity-on
residual is asserted to equal falling potential-energy loss in the full model.
