# Recorded campaign: 17–18 September 2026

**Conclusion: these tests falsify an unconditional energy-conservation claim.**
They support two isolated compressible corrections, but neither a full model
energy law nor a production-ready finite-step fix. Production code is unchanged.

Production base: `dae9e46d543720f4f1c4f3a57e8d7c7817e90f24` (PR #959).
Campaign: `18f0880968ddad57f40b41fbfdabb26657a0f207`.
Oceananigans: **0.111.0**, pinned in the dedicated manifest, not the manuscript's
0.112 baseline. Local Julia 1.12.7; Apple M5 Max; Metal.jl 1.11.0.
Each TOML contains source and script/environment hashes. See the parent
[README](../README.md) for commands, equations, tolerances and case definitions.

## Completed runs

| Backend | Precision | Isolated cases / harness assertions | Coupled cases / finite-state assertions |
|---|---|---|---|
| CPU | Float64 | 85 / 201 pass | 8 / 242 pass |
| CPU | Float32 | 85 / 201 pass | NOT RUN |
| Metal, Apple M5 Max | Float32 | 85 / 201 pass | 8 / 242 pass |
| CUDA, NVIDIA Tesla T4 | Float64 | 85 / 201 pass | 8 / 242 pass |
| CUDA, NVIDIA Tesla T4 | Float32 | 85 unique cases across two jobs; finite-only part 36 assertions pass | 8 / 242 pass |

These assertion counts validate the harness and controls, **not energy
conservation**. All 12 original isothermal composition-contrast cases fail the
physical invariant on each tested backend/precision. Raw data are the adjacent
`cpu-*`, `metal-*` and `cuda-*` TOML files; temperature fields/tendencies were computed on
the selected backend, with host Float64 diagnostic analysis.

CUDA used Julia 1.12.6, CUDA.jl 6.4.0, runtime 12.9.0/compiler 12.9.86,
system driver 550.90.12, and the existing Tesla T4 node. Each allocation used
one GPU/two CPUs and a 20-minute bound. No scalar indexing was enabled.

The initial CUDA64 isolated run completed 85 cases/201 assertions before its
allocation timed out. Its production source was dae9e46d, with an untracked
campaign directory; script/environment hashes match 18f08809. CUDA64 coupled
then completed in job1261: eight cases/242 finite-state assertions, source
a03c3b7e (clean). Its profiles match recorded CPU64 within 5.12e-13 K.

CUDA32 isolated completed across two allocations: job1259 saved 61 cases before
a compilation-bound timeout; job1266 executed only the missing 24 finite
implicit cases (36/36 assertions), plus the eight coupled cases (242/242).
The 61-case file has no completed flag or final assertion summary; no single-run
85-case/201-assertion pass is claimed. The two case-name sets are disjoint.
All four harness/environment hashes match the committed files. Both later
worktrees were clean at a03c3b7e, whose production tree is the original baseline.
See [continuation provenance](cuda-continuation-provenance.json) for hashes,
case selection, and job outcomes. The finite-only driver retained the exact
committed loop/functions/assertions and changed only selection, suite label,
and the absolute experiments.jl loader path.

CUDA64 reproduces the nonisothermal beta_cv ratios to <9e-14 from unity.
CUDA32 ratios range from 0.99999436 to 1.00005609, with maximum absolute
rate error 8.15e-8 K/s; its isothermal phase-only maximum is 1.01e-7 K/s.
The active bounded-limiter mass residual is -1.80607e-5 kg/m²/s on CUDA32.
These are diagnostic substitutions on the original source, distinct from
validation of the subsequently prepared compressible-only candidate.

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
(CPU64), 3.24e-5 K (CPU/Metal Float32), and 5.99e-5 K (CUDA32). The Float32 floor is
`eps(Float32)*280 = 3.34e-5 K`; these controls remain within the 2e-4 K
harness tolerance. CUDA32 potential-temperature errors reach 0.501220 K,
consistent with the finite-step limitation observed on the other backends.

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
