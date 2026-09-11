# Calibrating `TKEBasedTurbulenceClosure` against the Shen et al. (2022) LES library

Ensemble Kalman inversion (EKI) of the closure's stability-function and mixing-length coefficients against
the CNRM-CM6-1 `amip` members of the Shen et al. (2022) library of GCM-forced large-eddy simulations, run
through Breeze's column-ensemble mode: every (parameter set, LES member) pair is one column of a single
`AtmosphereModel`, so one model run evaluates the whole ensemble on all training members at once.

**Status (2026-09-11).** The pipeline runs on the GPU and is validated there. The numerics are *not* yet
settled: Δt = 60 s is measurably unconverged, and the time step and radiation interval quoted in the
commands below are **provisional** until the refinement study finishes. No coefficients are adopted yet.
`PROTOCOL_VERSION = 3`; checkpoints record it and refuse to resume across a change.

## The protocol: a column that replays the LES's own setup

Each LES member (a cfSite and a month) was forced by the GCM's time-invariant large-scale state (Shen et
al. 2022, §2). The column reproduces that forcing; only the turbulence closure differs from the LES.

| | LES (PyCLES) | Column (`run_ensemble`) |
|---|---|---|
| Subsidence | upstream differencing of −wˢ ∂z | `SubsidenceForcing(wˢ; advection = UpwindBiased(order = 1))`, acting on θ, qᵛ, cloud liquid and rain |
| Horizontal advection, GCM vertical eddy flux | time-invariant tendencies of θ, qᵗ | `Forcing` with the file's `hadv + fluc` |
| Surface fluxes | bulk, from the GCM SST | the LES's hourly SHF, LHF, stress as flux boundary conditions |
| Relaxation | winds to the GCM on 6 h; T, qᵗ above 3 km on 24 h | the same, toward the `*_mean_initial` (GCM) profiles; the moisture target is the vapor-plus-cloud sum |
| Microphysics | Kessler warm rain | `DCMIP2016KesslerMicrophysics()` with Tetens saturation |
| Radiation | RRTM every step, fixed sun, albedo 0.06, ε 0.95 | RRTMGP all-sky on `radiation_interval`, `FixedCosineZenith` and solar constant per column, same albedo and emissivity |
| Domain | 4 km, 20 m cells | LES grid or coarser to 4 km, then faces stretched 12 %/cell to 25 km; the monthly-mean GCM column above, relaxed toward on 10 min |
| Duration, scoring | 6 days; means over the last 2 days | the same window; time means of θˡ, qᵗ, qˡ, qʳ, u, v |

The observation vector of a column is its time-mean θˡ (K), qᵗ, qˡ (g kg⁻¹), u and v (m s⁻¹) as means over
100 m cells from the surface to 3 km (Oceananigans' conservative `regrid!`), so columns on different grids
are scored alike; the LES targets are regridded the same way. Observation noise per cell: 0.25 K, 0.25 and
0.1 g kg⁻¹, 0.5 m s⁻¹. Rain is diagnosed but not scored. Turbulent fluxes and TKE are not targets.

**Parameters.** `RiDependentSpace`, 17: the twelve endpoints of `RiDependentStabilityFunctions`, the
transition Ri⁰ and Riᵟ, the wall coefficient Cˢ of `GradientLimitedMixingLength`, and two surface-TKE-flux
coefficients (Jᵉ = Cᵂu★ u★³ + Cᵂʷ wΔ³, zero in the default closure). `ConstantSpace`, 7:
`ConstantStabilityFunctions(Cᵘ, Cᶜ, Cᵉ, Cᴰ)`, Cˢ and the two flux coefficients. Priors are independent
constrained Gaussians on (0, ∞) centred on the defaults (1 for the flux coefficients), standard deviation
half the centre.

## What the inversion optimizes, and when it stops

EnsembleKalmanProcesses' `Inversion()` with the `DataMisfitController` (Iglesias–Yang tempering), SECNice
localization and Nesterov acceleration.

Every iteration evaluates **one extra column at the constrained ensemble mean**, excluded from the EKI
update. This matters because the coefficients one would adopt are the ensemble mean ϕ̄, while the misfit
natural to report is the average over members of Φ(θⱼ), and Φ(ϕ̄) ≠ mean Φ(θⱼ). The extra column rides the
existing ensemble, so it costs ≈ 1/N_ens of a forward map rather than a second run.

Pseudo time 1 is where the tempering says the ensemble approximates the posterior. It is **not** where the
objective stops improving, so `optimize=true` continues past it until the directly evaluated Φ(ϕ̄) plateaus
(`objective_tolerance`, `objective_patience`, a minimum number of post-tempering iterations, and a hard
cap), retaining the best evaluated mean as `selected_mean` in the checkpoint.

Two cautions the scripts enforce in their wording. The spread of the terminal ensemble is **not** a
posterior uncertainty — EKI is not a sampler here — so it is reported as spread. And `selected_mean` is the
**best evaluated candidate**, not an established optimum, until its plateau is corroborated by independent
seeds and local perturbation.

## Held-out data: validation and a reserved test

Three roles, and the distinction is load-bearing:

- **Training**: the sites and months the inversion fits.
- **Validation** (sites 3, 12, 21, all months): chooses the design — the training split, the ensemble size,
  which closure family to adopt. Named explicitly and identical for every calibration compared, so the
  yardstick does not move with the design being judged.
- **Reserved test** (sites 6, 9, 15, 18, all months): touched by nothing. `evaluate.jl` integrates and saves
  these scores but does not print them without `reveal=true`, which is meant to be passed once, after the
  coefficients are frozen.

The 20 m LES grid has been used during design work, so it is an **untrained-resolution transfer** check, not
an untouched resolution test.

## Cost, measured on one A100-SXM4-40GB

The GPU reproduces the CPU to 2.7 × 10⁻¹² relative on the time means, and is 17.8× faster than the same
node's CPU at production size (1.62 s/step against 0.091 at 3 200 columns).

Cost per step is nearly flat in ensemble size — 16 training cases, 50 m grid, Δt = 60 s:

| N_ens | columns | s/step | VRAM |
|---|---|---|---|
| 200 | 3 200 | 0.086 | 1.3 GiB |
| 400 | 6 400 | 0.102 | 1.6 GiB |
| 800 | 12 800 | 0.112 | 2.0 GiB |
| 1600 | 25 600 | 0.140 | 4.5 GiB |

Eight times the columns for 1.62× the time. **Ensemble members and training cases are not
interchangeable, though**: the forward map depends on total columns, but the localized EKI update forms an
N_obs × N_obs covariance on the host, with N_obs = grids × cases × 5 × 30. Measured: 16 cases on one grid
is 1.9 s per update; 56 cases on three grids is 25 200 observations, a 4.73 GB covariance and ~150 s. Cases
are charged twice, ensemble members once.

Radiation is ~80 % of the step cost, and it fires on model time, so refining Δt adds no radiation calls.
Forward map on the 50 m grid at 3 200 columns (144 h = the latest training window end):

| Δt | radiation interval | min per forward map |
|---|---|---|
| 60 s | 10 min | 15.4 |
| 60 s | 30 min | 9.0 |
| 30 s | 30 min | 14.2 |
| 15 s | 30 min | 24.6 |
| 7.5 s | 30 min | 41.6 |

Measuring cost correctly takes more care than it looks. `run_ensemble` returns a `timing` whose
`integration_seconds` covers `run!` alone, excluding model setup — but `run!` still carries one-offs of its
own (the first `update_state!`, the first radiation call, late kernel specialization) worth tens of seconds
at production size. So: **difference `integration_seconds` between a short and a long run of the same
configuration**, with both long enough to contain several radiation calls at the same cadence. Differencing
wall times instead leaves setup variation in, badly enough to have produced a negative cost per step;
dividing a single short run by its step count instead charges the one-offs to every step, badly enough to
have overstated one by 6×. Discard the first configuration measured in a process, whichever it is: it is
inflated in both setup and per-step.

## Numerics: provisional, and why it matters

`scripts/discretization_sensitivity.jl` asks of each knob how far the *scored* observation vector moves
when the knob is refined, in units of the observation noise, next to the misfit the inversion is reducing.
A knob whose refinement moves the score by a sizeable fraction of that misfit is one the coefficients
would absorb.

- **Time step**: Δt = 60 s sits 1.20 σ from 15 s against a 4.68 σ misfit (a ratio of 0.26); 30 s is 0.80 σ;
  15 s is still 0.29 σ from 3.75 s. **Δt = 60 s is not converged for calibration.** The refinement study
  (3.75 and 1.875 s) is running; until it reports, every `dt=` below is provisional.
- **Radiation interval**: 0.085 σ between 10 and 30 minutes, a ratio of 0.018 — nearly free, and the
  cheapest source of time for a finer step.

## Layout

```
validation/tke_calibration/
├── Project.toml                 # the study's environment; Breeze from ../.. via [sources]
├── data/gcm_columns_CNRM-CM6-1_amip.nc   # monthly-mean GCM columns above the LES (committed, 0.3 MB)
├── src/                         # the BreezeCalibration module
│   ├── BreezeCalibration.jl     #   PROTOCOL_VERSION, LES members, parameter spaces, run_ensemble, scores
│   ├── grids.jl                 #   LES / uniform / hindcast faces, conservative regridding
│   ├── multiresolution.jl       #   several grids in one inversion
│   └── inversion.jl             #   priors, run_eki, the evaluated mean, optimization mode, checkpoints
├── scripts/
│   ├── smoke_test.jl                  # 2 × 2 columns for 3 hours; first thing to run on a new machine
│   ├── discretization_sensitivity.jl  # Δt and radiation interval against the observation noise
│   ├── calibrate.jl                   # one EKI run (checkpointed, resumable)
│   ├── batched_calibrate.jl           # several independent runs sharing forward evaluations
│   ├── evaluate.jl                    # a checkpoint's candidates on all 83 members, split train/validation/reserved
│   ├── compare_designs.jl             # designs against each other on the common validation sites
│   ├── design_objective.jl            # one pooled objective per design, from saved means
│   ├── compare_ensemble_sizes.jl      # trajectory, spread and agreement across ensemble sizes
│   ├── polish_candidate.jl            # local perturbation around a candidate
│   ├── diagnose_candidate.jl          # closure diagnostics over the scored window
│   └── …                              # visualization, export, data regeneration
└── test/
    ├── runtests.jl                    # replay, grids, spaces, observations, EKI end to end
    ├── gpu_pipeline.jl                # exact batching and diagnostics on CUDA (ConstantSpace)
    └── ri_space_column_independence.jl # the same exactness for RiDependentSpace
```

## Running

```bash
cd validation/tke_calibration
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t auto --project=. test/runtests.jl                       # a few minutes; downloads the LES artifact (27 MB)
julia -t auto --project=. test/gpu_pipeline.jl                   # on a CUDA machine
julia -t auto --project=. scripts/smoke_test.jl arch=gpu         # 2 members × 2 parameter sets, 3 hours
```

A small ensemble is *slower* on the GPU than on the CPU — the smoke test is a few hundred columns, where
launch overhead dominates. That is expected, not a regression; the GPU wins from ~1 000 columns up.

One calibration. The `dt` and `radiation_interval` here are provisional pending the refinement study:

```bash
julia -t auto --project=. scripts/calibrate.jl 400 space=ri resolutions=50,100,hindcast arch=gpu \
      optimize=true max_iterations=40 dt=7.5 radiation_interval=1800 \
      sites=2,5,8,11,14,17,20,23 months=01,04,07,10 output=results/eki_ri.jld2
```

Several independent runs — ensemble sizes and seeds — sharing forward evaluations while keeping separate
optimizers and checkpoints:

```bash
julia -t auto --project=. scripts/batched_calibrate.jl space=ri runs=200:1,400:1,400:2,800:1 \
      dt=7.5 radiation_interval=1800 resolutions=50,100,hindcast max_columns=32000 \
      sites=2,5,8,11,14,17,20,23 months=01,04,07,10 output=results/final_ri
```

Evaluation and comparison. `evaluate.jl` takes the time step and radiation interval from the checkpoint, so
a calibration is scored under the protocol it was fit under:

```bash
julia -t auto --project=. scripts/evaluate.jl checkpoint=results/eki_ri.jld2 resolutions=20,50,100,hindcast \
      arch=gpu output=results/evaluation_ri.jld2
julia --project=. scripts/compare_designs.jl A=results/design/A.jld2 B=results/design/B.jld2 \
      resolutions=50,20 arch=gpu output=results/design_comparison.jld2
julia --project=. scripts/design_objective.jl results/design_comparison.jld2
```

Resuming is automatic for the batched driver and explicit for a single run
(`resume=results/eki_ri.jld2`). A resume across a different protocol version, training split, grid set,
time step or optimizer is refused rather than silently accepted: EKI resumes by replaying saved forward
maps, so mixing them would leave the result meaningless while every dimension still matched.

## The design question: ensemble members against cases

Because cost is sublinear in columns, the interesting question is not "is N_ens = 200 enough" but how to
spend a budget. A first study at Δt = 30 s on the 50 m grid, three designs at ~6 400 columns each, scored on
the common validation sites and pooled into the noise-normalized objective Φ:

| design | N_ens × cases | Φ (mean of 20 and 50 m) |
|---|---|---|
| default coefficients | — | 8.195 |
| A | 400 × 16 | 3.046 |
| B | 200 × 32 | 3.034 |
| C | 114 × 56 | 5.785 |

Calibration is worth a factor 2.7. A and B are indistinguishable on one seed each. C is clearly worse —
**but that is not evidence against 56 cases.** Holding columns fixed forced its ensemble down to 114, only
6.7× the parameter dimension, and C also fit its *own* training set worse (misfit 4.33 against B's 2.79),
which is an under-resolved optimizer rather than poor generalization. More cases means more physics to
explore and so wants a *larger* ensemble, not a smaller one; the two should grow together. A design at
400 × 56 is the current test of that, and the final ladder varies N_ens at fixed cases rather than trading
one against the other.

Differences smaller than the seed-to-seed scatter are not rankings, and no run so far has error bars.

## Data

- **LES**: Breeze's lazy artifact `shen_et_al_2022_les_profiles`, a reduction of Shen et al.'s CC0 library
  (doi:10.22002/D1.20052) by `validation/cloud_les_library/`; 83 `amip` CNRM-CM6-1 members (cfSites 2–15,
  17–23 × January, April, July, October; site 15 January is absent).

  The archive stores some diagnostics in units the variable names do not announce — `tke_mean` is a density
  ρ₀ e, and `qt_flux_z`/`qt_sgs_flux_z` are mass fluxes, where the column model carries specific and
  kinematic quantities. These affect diagnostic plots only; the five scored fields are unaffected. Treat an
  archive array's units as unknown until checked against the original source.
- **GCM columns** (`data/gcm_columns_CNRM-CM6-1_amip.nc`): 2004–2008 monthly-mean `ta`, `hus`, `ps` from the
  CMIP6 CNRM-CM6-1 `amip` `r1i1p1f2` CFsubhr output at the 21 cfSites, plus the monthly-mean TOA insolation
  and insolation-weighted cos θ_z per site and month. `scripts/fetch_gcm_columns.jl` and
  `scripts/solar_parameters.jl` regenerate it.

## Hardware note

`sinfo` reports only `gpu:1` and does not distinguish models, and this cluster runs four, each name taken
from a job log on the node rather than from the partition name:

| partition | device | VRAM |
|---|---|---|
| `gpua100` | A100-SXM4-40GB | 41 GB |
| `gpua100largex4` | A100-SXM4-80GB | 82 GB |
| `gpuprod` | H100 80GB HBM3 | 79 GB |
| `gpudev` (default) | Tesla T4 | 15 GB |

**Exclude the T4 from production scheduling.** Its FP64 throughput is a small fraction of an A100's and it
measured 5.4× slower end to end on this workload; its 15 GB must not set the batching chunk limit, which
should be chosen for the A100-40GB and H100-80GB the study actually runs on. Query the node before
trusting a partition name, and never compare timings across models.

## References

Shen, Z., Sridhar, A., Tan, Z., Jaruga, A., Schneider, T. (2022). A library of large-eddy simulations forced by
global climate models. *J. Adv. Model. Earth Syst.* 14, e2021MS002631.
Iglesias, M., Yang, Y. (2021). Adaptive regularisation for ensemble Kalman inversion. *Inverse Problems* 37, 025008.
