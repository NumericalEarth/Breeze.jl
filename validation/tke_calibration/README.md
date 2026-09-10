# Calibrating `TKEBasedTurbulenceClosure` against the Shen et al. (2022) LES library

Ensemble Kalman inversion (EKI) of the closure's stability-function and mixing-length coefficients against
the CNRM-CM6-1 `amip` members of the Shen et al. (2022) library of GCM-forced large-eddy simulations, run
through Breeze's column-ensemble mode: every (parameter set, LES member) pair is one column of a single
`AtmosphereModel`, so one model run evaluates the whole ensemble on all training members at once.

Status (2026-09-10): the pipeline is complete and tested; the first calibration under the final protocol
(below) is running on a laptop CPU at roughly an hour per EKI iteration. It is meant to move to a GPU. All
earlier calibrations used a protocol that differed from the LES (no precipitation, centered subsidence,
replayed radiation) and are superseded; their numbers are kept in *Results so far* as a baseline.

## The protocol: a column that replays the LES's own setup

Each LES member (a cfSite and a month) was forced by the GCM's time-invariant large-scale state (Shen et
al. 2022, §2). The column approximates that setup while replacing resolved turbulence with the closure.
The prescribed LES surface fluxes, DCMIP Kessler implementation, liquid-water potential-temperature
relaxation, reconstructed solar forcing, and upper-column extension are additional modeling choices.

| | LES (PyCLES) | Column (`run_ensemble`) |
|---|---|---|
| Subsidence | upstream differencing of −wˢ ∂z | `SubsidenceForcing(wˢ; advection = UpwindBiased(order = 1))`, including vapor, cloud liquid and rain |
| Horizontal advection, GCM vertical eddy flux | time-invariant tendencies of θ, qᵗ | `Forcing` with the file's `hadv + fluc` |
| Surface fluxes | bulk, from the GCM SST | the LES's hourly SHF, LHF, stress as flux boundary conditions |
| Relaxation | winds to the GCM on 6 h; T, qᵗ above 3 km on 24 h | matching rates and height mask toward `*_mean_initial`; θˡ replaces T, moisture relaxation uses qᵛ + qᶜˡ |
| Microphysics | Kessler warm rain | `DCMIP2016KesslerMicrophysics()` with Tetens saturation |
| Radiation | RRTM every step, fixed sun (GCM insolation, insolation-weighted cos θ_z), albedo 0.06, ε 0.95 | RRTMGP all-sky every 10 min, `FixedCosineZenith` and solar constant per column, same albedo and emissivity |
| Domain | 4 km, 20 m cells; GCM profiles patched above for radiation | LES grid or coarser to 4 km, then faces stretched 12 %/cell to 25 km; the monthly-mean GCM column above 4 km, relaxed toward on 10 min |
| Duration, scoring | 6 days (one member 3.7); reduction uses the last 2 days (paper figures use the last day) | the reduction's window; time means of θˡ, qᵗ, qˡ, qʳ, u, v |

**Protocol version 2** corrects missing cloud/rain subsidence, total-water relaxation applied to vapor
alone, and rain incorrectly triggering the closure's saturated stability branch. Rain still contributes
to thermodynamic liquid water, heat capacity and buoyancy loading. Above the LES top, cloud and rain
relax toward zero alongside the prescribed GCM vapor profile. Checkpoints record this version and the
run configuration; resuming an older or incompatible forward map is rejected. Short diagnostic runs
must specify an averaging window they reach; an empty window is an error.

The observation vector of a column is its time-mean θˡ (K), qᵗ, qˡ (g kg⁻¹), u and v (m s⁻¹) as means over
100 m cells from the surface to 3 km (Oceananigans' conservative `regrid!`), so that columns on different
grids are scored alike; the LES targets are regridded the same way. Observation noise per cell: 0.25 K,
0.25 and 0.1 g kg⁻¹, 0.5 m s⁻¹. Rain water is diagnosed but not scored (the LES's horizontal-mean rain is
not the quantity a mean-field Kessler column produces, and scoring it diverged a calibration). Turbulent
fluxes and TKE are deliberately not targets: the column's TKE need not equal the LES's (see the CATKE paper).

**Parameters.** `RiDependentSpace`, 17: the twelve endpoints of `RiDependentStabilityFunctions` (Cᵘ, Cᶜ, Cᵉ,
Cᴰ in unstable, neutral and stable air), the transition Ri⁰ and Riᵟ, the wall coefficient Cˢ of
`GradientLimitedMixingLength`, and two surface-TKE-flux coefficients (Jᵉ = Cᵂu★ u★³ + Cᵂʷ wΔ³, CATKE's form,
zero in the default closure). `ConstantSpace`, 7: `ConstantStabilityFunctions(Cᵘ, Cᶜ, Cᵉ, Cᴰ)`, Cˢ and the
two flux coefficients, i.e. a Nakanishi–Niino-form closure. Priors are independent constrained Gaussians on
(0, ∞) centered on the defaults (1 for the flux coefficients) with a standard deviation of half the center.

**Inversion.** EnsembleKalmanProcesses' `Inversion()` with its defaults: the `DataMisfitController`
(Iglesias–Yang tempering, Δtₙ = 1 / mean squared normalized misfit, stop at pseudo time 1 where the ensemble
approximates the posterior), SECNice localization, Nesterov acceleration. The update is deterministic, so a
checkpoint (every iteration: parameters, forward map, misfit, step) resumes exactly by replaying the saved
forward maps. Ensemble sizes 200 (17 parameters) and 100 (7). Training members: sites 2, 5, 8, 11, 14, 17,
20, 23 × {January, July} = 16; the other 67 members (all April and October) are held out.

**Grids.** Calibration on three grids at once — 50 m, 100 m, and NumericalEarth's hindcast grid (50 m at the
surface stretching 10 % per level; 23 cells below 4 km) — so the parameters hold across the resolutions the
closure is used at; the 20 m LES grid is used for evaluation.

## Layout

```
validation/tke_calibration/
├── Project.toml                 # the study's environment; Breeze from ../.. via [sources]
├── data/gcm_columns_CNRM-CM6-1_amip.nc   # monthly-mean GCM columns above the LES (committed, 0.3 MB)
├── src/                         # the BreezeCalibration module
│   ├── BreezeCalibration.jl     #   LES members, parameter spaces, ColumnEnsembleProblem, run_ensemble, scores
│   ├── grids.jl                 #   LES / uniform / hindcast faces, conservative regridding
│   ├── multiresolution.jl       #   several grids in one inversion
│   └── inversion.jl             #   priors, run_eki, checkpoint and replay
├── scripts/
│   ├── smoke_test.jl            # 2 × 2 columns for 3 hours; first thing to run on a new machine
│   ├── time_forward_map.jl      # seconds per step versus ensemble size, to size a run
│   ├── validate_radiation.jl    # RRTMGP on the LES's initial state against the LES's own heating
│   ├── calibrate.jl             # the EKI run (checkpointed, resumable)
│   ├── evaluate.jl              # a checkpoint's parameters against the defaults on all 83 members
│   ├── compare_calibrations.jl  # held-out skill versus resolution for several calibrations
│   ├── visualize.jl             # parameters, misfit trajectory, importance, profiles
│   ├── profile_stamps.jl        # every member's θˡ and qᵗ profiles, LES versus column
│   ├── last_iteration.jl        # one line per saved iteration of a checkpoint
│   ├── fetch_gcm_columns.jl     # regenerate data/gcm_columns_… from CMIP6 CFsubhr (not needed to run)
│   └── solar_parameters.jl      #   … and add the fixed-sun insolation and cos θ_z per site and month
└── test/runtests.jl             # replay, grids, parameter spaces, observations, a 12-minute EKI end to end
```

## Running

```bash
cd validation/tke_calibration
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t auto --project=. test/runtests.jl                     # a few minutes; downloads the LES artifact (27 MB)
julia -t auto --project=. scripts/smoke_test.jl [arch=gpu]     # 2 members × 2 parameter sets, 3 hours
julia -t auto --project=. scripts/time_forward_map.jl 200 [arch=gpu]

# The two calibrations, then their evaluation on every member at four resolutions
julia -t auto --project=. scripts/calibrate.jl 200 space=ri       output=results/eki_ri.jld2       [arch=gpu]
julia -t auto --project=. scripts/calibrate.jl 100 space=constant output=results/eki_constant.jld2 [arch=gpu]
julia -t auto --project=. scripts/evaluate.jl checkpoint=results/eki_ri.jld2       output=results/evaluation_ri.jld2       [arch=gpu]
julia -t auto --project=. scripts/evaluate.jl checkpoint=results/eki_constant.jld2 output=results/evaluation_constant.jld2 [arch=gpu]

# Figures and tables
julia --project=. scripts/compare_calibrations.jl "Ri=results/evaluation_ri.jld2" "constant=results/evaluation_constant.jld2"
julia --project=. scripts/visualize.jl checkpoint=results/eki_ri.jld2 evaluation=results/evaluation_ri.jld2 resolution=50
julia --project=. scripts/profile_stamps.jl evaluation=results/evaluation_ri.jld2 resolution=50
julia --project=. scripts/last_iteration.jl results/eki_ri.jld2 all

# Continue an interrupted calibration (same space and resolutions)
julia -t auto --project=. scripts/calibrate.jl resume=results/eki_ri.jld2 output=results/eki_ri.jld2
```

`calibrate.jl` and `evaluate.jl` default to the protocol above (`resolutions=50,100,hindcast`, `top=25000`,
`radiation=interactive`). `top=les radiation=prescribed` gives the control experiment: a column ending at the
LES top with the LES's hourly heating replayed, which is the experiment to fall back on if the interactive
radiation misbehaves. `evaluate.jl` reads the column top and radiation from the checkpoint.

**GPU.** `arch=gpu` builds the column ensemble on `GPU()` (CUDA). Everything the kernels read is moved with
`on_architecture`; the regridding and the EKI update stay on the host. The GPU path has not been exercised yet
(the pipeline was developed on an Apple laptop): run `smoke_test.jl arch=gpu`, then `time_forward_map.jl`, first.

**Cost on the CPU** (Apple M5 Max, 9 threads; Oceananigans' column-ensemble kernels scale weakly with
threads): a forward map is one run per grid of 3200 columns (200 × 16) for 6 days at Δt = 60 s. With the
LES-top column and prescribed radiation the three coarse grids together took about 25 min per iteration; the
25 km column with RRTMGP roughly doubles that. Reaching pseudo time 1 has taken 10–15 iterations, so a
17-parameter calibration is a 10-hour job here, the 7-parameter one about half — hence the GPU.

## Results so far (superseded protocol)

Held-out medians over 67 members of the RMSE below 3 km of the calibrated ensemble mean, θˡ (K) / qᵗ (g kg⁻¹) /
wind (m s⁻¹). Columns ended at the LES top with the LES's heating replayed, saturation-adjustment microphysics
and centered subsidence, but the corrected GCM relaxation targets; both calibrations fit the five fields on the
50 m, 100 m and hindcast grids together.

| grid | Nakanishi–Niino defaults | Ri-dependent (17) | constant (7) |
|---|---|---|---|
| 20 m | 1.08 / 0.99 / 0.34 | 0.67 / 0.52 / 0.22 | 0.73 / 0.60 / 0.25 |
| 50 m | 1.24 / 1.10 / 0.35 | 0.78 / 0.56 / 0.23 | 0.84 / 0.65 / 0.24 |
| 100 m | 1.35 / 1.33 / 0.35 | 0.74 / 0.66 / 0.23 | 0.86 / 0.75 / 0.25 |
| hindcast | 1.26 / 1.33 / 0.35 | 0.90 / 0.76 / 0.26 | 1.01 / 0.78 / 0.27 |

What these established: the closure's error roughly halves under calibration and the improvement carries
across resolutions; the Ri-dependent stability functions beat the constant ones by 0.06–0.12 K; the mixing
length and dissipation coefficients are weakly identifiable jointly (Cˢ between 2.6 and 5.2 with similar
skill); the worst cases are deep-convective April members in the ITCZ (sites 13–15) and deep trade cumulus
(sites 8, 12, 21–23), where the column's inversion and entrainment zone are misplaced. The 2Δz noise seen
above the inversion came from the centered subsidence difference and is gone with the upwind scheme.

Radiation validation for the new protocol (`validate_radiation.jl`, RRTMGP on the LES's initial state
against the LES's first-hour heating, five members): cloud-top cooling peaks agree (−1.6 vs −1.75 K/day at
site 22 July), RMS heating difference 0.18–0.40 K/day on 100 m cells, column-integrated cooling below 4 km
4–15 % weaker in the column (46 vs 54 W m⁻² at site 22 July). Candidates for the residual: RRTMGP's 10 μm
droplets against PyCLES's density-dependent effective radius, RRTMGP against RRTMG, the hour-1 vs t = 0
comparison.

## Plan

1. **Finish the two calibrations under the final protocol** (`space=ri` 200 members, `space=constant` 100),
   evaluate both at 20 m, 50 m, 100 m and the hindcast grid, and compare them with the superseded results:
   the difference isolates what precipitation, upwind subsidence and interactive radiation change.
2. **Decide the defaults.** If the Ri-dependent set is worth its ten extra parameters, propose it as the
   closure's calibrated default in a follow-up PR; otherwise the constant set.
3. **Then**: refit with the 20 m grid included if the coarse-grid parameters do not transfer to it; test a
   subgrid condensation scheme on the cumulus cases, which no coefficient choice fixes; extend to the
   `amip4K` members (in the artifact) for the warming response.

## Data

- **LES**: Breeze's lazy artifact `shen_et_al_2022_les_profiles`, a reduction of Shen et al.'s CC0 library
  (doi:10.22002/D1.20052) by `validation/cloud_les_library/`; 83 `amip` CNRM-CM6-1 members (cfSites 2–15,
  17–23 × January, April, July, October; site 15 January is absent). Each file carries its forcing, hourly
  radiative heating and surface fluxes, initial and relaxation profiles and time-mean targets.
- **GCM columns** (`data/gcm_columns_CNRM-CM6-1_amip.nc`): 2004–2008 monthly-mean `ta`, `hus`, `ps` from the
  CMIP6 CNRM-CM6-1 `amip` `r1i1p1f2` CFsubhr output at the 21 cfSites (hybrid-sigma pressure and hydrostatic
  heights computed), plus the astronomical monthly-mean TOA insolation and insolation-weighted cos θ_z per site
  and month (S₀ = 1361 W m⁻²). `scripts/fetch_gcm_columns.jl` and `scripts/solar_parameters.jl` regenerate it
  (temperature streams from CEDA's OPeNDAP server; the humidity files must be downloaded from DKRZ).

## References

Shen, Z., Sridhar, A., Tan, Z., Jaruga, A., Schneider, T. (2022). A library of large-eddy simulations forced by
global climate models. *J. Adv. Model. Earth Syst.* 14, e2021MS002631.
Iglesias, M., Yang, Y. (2021). Adaptive regularisation for ensemble Kalman inversion. *Inverse Problems* 37, 025008.
