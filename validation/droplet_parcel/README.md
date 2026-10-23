# L0: adiabatic parcel with Lagrangian droplets vs. the NCAR parcel model

Rung L0 of the Breeze vs. NCAR parcel-DNS comparison (`ncar-dns/COMPARISON_PLAN.md`): a 0-D
closed adiabatic parcel rising at U = 2 m/s for 600 s from T = 284.3 K, p = 93850 Pa and a
supersaturation of −14.39%, carrying the 39-bin, three-mode marine aerosol of the NCAR parcel
model `activation.F90` (Chen et al. 2020; ammonium sulfate, 111.8 particles per parcel).
Each aerosol bin is one superdroplet with a multiplicity. Droplets grow with
`Breeze.LagrangianMicrophysics.condensational_growth_step` (implicit, Δt = 0.01 s unless noted);
the parcel thermodynamics is integrated in the script.

## Running

```sh
julia --project=validation/droplet_parcel -e 'using Pkg; Pkg.instantiate()'
julia --project=validation/droplet_parcel validation/droplet_parcel/droplet_parcel.jl [ncar_run_directory]
```

The NCAR reference is read from `new.out` and `new.rad` of an `activation.F90` run
(default `~/ncar-dns/runs/parcel0`). The runs take about a minute on one CPU core.
Outputs go to `output/`: `summary.csv` (all runs), `time_series.csv` (NCAR, (a), (b) at 1 s),
`dsd_600s.csv`, `parcel_comparison.png` and `timestep_sensitivity.png`.

## Runs

* **(a) NCAR-matched**: constants and formulas of `activation.F90`: pᵛ⁺ = 2.53×10¹¹ exp(−5420/T),
  constant ℒ = 2.5×10⁶ J/kg, Rᵈ = 287, Rᵛ = 467, cᵖ = 1004, g = 9.8, σ = 0.0761 N/m, its linear
  Dᵛ(T) and cubic Kᵀ(T) fits (as user-defined `vapor_diffusivity`/`thermal_conductivity` models),
  the dilute (exponential) solute term with κ = 2 (18/132.14)(1726/1000) = 0.470, gas-kinetic
  corrections with αᶜ = 0.036, αᵀ = 0.7, Δᵛ = 0.104 μm, Δᵀ = 0.216 μm, its ventilation
  (Stokes fall speed, Re and Sc as coded), its parcel equations (density update, p = ρRᵈ(1 + 0.62qᵛ)T,
  𝒮 = rᵛ/rᵛ⁺ − 1 with mixing ratios, one-step lag of the latent heating), and haze droplets
  initially in equilibrium.
* **(a1)–(a5)**: cumulative switches from (a) to Breeze's defaults, to attribute the differences:
  (a1) Breeze thermodynamics (Clausius–Clapeyron pᵛ⁺ with ℒˡ(T), Rᵛ = 461.5, moist cᵖᵐ, g = 9.81,
  hydrostatic ascent, 𝒮 = pᵛ/pᵛ⁺ − 1); (a2) σ = 0.072 N/m; (a3) κ-Köhler activity;
  (a4) Pruppacher–Klett Dᵛ(T, p), Kᵀ(T), Sutherland μ(T); (a5) αᶜ = αᵀ = 1.
* **(b) Breeze defaults**: (a5) plus Beard–Pruppacher ventilation evaluated at the Beard (1976)
  terminal velocity.
* Primed runs start from NCAR's vapor mixing ratio instead of its supersaturation.
* Time-step sensitivity of (a) (Δt = 0.001–2 s) and (b) (0.01–1 s).

Diagnostics use one code path for all runs including NCAR's output: liquid water
qˡ = Σ nᵢ (4π/3) ρᴸ (rᵢ³ − rᵈᵢ³) per kg of air, and "activated" means r > rᶜ(rᵈ, T)
of the NCAR-matched Köhler curve.

## Notes on `activation.F90`

* The condensate increment `deltaqp = (4π/3) ρʷ Σ n Δr³ / vol` (kg per m³ of a fixed 1 cm³ volume)
  is subtracted from the vapor *mixing ratio* without dividing by the air density. The vapor budget
  is therefore that of a closed parcel with 111.8 particles **per milligram of air** (number mixing
  ratio 1.118×10⁸ kg⁻¹, i.e. 128 cm⁻³ at the initial density 1.145 kg/m³), not 112 cm⁻³; the runs here
  use that number mixing ratio. The `lwc` column of `new.out` (divided by ρ) is not consistent with
  the vapor budget: it equals qˡ/ρ (0.9835 qˡ at 600 s, where ρ = 1.0168 kg/m³).
* The vapor gas-kinetic term uses √(2π/(Rᵃ T)) with the dry-air gas constant; this is reproduced
  exactly by αᶜ = 0.036 √(Rᵃ/Rᵛ) = 0.0282 in the standard form.
* Ventilation uses the Stokes fall speed for all sizes and the dynamic viscosity in place of the
  kinematic one; it matters only for the largest (giant-CCN) bins.

## Results (Δt = 0.01 s unless noted; `output/summary.csv`)

| run | 𝒮max (%) | t(𝒮max) (s) | 𝒮(600 s) (%) | qˡ(600 s) (g/kg) | activated | r̄ (μm) | σʳ (μm) |
|---|---|---|---|---|---|---|---|
| NCAR `activation.F90` (Δt = 10⁻⁴ s, explicit r³) | 1.5875 | 163.4 | 0.3198 | 1.5815 | 0.770 | 16.312 | 0.207 |
| (a) NCAR-matched | 1.5873 | 163.4 | 0.3197 | 1.5815 | 0.770 | 16.312 | 0.207 |
| (a1) + Breeze thermodynamics | 1.5728 | 168.3 | 0.3193 | 1.5607 | 0.770 | 16.240 | 0.214 |
| (a2) + σ = 0.072 N/m | 1.5646 | 168.2 | 0.3189 | 1.5607 | 0.770 | 16.241 | 0.200 |
| (a3) + κ-Köhler activity | 1.5648 | 168.2 | 0.3189 | 1.5607 | 0.770 | 16.241 | 0.200 |
| (a4) + PK Dᵛ(T, p), Kᵀ(T) | 1.5469 | 167.9 | 0.2956 | 1.5614 | 0.770 | 16.242 | 0.207 |
| (a5) + αᶜ = αᵀ = 1 | 0.9760 | 162.8 | 0.2586 | 1.5625 | 0.727 | 16.565 | 0.102 |
| (b) Breeze defaults | 0.9760 | 162.8 | 0.2586 | 1.5625 | 0.727 | 16.565 | 0.102 |

(a) reproduces NCAR to |Δ𝒮| ≤ 5×10⁻⁴ % (absolute) over the whole run, |Δqˡ| ≤ 7×10⁻⁵ g/kg, and every
bin's radius at 600 s to 9×10⁻⁵ (relative). Statistics are over activated droplets (r > rᶜ).

* **Gas-kinetic coefficients dominate** the (a)–(b) difference: the effective αᶜ = 0.028 of
  `activation.F90` (0.036 with Rᵃ in place of Rᵛ) throttles the growth of small droplets, so 𝒮
  overshoots to 1.55% instead of 0.98%, 4 percentage points more of the aerosol activates, and the DSD is twice as
  broad (σʳ 0.21 vs 0.10 μm).
* **Thermodynamics** (a1): with 𝒮₀ = −14.39% in each model's own definition, Breeze's
  Clausius–Clapeyron pᵛ⁺(284.3 K) is 0.7% larger than 2.53×10¹¹ exp(−5420/T), so the parcel holds
  0.7% more vapor; the moist cᵖᵐ (≈1011 vs 1004), g = 9.81 and ℒˡ(T) ≈ 2.48×10⁶ (vs 2.5×10⁶) change
  the cooling and the Clausius–Clapeyron sensitivity, delaying saturation and 𝒮max by ~5 s and
  lowering qˡ(600 s) by 1.3%. Starting from NCAR's vapor instead (a1′) gives 𝒮₀ = −14.97% and
  a further 6 s delay.
* σ (0.0761 → 0.072 N/m) lowers 𝒮max by 0.5%; κ-Köhler vs dilute activity is negligible here;
  Pruppacher–Klett Dᵛ (pressure dependent; 4% larger than NCAR's fit at the start, 20% at 600 s) lowers the
  quasi-steady 𝒮(600 s) by 7%.
* Ventilation is negligible for this case ((a5) = (b)).
* Time step: 𝒮max of (a) changes by +0.0004, +0.004 and +0.012 percentage points from Δt = 0.001 s
  to Δt = 0.1, 1 and 2 s; qˡ by +0.25% at 1 s. The spectral width is the most sensitive (−19% at 1 s),
  because the supersaturation is frozen over the step while the implicit step damps differential
  growth.
