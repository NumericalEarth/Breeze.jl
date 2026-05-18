# P3 Module File Reorganization

**Date:** 2026-05-18
**Module:** `src/Microphysics/PredictedParticleProperties/`
**Scope:** File-level partitioning (no logic changes, no API changes)

## Motivation

The P3 implementation has grown to 32 files / ~11,000 lines, all in a flat
directory. Six files have become monolithic, each mixing multiple distinct
concerns. The largest, `process_rates.jl`, is 2,508 lines and spans
seven independently-readable sections (utilities, kernel evaluators, CCN,
coupled vapor solver, derived-state structs, prognostic tendencies, sixth-moment
tendencies). Reading or modifying these files requires holding too much in
context at once.

This reorganization splits the six largest files along their natural seams.
Twenty-six already-focused files stay where they are.

## Non-goals

- **No public API changes.** Every name exported from
  `PredictedParticleProperties` stays exported, at the same call signature.
- **No logic changes.** This is pure file partitioning. Function bodies are
  moved verbatim. Docstrings move with their functions.
- **No dependency changes.** `Project.toml` `[deps]` / `[weakdeps]` / `[compat]`
  are untouched.
- **No subdirectory grouping.** Files stay flat. Subdirectories were considered
  and rejected as larger churn for a smaller readability gain than the file
  splits themselves.
- **No renames of existing focused files.** Only new files added; existing files
  shrink or stay the same.

## File-by-file plan

Line ranges below refer to the current `glw/p3` branch state. Function
groupings are listed in the order they should appear in the new file
(generally preserving current top-to-bottom order).

### 1. `process_rates.jl` (2508 → 7 files)

Current file contains seven independently-readable sections:

| New file | Contents |
|---|---|
| `process_rate_helpers.jl` | `sink_limiting_factor`, `limit_vapor_rates`, `safe_divide`, `ice_air_density_correction`, `mean_ice_distribution_state`, `mean_ice_particle_diameter`, `consistent_rime_state`, `liquid_fraction_on_ice`, `mean_total_ice_mass`, `rain_slope_parameter`, `ice_mean_density_for_bounds`, `bound_ice_sixth_moment` (both methods), `compute_ice_shape_parameter`, `ice_shape_parameter` (both methods), `liquid_psychrometric_correction`, `ice_psychrometric_correction`, `saturation_vapor_pressure_at_freezing` |
| `tabulated_kernels.jl` | `ventilation_sc_correction`, `deposition_ventilation` (both methods), `melting_ventilation`, `collection_kernel_per_particle` (both methods), `aggregation_kernel` (both methods) |
| `ccn_activation_rates.jl` | `ccn_activation_rate`, `compute_ccn_activation` (both methods), `ventilation_enhanced_deposition` |
| `coupled_saturation_adjustment.jl` | `P3CoupledVaporRates` struct, `predicted_supersaturation_adjustment`, `cloud_condensation_epsilon`, `rain_condensation_epsilon`, `ice_relaxation_epsilon`, `ice_deposition_epsilon`, `ice_coating_epsilon`, `coupled_saturation_adjustment_rates` |
| `process_rates.jl` (shrunk) | `P3DerivedState`, `P3Phase1Rates`, `P3Phase2Rates`, `P3ProcessRates` struct definition and constructor |
| `prognostic_tendencies.jl` | `tendency_ρqᶜˡ`, `tendency_ρqʳ`, `tendency_ρnʳ`, `tendency_ρqⁱ`, `tendency_ρnⁱ`, `tendency_ρqᶠ`, `tendency_ρbᶠ`, `split_splintering_mass`, `tendency_ρnᶜˡ`, `tendency_ρqʷⁱ`, `tendency_ρqᵛ`, `tendency_ρsˢᵃᵗ` |
| `sixth_moment_tendencies.jl` | `group2_ice_sixth_moment_tendency`, `active_ice_sixth_moment_tendency`, `tendency_ρzⁱ` (three methods), `tabulated_z_tendency` (two methods), `rain_riming_sixth_moment_factor` (three methods), `initiated_ice_sixth_moment_tendency`, `nucleation_sixth_moment_tendency` |

### 2. `lambda_solver.jl` (1172 → 4 files)

| New file | Contents |
|---|---|
| `ice_mass_relation.jl` | `IceMassPowerLaw` struct + constructor, `regime_threshold`, `deposited_ice_density`, `graupel_density`, `ice_mass_coefficients`, `ice_mass` |
| `ice_shape_closures.jl` | `FixedShapeParameter` + `shape_parameter(::FixedShapeParameter, …)`, `TwoMomentClosure` + `shape_parameter(::TwoMomentClosure, …)`, `ThreeMomentClosure`, `IceRegimeThresholds` + `ice_regime_thresholds` |
| `gamma_moments.jl` | `log_gamma_moment`, `log_gamma_inc_moment`, `logaddexp`, `log_mass_moment`, `log_mass_number_ratio`, `log_lambda_from_reflectivity`, `shape_parameter_from_moments`, `mass_residual_three_moment`, `g_of_mu`, `enforce_z_bounds` |
| `lambda_solver.jl` (shrunk) | P3_DM_* constants, `DiameterBounds` struct + both constructors, `lambda_bounds_from_diameter`, `enforce_diameter_bounds`, `IceDistributionParameters`, `solve_lambda` (two methods), `intercept_parameter`, `distribution_parameters` (two methods) |

### 3. `p3_interface.jl` (894 → 3 files)

| New file | Contents |
|---|---|
| `p3_microphysical_state.jl` | `const P3 = PredictedParticlePropertiesMicrophysics`, `P3MicrophysicalState` struct, `AM.prognostic_field_names`, `AM.specific_prognostic_moisture_from_total` (both methods), `AM.materialize_microphysical_fields`, `AM.microphysical_state`, `AM.grid_microphysical_state`, `AM.update_microphysical_fields!`, `AM.update_microphysical_auxiliaries!`, `P3IceProps`, `P3CacheResult`, `z̃ⁱ_tendency` (both methods), `p3_compute_and_cache!`, `AM.moisture_fractions`, `p3_ice_properties`, `p3_rates_and_properties`, `p3_ice_sixth_moment_tendency` (both methods) |
| `p3_microphysical_tendencies.jl` | The eleven `AM.microphysical_tendency(p3::P3, ::Val{…}, …)` dispatches (`:ρnᶜˡ`, `:ρqᶜˡ`, `:ρqʳ`, `:ρnʳ`, `:ρqⁱ`, `:ρnⁱ`, `:ρqᶠ`, `:ρbᶠ`, `:ρz̃ⁱ`, `:ρqʷⁱ`, `:ρsˢᵃᵗ`, `:ρqᵛ`) |
| `p3_driver.jl` | `AM.microphysics_model_update!`, `AM.compute_microphysical_tendencies!` |

### 4. `collection_rates.jl` (826 → 3 files)

| New file | Contents |
|---|---|
| `ice_aggregation_rates.jl` | `ice_rain_collection_lookup`, `ice_aggregation_rate` |
| `riming_rates.jl` | `cloud_riming_rate`, `cloud_warm_collection_rate`, `rain_warm_collection_rate`, `cloud_riming_number_rate`, `rain_riming_rate` (two methods), `rain_riming_mass_kernel`, `rain_riming_number_rate` (two methods), `rain_riming_number_kernel` |
| `wet_ice_processes.jl` | `rime_density` (two methods), `shedding_rate`, `shedding_integral`, `shedding_number_rate`, `wet_growth_capacity`, `refreezing_rate` |

### 5. `fortran_table_reader.jl` (690 → 2 files)

| New file | Contents |
|---|---|
| `fortran_table_format.jl` | All `FORTRAN_*`, `LOG_*` constants, `parse_fortran_line`, `parse_fortran_table_1`, `parse_fortran_table_3`, `make_fortran_tabulated_function`, `make_tabulated_function`, `fortran_table_1_ranges`, `fortran_table_2_ranges`, `fortran_table_3_ranges` |
| `fortran_table_reader.jl` (shrunk) | `read_fortran_lookup_tables`, `build_table_1_functions`, `build_table_2_functions`, `build_table_3_functions`, `assemble_lookup_tables`, `build_ice_properties_from_tables`, `tabulate_rain_from_quadrature` |

### 6. `process_rate_parameters.jl` (501)

**Left unchanged.** Contains one struct, one constructor, one `Base.show`.
Splitting hurts cohesion.

## Module file changes

`PredictedParticleProperties.jl` currently includes 33 files. After this
reorganization it will include 33 + 16 − 2 = **47 files** (16 new focused
files added; `collection_rates.jl` and `p3_interface.jl` are deleted because
their contents move entirely into newly named files; the other three monoliths
keep their original names in shrunk form). The new include list is grouped by
topic with comments. The proposed include order respects the existing
dependency graph (helpers before consumers):

```julia
# --- particle properties ---
include("ice_fall_speed.jl")
include("ice_deposition.jl")
include("ice_bulk_properties.jl")
include("ice_collection.jl")
include("ice_sixth_moment.jl")
include("ice_lambda_limiter.jl")
include("ice_rain_collection.jl")
include("ice_properties.jl")
include("rain_properties.jl")
include("cloud_droplet_properties.jl")
include("transport_properties.jl")
include("size_distribution.jl")
include("psd_corrections.jl")

# --- mass relation, shape closures, lambda solver ---
include("ice_mass_relation.jl")           # NEW (from lambda_solver)
include("ice_shape_closures.jl")          # NEW
include("gamma_moments.jl")               # NEW
include("lambda_solver.jl")               # SHRUNK

# --- lookup tables and tabulation ---
include("lookup_tables.jl")
include("lookup_table_3.jl")
include("tabulated_function_adapters.jl")
include("quadrature.jl")
include("rain_quadrature.jl")
include("table_generation_common.jl")
include("fortran_table_format.jl")        # NEW (from fortran_table_reader)
include("fortran_table_reader.jl")        # SHRUNK

# --- scheme types and GPU adaptation ---
include("p3_scheme.jl")
include("gpu_adaptation.jl")

# --- process-rate parameters ---
include("process_rate_parameters.jl")
include("aerosol_activation.jl")

# --- process rates: helpers, kernels, CCN, sat-adjust ---
include("process_rate_helpers.jl")        # NEW (from process_rates)
include("tabulated_kernels.jl")           # NEW
include("ccn_activation_rates.jl")        # NEW
include("coupled_saturation_adjustment.jl") # NEW

# --- process rates: collection, riming, wet ice ---
include("ice_aggregation_rates.jl")       # NEW (from collection_rates)
include("riming_rates.jl")                # NEW
include("wet_ice_processes.jl")           # NEW

# --- process rates: nucleation, melting, terminal v, rain ---
include("ice_nucleation_rates.jl")
include("melting_rates.jl")
include("terminal_velocities.jl")
include("rain_process_rates.jl")

# --- process-rate aggregate struct + tendencies ---
include("process_rates.jl")               # SHRUNK
include("prognostic_tendencies.jl")       # NEW
include("sixth_moment_tendencies.jl")     # NEW

# --- driver / Oceananigans interface ---
include("multi_ice_category.jl")
include("p3_microphysical_state.jl")      # NEW (from p3_interface)
include("p3_microphysical_tendencies.jl") # NEW
include("p3_driver.jl")                   # NEW
```

The `export` block at the top of `PredictedParticleProperties.jl` is unchanged.

## Mechanics

Each split is a sequence of:

1. Create the new file.
2. Move the function bodies from the source file (preserving docstrings,
   `@inline` annotations, exact text).
3. Remove the moved bodies from the source file.
4. Add the new file to `PredictedParticleProperties.jl` in the slot specified
   above.

Imports do not move into the new files — all `using`/`import` statements stay
at the top of `PredictedParticleProperties.jl`, where they already live.

## Verification

After all splits land:

- `julia --project -e 'using Breeze'` loads without errors.
- `julia --project=test test/runtests.jl` passes (matches the pre-split
  baseline). Per `.agents/testing.md`, P3-specific tests live under
  `test/Microphysics/`.
- `julia --project=docs docs/make.jl` (or whatever the local doctest command
  is) produces the same output as the pre-split baseline.
- `git diff main --stat` shows only file moves (new files + line removals from
  the six monoliths); no logic diffs inside any function body.

A useful spot-check: pick one function in each new file and confirm its body
is byte-identical to its pre-split version (`git log -p -- <path>` should
show only the move).

## Rollout

This is a single PR. The change is mechanical, the diff is large but local,
and there is no good intermediate state. Splitting across PRs would mean
intermediate commits where the same function exists in two files or where
includes are temporarily reordered, both of which are worse than a single
clean diff.

The PR title: `Split P3 monolithic files into focused per-concern files`.

The PR body lists the six files split and the count of new files, points at
this spec, and notes that no API or logic changed.
