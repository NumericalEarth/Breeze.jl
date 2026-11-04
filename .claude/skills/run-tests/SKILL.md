---
name: run-tests
description: Run targeted Breeze tests, prioritized by what's likely to break
---

# Run Tests

Run targeted tests one-by-one, prioritized by what's most likely to fail given recent changes.
Never run the full test suite blindly — it's too large. Fix each failure before moving on.

## Step 1: Identify What Changed

Look at the recent changes (staged, unstaged, or recent commits) and determine which test files
are most likely affected. Use this mapping:

| Changed area | Test files to run first |
|---|---|
| `src/AtmosphereModels/` | `atmosphere_model_construction`, `set_atmosphere_model`, `dynamics` |
| `src/AtmosphereModels/atmosphere_model.jl` | `atmosphere_model_construction` |
| `src/AtmosphereModels/update_*.jl` | `dynamics`, `set_atmosphere_model` |
| `src/AtmosphereModels/microphysics_interface.jl` | `cloud_microphysics_1M`, `cloud_microphysics_2M`, `saturation_adjustment` |
| `src/AnelasticEquations/` | `anelastic_pressure_solver_analytic`, `anelastic_pressure_solver_nonhydrostatic`, `dynamics` |
| `src/CompressibleEquations/` | `acoustic_substepping_components`, `acoustic_substepping_open_boundaries`, `acoustic_substepping_stability`, `dynamics` |
| `src/StaticEnergyFormulations/`, `src/PotentialTemperatureFormulations/` | `dynamics`, `set_atmosphere_model`, `moist_state_conversion` |
| `src/Thermodynamics/` | `saturation_adjustment`, `unit_tests`, `reference_states` |
| `src/Solvers.jl` | `solvers`, `saturation_adjustment` |
| `src/TurbulenceClosures/` | `turbulence_closures`, `vertical_diffusion`, `tke_closure` |
| `src/Microphysics/` | `cloud_microphysics_1M`, `cloud_microphysics_2M`, `saturation_adjustment` |
| `src/Microphysics/` (P3) | `predicted_particle_properties_processes`, `p3_reference_process_routing` |
| `src/Advection.jl` | `dynamics`, `tracer_dynamics`, `advection_schemes` |
| `src/Forcings/` | `forcing_and_boundary_conditions`, `geostrophic_subsidence_forcings`, `specific_forcing` |
| `src/BoundaryConditions/` | `forcing_and_boundary_conditions`, `boundary_conditions` |
| `src/TimeSteppers/` | `dynamics`, `runge_kutta_stage_times`, `acoustic_substepping_stability` |
| `src/TerrainFollowingDiscretization/` | `terrain_following_metrics`, `terrain_following_reference_state`, `terrain_following_split_explicit` |
| `src/KinematicDriver/` | `kinematic_driver` |
| `src/ParcelModels/` | `parcel_dynamics` |
| `src/single_column_mode.jl` | `single_column_mode` |
| `src/Breeze.jl` (exports) | `quality_assurance` |
| Docstrings | `doctests` |
| Radiation (`ext/`) | `gray_radiative_transfer`, `clear_sky_radiative_transfer`, `all_sky_radiative_transfer`, `radiation_scheduling` |

Every `test/<name>.jl` is a valid `test_args` name; `ls test/` lists them all.

## Step 2: Run the Most Likely Test First

```sh
julia --project -e '
using Pkg
ENV["CUDA_VISIBLE_DEVICES"] = "-1"
Pkg.test("Breeze"; test_args=`atmosphere_model_construction`)
'
```

## Step 3: Fix and Iterate

1. If a test fails, fix the issue
2. Re-run the same test to confirm the fix
3. Move on to the next most likely test
4. After direct tests pass, run `quality_assurance` to catch import/doctest issues

## Notes

- GPU tests may fail with "dynamic invocation error" — always test on CPU first
- `quality_assurance` checks explicit imports and Aqua.jl quality — run this for any change
- Reactant tests require `--check-bounds=auto` and Julia < 1.12
- If Julia version issues arise, delete `Manifest.toml` and run `Pkg.instantiate()`
