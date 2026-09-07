# Cloud LES library (Shen et al. 2022)

Scripts that turn the [library of GCM-forced large-eddy simulations](https://doi.org/10.22002/D1.20052)
of Shen, Sridhar, Tan, Jaruga & Schneider (2022, *J. Adv. Model. Earth Syst.* 14, e2021MS002631;
CC0) into the `cloud_les_library` artifact that
[`examples/single_column_tke_boundary_layer.jl`](../../examples/single_column_tke_boundary_layer.jl)
evaluates `TKEBasedTurbulenceClosure` against.

The library holds one PyCLES `Stats` file per member — 22 cfSites along the GPCI Pacific transect
(2–4 off Peru, 11–15 in the deep tropics, 17–18 off California, 19–23 across the northeast Pacific),
three GCMs, two climates (`amip`, `amip4K`), four months — of 200–330 MB each: 10-minute horizontal
means, second moments, fluxes and the TKE budget on 200 levels (Δz = 20 m to 4 km) over 3.7 days,
together with everything that forced the LES. The whole library is 149 GB; the CNRM-CM6-1 members
this artifact covers are 165 files and 54 GB.

## What the reduction keeps

Each member reduces to ~0.25 MB of NetCDF:

| Group | Variables | Notes |
|---|---|---|
| Forcing | `ls_subsidence`, `dtdt_hadv`, `dqtdt_hadv`, `dtdt_fluc`, `dqtdt_fluc` | Time-invariant in the LES (asserted); one profile each |
| Radiation | `dtdt_rad_hourly(z, time)` | The LES's RRTM heating, hourly, with its diurnal cycle |
| Nudging | `u_mean_nudge`, `v_mean_nudge`, `thetali_mean_nudge`, `qt_mean_nudge`; `d*dt_nudge` | Whole-run means, the natural relaxation targets; and the tendencies the LES applied |
| Surface | `shf_surface_mean`, `lhf_surface_mean`, `uw/vw_surface_mean`, `surface_temperature`, `friction_velocity_mean`, `obukhov_length_mean`, `lwp`, `cloud_fraction`, `cloud_base`, `cloud_top` | Hourly time series |
| Initial | `*_initial` | Profiles at t = 0 |
| Reference | `p0`, `rho0`, `temperature0`, `qv0` | PyCLES reference state |
| Targets | `*_mean` | Time means over the final two days: state, cloud, fluxes, TKE and its budget |

Every file records its source URL, size and SHA-256, the averaging window, and the DOI.

## Protocol of the LES (Shen et al. 2022, §2)

PyCLES, anelastic, prognostic entropy and total water; 6 km × 6 km × 4 km domain at 75 m × 20 m;
Smagorinsky–Lilly; Kessler warm rain; RRTM with the GCM's insolation; prescribed SST with bulk
surface fluxes; 6 simulated days from the 5-year mean GCM profiles. Horizontal winds are relaxed
to the GCM on 6 h everywhere; free-tropospheric temperature and humidity on 24 h, ramping in with
a half cosine between 3.0 and 3.5 km. The "fluctuation" tendencies are the GCM's vertical eddy
advection. The single-column runs in the example reproduce this protocol with the LES's own
radiative heating, surface fluxes and time-mean profiles as nudging targets, so that the closure
is the only unknown.

## Running

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'

# Download and reduce the CNRM-CM6-1 members, one file at a time (~54 GB of transfer, ~1 h)
julia --project reduce_shen_les_library.jl --gcm CNRM-CM6-1 --output reduced

# A subset, keeping the downloads
julia --project reduce_shen_les_library.jl --sites 17,22 --months 07 --output reduced --download downloads --keep

# Package the reduced directory as the artifact and print the Artifacts.toml entry
julia --project build_artifact.jl reduced cloud_les_library.tar.gz
```

The tarball is uploaded by hand to the Breeze.jl GitHub release named in the printed entry, as
`P3_lookup_tables` was; `Artifacts.toml` at the repository root carries the entry. To use a local
reduction before the release exists, point an
[`Overrides.toml`](https://pkgdocs.julialang.org/v1/artifacts/#Overriding-artifact-locations) at
the directory.
