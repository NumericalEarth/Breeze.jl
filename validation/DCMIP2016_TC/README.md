# DCMIP2016 tropical cyclone validation

End-to-end validation of Breeze.jl against the DCMIP2016 Reed–Jablonowski tropical-cyclone
benchmark (Reed & Jablonowski 2011; Ullrich et al. 2016; Willson et al. 2024). An analytic
balanced vortex in a quiescent moist tropical environment intensifies into a tropical
cyclone over ~10 days under the complete Reed–Jablonowski "simple physics": wind-dependent
surface drag, wind-dependent boundary-layer mixing, and large-scale condensation with
instantaneous rain-out (`InstantaneousPrecipitation`). The rendered study — figures, tables,
and discussion, including the comparison against Willson et al. (2024) — is the deliverable
[`dcmip2016_tc_intercomparison.md`](dcmip2016_tc_intercomparison.md).

These are 10-day global compressible runs on a GPU — they are intentionally **not** part of
the documentation build (the `examples/` are rendered, and run, by `docs/make.jl`; this
`validation/` tree is too expensive for that and is run manually). See the top-level
[`validation/README.md`](../README.md) for the rationale.

## Files

| File | Purpose |
|------|---------|
| `dcmip2016_tc.jl` | Defines `dcmip2016_tropical_cyclone_simulation(; resolution, advection_order, …)`, which builds a fully configured `Simulation`. Running the file directly executes the best configuration (0.25° + WENO9). |
| `dcmip2016_tc_intercomparison.jl` | Reuses the generator for two studies — (1) resolution (0.5° vs 0.25°) × advection order (WENO5 vs WENO9), and (2) vertical level configuration at 0.5° WENO9 — and renders the validation (Literate-formatted). |
| `extract_willson_comparison_data.jl` | Manual post-processing helper: reduces the 0.25° WENO5/WENO9 run fields to azimuthal-mean tangential-wind diagnostics in `postproc/` (matching the TempestExtremes processing Willson et al. applied to the ensemble). |
| `plot_willson_comparison.jl` | Manual post-processing helper: overlays those reductions on the DCMIP2016 ensemble (Willson et al. 2024, Figs. 5/7/8), writing `figures/willson_fig{5,7,8}.png`. |

## Running

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# best configuration (0.25° + WENO9, ≈ 921 hPa, ≈ 37 min on an H100)
julia --project=. dcmip2016_tc.jl

# full intercomparison study — resolution × advection + vertical (≈ 1.75 h on a single H100)
julia --project=. dcmip2016_tc_intercomparison.jl
```

A quicker 0.5° check (≈ 963 hPa, ≈ 8 min) is one keyword away:

```julia
include("dcmip2016_tc.jl")
sim = dcmip2016_tropical_cyclone_simulation(; resolution = 0.5, advection_order = 9)
run!(sim)
```

## Expected results

Minimum sea-level pressure over the 10-day run (deeper = more intense).

**Study 1 — horizontal resolution × advection order:**

| resolution | WENO5 | WENO9 |
|------------|-------|-------|
| **0.5°**   | 975.8 hPa | 963.2 hPa |
| **0.25°**  | 937.6 hPa | 921.4 hPa |

Both finer resolution and higher advection order deepen the storm; resolution is the larger
lever (~38–42 hPa vs ~13–16 hPa).

**Study 2 — vertical level configuration (0.5° WENO9):** doubling the level count uniformly
does not help (it slightly weakens the storm); re-placing levels into the 5–14 km updraft
layer recovers a few hPa. Vertical resolution is a second-order lever — what separates Breeze
from FV3 at equal (50 km) resolution is horizontal effective resolution, not vertical spacing.

See `dcmip2016_tc_intercomparison.jl` for the figures and full discussion.
