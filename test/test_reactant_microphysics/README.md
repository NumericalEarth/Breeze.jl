# Microphysics Differentiability Tests

Tests for Reactant/Enzyme automatic differentiation through Breeze microphysics schemes.

## Setup

- **Grid:** 4x4 doubly-periodic 2D `(Periodic, Periodic, Flat)`
- **Dynamics:** `CompressibleDynamics()` (no FFT pressure solve)
- **Timesteps:** 4 steps at Δt = 0.01 (nsteps is a perfect square for checkpointing)

## Schemes tested

| # | Scheme | Built-in? | Cloud formation | Precipitation |
|---|--------|-----------|----------------|---------------|
| 1 | Dry baseline | Yes | None | None |
| 2 | `SaturationAdjustment` (warm) | Yes | Equilibrium (liquid only) | None |
| 3 | `SaturationAdjustment` (mixed) | Yes | Equilibrium (liquid + ice) | None |
| 4 | `BulkMicrophysics()` (non-precipitating) | Yes | SA (default) | None |
| 5 | `DCMIP2016KesslerMicrophysics` | Yes | Internal SA | Warm rain (kernel) |
| 6 | `ZeroMomentCloudMicrophysics` (0M) | Extension | SA | Instant removal |
| 7 | `OneMomentCloudMicrophysics` (1M NE) | Extension | Non-equilibrium | Rain (tendency) |
| 8 | `OneMomentCloudMicrophysics` (1M SA) | Extension | SA | Rain (tendency) |
| 9 | `TwoMomentCloudMicrophysics` (2M) | Extension | Non-equilibrium | Rain (mass + number) |

## Running

```bash
cd Breeze.jl
julia --project=test test/test_reactant_microphysics/test_microphysics_ad.jl
```

To enable MLIR dumps for debugging, uncomment the lines near the top of `test_microphysics_ad.jl`.

## Known concerns

- **DCMIP2016 Kessler:** GPU kernel with in-place mutation and dynamic subcycling — likely hardest to differentiate
- **SaturationAdjustment:** Secant iteration with variable loop count — may cause tracing issues
- **1M / 2M:** Depend on CloudMicrophysics.jl function differentiability
- **`isnan` guards** in non-equilibrium condensation rate — non-differentiable
- **`max`/`min` clamping** throughout — zero gradient at kinks
