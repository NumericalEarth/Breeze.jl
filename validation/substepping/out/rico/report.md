## rico

RICO precipitating shallow cumulus, 128×128×100, Δt=2.0s, stop=1200s (shortened from 8h), GPU, Float32, WENO(5), OneMomentCloudMicrophysics, surface fluxes, subsidence, geostrophic, radiation, sponge.

| run | elapsed | iters reached | sim time | max|u| | max|w| | NaN? | ok |
|-----|--------:|--------------:|---------:|-------:|-------:|:-----|:---|
| anelastic | 35.46s | 600 | 1200.0s | 9.93 | 0.0678 | no | ✓ |
| compressible | 32.43s | 100 | 200.0s | NaN | NaN | yes | ✓ |

**Slowdown factor (compressible/anelastic): 0.91×**
