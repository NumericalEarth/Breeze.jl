## bomex

BOMEX shallow cumulus, 64×64×75, Δt=10.0s, stop=1800s (shortened from 6h), GPU, Float32, WENO(9), SatAdj, surface fluxes + subsidence + geostrophic + radiation.

| run | elapsed | iters reached | sim time | max|u| | max|w| | NaN? | ok |
|-----|--------:|--------------:|---------:|-------:|-------:|:-----|:---|
| anelastic | 24.92s | 180 | 1800.0s | 9.65 | 4.49 | no | ✓ |
| compressible | 39.44s | 100 | 1000.0s | NaN | NaN | yes | ✓ |

**Slowdown factor (compressible/anelastic): 1.58×**

