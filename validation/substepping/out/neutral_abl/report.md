## neutral_abl

Moeng-Sullivan shear-driven neutral ABL, 96³, Δt=0.5s, stop=600s (shortened from 5h), GPU, Float32, WENO(9), SmagorinskyLilly, capping inversion + geostrophic.

| run | elapsed | iters reached | sim time | max|u| | max|w| | NaN? | ok |
|-----|--------:|--------------:|---------:|-------:|-------:|:-----|:---|
| anelastic | 36.60s | 1200 | 600.0s | 15.1 | 0.252 | no | ✓ |
| compressible | 37.51s | 100 | 50.0s | NaN | NaN | yes | ✓ |

**Slowdown factor (compressible/anelastic): 1.02×**
