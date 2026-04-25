## dry_thermal_bubble_wizard

2D dry thermal bubble, 128×128, adaptive Δt via wizard (cfl=0.3 compressible, 0.3 anelastic for apples-to-apples), stop=1500s, CPU, WENO(9). Compressible uses PressureProjectionDamping(0.5) + forward_weight=0.8.

| run | elapsed | iters reached | sim time | max|u| | max|w| | NaN? | ok |
|-----|--------:|--------------:|---------:|-------:|-------:|:-----|:---|
| anelastic | 146.09s | 2187 | 1500.0s | 32.9 | 24.6 | no | ✓ |
| compressible | 502.33s | 2385 | 1500.0s | 49.7 | 23.2 | no | ✓ |

**Slowdown factor (compressible/anelastic): 3.44×**

