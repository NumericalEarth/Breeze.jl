## cloudy_thermal_bubble

2D bubble, 128×128 Bounded/Flat/Bounded, Δt = 2.0s, stop = 1000.0s, CPU, WENO(9), Δθ = 2.0K.

| run | elapsed | iters reached | sim time | max|u| | max|w| | NaN? | ok |
|-----|--------:|--------------:|---------:|-------:|-------:|:-----|:---|
| anelastic | 54.43s | 500 | 1000.0s | 9.23 | 14 | no | ✓ |
| compressible | 23.29s | 8 | 17.0s | 289 | 3.39 | no | ✗ |

**Compressible run crashed:** DomainError with -0.20988759637994653:
Exponentiation yielding a complex result requires a complex argument.
Replace x^y with (x+0im)^y, Complex(x)^y, or similar.
