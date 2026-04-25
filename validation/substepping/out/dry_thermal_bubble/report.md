## dry_thermal_bubble

2D bubble, 128×128, Δt = 2.0s, stop = 1500.0s, CPU, WENO(9)

| run | elapsed | iters reached | sim time | max|u| | max|w| | NaN? | ok |
|-----|--------:|--------------:|---------:|-------:|-------:|:-----|:---|
| anelastic | 54.40s | 750 | 1500.0s | 34.1 | 24.4 | no | ✓ |
| compressible | 19.46s | 6 | 12.0s | 53.8 | 725 | no | ✗ |

**Compressible run crashed:** DomainError with -12247.067050979173:
Exponentiation yielding a complex result requires a complex argument.
Replace x^y with (x+0im)^y, Complex(x)^y, or similar.

