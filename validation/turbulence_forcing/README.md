# Forced homogeneous isotropic turbulence

Statistically stationary turbulence in a triply-periodic box of side 2π, forced by
`ConstantPowerBandForcing` (the continuous-time form of the NCAR parcel-DNS forcing) or
`StochasticBandForcing` (Eswaran and Pope, 1988) at ε = 0.1, with `BoussinesqDynamics`,
second-order centered advection, and ν = 0.02. This is rung L1b of the Breeze vs. NCAR
parcel-DNS comparison.

```sh
julia --project=validation/turbulence_forcing validation/turbulence_forcing/forced_turbulence.jl 32 constant GPU 100
julia --project=validation/turbulence_forcing validation/turbulence_forcing/benchmark_band_forcing.jl
```

## Results (A100, Float64, statistics over 50 ≤ t ≤ 100)

| forcing      | N   | ⟨D⟩ / ε | (⟨D⟩ + ΔE/τ) / ε | u′    | Re_λ | k_max η | ⟨u²⟩ : ⟨v²⟩ : ⟨w²⟩    |
|--------------|-----|---------|-------------------|-------|------|---------|-----------------------|
| constant     | 32³ | 1.007   | 0.999             | 0.626 | 33.9 | 1.51    | 0.94 : 0.99 : 1.07    |
| stochastic   | 32³ | 0.887   | 0.911             | 0.576 | 30.5 | 1.56    | 0.97 : 0.98 : 1.05    |
| constant     | 64³ | 1.000   | 1.000             | 0.616 | 32.9 | 3.03    | 0.85 : 1.01 : 1.14    |

The dissipation rate is ν⟨|∇u|²⟩ with the second-order differences of the viscous term.
With T_f = 0.5, which is not short compared with the eddy turnover time, the stochastic
forcing injects about 0.9 ε.

## Cost (A100)

Computing the amplitudes of `ConstantPowerBandForcing` with partial Fourier transforms takes
0.20, 0.43, and 0.92 ms per Runge-Kutta stage at 64³, 128³, and 256³ (Float64), compared with
0.27, 1.3, and 10.7 ms for three FFT-based Poisson solves (one forward and one inverse 3D FFT each),
which is a lower bound for an FFT-based implementation.
