# [Round-off in reduced precision](@id round-off-section)

Each stage of a Runge–Kutta time stepper ends with a sum ``u^{(m)} = u^{(0)} + δ``. In `Float32`
that sum is rounded to the nearest representable number, and any increment ``δ`` smaller than half
the spacing between floats near ``u^{(0)}`` (half an ulp) is lost. For ``ρθ ≈ 100``–``600``, half an
ulp is ``4×10⁻⁶`` to ``3×10⁻⁵``, about the size of one stage's increment from radiative cooling of
``1.5`` K/day at ``Δt = 2.5`` s. The loss has the same sign at every step, so it is a bias, not
noise: the model stops warming or cooling while the radiation scheme keeps reporting the heating
rate ([issue #1122](https://github.com/NumericalEarth/Breeze.jl/issues/1122)).

The round-off treatment of the stage updates is selectable through `timestepper_kwargs`, for both
the [`SSPRungeKutta3`](@ref) stepper of the anelastic core and the [`AcousticRungeKutta3`](@ref)
stepper of the compressible core:

- [`RoundToNearest`](@ref) (default): IEEE rounding, bitwise identical to the original stage update
  and free.
- [`StochasticRounding`](@ref): each stage is evaluated in increment form,
  ``u^{(m)} = u^{(0)} + α \left[(u^{(m-1)} - u^{(0)}) + Δt \, G\right]``, and the final sum is
  rounded to one of its two neighboring floats, choosing each with probability proportional to its
  closeness to the exact sum. On CPUs a `Float32` sum is computed in `Float64`, random bits are
  added below the `Float32` significand and the result is truncated (`WidenedSum`); on GPUs, where
  `Float64` arithmetic is slow or unavailable, and for `Float64`, the rounding error is computed
  with TwoSum (`TwoSum`).
  Every stage is then exact in expectation, at the cost of an ulp-sized random walk. The random
  numbers are a counter-based hash of the seed, the iteration, the stage, the field and the grid
  indices, so there is no random state: runs are bitwise reproducible and restart reproducibly from
  checkpoints. It needs no extra memory.

```jldoctest
using Oceananigans, Breeze

grid = RectilinearGrid(Float32; size=(8, 8, 8), extent=(1000, 1000, 1000))
model = AtmosphereModel(grid; timestepper_kwargs = (; rounding = StochasticRounding(seed = 1)))
model.timestepper.rounding

# output
StochasticRounding(seed=1, summation=WidenedSum())
```

On the compressible core, the Wicker–Skamarock stages of [`AcousticRungeKutta3`](@ref) all restart
from ``Uⁿ``, so only the third stage's rounding reaches ``Uⁿ⁺¹``; that stage's recovery of the
acoustic prognostics and its scalar updates are rounded with the chosen treatment. The
vertically-implicit solves, the pressure correction, and the acoustic perturbation updates stay
round-to-nearest.

Stochastic rounding works in any precision; in `Float64` it is supported but rarely useful, since
increments are resolved. The default leaves `Float64` and `Float32` runs unchanged.

In a radiative-convective equilibrium LES on the anelastic core (liquid-ice potential temperature),
the energy residual over days 10–20 was:

| Precision                                   | Energy residual (W/m²) |
|:--------------------------------------------|-----------------------:|
| `Float64`                                   | +1.77                  |
| `Float32` with a `Float64` pressure solver  | −1.61                  |
| `Float32` with `StochasticRounding`         | +1.78                  |

On the split-explicit compressible core, `Float32` with `StochasticRounding` gave −2.56 W/m² against
−2.59 W/m² in `Float64`. Stochastic rounding cost about 20% in wall time in those runs.
