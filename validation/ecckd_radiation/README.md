# ecCKD radiation on the GPU

GPU cases for `BreezeNumericalRadiationExt`, the ecCKD radiation backend built on
[NumericalRadiation.jl](https://github.com/NumericalEarth/NumericalRadiation.jl). The CPU
validation ladder is in the test suite (`test/numerical_radiation_*.jl`, `test/ecckd_radiative_transfer.jl`):
analytic gray column, bit-level round trip against NumericalRadiation's array path, column-extension
convergence, and the clear-sky and all-sky comparisons against RRTMGP on one column. CI runs those
same files with the grid on the GPU, so kernel correctness is covered at small size. What remains is
scale, duration, and performance. The cases below are in priority order; each names what it checks
and what would count as a problem.

## 1. Radiative convection, as committed

`examples/radiative_convection.jl`: 2D, 128 × 51 stretched to 15 km, Float32, all-sky ecCKD with
`CloudScatteringTables()`, three days, radiation every five minutes, surface temperature a live `Field`
updated by a callback.

This is the only example that time-steps with interactive radiation, and it has run only as a
five-iteration CPU smoke. Run it twice, once as committed and once with `AllSkyOptics()` in place of
`EcCKDOptics(clouds = CloudScatteringTables())`, and compare:

- the OLR progress line over the three days (drift, spikes, NaN);
- the hourly-averaged heating profile `Fᴿ` at the end of day 3;
- the top-of-domain versus surface energy balance once the column is near equilibrium.

Expected: profiles within the clear-sky gates of `test/numerical_radiation_rrtmgp_comparison.jl`
outside cloud, and larger differences at cloud top where the cloud optics differ in kind. A growing
OLR difference or a NaN is a bug.

## 2. Performance against RRTMGP

A 3D `AtmosphereModel` on 32 × 32 × 64, all-sky, radiation every step, timing `update_state!` with
`EcCKDOptics(clouds = CloudScatteringTables())` and with `AllSkyOptics()` after one warm-up step.
Repeat at 128 × 128 × 64.

Expected: time per column flat between the two grids, memory scaling with the number of columns and
never with the g-point count, and a speedup over RRTMGP to quote in the PR. The earlier prototype of
this extension reported about 31× on an H100 at 32 × 32 × 64 with the 32 × 32 tables; a result far
below that means the streaming kernel is not doing what it is meant to.

## 3. A BOMEX-type domain

`examples/bomex.jl`'s 3 km domain with `EcCKDOptics()` in place of the prescribed cooling. First
radiation only: `set!` the model and read the surface downwelling longwave and the top-cell heating
against the numbers recorded in `test/numerical_radiation_column_extension.jl`. Then a few hours
interactively.

This is where the column extension above the grid top matters, and where the RRTMGP backend is
wrong by design (zero downwelling longwave at a 3 km top). If it behaves, BOMEX is the next example
to switch.

## 4. Float32 at scale

On the 32 × 32 × 64 grid of case 2, compare Float32 to Float64 fluxes and heating at one instant.

Expected: about 1 W m⁻² in the fluxes and tenths of a K day⁻¹ in heating. Larger differences point
at accumulation order in the g-point sums.

## 5. Schedule sensitivity

Case 1 with radiation every step, every 5 minutes, and every 10 minutes; compare the mean heating
profiles. This settles the default schedule the examples should use.

## What to watch

The surface emission is evaluated per g-point on the device at every update from the surface
temperature `Field`. The CPU tests exercise this only with a constant surface temperature; cases 1
and 3 are the first with a surface temperature that changes in time.
