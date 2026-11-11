# LASSO-ENA input adapter

The differentiation example uses the default ARM member
`20170718era5d25x100_sbmwrm-aer2-flxsst` ([ARM dataset](https://doi.org/10.5439/2572661)).
Extract its `samin` archive and point `ENA_LASSO_BUNDLE` at the directory containing
`snd`, `lsf`, `sfc`, `prm`, and `grd`. `LassoENA.INPUT_HASHES` pins these five files.
No Covert input or idealized forcing is substituted if files are absent.

```sh
ENA_LASSO_BUNDLE=/path/to/member julia --project=examples examples/ena_differentiability.jl
```

The case epoch is 2017-07-18 00:00 UTC. The input knots span 36 hours, from
2017-07-17 18:00 to 2017-07-19 06:00 UTC. The example spins up for one hour and
then differentiates a four-minute forecast. `ENA_NX`, `ENA_NY`,
`ENA_SPINUP_SECONDS`, and `ENA_FORECAST_STEPS` override those probe dimensions
and durations. The 32×32 default is deliberately smaller than the 256×256
reference simulation; the 100 m spacing and 260-level vertical grid are retained.

The original P3 configuration uses prognostic aerosol and droplet number,
the aer2 aerosol modes and diagnostic CCN-reservoir projection. The 75 cm⁻³
cloud-number initialization is not a fixed-number constraint. There is no
P3 coefficient scaling. Prescribed temperature/moisture advection, full-field
subsidence, geostrophic wind, two-hour mean-wind relaxation, top-boundary
relaxation and sponge follow the source protocol. Ocean heat, moisture and
momentum fluxes respond to SST and the evolving lowest-level state using SAM's
exchange law. ecCKD radiation remains active. The fixed 1 s probe timestep
uses a stateless 30-step radiation schedule, unlike the adaptive reference run.

The forecast restores its absolute clock time and all prognostic fields.
SST and CCN updates execute inside the forecast, including the differentiated
loop. Native/compiled agreement and directional finite differences are checked
before reporting sensitivities. Successful forward construction alone does not
validate these gradients or reproduce the published ARM scores.

The shared operators were adapted from BreezeLab revision
`a8337e25ea3da5cd193f73b7b3402418bb60a00f`; see `NOTICE` and `LICENSE`.

Run the input/forward checks separately from the expensive gradient experiment:

```sh
ENA_LASSO_BUNDLE=/path/to/member julia --project=examples examples/lasso_ena/validate.jl
```
