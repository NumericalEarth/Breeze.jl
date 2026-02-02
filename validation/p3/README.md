# P3 kin1d reference data

This directory holds a local reference dataset produced from the P3 Fortran
`kin1d` kinematic driver. The goal is to compare Breeze's P3 coupling against
the Fortran reference while the implementation is in progress.

Files
- `kin1d_reference.nc`: NetCDF version of the Fortran `out_p3.dat` output.
- `make_kin1d_reference.jl`: Script that converts `out_p3.dat` to NetCDF.

Case configuration (current reference)
- P3 repo commit: `24bf078ba70cb53818a03ddccc3a95cbb391fcd5`
- Driver: `kin1d/src/cld1d.f90`
- Config: `nCat=1`, `trplMomIce=true`, `liqFrac=true`
- `dt=10 s`, `outfreq=1 min`, `total=90 min`, `nk=41`
- Sounding: `snd_input.KOUN_00z1june2008.data`
- Lookup tables: `p3_lookupTable_1.dat-v6.9-3momI`, `p3_lookupTable_2.dat-v6.2`,
  `p3_lookupTable_3.dat-v1.4`
- Note: `cld1d.f90` has `version_p3 = 'v5.3.14'`, while `P3_INIT` prints `v5.5.0`.
  Both are recorded as NetCDF global attributes.

NetCDF contents
- Dimensions: `time` (seconds), `z` (meters)
- Variables:
  - `w` (time, z): vertical velocity [m s-1]
  - `prt_liq` (time): liquid precip rate [mm h-1]
  - `prt_sol` (time): solid precip rate [mm h-1]
  - `reflectivity` (time, z): radar reflectivity [dBZ]
  - `temperature` (time, z): temperature [C]
  - `q_cloud`, `q_rain`, `q_ice` (time, z): mixing ratios [kg kg-1]
  - `n_cloud`, `n_rain`, `n_ice` (time, z): number mixing ratios [kg-1]
  - `rime_fraction`, `liquid_fraction` (time, z): unitless fractions
  - `drm` (time, z): rain mean volume diameter [m]
  - Category-1 ice diagnostics: `q_ice_cat1`, `q_rime_cat1`,
    `q_liquid_on_ice_cat1`, `n_ice_cat1`, `b_rime_cat1`, `z_ice_cat1`,
    `rho_ice_cat1`, `d_ice_cat1`

The NetCDF variables map directly to the columns written by the Fortran driver.
See the `write(30, ...)` block in `kin1d/src/cld1d.f90` for the authoritative
column definitions.

How to reproduce

1) Build and run the Fortran driver (P3 repo):
```
P3_REPO=/Users/gregorywagner/Projects/P3-microphysics

cd $P3_REPO/lookup_tables
gunzip -k p3_lookupTable_1.dat-v6.9-2momI.gz \
        p3_lookupTable_1.dat-v6.9-3momI.gz \
        p3_lookupTable_2.dat-v6.2.gz \
        p3_lookupTable_3.dat-v1.4.gz

cd $P3_REPO/kin1d/src
ln -s ../soundings soundings
ln -s ../lookup_tables lookup_tables
ln -s ../levels levels

make execld
./execld
```
This produces `out_p3.dat` in `kin1d/src`.

2) Convert `out_p3.dat` to NetCDF:
```
cd /Users/gregorywagner/Projects/alt/Breeze.jl
P3_REPO=/Users/gregorywagner/Projects/P3-microphysics \
/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia \
  --project=validation/p3_env \
  validation/p3/make_kin1d_reference.jl
```

If you change `nk`, `outfreq`, or the sounding, set these environment variables
before running the script:
- `P3_NK` (default 41)
- `P3_OUTFREQ_MIN` (default 1)
- `P3_SOUNDING` (default `snd_input.KOUN_00z1june2008.data`)
