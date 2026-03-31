# Baroclinic wave validation plan

Two-test validation of Breeze compressible dynamics on `LatitudeLongitudeGrid`
against MPAS (~240 km, x1.10242, 26 levels).

## Test 1: Thermal wind balance (no perturbation)

Initialize the DCMIP2016 balanced jet without the localized perturbation.
Run for 5 days. The state should remain steady — any drift indicates
an imbalance in the discrete operators.

### Setup

**Breeze**: Use the DCMIP2016 IC functions but remove the perturbation
from `zonal_velocity`. No reference state subtraction.

```julia
dynamics = CompressibleDynamics(VerticallyImplicitTimeStepping(); surface_pressure = p₀)
```

If VITD doesn't work, fall back to:

```julia
dynamics = CompressibleDynamics(ExplicitTimeStepping(); surface_pressure = p₀)
```

**MPAS**: Modify `x1.10242.init.nc` in Python to zero out the perturbation
in the edge-normal velocity field, or recompile `init_atmosphere` with the
perturbation disabled in `mpas_init_atm_cases.F`.

### Diagnostics

Monitor every hour for 5 days:

| Quantity | How | Acceptable | Problem |
|---|---|---|---|
| max\|v\| | `maximum(abs, v)` | < 0.1 m/s | > 1 m/s |
| max\|w\| | `maximum(abs, w)` | < 0.01 m/s | > 0.1 m/s |
| Δu / u₀ | `(max\|u\| - u₀) / u₀` | < 1e-4 | > 1e-2 |

Compare drift rates between Breeze and MPAS. If Breeze drifts significantly
more than MPAS, investigate:
- Pressure gradient operators on LLG
- Metric term / Coriolis balance for sheared flow
- Hydrostatic consistency of discrete ρ, p, θ initialization

### What this tests

- Thermal wind balance: the pressure gradient, Coriolis, and gravity
  must be consistent on the discrete grid
- Curvature metric terms: must balance Coriolis for the sheared zonal jet
  (stronger test than solid body rotation which has no shear)
- Absence of reference state errors (run without reference state)
- Hydrostatic consistency of the discrete initialization

## Test 2: Full baroclinic wave (with perturbation)

Initialize the full DCMIP2016 state including the exponential u-perturbation
at (20°E, 40°N). Run for 15 days. Compare against MPAS at matching resolution.

### Setup

**Breeze**: 180×85×30 (2°), DCMIP2016 constants via custom `ThermodynamicConstants`,
no reference state. Try VITD first with appropriate Δt (~100-200s if vertical
CFL is removed). If VITD doesn't work, fall back to `ExplicitTimeStepping`
with Δt=2s.

**MPAS**: Already have output from x1.10242 run (~240 km, 26 levels, Δt=900s,
16 days). Output at `/tmp/mpas_jw_coarse/output.nc`.

### Diagnostics

1. **Meridional wind v at mid-level** — side-by-side snapshots at days 5, 7, 9, 12, 15.
   Expect wave-8/9 pattern developing from day 5, nonlinear wave breaking by day 9.

2. **Zonal wavenumber spectrum** of v at 45°N — compare peak wavenumber at days 7, 9, 12.
   Reference: MPAS shows wn 8 (day 7-9), shifting to wn 6 (day 12) as nonlinear
   inverse cascade sets in.

3. **max|v| time series** — growth rate comparison. Expect exponential growth
   from day 4-8, saturation near day 9.

4. **max|u| time series** — jet acceleration. Initial u₀ ≈ 35 m/s, growing to
   50-60 m/s by day 15.

### Success criteria

- Same dominant wavenumber as MPAS (wn 8-9 at day 7-9)
- Growth onset within ±1 day of MPAS
- Amplitude within factor of 2 of MPAS (resolution differences are expected)
- No blow-up or spurious oscillations

### Run order

1. Breeze with `VerticallyImplicitTimeStepping`, Δt chosen by horizontal acoustic
   CFL at highest latitude (Δx ≈ 19 km at 85° → Δt < 56s; with polar filter Δt ~ 200s)
2. If VITD fails or is unstable: Breeze with `ExplicitTimeStepping`, Δt = 2s
3. Compare against existing MPAS coarse output

## Pre-flight checks (run before Test 1)

These are quick offline checks that don't require a full simulation.
Run on CPU, no GPU needed.

### A. Pressure gradient operator on flat LLG

```julia
grid = LatitudeLongitudeGrid(CPU(); size=(10, 10, 10), halo=(3,3,3),
           longitude=(0,360), latitude=(-80,80), z=(0,10000))
# ∂x_z should be zero on a flat (non-terrain-following) grid
@test Oceananigans.Operators.∂x_zᶠᶜᶜ(5, 5, 5, grid) == 0
```

If nonzero, the horizontal PGF has a spurious vertical correction
that would corrupt the thermal wind balance.

### B. Divergence operator on LLG

Set up solid-body-rotation mass flux ρu = ρ₀ cosφ, ρv = 0, ρw = 0.
Compute `divᶜᶜᶜ(grid, ρu, ρv, ρw)`. Should be zero everywhere
(mass is conserved for non-divergent flow). Any nonzero residual
indicates the divergence operator doesn't correctly handle LLG metrics.

### C. Extended solid body rotation

Run solid body rotation (u = u₀ cosφ) for 1000 steps at 2° with Δt=10s.
Record max|v| drift rate per day. Compare with MPAS running the same
solid body rotation. This extends the existing 10-step test to detect
slow-growing imbalances.

### D. Reference state isolation

Run Test 1 (thermal wind, no perturbation) twice:
1. Without reference state: `dynamics = CompressibleDynamics(ExplicitTimeStepping(); surface_pressure=p₀)`
2. With reference state: `dynamics = CompressibleDynamics(ExplicitTimeStepping(); surface_pressure=p₀, reference_potential_temperature=θ_ref)`

Compare drift rates. If (2) drifts more than (1), the reference state
is confirmed as a source of error.

## Convergence test

Run Test 2 at both 4° and 2° resolution. Compare:
- Dominant wavenumber at day 9
- Growth rate (slope of log(max|v|) vs time during days 4-8)

If wavenumber converges toward MPAS's wn 8-9 with increasing resolution,
the issue is numerical diffusion (acceptable). If it stays at wn 6,
there's an operator bug.

### Reference data locations

- MPAS coarse output: `/tmp/mpas_jw_coarse/output.nc` (x1.10242, 17 daily snapshots)
- MPAS fine output: `/tmp/mpas_jw/jw_baroclinic_wave/output.nc` (x1.40962, 17 daily snapshots)
- WRF output: `/tmp/wrf_bwave/wrfout_d01_0001-01-01_00:00:00` (41×81×64 channel, 61 6-hourly snapshots)
- MPAS executables: `/teamspace/studios/this_studio/MPAS-Model/`
- WRF executables: `/teamspace/studios/this_studio/WRF/main/`
