# MPAS vs Breeze 1D acoustic comparison

## Setup

### MPAS planar hex mesh
- **Grid**: 86×86 periodic planar hex, `dc = 500m` → 7396 cells
- **Vertical**: 40 levels, Δz = 500m, ztop = 20000m (squall line setup)
- **Aspect ratio**: Δx/Δz = 1 (the stability-critical regime)
- **IC**: Horizontally uniform (all cells identical), zero wind, dry
  - θ profile from squall line base state: 300.5K at surface to 489.4K at top
  - ρ from squall line initialization (includes warm bubble residual ρ_p)
  - w = 0, u = 0
- **Time step**: Δt = 6s, N = 6 substeps → Δτ = 1s
- **Parameters**: epssm = 0.1, smdiv = 0.1, len_disp = 500m
- **Physics**: none

### How to reproduce the MPAS run

```bash
# 1. Generate mesh (requires mpas_tools Python package)
cd /tmp/mpas_squall
PYTHONPATH=/tmp/mpas_tools/conda_package python3 -c "
from mpas_tools.planar_hex import make_planar_hex_mesh
make_planar_hex_mesh(nx=86, ny=86, dc=500, nonperiodic_x=False,
                     nonperiodic_y=False, outFileName='grid.nc')
"

# 2. Run init_atmosphere (config_init_case=4, squall line)
./init_atmosphere_model

# 3. Uniformize IC (make all cells identical, zero wind)
python3 uniformize_ic.py   # see script below

# 4. Run atmosphere
./atmosphere_model 2> mpas_debug.txt
```

### Uniformize IC script
```python
from netCDF4 import Dataset
import numpy as np
ds = Dataset('init.nc', 'r+')
state_cell = ['theta', 'rho_zz', 'rho', 'rho_base', 'qv', 'qc', 'qr', 'relhum', 'dss']
for vn in state_cell:
    if vn in ds.variables:
        v = ds.variables[vn]
        if 'nCells' in v.dimensions:
            ci = list(v.dimensions).index('nCells')
            s = [slice(None)]*len(v.dimensions); s[ci]=0
            p = np.array(v[tuple(s)])
            for i in range(v.shape[ci]):
                s2=list(s); s2[ci]=i; v[tuple(s2)]=p
for vn in ['u', 'ru']:
    if vn in ds.variables: ds.variables[vn][:] = 0.0
for vn in ['w', 'rw']:
    if vn in ds.variables: ds.variables[vn][:] = 0.0
ds.variables['xtime'][0] = list('0000-01-01_00:00:00'.ljust(64))
ds.close()
```

### Debug prints added to MPAS
After `!$acc end parallel` at line 2976 of `mpas_atm_time_integration.F`:
```fortran
if (small_step <= 2) then
   write(0,'(A,I3)') 'SUBSTEP=', small_step
   write(0,'(A,ES15.8)') '  dts=', dts
   do k=2,min(10,nVertLevels)
      write(0,'(A,I3,6(A,ES20.13))') '  k=',k, &
         ' rw_p=',rw_p(k,1),' rtheta_pp=',rtheta_pp(k,1),' rho_pp=',rho_pp(k,1), &
         ' cofwz=',cofwz(k,1),' cofwr=',cofwr(k,1),' tend_rw=',tend_rw(k,1)
   end do
end if
```

## MPAS reference data (first 2 substeps of RK stage 1)

### Substep 1 (Δτ = 1s)

| k | rw_p | rtheta_pp | rho_pp | cofwz | cofwr | tend_rw |
|---|------|-----------|--------|-------|-------|---------|
| 2 | -3.043e-04 | 6.967e-05 | 2.247e-07 | 4.289e-01 | 2.697e+00 | -3.177e-04 |
| 3 | -5.085e-04 | 1.884e-04 | 6.079e-07 | 4.219e-01 | 2.697e+00 | -4.571e-04 |
| 4 | -1.061e-03 | -5.842e-05 | -1.899e-07 | 4.155e-01 | 2.697e+00 | -1.163e-03 |
| 5 | -8.885e-04 | -4.698e-05 | -1.533e-07 | 4.095e-01 | 2.697e+00 | -8.844e-04 |
| 6 | -7.492e-04 | -5.894e-05 | -1.917e-07 | 4.032e-01 | 2.697e+00 | -7.546e-04 |
| 7 | -5.749e-04 | -3.978e-05 | -1.295e-07 | 3.969e-01 | 2.697e+00 | -5.679e-04 |
| 8 | -4.572e-04 | -3.283e-05 | -1.067e-07 | 3.905e-01 | 2.697e+00 | -4.549e-04 |
| 9 | -3.602e-04 | -3.409e-05 | -1.101e-07 | 3.840e-01 | 2.697e+00 | -3.610e-04 |
| 10 | -2.602e-04 | -1.544e-05 | -5.017e-08 | 3.775e-01 | 2.697e+00 | -2.534e-04 |

### Substep 2 (Δτ = 1s)

| k | rw_p | rtheta_pp | rho_pp |
|---|------|-----------|--------|
| 2 | -5.835e-04 | 3.122e-04 | 1.008e-06 |
| 3 | -1.128e-03 | 5.866e-04 | 1.892e-06 |
| 4 | -1.843e-03 | -1.350e-04 | -4.402e-07 |
| 5 | -1.757e-03 | -1.778e-04 | -5.802e-07 |
| 6 | -1.483e-03 | -2.131e-04 | -6.933e-07 |
| 7 | -1.169e-03 | -1.556e-04 | -5.065e-07 |
| 8 | -9.232e-04 | -1.282e-04 | -4.164e-07 |
| 9 | -7.209e-04 | -1.237e-04 | -3.997e-07 |
| 10 | -5.394e-04 | -6.712e-05 | -2.175e-07 |

### Key observations
- `cofwr = 2.697` = `dtseps * g/2` = `0.55 * 1 * 9.80616/2` = 2.697 ✓
- `tend_rw` is O(1e-4 to 1e-3) — dominated by the warm bubble residual PGF+buoyancy
- `rw_p` grows from 0 → O(1e-4) after substep 1 → O(1e-3) after substep 2
- `rtheta_pp` is O(1e-5 to 1e-4) — small acoustic compressibility correction
- `rho_pp` is O(1e-7 to 1e-6) — tiny density perturbation

## Breeze comparison setup

Match the MPAS setup on a RectilinearGrid:
```julia
grid = RectilinearGrid(CPU(); size=(6, 6, 40), halo=(5, 5, 5),
                       x=(0, 3000), y=(0, 3000), z=(0, 20000),
                       topology=(Periodic, Periodic, Bounded))
```

- Δx = Δy = 500m, Δz = 500m (matches MPAS)
- Δt = 6s, N = 6 substeps → Δτ = 1s
- epssm = 0.1 (forward_weight = 0.55)
- smdiv = 0.1, len_disp = 500m
- IC: same θ and ρ profile from MPAS init.nc (table in profile.txt)
- Reference state: isothermal T₀ = 288K (squall line default) with discrete Exner integration

Print the SAME quantities at cell (3,3) after each substep:
- rw_p(k), rtheta_pp(k), rho_pp(k) for k=2:10
- cofwz(k), cofwr(k) — should match MPAS exactly
- Gˢw(k) * ρ_face — should match MPAS tend_rw

If these match to ~12 digits, the acoustic solve is identical.
If they diverge, the first level/substep where they differ reveals the bug.

## Files
- `/tmp/mpas_squall/` — MPAS run directory
- `/tmp/mpas_squall/init.nc` — uniformized init file
- `/tmp/mpas_squall/grid.nc` — planar hex mesh (86×86, dc=500m)
- `/tmp/mpas_squall/mpas_debug.txt` — debug output (66 lines)
- `/tmp/mpas_squall/profile.txt` — 1D θ, ρ, ρ_base profile
- `/tmp/mpas_squall/namelist.atmosphere` — run configuration
