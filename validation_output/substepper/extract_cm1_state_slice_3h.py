#!/usr/bin/env python3
"""
Extract a Breeze-schema state-slice CSV (`i, k, x, z, u, w_center,
theta_perturbation, pressure_perturbation`) from a single CM1 cm1out_*.nc
frame. Defaults to t = 10800 s (3 h) using the existing 6 h CM1 reference.

The CM1 grid has `u` at x-faces (xf) and `w` at z-faces (zf); we interpolate
both to cell centers to match the Breeze `w_center` / `u` convention.

CM1 horizontal coordinates are stored in km; we convert to metres in the CSV
output for direct comparison with Breeze state slices.
"""

import csv
import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

root = Path.cwd()
src = root / os.environ.get(
    "CM1_NC",
    "validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference/cm1out_000019.nc",
)
out = root / os.environ.get(
    "CM1_STATE_CSV",
    "validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference/"
    "external_schar_400x200_periodic_theta300_t10800s_state_slice.csv",
)

with Dataset(src) as ds:
    xh = np.array(ds.variables["xh"][:]) * 1000.0    # km → m
    zh = np.array(ds.variables["zh"][:]) * 1000.0    # km → m
    u  = np.array(ds.variables["u"][0, :, 0, :])     # (zh, xf)
    w  = np.array(ds.variables["w"][0, :, 0, :])     # (zf, xh)
    th = np.array(ds.variables["th"][0, :, 0, :])    # (zh, xh)
    th0 = np.array(ds.variables["th0"][0, :, 0, :])
    p  = np.array(ds.variables["prs"][0, :, 0, :])
    p0 = np.array(ds.variables["prs0"][0, :, 0, :])
    time = float(ds.variables["time"][0])

nz, nx = th.shape

# Interpolate u (x-face) and w (z-face) to cell centres.
u_cc = 0.5 * (u[:, :-1] + u[:, 1:])         # (zh, xh)
w_cc = 0.5 * (w[:-1, :] + w[1:, :])         # (zh, xh)
theta_pert = th - th0                       # (zh, xh)
p_pert = p - p0                             # (zh, xh)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as h:
    writer = csv.writer(h)
    writer.writerow(["i", "k", "x", "z", "u", "w_center",
                     "theta_perturbation", "pressure_perturbation"])
    for k in range(nz):
        for i in range(nx):
            writer.writerow([
                i + 1, k + 1,
                f"{xh[i]:.7e}", f"{zh[k]:.7e}",
                f"{u_cc[k, i]:.7e}", f"{w_cc[k, i]:.7e}",
                f"{theta_pert[k, i]:.7e}", f"{p_pert[k, i]:.7e}",
            ])

print(f"wrote {out}")
print(f"  time = {time:.0f} s, grid = {nx} x {nz}, "
      f"x in [{xh.min():.0f}, {xh.max():.0f}] m, z in [{zh.min():.0f}, {zh.max():.0f}] m")
