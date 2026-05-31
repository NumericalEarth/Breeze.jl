#!/usr/bin/env python3
"""
Extract a Breeze-schema state-slice CSV (`i, k, x, z, u, w_center,
theta_perturbation, pressure_perturbation`) from a single CM1 cm1out_*.nc
frame, downsampled by 2×2 block-averaging from 400×200 → 200×100.

The 200×100 grid uses cells of size 1000 m × 300 m centered at the same
domain centroids as the original CM1 grid. The block-average is a simple
unweighted mean of the four CM1 cell-center values that fall inside each
new cell.
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
    "external_schar_400x200_periodic_theta300_t10800s_state_slice_200x100.csv",
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

# Interpolate to cell centres first (matching the original extractor).
u_cc = 0.5 * (u[:, :-1] + u[:, 1:])    # (zh, xh)
w_cc = 0.5 * (w[:-1, :] + w[1:, :])    # (zh, xh)
theta_pert = th - th0
p_pert = p - p0

# Block-average 2x2 → 1x1.
def block_avg_2x2(a):
    nz, nx = a.shape
    return 0.25 * (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2])

xh_ds = 0.5 * (xh[0::2] + xh[1::2])    # new cell centres
zh_ds = 0.5 * (zh[0::2] + zh[1::2])
u_ds  = block_avg_2x2(u_cc)
w_ds  = block_avg_2x2(w_cc)
theta_ds = block_avg_2x2(theta_pert)
p_ds  = block_avg_2x2(p_pert)

nz, nx = theta_ds.shape  # should be (100, 200)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as h:
    writer = csv.writer(h)
    writer.writerow(["i", "k", "x", "z", "u", "w_center",
                     "theta_perturbation", "pressure_perturbation"])
    for k in range(nz):
        for i in range(nx):
            writer.writerow([
                i + 1, k + 1,
                f"{xh_ds[i]:.7e}", f"{zh_ds[k]:.7e}",
                f"{u_ds[k, i]:.7e}", f"{w_ds[k, i]:.7e}",
                f"{theta_ds[k, i]:.7e}", f"{p_ds[k, i]:.7e}",
            ])

print(f"wrote {out}")
print(f"  time = {time:.0f} s, grid = {nx} x {nz} (downsampled 2×2 from CM1 400×200)")
print(f"  x in [{xh_ds.min():.0f}, {xh_ds.max():.0f}] m  z in [{zh_ds.min():.0f}, {zh_ds.max():.0f}] m")
