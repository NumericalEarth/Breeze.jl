#!/usr/bin/env python3
"""
Six-panel CM1 vs Breeze-SLEVE comparison at the Schär mountain-wave validation
final time. Layout matches `schar_cm1_vs_breeze_3h_six_panel.png`:

    [ CM1 w ]  [ Breeze w ]  [ Breeze − CM1 w ]
    [ CM1 p' ] [ Breeze p' ] [ Breeze − CM1 p' ]

Reads the `state_slice.csv` produced by `terrain_schar_mountain_wave_validation.jl`
(both CM1 reference and Breeze share the same column schema:
`i, k, x, z, u, w_center, theta_perturbation, pressure_perturbation`).

Env knobs:
    SCHAR_CM1_STATE      path to CM1 state-slice CSV
    SCHAR_BREEZE_STATE   path to Breeze state-slice CSV
    SCHAR_OUTPUT_PNG     destination .png path
    SCHAR_BREEZE_LABEL   Breeze panel column header (default: "Breeze SLEVE")
    SCHAR_TIME_HOURS     time label for the figure title (default: 3.00)
    SCHAR_SPONGE_BASE_KM dashed-line z (default: 20)
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

root = Path.cwd()
cm1_path = root / os.environ.get(
    "SCHAR_CM1_STATE",
    "validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference/"
    "external_schar_400x200_periodic_theta300_t10800s_state_slice.csv",
)
breeze_path = root / os.environ.get(
    "SCHAR_BREEZE_STATE",
    "validation_output/substepper/terrain_schar_3h_400x200_sleve_outside/"
    "terrain_schar_mountain_wave_state_slice.csv",
)
output_png = root / os.environ.get(
    "SCHAR_OUTPUT_PNG",
    "validation_output/substepper/schar_cm1_vs_breeze_sleve_400x200_3h_six_panel.png",
)
breeze_label = os.environ.get("SCHAR_BREEZE_LABEL", "Breeze SLEVE")
time_hours = float(os.environ.get("SCHAR_TIME_HOURS", "3.00"))
sponge_base_km = float(os.environ.get("SCHAR_SPONGE_BASE_KM", "20"))


def read_state(path: Path):
    """Return (nx, nz, x_km[nx], z_km[nz], w[nx,nz], p_prime[nx,nz])."""
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    nx = max(int(r["i"]) for r in rows)
    nz = max(int(r["k"]) for r in rows)
    x_km = np.zeros(nx)
    z_km = np.zeros(nz)
    w = np.zeros((nx, nz))
    p = np.zeros((nx, nz))
    for r in rows:
        i = int(r["i"]) - 1
        k = int(r["k"]) - 1
        x_km[i] = float(r["x"]) / 1000.0
        z_km[k] = float(r["z"]) / 1000.0
        w[i, k] = float(r["w_center"])
        p[i, k] = float(r["pressure_perturbation"])
    return nx, nz, x_km, z_km, w, p


cm1_nx, cm1_nz, cm1_x_km, cm1_z_km, cm1_w, cm1_p = read_state(cm1_path)
br_nx, br_nz, br_x_km, br_z_km, br_w, br_p = read_state(breeze_path)

if (cm1_nx, cm1_nz) != (br_nx, br_nz):
    # Interpolate Breeze to CM1 grid along z if needed (x grids agree by construction).
    if cm1_nx == br_nx and cm1_nz != br_nz:
        br_w = np.stack([np.interp(cm1_z_km, br_z_km, br_w[i, :]) for i in range(br_nx)], 0)
        br_p = np.stack([np.interp(cm1_z_km, br_z_km, br_p[i, :]) for i in range(br_nx)], 0)
    else:
        raise SystemExit(
            f"grid mismatch CM1 {cm1_nx}x{cm1_nz} vs Breeze {br_nx}x{br_nz}; "
            "x dimension must match"
        )

# CM1 x grid is centred at zero (-100..100 km); Breeze's is 0..200 km. Shift CM1 to
# Breeze's convention so panel x-axes match.
if cm1_x_km[0] < 0:
    cm1_x_km = cm1_x_km - cm1_x_km[0]

# Errors (Breeze - CM1)
dw = br_w - cm1_w
dp = br_p - cm1_p

# Symmetric colour limits — clip to 95th percentile of |field| so a few rogue
# cells near the boundaries don't compress the rest of the dynamic range.
def vlim(*arrays):
    stacked = np.concatenate([np.abs(a).ravel() for a in arrays])
    return float(np.percentile(stacked, 99.5)) or 1.0


w_lim = vlim(cm1_w, br_w)
p_lim = vlim(cm1_p, br_p)
dw_lim = vlim(dw)
dp_lim = vlim(dp)

# Plot
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True,
                        sharex=True, sharey=True)
fig.suptitle(
    f"Schär mountain wave  ·  t = {time_hours:.2f} h  ·  "
    f"{br_nx}×{br_nz}  ·  {breeze_label} vs CM1",
    fontsize=11,
)

X, Z = np.meshgrid(cm1_x_km, cm1_z_km, indexing="ij")

panels = [
    ("CM1  w", cm1_w, w_lim, "w (m/s)", "RdBu_r"),
    (f"{breeze_label}  w", br_w, w_lim, "w (m/s)", "RdBu_r"),
    (f"{breeze_label} − CM1  w", dw, dw_lim, "Δw (m/s)", "RdBu_r"),
    ("CM1  p′", cm1_p, p_lim, "p′ (Pa)", "RdBu_r"),
    (f"{breeze_label}  p′", br_p, p_lim, "p′ (Pa)", "RdBu_r"),
    (f"{breeze_label} − CM1  p′", dp, dp_lim, "Δp′ (Pa)", "RdBu_r"),
]

for ax, (title, data, lim, cb_label, cmap) in zip(axes.flat, panels):
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    pc = ax.pcolormesh(X, Z, data, cmap=cmap, norm=norm, shading="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(40, 160)
    ax.set_ylim(0, cm1_z_km[-1])
    ax.axhline(sponge_base_km, color="k", linestyle="--", linewidth=0.5)
    fig.colorbar(pc, ax=ax, fraction=0.05, pad=0.02).set_label(cb_label, fontsize=8)

for ax in axes[-1, :]:
    ax.set_xlabel("x (km)")
for ax in axes[:, 0]:
    ax.set_ylabel("z (km)")

fig.text(0.5, -0.01, "dashed line: sponge base (z = {:.0f} km)".format(sponge_base_km),
         ha="center", fontsize=8)

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, dpi=150, bbox_inches="tight")
print(f"wrote {output_png}")
