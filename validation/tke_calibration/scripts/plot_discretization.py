"""Plot exported numerical-convergence profiles. Requires numpy and matplotlib."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
with args.input.open() as stream:
    rows = list(csv.DictReader(stream))
settings = sorted({float(r["setting"]) for r in rows}, reverse=True)
reference = float(rows[0]["reference"])
cases = [(22, "07"), (2, "01")]
variables = [("theta_l", "Liquid-water potential temperature (K)"),
             ("total_water", "Nonprecipitating water (g/kg)"),
             ("cloud_water", "Cloud liquid water (g/kg)")]
colors = ["#D97732", "#7353A6", "#007F87", "#2E5F9A"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180,
                     "axes.titleweight": "bold", "font.family": "DejaVu Sans"})

def select(site, month, variable, setting):
    data = [r for r in rows if int(r["site"]) == site and r["month"] == month
            and r["variable"] == variable and float(r["setting"]) == setting]
    return sorted(data, key=lambda r: float(r["z"]))

fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharey=True, constrained_layout=True)
for column, (site, month) in enumerate(cases):
    cloud = select(site, month, "cloud_water", reference)
    cloud_z = [float(r["z"]) / 1000 for r in cloud if float(r["target"]) > .001]
    for row, (variable, xlabel) in enumerate(variables):
        ax = axes[row, column]
        for setting, color in zip(settings, colors):
            data = select(site, month, variable, setting)
            ax.plot([float(r["profile"]) for r in data], [float(r["z"]) / 1000 for r in data],
                    color=color, lw=2, label=f"{setting:g} s")
        data = select(site, month, variable, reference)
        ax.plot([float(r["target"]) for r in data], [float(r["z"]) / 1000 for r in data],
                color="#20262E", lw=2, ls="--", label="LES target")
        if cloud_z:
            ax.axhspan(min(cloud_z) - .05, max(cloud_z) + .05, color="#98BCC5", alpha=.15, zorder=-1)
        ax.set(xlabel=xlabel, ylim=(0, 3))
        ax.grid(axis="y", alpha=.18)
        if column == 0:
            ax.set_ylabel("Height (km)")
        if row == 0:
            ax.set_title(f"Site {site} · month {month}", loc="left", pad=12)
    axes[0, column].legend(frameon=False, fontsize=9)
fig.suptitle("The timestep changes the cloud-layer structure", fontsize=19, fontweight="bold")
args.output.mkdir(parents=True, exist_ok=True)
for extension in ("png", "pdf", "svg"):
    fig.savefig(args.output / f"timestep_profiles.{extension}", facecolor="white")
plt.close(fig)

comparisons = [s for s in settings if s != reference]
fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, constrained_layout=True)
for ax, (site, month) in zip(axes, cases):
    image_columns, labels = [], []
    for setting in comparisons:
        for variable, _ in variables:
            data = select(site, month, variable, setting)
            image_columns.append([float(r["normalized_difference"]) for r in data])
            labels.append(f"{setting:g} s\n" + {"theta_l": "temperature", "total_water": "water", "cloud_water": "cloud"}[variable])
    matrix = np.array(image_columns).T
    im = ax.imshow(matrix, origin="lower", aspect="auto", extent=(-.5, len(labels)-.5, 0, 3),
                   cmap="RdBu_r", vmin=-16, vmax=16, interpolation="nearest")
    ax.set(xticks=range(len(labels)), xticklabels=labels, title=f"Site {site} · month {month}")
axes[0].set_ylabel("Height (km)")
fig.colorbar(im, ax=axes, label="Profile difference / observation noise", shrink=.85)
fig.suptitle(f"Where refinement matters: differences from {reference:g} s", fontsize=17, fontweight="bold")
for extension in ("png", "pdf", "svg"):
    fig.savefig(args.output / f"timestep_difference.{extension}", facecolor="white")
plt.close(fig)
print(args.output)
