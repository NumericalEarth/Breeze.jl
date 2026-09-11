#!/usr/bin/env python3
"""Plot exports from export_closure_diagnostics.jl; no SCM integration or inferred LES budget.

python scripts/plot_closure_diagnostics.py export_dir [--members 22/07,14/01] [--top 3000]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--members", help="Comma-separated site/month pairs; default: all saved cases")
    parser.add_argument("--top", type=float, default=3000)
    args = parser.parse_args()
    with (args.directory / "profiles.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (args.directory / "scores.csv").open() as stream:
        scores = list(csv.DictReader(stream))
    context = (args.directory / "context.txt").read_text().splitlines()
    caption = context[0]
    if "stop-time override: nothing" not in context[1]:
        caption = "SHORT INTEGRATION CHECK — " + caption
    models = list(dict.fromkeys(row["model"] for row in rows if row["model"] != "LES"))
    members = list(dict.fromkeys((row["site"], row["month"]) for row in rows))
    if args.members:
        requested = [tuple(item.split("/")) for item in args.members.split(",")]
        absent = set(requested) - set(members)
        if absent:
            raise ValueError(f"Cases not present in saved diagnostics: {sorted(absent)}")
        members = requested
    if not members or not models or args.top <= 0:
        raise ValueError("Need saved models, cases, and a positive plot top")
    colors = dict(zip(models, ["#687782", "#d97930", "#007f83", "#8253a5", "#c64d75"]))
    if len(models) > len(colors):
        raise ValueError("At most five model curves can be compared legibly")
    colors["LES"] = "#18232b"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#243746", "text.color": "#243746",
                         "figure.facecolor": "#fbfaf7", "axes.facecolor": "#fbfaf7",
                         "savefig.facecolor": "#fbfaf7"})

    def profile(model, member, variable):
        chosen = [row for row in rows if row["model"] == model and
                  (row["site"], row["month"]) == member and row["variable"] == variable]
        if not chosen:
            raise ValueError(f"Missing {model} {member} {variable}")
        chosen.sort(key=lambda row: float(row["z_m"]))
        z = np.array([float(row["z_m"]) for row in chosen])
        values = np.array([float(row["value"]) for row in chosen])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite {model} {member} {variable}")
        return z, values

    def draw(axis, member, variable, factor=1, les=False):
        for model in models + (["LES"] if les else []):
            z, values = profile(model, member, variable)
            mask = z <= args.top
            axis.plot(values[mask] * factor, z[mask] / 1000, color=colors[model],
                      lw=1.8 if model == "Default" else 2.1,
                      ls="--" if model == "LES" else "-", label=model)

    def format_axis(axis, member, xlabel):
        axis.set_ylim(0, args.top / 1000)
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.15)
        z, cloud = profile("LES", member, "ql")
        cloudy = z[(cloud > 1e-6) & (z <= args.top)]
        if cloudy.size > 1:
            axis.axhspan(cloudy.min() / 1000, cloudy.max() / 1000,
                         color="#80bac5", alpha=0.12, zorder=0)

    def save(figure, name):
        for extension in ("png", "pdf", "svg"):
            figure.savefig(args.directory / f"{name}.{extension}", dpi=180)
        plt.close(figure)

    for member in members:
        site, month = member
        figure, axes = plt.subplots(2, 3, figsize=(12.4, 9), sharey=True)
        panels = [("theta_l", r"$\theta_l$ (K)", 1, True),
                  ("qt", "Nonprecipitating water (g/kg)", 1000, True),
                  ("ql", "Cloud liquid water (g/kg)", 1000, True),
                  ("e", r"Specific TKE (m$^2$/s$^2$)", 1, False),
                  ("mixing_length", "Mixing length (m)", 1, False),
                  ("K_c", r"Scalar diffusivity (m$^2$/s)", 1, False)]
        for axis, (variable, xlabel, factor, les) in zip(axes.flat, panels):
            draw(axis, member, variable, factor, les)
            format_axis(axis, member, xlabel)
        for axis in axes[:, 0]:
            axis.set_ylabel("Height (km)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                      ncol=len(labels), frameon=False)
        figure.suptitle(f"Site {site}, month {month} · structure and turbulent mixing", y=0.985, fontsize=17)
        figure.text(0.5, 0.866, caption, ha="center", fontsize=9)
        paths = []
        for model in models + ["LES"]:
            entry = next(row for row in scores if row["model"] == model and
                         (row["site"], row["month"]) == member and row["variable"] == "cloud_water_path")
            paths.append(f"{model}: {1000 * float(entry['value']):.1f}")
        figure.text(0.07, 0.042, "Cloud water path (g/m², LES domain)   " + "   |   ".join(paths), fontsize=10)
        figure.text(0.07, 0.019, "Shading: vertical extent of LES mean cloud water > 0.001 g/kg. See context.txt for protocol and candidate status.", fontsize=9)
        figure.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.835, wspace=0.25, hspace=0.29)
        save(figure, f"structure_site{site}_month{month}")

        figure, axes = plt.subplots(1, 4, figsize=(14.4, 5.9), sharey=True)
        draw(axes[0], member, "total_water_flux", 1e5)
        format_axis(axes[0], member, "Water flux (10⁻⁵ kg/kg m/s)")
        draw(axes[1], member, "shear_production", 1e4)
        format_axis(axes[1], member, "Shear production (10⁻⁴ m²/s³)")
        draw(axes[2], member, "buoyancy_production", 1e4)
        format_axis(axes[2], member, "Buoyancy production (10⁻⁴ m²/s³)")
        draw(axes[3], member, "dissipation", 1e4)
        format_axis(axes[3], member, "Dissipation magnitude (10⁻⁴ m²/s³)")
        for axis in axes:
            axis.axvline(0, color="#a7afb3", lw=0.7, zorder=0)
        axes[0].set_ylabel("Height (km)")
        figure.suptitle(f"Site {site}, month {month} · transport and selected TKE terms", y=0.985, fontsize=17)
        figure.text(0.5, 0.839, caption, ha="center", fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                      ncol=len(labels), frameon=False)
        figure.text(0.065, 0.058, "Positive water flux is upward; positive buoyancy production feeds TKE. Dissipation removes TKE.", fontsize=10)
        figure.text(0.065, 0.026, "SCM instantaneous terms averaged over the target window. Transport/tendency terms are omitted; this is not a closed TKE budget.", fontsize=9)
        figure.subplots_adjust(left=0.065, right=0.98, bottom=0.19, top=0.80, wspace=0.27)
        save(figure, f"processes_site{site}_month{month}")


if __name__ == "__main__":
    main()
