"""Show whether a calibration gain survives changed numerical settings, using exported RMSEs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--resolution", default="50")
args = parser.parse_args()
with args.input.open() as stream:
    rows = [row for row in csv.DictReader(stream) if row["resolution"] == args.resolution]
settings = sorted({(float(r["dt"]), float(r["radiation_interval"])) for r in rows}, reverse=True)
if len(settings) != 2:
    parser.error("Choose a resolution present at exactly two numerical settings")
models = list(dict.fromkeys(r["model"] for r in rows))
noise = {"θˡ": .25, "qᵗ": .25, "qˡ": .1, "u": .5, "v": .5}
values = defaultdict(list)
cases = defaultdict(set)
for row in rows:
    key = ((float(row["dt"]), float(row["radiation_interval"])), row["model"], row["variable"])
    values[key].append(.5 * (float(row["rmse"]) / noise[row["variable"]]) ** 2)
    cases[key].add((row["site"], row["month"]))
if len({frozenset(case_ids) for case_ids in cases.values()}) != 1:
    parser.error("The evaluated cases differ between comparisons")
scores = np.array([[np.mean([np.mean(values[setting, model, variable]) for variable in noise])
                    for model in models] for setting in settings])
labels = ["Default" if m.startswith("default") else m.replace("_ens", " · N=").replace("_mem", " · cases=") for m in models]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(models))
for k, ((dt, radiation), color) in enumerate(zip(settings, ("#8BA5B7", "#007F87"))):
    bars = axes[0].bar(x + (k - .5) * .36, scores[k], width=.36, color=color,
                       label=f"Δt {dt:g} s · radiation {radiation / 60:g} min")
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
axes[0].set(xticks=x, xticklabels=labels, ylabel="Validation objective Φ · lower is better",
            ylim=(0, 1.25 * scores.max()), title="The same coefficients can lose their advantage")
axes[0].tick_params(axis="x", labelrotation=20)
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(axis="y", alpha=.15)
ratios = scores / scores[:, :1]
for i, model in enumerate(models[1:], start=1):
    color = ("#007F87", "#C36335", "#7353A6")[(i-1) % 3]
    axes[1].plot([0, 1], ratios[:, i], "o-", lw=2.5, color=color, label=labels[i])
axes[1].axhline(1, color="#243D50", ls="--", lw=1.3)
axes[1].set(xticks=[0, 1], xticklabels=[f"{dt:g} s / {rad / 60:g} min" for dt, rad in settings],
            ylabel="Objective / default at the same numerical settings",
            title="A gain must survive numerical refinement", ylim=(0, max(1.5, ratios.max()*1.2)))
axes[1].text(.02, 1.02, "Default performance", fontsize=9, color="#526574")
axes[1].legend(frameon=False, loc="lower right", fontsize=9)
axes[1].grid(axis="y", alpha=.15)
fig.suptitle("Calibration can compensate for numerical error", fontsize=20, fontweight="bold")
fig.supxlabel(f"Unchanged exploratory coefficients · common validation sites 3, 12, 21 · {args.resolution} m grid", fontsize=10)
args.output.mkdir(parents=True, exist_ok=True)
for extension in ("png", "pdf", "svg"):
    fig.savefig(args.output / f"design_numerics.{extension}", facecolor="white")
print(args.output)
