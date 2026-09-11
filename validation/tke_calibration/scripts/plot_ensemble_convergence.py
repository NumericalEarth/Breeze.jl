#!/usr/bin/env python3
"""Plot slim, compatibility-gated exports from compare_ensemble_sizes.jl.

python scripts/plot_ensemble_convergence.py results/final/ladder --title 'Constant stability functions'
Reads <stem>_iterations.csv and <stem>_parameters.csv; writes standalone PNG/PDF/SVG figures.
"""
import argparse
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    value = row.get(key, "")
    return float(value) if value else np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", type=Path)
    parser.add_argument("--title", default="Ensemble-size investigation")
    args = parser.parse_args()
    rows = read_rows(Path(str(args.stem) + "_iterations.csv"))
    parameters = read_rows(Path(str(args.stem) + "_parameters.csv"))
    if not rows:
        raise ValueError("The iteration table is empty")
    # The exporter gates compatibility. Refuse a manually concatenated CSV that visibly mixes
    # protocols; all variables it exports must continue to describe a single numerical experiment.
    context = []
    for field, label in [("protocol", "Protocol"), ("dt", "dt (s)"),
                         ("radiation_interval", "radiation (s)"), ("cases", "cases")]:
        values = set(row.get(field, "") for row in rows)
        if len(values) != 1:
            raise ValueError(f"CSV mixes {field}: {values}")
        value = next(iter(values))
        if value:
            context.append(f"{label}: {value}")
    runs = defaultdict(list)
    for row in rows:
        runs[row["run"]].append(row)
    for history in runs.values():
        history.sort(key=lambda row: int(row["iteration"]))
        iterations = [int(row["iteration"]) for row in history]
        if len(set(iterations)) != len(iterations):
            raise ValueError("Duplicate run/iteration rows; do not concatenate repeated exports")
    order = sorted(runs, key=lambda name: (int(runs[name][0]["N_ens"]), runs[name][0]["seed"], name))
    sizes = sorted(set(int(runs[name][0]["N_ens"]) for name in order))
    palette = ["#687782", "#258f92", "#d97930", "#8253a5", "#c64d75"]
    colors = {size: palette[i % len(palette)] for i, size in enumerate(sizes)}
    seeds = list(dict.fromkeys(runs[name][0]["seed"] for name in order))
    styles = {seed: ["-", "--", ":", "-."][i % 4] for i, seed in enumerate(seeds)}
    markers = {seed: ["o", "s", "^", "D", "v"][i % 5] for i, seed in enumerate(seeds)}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fbfaf7", "axes.facecolor": "#fbfaf7",
                         "text.color": "#243746"})

    def save(figure, suffix):
        for extension in ("png", "pdf", "svg"):
            figure.savefig(f"{args.stem}_{suffix}.{extension}", dpi=180, facecolor=figure.get_facecolor())
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [1.5, 1]})
    objectives = {}
    for name in order:
        history = runs[name]
        first, last = history[0], history[-1]
        size, seed = int(first["N_ens"]), first["seed"]
        x = np.array([int(row["iteration"]) for row in history])
        mean = np.array([number(row, "mean_objective") for row in history])
        best = np.array([number(row, "best_so_far_mean_objective") for row in history])
        label = f"N={size}, seed {seed or 'unknown'}"
        axes[0].plot(x, mean, color=colors[size], alpha=.23, lw=1)
        axes[0].plot(x, best, styles[seed], color=colors[size], lw=2.1, label=label)
        selected = number(last, "selected_objective")
        if not np.isfinite(selected):
            continue
        objectives[name] = selected
        axes[1].scatter(size, selected, marker=markers[seed], color=colors[size], s=70, zorder=3)
        axes[1].annotate(f"s{seed or '?'}", (size, selected), xytext=(7, 4),
                         textcoords="offset points", fontsize=9)
        selected_iteration = number(last, "selected_iteration")
        axes[0].scatter(selected_iteration, selected, marker=markers[seed], color=colors[size], s=35, zorder=3)
    if not objectives:
        raise ValueError("No directly evaluated selected mean; cannot draw a candidate-convergence figure")
    for seed in seeds:
        if not seed:
            continue
        paired = [(int(runs[name][0]["N_ens"]), objectives[name]) for name in order
                  if runs[name][0]["seed"] == seed and name in objectives]
        if len({size for size, _ in paired}) != len(paired):
            raise ValueError("Duplicate N/seed experiments; compare distinct recorded seeds")
        if len(paired) > 1:
            axes[1].plot(*zip(*paired), color="#a9b2b8", ls=styles[seed], lw=1, zorder=1)
    axes[0].set_xlabel("Forward evaluation / iteration")
    axes[0].set_ylabel("Training objective Φ at evaluated mean coefficients")
    axes[0].set_title("Does the optimizer stop improving?", loc="left", fontsize=13)
    axes[1].set_xlabel("Ensemble members N")
    axes[1].set_ylabel("Best directly evaluated mean objective Φ")
    axes[1].set_title("Does increasing N change the result?", loc="left", fontsize=13)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(sizes, [str(size) for size in sizes])
    if len(sizes) == 1:
        axes[1].set_xlim(sizes[0] / 1.3, sizes[0] * 1.3)
    for axis in axes:
        axis.grid(alpha=.15)
    figure.suptitle(args.title, fontsize=19, y=.98)
    figure.text(.08, .9, " · ".join(context), fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .11),
                  ncol=min(4, len(labels)), frameon=False, fontsize=9)
    figure.text(.08, .072, "Faint curves: current mean objective. Strong curves: best mean evaluated so far. Symbols: selected candidates.", fontsize=9)
    reasons = "; ".join(f"N={runs[name][0]['N_ens']}/s{runs[name][0]['seed'] or '?'}: "
                        f"{runs[name][-1]['stop_reason'] or 'unrecorded'}" for name in order)
    figure.text(.08, .04, reasons, fontsize=8)
    figure.text(.08, .014, "Training diagnostics alone do not establish ensemble-size convergence, held-out skill, or a global optimum.", fontsize=9)
    figure.subplots_adjust(left=.08, right=.96, bottom=.29, top=.82, wspace=.34)
    save(figure, "objectives")

    reference = min(objectives, key=objectives.get)
    by_run = defaultdict(dict)
    for row in parameters:
        if row["run"] not in runs:
            raise ValueError("Parameter table contains a run absent from the iteration table")
        last = runs[row["run"]][-1]
        for field in ("N_ens", "cases", "seed", "protocol", "dt", "radiation_interval", "selected_iteration"):
            if field in row and row[field] != last.get(field, ""):
                raise ValueError(f"Parameter and iteration tables disagree in {field}; regenerate both exports")
        if row["parameter"] in by_run[row["run"]]:
            raise ValueError("Duplicate run/parameter rows")
        by_run[row["run"]][row["parameter"]] = number(row, "selected_value")
    names = list(by_run[reference])
    chosen = [name for name in order if name in objectives]
    if not names or any(set(by_run[name]) != set(names) for name in chosen):
        raise ValueError("Parameter table lacks a complete, common parameter set")
    values = np.array([[by_run[run][name] for run in chosen] for name in names])
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Log-ratio coefficient figure requires finite positive coefficients")
    baseline = np.array([by_run[reference][name] for name in names])
    ratios = np.log2(values / baseline[:, None])
    limit = max(float(np.max(np.abs(ratios))), .1)
    figure, axis = plt.subplots(figsize=(max(10, .9 * len(chosen) + 4), max(5.4, .32 * len(names) + 2.4)))
    heatmap = axis.imshow(ratios, cmap="BrBG", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_yticks(range(len(names)), names)
    axis.set_xticks(range(len(chosen)), [f"N={runs[name][0]['N_ens']}\ns{runs[name][0]['seed'] or '?'}" for name in chosen])
    for i in range(len(names)):
        for j in range(len(chosen)):
            axis.text(j, i, f"{2 ** ratios[i, j]:.2g}×", ha="center", va="center", fontsize=9,
                      color="white" if abs(ratios[i, j]) > .65 * limit else "#243746")
    bar = figure.colorbar(heatmap, ax=axis, pad=.04, shrink=.8)
    bar.set_label("log₂(coefficient / reference coefficient)")
    figure.suptitle(textwrap.fill(args.title + " · coefficient agreement", width=85), fontsize=16, y=.985)
    figure.text(.13, .90, textwrap.fill(f"Reference: {reference}, lowest directly evaluated mean objective in this export", width=110), fontsize=9)
    figure.text(.13, .045, "Ratios compare selected coefficients; ensemble spread is not a posterior uncertainty.", fontsize=9)
    figure.text(.13, .019, "Read alongside response differences and stability-function combinations; different coefficients may compensate.", fontsize=9)
    figure.subplots_adjust(left=.14, right=.94, bottom=.15, top=.86)
    save(figure, "coefficients")


if __name__ == "__main__":
    main()
