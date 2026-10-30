# Figures for compensated_storage.jl. Needs CairoMakie and DelimitedFiles.
#   julia --project=<env with CairoMakie> plot_compensated_storage.jl <output dir> [extra figure dir]

using CairoMakie
using DelimitedFiles
using Statistics
using Printf

outdir = get(ARGS, 1, joinpath(@__DIR__, "output"))
extra = get(ARGS, 2, nothing)

function load(path)
    data, header = readdlm(path, ','; header = true)
    columns = Dict(String(h) => data[:, j] for (j, h) in enumerate(vec(header)))
    return [Dict(k => v[i] for (k, v) in columns) for i in 1:size(data, 1)]
end

# The signed-int8 residual runs in compensated_methods.csv saturated (see compensated_storage.jl);
# they are replaced by the unsigned-residual reruns in compensated_uint8.csv.
methods_csv = filter(r -> !occursin("int8", r["method"]), load(joinpath(outdir, "compensated_methods.csv")))
rows = vcat(methods_csv, load(joinpath(outdir, "compensated_uint8.csv")), load(joinpath(outdir, "compensated_interval.csv")))
floor_percent = 1e-3
magnitude(x) = max(abs(x), floor_percent)

set_theme!(fontsize = 15, Axis = (xgridvisible = false, ygridvisible = false))

function savefig(fig, name)
    save(joinpath(outdir, name), fig; px_per_unit = 3)
    extra === nothing || save(joinpath(extra, name), fig; px_per_unit = 3)
end

#####
##### 1. σ error vs storage method, Nsd = 1000, three Δt
#####

methods_rows = filter(r -> r["experiment"] == "methods", rows)
labels = unique(r["method"] for r in methods_rows)
cases = [(1000, 1.0), (1000, 0.1), (1000, 0.01), (100, 0.1), (10000, 0.1)]
colors = Makie.wong_colors()

fig = Figure(size = (900, 820))
ax = Axis(fig[1, 1]; xscale = log10, yticks = (1:length(labels), labels), yreversed = true,
          xlabel = "|Δσ| at 300 m above cloud base [%] (relative to Float64)",
          title = "Spectral width error by stored format / method (3 seeds; Jensen & Nugent 2017 parcel)")
vspan!(ax, floor_percent, 1; color = (:green, 0.08))
vlines!(ax, [1, 10]; color = :gray70, linestyle = :dash)
for (c, (Nsd, Δt)) in enumerate(cases)
    for (j, label) in enumerate(labels)
        sel = filter(r -> r["method"] == label && r["Nsd"] == Nsd && r["dt"] == Δt, methods_rows)
        isempty(sel) && continue
        y = j + (c - 3) * 0.13
        values = magnitude.(Float64[r["dsigma"] for r in sel])
        scatter!(ax, values, fill(y, length(values)); color = (colors[c], 0.9), markersize = 8,
                 marker = Δt == 0.1 && Nsd != 1000 ? :diamond : :circle,
                 label = "N = $Nsd, Δt = $Δt s")
    end
end
axislegend(ax; position = :rb, merge = true, unique = true, framevisible = false)
xlims!(ax, floor_percent / 2, 300)
savefig(fig, "sigma_error_by_method.png")

#####
##### 2. σ inflation vs number of stores, with theory
#####

fig = Figure(size = (1000, 430))
for (p, experiment) in enumerate(("interval", "interval_dt"))
    sel = filter(r -> r["experiment"] == experiment, rows)
    ax = Axis(fig[1, p]; xscale = log10, yscale = log10,
              xlabel = "number of stores per particle (t = 0 → 1500 s)",
              ylabel = "σ inflation  (σ − σ_ref)/σ_ref  [%]",
              title = experiment == "interval" ? "Δt = 0.01 s, store every k steps (k = 1…300)" :
                                                  "store every step, Δt = 0.003…3 s")
    for (c, label) in enumerate(unique(r["method"] for r in sel))
        m = filter(r -> r["method"] == label, sel)
        stores = Float64[r["stores"] for r in m]
        scatter!(ax, stores, magnitude.(Float64[r["dsigma"] for r in m]); color = colors[c], label)
        groups = sort(unique(stores))
        predicted = [mean(100 * (r["sigma_pred"] - r["sigma_ref"]) / r["sigma_ref"] for r in m if r["stores"] == s) for s in groups]
        lines!(ax, groups, magnitude.(predicted); color = colors[c])
    end
    p == 1 && axislegend(ax; position = :lb, framevisible = false, labelsize = 12)
end
Label(fig[0, :], "Markers: experiment (3 seeds). Lines: linear SR-noise theory, σ² = σ_ref² + ⟨Σ ulp² f(1−f) (r/s′)²⟩ / r²"; fontsize = 13)
savefig(fig, "sigma_inflation_vs_stores.png")

#####
##### 3. Pareto: bytes per particle vs σ error
#####

fig = Figure(size = (820, 560))
ax = Axis(fig[1, 1]; yscale = log10, xlabel = "bytes per particle for the radius attribute",
          ylabel = "worst |Δσ| over Δt ∈ {1, 0.1, 0.01} s, N = 1000, 3 seeds [%]",
          title = "Storage cost vs spectral-width error", xticks = 1:8)
hlines!(ax, [1]; color = :gray70, linestyle = :dash)
kind_colors = Dict("plain" => colors[1], "compensated" => colors[2], "deviation" => colors[3],
                   "deviation_mean" => colors[4], "window" => colors[5])
for label in labels
    sel = filter(r -> r["method"] == label && r["Nsd"] == 1000, methods_rows)
    worst = maximum(magnitude(r["dsigma"]) for r in sel)
    bytes = sel[1]["bytes"]
    stalls = any(abs(r["dLWC"]) > 1 for r in sel)
    scatter!(ax, [bytes], [worst]; color = kind_colors[sel[1]["kind"]], markersize = 11,
             marker = stalls ? :xcross : :circle, label = sel[1]["kind"])
    text!(ax, bytes, worst; text = label, fontsize = 10, offset = (6, 0), align = (:left, :center))
end
scatter!(ax, [8.0], [floor_percent]; color = :black, marker = :star5, markersize = 14, label = "Float64 (reference)")
axislegend(ax; position = :rt, merge = true, unique = true, framevisible = false)
xlims!(ax, 1.5, 10)
savefig(fig, "bytes_vs_sigma_error_pareto.png")

#####
##### 4. Theory vs experiment, all SR runs
#####

fig = Figure(size = (560, 520))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "predicted σ inflation [%]",
          ylabel = "measured σ inflation [%]", title = "Linear SR-noise theory vs experiment")
sr = filter(r -> !startswith(r["method"], "Float32") && !occursin("FP32 r", r["method"]), rows)
x = [magnitude(100 * (r["sigma_pred"] - r["sigma_ref"]) / r["sigma_ref"]) for r in sr]
y = [magnitude(r["dsigma"]) for r in sr]
scatter!(ax, x, y; color = (colors[1], 0.5), markersize = 6)
lines!(ax, [floor_percent, 300], [floor_percent, 300]; color = :black, linestyle = :dash)
savefig(fig, "sigma_theory_vs_experiment.png")

@info "figures written to $outdir"
