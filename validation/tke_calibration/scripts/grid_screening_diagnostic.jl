# Would fitting on a cheap subset of the grids select the same coefficients as fitting on all of them?
#
# A multi-resolution calibration concatenates the grids: the forward map is [G₅₀; G₁₀₀; G_hindcast] and
# the observations are the same targets repeated, so with 56 cases and three grids the observation
# vector is 25,200 long. That costs twice over — the forward map scales with the cells summed over the
# grids, and the EKI update scales with the *cube* of the observation length. Screening the early
# iterations on one cheap grid and refining on all three would cut both.
#
# The necessary condition is that the cheap objective ranks the ensemble the way the full one does. If
# it does not, screening actively selects the wrong members and the schedule is dead — and that can be
# established with no new integration at all, because every iteration's forward map is stored in the
# checkpoint and each grid's block is a slice of it.
#
# WHAT THIS CANNOT DO. It compares objectives on a *fixed* ensemble. A real coarse-to-fine run would
# take different update steps from the first iteration onward, so its trajectory would diverge from
# this one. Agreement here does not prove a transferred optimum works — only a refit does. Disagreement
# does disprove it. Treat a pass as a licence to spend GPU time on the refit, not as the result.
#
#     julia --project scripts/grid_screening_diagnostic.jl results/final/constant/n400_seed1.jld2
#
# Reading a checkpoint of a running job is safe — they are written to `.tmp` and renamed, so a reader
# sees a complete version — but the whole history is loaded, which is N_obs × N_ens × 8 bytes per
# iteration (81 MB per iteration at 25,200 × 400), and the answer is as of whatever iteration was on
# disk when it was read.
using JLD2, Printf, Statistics

path = first(filter(a -> !occursin('=', a), ARGS))
options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
iterations_wanted = haskey(options, "iterations") ? parse.(Int, split(options["iterations"], ',')) : nothing
quantile_fraction = parse(Float64, get(options, "top", "0.1"))

saved = load(path)
z_faces = saved["z_faces"]
length(z_faces) > 1 || error("$path was calibrated on a single grid ($(length(z_faces[1]) - 1) cells); " *
                             "there is no grid subset to screen on")
members, variables = saved["members"], saved["variables"]
n_cells = length(saved["observation_faces"]) - 1
block = length(members) * length(variables) * n_cells
y, σ = saved["y"], sqrt.(saved["Γ"])
length(y) == block * length(z_faces) ||
    error("Observation vector is $(length(y)) long; $(length(z_faces)) grids × $block per grid = $(block * length(z_faces))")

cells = [length(z) - 1 for z in z_faces]
names = [get(Dict(114 => "50 m", 68 => "100 m", 40 => "hindcast"), c, "grid$i ($c cells)")
         for (i, c) in enumerate(cells)]
ranges = [(r - 1) * block + 1 : r * block for r in eachindex(z_faces)]

# Every non-empty subset of the grids, cheapest first. Enumerated by bit mask rather than with
# Combinatorics, which is not a dependency of this project and must not become one for a diagnostic.
subsets = sort([[r for r in eachindex(z_faces) if (mask >> (r - 1)) & 1 == 1]
                for mask in 1:(2^length(z_faces) - 1)], by = s -> sum(cells[s]))

"""Φ = ½⟨((G − y)/σ)²⟩ over the observations of the grids in `s`, for every ensemble member."""
function objectives(G, s)
    rows = vcat([collect(ranges[r]) for r in s]...)
    yₛ, σₛ = y[rows], σ[rows]
    return [mean(abs2, (view(G, rows, i) .- yₛ) ./ σₛ) / 2 for i in axes(G, 2)]
end

"""Spearman's ρ, on the members where both objectives are finite."""
function rank_correlation(a, b)
    keep = findall(i -> isfinite(a[i]) && isfinite(b[i]), eachindex(a))
    length(keep) > 2 || return NaN
    rank(x) = invperm(sortperm(x[keep]))
    ra, rb = rank(a), rank(b)
    return cor(Float64.(ra), Float64.(rb))
end

history = saved["history"]
wanted = isnothing(iterations_wanted) ? eachindex(history) :
         [i for i in eachindex(history) if history[i].iteration in iterations_wanted]

@printf "%s\n%d grids %s, %d cases, %d ensemble members, %d iterations stored\n" path length(z_faces) string(names) length(members) size(history[1].G, 2) length(history)
println("Φ is the objective the inversion minimizes; the full objective is over all $(length(z_faces)) grids.\n")

# Cost is reported as an ESTIMATE. The forward map is assumed proportional to the cells summed over the
# grids, which has not been measured — scripts/time_forward_map.jl measures it per configuration. The
# update's N_obs³ scaling rests on the measured 150-172 s host solve at 25,200 observations.
println("Estimated relative cost of screening on each subset (forward map ∝ Σ cells is UNMEASURED;")
println("update ∝ N_obs³ from the measured 150-172 s solve at 25,200 observations):")
@printf "  %-22s %7s %9s %12s %14s\n" "subset" "cells" "N_obs" "map (est.)" "update (est.)"
for s in subsets
    @printf "  %-22s %7d %9d %11.2f× %13.2f×\n" join(names[s], "+") sum(cells[s]) (block * length(s)) (sum(cells) / sum(cells[s])) (length(z_faces) / length(s))^3
end

for i in wanted
    h = history[i]
    full = objectives(h.G, collect(eachindex(z_faces)))
    finite = findall(isfinite, full)
    best_full = isempty(finite) ? 0 : finite[argmin(full[finite])]
    n_top = max(1, round(Int, quantile_fraction * length(finite)))
    top_full = Set(finite[partialsortperm(full[finite], 1:n_top)])
    @printf "\n=== iteration %d: %d of %d members finite, full Φ best %.4f, mean %.4f\n" h.iteration length(finite) length(full) full[best_full] mean(full[finite])
    @printf "  %-22s %8s %9s %12s %10s %12s\n" "screen on" "Φ best" "Spearman" "regret in Φ" "relative" "top $(round(Int, 100quantile_fraction))% kept"
    for s in subsets
        length(s) == length(z_faces) && continue
        Φ = objectives(h.G, s)
        ok = [j for j in finite if isfinite(Φ[j])]
        isempty(ok) && continue
        picked = ok[argmin(Φ[ok])]
        # What the full objective thinks of the member this cheap objective would have selected.
        regret = full[picked] - full[best_full]
        top_s = Set(ok[partialsortperm(Φ[ok], 1:min(n_top, length(ok)))])
        @printf "  %-22s %8.4f %9.3f %12.4f %9.1f%% %11.0f%%\n" join(names[s], "+") Φ[picked] rank_correlation(Φ, full) regret (100regret / full[best_full]) (100length(intersect(top_s, top_full)) / n_top)
    end
end

println("\nSpearman near 1 and a small regret mean the cheap objective selects what the full one would.")
println("A low rank correlation kills the schedule outright. A high one licenses a refit — it does not")
println("replace one: this compares objectives on a fixed ensemble, and a real coarse-to-fine run would")
println("take different steps from the first update onward.")
