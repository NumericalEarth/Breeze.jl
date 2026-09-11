# Would fitting on a cheap subset of the grids select the same coefficients as fitting on all of them?
#
# A multi-resolution calibration concatenates the grids: the forward map is [G₅₀; G₁₀₀; G_hindcast] and
# the observations are the same targets repeated, so with 56 cases and three grids the observation
# vector is 25,200 long. That costs twice over — the forward map scales with the cells summed over the
# grids, and the EKI update scales with the *cube* of the observation length. Screening the early
# iterations on one cheap grid and refining on all three would cut both.
#
# The question it asks is whether the cheap objective carries the same information about the ensemble
# as the full one — and it can be asked with no new integration at all, because every iteration's
# forward map is stored in the checkpoint and each grid's block is a slice of it.
#
# WHAT THIS IS NOT. EKI does not select the best member: it updates using the covariance between the
# parameters and the forward maps across the whole ensemble. So rank agreement and the regret of the
# member a cheap objective *would* pick are proxies for whether the cheap objective orders the ensemble
# the same way, not the mechanism by which a coarse-to-fine run would actually move. It also compares
# objectives on a *fixed* ensemble, and a real coarse-to-fine run would take different update steps
# from the first iteration onward, so its trajectory would diverge from this one.
#
# Read the result accordingly, in both directions. High agreement is motivation to spend GPU time on a
# refit, not a substitute for one. Low agreement on a single ensemble is a warning that deprioritizes a
# subset; it does not disprove the schedule, because the update could still move sensibly on covariances
# whose ordering these statistics do not capture.
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

"""
Ranks with ties averaged, which is what Spearman's ρ is defined on. `invperm(sortperm(x))` breaks ties
by position instead, which would report a spurious ordering between members whose objectives are equal
— and equal objectives are not exotic here: a degenerate parameter set can give several members the
same profile.
"""
function tied_ranks(x)
    order = sortperm(x)
    r = zeros(Float64, length(x))
    i = 1
    while i <= length(x)
        j = i
        while j < length(x) && x[order[j + 1]] == x[order[i]]
            j += 1
        end
        for k in i:j
            r[order[k]] = (i + j) / 2   # the mean of the ranks the tied block occupies
        end
        i = j + 1
    end
    return r
end

"""Spearman's ρ, on the members where both objectives are finite."""
function rank_correlation(a, b)
    keep = findall(i -> isfinite(a[i]) && isfinite(b[i]), eachindex(a))
    length(keep) > 2 || return NaN
    ra, rb = tied_ranks(a[keep]), tied_ranks(b[keep])
    # With every value tied, a rank vector is constant and its correlation is 0/0. Report that as
    # undefined rather than letting a NaN read as a computed correlation of zero.
    (allequal(ra) || allequal(rb)) && return NaN
    return cor(ra, rb)
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
    # An early iteration from a broad prior can leave no member with a finite objective on every grid.
    # Say so and move on: indexing the best of an empty set would crash, and reporting a comparison
    # over no members would be worse.
    if isempty(finite)
        @printf "\n=== iteration %d: no member has a finite objective on all grids; nothing to compare\n" h.iteration
        continue
    end
    best_full = finite[argmin(full[finite])]
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

println("\nSpearman near 1 with a small regret means the cheap objective orders this ensemble the way the")
println("full one does. That is motivation to try a refit on the subset, not evidence that one would")
println("succeed: EKI updates on the covariance between parameters and forward maps over the whole")
println("ensemble rather than by selecting a best member, and a real coarse-to-fine run would take")
println("different steps from the first update onward, so its trajectory leaves this one immediately.")
println("Low agreement on a single ensemble is a reason to deprioritize a subset, not to rule it out.")
println("\nRanking can change as the ensemble contracts, which is when small differences between grids")
println("start to matter, so a subset should be judged over several iterations rather than the first.")
