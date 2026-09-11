# One number per design, on the scale the inversion works in.
#
# `compare_designs.jl` reports RMSE per field, which is readable but does not say which design is
# better overall: the fields are in different units and the trade-offs run in different directions
# (the 56-case design won on qᵗ and lost on θˡ). The objective the inversion minimizes is the
# noise-normalized misfit, Φ = ½ mean over scored cells of ((model − LES)/σ)², pooled over fields
# and members, so that is the common currency. Computed here from the saved time means — no model
# integration — so it can be applied to an existing comparison file.
#
#     julia --project scripts/design_objective.jl [results/design/design_comparison.jld2]
using BreezeCalibration, JLD2, Statistics, Printf

const scales = BreezeCalibration.observation_scales
const scored = (:θˡ, :qᵗ, :qˡ, :u, :v)

path = isempty(ARGS) ? joinpath(@__DIR__, "..", "results", "design_comparison.jld2") : ARGS[1]
saved = load(path)
results, labels = saved["results"], saved["labels"]
member_ids, validation_sites = saved["members"], saved["validation_sites"]
members = [load_member(s, m) for (s, m) in member_ids]
les_zf = les_faces(members[1].z)

@printf "\n%s\nvalidation sites %s, %d members, %d scored fields\n" path join(validation_sites, ", ") length(members) length(scored)
@printf "\n%-28s" "design"
for r in sort(collect(keys(results))); @printf "%14s" "Φ at $(r) m"; end
@printf "%14s\n" "mean"

rows = map(enumerate(labels)) do (i, label)
    per_resolution = map(sort(collect(keys(results)))) do r
        res = results[r]
        zf = res.zf
        zo = collect(0.0:100.0:3000.0)
        # ½ ⟨((model − LES)/σ)²⟩ over every scored cell of every member and field: the inversion's own
        # objective, restricted to the validation members
        sq = Float64[]
        for (j, m) in enumerate(members), v in scored
            model = scales[v] .* regrid_column(res.means[v][i, j, :], zf, zo)
            target = scales[v] .* regrid_column(m.targets[v], les_zf, zo)
            append!(sq, ((model .- target) ./ default_observation_noise[v]) .^ 2)
        end
        0.5 * mean(sq)
    end
    (; label, per_resolution, mean_objective = mean(per_resolution))
end

for row in rows
    @printf "%-28s" row.label
    for Φ in row.per_resolution; @printf "%14.3f" Φ; end
    @printf "%14.3f\n" row.mean_objective
end

best = rows[argmin([r.mean_objective for r in rows])]
baseline = rows[1]
println()
@printf "best: %s (Φ = %.3f), against the default's %.3f — a factor %.2f\n" best.label best.mean_objective baseline.mean_objective baseline.mean_objective / best.mean_objective
println("\nΦ pools the five fields through their observation noise, so it is comparable across designs")
println("in a way the per-field RMSEs are not. It carries no error bars: with one seed per design a")
println("difference smaller than the seed-to-seed scatter is not a ranking. Repeat with seeds before")
println("selecting on small differences.")
