# Selection for direct local checks; this does not change the mean-based EKI stopping rule.
using Statistics: mean

function evaluated_candidate(saved; selection = :mean)
    selection in (:mean, :best_evaluated) || error("selection must be mean or best_evaluated")
    selected = get(saved, "selected_mean", nothing)
    isnothing(selected) && error("No directly evaluated mean candidate")
    candidate = (; parameters = copy(selected.parameters), G = copy(selected.G),
                   objective = selected.objective, iteration = selected.iteration,
                   kind = :ensemble_mean, member = 0)
    isfinite(candidate.objective) || error("Selected mean objective is nonfinite")
    if selection == :best_evaluated
        for h in saved["history"], j in eachindex(h.misfit)
            objective = h.misfit[j]^2 / 2
            if isfinite(objective) && objective < candidate.objective
                candidate = (; parameters = copy(h.ϕ[:, j]), G = copy(h.G[:, j]),
                               objective, iteration = h.iteration, kind = :ensemble_member, member = j)
            end
        end
    end
    objective = mean(abs2, (candidate.G .- saved["y"]) ./ sqrt.(saved["Γ"])) / 2
    isapprox(objective, candidate.objective; rtol = 1e-10, atol = 1e-12) ||
        error("Selected candidate objective disagrees with its saved forward map")
    return candidate
end
