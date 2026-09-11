# One line per saved iteration of a checkpoint: julia --project scripts/last_iteration.jl results/eki.jld2 [all]
using JLD2, Statistics, Printf
path = ARGS[1]
history = load(path, "history")
shown = length(ARGS) ≥ 2 && ARGS[2] == "all" ? history : history[end:end]
for h in shown
    Δt = haskey(h, :Δt) ? h.Δt : NaN
    T = haskey(h, :pseudotime) ? h.pseudotime : NaN
    @printf "%s iteration %d: misfit mean %.2f median %.2f best %.2f  Δt %.3f  pseudo time %.3f  forward map %.0f min\n" basename(path) h.iteration mean(h.misfit) median(h.misfit) minimum(h.misfit) Δt T h.wall / 60
end
