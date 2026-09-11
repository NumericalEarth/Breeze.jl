# What does the ensemble size buy? On a GPU the per-step cost is nearly flat in the number of
# parameter sets — 8× the columns for 1.4× the cost — so the ensemble is close to free per iteration
# and the question becomes whether a larger one converges in fewer iterations, lands in the same
# place, and needs less help from localization.
#
# Reads the checkpoints of calibrations that differ only in `N_ens` (and possibly localization) and
# tabulates the trajectory, the cost, and the posterior.
#
#     julia --project scripts/compare_ensemble_sizes.jl results/ensemble_size/eki_n*.jld2
#     julia --project scripts/compare_ensemble_sizes.jl "50=...jld2" "400 (no loc)=...jld2"
using BreezeCalibration, JLD2, Printf, Statistics, LinearAlgebra

entries = map(ARGS) do a
    label, path = occursin('=', a) ? split(a, '=', limit = 2) : (replace(basename(a), r"^eki_|\.jld2$" => ""), a)
    saved = load(String(path))
    (; label = String(label), path = String(path), saved, history = saved["history"],
       names = saved["parameter_names"], y = saved["y"], Γ = saved["Γ"])
end
isempty(entries) && error("Pass one or more checkpoints")

allsame(f) = all(e -> isequal(f(e), f(entries[1])), entries)
allsame(e -> e.names) || error("The checkpoints calibrate different parameters")
allsame(e -> e.y) || @warn "The checkpoints were fit to different observations; the misfits are not comparable"
for e in entries
    v = get(e.saved, "protocol_version", missing)
    ismissing(v) && @warn "$(e.label): no protocol version recorded"
end
protocols = unique(get(e.saved, "protocol_version", missing) for e in entries)
length(protocols) == 1 || @warn "Checkpoints span protocol versions $protocols; their forward maps differ"

println("\n===== trajectory and cost")
@printf "%-14s %6s %6s %10s %11s %10s %11s %11s\n" "run" "N_ens" "iters" "pseudo t" "terminated" "min/iter" "total h" "misfit"
for e in entries
    h = e.history
    N_ens = size(h[end].ϕ, 2)
    walls = [x.wall for x in h]
    final = h[end]
    @printf "%-14s %6d %6d %10.3f %11s %10.1f %11.2f %11.3f\n" e.label N_ens length(h) final.pseudotime (final.pseudotime ≥ 0.999 ? "yes" : "NO") mean(walls)/60 sum(walls)/3600 mean(final.misfit)
end

println("\n===== misfit by iteration (ensemble mean of the normalized misfit)")
maxit = maximum(length(e.history) for e in entries)
@printf "%-14s" "iteration"
for i in 1:maxit; @printf "%7d" i; end
println()
for e in entries
    @printf "%-14s" e.label
    for i in 1:maxit
        i ≤ length(e.history) ? @printf("%7.2f", mean(e.history[i].misfit)) : @printf("%7s", "")
    end
    println()
end

println("\n===== pseudo time by iteration (termination is 1; a larger step means a faster ascent)")
@printf "%-14s" "iteration"
for i in 1:maxit; @printf "%7d" i; end
println()
for e in entries
    @printf "%-14s" e.label
    for i in 1:maxit
        i ≤ length(e.history) ? @printf("%7.3f", e.history[i].pseudotime) : @printf("%7s", "")
    end
    println()
end

println("\n===== posterior: ensemble mean ± std of each parameter")
@printf "%-6s %14s" "param" "default"
for e in entries; @printf "%20s" e.label; end
println()
space = space_of(length(entries[1].names))
for (k, name) in enumerate(entries[1].names)
    @printf "%-6s %14.3f" name getproperty(default_parameters(space), Symbol(name))
    for e in entries
        ϕ = e.history[end].ϕ
        @printf "%12.3f ±%6.3f" mean(ϕ[k, :]) std(ϕ[k, :])
    end
    println()
end

# Do the runs agree? Difference of posterior means between each run and the largest ensemble,
# measured in that run's own posterior standard deviations — the scale on which a difference matters.
largest = entries[argmax([size(e.history[end].ϕ, 2) for e in entries])]
println("\n===== agreement with the largest ensemble ($(largest.label)), in posterior standard deviations")
@printf "%-14s %12s %12s %-20s\n" "run" "RMS" "max" "worst parameter"
for e in entries
    e === largest && continue
    ϕ, ϕᴸ = e.history[end].ϕ, largest.history[end].ϕ
    d = [(mean(ϕ[k, :]) - mean(ϕᴸ[k, :])) / max(std(ϕᴸ[k, :]), eps()) for k in eachindex(e.names)]
    @printf "%-14s %12.3f %12.3f %-20s\n" e.label sqrt(mean(d .^ 2)) maximum(abs.(d)) e.names[argmax(abs.(d))]
end
println("\nA small ensemble that lands within a fraction of a posterior standard deviation of the largest\n" *
        "is telling the same story more cheaply. Differences of order one mean the sample covariance —\n" *
        "or the localization correcting it — is still setting the answer.")

println("\n===== spread collapse (mean over parameters of posterior std / prior std)")
@printf "%-14s %12s\n" "run" "std ratio"
for e in entries
    ϕ₀, ϕ = e.history[1].ϕ, e.history[end].ϕ
    ratios = [std(ϕ[k, :]) / max(std(ϕ₀[k, :]), eps()) for k in eachindex(e.names)]
    @printf "%-14s %12.3f\n" e.label mean(ratios)
end
println("\nEKI's spread shrinks as it conditions on the data; too small an ensemble collapses further\n" *
        "than the posterior warrants, so an over-tight spread at small N_ens is the symptom to look for.")
