# Compare calibrations that differ in ensemble size, seed, or both.
#
# On a GPU the forward map's cost is nearly flat in ensemble size, so the ensemble is close to free
# per iteration and the questions become whether a larger one reaches its stopping criterion sooner,
# lands on the same coefficients, and needs less help from localization. None of that is answered by
# the misfit the EKI update happens to report: that is an average over members of Φ(θⱼ), while the
# coefficients one would adopt are a *single* parameter set whose objective is evaluated directly.
#
# Two vocabulary rules it keeps. The spread of a terminal ensemble is **spread**, never a posterior
# uncertainty — EKI here is an optimizer, not a sampler. And a run that reached pseudo time 1 has
# exhausted its *tempering budget*, which is not the same as having converged; the stop reason
# recorded in the checkpoint says which.
#
# Everything below is a diagnostic. None of these quantities proves a property of the optimizer or of
# the parameter space on its own; they are evidence to be weighed with the seeds and the refinement
# checks, and the prose is written to keep that distinction.
#
#     julia --project scripts/compare_ensemble_sizes.jl results/ladder/*.jld2
#     julia --project scripts/compare_ensemble_sizes.jl "small=a.jld2" "large=b.jld2"
# No BreezeCalibration dependency: this reads checkpoints and does arithmetic, so it loads in seconds
# and never touches the package precompile cache that the GPU jobs are queued behind.
using JLD2, Printf, Statistics, LinearAlgebra

# Metadata normalization is shared with the other analyses: `saved_metadata_value` rebuilds a value
# JLD2 had to reconstruct — `MoistStaticStability` here, and the NamedTuple containing it — from
# `propertynames`, keeping the saved type name. Without it, such values have no `keys` and compare
# unequal to themselves, so two loads of the SAME file would be reported as incompatible: a gate that
# invents differences is worse than the missing gate it replaced.
include(joinpath(@__DIR__, "saved_metadata.jl"))
const canonical = saved_metadata_value

"""Property `k` of a loaded container, or `missing`. Works on reconstructed types, which lack `keys`."""
field(nt, k) = (isnothing(nt) || ismissing(nt) || k ∉ propertynames(nt)) ? missing : canonical(getproperty(nt, k))

# A checkpoint is summarized here and the file's big arrays are dropped before the next one opens.
# Nothing downstream reads `history[n].G` — only ϕ, misfit, wall and `selected_mean.G` — and at
# 25 200 observations, 800 members and 30 iterations that unread field is about 4.8 GB per run, so
# retaining every history would exhaust a login node on a replicated ladder.
function load_entry(label, path)
    saved = load(String(path))
    history = saved["history"]
    final = history[end]
    selected = get(saved, "selected_mean", nothing)
    run_configuration = get(saved, "run_configuration", nothing)
    metadata = get(saved, "experiment_metadata", nothing)
    # The batched driver records the seed in `experiment_metadata`; a single run records it in the
    # run configuration. Neither is guaranteed, and an unknown seed must not be treated as a value.
    seed = coalesce(field(metadata, :seed), field(run_configuration, :seed))
    # Φ of a member, derived from its normalized misfit (Φ = ½ misfit²). Reported over the whole
    # history, not just the final iteration: EKI's objective is not monotone in the iteration.
    # Over finite members only: one diverged member turns `minimum` into NaN and would discard the
    # information the other members carry. A wholly failed iteration stays `missing`, not zero.
    function best_finite_member(misfit)
        finite = filter(isfinite, misfit)
        return isempty(finite) ? missing : 0.5 * minimum(finite)^2
    end
    member_Φ = [best_finite_member(h.misfit) for h in history]
    # A per-iteration table of scalars. Kept where the arrays are dropped: it is what a convergence
    # plot needs, and it is a few hundred bytes against the gigabytes of `G` that go.
    iteration_table = [(; iteration = h.iteration,
                     mean_objective = get(h, :mean_objective, missing),
                     best_member_objective = member_Φ[n],
                     # The mean was evaluated BEFORE this iteration's update, so its own pseudo time is
                     # not the post-update total. Keep both rather than conflate them.
                     evaluation_pseudotime = get(h, :evaluation_pseudotime, missing),
                     pseudotime = get(h, :pseudotime, missing),
                     wall = get(h, :wall, missing),
                     shared_forward_wall = get(h, :shared_forward_wall, missing),
                     optimizer_wall = get(h, :optimizer_wall, missing),
                     stop_reason = get(h, :stop_reason, missing))
                  for (n, h) in enumerate(history)]
    selected_iteration = isnothing(selected) ? missing : selected.iteration
    # The only ensembles kept: the one the mean was selected from, for spreads, and the last, for N_ens.
    selected_ϕ = ismissing(selected_iteration) ? nothing : copy(history[selected_iteration].ϕ)
    entry = (; label = String(label), path = String(path), final_ϕ = copy(final.ϕ), selected_ϕ, iteration_table,
               N_ens = size(final.ϕ, 2), cases = length(saved["members"]), iterations = length(history),
               protocol = get(saved, "protocol_version", missing), run_configuration, metadata, seed,
               Δt = field(run_configuration, :Δt),
               stop_reason = get(final, :stop_reason, missing), pseudotime = final.pseudotime,
               wall_hours = sum(h.wall for h in history) / 3600,
               members = get(saved, "members", missing), z_faces = get(saved, "z_faces", missing),
               radiation = get(saved, "radiation", missing),
               algorithm = get(saved, "algorithm_configuration", missing),
               selected, selected_Φ = isnothing(selected) ? missing : selected.objective,
               selected_iteration,
               final_mean_Φ = get(final, :mean_objective, missing),
               best_member_Φ = all(ismissing, member_Φ) ? missing : minimum(skipmissing(member_Φ)),
               best_member_iteration = all(ismissing, member_Φ) ? missing :
                   argmin(i -> coalesce(member_Φ[i], Inf), eachindex(member_Φ)),
               final_member_Φ = last(member_Φ),
               names = saved["parameter_names"], y = saved["y"], Γ = saved["Γ"])
    saved = nothing; history = nothing; final = nothing
    GC.gc()
    return entry
end

# `csv=` is an option, not a checkpoint; everything else is `label=path` or a bare path.
const options = filter(a -> startswith(a, "csv="), ARGS)
const checkpoint_args = filter(a -> !startswith(a, "csv="), ARGS)
entries = [load_entry((occursin('=', a) ? split(a, '=', limit = 2) : (replace(basename(a), r"\.jld2$" => ""), a))...) for a in checkpoint_args]
isempty(entries) && error("Pass one or more checkpoints")

#####
##### Compatibility. This is a gate, not a warning: a difference here means the runs are not
##### answering the same question, and every cross-run number below would be measuring it.
#####

"""Fields of the run configuration that may legitimately differ across a ladder."""
const allowed_to_differ = (:seed, :N_ens)

function incompatibilities(entries)
    problems = String[]
    function check(name, f)
        values = [canonical(f(e)) for e in entries]
        # Missing metadata is not agreement. Two checkpoints that both lack a field compare equal as
        # `missing`, which would let the gate pass on runs whose provenance is simply unrecorded.
        any(ismissing, values) && return push!(problems, "$name is not recorded in " *
            join(entries[i].label for i in eachindex(values) if ismissing(values[i]), ", "))
        length(unique(values)) == 1 ||
            push!(problems, "$name differs: " * join(unique(string(v) for v in values), " | "))
    end
    check("parameter space", e -> e.names)
    check("observations y", e -> e.y)
    check("observation noise Γ", e -> e.Γ)
    check("protocol version", e -> e.protocol)
    check("members", e -> e.members)
    check("grid faces", e -> e.z_faces)
    check("radiation mode", e -> e.radiation)
    check("algorithm configuration", e -> e.algorithm)
    # The run configuration compared field by field, so the message names the field
    configs = [e.run_configuration for e in entries]
    if any(isnothing, configs)
        push!(problems, "some checkpoints record no run configuration")
    else
        for key in union((propertynames(c) for c in configs)...)
            key in allowed_to_differ && continue
            length(unique(field(c, key) for c in configs)) == 1 ||
                push!(problems, "run configuration $key differs: " * join(unique(string(field(c, key)) for c in configs), " | "))
        end
    end
    return problems
end

if length(entries) > 1
    problems = incompatibilities(entries)
    isempty(problems) || error("These calibrations do not answer the same question, so they cannot be " *
                               "compared:\n  " * join(problems, "\n  ") *
                               "\nCompare only runs that differ in ensemble size or seed.")
end

#####
##### What each run did, and why it stopped
#####

println("\n===== runs")
@printf "%-20s %6s %6s %7s %8s %6s %9s %9s %24s\n" "run" "N_ens" "cases" "seed" "Δt (s)" "iters" "pseudo t" "wall h" "stop reason"
for e in entries
    @printf "%-20s %6d %6d %7s %8s %6d %9.3f %9.2f %24s\n" e.label e.N_ens e.cases string(coalesce(e.seed, "unknown")) string(coalesce(e.Δt, "?")) e.iterations e.pseudotime e.wall_hours string(e.stop_reason)
end
println("\n`tempering_budget`: the scheduler's pseudo time ran out. `mean_objective_plateau`: the")
println("optimization criterion was met. `none`: no stopping criterion was recorded at the last saved")
println("iteration — the run may still be going, have been interrupted, or have hit the cap. Only the")
println("second is a statement about the objective.")

#####
##### The objective, evaluated rather than inferred
#####

println("\n===== objective Φ = ½⟨((G − y)/σ)²⟩, directly evaluated")
@printf "%-20s %16s %8s %16s %16s %10s\n" "run" "selected mean" "at iter" "final mean" "best member" "at iter"
for e in entries
    fmt(x) = ismissing(x) ? "—" : @sprintf("%.4f", x)
    @printf "%-20s %16s %8s %16s %16s %10s\n" e.label fmt(e.selected_Φ) string(coalesce(e.selected_iteration, "—")) fmt(e.final_mean_Φ) fmt(e.best_member_Φ) string(coalesce(e.best_member_iteration, "—"))
end
println("\nThe selected mean is what a calibration adopts. The best-member column is the lowest member")
println("objective over the whole history, a diagnostic only: a member is a draw from a contracting")
println("ensemble, not a candidate the procedure selects. If it sits well below every mean, that is")
println("worth investigating — it is consistent with an ensemble still spread across dissimilar")
println("parameter sets, among other explanations — but it does not by itself establish any of them.")

#####
##### Seeds
#####

seed_of(e) = e.seed
usable = [i for (i, e) in enumerate(entries) if !ismissing(seed_of(e))]
groups = Dict{Tuple{Int, Int}, Vector{Int}}()
for i in usable
    push!(get!(groups, (entries[i].N_ens, entries[i].cases), Int[]), i)
end
# Distinct seeds only, and one run per seed. Two runs at the same configuration with the same seed are
# one experiment repeated: counting both would weight it twice in the scatter and understate it.
# A mixed group like seeds [1, 1, 2] passes a bare "more than one distinct seed" test, so deduplicate
# before testing rather than after.
function one_per_seed(indices)
    seen = Set()
    kept = Int[]
    for i in indices
        s = seed_of(entries[i])
        s in seen && continue
        push!(seen, s)
        push!(kept, i)
    end
    return kept
end
deduplicated = Dict(k => one_per_seed(v) for (k, v) in groups)
for (k, v) in groups
    dropped = setdiff(v, deduplicated[k])
    isempty(dropped) ||
        @warn "Duplicate seeds at N_ens = $(k[1]), cases = $(k[2]): keeping one run per seed, dropping " *
              join((entries[i].label for i in dropped), ", ")
end
paired = Dict(k => v for (k, v) in deduplicated if length(v) > 1)

if length(usable) < length(entries)
    unknown = [entries[i].label for i in eachindex(entries) if i ∉ usable]
    @warn "No seed recorded for $(join(unknown, ", ")); excluded from across-seed statistics"
end

if isempty(paired)
    println("\n===== seeds: no configuration has two runs with distinct recorded seeds")
    println("Across-seed scatter cannot be estimated, so nothing here separates a coefficient the data")
    println("determines from one the initial draw chose.")
else
    println("\n===== across-seed scatter of the selected mean, per coefficient, over the ensemble spread")
    println("      at the iteration the mean was selected")
    for ((N_ens, cases), idx) in sort(collect(paired), by = first)
        group = entries[idx]
        all(e -> !isnothing(e.selected), group) || continue
        ϕs = hcat([vec(e.selected.parameters) for e in group]...)
        @printf "\n  N_ens = %d, cases = %d, seeds %s\n" N_ens cases join(string.(seed_of.(group)), ", ")
        @printf "    %-8s %12s %14s %16s\n" "param" "mean" "across-seed" "over spread"
        for (k, name) in enumerate(group[1].names)
            # The spread to compare against is the one at the selected iteration, not at the end:
            # the ensemble keeps contracting after the mean that was selected.
            within = mean(std(e.selected_ϕ[k, :]) for e in group)
            across = std(ϕs[k, :])
            @printf "    %-8s %12.4f %14.4f %16s\n" name mean(ϕs[k, :]) across (within > 0 ? @sprintf("%.2f", across / within) : "—")
        end
        Φs = [e.selected_Φ for e in group if !ismissing(e.selected_Φ)]
        isempty(Φs) || @printf "    selected Φ across seeds: %s\n" join([@sprintf("%.4f", Φ) for Φ in Φs], ", ")
    end
    println("\nA ratio near or above 1 means the seeds disagree about that coefficient by as much as one")
    println("run's own ensemble spans, which is a reason not to quote it as determined by the data.")
end

#####
##### Coefficients and response
#####

with_G = [e for e in entries if !isnothing(e.selected) && haskey(e.selected, :G)]
if length(with_G) > 1
    reference = with_G[argmin([coalesce(e.selected_Φ, Inf) for e in with_G])]
    σ = sqrt.(reference.Γ)
    println("\n===== difference from $(reference.label) (lowest selected Φ)")
    @printf "%-20s %24s %28s\n" "run" "coefficients (spreads)" "response (σ per obs cell)"
    for e in with_G
        e === reference && continue
        within = [std(reference.selected_ϕ[k, :]) for k in eachindex(e.names)]
        dϕ = (vec(e.selected.parameters) .- vec(reference.selected.parameters)) ./ max.(within, eps())
        dG = (e.selected.G .- reference.selected.G) ./ σ
        @printf "%-20s %24s %28s\n" e.label @sprintf("RMS %.2f, max %.2f", sqrt(mean(dϕ.^2)), maximum(abs.(dϕ))) @sprintf("RMS %.3f, max %.2f", sqrt(mean(dG.^2)), maximum(abs.(dG)))
    end
    println("\nThe response column is the difference between the two forward maps on the *training*")
    println("observations, in units of their noise. Coefficients differing by much while the response")
    println("differs by little is the signature of a direction the training data constrains weakly; it")
    println("says nothing about predictions on other cases, resolutions or regimes, where the same two")
    println("parameter sets may well diverge.")
end

#####
##### Optional CSV export
#####
#
# Two slim tables, so a convergence or ladder figure can be drawn without reopening multi-gigabyte
# histories and without a second copy of the compatibility gate above. They are written only after
# that gate passes, so a CSV cannot describe a comparison the script would have refused.
#
#     ... csv=results/final/ladder     writes ladder_iterations.csv and ladder_parameters.csv

let
    if !isempty(options)
        stem = first(options)[5:end]
        isempty(dirname(stem)) || mkpath(dirname(stem))
        # Quote every cell: labels and paths are user-supplied and may contain commas or quotes.
        function cell(x)
            (ismissing(x) || isnothing(x)) && return ""
            s = string(x)
            return any(c -> c in (',', '"', '\n'), s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
        end

        open(stem * "_iterations.csv", "w") do io
            println(io, "run,N_ens,cases,seed,protocol,dt,radiation_interval,iteration,mean_objective," *
                        "best_so_far_mean_objective,best_member_objective,evaluation_pseudotime,pseudotime," *
                        "stop_reason,attributed_forward_seconds,shared_forward_seconds,optimizer_seconds," *
                        "selected_iteration,selected_objective")
            for e in entries
                best_so_far = missing
                for it in e.iteration_table
                    # Best *so far* rather than best overall: a convergence plot should show what the
                    # run knew at each iteration, not information from its future.
                    # A nonfinite mean is a failed evaluation, not a candidate: it must not become the
                    # best so far. The raw column keeps it visible as the invalid diagnostic it is.
                    if !ismissing(it.mean_objective) && isfinite(it.mean_objective)
                        best_so_far = ismissing(best_so_far) ? it.mean_objective : min(best_so_far, it.mean_objective)
                    end
                    println(io, join(cell.((e.label, e.N_ens, e.cases, coalesce(e.seed, missing),
                                            e.protocol, coalesce(e.Δt, missing),
                                            field(e.run_configuration, :radiation_interval),
                                            it.iteration, it.mean_objective, best_so_far,
                                            it.best_member_objective, it.evaluation_pseudotime,
                                            it.pseudotime, it.stop_reason, it.wall,
                                            it.shared_forward_wall, it.optimizer_wall,
                                            e.selected_iteration, e.selected_Φ)), ","))
                end
            end
        end

        open(stem * "_parameters.csv", "w") do io
            println(io, "run,N_ens,cases,seed,protocol,dt,radiation_interval,selected_iteration,parameter,selected_value,spread_at_selected")
            for e in entries
                isnothing(e.selected) && continue
                for (k, name) in enumerate(e.names)
                    spread = isnothing(e.selected_ϕ) ? missing : std(e.selected_ϕ[k, :])
                    println(io, join(cell.((e.label, e.N_ens, e.cases, coalesce(e.seed, missing),
                                            e.protocol, coalesce(e.Δt, missing),
                                            field(e.run_configuration, :radiation_interval),
                                            e.selected_iteration, name, vec(e.selected.parameters)[k], spread)), ","))
                end
            end
        end
        println("\nwrote $(stem)_iterations.csv and $(stem)_parameters.csv")
    end
end
