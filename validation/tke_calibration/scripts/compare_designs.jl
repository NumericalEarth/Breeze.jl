# Compare calibrations that differ in their training design, on a validation set common to all of
# them. Their own training objectives are not comparable — a design trained on 56 cases has a longer
# and harder observation vector than one trained on 16, so a smaller normalized misfit may mean only
# that it was asked an easier question. The common set asks all of them the same question.
#
# Every candidate is one column of a single ensemble, so the whole comparison is one model run per
# grid regardless of how many designs are compared.
#
#     julia -t auto --project scripts/compare_designs.jl "A=results/design/A.jld2" "B=..." \
#           [resolutions=50,20] [validation=3,12,21] [arch=gpu] [dt=] [radiation_interval=] [output=...]
using BreezeCalibration, JLD2, Statistics, Printf
using Oceananigans: CPU, GPU
include(joinpath(@__DIR__, "calibration_data_manifest.jl"))

options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
resolutions = split(pop!(options, "resolutions", "50,20"), ',')
validation_sites = parse.(Int, split(pop!(options, "validation", "3,12,21"), ','))
arch_name = pop!(options, "arch", "cpu")
dt_option = pop!(options, "dt", nothing)
interval_option = pop!(options, "radiation_interval", nothing)
output = pop!(options, "output", joinpath(@__DIR__, "..", "results", "design_comparison.jld2"))
arch_name == "gpu" && @eval using CUDA
architecture = arch_name == "gpu" ? GPU() : CPU()
isempty(options) && error("Pass at least one label=checkpoint")

# Each design contributes one candidate. `selected_mean` — the mean whose objective was evaluated
# directly — is preferred; without it the final iteration's mean is the only thing available, and it
# is a weaker candidate, so say so rather than let the label imply otherwise.
entries = sort!([(label = String(l), path = String(p)) for (l, p) in options], by = e -> e.label)
source_hashes = [(; e.label, sha256 = file_sha256(e.path)) for e in entries]
candidates = map(entries) do e
    saved = load(e.path)
    # A frozen candidate rescored against changed forcing data would credit the difference to the
    # coefficients. Warn-only for legacy checkpoints, which record no hashes.
    validate_data_manifest(saved)
    history = saved["history"]
    selected = get(saved, "selected_mean", nothing)
    ϕ = isnothing(selected) ? vec(mean(history[end].ϕ, dims = 2)) : vec(selected.parameters)
    run_configuration = get(saved, "run_configuration", nothing)
    (; e.label, e.path, ϕ, evaluated = !isnothing(selected),
       cases = length(saved["members"]), N_ens = size(history[end].ϕ, 2), iterations = length(history),
       Δt = isnothing(run_configuration) ? nothing : get(run_configuration, :Δt, nothing),
       radiation_interval = isnothing(run_configuration) ? nothing : get(run_configuration, :radiation_interval, nothing),
       protocol = get(saved, "protocol_version", missing))
end

space = space_of(length(first(candidates).ϕ))
all(c -> length(c.ϕ) == length(first(candidates).ϕ), candidates) || error("The checkpoints calibrate different parameter spaces")
protocols = unique(c.protocol for c in candidates)
length(protocols) == 1 || @warn "Candidates span protocol versions $protocols: their forward maps differ, so this comparison mixes physics"
Δts = unique(c.Δt for c in candidates)
length(Δts) == 1 || @warn "Candidates were fit at different time steps $Δts"
any(c -> !c.evaluated, candidates) &&
    @warn "Some checkpoints record no directly evaluated mean; using the final iteration's ensemble mean for those, which is a weaker candidate"

Δt = isnothing(dt_option) ? something(first(Δts), 60.0) : parse(Float64, dt_option)
radiation_interval = isnothing(interval_option) ? something(first(candidates).radiation_interval, 600.0) : parse(Float64, interval_option)

members = [load_member(s, m) for (s, m) in library_members() if s in validation_sites]
isempty(members) && error("No library members at sites $validation_sites")
member_ids = [(m.site, m.month) for m in members]
validation_data = calibration_data_manifest(member_ids)
source_code = (; revision = readchomp(`git rev-parse HEAD`),
                 diff_sha256 = bytes2hex(sha256(read(`git diff --binary`))))
for c in candidates
    saved_training = load(c.path, "members")
    bad = intersect(unique(s for (s, _) in saved_training), validation_sites)
    isempty(bad) || error("$(c.label) trained on validation sites $bad; the comparison would not be out of sample")
end

labels = vcat("default (Nakanishi–Niino)", [c.label for c in candidates])
params = hcat(collect(Float64, default_parameters(space)), [c.ϕ for c in candidates]...)

println("\ndesigns compared (all scored on sites $(join(validation_sites, ", ")), $(length(members)) members):")
@printf "  %-18s %8s %7s %7s %11s %10s %s\n" "label" "N_ens" "cases" "iters" "columns" "Δt (s)" "candidate"
for c in candidates
    @printf "  %-18s %8d %7d %7d %11d %10s %s\n" c.label c.N_ens c.cases c.iterations c.N_ens*c.cases string(something(c.Δt, "?")) (c.evaluated ? "evaluated mean" : "final mean")
end
@info "Scoring $(length(labels)) parameter sets at Δt = $Δt s, radiation every $(radiation_interval / 60) min"

results = Dict{String, Any}()
for resolution in resolutions
    problem = resolution == "hindcast" ? ColumnEnsembleProblem(members; z_faces = hindcast_faces()) :
              resolution == "20" ? ColumnEnsembleProblem(members) :
              ColumnEnsembleProblem(members; Δz = parse(Float64, resolution))
    @info "resolution $resolution: $(length(labels)) × $(length(members)) = $(length(labels) * length(members)) columns, $(length(problem.zf) - 1) cells"
    scores, means = evaluate(params, problem; space, architecture, Δt, radiation_interval)
    println("\n===== resolution $resolution, validation sites $(join(validation_sites, ", ")):")
    rmse_table(scores, labels)
    results[resolution] = (; scores, means, zf = problem.zf)
end

isempty(dirname(output)) || mkpath(dirname(output))
all(file_sha256(e.path) == source.sha256 for (e, source) in zip(entries, source_hashes)) ||
    error("A candidate checkpoint changed during evaluation; freeze it before comparing")
isequal(validation_data, calibration_data_manifest(member_ids)) ||
    error("Validation forcing data changed during evaluation")
comparison_manifest = (; complete = true, source_hashes, validation_data, source_code, architecture = arch_name)
# A failed serialization must not leave a final-looking file for a watcher to accept.
temporary, io = mktemp(dirname(abspath(output)))
close(io)
try
    jldsave(temporary; results, labels, params, validation_sites, Δt, radiation_interval,
                      members = member_ids, comparison_manifest,
                      designs = [(; c.label, c.cases, c.N_ens, c.iterations, c.evaluated) for c in candidates],
                      protocol_version = PROTOCOL_VERSION)
    mv(temporary, output; force = true)
finally
    isfile(temporary) && rm(temporary)
end
@info "wrote $output"
