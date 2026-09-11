# Evaluate calibrated parameters against the defaults on every member of the library (83 columns
# per parameter set, one column-ensemble run per parameter set and grid).
#
#     julia -t auto --project scripts/evaluate.jl [checkpoint=results/eki.jld2] [resolutions=20,50,100,hindcast] [arch=cpu|gpu]
#                                                 [top=25000|les] [radiation=interactive|prescribed]
#                                                 [dt=] [radiation_interval=] [validation=3,12,21] [reserved=6,9,15,18] [reveal=true]
#                                                 [output=results/evaluation.jld2]
#
# `resolutions` is a comma-separated list of uniform spacings in m and/or `hindcast`. The column top, the
# radiation, the time step and the radiation interval all default to what the checkpoint records, so a
# calibration is evaluated under the protocol it was fit under; overriding any of them is measuring
# something else as well as skill, and the script says so.
using BreezeCalibration, JLD2, Statistics, Printf
using Oceananigans: CPU, GPU

options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
path = get(options, "checkpoint", joinpath(@__DIR__, "..", "results", "eki.jld2"))
output = get(options, "output", joinpath(@__DIR__, "..", "results", "evaluation.jld2"))
resolutions = split(get(options, "resolutions", "20,50,100,hindcast"), ',')

saved = load(path)
history = saved["history"]
ϕ_last = history[end].ϕ
space = space_of(size(ϕ_last, 1))
best = argmin(history[end].misfit)
calibrated_mean = named(space, vec(mean(ϕ_last, dims = 2)))
calibrated_best = named(space, ϕ_last[:, best])
training = saved["members"]

# `selected_mean` is the ensemble mean whose objective, evaluated directly
# rather than inferred from the members, was lowest over the whole run. It is not in general the
# last iteration's mean — EKI's objective is not monotone in the iteration, and under `optimize`
# the run deliberately continues past the tempering budget. Score it whenever the checkpoint has it.
selected = get(saved, "selected_mean", nothing)
if isnothing(selected)
    labels = ["default (Nakanishi–Niino)", "EKI final mean", "EKI best member"]
    sets = (default_parameters(space), calibrated_mean, calibrated_best)
    @warn "The checkpoint records no directly evaluated mean; scoring the final iteration's mean instead, which is not the best evaluated candidate and may not be near it"
else
    labels = ["default (Nakanishi–Niino)", "best evaluated mean", "EKI final mean", "EKI best member"]
    sets = (default_parameters(space), named(space, vec(selected.parameters)), calibrated_mean, calibrated_best)
    @info @sprintf("Best directly evaluated ensemble mean is from iteration %d, objective %.4f — a candidate, not an established optimum until the plateau is corroborated by independent seeds", selected.iteration, selected.objective)
end

println("parameters ($space):")
for (label, p) in zip(labels, sets)
    println(rpad(label, 28), join([@sprintf("%s = %.3f", n, getproperty(p, n)) for n in parameter_names(space)], "  "))
end

members = [load_member(s, m) for (s, m) in library_members()]
istrain = [(m.site, m.month) in training for m in members]

# Held-out members split into two roles. A set that chooses anything — the training design, the
# ensemble size, which closure family to adopt — is a *validation* set, and reporting a final skill
# on it overstates the skill, because the choice already fitted it. The reserved sites are kept out
# of every such decision so that one number at the end is an honest out-of-sample test. They are
# integrated and saved either way (no extra model runs); only the reporting is gated.
reserved_sites = parse.(Int, split(get(options, "reserved", "6,9,15,18"), ','))
validation_sites = parse.(Int, split(get(options, "validation", "3,12,21"), ','))
reveal = get(options, "reveal", "false") == "true"

# The validation set must be the *same* sites for every calibration being compared, so it is named
# explicitly rather than derived as "everything not trained on". Derived that way it would shrink and
# shift as the training split changes — a design comparison whose yardstick moves with the design.
isreserved = [m.site in reserved_sites for m in members]
isvalidation = [m.site in validation_sites for m in members]
isdiagnostic = .!istrain .& .!isreserved .& .!isvalidation   # neither trained on nor part of either judgement
isempty(intersect(validation_sites, reserved_sites)) || error("Validation and reserved sites overlap: $(intersect(validation_sites, reserved_sites))")
trained_sites = unique(s for (s, _) in training)
bad = intersect(trained_sites, union(validation_sites, reserved_sites))
isempty(bad) || error("Sites $bad are trained on but are also held out for judgement")
params = hcat([collect(Float64, p) for p in sets]...)

variables = Tuple(Symbol.(split(get(options, "variables", join(String.(default_variables), ',')), ',')))
# The checkpoint records the top face of its first grid (the LES top, about 4 km, for a column that was not extended)
saved_top = get(saved, "top", 25_000.0)
top = get(options, "top", saved_top > 4001 ? string(saved_top) : "les"); top = top == "les" ? nothing : parse(Float64, top)
radiation = Symbol(get(options, "radiation", get(saved, "radiation", "interactive")))

# The discretization is part of the protocol the coefficients were fit under, so it must be read from
# the checkpoint like the column top and the radiation. Evaluating a calibration at a step it was not
# fit at measures the discretization difference along with the skill — and since Δt = 60 s sits 1.20 σ
# from Δt = 15 s against a 4.68 σ misfit, that is not a small contamination.
run_configuration = get(saved, "run_configuration", nothing)
saved_Δt = isnothing(run_configuration) ? nothing : get(run_configuration, :Δt, nothing)
saved_interval = isnothing(run_configuration) ? nothing : get(run_configuration, :radiation_interval, nothing)
Δt = parse(Float64, get(options, "dt", string(something(saved_Δt, 60.0))))
radiation_interval = parse(Float64, get(options, "radiation_interval", string(something(saved_interval, 600.0))))
if isnothing(saved_Δt)
    @warn "The checkpoint records no time step; evaluating at Δt = $Δt s, which may differ from the calibration's"
elseif Δt != saved_Δt
    @warn "Evaluating at Δt = $Δt s, but the calibration used $(saved_Δt) s: the scores include that difference"
end
@info "Evaluating at Δt = $Δt s with radiation every $(radiation_interval / 60) min (protocol $(get(saved, "protocol_version", "unrecorded")))"
get(options, "arch", "cpu") == "gpu" && @eval using CUDA
architecture = get(options, "arch", "cpu") == "gpu" ? GPU() : CPU()
problem_for(resolution) = resolution == "hindcast" ? ColumnEnsembleProblem(members; z_faces = hindcast_faces(), variables, top) :
                          resolution == "20" ? ColumnEnsembleProblem(members; variables, top) :
                          ColumnEnsembleProblem(members; Δz = parse(Float64, resolution), variables, top)

results = Dict{String, Any}()
for resolution in resolutions
    problem = problem_for(resolution)
    @info "Evaluating $(length(sets)) parameter sets on $(length(members)) members at resolution $resolution ($(length(problem.zf) - 1) cells, top $(problem.zf[end]) m, radiation $radiation)"
    scores, means = evaluate(params, problem; space, radiation, architecture, Δt, radiation_interval)
    println("\n===== resolution $resolution")
    println("training members ($(count(istrain))):")
    rmse_table(scores[:, istrain], labels)
    println("validation members ($(count(isvalidation)), sites $(join(validation_sites, ", ")) — the same sites for every calibration; these choose the design):")
    rmse_table(scores[:, isvalidation], labels)
    if any(isdiagnostic)
        println("further non-training members ($(count(isdiagnostic))) — diagnostic only, not a basis for selection:")
        rmse_table(scores[:, isdiagnostic], labels)
    end
    if reveal
        println("RESERVED TEST members ($(count(isreserved)), sites $(join(reserved_sites, ", "))) — reveal only after the coefficients are frozen:")
        rmse_table(scores[:, isreserved], labels)
    else
        println("reserved test members ($(count(isreserved)), sites $(join(reserved_sites, ", "))): computed and saved, not shown.")
        println("  They are the final test. Anything that selects a design or a coefficient must use the")
        println("  validation rows above; pass reveal=true only once nothing further will be chosen.")
    end
    results[resolution] = (; scores, means, zc = problem.zc, zf = problem.zf)
end

les = (zc = members[1].z, θˡ = hcat([m.targets.θˡ for m in members]...), qᵗ = hcat([m.targets.qᵗ for m in members]...),
       qˡ = hcat([m.targets.qˡ for m in members]...), cloud_fraction = [m.targets.cloud_fraction for m in members])
isempty(dirname(output)) || mkpath(dirname(output))
jldsave(output; results, labels, members = [(m.site, m.month) for m in members], istrain, isvalidation, isreserved, isdiagnostic, reserved_sites, validation_sites, params, Δt, radiation_interval,
                protocol_version = PROTOCOL_VERSION, checkpoint = path,
                parameter_names = collect(String.(parameter_names(space))), les)
