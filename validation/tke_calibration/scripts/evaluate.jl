# Evaluate calibrated parameters against the defaults on every member of the library (83 columns
# per parameter set, one column-ensemble run per parameter set and grid).
#
#     julia -t auto --project scripts/evaluate.jl [checkpoint=results/eki.jld2] [resolutions=20,50,100,hindcast] [arch=cpu|gpu]
#                                                 [top=25000|les] [radiation=interactive|prescribed] [output=results/evaluation.jld2]
#
# `resolutions` is a comma-separated list of uniform spacings in m and/or `hindcast`. The column top and the
# radiation default to what the checkpoint records, so a calibration is evaluated under its own protocol.
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
labels = ["default (Nakanishi–Niino)", "EKI ensemble mean", "EKI best member"]
sets = (default_parameters(space), calibrated_mean, calibrated_best)

println("parameters ($space):")
for (label, p) in zip(labels, sets)
    println(rpad(label, 28), join([@sprintf("%s = %.3f", n, getproperty(p, n)) for n in parameter_names(space)], "  "))
end

members = [load_member(s, m) for (s, m) in library_members()]
istrain = [(m.site, m.month) in training for m in members]
params = hcat([collect(Float64, p) for p in sets]...)

variables = Tuple(Symbol.(split(get(options, "variables", join(String.(default_variables), ',')), ',')))
# The checkpoint records the top face of its first grid (the LES top, about 4 km, for a column that was not extended)
saved_top = get(saved, "top", 25_000.0)
top = get(options, "top", saved_top > 4001 ? string(saved_top) : "les"); top = top == "les" ? nothing : parse(Float64, top)
radiation = Symbol(get(options, "radiation", get(saved, "radiation", "interactive")))
get(options, "arch", "cpu") == "gpu" && @eval using CUDA
architecture = get(options, "arch", "cpu") == "gpu" ? GPU() : CPU()
problem_for(resolution) = resolution == "hindcast" ? ColumnEnsembleProblem(members; z_faces = hindcast_faces(), variables, top) :
                          resolution == "20" ? ColumnEnsembleProblem(members; variables, top) :
                          ColumnEnsembleProblem(members; Δz = parse(Float64, resolution), variables, top)

results = Dict{String, Any}()
for resolution in resolutions
    problem = problem_for(resolution)
    @info "Evaluating $(length(sets)) parameter sets on $(length(members)) members at resolution $resolution ($(length(problem.zf) - 1) cells, top $(problem.zf[end]) m, radiation $radiation)"
    scores, means = evaluate(params, problem; space, radiation, architecture)
    println("\n===== resolution $resolution, all $(length(members)) members:")
    rmse_table(scores, labels)
    println("training members ($(count(istrain))):")
    rmse_table(scores[:, istrain], labels)
    println("held-out members ($(count(.!istrain))):")
    rmse_table(scores[:, .!istrain], labels)
    results[resolution] = (; scores, means, zc = problem.zc, zf = problem.zf)
end

les = (zc = members[1].z, θˡ = hcat([m.targets.θˡ for m in members]...), qᵗ = hcat([m.targets.qᵗ for m in members]...),
       qˡ = hcat([m.targets.qˡ for m in members]...), cloud_fraction = [m.targets.cloud_fraction for m in members])
isempty(dirname(output)) || mkpath(dirname(output))
jldsave(output; results, labels, members = [(m.site, m.month) for m in members], istrain, params,
                parameter_names = collect(String.(parameter_names(space))), les)
