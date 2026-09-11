# Compare saved evaluations of exactly the same coefficients, without loading the SCM or reintegrating.
# julia --project scripts/export_design_scores.jl output.csv comparison1.jld2 comparison2.jld2
using JLD2

length(ARGS) >= 2 || error("Pass output.csv and one or more saved design comparisons")
output, paths = first(ARGS), ARGS[2:end]
runs = load.(paths)
reference = first(runs)
for run in runs, key in ("protocol_version", "labels", "params", "members", "validation_sites")
    isequal(run[key], reference[key]) || error("Comparisons differ in $key; this would confound the numerical experiment")
end
quote_csv(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""
mkpath(dirname(abspath(output)))
open(output, "w") do io
    println(io, "protocol,dt,radiation_interval,resolution,model,site,month,variable,rmse,source")
    for (run, path) in zip(runs, paths), (resolution, result) in run["results"],
        (i, label) in enumerate(run["labels"]), (j, (site, month)) in enumerate(run["members"]),
        variable in (:θˡ, :qᵗ, :qˡ, :u, :v)
        println(io, join(quote_csv.((run["protocol_version"], run["Δt"], run["radiation_interval"],
                          resolution, label, site, month, variable, result.scores[i, j][variable], abspath(path))), ','))
    end
end
println(output)
