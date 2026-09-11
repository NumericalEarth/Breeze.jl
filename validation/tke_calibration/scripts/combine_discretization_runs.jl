# Assemble compatible saved runs without repeating a forward simulation.
# julia --project scripts/combine_discretization_runs.jl output.jld2 partial1.jld2 partial2.jld2 ...
using JLD2

length(ARGS) >= 3 || error("Pass an output path and at least two partial runs")
output, paths = first(ARGS), ARGS[2:end]
saved = load.(paths)
reference = first(saved)
label = reference["label"]
label in ("Δt", "Δtᵣ") || error("Unknown sweep label $label")
fixed_knob = label == "Δt" ? "radiation_interval" : "radiation_dt"
for run in saved, key in ("protocol_version", "label", "params", "members", "zf", "observation_zf", fixed_knob)
    isequal(run[key], reference[key]) || error("Incompatible partial runs: $key differs")
end
values = [run["value"] for run in saved]
length(unique(values)) == length(values) || error("Duplicate sweep setting")
combined = copy(reference)
combined["dt_means"] = label == "Δt" ? Dict(run["value"] => run["means"] for run in saved) : Dict()
combined["interval_means"] = label == "Δtᵣ" ? Dict(run["value"] => run["means"] for run in saved) : Dict()
combined["source_partials"] = abspath.(paths)
delete!(combined, "means")
mkpath(dirname(abspath(output)))
jldsave(output; (Symbol(key) => value for (key, value) in combined)...)
println(output)
