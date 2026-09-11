# Export coefficient vectors for the pedagogical plots. Labels are explicitly candidates until
# the numerical, seed and ensemble-size checks establish that they can be adopted.
# julia --project scripts/export_parameter_sets.jl output=parameters.csv constant=... ri=...
using BreezeCalibration, JLD2, Statistics

options = Dict(split(a, '='; limit = 2) for a in ARGS)
output = pop!(options, "output")
mkpath(dirname(abspath(output)))
quote_csv(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""
open(output, "w") do io
    println(io, "model,parameter_space,protocol,status,parameter,value,source")
    default = default_parameters(ConstantSpace())
    for (name, value) in pairs(default)
        println(io, join(quote_csv.(("Default", "constant", PROTOCOL_VERSION, "reference", name, value, "BreezeCalibration.default_parameters")), ','))
    end
    for (label, path) in sort(collect(options))
        saved = load(path)
        if haskey(saved, "parameters") && haskey(saved, "objective")
            parameters = saved["parameters"]
            status = get(saved, "converged", false) ? "candidate: local direction checks passed" : "candidate: local refinement incomplete"
        else
            selected = get(saved, "selected_mean", nothing)
            parameters = isnothing(selected) ? vec(mean(saved["history"][end].ϕ; dims = 2)) : selected.parameters
            status = isnothing(selected) ? "exploratory final ensemble mean" : "candidate: best evaluated ensemble mean"
        end
        space = space_of(length(parameters))
        for (name, value) in zip(parameter_names(space), parameters)
            println(io, join(quote_csv.((label, space isa ConstantSpace ? "constant" : "ri", get(saved, "protocol_version", "unknown"), status, name, value, path)), ','))
        end
    end
end
println(output)
