# Export saved diagnostics without loading the SCM, changing data, or integrating again.
# julia --compiled-modules=existing --project scripts/export_closure_diagnostics.jl output_dir Constant=constant.jld2 Ri=ri.jld2
using JLD2
include(joinpath(@__DIR__, "saved_metadata.jl"))

function export_closure_diagnostics(output, specifications)
    isempty(specifications) && error("Pass at least one label=diagnostics.jld2")
    labels, paths = first.(specifications), last.(specifications)
    length(unique(labels)) == length(labels) || error("Duplicate candidate labels")
    any(label -> label in ("Default", "LES"), labels) && error("Default and LES are reserved labels")
    runs = load.(paths)
    reference = first(runs)
    for run in runs, key in ("protocol_version", "members", "zf", "zc", "les_zf", "observation_zf",
                            "Δt", "radiation_interval", "stop_time", "averaging_window", "targets",
                            "source_run_configuration")
        isequal(saved_metadata_value(run[key]), saved_metadata_value(reference[key])) || error("Diagnostics differ in $key")
    end
    for run in runs
        all(>(0), run["diagnostics"].counts) || error("Diagnostics contain unsampled cases")
        for key in keys(reference["means"])
            isapprox(run["means"][key][1, :, :], reference["means"][key][1, :, :]; rtol = 1e-10, atol = 1e-12) ||
                error("Default profiles differ in $key; the candidates cannot share one baseline")
        end
        for key in keys(reference["diagnostics"].means)
            isapprox(run["diagnostics"].means[key][1, :, :], reference["diagnostics"].means[key][1, :, :];
                     rtol = 1e-10, atol = 1e-12) || error("Default diagnostics differ in $key")
        end
    end
    mkpath(output)
    quote_csv(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""
    row(io, values) = println(io, join(quote_csv.(values), ','))
    profile_variables = ((:θˡ, "theta_l"), (:qᵗ, "qt"), (:qˡ, "ql"), (:u, "u"), (:v, "v"))
    # Keep native staggering: the TKE and dissipation are at centers; diffusivities, mixing
    # length, production terms and kinematic fluxes are at faces. No interpolation for plotting.
    center_diagnostics = (:e, :dissipation)
    sources = vcat([("Default", reference, 1, first(paths))],
                   [(label, run, 2, path) for (label, run, path) in zip(labels, runs, paths)])
    open(joinpath(output, "profiles.csv"), "w") do io
        println(io, "model,site,month,kind,variable,z_m,value,source")
        for (label, run, i, path) in sources, (j, (site, month)) in enumerate(run["members"])
            for (variable, name) in profile_variables, (k, z) in enumerate(run["zc"])
                row(io, (label, site, month, "profile", name, z, run["means"][variable][i, j, k], abspath(path)))
            end
            for (variable, values) in run["diagnostics"].means
                z = variable in center_diagnostics ? run["zc"] : run["zf"]
                size(values, 3) == length(z) || error("Unexpected vertical staggering for $variable")
                for k in eachindex(z)
                    row(io, (label, site, month, "diagnostic", variable, z[k], values[i, j, k], abspath(path)))
                end
            end
        end
        les_z = (reference["les_zf"][1:end-1] + reference["les_zf"][2:end]) / 2
        for (j, (site, month)) in enumerate(reference["members"]), (variable, name) in profile_variables,
            (k, z) in enumerate(les_z)
            row(io, ("LES", site, month, "profile", name, z, reference["targets"][j][variable][k], abspath(first(paths))))
        end
    end
    open(joinpath(output, "scores.csv"), "w") do io
        println(io, "model,site,month,variable,value,source")
        for (label, run, i, path) in sources, (j, (site, month)) in enumerate(run["members"])
            for (variable, name) in profile_variables
                row(io, (label, site, month, name * "_rmse", run["scores"][i, j][variable], abspath(path)))
            end
            row(io, (label, site, month, "cloud_water_path", run["cloud_water_path"][i, j], abspath(path)))
        end
        for (j, (site, month)) in enumerate(reference["members"])
            row(io, ("LES", site, month, "cloud_water_path", reference["les_cloud_water_path"][j], abspath(first(paths))))
        end
    end
    open(joinpath(output, "context.txt"), "w") do io
        println(io, "Protocol $(reference["protocol_version"]); dt=$(reference["Δt"]) s; radiation=$(reference["radiation_interval"]) s")
        println(io, "Resolution: $(reference["resolution"]); averaging override: $(reference["averaging_window"]); stop-time override: $(reference["stop_time"])")
        println(io, "Production terms and fluxes are averages of instantaneous quantities, not products of mean profiles.")
        println(io, "Only selected SCM TKE source/sink terms are shown; this is not a closed TKE budget or an LES dissipation estimate.")
        println(io, "Cloud water path covers the LES domain; profile scores use the saved common observation grid.")
        for (label, run, i, path) in sources
            println(io, "$label: objective=$(run["objectives"][i]); source=$(abspath(path)); status=$(run["labels"][i])")
        end
    end
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("Pass output_dir and label=diagnostics.jld2 entries")
    specifications = [Tuple(split(argument, '='; limit = 2)) for argument in ARGS[2:end]]
    all(s -> length(s) == 2, specifications) || error("Expected label=diagnostics.jld2")
    println(export_closure_diagnostics(first(ARGS), specifications))
end
