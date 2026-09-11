# Export the scored profiles and their numerical differences without rerunning the model.
# julia --project scripts/export_discretization_profiles.jl input=results/discretization_sensitivity.jld2 output=results/discretization_profiles.csv family=dt
using BreezeCalibration, JLD2, Printf, Statistics

options = Dict(split(a, '='; limit = 2) for a in ARGS)
saved = load(options["input"])
family = get(options, "family", "dt")
runs = saved[family == "dt" ? "dt_means" : "interval_means"]
isempty(runs) && error("No $family runs in this file")
settings = sort(collect(keys(runs)))
reference = first(settings)
faces = saved["observation_zf"]
z = (faces[1:end-1] + faces[2:end]) / 2
output = options["output"]
mkpath(dirname(abspath(output)))
scored = (:θˡ, :qᵗ, :qˡ, :u, :v)
names = ("theta_l", "total_water", "cloud_water", "u", "v")
println("Reference $family = $reference; protocol ", saved["protocol_version"])
open(output, "w") do io
    println(io, "protocol,family,setting,reference,site,month,variable,z,profile,target,difference,normalized_difference")
    for (j, (site, month)) in enumerate(saved["members"])
        member = load_member(site, month)
        for (v, name) in zip(scored, names)
            scale = BreezeCalibration.observation_scales[v]
            target = scale .* regrid_column(member.targets[v], les_faces(member.z), faces)
            ref = scale .* regrid_column(runs[reference][v][1, j, :], saved["zf"], faces)
            for setting in settings
                profile = scale .* regrid_column(runs[setting][v][1, j, :], saved["zf"], faces)
                difference = profile - ref
                normalized = difference ./ default_observation_noise[v]
                k = argmax(abs.(normalized))
                @printf "%d/%s %-12s %7g: RMS %.4f, max %.3f sigma at %.0f m\n" site month name setting sqrt(mean(normalized .^ 2)) normalized[k] z[k]
                for k in eachindex(z)
                    println(io, join((saved["protocol_version"], family, setting, reference, site, month, name,
                                      z[k], profile[k], target[k], difference[k], normalized[k]), ','))
                end
            end
        end
    end
end
println("Wrote $output")
