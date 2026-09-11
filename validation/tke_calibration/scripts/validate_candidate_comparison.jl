# Lightweight structural completion check; never interprets completion as numerical convergence.
# julia --compiled-modules=existing --project scripts/validate_candidate_comparison.jl \
#       output=result.jld2 snapshot=candidate.jld2 label=ri dt=7.5
using JLD2, SHA
include(joinpath(@__DIR__, "saved_metadata.jl"))

function validate_candidate_comparison(output, snapshot, label; Δt, radiation_interval = 600.0,
                                       resolutions = ["50", "100", "hindcast"],
                                       validation_sites = [3, 12, 21], reference = nothing)
    run = load(output)
    selected, protocol = jldopen(snapshot, "r") do file
        file["selected_mean"], file["protocol_version"]
    end
    isnothing(selected) && error("Snapshot has no evaluated selected_mean")
    isfinite(selected.objective) && all(isfinite, selected.parameters) || error("Nonfinite selected_mean")
    run["protocol_version"] == protocol || error("protocol_version mismatch")
    run["Δt"] == Δt || error("Δt mismatch")
    run["radiation_interval"] == radiation_interval || error("radiation_interval mismatch")
    run["validation_sites"] == validation_sites || error("validation_sites mismatch")
    run["labels"] == ["default (Nakanishi–Niino)", label] || error("labels mismatch")
    params = run["params"]
    size(params) == (length(selected.parameters), 2) || error("params shape mismatch")
    isequal(params[:, 2], vec(selected.parameters)) || error("params differ from frozen candidate")
    all(isfinite, params) || error("Nonfinite params")
    manifest = saved_metadata_value(run["comparison_manifest"])
    manifest.complete === true || error("comparison_manifest incomplete")
    manifest.architecture == "gpu" || error("architecture is not gpu")
    digest = open(io -> bytes2hex(sha256(io)), snapshot)
    manifest.source_hashes == [(; label, sha256 = digest)] || error("source_hashes mismatch")
    isempty(manifest.source_code.revision) && error("Missing source_code revision")
    length(manifest.source_code.diff_sha256) == 64 || error("Invalid source_code diff_sha256")
    expected_members = [(site, month) for site in validation_sites for month in ("01", "04", "07", "10")]
    members = run["members"]
    length(members) == length(expected_members) && Set(members) == Set(expected_members) ||
        error("members mismatch")
    data = manifest.validation_data
    length(data.gcm_sha256) == 64 || error("Invalid validation_data gcm_sha256")
    [(item.site, item.month) for item in data.les] == members || error("validation_data members mismatch")
    all(item -> length(item.sha256) == 64, data.les) || error("Invalid validation_data LES hash")
    Set(keys(run["results"])) == Set(resolutions) || error("resolutions mismatch")
    variables = (:θˡ, :qᵗ, :qˡ, :qʳ, :u, :v)
    for resolution in resolutions
        result = run["results"][resolution]
        zf = result.zf
        length(zf) >= 2 && all(isfinite, zf) && all(>(0), diff(zf)) || error("Invalid zf at $resolution")
        size(result.scores) == (2, length(members)) || error("scores shape mismatch at $resolution")
        for score in result.scores, variable in (variables..., :wind)
            value = getproperty(score, variable)
            isfinite(value) && value >= 0 || error("Invalid score $variable at $resolution")
        end
        for variable in variables
            profile = getproperty(result.means, variable)
            size(profile) == (2, length(members), length(zf) - 1) || error("means shape mismatch at $resolution")
            all(isfinite, profile) || error("Nonfinite means $variable at $resolution")
        end
    end
    if !isnothing(reference)
        other = load(reference)
        for key in ("protocol_version", "radiation_interval", "validation_sites", "labels", "members", "params", "comparison_manifest")
            isequal(saved_metadata_value(run[key]), saved_metadata_value(other[key])) ||
                error("Paired outputs differ in $key")
        end
        Set(keys(other["results"])) == Set(resolutions) || error("Paired outputs differ in resolutions")
        for resolution in resolutions
            isequal(run["results"][resolution].zf, other["results"][resolution].zf) ||
                error("Paired outputs differ in zf at $resolution")
        end
    end
    return manifest
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = Dict(split(argument, '='; limit = 2) for argument in ARGS)
    validate_candidate_comparison(options["output"], options["snapshot"], options["label"];
        Δt = parse(Float64, options["dt"]),
        radiation_interval = parse(Float64, get(options, "radiation_interval", "600")),
        resolutions = split(get(options, "resolutions", "50,100,hindcast"), ','),
        reference = get(options, "reference", nothing))
    println("COMPLETE: finite payload, frozen candidate and requested settings verified; numerical acceptance remains unevaluated")
end
