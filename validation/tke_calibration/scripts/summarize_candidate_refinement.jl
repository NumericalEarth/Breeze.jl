# Analyze a completed frozen-candidate pair without Breeze, mutable target files or another GPU run.
# julia --compiled-modules=existing --project scripts/summarize_candidate_refinement.jl \
#       output=dt75.jld2 reference=dt375.jld2 snapshot=candidate.jld2 label=ri prefix=refinement_ri
using Statistics, Printf
include(joinpath(@__DIR__, "validate_candidate_comparison.jl"))

function candidate_refinement_rows(run, reference; profile_tolerance = 0.1, objective_tolerance = 0.01)
    resolutions = sort!(collect(keys(run["results"])))
    Set(resolutions) == Set(keys(reference["results"])) || error("Different resolutions")
    run["labels"] == reference["labels"] || error("Different labels")
    run["members"] == reference["members"] || error("Different members")
    number_sets, number_members = length(run["labels"]), length(run["members"])
    data_pairs = map(resolutions) do resolution
        a = run["results"][resolution].observation_data
        b = reference["results"][resolution].observation_data
        for key in (:y, :variance, :variables, :zf)
            isequal(getproperty(a, key), getproperty(b, key)) || error("Different observation $key at $resolution")
        end
        a.variables == (:θˡ, :qᵗ, :qˡ, :u, :v) || error("Unexpected observation variables")
        length(a.zf) >= 2 && all(isfinite, a.zf) && all(>(0), diff(a.zf)) || error("Invalid observation zf")
        observations = number_members * length(a.variables) * (length(a.zf) - 1)
        length(a.y) == length(a.variance) == observations || error("Observation vector length mismatch")
        size(a.G) == size(b.G) == (observations, number_sets) || error("Observation G shape mismatch")
        all(isfinite, a.y) && all(x -> isfinite(x) && x > 0, a.variance) || error("Invalid targets or variance")
        all(isfinite, a.G) && all(isfinite, b.G) || error("Nonfinite observation G")
        (; a, b)
    end
    rows = map(Iterators.product(eachindex(run["labels"]), vcat(resolutions, ["pooled"]))) do (i, resolution)
        selected = resolution == "pooled" ? eachindex(resolutions) : [findfirst(==(resolution), resolutions)]
        normalized = [((data_pairs[k].a.G[:, i] - data_pairs[k].b.G[:, i]) ./ sqrt.(data_pairs[k].a.variance)) for k in selected]
        differences = vcat(normalized...)
        residual_a = vcat([((data_pairs[k].a.G[:, i] - data_pairs[k].a.y) ./ sqrt.(data_pairs[k].a.variance)) for k in selected]...)
        residual_b = vcat([((data_pairs[k].b.G[:, i] - data_pairs[k].b.y) ./ sqrt.(data_pairs[k].b.variance)) for k in selected]...)
        objective = mean(abs2, residual_a) / 2
        reference_objective = mean(abs2, residual_b) / 2
        relative_change = reference_objective == 0 ? (objective == 0 ? 0.0 : Inf) :
                          abs(objective - reference_objective) / reference_objective
        rms = sqrt(mean(abs2, differences))
        worst_block = argmax([maximum(abs, values) for values in normalized])
        k = selected[worst_block]
        a = data_pairs[k].a
        # observation_vector flattens heights, then variables, then members; grids concatenate last.
        index = argmax(abs.(normalized[worst_block])) - 1
        cells, variables = length(a.zf) - 1, length(a.variables)
        height_index = index % cells + 1
        variable_index = (index ÷ cells) % variables + 1
        member_index = index ÷ (cells * variables) + 1
        site, month = run["members"][member_index]
        (; label = run["labels"][i], resolution, objective, reference_objective, relative_change,
           rms_noise = rms, max_noise = maximum(abs, differences),
           worst_grid = resolutions[k], site, month, variable = a.variables[variable_index],
           z_m = (a.zf[height_index] + a.zf[height_index + 1]) / 2,
           aggregate_target_met = rms <= profile_tolerance && relative_change <= objective_tolerance)
    end
    return vec(rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = Dict(split(argument, '='; limit = 2) for argument in ARGS)
    output, reference, snapshot, label = (options[key] for key in ("output", "reference", "snapshot", "label"))
    Δt = parse(Float64, get(options, "dt", "7.5"))
    reference_Δt = parse(Float64, get(options, "reference_dt", "3.75"))
    Δt > reference_Δt > 0 || error("Reference must use a smaller positive timestep")
    interval = parse(Float64, get(options, "radiation_interval", "600"))
    validate_candidate_comparison(reference, snapshot, label; Δt = reference_Δt, radiation_interval = interval)
    validate_candidate_comparison(output, snapshot, label; Δt, radiation_interval = interval, reference)
    rows = candidate_refinement_rows(load(output), load(reference))
    prefix = get(options, "prefix", "candidate_refinement_$label")
    digest = open(io -> bytes2hex(sha256(io)), snapshot)
    provenance = (; dt_s = Δt, reference_dt_s = reference_Δt, radiation_interval_s = interval,
                    snapshot_sha256 = digest)
    mkpath(dirname(abspath(prefix)))
    open(prefix * ".csv", "w") do io
        println(io, join(keys(merge(first(rows), provenance)), ','))
        for row in rows
            println(io, join(["\"" * replace(string(value), "\"" => "\"\"") * "\"" for value in values(merge(row, provenance))], ','))
        end
    end
    open(prefix * ".txt", "w") do io
        println(io, "Frozen $label: dt=$Δt versus $reference_Δt s, radiation=$interval s")
        println(io, "Snapshot: $(abspath(snapshot))")
        println(io, "Evaluations: $(abspath(output)); $(abspath(reference))")
        println(io, "Targets: RMS <= 0.1 noise units AND relative objective change <= 1%.")
        println(io, "These aggregate sensitivity targets do not establish pointwise or asymptotic convergence.")
        for row in rows
            @printf io "%s / %s: Phi %.8g vs %.8g; change %.3f%%; RMS %.5f sigma; max %.5f sigma\n" row.label row.resolution row.objective row.reference_objective 100row.relative_change row.rms_noise row.max_noise
            println(io, "  worst: grid=$(row.worst_grid), site=$(row.site), month=$(row.month), variable=$(row.variable), z=$(row.z_m) m")
            println(io, "  aggregate target met: $(row.aggregate_target_met)")
        end
        println(io, "Assess the fitted candidate, per-grid differences and worst profiles; if borderline or nonmonotone, check 1.875 s with radiation held fixed.")
    end
    print(read(prefix * ".txt", String))
end
