# Save scored profiles and instantaneous closure-budget averages for a candidate and its baseline.
# julia --project scripts/diagnose_candidate.jl checkpoint=... output=... resolution=50
#   members=22/07,17/07,14/01,2/01 [arch=gpu] [dt=...] [radiation_interval=...]
# Reserved test sites require reveal_reserved=true, after the fitting decisions are frozen.
using BreezeCalibration, JLD2, Statistics, NCDatasets, CUDA
using Oceananigans: CPU, GPU, interior
include(joinpath(@__DIR__, "closure_diagnostics.jl"))
include(joinpath(@__DIR__, "calibration_data_manifest.jl"))

function main()
    options = Dict(split(a, '='; limit = 2) for a in ARGS)
    source_path = options["checkpoint"]
    source = load(source_path)
    if haskey(source, "parameters") && haskey(source, "source_checkpoint")
        candidate = source["parameters"]
        metadata = load(source["source_checkpoint"])
        status = get(source, "converged", false) ? "candidate passing local direction checks" : "candidate with incomplete local refinement"
    else
        metadata = source
        selected = get(source, "selected_mean", nothing)
        candidate = isnothing(selected) ? vec(mean(source["history"][end].ϕ; dims = 2)) : selected.parameters
        status = isnothing(selected) ? "exploratory final mean" : "best directly evaluated mean candidate"
    end
    source["protocol_version"] == metadata["protocol_version"] == PROTOCOL_VERSION || error("Physics protocol mismatch")
    validate_data_manifest(metadata)
    configuration = metadata["run_configuration"]
    space = space_of(length(candidate))
    member_ids = [(parse(Int, first(split(s, '/'))), String(last(split(s, '/'))))
                  for s in split(get(options, "members", "22/07,17/07,14/01,2/01"), ',')]
    reserved = any(s in (6, 9, 15, 18) for (s, _) in member_ids)
    reserved && get(options, "reveal_reserved", "false") != "true" &&
        error("Reserved test cases require reveal_reserved=true after coefficients are frozen")
    members = [load_member(s, m) for (s, m) in member_ids]
    resolution = get(options, "resolution", "50")
    variables = Tuple(Symbol.(metadata["variables"]))
    stability = first(configuration.static_stability)
    all(isequal(stability), configuration.static_stability) || error("Source grids use different static stability models")
    common = (; top = metadata["top"], variables, observation_faces = metadata["observation_faces"], static_stability = stability)
    problem = resolution == "hindcast" ? ColumnEnsembleProblem(members; z_faces = hindcast_faces(), common...) :
              resolution == "20" ? ColumnEnsembleProblem(members; common...) :
              ColumnEnsembleProblem(members; Δz = parse(Float64, resolution), common...)
    Δt = parse(Float64, get(options, "dt", string(configuration.Δt)))
    radiation_interval = parse(Float64, get(options, "radiation_interval", string(configuration.radiation_interval)))
    stop_time = haskey(options, "stop_time") ? parse(Float64, options["stop_time"]) : configuration.stop_time
    averaging_window = haskey(options, "stop_time") ? (0.0, stop_time) : configuration.averaging_window
    haskey(options, "stop_time") && @warn "Short diagnostic test: averaging from zero to $stop_time seconds"
    architecture_name = get(options, "arch", "gpu")
    architecture_name in ("gpu", "cpu") || error("arch must be gpu or cpu")
    architecture = architecture_name == "gpu" ? GPU() : CPU()
    if architecture isa GPU
        CUDA.functional() || error("CUDA is required")
        CUDA.allowscalar(false)
    end
    labels = ["Default", status]
    parameters = hcat(collect(Float64, default_parameters(space)), candidate)
    recorder = closure_diagnostic_recorder(length(members))
    wall = @elapsed means, model, timing = run_ensemble(problem, parameters; space, architecture, Δt, radiation_interval,
                       radiation = Symbol(metadata["radiation"]), upper_relaxation_rate = configuration.upper_relaxation_rate,
                       stop_time, averaging_window,
                       sample_callback = recorder.sample)
    diagnostics = recorder.result()
    fields = BreezeCalibration.observable_variables
    profiles(i, j) = NamedTuple{fields}(Tuple(means[v][i, j, :] for v in fields))
    scores = [BreezeCalibration.rmse(problem, profiles(i, j), members[j]) for i in 1:2, j in eachindex(members)]
    σ = NamedTuple{keys(default_observation_noise)}(Tuple(metadata["σ"]))
    objectives = [mean(mean((s[v] / σ[v])^2 for v in variables) for s in scores[i, :]) / 2 for i in 1:2]
    density = Array(interior(model.dynamics.reference_state.density))
    # Compare the integrated cloud mass over the common LES domain, not the extended SCM top.
    thickness = diff(clamp.(problem.zf, 0, last(problem.les_zf)))
    cloud_water_path = dropdims(sum(means.qˡ .* density .* reshape(thickness, 1, 1, :); dims = 3); dims = 3)
    les_cloud_water_path = [NCDataset(BreezeCalibration.member_path(m.site, m.month)) do ds
        sum(ds["rho0"][:] .* m.targets.qˡ .* diff(problem.les_zf))
    end for m in members]
    output = options["output"]
    mkpath(dirname(abspath(output)))
    jldsave(output * ".tmp"; protocol_version = PROTOCOL_VERSION, source_checkpoint = abspath(source_path),
             source_run_configuration = configuration, Δt, radiation_interval, stop_time, averaging_window, resolution,
             labels, parameters, parameter_names = parameter_names(space), members = member_ids,
             zf = problem.zf, zc = problem.zc, observation_zf = problem.observation_zf,
             means, diagnostics, scores, objectives, density, cloud_water_path, les_cloud_water_path,
             targets = [m.targets for m in members], les_zf = problem.les_zf, wall, timing,
             hardware = architecture isa GPU ? CUDA.name(CUDA.device()) : "CPU")
    mv(output * ".tmp", output; force = true)
    println("Saved candidate diagnostics to $output; objectives $objectives")
end

main()
