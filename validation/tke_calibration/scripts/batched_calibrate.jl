# Independent optimizers share GPU forward evaluations, never their empirical covariances.
# julia -t auto --project scripts/batched_calibrate.jl space=ri runs=200:1,400:1,400:2
#   dt=7.5 radiation_interval=600 resolutions=50,100,hindcast max_columns=32000
#   sites=2,5,8,11,14,17,20,23 months=01,04,07,10 output=results/final_ri
# Each run has its own atomic checkpoint and resumes automatically after a preemption.
using BreezeCalibration, EnsembleKalmanProcesses, JLD2, Random, Statistics, Printf, SHA, CUDA
using Oceananigans: CPU, GPU

const BC = BreezeCalibration
function main()
    options = Dict(split(a, '='; limit = 2) for a in ARGS)
    space_name = get(options, "space", "ri")
    space_name in ("ri", "constant") || error("space must be ri or constant")
    space = space_name == "ri" ? RiDependentSpace() : ConstantSpace()
    specs = [Tuple(parse.(Int, split(item, ':'))) for item in split(options["runs"], ',')]
    all(s -> length(s) == 2 && s[1] >= 2, specs) || error("runs must list Nens:seed pairs")
    length(unique(specs)) == length(specs) || error("Duplicate experiment would overwrite a checkpoint")
    sites = parse.(Int, split(get(options, "sites", "2,5,8,11,14,17,20,23"), ','))
    months = split(get(options, "months", "01,04,07,10"), ',')
    members = [load_member(s, m) for s in sites for m in months]
    resolutions = split(get(options, "resolutions", "50,100,hindcast"), ',')
    problems = [r == "hindcast" ? ColumnEnsembleProblem(members; z_faces = hindcast_faces()) :
                ColumnEnsembleProblem(members; Δz = parse(Float64, r)) for r in resolutions]
    problem = length(problems) == 1 ? only(problems) : MultiResolutionProblem(problems)
    Δt = parse(Float64, options["dt"])
    radiation_interval = parse(Float64, get(options, "radiation_interval", "600"))
    radiation = :interactive
    architecture_name = get(options, "arch", "gpu")
    architecture_name in ("gpu", "cpu") || error("arch must be gpu or cpu")
    architecture = architecture_name == "gpu" ? GPU() : CPU()
    if architecture isa GPU
        CUDA.functional() || error("CUDA is required")
        CUDA.allowscalar(false)
        @info "GPU: $(CUDA.name(CUDA.device()))"
    end
    max_columns = parse(Int, get(options, "max_columns", "32000"))
    sets_per_chunk = max_columns ÷ length(members)
    sets_per_chunk >= 1 || error("max_columns must accommodate at least one parameter set")
    max_iterations = parse(Int, get(options, "max_iterations", "40"))
    max_iterations > 0 || error("max_iterations must be positive")
    localization_name = get(options, "localization", "secnice")
    localization_name in ("secnice", "none") || error("localization must be secnice or none")
    stop_time = haskey(options, "stop_time") ? parse(Float64, options["stop_time"]) : nothing
    averaging_window = isnothing(stop_time) ? nothing : (0.0, stop_time)
    isnothing(stop_time) || @warn "Short diagnostic run: averaging from zero to $stop_time seconds"
    output = options["output"]; mkpath(output)
    σ = default_observation_noise
    spread = parse(Float64, get(options, "spread", "0.5"))
    prior = prior_distribution(space; spread)
    y, Γ = observations(problem; σ)
    stopping = (; optimize = true, objective_tolerance = parse(Float64, get(options, "tolerance", "0.005")),
                  objective_patience = 3, minimum_optimization_iterations = 6, target_pseudotime = 1.0)
    run_configuration = (; Δt, radiation_interval, upper_relaxation_rate = 1 / 600,
                           stop_time, averaging_window, spread,
                           windows = [m.window for m in members],
                           static_stability = [p.static_stability for p in problems])
    protocol = BC.checkpoint_protocol(problem, space, y, Γ, radiation)
    revision = readchomp(`git rev-parse HEAD`)
    diff_hash = bytes2hex(sha256(read(`git diff --binary`)))
    driver_hash = bytes2hex(sha256(read(@__FILE__)))

    states = map(specs) do (N_ens, seed)
        label = "n$(N_ens)_seed$(seed)"
        path = joinpath(output, "$label.jld2")
        rng = MersenneTwister(seed)
        scheduler = DataMisfitController(on_terminate = "continue")
        accelerator = NesterovAccelerator()
        localization = localization_name == "none" ? NoLocalization() : SECNice()
        algorithm = merge(BC.algorithm_configuration(scheduler, accelerator, localization), (; stopping))
        provenance = [(; revision, diff_hash, driver_hash)]
        if isfile(path)
            saved = load(path)
            BC.validate_checkpoint(saved, protocol, run_configuration; algorithm)
            saved["experiment_metadata"].seed == seed || error("Seed differs from checkpoint")
            previous = get(saved["experiment_metadata"], :provenance, provenance)
            provenance = unique(vcat(previous, provenance))
            ekp, history = replay(prior, saved["history"], y, Γ; rng, scheduler, accelerator,
                                  localization_method = localization)
            size(history[1].ϕ, 2) == N_ens || error("Ensemble size differs from checkpoint")
        else
            initial = construct_initial_ensemble(rng, prior, N_ens)
            ekp = BC.build_process(initial, y, Γ; rng, scheduler, accelerator, localization_method = localization)
            history = []
        end
        metadata = (; label, seed, N_ens, revision, diff_hash, driver_hash, provenance,
                      batched = true, wall_attribution = "shared forward seconds times fraction of parameter columns")
        (; label, path, ekp, history, algorithm, metadata)
    end

    while true
        active = [state for state in states if length(state.history) < max_iterations &&
                  (isempty(state.history) || state.history[end].stop_reason == :none)]
        isempty(active) && break
        ensembles = [get_ϕ_final(prior, state.ekp) for state in active]
        parameters, ranges = BC.batched_forward_parameters(ensembles)
        G = Matrix{Float64}(undef, length(y), size(parameters, 2))
        @info "Batched forward: $(length(active)) independent optimizers, $(size(parameters, 2)) parameter sets, $(length(members)) cases; maximum $max_columns columns per chunk"
        wall = @elapsed for first_column in 1:sets_per_chunk:size(parameters, 2)
            columns = first_column:min(first_column + sets_per_chunk - 1, size(parameters, 2))
            result, _ = forward_map(problem, parameters[:, columns]; space, Δt, radiation_interval,
                                     radiation, architecture, stop_time, averaging_window)
            G[:, columns] .= result
        end
        for (state, ϕ, columns) in zip(active, ensembles, ranges)
            @info "$(state.label): shared forward cost $wall seconds"
            attributed_wall = wall * length(columns) / size(parameters, 2)
            reason = BC.record_eki_iteration!(state.history, state.ekp, ϕ, G[:, columns], y, Γ,
                                               attributed_wall; stopping)
            state.history[end] = merge(state.history[end], (; shared_forward_wall = wall,
                                          shared_parameter_sets = size(parameters, 2)))
            io_wall = @elapsed BC.checkpoint!(state.path, state.history, state.ekp, y, Γ, problem, space, σ, radiation,
                           run_configuration, state.algorithm; experiment_metadata = state.metadata)
            @info "$(state.label): checkpoint I/O $io_wall seconds"
            reason == :none || @info "$(state.label) stopped: $reason"
            flush(stdout); flush(stderr)
        end
    end

    for state in states
        best = argmin([h.mean_objective for h in state.history])
        @printf "%s: %d evaluations, best mean objective %.8g at iteration %d; last reason %s\n" state.label length(state.history) state.history[best].mean_objective best string(state.history[end].stop_reason)
    end
end

main()
