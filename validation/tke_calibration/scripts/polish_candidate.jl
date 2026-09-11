# Direct local checks/refinement of an EKI candidate, in positive log-parameter coordinates.
# Every coordinate direction plus fixed random directions is evaluated in one GPU ensemble.
# julia --project scripts/polish_candidate.jl checkpoint=... output=... [radius=0.1]
#   [minimum_radius=0.0125] [max_iterations=12] [random_directions=8] [seed=41] [arch=gpu]
#   [selection=mean|best_evaluated] -- the latter also searches all previously evaluated members
using BreezeCalibration, JLD2, Statistics, LinearAlgebra, Random, Printf, CUDA
using Oceananigans: CPU, GPU
include(joinpath(@__DIR__, "evaluated_candidate.jl"))
include(joinpath(@__DIR__, "calibration_data_manifest.jl"))

function main()
    options = Dict(split(a, '='; limit = 2) for a in ARGS)
    source = load(options["checkpoint"])
    source["protocol_version"] == PROTOCOL_VERSION || error("Physics protocol mismatch")
    validate_data_manifest(source)
    starting_candidate = evaluated_candidate(source; selection = Symbol(get(options, "selection", "mean")))
    parameters = copy(starting_candidate.parameters)
    all(>(0), parameters) || error("Log-coordinate refinement requires positive parameters")
    space = space_of(length(parameters))
    members = [load_member(s, m) for (s, m) in source["members"]]
    configuration = source["run_configuration"]
    variables = Tuple(Symbol.(source["variables"]))
    problems = [ColumnEnsembleProblem(members; z_faces = faces, top = last(faces), variables,
                 observation_faces = source["observation_faces"],
                 static_stability = configuration.static_stability[k]) for (k, faces) in enumerate(source["z_faces"])]
    problem = length(problems) == 1 ? only(problems) : MultiResolutionProblem(problems)
    y, Γ = source["y"], Diagonal(source["Γ"])
    protocol = BreezeCalibration.checkpoint_protocol(problem, space, y, Γ, Symbol(source["radiation"]))
    BreezeCalibration.validate_checkpoint(source, protocol, configuration)
    architecture_name = get(options, "arch", "gpu")
    architecture_name in ("cpu", "gpu") || error("arch must be gpu or cpu")
    architecture = architecture_name == "gpu" ? GPU() : CPU()
    if architecture isa GPU
        CUDA.functional() || error("CUDA is required")
        CUDA.allowscalar(false)
        println("GPU: ", CUDA.name(CUDA.device()))
    end
    radius = parse(Float64, get(options, "radius", "0.1"))
    minimum_radius = parse(Float64, get(options, "minimum_radius", "0.0125"))
    max_iterations = parse(Int, get(options, "max_iterations", "12"))
    number_of_random_directions = parse(Int, get(options, "random_directions", "8"))
    relative_tolerance = parse(Float64, get(options, "tolerance", "0.0001"))
    0 < minimum_radius <= radius || error("Require 0 < minimum_radius <= radius")
    max_iterations > 0 || error("max_iterations must be positive")
    number_of_random_directions >= 0 || error("random_directions must be nonnegative")
    relative_tolerance > 0 || error("tolerance must be positive")
    rng = MersenneTwister(parse(Int, get(options, "seed", "41")))
    random = randn(rng, length(parameters), number_of_random_directions)
    random ./= sqrt.(sum(abs2, random; dims = 1))
    directions = hcat(Matrix{Float64}(I, length(parameters), length(parameters)), random)
    directions = hcat(zeros(length(parameters)), directions, -directions)
    history = []
    output = options["output"]; mkpath(dirname(abspath(output)))
    best_G = copy(starting_candidate.G)
    best_objective = starting_candidate.objective
    converged = false

    for iteration in 1:max_iterations
        candidates = parameters .* exp.(radius .* directions)
        wall = @elapsed G, _ = forward_map(problem, candidates; space, architecture,
                          Δt = configuration.Δt, radiation_interval = configuration.radiation_interval,
                          upper_relaxation_rate = configuration.upper_relaxation_rate,
                          stop_time = configuration.stop_time, averaging_window = configuration.averaging_window,
                          radiation = Symbol(source["radiation"]))
        objectives = vec(mean(abs2, (G .- y) ./ sqrt.(diag(Γ)); dims = 1)) / 2
        all(isfinite, objectives) || error("A local perturbation produced a nonfinite objective")
        if iteration == 1
            isapprox(objectives[1], best_objective; rtol = 1e-8, atol = 1e-10) ||
                error("The source candidate does not reproduce its saved objective")
        end
        selected = argmin(objectives)
        improvement = objectives[1] - objectives[selected]
        accepted = improvement > relative_tolerance * max(objectives[1], eps())
        push!(history, (; iteration, center = copy(parameters), radius, candidates, objectives, G, wall, selected, accepted))
        if accepted
            parameters = copy(candidates[:, selected])
            best_G = copy(G[:, selected])
            best_objective = objectives[selected]
        elseif radius <= minimum_radius
            converged = true
        else
            radius = max(radius / 2, minimum_radius)
        end
        jldsave(output * ".tmp"; protocol_version = PROTOCOL_VERSION, source_checkpoint = abspath(options["checkpoint"]),
                 starting_candidate,
                 parameters, parameter_names = source["parameter_names"], objective = best_objective,
                 G = best_G, history, converged, run_configuration = configuration,
                 members = source["members"], z_faces = source["z_faces"], relative_tolerance,
                 minimum_radius, directions)
        mv(output * ".tmp", output; force = true)
        @printf "Local check %d: center %.8g, best %.8g, relative gain %.4g, accepted %s, next radius %.4g\n" iteration objectives[1] objectives[selected] improvement / objectives[1] accepted radius
        flush(stdout)
        converged && break
    end
    println(converged ? "No material improvement in the tested local directions at the minimum radius." :
                        "Iteration cap reached; local convergence is not established.")
end

main()
