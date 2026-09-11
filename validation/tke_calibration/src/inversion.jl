#####
##### Ensemble Kalman inversion
#####

"""
Priors: independent constrained Gaussians (lognormal-like, positive) centered on the defaults with
relative width `spread`.
"""
function prior_distribution(space::ParameterSpace = RiDependentSpace(); center = prior_center(space), spread = 0.5)
    dists = [constrained_gaussian(String(name), getproperty(center, name), spread * getproperty(center, name), 0, Inf) for name in parameter_names(space)]
    return combine_distributions(dists)
end

#####
##### Step size
#####
#
# The pseudo time step is EnsembleKalmanProcesses' default for `Inversion()`, the `DataMisfitController`
# of Iglesias and Yang (2021): Bayesian tempering, in which Δtₙ = max(M / 2Φ̄, √(M / 2 var Φ)) with
# Φⱼ = ½‖Γ^{-1/2}(G(θⱼ) − y)‖² the misfit of member j and M the number of observations, and the algorithm
# terminates once the pseudo time reaches T = 1. This is a tempering budget, not a test of
# optimization stationarity or calibrated posterior uncertainty. The step is thus related to
# the inverse mean squared normalized misfit: small while the ensemble is far from
# the data and growing as it converges, so the number of iterations to termination is not fixed in advance.

# A scheduler with `T` units of pseudo time already spent: only the terminating one needs to know
remaining(scheduler, T) = scheduler
remaining(s::DataMisfitController, T) = DataMisfitController(terminate_at = s.terminate_at - T, on_terminate = s.on_terminate)

#####
##### Running and resuming the inversion
#####

build_process(initial, y, Γ; rng, scheduler, accelerator, localization_method) =
    EnsembleKalmanProcess(initial, y, Γ, Inversion(); rng, scheduler, accelerator, localization_method)

unconstrained(prior, ϕ) = transform_constrained_to_unconstrained(prior, ϕ)

const checkpoint_fields = (:iteration, :ϕ, :G, :misfit, :wall, :Δt, :pseudotime)

# Snapshot configuration before the scheduler and accelerator acquire mutable iteration state.
configuration_signature(x::Union{Number, AbstractString, Symbol, Nothing}) = x
configuration_signature(x::Type) = string(x)
configuration_signature(x::Tuple) = map(configuration_signature, x)
configuration_signature(x::AbstractArray) = (; type = string(typeof(x)), size = size(x),
                                             values = Tuple(configuration_signature(v) for v in x))
configuration_signature(x) = (; type = string(typeof(x)),
                               fields = NamedTuple{fieldnames(typeof(x))}(Tuple(configuration_signature(getfield(x, n)) for n in fieldnames(typeof(x)))))

function algorithm_configuration(scheduler, accelerator, localization_method)
    return (; scheduler = configuration_signature(scheduler),
              accelerator = configuration_signature(accelerator),
              localization = configuration_signature(localization_method))
end

# The extra forward column measures the objective at the constrained ensemble mean;
# it must never participate in the empirical covariance or the EKI update.
function ensemble_mean_diagnostics(ϕ, augmented_G, y, Γ)
    N = size(ϕ, 2)
    size(augmented_G, 2) == N + 1 || error("Expected one diagnostic column after the EKI ensemble")
    mean_G = copy(augmented_G[:, end])
    mean_parameters = vec(mean(ϕ; dims = 2))
    residual = (mean_G - y) ./ sqrt.(diag(Γ))
    mean_objective = mean(abs2, residual) / 2
    return copy(augmented_G[:, 1:N]), (; mean_parameters, mean_G, mean_objective,
                                       mean_misfit = sqrt(2mean_objective))
end

function mean_objective_plateau(history; patience = 3, minimum_iterations = 6,
                               relative_tolerance = 0.005, minimum_pseudotime = 1)
    eligible = [h.mean_objective for h in history if
                haskey(h, :mean_objective) && h.evaluation_pseudotime >= minimum_pseudotime]
    length(eligible) >= max(minimum_iterations, patience + 1) || return false
    all(isfinite, eligible) || return false
    recent = eligible[end-patience:end]
    best = minimum(eligible)
    scale = max(abs(best), eps(Float64))
    # Require both a stable recent objective and proximity to the best evaluated mean.
    return maximum(recent) - minimum(recent) <= relative_tolerance * scale &&
           maximum(recent) - best <= relative_tolerance * scale
end

"""
Rebuild an ensemble Kalman process from a saved `history` of (constrained ensemble, forward map) pairs by
replaying the saved forward maps through the update. The update is deterministic (no stochastic
perturbation with an accelerator, and both `SECNice` and the Nesterov accelerator are deterministic), so
the replay reproduces the state of the original process — including the accelerator's momentum and the
scheduler's pseudo time — which is verified against the saved ensembles at every step. If the replay drifts
from what was saved, the process is rebuilt from the last saved ensemble instead, losing only momentum.

Returns the process and the history with the pseudo-time bookkeeping of the replay filled in.
"""
function replay(prior, history, y, Γ; rng, scheduler, accelerator, localization_method, tolerance = 1e-8)
    # Schedulers and accelerators carry mutable state, so each process gets its own copy
    ekp = build_process(unconstrained(prior, history[1].ϕ), y, Γ; rng, scheduler = deepcopy(scheduler), accelerator = deepcopy(accelerator), localization_method)
    replayed = []
    exact = true
    for h in history
        ϕ = get_ϕ_final(prior, ekp)
        discrepancy = maximum(abs.(ϕ .- h.ϕ) ./ abs.(h.ϕ))
        if discrepancy > tolerance
            @warn @sprintf("Replay of iteration %d differs from the checkpoint (max relative discrepancy %.2e)", h.iteration, discrepancy)
            exact = false
        end
        if get(h, :applied_update, true)
            terminate = update_ensemble!(ekp, h.G)
            Δt = isnothing(terminate) ? get_Δt(ekp)[end] : NaN
        else
            Δt = h.Δt
        end
        push!(replayed, merge(h, (; Δt, pseudotime = sum(get_Δt(ekp)))))
    end

    if !exact # start afresh from the last saved ensemble, applying its saved update
        T = sum(get_Δt(ekp)[1:end-1])
        ekp = build_process(unconstrained(prior, history[end].ϕ), y, Γ; rng, scheduler = remaining(deepcopy(scheduler), T), accelerator = deepcopy(accelerator), localization_method)
        update_ensemble!(ekp, history[end].G)
        @info @sprintf("Rebuilt the process from the ensemble of iteration %d at pseudo time %.3f", history[end].iteration, T)
    end

    return ekp, replayed
end

function checkpoint_protocol(problem, space, y, Γ, radiation)
    return (; protocol_version = PROTOCOL_VERSION, y, Γ = diag(Γ), space = summary(space),
              parameter_names = collect(String.(parameter_names(space))),
              members = [(m.site, m.month) for m in members(problem)],
              z_faces = [p.zf for p in problems(problem)],
              observation_faces = first(problems(problem)).observation_zf,
              variables = collect(String.(first(problems(problem)).variables)),
              radiation = String(radiation), top = first(problems(problem)).zf[end])
end

function validate_checkpoint(saved, protocol, run_configuration; algorithm = nothing)
    for (key, value) in pairs(protocol)
        name = String(key)
        haskey(saved, name) || error("Checkpoint lacks $name; its calibration protocol cannot be verified")
        isequal(saved[name], value) || error("Checkpoint $name differs from the requested calibration protocol; start a new calibration")
    end
    haskey(saved, "run_configuration") || error("Checkpoint lacks run configuration; its calibration protocol cannot be verified")
    isequal(saved["run_configuration"], run_configuration) ||
        error("Checkpoint run configuration differs (time step, radiation interval, windows, prior, or static stability); start a new calibration")
    if !isnothing(algorithm)
        haskey(saved, "algorithm_configuration") || error("Checkpoint lacks algorithm configuration; its optimizer cannot be verified")
        isequal(saved["algorithm_configuration"], algorithm) ||
            error("Checkpoint optimizer differs (scheduler, accelerator, or localization); start a new calibration")
    end
    return nothing
end

function checkpoint!(output, history, ekp, y, Γ, problem, space, σ, radiation, run_configuration, algorithm)
    temporary = output * ".tmp"
    candidates = [h for h in history if haskey(h, :mean_objective) && isfinite(h.mean_objective)]
    selected = isempty(candidates) ? nothing : candidates[argmin([h.mean_objective for h in candidates])]
    jldsave(temporary; protocol_version = PROTOCOL_VERSION, checkpoint_format_version = 2,
                    algorithm_configuration = algorithm, history, y, Γ = diag(Γ), space = summary(space), σ = collect(Float64, values(σ)), parameter_names = collect(String.(parameter_names(space))),
                    members = [(m.site, m.month) for m in members(problem)],
                    z_faces = [p.zf for p in problems(problem)], observation_faces = first(problems(problem)).observation_zf,
                    variables = collect(String.(first(problems(problem)).variables)), radiation = String(radiation),
                    top = first(problems(problem)).zf[end],
                    run_configuration, selected_mean = isnothing(selected) ? nothing :
                    (; parameters = selected.mean_parameters, G = selected.mean_G,
                       objective = selected.mean_objective, iteration = selected.iteration),
                    u_final = get_u_final(ekp), Δts = get_Δt(ekp))
    mv(temporary, output; force = true)
    return nothing
end

"""
Run EKI over the parameters of `space` with `N_ens` parameter sets until the pseudo time reaches
`target_pseudotime` (1 by default, a tempering budget rather than an optimality criterion), each iteration one
column-ensemble model run per grid of the training `problem` on `architecture`, saving every iteration's
parameters, forward map, misfit and pseudo time step to `output`. The step size is set by `scheduler`, by
default EnsembleKalmanProcesses' adaptive `DataMisfitController`, and the number of iterations follows from
it; `max_iterations` is only a safety cap.

Pass `resume = "results/eki.jld2"` to continue from a checkpoint: the saved forward maps are replayed
through the update (see [`replay`](@ref)) and the run continues toward the target.
The protocol version, targets, noise, members, grids, radiation, run configuration, and optimizer must agree.
For a short diagnostic run, pass both `stop_time` and an `averaging_window` within that run.
"""
function run_eki(problem::AnyProblem; space = RiDependentSpace(), N_ens = 20, target_pseudotime = 1, max_iterations = 50,
                 rng = MersenneTwister(1), σ = default_observation_noise, spread = 0.5, output = "results/eki.jld2",
                 Δt = 1minute, architecture = CPU(), stop_time = nothing, averaging_window = nothing, radiation = :interactive,
                 radiation_interval = 10minutes, upper_relaxation_rate = 1 / 600,
                 optimize = false, objective_tolerance = 0.005, objective_patience = 3,
                 minimum_optimization_iterations = 6,
                 scheduler = DataMisfitController(terminate_at = target_pseudotime, on_terminate = optimize ? "continue" : "stop"),
                 accelerator = NesterovAccelerator(),
                 localization_method = SECNice(),
                 resume = nothing)

    prior = prior_distribution(space; spread)
    y, Γ = observations(problem; σ)
    protocol = checkpoint_protocol(problem, space, y, Γ, radiation)
    run_configuration = (; Δt, radiation_interval, upper_relaxation_rate, stop_time, averaging_window, spread,
                           windows = [m.window for m in members(problem)],
                           static_stability = [p.static_stability for p in problems(problem)])
    stopping = (; optimize, objective_tolerance, objective_patience, minimum_optimization_iterations,
                  target_pseudotime)
    algorithm = merge(algorithm_configuration(scheduler, accelerator, localization_method), (; stopping))
    isempty(dirname(output)) || mkpath(dirname(output))

    if isnothing(resume)
        initial = construct_initial_ensemble(rng, prior, N_ens)
        ekp = build_process(initial, y, Γ; rng, scheduler, accelerator, localization_method)
        history = []
    else
        checkpoint = load(resume)
        validate_checkpoint(checkpoint, protocol, run_configuration; algorithm)
        saved = checkpoint["history"]
        size(saved[1].ϕ, 1) == length(parameter_names(space)) || error("The checkpoint has $(size(saved[1].ϕ, 1)) parameters; $space has $(length(parameter_names(space)))")
        @info "Resuming from $resume after $(length(saved)) iterations"
        ekp, history = replay(prior, saved, y, Γ; rng, scheduler, accelerator, localization_method)
        N_ens = get_N_ens(ekp)
    end

    terminated = false
    for _ in 1:max_iterations
        n = length(history) + 1
        ϕ = get_ϕ_final(prior, ekp)                       # constrained parameters, (N_params, N_ens)
        evaluation_pseudotime = sum(get_Δt(ekp))
        augmented = hcat(ϕ, mean(ϕ; dims = 2))
        wall = @elapsed augmented_G, means = forward_map(problem, augmented; space, Δt, architecture,
                              stop_time, averaging_window, radiation, radiation_interval, upper_relaxation_rate)
        G, diagnostics = ensemble_mean_diagnostics(ϕ, augmented_G, y, Γ)
        misfit = [sqrt(mean(((G[:, i] .- y) ./ sqrt.(diag(Γ))) .^ 2)) for i in 1:N_ens]
        best = argmin(misfit)

        evaluated = merge((; iteration = n, ϕ = copy(ϕ), G, misfit, wall, evaluation_pseudotime), diagnostics)
        plateau = optimize && mean_objective_plateau(vcat(history, [evaluated]);
                  patience = objective_patience, minimum_iterations = minimum_optimization_iterations,
                  relative_tolerance = objective_tolerance, minimum_pseudotime = target_pseudotime)
        terminate = plateau ? true : update_ensemble!(ekp, G)
        applied_update = isnothing(terminate)
        Δtₙ = plateau ? 0.0 : applied_update ? get_Δt(ekp)[end] : NaN
        T = sum(get_Δt(ekp))

        @info @sprintf("EKI iteration %d: forward map %.0f s; normalized misfit mean %.2f, best %.2f (member %d); Δt = %.4f, pseudo time %.3f",
                       n, wall, mean(misfit), misfit[best], best, Δtₙ, T)
        @info "  ensemble mean parameters: " * join([@sprintf("%s = %.3f", name, mean(ϕ[k, :])) for (k, name) in enumerate(parameter_names(space))], ", ")
        @info @sprintf("  evaluated mean: objective %.8g, normalized RMS %.6f (evaluation pseudo time %.3f)",
                       diagnostics.mean_objective, diagnostics.mean_misfit, evaluation_pseudotime)

        stop_reason = plateau ? :mean_objective_plateau : applied_update ? :none : :tempering_budget
        push!(history, merge(evaluated, (; Δt = Δtₙ, pseudotime = T, applied_update, stop_reason)))
        checkpoint!(output, history, ekp, y, Γ, problem, space, σ, radiation, run_configuration, algorithm)
        flush(stdout); flush(stderr)      # Julia buffers both when they are redirected to a file

        if !isnothing(terminate)
            @info "Stopped at iteration $n: $stop_reason; inspect numerical, seed and ensemble-size convergence before adopting coefficients"
            terminated = true
            break
        end
    end
    terminated || @warn "Stopped after $max_iterations iterations at pseudo time $(sum(get_Δt(ekp))) without reaching the target"

    ϕ_final = get_ϕ_final(prior, ekp)
    return ekp, prior, ϕ_final, history
end
