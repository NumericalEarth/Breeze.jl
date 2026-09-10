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
# terminates once the pseudo time reaches T = 1 — at which point the ensemble approximates the
# posterior. The step is thus 1 / (mean squared normalized misfit): small while the ensemble is far from
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
        terminate = update_ensemble!(ekp, h.G)
        Δt = isnothing(terminate) ? get_Δt(ekp)[end] : NaN
        push!(replayed, (iteration = h.iteration, ϕ = h.ϕ, G = h.G, misfit = h.misfit, wall = h.wall, Δt, pseudotime = sum(get_Δt(ekp))))
    end

    if !exact # start afresh from the last saved ensemble, applying its saved update
        T = sum(get_Δt(ekp)[1:end-1])
        ekp = build_process(unconstrained(prior, history[end].ϕ), y, Γ; rng, scheduler = remaining(deepcopy(scheduler), T), accelerator = deepcopy(accelerator), localization_method)
        update_ensemble!(ekp, history[end].G)
        @info @sprintf("Rebuilt the process from the ensemble of iteration %d at pseudo time %.3f", history[end].iteration, T)
    end

    return ekp, replayed
end

function checkpoint!(output, history, ekp, y, Γ, problem, space, σ, radiation)
    jldsave(output; history, y, Γ = diag(Γ), space = summary(space), σ = collect(Float64, values(σ)), parameter_names = collect(String.(parameter_names(space))),
                    members = [(m.site, m.month) for m in members(problem)],
                    z_faces = [p.zf for p in problems(problem)], observation_faces = first(problems(problem)).observation_zf,
                    variables = collect(String.(first(problems(problem)).variables)), radiation = String(radiation),
                    top = first(problems(problem)).zf[end],
                    u_final = get_u_final(ekp), Δts = get_Δt(ekp))
    return nothing
end

"""
Run EKI over the parameters of `space` with `N_ens` parameter sets until the pseudo time reaches
`target_pseudotime` (1 by default, where the ensemble approximates the posterior), each iteration one
column-ensemble model run per grid of the training `problem` on `architecture`, saving every iteration's
parameters, forward map, misfit and pseudo time step to `output`. The step size is set by `scheduler`, by
default EnsembleKalmanProcesses' adaptive `DataMisfitController`, and the number of iterations follows from
it; `max_iterations` is only a safety cap.

Pass `resume = "results/eki.jld2"` to continue from a checkpoint: the saved forward maps are replayed
through the update (see [`replay`](@ref)) and the run continues toward the target.
"""
function run_eki(problem::AnyProblem; space = RiDependentSpace(), N_ens = 20, target_pseudotime = 1, max_iterations = 50,
                 rng = MersenneTwister(1), σ = default_observation_noise, spread = 0.5, output = "results/eki.jld2",
                 Δt = 1minute, architecture = CPU(), stop_time = nothing, radiation = :interactive,
                 scheduler = DataMisfitController(terminate_at = target_pseudotime),
                 accelerator = NesterovAccelerator(),
                 localization_method = SECNice(),
                 resume = nothing)

    prior = prior_distribution(space; spread)
    y, Γ = observations(problem; σ)
    isempty(dirname(output)) || mkpath(dirname(output))

    if isnothing(resume)
        initial = construct_initial_ensemble(rng, prior, N_ens)
        ekp = build_process(initial, y, Γ; rng, scheduler, accelerator, localization_method)
        history = []
    else
        saved = load(resume, "history")
        size(saved[1].ϕ, 1) == length(parameter_names(space)) || error("The checkpoint has $(size(saved[1].ϕ, 1)) parameters; $space has $(length(parameter_names(space)))")
        @info "Resuming from $resume after $(length(saved)) iterations"
        ekp, history = replay(prior, saved, y, Γ; rng, scheduler, accelerator, localization_method)
        N_ens = get_N_ens(ekp)
    end

    terminated = false
    for _ in 1:max_iterations
        n = length(history) + 1
        ϕ = get_ϕ_final(prior, ekp)                       # constrained parameters, (N_params, N_ens)
        wall = @elapsed G, means = forward_map(problem, ϕ; space, Δt, architecture, stop_time, radiation)
        misfit = [sqrt(mean(((G[:, i] .- y) ./ sqrt.(diag(Γ))) .^ 2)) for i in 1:N_ens]
        best = argmin(misfit)

        terminate = update_ensemble!(ekp, G)
        Δtₙ = isnothing(terminate) ? get_Δt(ekp)[end] : NaN
        T = sum(get_Δt(ekp))

        @info @sprintf("EKI iteration %d: forward map %.0f s; normalized misfit mean %.2f, best %.2f (member %d); Δt = %.4f, pseudo time %.3f",
                       n, wall, mean(misfit), misfit[best], best, Δtₙ, T)
        @info "  ensemble mean parameters: " * join([@sprintf("%s = %.3f", name, mean(ϕ[k, :])) for (k, name) in enumerate(parameter_names(space))], ", ")

        push!(history, (iteration = n, ϕ = copy(ϕ), G, misfit, wall, Δt = Δtₙ, pseudotime = T))
        checkpoint!(output, history, ekp, y, Γ, problem, space, σ, radiation)

        if !isnothing(terminate)
            @info "The scheduler has reached its termination time; the ensemble of iteration $n is final"
            terminated = true
            break
        end
    end
    terminated || @warn "Stopped after $max_iterations iterations at pseudo time $(sum(get_Δt(ekp))) without reaching the target"

    ϕ_final = get_ϕ_final(prior, ekp)
    return ekp, prior, ϕ_final, history
end
