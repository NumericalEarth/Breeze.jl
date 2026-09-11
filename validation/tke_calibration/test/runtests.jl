# julia --project test/runtests.jl — a few minutes; downloads the LES artifact on first use
using Test
using BreezeCalibration
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using Random, Statistics, LinearAlgebra, JLD2, NCDatasets
using Breeze.TurbulenceClosures: ConstantStabilityFunctions, RiDependentStabilityFunctions
using Oceananigans.Units
using Oceananigans: RectilinearGrid, Flat, Bounded

@testset "the ensemble mean is an independent forward evaluation" begin
    parameters = reshape([1.0, 3.0], 1, :)
    augmented = hcat(parameters, mean(parameters; dims = 2))
    # A nonlinear forward map makes the two notions of mean distinguishable.
    G, diagnostic = BreezeCalibration.ensemble_mean_diagnostics(parameters, augmented .^ 2,
                                                                [0.0], Diagonal([1.0]))
    @test size(G) == (1, 2) && G == [1.0 9.0]
    @test diagnostic.mean_G == [4.0] && diagnostic.mean_parameters == [2.0]
    @test diagnostic.mean_objective == 8.0
    @test diagnostic.mean_G != vec(mean(G; dims = 2))
    history(values, T = 2.0) = [(; mean_objective = value, evaluation_pseudotime = T) for value in values]
    @test BreezeCalibration.mean_objective_plateau(history([2.0, 1.5, 1.0, 1.001, 1.002, 1.001]))
    @test !BreezeCalibration.mean_objective_plateau(history([2.0, 1.5, 1.2, 1.1, 1.0, 0.9]))
    @test !BreezeCalibration.mean_objective_plateau(history(fill(1.0, 6), 0.5))
    @test !BreezeCalibration.mean_objective_plateau(history([1.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
end

@testset "optimizer configuration survives checkpoint serialization" begin
    scheduler = DataMisfitController()
    accelerator = NesterovAccelerator()
    algorithm = BreezeCalibration.algorithm_configuration(scheduler, accelerator, SECNice())
    protocol = (; protocol_version = PROTOCOL_VERSION)
    configuration = (; Δt = 15.0, radiation_interval = 600.0)
    path = joinpath(mktempdir(), "configuration.jld2")
    jldsave(path; protocol_version = PROTOCOL_VERSION, run_configuration = configuration,
                 algorithm_configuration = algorithm)
    saved = load(path)
    @test isnothing(BreezeCalibration.validate_checkpoint(saved, protocol, configuration; algorithm))
    # A snapshot must not acquire the mutable runtime state of its source objects.
    push!(scheduler.iteration, 1)
    accelerator.θ_prev = 0.5
    fresh = BreezeCalibration.algorithm_configuration(DataMisfitController(), NesterovAccelerator(), SECNice())
    @test algorithm == fresh
    for alternative in (BreezeCalibration.algorithm_configuration(DataMisfitController(), NesterovAccelerator(), NoLocalization()),
                        BreezeCalibration.algorithm_configuration(DataMisfitController(terminate_at = 2), NesterovAccelerator(), SECNice()),
                        BreezeCalibration.algorithm_configuration(DataMisfitController(), accelerator, SECNice()))
        @test_throws ErrorException BreezeCalibration.validate_checkpoint(saved, protocol, configuration; algorithm = alternative)
    end
    delete!(saved, "algorithm_configuration")
    @test_throws ErrorException BreezeCalibration.validate_checkpoint(saved, protocol, configuration; algorithm)
end

@testset "batched forward maps preserve independent optimizer trajectories" begin
    prior = prior_distribution(ConstantSpace())
    rng = MersenneTwister(18)
    A = randn(rng, 12, 7)
    y, Γ = ones(12), Diagonal(ones(12))
    initials = [construct_initial_ensemble(MersenneTwister(n), prior, n) for n in (20, 30)]
    function process(initial)
        BreezeCalibration.build_process(copy(initial), y, Γ; rng = MersenneTwister(4),
            scheduler = DataMisfitController(on_terminate = "continue"),
            accelerator = NesterovAccelerator(), localization_method = SECNice())
    end
    grouped, separate = process.(initials), process.(initials)
    grouped_history, separate_history = [[], []], [[], []]
    stopping = (; optimize = true, objective_tolerance = 0.005, objective_patience = 3,
                  minimum_optimization_iterations = 6, target_pseudotime = 1.0)
    for iteration in 1:3
        ensembles = [get_ϕ_final(prior, process) for process in grouped]
        parameters, ranges = BreezeCalibration.batched_forward_parameters(ensembles)
        @test ranges == [1:21, 22:52]
        all_G = A * parameters .^ 2
        for k in eachindex(grouped)
            BreezeCalibration.record_eki_iteration!(grouped_history[k], grouped[k], ensembles[k],
                all_G[:, ranges[k]], y, Γ, 0.0; stopping)
            ϕ = get_ϕ_final(prior, separate[k])
            G = A * hcat(ϕ, mean(ϕ; dims = 2)) .^ 2
            BreezeCalibration.record_eki_iteration!(separate_history[k], separate[k], ϕ, G, y, Γ, 0.0; stopping)
            @test get_u_final(grouped[k]) ≈ get_u_final(separate[k]) rtol = 1e-10
            @test grouped_history[k][end].mean_objective ≈ separate_history[k][end].mean_objective rtol = 1e-10
        end
    end
end

@testset "total-water relaxation acts on the vapor-cloud sum" begin
    grid = RectilinearGrid(size = 4, z = (3000, 4000), topology = (Flat, Flat, Bounded))
    density = fill(0.8, 1, 1, 4)
    fields = (ρqᵛ = density .* 0.006, ρqᶜˡ = density .* 0.002)
    p = (; density, rate = 1 / 86400, mask = Returns(1), target = fill(0.008, 1, 1, 4))
    @test BreezeCalibration.total_water_relaxation(1, 1, 2, grid, nothing, fields, p) ≈ 0 atol=1e-20
    wet = merge(p, (; target = fill(0.009, 1, 1, 4)))
    @test BreezeCalibration.total_water_relaxation(1, 1, 2, grid, nothing, fields, wet) ≈ 0.001 / 86400
end

# A linear toy problem in the closure's 17 parameters: G(θ) = A θ, observed with noise 0.1
function toy_problem(rng; N_obs = 40)
    prior = prior_distribution()
    N = length(parameter_names())
    A = randn(rng, N_obs, N)
    θ★ = collect(Float64, default_parameters()) .* exp.(0.2 .* randn(rng, N))
    y = A * θ★
    Γ = Diagonal(fill(0.1^2, N_obs))
    return prior, A, y, Γ
end

@testset "replay reproduces the process" begin
    # Schedulers and accelerators are stateful, so each process gets freshly constructed ones
    options() = (scheduler = DataMisfitController(terminate_at = 1), accelerator = NesterovAccelerator(), localization_method = SECNice())
    rng = MersenneTwister(3)
    prior, A, y, Γ = toy_problem(rng)
    ekp = EnsembleKalmanProcess(construct_initial_ensemble(rng, prior, 100), y, Γ, Inversion(); rng, options()...)
    history = []
    for it in 1:4
        ϕ = get_ϕ_final(prior, ekp)
        G = A * ϕ
        misfit = [sqrt(mean(((G[:, i] .- y) ./ sqrt.(diag(Γ))) .^ 2)) for i in axes(G, 2)]
        push!(history, (iteration = it, ϕ = copy(ϕ), G, misfit, wall = 0.0))
        update_ensemble!(ekp, G)
    end

    ekp′, replayed = replay(prior, history, y, Γ; rng = MersenneTwister(3), options()...)
    @test get_u_final(ekp′) ≈ get_u_final(ekp) rtol = 1e-8
    @test get_Δt(ekp′) ≈ get_Δt(ekp)
    @test length(replayed) == 4
    @test replayed[end].pseudotime ≈ sum(get_Δt(ekp))
    @test all(h -> isfinite(h.Δt), replayed)
end

@testset "grids and regridding" begin
    faces = hindcast_faces()
    @test length(faces) == 24
    @test faces[1] == 0 && faces[2] == 50
    @test isapprox(faces[end], 3977.31, atol = 0.01)
    @test all(diff(diff(faces)) .> 0)          # spacing grows monotonically

    les = les_faces(collect(10.0:20.0:3990.0))
    @test les == collect(0.0:20.0:4000.0)
    @test uniform_faces(100, 4000) == collect(0.0:100.0:4000.0)
    @test_throws ErrorException uniform_faces(300, 4000)

    # A linear profile regrids exactly onto grids whose faces align with the LES faces (cell means of a linear
    # function are its center values); on the stretched grid the piecewise-constant source leaves an error
    # bounded by the slope times half an LES cell
    zc = collect(10.0:20.0:3990.0)
    linear = 300 .+ 0.004 .* zc
    to = uniform_faces(100, 4000)
    @test regrid_column(linear, les, to) ≈ 300 .+ 0.004 .* (to[1:end-1] .+ to[2:end]) ./ 2
    @test isapprox(regrid_column(linear, les, faces), 300 .+ 0.004 .* (faces[1:end-1] .+ faces[2:end]) ./ 2, atol = 0.004 * 10)

    # Conservation: the integral over a common depth is preserved, and coarse → fine is piecewise constant
    profile = @. 300 + 3 * exp(-((zc - 1000) / 300)^2) + 0.01 * zc
    coarse = regrid_column(profile, les, uniform_faces(100, 4000))
    @test sum(coarse) * 100 ≈ sum(profile) * 20
    back = regrid_column(coarse, uniform_faces(100, 4000), les)
    @test all(back[1:5] .≈ coarse[1])

    # The 3D version agrees with the column version
    A = reshape(profile, 1, 1, :) .* ones(3, 2, 1)
    @test regrid_columns(A, les, faces)[2, 1, :] ≈ regrid_column(profile, les, faces)

    @test_throws ErrorException regrid_column(profile, les, collect(0.0:500.0:4500.0))
end

@testset "parameter spaces" begin
    @test parameter_names() == parameter_names(RiDependentSpace())
    @test length(parameter_names(ConstantSpace())) == 7
    @test space_of(17) isa RiDependentSpace && space_of(7) isa ConstantSpace
    @test_throws ErrorException space_of(5)

    for space in (RiDependentSpace(), ConstantSpace())
        p = default_parameters(space)
        @test p.Cᵂu★ == 0 && prior_center(space).Cᵂu★ == 1
        closure = closure_from(space, collect(p))
        @test closure.mixing_length.Cˢ == 1.316
        @test closure_from(p).stability_functions == closure.stability_functions
    end
    @test closure_from(ConstantSpace(), collect(default_parameters(ConstantSpace()))).stability_functions isa ConstantStabilityFunctions
    @test closure_from(default_parameters()).stability_functions isa RiDependentStabilityFunctions
    @test BreezeCalibration.parameter_index(ConstantSpace(), :Cᵂʷ) == 7
    @test BreezeCalibration.parameter_index(RiDependentSpace(), :Cᵂu★) == 16
end

@testset "multi-resolution observations" begin
    members = [load_member(22, "07"), load_member(17, "07")]
    # Columns ending at the LES top
    fine = ColumnEnsembleProblem(members; variables = (:θˡ, :qᵗ), top = nothing)
    coarse = ColumnEnsembleProblem(members; Δz = 100, variables = (:θˡ, :qᵗ), top = nothing)
    hindcast = ColumnEnsembleProblem(members; z_faces = hindcast_faces(), variables = (:θˡ, :qᵗ), top = nothing)
    @test length(fine.zf) == 201 && length(coarse.zf) == 41 && length(hindcast.zf) == 24
    # The default column continues to 25 km on stretched faces; the observations do not depend on it
    tall = ColumnEnsembleProblem(members; Δz = 100, variables = (:θˡ, :qᵗ))
    @test tall.zf[end] == 25_000 && tall.les_top == 4000 && tall.zf[1:41] == coarse.zf
    @test all(diff(diff(tall.zf[41:end-1])) .> 0)          # spacing grows geometrically; the last cell is clipped at the top
    @test observations(tall)[1] == observations(coarse)[1]
    y₁, Γ₁ = observations(fine)
    y₂, _ = observations(coarse)
    @test y₁ ≈ y₂                                 # targets live on the observation cells, not the model grid
    @test length(y₁) == 2 * 30 * 2
    multi = MultiResolutionProblem([fine, coarse, hindcast]; weights = [1, 1, 2])
    y, Γ = observations(multi)
    @test y == vcat(y₁, y₁, y₁)
    @test diag(Γ)[end] ≈ diag(Γ₁)[end] / 4
    @test BreezeCalibration.members(multi) === members
    full = ColumnEnsembleProblem(members; Δz = 100)          # the default five variables (rain is diagnosed, not scored)
    yf, Γf = observations(full)
    @test length(yf) == 5 * 30 * 2 && yf[1:60] == y₂[1:60]
    @test diag(Γf)[61] == 0.1^2 && diag(Γf)[end] == 0.5^2
    rainy = ColumnEnsembleProblem(members; Δz = 100, variables = observable_variables)
    @test length(observations(rainy)[1]) == 6 * 30 * 2
    @test_throws ErrorException ColumnEnsembleProblem(members; variables = (:θˡ, :T))
end

@testset "relaxation targets are the GCM profiles" begin
    # The LES relaxed its winds toward the GCM winds on 6 h (Shen et al. 2022, §2.2.3); the file's
    # `u_mean_nudge` is the LES run mean, and adding 6 h × the run-mean nudging tendency recovers the GCM
    # (initial) profile. The column must relax toward the latter.
    ds = NCDataset(member_path(22, "07"))
    z = ds["z"][:]; kb = z .< 2500
    u₀ = ds["u_mean_initial"][:]; uₙ = ds["u_mean_nudge"][:]; du = ds["dudt_nudge"][:]
    close(ds)
    @test sqrt(mean((u₀[kb] .- (uₙ[kb] .+ 21600 .* du[kb])) .^ 2)) < 0.05
    @test sqrt(mean((u₀[kb] .- uₙ[kb]) .^ 2)) > 0.3
    m = load_member(22, "07")
    @test m.nudging.u == m.initial.u
    @test !isnothing(m.gcm) && m.gcm.z[end] > 25_000 && isfinite(m.gcm.coszen)
end

@testset "EKI end to end: tall column, interactive radiation, checkpoint and resume" begin
    members = [load_member(22, "07")]
    problem = MultiResolutionProblem([ColumnEnsembleProblem(members; Δz = 100), ColumnEnsembleProblem(members; z_faces = hindcast_faces())])
    output = joinpath(mktempdir(), "eki.jld2")
    # Include the 10-minute callback, so this exercises evolved profiles as well as t = 0.
    short_run = (; stop_time = 12minutes, averaging_window = (0.0, 12minutes))
    ekp, prior, ϕ, history = run_eki(problem; space = ConstantSpace(), N_ens = 3, max_iterations = 1, output, short_run...)
    @test length(history) == 1 && size(ϕ) == (7, 3) && all(isfinite, history[1].G)
    saved = load(output)
    @test saved["radiation"] == "interactive" && saved["top"] == 25_000 && length(saved["z_faces"]) == 2
    @test saved["parameter_names"] == collect(String.(parameter_names(ConstantSpace())))
    @test maximum(history[1].G) > 250 # real potential temperatures, not empty-window zero means
    @test maximum(abs, history[1].G[:, 1] - history[1].G[:, 2]) > 1e-8
    @test size(history[1].G, 2) == 3 && length(history[1].mean_G) == size(history[1].G, 1)
    @test history[1].mean_parameters == vec(mean(history[1].ϕ; dims = 2))
    @test history[1].mean_misfit > 0 && isfinite(history[1].mean_objective)
    @test saved["selected_mean"].objective == history[1].mean_objective
    @test saved["run_configuration"].averaging_window == short_run.averaging_window
    protocol = BreezeCalibration.checkpoint_protocol(problem, ConstantSpace(), saved["y"], Diagonal(saved["Γ"]), :interactive)
    old = merge(saved, Dict("protocol_version" => PROTOCOL_VERSION - 1))
    @test_throws ErrorException BreezeCalibration.validate_checkpoint(old, protocol, saved["run_configuration"])
    # Incompatible forward maps must be rejected before replay, even when their dimensions agree.
    @test_throws ErrorException run_eki(problem; space = ConstantSpace(), resume = output, radiation = :prescribed, short_run...)
    @test_throws ErrorException run_eki(problem; space = ConstantSpace(), resume = output, Δt = 30, short_run...)
    @test_throws ErrorException run_eki(problem; space = ConstantSpace(), resume = output, radiation_interval = 300, short_run...)
    @test_throws ErrorException run_eki(problem; space = ConstantSpace(), resume = output, localization_method = NoLocalization(), short_run...)
    reversed_problem = MultiResolutionProblem(reverse(problem.problems))
    @test_throws ErrorException run_eki(reversed_problem; space = ConstantSpace(), resume = output, short_run...)
    # Resuming replays the saved forward map exactly and runs one more iteration
    _, _, _, history₂ = run_eki(problem; space = ConstantSpace(), N_ens = 3, max_iterations = 1, output, short_run..., resume = output)
    @test length(history₂) == 2 && history₂[1].ϕ == history[1].ϕ
end
