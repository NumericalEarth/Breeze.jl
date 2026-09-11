# GPU integration regression: julia -t auto --project test/gpu_pipeline.jl
# Requires a CUDA device. Covers exact three-seed column independence, diagnostics,
# chunked calibration, independent checkpoint/resume and direct local candidate checks.
using BreezeCalibration, CUDA, Test, Statistics, Random
using EnsembleKalmanProcesses: construct_initial_ensemble, transform_unconstrained_to_constrained
using Oceananigans: GPU
include(joinpath(@__DIR__, "..", "scripts", "closure_diagnostics.jl"))
CUDA.functional() || error("GPU required")
CUDA.allowscalar(false)
println("GPU batching/diagnostics test: ", CUDA.name(CUDA.device())); flush(stdout)
members = [load_member(22, "07"), load_member(17, "07")]
problem = ColumnEnsembleProblem(members; Δz = 100)
space = ConstantSpace()
prior = prior_distribution(space)
ensembles = [transform_unconstrained_to_constrained(prior, construct_initial_ensemble(MersenneTwister(seed), prior, 12)) for seed in 1:3]
parameters, ranges = BreezeCalibration.batched_forward_parameters(ensembles)
kw = (; space, architecture = GPU(), Δt = 60.0, stop_time = 720.0, averaging_window = (0.0, 720.0))
recorder = closure_diagnostic_recorder(length(members))
G, _ = forward_map(problem, parameters; kw..., sample_callback = recorder.sample)
diagnostics = recorder.result()
@testset "GPU grouped columns and diagnostic sampling" begin
    @test all(==(2), diagnostics.counts)
    @test all(values -> all(isfinite, values), values(diagnostics.means))
    @test maximum(diagnostics.means[:e]) > 0
    @test maximum(diagnostics.means[:K_u]) > 0
    @test maximum(diagnostics.means[:shear_production]) > 0
    @test maximum(diagnostics.means[:dissipation]) > 0
    @test maximum(abs, diagnostics.means[:total_water_flux]) > 0
    for (ensemble, columns) in zip(ensembles, ranges)
        separate, _ = forward_map(problem, hcat(ensemble, mean(ensemble; dims = 2)); kw...)
        println("Grouped/separate exact equality: ", G[:, columns] == separate, "; max abs error: ", maximum(abs, G[:, columns] - separate))
        @test G[:, columns] == separate
    end
end

module DriverSmoke
    empty!(ARGS)
    const smoke_directory = mktempdir(; prefix = "breeze_batch_pipeline_")
    println("Persistent pipeline test output: ", smoke_directory); flush(stdout)
    append!(ARGS, ["space=constant", "runs=12:1,16:2", "sites=22", "months=07",
                   "resolutions=100", "dt=60", "radiation_interval=600", "stop_time=720",
                   "max_iterations=1", "max_columns=20", "output=$smoke_directory"])
    include(joinpath(@__DIR__, "..", "scripts", "batched_calibrate.jl"))
end
@testset "batched driver checkpoints" begin
    for label in ("n12_seed1", "n16_seed2")
        saved = DriverSmoke.load(joinpath(DriverSmoke.smoke_directory, "$label.jld2"))
        @test length(saved["history"]) == 1
        @test saved["selected_mean"].objective > 0
        @test saved["experiment_metadata"].batched
        @test saved["run_configuration"].Δt == 60
    end
end

# Resume after the initial chunked batch, retaining each independent optimizer state.
empty!(DriverSmoke.ARGS)
append!(DriverSmoke.ARGS, ["space=constant", "runs=12:1,16:2", "sites=22", "months=07",
                         "resolutions=100", "dt=60", "radiation_interval=600", "stop_time=720",
                         "max_iterations=2", "max_columns=20", "output=$(DriverSmoke.smoke_directory)"])
DriverSmoke.main()
@testset "batched driver resumes both optimizers" begin
    for label in ("n12_seed1", "n16_seed2")
        saved = DriverSmoke.load(joinpath(DriverSmoke.smoke_directory, "$label.jld2"))
        @test length(saved["history"]) == 2
        @test length(saved["experiment_metadata"].provenance) == 1
        @test all(h -> h.applied_update, saved["history"])
    end
end
module PolishSmoke
    empty!(ARGS)
    append!(ARGS, ["checkpoint=$(Main.DriverSmoke.smoke_directory)/n12_seed1.jld2",
                   "output=$(Main.DriverSmoke.smoke_directory)/polish.jld2",
                   "max_iterations=1", "random_directions=0"])
    include(joinpath(@__DIR__, "..", "scripts", "polish_candidate.jl"))
end
@testset "local refinement reproduces the saved candidate" begin
    source = DriverSmoke.load(joinpath(DriverSmoke.smoke_directory, "n12_seed1.jld2"))
    polished = DriverSmoke.load(joinpath(DriverSmoke.smoke_directory, "polish.jld2"))
    @test polished["history"][1].objectives[1] ≈ source["selected_mean"].objective rtol=1e-8
    @test polished["objective"] <= source["selected_mean"].objective
    @test size(polished["history"][1].candidates, 2) == 15
end
