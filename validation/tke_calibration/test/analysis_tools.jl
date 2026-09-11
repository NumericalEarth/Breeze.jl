# Saved-result analysis must preserve the evaluated candidate and refuse incompatible numerical runs.
# julia --project test/analysis_tools.jl
using BreezeCalibration, JLD2, Test
include(joinpath(@__DIR__, "..", "scripts", "evaluated_candidate.jl"))
include(joinpath(@__DIR__, "..", "scripts", "calibration_data_manifest.jl"))

@testset "Forcing-data identity" begin
    current = calibration_data_manifest([(22, "07")])
    @test length(current.gcm_sha256) == 64
    @test length(only(current.les).sha256) == 64
    saved = Dict("run_configuration" => (; data_manifest = current), "members" => [(22, "07")])
    @test isnothing(validate_data_manifest(saved))
    changed = merge(current, (; gcm_sha256 = repeat("0", 64)))
    saved["run_configuration"] = (; data_manifest = changed)
    @test_throws ErrorException validate_data_manifest(saved)
    mktemp() do path, io
        write(io, "first"); flush(io)
        before = file_sha256(path)
        write(io, "second"); flush(io)
        @test file_sha256(path) != before
    end
end

@testset "Best evaluated point is distinct from mean-based stopping" begin
    selected = (; parameters = ones(7), G = [1.0, 2.0], objective = 0.5, iteration = 2)
    history = [(; iteration = 1, ϕ = hcat(fill(2.0, 7), fill(3.0, 7)),
                 G = [0.5 2.0; 1.0 4.0], misfit = [0.5, 2.0]),
               (; iteration = 2, ϕ = hcat(fill(4.0, 7), fill(5.0, 7)),
                 G = [0.8 NaN; 1.6 NaN], misfit = [0.8, NaN])]
    saved = Dict("selected_mean" => selected, "history" => history, "y" => zeros(2), "Γ" => [1.0, 4.0])
    @test evaluated_candidate(saved).kind == :ensemble_mean
    best = evaluated_candidate(saved; selection = :best_evaluated)
    @test best.kind == :ensemble_member
    @test (best.iteration, best.member) == (1, 1)
    @test best.objective == 0.125
    @test best.parameters == fill(2.0, 7)
    @test saved["selected_mean"] == selected
    @test_throws ErrorException evaluated_candidate(saved; selection = :unknown)
    bad = merge(saved, Dict("selected_mean" => merge(selected, (; objective = 9.0))))
    @test_throws ErrorException evaluated_candidate(bad)
end

function run_analysis_script(name, arguments)
    previous = copy(ARGS)
    try
        empty!(ARGS)
        append!(ARGS, arguments)
        redirect_stdout(devnull) do
            Base.include(Module(gensym(:AnalysisScript)), joinpath(@__DIR__, "..", "scripts", name))
        end
    finally
        empty!(ARGS)
        append!(ARGS, previous)
    end
end

@testset "Saved candidate export" begin
    mktempdir() do directory
        selected_path = joinpath(directory, "selected.jld2")
        polished_path = joinpath(directory, "polished.jld2")
        legacy_path = joinpath(directory, "legacy.jld2")
        output = joinpath(directory, "parameters.csv")
        parameters = collect(Float64, default_parameters(ConstantSpace()))
        selected = (; parameters = 2 .* parameters)
        jldsave(selected_path; protocol_version = PROTOCOL_VERSION, selected_mean = selected,
                  history = [(; ϕ = hcat(parameters, parameters))])
        jldsave(polished_path; protocol_version = PROTOCOL_VERSION, parameters = 3 .* parameters,
                  objective = 1.0, converged = false)
        jldsave(legacy_path; protocol_version = PROTOCOL_VERSION,
                  history = [(; ϕ = hcat(parameters, 3 .* parameters))])
        run_analysis_script("export_parameter_sets.jl",
                            ["output=$output", "selected=$selected_path", "polished=$polished_path", "legacy=$legacy_path"])
        lines = readlines(output)
        @test length(lines) == 1 + 4length(parameters)
        @test occursin("best evaluated ensemble mean", only(filter(l -> startswith(l, "\"selected\"") && occursin("\"Cᵘ\"", l), lines)))
        selected_line = only(filter(l -> startswith(l, "\"selected\"") && occursin("\"Cᵘ\"", l), lines))
        @test occursin("\"$(2parameters[1])\"", selected_line)
        @test occursin("local refinement incomplete", only(filter(l -> startswith(l, "\"polished\"") && occursin("\"Cᵘ\"", l), lines)))
        @test occursin("exploratory final ensemble mean", only(filter(l -> startswith(l, "\"legacy\"") && occursin("\"Cᵘ\"", l), lines)))
    end
end

@testset "Numerical partial compatibility" begin
    mktempdir() do directory
        first_path, second_path, bad_path, output = joinpath.(directory, ("first.jld2", "second.jld2", "bad.jld2", "combined.jld2"))
        common = (; protocol_version = PROTOCOL_VERSION, label = "Δt", params = ones(7, 1),
                    members = [(22, "07")], zf = [0.0, 100.0], observation_zf = [0.0, 100.0],
                    radiation_interval = 600.0)
        first_means, second_means = (; θˡ = fill(300.0, 1, 1, 1)), (; θˡ = fill(301.0, 1, 1, 1))
        jldsave(first_path; common..., value = 7.5, means = first_means)
        jldsave(second_path; common..., value = 3.75, means = second_means)
        run_analysis_script("combine_discretization_runs.jl", [output, first_path, second_path])
        combined = load(output)
        @test combined["dt_means"][7.5] == first_means
        @test combined["dt_means"][3.75] == second_means
        @test combined["source_partials"] == [first_path, second_path]
        @test isempty(combined["interval_means"])
        @test_throws LoadError run_analysis_script("combine_discretization_runs.jl", [output, first_path, first_path])
        changed = merge(common, (; radiation_interval = 1800.0))
        jldsave(bad_path; changed..., value = 3.75, means = second_means)
        @test_throws LoadError run_analysis_script("combine_discretization_runs.jl", [output, first_path, bad_path])
        @test load(output)["dt_means"] == combined["dt_means"]
    end
end
