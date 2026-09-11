# Run without loading Breeze or writing its precompile cache.
using JLD2, Test
include(joinpath(@__DIR__, "..", "scripts", "export_closure_diagnostics.jl"))

@testset "Saved closure diagnostics preserve staggering and reject mixed experiments" begin
    mktempdir() do directory
        path = joinpath(directory, "first.jld2")
        second = joinpath(directory, "second.jld2")
        output = joinpath(directory, "export")
        variables = (:θˡ, :qᵗ, :qˡ, :u, :v)
        means = NamedTuple{variables}(Tuple(fill(Float64(k), 2, 1, 2) for k in 1:5))
        targets = [NamedTuple{variables}(Tuple(fill(Float64(k), 2) for k in 1:5))]
        scores = fill(NamedTuple{variables}(Tuple(Float64(k) for k in 1:5)), 2, 1)
        diagnostics = (; counts = [2], means = Dict(:e => fill(1.0, 2, 1, 2),
                            :dissipation => fill(0.1, 2, 1, 2), :K_c => fill(3.0, 2, 1, 3)))
        run = (; protocol_version = 3, members = [(22, "07")],
                 zf = [0.0, 100.0, 300.0], zc = [50.0, 200.0], les_zf = [0.0, 100.0, 300.0],
                 observation_zf = [0.0, 100.0, 200.0, 300.0], Δt = 7.5, radiation_interval = 600.0,
                 stop_time = nothing, averaging_window = nothing,
                 source_run_configuration = (; upper_relaxation_rate = 1 / 600),
                 targets, means, diagnostics, scores, cloud_water_path = fill(0.2, 2, 1),
                 les_cloud_water_path = [0.1], objectives = [5.0, 2.0], resolution = "50",
                 labels = ["Default", "best directly evaluated mean candidate"])
        jldsave(path; run...)
        jldsave(second; run...)
        @test export_closure_diagnostics(output, [("Constant", path), ("Ri", second)]) == output
        rows = readlines(joinpath(output, "profiles.csv"))
        @test length(rows) == 1 + 3 * (5 * 2 + 2 + 2 + 3) + 5 * 2
        @test count(row -> occursin("\"LES\"", row), rows) == 10
        @test count(row -> occursin("\"K_c\",\"300.0\"", row), rows) == 3
        @test count(row -> occursin("\"e\",\"200.0\"", row), rows) == 3
        @test length(readlines(joinpath(output, "scores.csv"))) == 1 + 3 * 6 + 1
        @test occursin("not a closed TKE budget", read(joinpath(output, "context.txt"), String))
        @test_throws ErrorException export_closure_diagnostics(output, [("Default", path)])
        @test_throws ErrorException export_closure_diagnostics(output, [("Ri", path), ("Ri", second)])
        for key in (:Δt, :radiation_interval, :protocol_version)
            changed = merge(run, NamedTuple{(key,)}((getproperty(run, key) + 1,)))
            jldsave(second; changed...)
            err = try
                export_closure_diagnostics(output, [("Constant", path), ("Ri", second)])
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin(string(key), sprint(showerror, err))
        end
        changed_means = deepcopy(means)
        changed_means.qˡ[1, 1, 1] += 1
        jldsave(second; merge(run, (; means = changed_means))...)
        @test_throws ErrorException export_closure_diagnostics(output, [("Constant", path), ("Ri", second)])
        no_samples = merge(diagnostics, (; counts = [0]))
        jldsave(second; merge(run, (; diagnostics = no_samples))...)
        @test_throws ErrorException export_closure_diagnostics(output, [("Ri", second)])
        # Rejections happen before writing the exports; a bad comparison must not replace good output.
        @test readlines(joinpath(output, "profiles.csv")) == rows
    end
end
