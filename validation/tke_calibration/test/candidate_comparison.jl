using Test, JLD2, SHA
include(joinpath(@__DIR__, "..", "scripts", "validate_candidate_comparison.jl"))

@testset "Candidate comparison completion and provenance" begin
    mktempdir() do directory
        snapshot = joinpath(directory, "snapshot.jld2")
        output = joinpath(directory, "comparison.jld2")
        reference = joinpath(directory, "reference.jld2")
        selected_mean = (; parameters = [1.0, 2.0, 3.0], objective = 4.0, iteration = 2)
        jldsave(snapshot; selected_mean, protocol_version = 3)
        digest = open(io -> bytes2hex(sha256(io)), snapshot)
        members = [(site, month) for site in [3, 12, 21] for month in ("01", "04", "07", "10")]
        variables = (:θˡ, :qᵗ, :qˡ, :qʳ, :u, :v)
        means = NamedTuple{variables}(Tuple(fill(1.0, 2, 12, 2) for variable in variables))
        score = NamedTuple{(variables..., :wind)}(ntuple(i -> 1.0, 7))
        result = (; means, scores = fill(score, 2, 12), zf = [0.0, 50.0, 100.0])
        results = Dict(resolution => deepcopy(result) for resolution in ("50", "100", "hindcast"))
        validation_data = (; gcm_sha256 = "a"^64,
                             les = [(; site, month, sha256 = "b"^64) for (site, month) in members])
        comparison_manifest = (; complete = true, source_hashes = [(; label = "ri", sha256 = digest)],
                                 validation_data, source_code = (; revision = "fixture", diff_sha256 = "c"^64),
                                 architecture = "gpu")
        run = (; results, params = hcat(selected_mean.parameters, selected_mean.parameters),
                 labels = ["default (Nakanishi–Niino)", "ri"], members, comparison_manifest,
                 protocol_version = 3, Δt = 7.5, radiation_interval = 600.0, validation_sites = [3, 12, 21])
        check(; kw...) = validate_candidate_comparison(output, snapshot, "ri"; Δt = 7.5, kw...)
        jldsave(output; run...)
        @test check() == comparison_manifest
        jldsave(reference; merge(run, (; Δt = 3.75))...)
        @test check(; reference) == comparison_manifest
        # One fault at a time, including plausible complete files from the wrong experiment.
        faults = [
            ("Δt", (; Δt = 30.0)),
            ("radiation_interval", (; radiation_interval = 1800.0)),
            ("protocol_version", (; protocol_version = 2)),
            ("validation_sites", (; validation_sites = [3, 12])),
            ("labels", (; labels = ["default (Nakanishi–Niino)", "constant"])),
            ("params", (; params = ones(3, 2))),
            ("members", (; members = members[1:end-1])),
            ("incomplete", (; comparison_manifest = merge(comparison_manifest, (; complete = false)))),
            ("source_hashes", (; comparison_manifest = merge(comparison_manifest,
                (; source_hashes = [(; label = "ri", sha256 = "d"^64)])))),
            ("architecture", (; comparison_manifest = merge(comparison_manifest, (; architecture = "cpu")))),
            ("resolutions", (; results = Dict("50" => result))),
        ]
        for (message, replacement) in faults
            jldsave(output; merge(run, replacement)...)
            err = try check() catch e; e end
            @test err isa ErrorException
            @test occursin(message, sprint(showerror, err))
        end
        for replacement in ((; means = merge(means, (; θˡ = fill(NaN, 2, 12, 2)))),
                            (; means = merge(means, (; θˡ = ones(2, 11, 2)))),
                            (; scores = fill(merge(score, (; wind = NaN)), 2, 12)),
                            (; zf = [0.0, 100.0, 50.0]))
            broken_results = merge(results, Dict("50" => merge(result, replacement)))
            jldsave(output; merge(run, (; results = broken_results))...)
            @test_throws ErrorException check()
        end
        jldsave(output; run...)
        for manifest_change in ((; source_code = (; revision = "other", diff_sha256 = "c"^64)),
                                (; validation_data = merge(validation_data, (; gcm_sha256 = "e"^64))))
            changed = merge(comparison_manifest, manifest_change)
            jldsave(reference; merge(run, (; comparison_manifest = changed, Δt = 3.75))...)
            @test_throws ErrorException check(; reference)
        end
        jldsave(output; results) # Partial schema left by an interrupted writer.
        @test_throws Exception check()
        write(output, "truncated")
        @test_throws Exception check()
    end
end
