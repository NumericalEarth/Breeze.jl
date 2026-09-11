using Test
include(joinpath(@__DIR__, "..", "scripts", "summarize_candidate_refinement.jl"))

@testset "Frozen-candidate numerical sensitivity" begin
    variables = (:θˡ, :qᵗ, :qˡ, :u, :v)
    observation_data = (; G = hcat(zeros(20), ones(20)), y = fill(10.0, 20),
                          variance = fill(4.0, 20), variables, zf = [0.0, 100.0, 200.0])
    reference = Dict("labels" => ["default", "ri"], "members" => [(3, "01"), (12, "07")],
                     "results" => Dict(grid => (; observation_data = deepcopy(observation_data))
                                       for grid in ("50", "100", "hindcast")))
    run = deepcopy(reference)
    rows = candidate_refinement_rows(run, reference)
    @test length(rows) == 8
    @test all(row -> row.rms_noise == row.max_noise == row.relative_change == 0, rows)
    @test all(row -> row.aggregate_target_met, rows)
    @test only(filter(row -> row.label == "ri" && row.resolution == "pooled", rows)).objective == 10.125
    # One local change can exceed the pointwise scale while meeting the pooled accuracy targets.
    # Index 16 is case 2, variable 3, height 2 in the actual observation-vector ordering.
    run["results"]["100"].observation_data.G[16, 2] += 0.4
    rows = candidate_refinement_rows(run, reference)
    pooled = only(filter(row -> row.label == "ri" && row.resolution == "pooled", rows))
    @test pooled.rms_noise ≈ 0.2 / sqrt(60)
    @test pooled.max_noise ≈ 0.2
    @test pooled.objective ≈ (59 * 9^2 + 8.6^2) / (2 * 4 * 60)
    @test pooled.relative_change ≈ abs(pooled.objective - 10.125) / 10.125
    @test (pooled.worst_grid, pooled.site, pooled.month, pooled.variable, pooled.z_m) ==
          ("100", 12, "07", :qˡ, 150.0)
    @test pooled.aggregate_target_met
    @test only(filter(row -> row.label == "ri" && row.resolution == "50", rows)).rms_noise == 0
    changed = deepcopy(reference)
    for result in values(changed["results"])
        result.observation_data.G[:, 2] .+= 2
    end
    rows = candidate_refinement_rows(changed, reference)
    @test all(row -> !row.aggregate_target_met, filter(row -> row.label == "ri", rows))
    for field in (:y, :variance, :zf)
        changed = deepcopy(reference)
        getproperty(changed["results"]["50"].observation_data, field)[1] += 1
        @test_throws ErrorException candidate_refinement_rows(changed, reference)
    end
    for value in (NaN, Inf)
        changed = deepcopy(reference)
        changed["results"]["50"].observation_data.G[1, 1] = value
        @test_throws ErrorException candidate_refinement_rows(changed, reference)
    end
    zero_reference = deepcopy(reference)
    for result in values(zero_reference["results"])
        result.observation_data.G .= 0
        result.observation_data.y .= 0
    end
    @test all(row -> row.relative_change == 0, candidate_refinement_rows(zero_reference, zero_reference))
    changed = deepcopy(zero_reference)
    changed["results"]["50"].observation_data.G[1, 2] = 1
    rows = candidate_refinement_rows(changed, zero_reference)
    @test only(filter(row -> row.label == "ri" && row.resolution == "pooled", rows)).relative_change == Inf
    @test !only(filter(row -> row.label == "ri" && row.resolution == "pooled", rows)).aggregate_target_met
end
