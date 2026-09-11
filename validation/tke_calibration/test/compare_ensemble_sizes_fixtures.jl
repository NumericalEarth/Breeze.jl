# Fixtures for `scripts/compare_ensemble_sizes.jl`.
#
# The script's job is to refuse bad comparisons, so the cases that matter are the ones it must
# reject: runs that differ in anything but ensemble size and seed, and runs whose seeds are absent
# or duplicated. Synthetic checkpoints with the shape of real ones are enough, and they let the
# incompatibilities be introduced one at a time.
#
# No BreezeCalibration dependency, by design — the script has none either, so this runs in seconds
# and never touches the package precompile cache.
#
#     julia --compiled-modules=existing --project test/compare_ensemble_sizes_fixtures.jl
using Test, JLD2, Random, Statistics

const script = joinpath(@__DIR__, "..", "scripts", "compare_ensemble_sizes.jl")
const N_obs, N_par = 24, 4

"""A checkpoint with the fields the script reads. `overrides` replaces any top-level entry."""
function fixture(path; N_ens = 8, cases = 3, seed = 1, iterations = 4, protocol = 3,
                 Δt = 7.5, radiation_interval = 1800.0, radiation = "interactive",
                 y = collect(range(280, 300, length = N_obs)), Γ = fill(0.25^2, N_obs),
                 members = [(2, "01"), (5, "01"), (8, "01")][1:cases],
                 z_faces = [collect(0.0:50.0:200.0)], selected_iteration = 3,
                 include_selected = true, overrides = Dict{String, Any}())
    rng = MersenneTwister(seed)
    history = map(1:iterations) do n
        ϕ = 0.5 .+ 0.1 .* rand(rng, N_par, N_ens)
        G = y .+ 0.3 .* randn(rng, N_obs, N_ens)
        misfit = [sqrt(mean(((G[:, i] .- y) ./ sqrt.(Γ)) .^ 2)) for i in 1:N_ens]
        mean_G = y .+ 0.2 .* randn(rng, N_obs)
        (; iteration = n, ϕ, G, misfit, wall = 100.0 + n,
           mean_parameters = vec(mean(ϕ; dims = 2)), mean_G,
           mean_objective = mean(abs2, (mean_G .- y) ./ sqrt.(Γ)) / 2,
           evaluation_pseudotime = 0.25n, pseudotime = 0.25n,
           applied_update = true, stop_reason = n == iterations ? :mean_objective_plateau : :none,
           optimizer_wall = 1.0)
    end
    selected = include_selected ? (; parameters = history[selected_iteration].mean_parameters,
                                    G = history[selected_iteration].mean_G,
                                    objective = history[selected_iteration].mean_objective,
                                    iteration = selected_iteration) : nothing
    data = Dict{String, Any}("history" => history, "y" => y, "Γ" => Γ, "protocol_version" => protocol,
        "parameter_names" => ["Cᵘ", "Cᶜ", "Cᵉ", "Cᴰ"], "members" => members, "z_faces" => z_faces,
        "radiation" => radiation, "selected_mean" => selected,
        "algorithm_configuration" => (; scheduler = "DataMisfitController", accelerator = "Nesterov"),
        "experiment_metadata" => (; seed),
        "run_configuration" => (; Δt, radiation_interval, spread = 0.5, N_ens, seed))
    merge!(data, overrides)
    jldsave(path; (Symbol(k) => v for (k, v) in data)...)
    return path
end

"""Run the script on `paths`; return (ok, combined output)."""
function run_script(paths...)
    cmd = `$(Base.julia_cmd()) --compiled-modules=existing --startup-file=no --project=$(joinpath(@__DIR__, "..")) $script $(collect(paths))`
    out = IOBuffer()
    ok = success(pipeline(cmd; stdout = out, stderr = out))
    return ok, String(take!(out))
end

dir = mktempdir()

@testset "accepts a ladder that differs only in ensemble size and seed" begin
    a = fixture(joinpath(dir, "a.jld2"); N_ens = 8, seed = 1)
    b = fixture(joinpath(dir, "b.jld2"); N_ens = 16, seed = 2)
    ok, out = run_script("small=$a", "large=$b")
    @test ok
    @test occursin("selected mean", out)
    @test occursin("mean_objective_plateau", out)
    @test !occursin("posterior uncertainty", out)   # the word only survives in the rule forbidding it
end

@testset "rejects each incompatibility, naming it" begin
    base = fixture(joinpath(dir, "base.jld2"))
    cases = [
        ("time step",        Dict("run_configuration" => (; Δt = 15.0, radiation_interval = 1800.0, spread = 0.5, N_ens = 8, seed = 2)), "Δt"),
        ("radiation interval", Dict("run_configuration" => (; Δt = 7.5, radiation_interval = 600.0, spread = 0.5, N_ens = 8, seed = 2)), "radiation_interval"),
        ("protocol version", Dict("protocol_version" => 2), "protocol version"),
        ("members",          Dict("members" => [(2, "01"), (5, "01")]), "members"),
        ("grid faces",       Dict("z_faces" => [collect(0.0:100.0:200.0)]), "grid faces"),
        ("radiation mode",   Dict("radiation" => "prescribed"), "radiation mode"),
        ("observation noise", Dict("Γ" => fill(0.5^2, N_obs)), "Γ"),
        ("algorithm",        Dict("algorithm_configuration" => (; scheduler = "Constant", accelerator = "Nesterov")), "algorithm"),
    ]
    for (name, overrides, expected) in cases
        other = fixture(joinpath(dir, "bad_$(replace(name, ' ' => '_')).jld2"); seed = 2, overrides)
        ok, out = run_script(base, other)
        @test !ok                              # a difference here must be fatal, not a warning
        @test occursin(expected, out)          # and the message must name the offending field
    end
end

@testset "a single checkpoint needs no compatibility check" begin
    ok, out = run_script(fixture(joinpath(dir, "solo.jld2")))
    @test ok
    @test occursin("no configuration has two runs with distinct recorded seeds", out)
end

@testset "seeds must be distinct and recorded before scatter is claimed" begin
    # Same configuration, same seed: one experiment repeated, not two samples
    d1 = fixture(joinpath(dir, "dup1.jld2"); seed = 5)
    d2 = fixture(joinpath(dir, "dup2.jld2"); seed = 5)
    ok, out = run_script(d1, d2)
    @test ok
    @test occursin("no configuration has two runs with distinct recorded seeds", out)

    # Seed absent from both places: must be excluded, not defaulted
    bare = Dict("experiment_metadata" => (;), "run_configuration" => (; Δt = 7.5, radiation_interval = 1800.0, spread = 0.5, N_ens = 8))
    u1 = fixture(joinpath(dir, "unseeded1.jld2"); overrides = bare)
    u2 = fixture(joinpath(dir, "unseeded2.jld2"); overrides = bare)
    ok, out = run_script(u1, u2)
    @test ok
    @test occursin("No seed recorded", out)

    # Distinct seeds at one configuration: scatter is reported
    s1 = fixture(joinpath(dir, "seed1.jld2"); seed = 1)
    s2 = fixture(joinpath(dir, "seed2.jld2"); seed = 2)
    ok, out = run_script(s1, s2)
    @test ok
    @test occursin("across-seed scatter", out)
end

@testset "the seed is read from experiment_metadata, as the batched driver writes it" begin
    # Batched runs record the seed only in experiment_metadata; the run configuration has none
    batched(path, seed) = fixture(path; seed, overrides = Dict(
        "experiment_metadata" => (; seed, provenance = "batched"),
        "run_configuration" => (; Δt = 7.5, radiation_interval = 1800.0, spread = 0.5, N_ens = 8)))
    ok, out = run_script(batched(joinpath(dir, "batch1.jld2"), 11), batched(joinpath(dir, "batch2.jld2"), 12))
    @test ok
    @test occursin("seeds 11, 12", out)
    @test !occursin("No seed recorded", out)
end

@testset "a mixed group of duplicate and distinct seeds keeps one run per seed" begin
    # Seeds [1, 1, 2] pass a bare "more than one distinct seed" test while double-weighting seed 1 in
    # the scatter. The duplicate must be dropped before the pairing test, not after.
    m1 = fixture(joinpath(dir, "mix1.jld2"); seed = 1)
    m2 = fixture(joinpath(dir, "mix2.jld2"); seed = 1)
    m3 = fixture(joinpath(dir, "mix3.jld2"); seed = 2)
    ok, out = run_script(m1, m2, m3)
    @test ok
    @test occursin("Duplicate seeds", out)
    @test occursin("seeds 1, 2", out)            # the surviving pair, not 1, 1, 2
end

@testset "metadata that is absent in every checkpoint is not agreement" begin
    # Two checkpoints both lacking a field compare equal as `missing`, which would let the gate pass
    # on runs whose provenance is simply unrecorded.
    for (name, key) in (("protocol version", "protocol_version"), ("radiation mode", "radiation"),
                        ("algorithm configuration", "algorithm_configuration"))
        strip_key = Dict(key => missing)
        b1 = fixture(joinpath(dir, "nometa1_$key.jld2"); seed = 1, overrides = strip_key)
        b2 = fixture(joinpath(dir, "nometa2_$key.jld2"); seed = 2, overrides = strip_key)
        ok, out = run_script(b1, b2)
        @test !ok
        @test occursin("is not recorded in", out)
    end
end

@testset "a checkpoint without a directly evaluated mean still reports" begin
    n1 = fixture(joinpath(dir, "nomean1.jld2"); include_selected = false, seed = 1)
    n2 = fixture(joinpath(dir, "nomean2.jld2"); include_selected = false, seed = 2)
    ok, out = run_script(n1, n2)
    @test ok
    @test occursin("—", out)   # the selected-mean column is blank rather than fabricated
end
