# Fixtures for scripts/grid_screening_diagnostic.jl.
#
# Every case runs the real script as a subprocess against a synthetic checkpoint. Asserting on a
# restatement of the script's arithmetic inside this file would pass while the script itself was
# broken — that has happened here before.
#
# The checkpoints are built so the expected numbers can be derived by hand: one observation cell, one
# variable, one member, sigma = 1 and y = 0, so a member's objective on a grid is exactly G^2/2 for
# that grid's single entry.
#
#     julia --project test/grid_screening_fixtures.jl
using JLD2, Printf, Test

const script = joinpath(@__DIR__, "..", "scripts", "grid_screening_diagnostic.jl")
const project = joinpath(@__DIR__, "..")

"""A checkpoint with `size(G, 1)` observations: `n_grids` grids of one cell, one variable, one member."""
function write_checkpoint(path, G; n_grids = 2, cells = [3, 5])
    length(cells) == n_grids || error("fixture needs one cell count per grid")
    size(G, 1) == n_grids || error("fixture uses one observation per grid")
    history = [(; iteration = 1, G = G, misfit = vec(sqrt.(2 .* sum(abs2, G, dims = 1) ./ n_grids)))]
    jldsave(path; z_faces = [collect(0.0:1.0:c) for c in cells],
                  members = [(1, "01")], variables = ["thetal"],
                  observation_faces = [0.0, 100.0],
                  y = zeros(n_grids), Γ = ones(n_grids), history)
    return path
end

run_script(path) = (io = IOBuffer();
                    ok = success(pipeline(`julia --startup-file=no --project=$project $script $path`;
                                          stdout = io, stderr = io));
                    (ok, String(take!(io))))

"""
The row for `subset` in the per-iteration table. The cost-estimate table above it has rows with the
same leading label, so matching on the label alone would pick up both — which is how the first version
of this fixture failed.
"""
function iteration_row(out, subset)
    sections = split(out, "=== iteration")
    length(sections) > 1 || error("no per-iteration table in:\n$out")
    return only(filter(l -> startswith(l, "  " * subset * " "), split(last(sections), '\n')))
end

"""
The numeric columns of that row — Phi, Spearman, regret, relative regret, top-decile overlap — with the
label removed. Matching a number anywhere in the row is not enough: 1.0000 in the Phi column would
satisfy an assertion meant for the correlation.
"""
iteration_fields(out, subset) = split(replace(iteration_row(out, subset), subset => ""))

mktempdir() do dir
    @testset "ties use averaged ranks" begin
        # Grid 1 objectives [1, 2, 2, 3]: two members tie. Full objectives (2*grid1 + grid2^2)/4 are
        # [0.5, 1.0, 1.25, 1.5], strictly increasing, so their ranks are 1, 2, 3, 4.
        #
        # Spearman on tied (mid) ranks [1, 2.5, 2.5, 4] against [1, 2, 3, 4] is
        #     cov 4.5, var 4.5 and 5  ->  4.5 / sqrt(22.5) = 0.9487
        # Breaking ties by position instead would give ranks [1, 2, 3, 4] and exactly 1.000, so this
        # case distinguishes the two implementations rather than merely exercising one.
        G = [sqrt(2.0) 2.0 2.0 sqrt(6.0)
             0.0       0.0 1.0 0.0]
        ok, out = run_script(write_checkpoint(joinpath(dir, "ties.jld2"), G))
        @test ok
        fields = iteration_fields(out, "grid1 (3 cells)")
        @test fields[1] == "1.0000"          # Phi of the member this subset would pick
        @test fields[2] == "0.949"           # averaged ranks; position-broken ties would give 1.000
        @test fields[2] != "1.000"

        # Grid 2 objectives [0, 0, 0.5, 0]: three tied at mid-rank 2, one at 4. Against [1, 2, 3, 4]
        # that is cov 1, var 3 and 5 -> 1 / sqrt(15) = 0.2582.
        @test iteration_fields(out, "grid2 (5 cells)")[2] == "0.258"
    end

    @testset "an all-nonfinite iteration is reported, not crashed on" begin
        # Indexing the best of an empty set is what this used to do.
        ok, out = run_script(write_checkpoint(joinpath(dir, "nan.jld2"), fill(NaN, 2, 4)))
        @test ok
        @test occursin("no member has a finite objective on all grids", out)
    end

    @testset "a partially nonfinite iteration keeps the finite members" begin
        G = [sqrt(2.0) 2.0 NaN sqrt(6.0)
             0.0       0.0 NaN 0.0]
        ok, out = run_script(write_checkpoint(joinpath(dir, "partial.jld2"), G))
        @test ok
        @test occursin("3 of 4 members finite", out)
    end

    @testset "a single-grid checkpoint is refused" begin
        # There is no subset to screen on, and silently reporting nothing would read as a pass.
        ok, out = run_script(write_checkpoint(joinpath(dir, "one.jld2"), reshape([1.0, 2.0], 1, 2);
                                              n_grids = 1, cells = [3]))
        @test !ok
        @test occursin("single grid", out)
    end

    @testset "a mislabelled observation vector is refused" begin
        # Three grids' worth of z_faces with two grids' worth of observations: the block boundaries
        # would be wrong and every objective silently computed over the wrong rows.
        path = joinpath(dir, "mismatch.jld2")
        jldsave(path; z_faces = [collect(0.0:1.0:c) for c in (3, 5, 7)],
                      members = [(1, "01")], variables = ["thetal"],
                      observation_faces = [0.0, 100.0], y = zeros(2), Γ = ones(2),
                      history = [(; iteration = 1, G = zeros(2, 4), misfit = zeros(4))])
        ok, out = run_script(path)
        @test !ok
        @test occursin("Observation vector", out)
    end
end
