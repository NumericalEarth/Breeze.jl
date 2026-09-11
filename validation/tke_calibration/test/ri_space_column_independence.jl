# Batching independent calibrations into one forward map is only valid if a column's result does not
# depend on which other columns share the grid. `test/gpu_pipeline.jl` establishes that for
# `ConstantSpace`. The column indexing is shared, but the closure *array* is not: `RiDependentSpace`
# builds `RiDependentStabilityFunctions` per column, a larger struct with a different adapted type on
# the device, so the property is worth demonstrating rather than inherited.
#
# Six parameter sets are run as one ensemble and as two ensembles of three, and the time means are
# required to agree exactly. Exactly, not approximately: the columns are independent by construction
# — `(Flat, Flat, Bounded)` topology, per-column reference state, RRTMGP solving column by column —
# so any difference is a coupling that should not exist. A tolerance would hide it.
#
#     julia -t auto --project test/ri_space_column_independence.jl [arch=gpu]
using Test, BreezeCalibration, Printf, Statistics
using Oceananigans: CPU, GPU
using Oceananigans.Units

options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
get(options, "arch", "cpu") == "gpu" && @eval using CUDA
architecture = get(options, "arch", "cpu") == "gpu" ? GPU() : CPU()

members = [load_member(22, "07"), load_member(17, "07"), load_member(2, "01")]
problem = ColumnEnsembleProblem(members; Δz = 100.0)
space = RiDependentSpace()
base = collect(Float64, default_parameters(space))
sets = hcat([base .* (1 .+ 0.05 .* sin.((1:length(base)) .* i)) for i in 1:6]...)

run_kw = (; Δt = 60.0, radiation = :interactive, radiation_interval = 1800.0, architecture,
            stop_time = 40 * 60.0, averaging_window = (0.0, 40 * 60.0))

@info "RiDependentSpace column independence on $(summary(architecture)): 6 sets × $(length(members)) members"
together, _ = run_ensemble(problem, sets; space, run_kw...)
first_half, _ = run_ensemble(problem, sets[:, 1:3]; space, run_kw...)
second_half, _ = run_ensemble(problem, sets[:, 4:6]; space, run_kw...)

@testset "RiDependentSpace columns are independent of the ensemble they sit in" begin
    for v in observable_variables
        @test together[v][1:3, :, :] == first_half[v]
        @test together[v][4:6, :, :] == second_half[v]
    end
    # A guard against the test passing because nothing varies: the six sets must actually differ
    @test !all(together.θˡ[1, :, :] .== together.θˡ[6, :, :])
    @test all(isfinite, together.θˡ)
end

for v in observable_variables
    d = max(maximum(abs, together[v][1:3, :, :] .- first_half[v]), maximum(abs, together[v][4:6, :, :] .- second_half[v]))
    @printf "  %-4s max |batched - separate| = %.3e\n" v d
end
