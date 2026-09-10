# Wall time per step of the column-ensemble forward map as a function of ensemble size, to size the EKI:
# N_ens parameter sets × 8 members on the LES grid extended to 25 km with interactive radiation.
#     julia -t auto --project scripts/time_forward_map.jl N_ens [arch=cpu|gpu]
using BreezeCalibration, Printf, Random
using Oceananigans.Units
using Oceananigans: CPU, GPU
options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
get(options, "arch", "cpu") == "gpu" && @eval using CUDA
architecture = get(options, "arch", "cpu") == "gpu" ? GPU() : CPU()
N_ens = parse(Int, first(filter(a -> !occursin('=', a), ARGS)))
# Training members with short (3.7-day) records, so the ensemble stops at 88 h
members = [load_member(s, m) for (s, m) in ((22, "07"), (23, "07"), (20, "07"), (21, "07"), (14, "01"), (12, "01"), (5, "07"), (8, "07"))]
problem = ColumnEnsembleProblem(members)
println("record lengths (h): ", [round(Int, m.window[2] / 3600) for m in members], "  common axis end: ", problem.times[end] / 3600, " h")
rng = MersenneTwister(1)
params = hcat([collect(Float64, default_parameters()) .* (1 .+ 0.2 .* (rand(rng, 17) .- 0.5)) for i in 1:N_ens]...)
steps = 120
t = @elapsed run_ensemble(problem, params; stop_time = steps * 60.0, architecture)
@printf "N_ens = %d (%d columns), %s, threads = %d: %.1f s for %d steps incl. compilation\n" N_ens N_ens * length(members) summary(architecture) Threads.nthreads() t steps
t = @elapsed run_ensemble(problem, params; stop_time = steps * 60.0, architecture)
@printf "N_ens = %d (%d columns), %s, threads = %d: %.3f s/step → %.0f min per 88-h forward map\n" N_ens N_ens * length(members) summary(architecture) Threads.nthreads() t / steps (t / steps) * 88 * 60 / 60
