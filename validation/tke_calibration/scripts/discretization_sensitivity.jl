# Are the calibration's discretization choices converged? The coefficients are fit to the column's
# time-mean profiles, so what matters is not whether a run is stable at Δt = 60 s with radiation every
# 10 min, but whether the *scored* observation vector still moves as those are refined. A knob whose
# refinement moves the score by as much as the misfit the inversion is reducing is a knob the
# coefficients would absorb.
#
# Two sweeps, each against its own finest setting:
#   Δt                  — the time step (default 60, 30, 15 s)
#   radiation_interval   — how often RRTMGP is called (default 600, 1200, 1800 s)
#
# The second is also the main performance knob: interactive radiation is ~80 % of the step cost at
# production ensemble size, so if the score is insensitive between 10 and 30 min, the calibration runs
# roughly twice as fast for nothing.
#
#     julia -t auto --project scripts/discretization_sensitivity.jl [dts=60,30,15] [intervals=600,1200,1800]
#                                                                   [resolution=50] [arch=cpu|gpu]
#                                                                   [members=22/07,17/07,14/01,2/01]
#                                                                   [top=25000|les] [output=...]
using BreezeCalibration, Printf, Statistics, JLD2
using Oceananigans: CPU, GPU

const observation_scales = BreezeCalibration.observation_scales
const scored = (:θˡ, :qᵗ, :qˡ, :u, :v)

options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
dts = parse.(Float64, split(get(options, "dts", "60,30,15"), ','))
intervals = parse.(Float64, split(get(options, "intervals", "600,1200,1800"), ','))
resolution = get(options, "resolution", "50")
spec = split(get(options, "members", "22/07,17/07,14/01,2/01"), ',')
top = get(options, "top", "25000"); top = top == "les" ? nothing : parse(Float64, top)
get(options, "arch", "cpu") == "gpu" && @eval using CUDA
architecture = get(options, "arch", "cpu") == "gpu" ? GPU() : CPU()
output = get(options, "output", joinpath(@__DIR__, "..", "results", "discretization_sensitivity.jld2"))

members = [load_member(parse(Int, first(split(s, '/'))), last(split(s, '/'))) for s in spec]
problem = resolution == "hindcast" ? ColumnEnsembleProblem(members; z_faces = hindcast_faces(), top) :
          resolution == "20" ? ColumnEnsembleProblem(members; top) :
          ColumnEnsembleProblem(members; Δz = parse(Float64, resolution), top)
params = reshape(collect(Float64, default_parameters()), :, 1)

@info "Discretization sensitivity (protocol $PROTOCOL_VERSION): $(length(members)) members on the " *
      "$resolution m grid ($(length(problem.zf) - 1) cells to $(problem.zf[end]) m), $(summary(architecture))"

zo = problem.observation_zf
cell(means, v, j) = observation_scales[v] .* regrid_column(means[v][1, j, :], problem.zf, zo)
les(v, j) = observation_scales[v] .* regrid_column(members[j].targets[v], problem.les_zf, zo)

"""Run the sweep over `values` of one knob, calling `run(value)`, and report against the finest."""
function sweep(label, values, run, finest)
    runs = Dict{Float64, Any}()
    for value in values
        t = @elapsed means = run(value)
        runs[value] = means
        @printf "  %s = %-7g %7.0f s wall\n" label value t
        flush(stdout)
    end

    println("\n  RMSE against the LES below 3 km / RMS change from $label = $finest, in observation units:")
    for (j, m) in enumerate(members)
        @printf "\n  cfSite %d %s\n" m.site m.month
        @printf "    %-4s %-7s" "" "noise"
        for v in values; @printf "  %13s" @sprintf("%s=%g", label, v); end
        println()
        for v in scored
            @printf "    %-4s %-7.3f" v default_observation_noise[v]
            for value in values
                @printf "  %6.3f/%6.3f" sqrt(mean((cell(runs[value], v, j) .- les(v, j)) .^ 2)) sqrt(mean((cell(runs[value], v, j) .- cell(runs[finest], v, j)) .^ 2))
            end
            println()
        end
    end

    println("\n  Normalized by the observation noise, over every member and scored cell:")
    for value in values
        d = vcat([vcat([(cell(runs[value], v, j) .- cell(runs[finest], v, j)) ./ default_observation_noise[v] for v in scored]...) for j in eachindex(members)]...)
        e = vcat([vcat([(cell(runs[value], v, j) .- les(v, j)) ./ default_observation_noise[v] for v in scored]...) for j in eachindex(members)]...)
        @printf "    %s = %-7g RMS change from finest %6.3f σ (max %6.2f σ)   RMS misfit to the LES %6.3f σ   ratio %.3f\n" label value sqrt(mean(d .^ 2)) maximum(abs.(d)) sqrt(mean(e .^ 2)) sqrt(mean(d .^ 2)) / sqrt(mean(e .^ 2))
    end
    println("\n  The knob is converged for calibration when the change from the finest setting is small\n" *
            "  next to the misfit the inversion reduces — the last column. A ratio near 1 means the\n" *
            "  coefficients would be fitting the discretization.")
    return runs
end

# The accumulation fires on TimeInterval(10minutes), which every Δt here divides, so all runs average
# the same instants of model time over each member's own target window.
println("\n===== time step, radiation every 10 min")
dt_runs = sweep("Δt", dts, Δt -> first(run_ensemble(problem, params; Δt, architecture, radiation = :interactive)), minimum(dts))

println("\n===== radiation interval, Δt = 60 s")
interval_runs = sweep("Δtᵣ", intervals, Δtᵣ -> first(run_ensemble(problem, params; Δt = 60.0, radiation_interval = Δtᵣ, architecture, radiation = :interactive)), minimum(intervals))

isempty(dirname(output)) || mkpath(dirname(output))
jldsave(output; protocol_version = PROTOCOL_VERSION, dts, intervals,
                dt_means = Dict(k => v for (k, v) in dt_runs), interval_means = Dict(k => v for (k, v) in interval_runs),
                members = [(m.site, m.month) for m in members], zf = problem.zf, observation_zf = zo,
                resolution, top = problem.zf[end])
@info "wrote $output"
