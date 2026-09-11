# Are the calibration's discretization choices converged? The coefficients are fit to the column's
# time-mean profiles, so what matters is not whether a run is stable at Δt = 60 s with radiation every
# 10 min, but whether the *scored* observation vector still moves as those are refined. A knob whose
# refinement moves the score by as much as the misfit the inversion is reducing is a knob the
# coefficients would absorb.
#
# Three sweeps, each against its own finest setting:
#   Δt                  — the time step (default 60, 30, 15 s)
#   radiation_interval   — how often RRTMGP is called (default 600, 1200, 1800 s)
#   stretching           — the growth ratio of the faces *above the LES top* (default 1.12, 1.3, 1.4)
#
# Radiation is a substantial part of the cost. Test its sampling interval independently: a faster
# forward map is useful only if the scored profiles remain sufficiently accurate.
#
# The stretched extension above the LES top exists only so RRTMGP has an atmosphere, and it is 30 %,
# 41 % and 43 % of the 50 m, 100 m and hindcast grids at the default 1.12. `extend_faces` grows the
# faces geometrically from the LES top upward, so coarsening it leaves every cell below the LES top —
# and so every scored cell — unchanged. The only paths by which it can move the score are the radiative
# heating profile and the upper relaxation. That makes it a narrow, physical sensitivity rather than a
# general accuracy trade, which is why it is worth testing separately from Δz.
#
# THE DECISION RULE, for all three: a setting is adopted when the RMS change from the finest setting is
# ≤ 0.1 σ AND the relative change in the objective Φ is ≤ 1 %. The ratio of the change to the misfit is
# printed as context, but it is not the criterion — a small ratio can still hide a change in Φ large
# enough to move where the inversion goes. Both numbers are reported for every setting.
#
#     julia -t auto --project scripts/discretization_sensitivity.jl [dts=60,30,15] [intervals=600,1200,1800]
#                                                                   [stretchings=1.12,1.3,1.4] [modes=dt,radiation,stretching]
#                                                                   [resolution=50] [arch=cpu|gpu]
#                                                                   [members=22/07,17/07,14/01,2/01]
#                                                                   [top=25000|les] [output=...]
#
# NOTE the sweeps run at whatever `params` is — `default_parameters()` unless `checkpoint=` supplies a
# fitted candidate. A fitted candidate can be far more sensitive to a numerical choice than the default
# it was measured around (design B was), so a pass at the default licenses a setting for exploration
# only, and must be repeated against a real candidate before it is adopted for production.
using BreezeCalibration, Printf, Statistics, JLD2
using LinearAlgebra: diag
using Oceananigans: CPU, GPU
include(joinpath(@__DIR__, "calibration_data_manifest.jl"))

const observation_scales = BreezeCalibration.observation_scales
const scored = (:θˡ, :qᵗ, :qˡ, :u, :v)

options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
dts = parse.(Float64, split(get(options, "dts", "60,30,15"), ','))
intervals = parse.(Float64, split(get(options, "intervals", "600,1200,1800"), ','))
stretchings = parse.(Float64, split(get(options, "stretchings", "1.12,1.3,1.4"), ','))
modes = split(get(options, "modes", "dt,radiation"), ',')
radiation_interval = parse(Float64, get(options, "radiation_interval", "600"))
radiation_dt = parse(Float64, get(options, "radiation_dt", "15"))
# The time step and radiation interval the stretching sweep is run at; the extension is being tested
# for production, so it is tested at the production discretization rather than at the finest one.
stretching_dt = parse(Float64, get(options, "stretching_dt", "7.5"))
base_stretching = parse(Float64, get(options, "stretching", "1.12"))
resolution = get(options, "resolution", "50")
spec = split(get(options, "members", "22/07,17/07,14/01,2/01"), ',')
top = get(options, "top", "25000"); top = top == "les" ? nothing : parse(Float64, top)
get(options, "arch", "cpu") == "gpu" && @eval using CUDA
architecture = get(options, "arch", "cpu") == "gpu" ? GPU() : CPU()
output = get(options, "output", joinpath(@__DIR__, "..", "results", "discretization_sensitivity.jld2"))

# `members=` names site/month pairs one by one; `sites=`/`months=` take their cross product, which is
# how the production runs are specified, so a sweep can be pointed at the training set without
# spelling out 56 pairs and risking a typo in one of them.
members = if haskey(options, "sites") || haskey(options, "months")
    haskey(options, "members") && error("Pass either members= or sites=/months=, not both")
    [load_member(s, m) for s in parse.(Int, split(options["sites"], ','))
                       for m in split(get(options, "months", "01,04,07,10"), ',')]
else
    [load_member(parse(Int, first(split(s, '/'))), last(split(s, '/'))) for s in spec]
end
# The stretching sweep changes the grid itself, so each setting gets its own problem. The observation
# cells do not change with it, so every grid is still scored on the same 100 m cells against the same
# targets, and the objectives below are directly comparable.
grid_for(stretching) =
    resolution == "hindcast" ? ColumnEnsembleProblem(members; z_faces = hindcast_faces(), top, stretching) :
    resolution == "20" ? ColumnEnsembleProblem(members; top, stretching) :
    ColumnEnsembleProblem(members; Δz = parse(Float64, resolution), top, stretching)
problem = grid_for(base_stretching)
params = reshape(collect(Float64, default_parameters()), :, 1)
source_checkpoint = get(options, "checkpoint", nothing)
if !isnothing(source_checkpoint)
    saved = load(source_checkpoint)
    saved["protocol_version"] == PROTOCOL_VERSION || error("Checkpoint physics version differs")
    validate_data_manifest(saved)
    haskey(saved, "selected_mean") && !isnothing(saved["selected_mean"]) ||
        error("Checkpoint must contain a directly evaluated selected mean")
    params = reshape(copy(saved["selected_mean"].parameters), :, 1)
end

@info "Discretization sensitivity (protocol $PROTOCOL_VERSION): $(length(members)) members on the " *
      "$resolution m grid ($(length(problem.zf) - 1) cells to $(problem.zf[end]) m), $(summary(architecture))"

zo = problem.observation_zf
# A run carries the grid it was produced on, because the stretching sweep changes it: regridding a
# coarse-extension run onto the observation cells through the *reference* grid's faces would silently
# misplace the profile.
cell(run, v, j) = observation_scales[v] .* regrid_column(run.means[v][1, j, :], run.problem.zf, zo)
les(v, j) = observation_scales[v] .* regrid_column(members[j].targets[v], problem.les_zf, zo)

# The objective the inversion actually minimizes, Φ = ½⟨((G − y)/σ)²⟩, so a setting can be judged by
# what it does to the quantity being optimized and not only by how far the profiles move. `y` and `σ`
# depend on the members and the observation cells alone, so they are common to every setting here.
y_reference, Γ_reference = observations(problem)
σ_reference = sqrt.(diag(Γ_reference))
function observation_vector_of(run)
    vars = run.problem.variables
    return vcat([BreezeCalibration.observation_vector(run.problem,
                    NamedTuple{vars}(Tuple(run.means[v][1, j, :] for v in vars))) for j in eachindex(members)]...)
end
function objective_of(run)
    y, Γ = observations(run.problem)
    y == y_reference && sqrt.(diag(Γ)) == σ_reference ||
        error("A sweep setting changed the observations themselves; its Φ would not be comparable")
    return mean(abs2, (observation_vector_of(run) .- y) ./ σ_reference) / 2
end

"""
Run the sweep over `values` of one knob and report against the finest. `run(value)` returns the time
means; `grid(value)` the problem they were produced on, which only the stretching sweep varies; `settings`
the (Δt, radiation interval, stretching) each value implies, which is what a saved partial is validated
against.
"""
function sweep(label, values, run, finest, grid, settings)
    runs = Dict{Float64, Any}()
    tag = label == "Δt" ? "dt" : label == "Δtᵣ" ? "radiation" : "stretching"
    for value in values
        problem_value = grid(value)
        wanted = settings(value)
        # Preserve every expensive run independently, even if a later refinement fails: a sweep is
        # several hours of GPU time and a failure in the ninth run must not discard the first eight.
        isempty(dirname(output)) || mkpath(dirname(output))
        partial = output * "." * tag * ".$value.jld2"
        reference = get(options, "reference", "")
        reusable = isfile(partial) ? partial : reference
        reused = false
        if isfile(reusable)
            saved = load(reusable)
            same_protocol = saved["protocol_version"] == PROTOCOL_VERSION &&
                            saved["params"] == params && saved["zf"] == problem_value.zf &&
                            saved["observation_zf"] == zo &&
                            saved["members"] == [(m.site, m.month) for m in members]
            # New partials record the settings they were run at outright. Older ones record only the
            # swept value and have to have the other two derived from the sweep they belonged to, so
            # keep that fallback rather than discard partials that are still perfectly valid.
            saved_settings = if haskey(saved, "run_settings")
                saved["run_settings"]
            else
                (; Δt = saved["label"] == "Δt" ? saved["value"] : saved["radiation_dt"],
                   radiation_interval = saved["label"] == "Δt" ? saved["radiation_interval"] : saved["value"],
                   stretching = get(saved, "stretching", base_stretching))
            end
            reused = same_protocol && saved_settings.Δt == wanted.Δt &&
                     saved_settings.radiation_interval == wanted.radiation_interval &&
                     saved_settings.stretching == wanted.stretching
            if reused
                means, t = saved["means"], 0.0
                @info "Reusing validated forward evaluation" reusable
            elseif reusable == partial
                error("Existing partial result has incompatible settings: $partial " *
                      "(saved $(saved_settings), wanted $(wanted))")
            end
        end
        if !reused
            t = @elapsed means = run(value)
        end
        runs[value] = (; means, problem = problem_value)
        if !(reused && reusable == partial)
            jldsave(partial; protocol_version = PROTOCOL_VERSION, label, value, means, params,
                        radiation_interval, radiation_dt, source_checkpoint, run_settings = wanted,
                        stretching = wanted.stretching,
                        members = [(m.site, m.month) for m in members], zf = problem_value.zf,
                        observation_zf = zo, resolution, top = problem_value.zf[end], wall = t,
                            cells = length(problem_value.zf) - 1,
                            reused_from = reused ? reusable : nothing)
        end
        @printf "  %s = %-7g %6d cells %7.0f s wall\n" label value (length(problem_value.zf) - 1) t
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

    println("\n  Normalized by the observation noise, over every member and scored cell, with the")
    println("  objective Φ = ½⟨((G − y)/σ)²⟩ the inversion actually minimizes:")
    Φ_finest = objective_of(runs[finest])
    @printf "    %-14s %10s %8s %10s %10s %9s   %s\n" label "RMS change" "max" "Φ" "ΔΦ/Φ" "ratio" "verdict"
    verdicts = Dict{Float64, Any}()
    for value in values
        d = vcat([vcat([(cell(runs[value], v, j) .- cell(runs[finest], v, j)) ./ default_observation_noise[v] for v in scored]...) for j in eachindex(members)]...)
        e = vcat([vcat([(cell(runs[value], v, j) .- les(v, j)) ./ default_observation_noise[v] for v in scored]...) for j in eachindex(members)]...)
        rms, Φ = sqrt(mean(d .^ 2)), objective_of(runs[value])
        relative = abs(Φ - Φ_finest) / Φ_finest
        pass = rms ≤ 0.1 && relative ≤ 0.01
        verdicts[value] = (; rms, maximum_change = maximum(abs.(d)), Φ, relative_ΔΦ = relative,
                             misfit = sqrt(mean(e .^ 2)), ratio = rms / sqrt(mean(e .^ 2)), pass)
        @printf "    %-14g %8.3f σ %6.2f σ %10.4f %9.2f%% %9.3f   %s\n" value rms maximum(abs.(d)) Φ (100relative) (rms / sqrt(mean(e .^ 2))) (value == finest ? "reference" : pass ? "PASS" : "FAIL")
    end
    println("\n  A setting is adopted when the RMS change from the finest is ≤ 0.1 σ AND the relative")
    println("  change in Φ is ≤ 1 %. Both are required: the profiles can move little on average while")
    println("  still shifting the objective enough to move where the inversion goes. The ratio to the")
    println("  misfit is context, not the criterion.")
    println("\n  These pass/fail marks are for this knob alone. Two knobs that each pass separately are")
    println("  not thereby jointly safe — Δt and the radiation interval each passed and their")
    println("  combination (7.5 s with 1800 s radiation) was rejected — so any pairing, including a")
    println("  coarse extension with Float32, needs its own joint check before it is used together.")
    return runs, verdicts
end

# The accumulation fires on TimeInterval(10minutes), which every Δt here divides, so all runs average
# the same instants of model time over each member's own target window.
println("\n===== time step, radiation every $(radiation_interval / 60) min")
dt_runs, dt_verdicts = "dt" in modes ?
    sweep("Δt", dts, Δt -> first(run_ensemble(problem, params; Δt, architecture,
              radiation_interval, radiation = :interactive)), minimum(dts), _ -> problem,
          Δt -> (; Δt, radiation_interval, stretching = base_stretching)) : (Dict(), Dict())

println("\n===== radiation interval, Δt = $radiation_dt s")
interval_runs, interval_verdicts = "radiation" in modes ?
    sweep("Δtᵣ", intervals, Δtᵣ -> first(run_ensemble(problem, params;
              Δt = radiation_dt, radiation_interval = Δtᵣ, architecture, radiation = :interactive)),
          minimum(intervals), _ -> problem,
          Δtᵣ -> (; Δt = radiation_dt, radiation_interval = Δtᵣ, stretching = base_stretching)) : (Dict(), Dict())

# The extension above the LES top. Each setting runs on its own grid; nothing below the LES top moves,
# so the scored cells are resolved identically and only the radiation and the upper relaxation differ.
println("\n===== stretching above the LES top, Δt = $stretching_dt s, radiation every $(radiation_interval / 60) min")
stretching_runs, stretching_verdicts = "stretching" in modes ?
    sweep("s", stretchings, s -> first(run_ensemble(grid_for(s), params; Δt = stretching_dt,
              radiation_interval, architecture, radiation = :interactive)), minimum(stretchings), grid_for,
          s -> (; Δt = stretching_dt, radiation_interval, stretching = s)) : (Dict(), Dict())

isempty(dirname(output)) || mkpath(dirname(output))
# Only the time means are saved, never the problems the runs carry: a `ColumnEnsembleProblem` holds
# `static_stability`, which JLD2 reconstructs as a type that has no `keys` and does not compare equal
# to itself across two loads of the same file.
jldsave(output; protocol_version = PROTOCOL_VERSION, dts, intervals, stretchings, modes, params,
                radiation_interval, radiation_dt, stretching_dt, base_stretching, source_checkpoint,
                dt_means = Dict(k => v.means for (k, v) in dt_runs),
                interval_means = Dict(k => v.means for (k, v) in interval_runs),
                stretching_means = Dict(k => v.means for (k, v) in stretching_runs),
                stretching_zf = Dict(k => v.problem.zf for (k, v) in stretching_runs),
                verdicts = (; dt = dt_verdicts, radiation = interval_verdicts, stretching = stretching_verdicts),
                members = [(m.site, m.month) for m in members], zf = problem.zf, observation_zf = zo,
                resolution, top = problem.zf[end])
@info "wrote $output"
