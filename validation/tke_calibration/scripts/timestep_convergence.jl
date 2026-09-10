# Is the calibration's time step converged? The coefficients are fit to the column's time-mean profiles,
# so what matters is not whether the run is stable at Δt = 60 s but whether the *scored* observation
# vector still moves as Δt falls. If it moves by as much as the observation noise, the inversion is
# fitting a discretization error and the calibration should use a smaller step.
#
#     julia -t auto --project scripts/timestep_convergence.jl [dts=60,30,15] [resolution=50] [arch=cpu|gpu]
#                                                             [members=22/07,17/07,14/01,2/01] [top=25000|les]
#                                                             [radiation=interactive|prescribed] [output=...]
#
# One column-ensemble run of the default closure per Δt, over the same members, on the production protocol.
# Prints, per member and field, the RMSE against the LES and — the actual question — the RMS change from
# the finest step, both in observation units next to the observation noise.
using BreezeCalibration, Printf, Statistics, JLD2
using Oceananigans: CPU, GPU

options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
dts = parse.(Float64, split(get(options, "dts", "60,30,15"), ','))
resolution = get(options, "resolution", "50")
spec = split(get(options, "members", "22/07,17/07,14/01,2/01"), ',')
top = get(options, "top", "25000"); top = top == "les" ? nothing : parse(Float64, top)
radiation = Symbol(get(options, "radiation", "interactive"))
get(options, "arch", "cpu") == "gpu" && @eval using CUDA
architecture = get(options, "arch", "cpu") == "gpu" ? GPU() : CPU()
output = get(options, "output", joinpath(@__DIR__, "..", "results", "timestep_convergence.jld2"))

members = [load_member(parse(Int, first(split(s, '/'))), last(split(s, '/'))) for s in spec]
problem = resolution == "hindcast" ? ColumnEnsembleProblem(members; z_faces = hindcast_faces(), top) :
          resolution == "20" ? ColumnEnsembleProblem(members; top) :
          ColumnEnsembleProblem(members; Δz = parse(Float64, resolution), top)
params = reshape(collect(Float64, default_parameters()), :, 1)

@info "Δt convergence: $(length(members)) members × $(length(dts)) steps on the $resolution m grid " *
      "($(length(problem.zf) - 1) cells to $(problem.zf[end]) m, radiation $radiation, $(summary(architecture)))"

# The accumulation fires on TimeInterval(10minutes), which every Δt here divides, so all runs average the
# same instants of model time over each member's own target window.
runs = Dict{Float64, Any}()
for Δt in dts
    t = @elapsed means, _ = run_ensemble(problem, params; Δt, radiation, architecture)
    runs[Δt] = means
    @printf "  Δt = %5.1f s: %6.0f s wall (%d steps)\n" Δt t round(Int, problem.times[end] / Δt)
    flush(stdout)
end

# Scores and differences on the observation cells, in observation units
finest = minimum(dts)
zo = problem.observation_zf
nk = length(zo) - 1
scored = (:θˡ, :qᵗ, :qˡ, :u, :v)
const observation_scales = BreezeCalibration.observation_scales
cell(means, v, j) = observation_scales[v] .* regrid_column(means[v][1, j, :], problem.zf, zo)
les(v, j) = observation_scales[v] .* regrid_column(members[j].targets[v], problem.les_zf, zo)

println("\nRMSE against the LES below 3 km, and the RMS change from Δt = $finest s (observation units):")
for (j, m) in enumerate(members)
    @printf "\ncfSite %d %s:\n" m.site m.month
    @printf "  %-4s %-8s" "" "noise"
    for Δt in dts; @printf "  %11s" @sprintf("Δt=%gs", Δt); end
    println()
    for v in scored
        @printf "  %-4s %-8.3f" v default_observation_noise[v]
        for Δt in dts
            @printf "  %5.3f/%5.3f" sqrt(mean((cell(runs[Δt], v, j) .- les(v, j)) .^ 2)) sqrt(mean((cell(runs[Δt], v, j) .- cell(runs[finest], v, j)) .^ 2))
        end
        println("     (RMSE vs LES / RMS change from finest)")
    end
end

# The verdict: the change from the finest step, relative to the observation noise, over all members and
# fields at once — the same normalization the inversion's misfit uses
println("\nNormalized by the observation noise, over every member and scored cell:")
for Δt in dts
    d = vcat([vcat([(cell(runs[Δt], v, j) .- cell(runs[finest], v, j)) ./ default_observation_noise[v] for v in scored]...) for j in eachindex(members)]...)
    e = vcat([vcat([(cell(runs[Δt], v, j) .- les(v, j)) ./ default_observation_noise[v] for v in scored]...) for j in eachindex(members)]...)
    @printf "  Δt = %5.1f s: RMS change from finest %.3f σ (max %.2f σ), RMS misfit to the LES %.3f σ\n" Δt sqrt(mean(d .^ 2)) maximum(abs.(d)) sqrt(mean(e .^ 2))
end
println("\nThe step is converged for calibration when the change from the finest step is small next to " *
        "the misfit the inversion is trying to reduce; if it is a sizeable fraction of it, the " *
        "coefficients would absorb the discretization error.")

isempty(dirname(output)) || mkpath(dirname(output))
jldsave(output; dts, means = Dict(Δt => runs[Δt] for Δt in dts), members = [(m.site, m.month) for m in members],
                zf = problem.zf, observation_zf = zo, resolution, radiation = String(radiation),
                top = problem.zf[end])
@info "wrote $output"
