# Calibrate the parameters of `TKEBasedTurbulenceClosure` against a training subset of the Shen et al. (2022)
# LES library with ensemble Kalman inversion, on one or several vertical grids at once.
#
#     julia -t auto --project scripts/calibrate.jl [N_ens] [space=ri|constant] [resolutions=50,100,hindcast] [arch=cpu|gpu]
#                                                  [top=25000|les] [radiation=interactive|prescribed] [variables=θˡ,qᵗ,qˡ,u,v]
#                                                  [pseudotime=1] [max_iterations=50] [localization=secnice|none]
#                                                  [sites=2,5,8,...] [months=01,07]
#                                                  [output=...] [resume=...]
#
# `space=ri` (default) calibrates the 17 parameters of the Ri-dependent stability functions, `space=constant`
# the 7 of the constant-coefficient (Nakanishi–Niino-form) closure. `resolutions` is a comma-separated list of
# uniform spacings in m and/or `hindcast` (NumericalEarth's stretched grid truncated to the LES depth); with
# several, every EKI iteration runs one column ensemble per grid and fits all of them at once. Every column
# ensemble is N_ens parameter sets × N_members LES members as independent columns, 3.7–6 days at Δt = 60 s, on
# the CPU (one thread per core helps) or, with `arch=gpu`, on a CUDA GPU.
#
# By default the columns extend to 25 km with interactive RRTMGP radiation (the protocol of the LES); `top=les
# radiation=prescribed` instead ends the column at the LES top and replays the LES's hourly radiative heating.
#
# The run stops when the pseudo time reaches `pseudotime`, not after a set number of iterations: the step is
# EnsembleKalmanProcesses' adaptive Iglesias–Yang data-misfit controller, Δtₙ = 1 / (mean squared normalized misfit),
# and the pseudo time 1 is where the ensemble approximates the posterior. `resume=<checkpoint>` replays a saved run
# and continues it toward the target; pass the same space and resolutions. `variables=` selects the observed
# fields (all five by default, with noise 0.25 K, 0.25 and 0.1 g/kg, 0.5 m/s).
using BreezeCalibration, Statistics, Printf, Random
using Oceananigans: CPU, GPU

positional = filter(a -> !occursin('=', a), ARGS)
options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))

space = get(options, "space", "ri") == "ri" ? RiDependentSpace() :
        get(options, "space", "ri") == "constant" ? ConstantSpace() : error("space must be ri or constant")
N_ens = length(positional) ≥ 1 ? parse(Int, positional[1]) : (space isa RiDependentSpace ? 200 : 100)
resolutions = split(get(options, "resolutions", "50,100,hindcast"), ',')
target_pseudotime = parse(Float64, get(options, "pseudotime", "1"))
max_iterations = parse(Int, get(options, "max_iterations", "50"))
resume = get(options, "resume", nothing)
top = get(options, "top", "25000"); top = top == "les" ? nothing : parse(Float64, top)   # the column top; `les` ends it at the LES top
radiation = Symbol(get(options, "radiation", "interactive"))                              # interactive (RRTMGP) or prescribed (the LES's heating)
get(options, "arch", "cpu") == "gpu" && @eval using CUDA
architecture = get(options, "arch", "cpu") == "gpu" ? GPU() : CPU()
# Localization corrects the sampling error of a small ensemble's covariance. It is worth turning off
# to see what a large ensemble does without it, since the correction itself biases the update.
localization_method = get(options, "localization", "secnice") == "none" ? NoLocalization() : SECNice()
# Training members: by default two months at eight sites along the transect — Peru and California
# stratocumulus, the deep tropics, and the trades — leaving the other months and sites for evaluation.
#
# The forward map's cost depends on the total number of columns, N_ens × N_members, and on a GPU that
# cost is strongly sublinear (eight times the columns for 1.62 times the time). Training members and
# ensemble members are therefore interchangeable in the budget, and they buy different things: a
# larger ensemble reduces the sampling error of the covariance, more members reduce generalization
# error. `sites` and `months` spend the budget on the second.
sites = parse.(Int, split(get(options, "sites", "2,5,8,11,14,17,20,23"), ','))
months = String.(split(get(options, "months", "01,07"), ','))
training = [(site, month) for site in sites for month in months]
available = Set(library_members())
missing_members = filter(m -> m ∉ available, training)
isempty(missing_members) || error("The library has no member for $missing_members")

# The training split is part of what identifies a calibration, so it belongs in the default filename
tag = (space isa RiDependentSpace ? "ri" : "constant") * "_" * join(resolutions, "_") *
      (isnothing(top) ? "" : "_top$(round(Int, top))") * (radiation == :interactive ? "_rrtmgp" : "") *
      "_n$(length(training))"
output = get(options, "output", joinpath(@__DIR__, "..", "results", "eki_$tag.jld2"))

@info "Loading $(length(training)) training members"
members = [load_member(s, m) for (s, m) in training]

variables = Tuple(Symbol.(split(get(options, "variables", join(String.(default_variables), ',')), ',')))
problem_for(resolution) = resolution == "hindcast" ? ColumnEnsembleProblem(members; z_faces = hindcast_faces(), variables, top) :
                          resolution == "20" ? ColumnEnsembleProblem(members; variables, top) :
                          ColumnEnsembleProblem(members; Δz = parse(Float64, resolution), variables, top)
problems = [problem_for(r) for r in resolutions]
problem = length(problems) == 1 ? problems[1] : MultiResolutionProblem(problems)
cells = join([string(length(p.zf) - 1) for p in problems], ", ")

if isnothing(resume)
    @info "Running EKI in $space with $N_ens ensemble members to pseudo time $target_pseudotime on grids of $cells cells ($(N_ens * length(training)) columns per grid per forward map, $(summary(architecture)), $(Threads.nthreads()) threads, radiation $radiation) → $output"
else
    @info "Resuming EKI from $resume toward pseudo time $target_pseudotime ($space, grids of $cells cells)"
end
ekp, prior, ϕ, history = run_eki(problem; space, N_ens, target_pseudotime, max_iterations, output, resume, radiation, architecture, localization_method)

println("\nfinal ensemble (constrained parameters):")
for (k, name) in enumerate(parameter_names(space))
    @printf "  %-5s mean %.3f  std %.3f   (default %.3f)\n" name mean(ϕ[k, :]) std(ϕ[k, :]) getproperty(default_parameters(space), name)
end
