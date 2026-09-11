# Wall time per step of the column-ensemble forward map, measured *inside* one run.
#
# The cost of a configuration is not a whole run's wall time divided by its steps, and it is not the
# difference of two runs' wall times. Setup — compilation, artifact reads, regridding every member's
# forcing onto the model grid, RRTMGP's lookup tables — is tens of seconds and varies with GPU
# contention, so dividing by steps overstates a short run several-fold and differencing two runs can
# return a negative cost per step. Both mistakes were made here and both produced confident wrong
# numbers. So: two timestamps taken during the integration, on the device, and the steps between them.
#
# The block between the timestamps must contain a whole number of radiation calls. RRTMGP is a large
# fraction of the step cost and fires only every `radiation_interval`, so a block that happens to span
# five radiation calls in one configuration and four in another compares the cadence, not the grid.
# The timestamps themselves can only be taken where `run_ensemble` samples (every 600 s of model time),
# so the block is a multiple of the least common multiple of the two cadences and is placed after a
# warmup. A requested mark the sampler never reaches is an error, not a missing key.
#
#     julia -t auto --project scripts/time_forward_map.jl nens=400 arch=gpu resolution=100 stretching=1.3
#     julia -t auto --project scripts/time_forward_map.jl nens=400 arch=gpu resolution=50 dt=7.5 radiation_interval=600
#
# Reports seconds per step, seconds per step per 1000 columns, and the hours one production forward map
# would take at this cost. That projection is per *iteration*: how many iterations a calibration needs
# is not known until its objective is stationary, so it must never be multiplied into a time to
# convergence.
using BreezeCalibration, Printf, Random, JLD2, Statistics
using SHA: sha256
using Oceananigans.Units
using Oceananigans: CPU, GPU

options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
arch_name = get(options, "arch", "cpu")
arch_name == "gpu" && @eval using CUDA
architecture = arch_name == "gpu" ? GPU() : CPU()
N_ens = parse(Int, get(options, "nens", "1"))
Δt = parse(Float64, get(options, "dt", "7.5"))
radiation_interval = parse(Float64, get(options, "radiation_interval", "600"))
# Several configurations are swept in ONE process: Julia's load and compilation is minutes, and paying
# it nine times would cost more than the measurements. Each grid size does still specialize — this
# Oceananigans encodes the size in the grid type, `GridSize{2, 4, 114, 0, 0, 3}` — but that compilation
# happens during setup and the first steps, before the warmed timing block, so it does not enter the
# measurement.
resolutions = split(get(options, "resolutions", get(options, "resolution", "50")), ',')
stretchings = parse.(Float64, split(get(options, "stretchings", get(options, "stretching", "1.12")), ','))
top = parse(Float64, get(options, "top", "25000"))
warmup_steps = parse(Int, get(options, "warmup", "240"))
minimum_block = parse(Int, get(options, "block", "400"))
sites = parse.(Int, split(get(options, "sites", "2,4,5,7,8,10,11,13,14,17,19,20,22,23"), ','))
months = split(get(options, "months", "01,04,07,10"), ',')
output = get(options, "output", nothing)

members = [load_member(s, m) for s in sites for m in months]
grid_for(resolution, stretching) =
    resolution == "hindcast" ? ColumnEnsembleProblem(members; z_faces = hindcast_faces(), top, stretching) :
    resolution == "20" ? ColumnEnsembleProblem(members; top, stretching) :
    ColumnEnsembleProblem(members; Δz = parse(Float64, resolution), top, stretching)

# The parameter values do not set the cost, but identical columns are avoided so that no branch is
# uniformly taken across the ensemble. Column 1 is the exact default.
rng = MersenneTwister(1)
ϕ = collect(Float64, default_parameters())
params = hcat(ϕ, [ϕ .* (1 .+ 0.05 .* (rand(rng, length(ϕ)) .- 0.5)) for _ in 2:N_ens]...)

# The sampler fires on TimeInterval(10minutes); radiation fires on `radiation_interval`. Only step
# counts that are multiples of the sampling cadence are observable, and only blocks that are multiples
# of both cadences compare like with like.
sample_seconds = 600.0
isinteger(sample_seconds / Δt) || error("Δt = $Δt s does not divide the $(sample_seconds) s sampling interval, so no timestamp can be taken at a known step")
isinteger(radiation_interval / Δt) || error("Δt = $Δt s does not divide the radiation interval $radiation_interval s")
n_sample = round(Int, sample_seconds / Δt)
n_radiation = round(Int, radiation_interval / Δt)
period = lcm(n_sample, n_radiation)
first_mark = cld(warmup_steps, n_sample) * n_sample
block = cld(minimum_block, period) * period
last_mark = first_mark + block
stop_seconds = last_mark * Δt

columns = N_ens * length(members)
production_steps = maximum(m.window[2] for m in members) / Δt
@printf "%d cases × %d parameter sets = %d columns, %s\n" length(members) N_ens columns summary(architecture)
@printf "Δt = %.4g s, radiation every %.0f s (%d steps), sampled every %d steps\n" Δt radiation_interval n_radiation n_sample
@printf "timing steps %d→%d (%d steps = %d radiation calls), each run stops at %.0f s of model time\n\n" first_mark last_mark block (block ÷ n_radiation) stop_seconds
# Redirected stdout is block-buffered, so without this the header does not reach the log until the
# first measurement finishes — minutes in which a running job looks hung.
flush(stdout)

function measure(resolution, stretching)
    problem = grid_for(resolution, stretching)
    marks = Dict{Int, Float64}()
    function stamp(model, t, active)
        iteration = model.clock.iteration
        if iteration == first_mark || iteration == last_mark
            # Kernel launches are asynchronous: without this the timestamp records when the work was
            # queued, not when it finished.
            architecture isa GPU && CUDA.synchronize()
            marks[iteration] = time_ns() / 1e9
        end
        return nothing
    end
    _, _, timing = run_ensemble(problem, params; Δt, architecture, radiation = :interactive, radiation_interval,
                                stop_time = stop_seconds, averaging_window = (0.0, stop_seconds),
                                sample_callback = stamp)
    for mark in (first_mark, last_mark)
        haskey(marks, mark) && continue
        error("No timestamp at step $mark: the sampler fires every $n_sample steps and reached " *
              "$(sort(collect(keys(marks)))) — steps run were $(timing.steps). Choose marks that are " *
              "multiples of $n_sample.")
    end
    return (; resolution = String(resolution), stretching, cells = length(problem.zf) - 1, zf = problem.zf,
              seconds_per_step = (marks[last_mark] - marks[first_mark]) / block,
              whole_run_seconds_per_step = timing.seconds_per_step, setup_seconds = timing.setup_seconds)
end

# Enough identity to tell whether two result files may be compared at all. A retry that lands on a
# different GPU, or after a source change, is a different measurement and must not be merged with the
# earlier one; and the parameters are recorded as the values actually run, not as the name of the
# function they were derived from — they are perturbed around the default, not equal to it.
device_identity() =
    architecture isa GPU ?
        (; name = CUDA.name(CUDA.device()),
           uuid = try string(CUDA.uuid(CUDA.device())) catch; "unavailable" end) :
        (; name = "CPU", uuid = "")

const provenance = (; revision = readchomp(`git rev-parse HEAD`),
                      diff_sha256 = bytes2hex(sha256(read(`git diff --binary`))),
                      driver_sha256 = bytes2hex(sha256(read(@__FILE__))),
                      device = device_identity(), hostname = gethostname(),
                      slurm_job = get(ENV, "SLURM_JOB_ID", ""))

function write_results(path, results)
    isempty(dirname(path)) || mkpath(dirname(path))
    temporary = path * ".tmp"
    jldsave(temporary; protocol_version = PROTOCOL_VERSION, results, columns, N_ens, Δt,
                       radiation_interval, top, first_mark, last_mark, block, production_steps,
                       architecture = arch_name, params, provenance,
                       members = [(m.site, m.month) for m in members])
    mv(temporary, path; force = true)
    return nothing
end

@printf "device %s\n" device_identity().name
@printf "  %-10s %10s %7s %12s %12s %11s %12s\n" "grid" "stretching" "cells" "s/step" "per 1k col" "map (h)" "vs 1st"
results = NamedTuple[]
for resolution in resolutions, stretching in stretchings
    r = measure(resolution, stretching)
    # The reference is read back out of `results` rather than held in a variable assigned inside this
    # loop: a top-level `for` body is soft scope, so such an assignment creates a new local and the
    # read of it fails with UndefVarError. `results` is only mutated, never reassigned, so it is fine.
    push!(results, r)
    @printf "  %-10s %10.2f %7d %12.5f %12.5f %11.2f %11.3f×\n" r.resolution r.stretching r.cells r.seconds_per_step (r.seconds_per_step * 1000 / columns) (r.seconds_per_step * production_steps / 3600) (results[1].seconds_per_step / r.seconds_per_step)
    flush(stdout)
    # Each measurement is written as it is taken. The write rebuilds the whole file, so it goes to a
    # temporary and is renamed into place: a preemption during a plain in-place rewrite would destroy
    # the measurements already in it, which is exactly the failure this is meant to survive.
    isnothing(output) || write_results(output, results)
end

println("\n  `s/step` is the in-run block; `whole run / steps` is the discredited estimate, kept only")
println("  to show how far off it is:")
for r in results
    @printf "    %-10s s=%.2f   block %.5f   whole run / steps %.5f   setup %.0f s\n" r.resolution r.stretching r.seconds_per_step r.whole_run_seconds_per_step r.setup_seconds
end
println("\n  `map (h)` is per ITERATION. How many iterations a calibration needs is not known until its")
println("  objective is stationary, so this must not be multiplied out into a time to convergence.")
isnothing(output) || @info "wrote $output"
