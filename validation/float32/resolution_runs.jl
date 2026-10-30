# Resolution study: Breeze decaying (L1a) and forced (L1b) turbulence in the NCAR 32³ box (L = 41.4 mm),
# started from an NCAR spectral field, with the headline small-scale diagnostics of the study.
#
#   julia --project=validation/ncar_dns validation/ncar_dns/resolution_runs.jl MODE ZK_FILE RUNS STOP_TIME OUTPUT
#
# - MODE: decay (no forcing) or forced (ConstantPowerBandForcing, ε = 0.05, the 9 NCAR modes |k| ≤ 1.5·2π/L)
# - ZK_FILE: NCAR Zk*.ncf initial condition (imported exactly onto each C-grid; t = 0 of the run)
# - RUNS: comma-separated M:advection list, e.g. "64:centered4,128:centered4,128:centered6"
# - STOP_TIME (s)
# - OUTPUT: decay: text file of output times (s since the start); forced: output interval (s)
#
# Environment: CFL (0.3), OUTPUT_DIR (validation/ncar_dns/output/resolution), LABEL (case label, default R),
#   SLOW_EVERY (forced: gradient moments every SLOW_EVERY outputs, default 5), SNAPSHOTS (decay: write
#   Fourier snapshots truncated to 128³ for pointwise comparisons, default 1), BOX_N (32).
#
# Per run, writes <OUTPUT_DIR>/breeze_<LABEL>_M<M>_<advection>_{timeseries,spectra}.csv, snapshots
# <OUTPUT_DIR>/snapshots/<...>_t<n>.nc (decay), and a timing line (GPU ms per step without diagnostics).

include(joinpath(@__DIR__, "NCARDNS.jl"))
using .NCARDNS
using Breeze
using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using Printf
using Statistics
using CUDA
using FFTW

FFTW.set_num_threads(max(1, Threads.nthreads()))

const ε_injection = 0.05

function new_rows()
    names = (:time, :iteration, :time_step, :kinetic_energy, :rms_velocity, :dissipation, :taylor_microscale,
             :taylor_reynolds, :kolmogorov_length, :kmax_eta, :integral_scale, :eddy_time, :injection_power,
             :forced_band_energy, :max_velocity,
             :skewness, :flatness, :transverse_flatness, :gradient_variance_ratio,
             :skewness_fd, :flatness_fd, :transverse_flatness_fd, :dissipation_fd, :relative_divergence_fd)
    return NamedTuple{names}(Tuple(n === :iteration ? Int[] : Float64[] for n in names))
end

function record!(state, simulation)
    model = simulation.model
    (; rows, spectra, forcing, L, ν, M, slow_every, snapshot_prefix, mode) = state
    !isempty(rows.time) && model.clock.time == rows.time[end] && return nothing
    u = Array{Float64}(interior(model.velocities.u))
    v = Array{Float64}(interior(model.velocities.v))
    w = Array{Float64}(interior(model.velocities.w))
    F = fourier_velocity_from_arrays(u, v, w, L; ν)
    n = length(rows.time)
    slow = mode == "decay" || n % slow_every == 0
    s = turbulence_statistics(F; skewness=false)
    spectrum = shell_spectrum(F)
    t = model.clock.time
    P = forcing === nothing || model.clock.iteration == 0 ? NaN :
        injection_power(F, forcing.band.wavevectors, Array(forcing.amplitudes), L)

    push!(rows.time, t); push!(rows.iteration, model.clock.iteration); push!(rows.time_step, simulation.Δt)
    for name in (:kinetic_energy, :rms_velocity, :dissipation, :taylor_microscale, :taylor_reynolds,
                 :kolmogorov_length, :kmax_eta, :integral_scale, :eddy_time)
        push!(getfield(rows, name), getfield(s, name))
    end
    push!(rows.injection_power, P); push!(rows.forced_band_energy, spectrum.energy[1])
    push!(rows.max_velocity, max(maximum(abs, u), maximum(abs, v), maximum(abs, w)))

    if slow
        g = gradient_moments(F)
        gf = finite_difference_gradient_moments(u, v, w, L)
        fd = finite_difference_statistics(u, v, w, L; ν)
        push!(rows.skewness, g.skewness); push!(rows.flatness, g.flatness)
        push!(rows.transverse_flatness, g.transverse_flatness)
        push!(rows.gradient_variance_ratio, g.transverse_variance / g.longitudinal_variance)
        push!(rows.skewness_fd, gf.skewness); push!(rows.flatness_fd, gf.flatness)
        push!(rows.transverse_flatness_fd, gf.transverse_flatness); push!(rows.dissipation_fd, fd.dissipation)
        push!(rows.relative_divergence_fd, fd.rms_divergence / fd.rms_gradient)
    else
        for name in (:skewness, :flatness, :transverse_flatness, :gradient_variance_ratio, :skewness_fd,
                     :flatness_fd, :transverse_flatness_fd, :dissipation_fd, :relative_divergence_fd)
            push!(getfield(rows, name), NaN)
        end
    end
    for j in eachindex(spectrum.k)
        push!(spectra.time, t); push!(spectra.k, spectrum.k[j]); push!(spectra.energy, spectrum.energy[j])
        push!(spectra.dissipation, spectrum.dissipation[j])
    end
    if snapshot_prefix !== nothing
        Fc = collocated(F, staggering_offsets(L / M))
        save_fourier_velocity(@sprintf("%s_n%02d.nc", snapshot_prefix, n), Fc, t)
    end
    slow && @info @sprintf("t = %.4f s (iter %d, Δt = %.2e): K = %.5e, ε = %.5e (FD %.5e), Re_λ = %.2f, S = %.4f, F = %.3f, F_T = %.3f",
                           t, model.clock.iteration, simulation.Δt, s.kinetic_energy, s.dissipation, rows.dissipation_fd[end],
                           s.taylor_reynolds, rows.skewness[end], rows.flatness[end], rows.transverse_flatness[end])
    return nothing
end

function run_case(z, mode, M, advection_name; arch, cfl=0.3, stop_time, output, output_dir, label, slow_every=5,
                  snapshots=true, FT=Float64)
    advection = Dict("centered2" => Centered(order=2), "centered4" => Centered(order=4),
                     "centered6" => Centered(order=6), "centered8" => Centered(order=8))[advection_name]
    name = @sprintf("%s_M%d_%s_%s", label, M, advection_name, FT)
    p = z.parameters
    L = p.domain_length
    ν = p.viscosity
    @info "Resolution study run" name mode M advection_name L ν
    halo = advection_name == "centered8" ? 4 : 3
    grid = RectilinearGrid(arch, FT; size=(M, M, M), halo=(halo, halo, halo), extent=(L, L, L),
                           topology=(Periodic, Periodic, Periodic))
    forcing = mode == "forced" ? ConstantPowerBandForcing(grid; energy_injection_rate=ε_injection) : nothing
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    dynamics = BoussinesqDynamics(grid, constants; pressure=90220, temperature=281.2)
    kw = forcing === nothing ? (;) : (; forcing=(u=forcing, v=forcing, w=forcing))
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, advection,
                            closure=ScalarDiffusivity(FT; ν, κ=2.22e-5), kw...)
    u₀, v₀, w₀ = staggered_velocities(z, M)
    θᵣ = model.dynamics.reference_state.potential_temperature
    set!(model; θ=θᵣ, u=u₀, v=v₀, w=w₀)

    snapshot_prefix = nothing
    if mode == "decay" && snapshots
        mkpath(joinpath(output_dir, "snapshots"))
        snapshot_prefix = joinpath(output_dir, "snapshots", name)
    end
    state = (rows=new_rows(), spectra=(time=Float64[], k=Int[], energy=Float64[], dissipation=Float64[]),
             forcing, L, ν, M, slow_every, snapshot_prefix, mode)

    Δ = L / M
    umax₀ = max(maximum(abs, u₀), maximum(abs, v₀), maximum(abs, w₀))
    simulation = Simulation(model; Δt=cfl * Δ / umax₀, stop_time, verbose=false)
    record!(state, simulation)
    wizard = TimeStepWizard(; cfl, max_change=1.1, diffusive_cfl=0.2)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))
    schedule = mode == "decay" ? SpecifiedTimes(output...) : TimeInterval(output)
    simulation.callbacks[:diagnostics] = Callback(sim -> record!(state, sim), schedule)

    GC.gc(true); arch isa GPU && CUDA.reclaim()
    memory = arch isa GPU ? CUDA.used_memory() : 0
    wall = time_ns()
    run!(simulation)
    elapsed = (time_ns() - wall) / 1e9

    # GPU cost without diagnostics: 200 steps at the final Δt (after the recorded run, so results are unaffected)
    Δt = simulation.Δt
    for _ in 1:10; time_step!(model, Δt); end
    arch isa GPU && CUDA.synchronize()
    t₀ = time_ns()
    for _ in 1:200; time_step!(model, Δt); end
    arch isa GPU && CUDA.synchronize()
    ms_per_step = (time_ns() - t₀) / 1e6 / 200
    iterations = state.rows.iteration[end]
    @info @sprintf("TIMING %s: %d iterations to t = %.4f s, wall %.1f s incl. diagnostics; %.3f ms/step without diagnostics -> %.1f s GPU time to solution",
                   name, iterations, state.rows.time[end], elapsed, ms_per_step, ms_per_step * iterations / 1e3)
    open(joinpath(output_dir, "timing.csv"), "a") do io
        println(io, join((name, mode, M, advection_name, iterations, state.rows.time[end], elapsed, ms_per_step,
                          ms_per_step * iterations / 1e3, FT, memory), ","))
    end
    write_csv(joinpath(output_dir, "breeze_$(name)_timeseries.csv"), state.rows)
    write_csv(joinpath(output_dir, "breeze_$(name)_spectra.csv"), state.spectra)
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = ARGS[1]
    @assert mode in ("decay", "forced")
    zk_file = ARGS[2]
    runs = [(parse(Int, a), String(b)) for (a, b) in split.(split(ARGS[3], ','), ':')]
    stop_time = parse(Float64, ARGS[4])
    output = mode == "decay" ? parse.(Float64, filter(!isempty, strip.(readlines(ARGS[5])))) : parse(Float64, ARGS[5])
    arch = CUDA.functional() ? GPU() : CPU()
    cfl = parse(Float64, get(ENV, "CFL", "0.3"))
    output_dir = get(ENV, "OUTPUT_DIR", joinpath(@__DIR__, "output", "resolution"))
    label = get(ENV, "LABEL", "R")
    slow_every = parse(Int, get(ENV, "SLOW_EVERY", "5"))
    snapshots = get(ENV, "SNAPSHOTS", "1") == "1"
    box_N = parse(Int, get(ENV, "BOX_N", "32"))
    FT = get(ENV, "FLOAT", "Float64") == "Float32" ? Float32 : Float64
    mkpath(output_dir)
    z = read_zk(zk_file; box_N)
    @info "Initial condition" zk_file z z.parameters
    for (M, advection_name) in runs
        run_case(z, mode, M, advection_name; arch, cfl, stop_time, output, output_dir, label, slow_every, snapshots, FT)
    end
end
