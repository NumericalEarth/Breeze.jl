"""
ABCompare — shared utilities for comparing anelastic vs compressible-substepper
example runs. Each example provides two build functions: `build_anelastic(grid)`
and `build_compressible(grid)`. The harness runs both, captures wall-clock,
peak diagnostics, NaN status, and writes a markdown row.
"""
module ABCompare

using Oceananigans
using Oceananigans.Units
using Printf
using Statistics

export run_pair, peak_tracker, write_row, REPORT_PATH

const REPORT_PATH = joinpath(@__DIR__, "REPORT.md")

# Tracker mutable struct: peak |w|, peak |u|, NaN status, wall-clock.
mutable struct RunResult
    label::String
    Δt::Float64
    stop_time::Float64
    iterations::Int
    elapsed_s::Float64        # total wall-clock
    sim_per_real::Float64      # simulated seconds per real second
    wmax_overall::Float64
    umax_overall::Float64
    wmax_final::Float64
    has_nan::Bool
    extra::NamedTuple
end

"""
Run a model to `stop_time` with `Δt`, tracking max|w|, max|u| at the
given iteration interval. Returns a RunResult.
"""
function run_one(label, model; Δt, stop_time, callback_iters = 50, extra_track = nothing)
    sim = Simulation(model; Δt, stop_time, verbose = false)
    wmax_overall = Ref(0.0)
    umax_overall = Ref(0.0)
    extra_acc = Ref{Any}(nothing)

    function _track(sim)
        u = sim.model.velocities.u
        w = sim.model.velocities.w
        wmax = Float64(maximum(abs, interior(w)))
        umax = Float64(maximum(abs, interior(u)))
        wmax_overall[] = max(wmax_overall[], wmax)
        umax_overall[] = max(umax_overall[], umax)
        if extra_track !== nothing
            extra_acc[] = extra_track(sim, extra_acc[])
        end
        return nothing
    end
    add_callback!(sim, _track, IterationInterval(callback_iters))

    t0 = time()
    crashed = false
    try
        run!(sim)
    catch err
        crashed = true
        @info "  $(label) CRASHED: $(sprint(showerror, err))"
    end
    elapsed = time() - t0

    w = model.velocities.w
    has_nan = any(isnan, parent(w)) || crashed
    wmax_final = has_nan ? NaN : Float64(maximum(abs, interior(w)))

    raw_time = sim.model.clock.time
    sim_time = raw_time isa Number ? Float64(raw_time) : Float64(stop_time)
    sim_per_real = sim_time / max(elapsed, 1e-9)
    iters = sim.model.clock.iteration

    extra_nt = extra_acc[] === nothing ? NamedTuple() : (; extra = extra_acc[])

    @info @sprintf("  %-22s | Δt=%-5.2f t_end=%6.1fs iters=%5d wall=%6.2fs (%.1fx realtime)  wmax=%.3e umax=%.3e %s",
                   label, Δt, sim_time, iters, elapsed, sim_per_real,
                   wmax_overall[], umax_overall[], has_nan ? "NaN" : "✓")

    return RunResult(label, Δt, stop_time, iters, elapsed, sim_per_real,
                     wmax_overall[], umax_overall[], wmax_final, has_nan, extra_nt)
end

"""
Run both anelastic and compressible models built by builder functions, return
both RunResults plus a summary string for the report.
"""
function run_pair(name::String;
                  build_anelastic, build_compressible,
                  Δt_anel, Δt_comp, stop_time,
                  callback_iters = 50,
                  extra_track = nothing,
                  notes = "")
    @info "=========== $name ==========="
    anel = run_one("anelastic", build_anelastic(); Δt = Δt_anel, stop_time, callback_iters, extra_track)
    comp = run_one("compressible", build_compressible(); Δt = Δt_comp, stop_time, callback_iters, extra_track)
    return (; name, anel, comp, notes)
end

"Format a markdown row for REPORT.md."
function row(result)
    a, c = result.anel, result.comp
    stab_a = a.has_nan ? "✗" : "✓"
    stab_c = c.has_nan ? "✗" : "✓"
    ratio  = (a.wmax_overall == 0 || a.has_nan || c.has_nan) ? "—" :
             @sprintf("%.3f", c.wmax_overall / a.wmax_overall)
    speedup = (a.elapsed_s == 0 || c.elapsed_s == 0) ? "—" :
              @sprintf("%.2fx", a.elapsed_s / c.elapsed_s)
    return @sprintf("| %s | %s | %s | Δt=%.2f→%.2f | %s | %s | %.1fs / %.1fs | %s |",
                    result.name, stab_a, stab_c,
                    a.Δt, c.Δt, ratio, speedup, a.elapsed_s, c.elapsed_s, result.notes)
end

function header()
    return """
## Anelastic vs Compressible-substepper comparison

| Example | Anel stable | Comp stable | Δt anel→comp | wmax ratio (comp/anel) | Speedup (anel/comp) | Wall-clock (anel / comp) | Notes |
|---------|-------------|-------------|---------------|-------------------------|----------------------|---------------------------|-------|
"""
end

function write_row(result; append = true)
    if !isfile(REPORT_PATH) || !append
        open(REPORT_PATH, "w") do io
            write(io, header())
        end
    end
    open(REPORT_PATH, "a") do io
        write(io, row(result), "\n")
    end
    @info "Wrote row for '$(result.name)' to $REPORT_PATH"
    return nothing
end

end # module
