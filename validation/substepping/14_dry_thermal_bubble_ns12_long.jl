#####
##### validation/substepping/14_dry_thermal_bubble_ns12_long.jl
#####
##### Long (25-minute) dry thermal bubble with the substepper at FIXED Ns=12,
##### compared against the anelastic baseline and the fully-explicit
##### compressible ground truth. The visualization step is in this same
##### script so we pay the Julia load + JIT cost only once.
#####
##### Output:
#####   out/dry_thermal_bubble_ns12_long/anelastic.jld2
#####   out/dry_thermal_bubble_ns12_long/explicit.jld2
#####   out/dry_thermal_bubble_ns12_long/substepped.jld2
#####   out/dry_thermal_bubble_ns12_long/bubble_three_way.{mp4,gif}
#####   out/dry_thermal_bubble_ns12_long/peak_w.png
#####
##### Setup follows 07/08 (128×128 GPU, x ∈ [-10,10] km, z ∈ [0,10] km,
##### WENO(9), Δθ=10K bubble at z=3km, N²=1e-6 background).
#####
##### Substepper: substeps=12 (FIXED), forward_weight=0.8,
##### PressureProjectionDamping(0.5). Outer Δt: time-step wizard at cfl=0.3
##### so the substep acoustic CFL stays bounded as |w| evolves through the
##### bubble lifecycle.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using CairoMakie
using Statistics
using Printf
using JLD2

CUDA.functional() || error("GPU required")
const arch = GPU()

const CASE = "dry_thermal_bubble_ns12_long"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 25minutes
const CFL_WIZARD = 0.3

# Same physical setup as 07/08.
const θ₀_ref = 300.0
const N²     = 1e-6
const r₀     = 2e3
const Δθ     = 10.0
const g_phys = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g_phys)

function build_grid()
    RectilinearGrid(arch; size = (128, 128), halo = (5, 5),
                    x = (-10e3, 10e3), z = (0, 10e3),
                    topology = (Periodic, Flat, Bounded))
end

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
    function θᵢ(x, z)
        r = sqrt((x - x₀)^2 + (z - z₀)^2)
        return θᵇᵍ(z) + Δθ * max(0, 1 - r / r₀)
    end
end

function build_anelastic_model(grid)
    constants = ThermodynamicConstants(eltype(grid))
    reference_state = ReferenceState(grid, constants; potential_temperature = θᵇᵍ)
    dynamics = AnelasticDynamics(reference_state)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9))
end

function build_explicit_model(grid)
    constants = ThermodynamicConstants(eltype(grid))
    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants)
end

function build_substepped_model(grid)
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(substeps = 12,
                                         forward_weight = 0.55,
                                         damping = NoDivergenceDamping())
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder; Δt_init, use_wizard, sample_every = 5seconds)
    grid = build_grid()
    model = builder(grid)
    if label == "anelastic"
        set!(model; θ = θᵢ_builder(grid))
    else
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ_builder(grid), ρ = ref.density)
    end

    sim = Simulation(model; Δt = Δt_init, stop_time = STOP_T, verbose = false)
    if use_wizard
        conjure_time_step_wizard!(sim; cfl = CFL_WIZARD)
    end

    counter = Ref(0)
    function _progress(sim)
        counter[] += 1
        if counter[] % 5 == 0
            @info @sprintf("[%s] iter=%6d t=%7.1fs Δt=%.3fs max|w|=%.2f max|u|=%.2f",
                           label, iteration(sim), sim.model.clock.time, sim.Δt,
                           Float64(maximum(abs, interior(sim.model.velocities.w))),
                           Float64(maximum(abs, interior(sim.model.velocities.u))))
        end
    end
    progress_iter = label == "explicit" ? 500 : 50
    add_callback!(sim, _progress, IterationInterval(progress_iter))

    if label == "anelastic"
        outputs = (; w = model.velocities.w,
                     u = model.velocities.u,
                     θ = PotentialTemperature(model))
    else
        outputs = (; w = model.velocities.w,
                     u = model.velocities.u,
                     θ = PotentialTemperature(model),
                     ρ = dynamics_density(model.dynamics))
    end
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(sample_every),
                                           overwrite_existing = true)

    res = timed_run!(sim; label)
    return summarize_result(label, res, model)
end

@info "[$CASE] Anelastic run (wizard cfl=$(CFL_WIZARD))…"
a = run_one("anelastic",  build_anelastic_model;  Δt_init = 0.5, use_wizard = true)
@info "[$CASE] Substepped run (Ns=12 fixed, wizard cfl=$(CFL_WIZARD))…"
s = run_one("substepped", build_substepped_model; Δt_init = 0.5, use_wizard = true)
@info "[$CASE] Explicit-compressible run (Δt=0.1 fixed)…"
e = run_one("explicit",   build_explicit_model;   Δt_init = 0.1, use_wizard = false)

@info "[$CASE] === SUMMARY ==="
for r in (a, s, e)
    mark = r.has_nan ? "NaN" : (r.ok ? "✓" : "✗")
    @info @sprintf("  %3s %-12s t=%7.1f wmax=%.3g umax=%.3g  %.1fs",
                   mark, r.label, r.time_sim, r.wmax, r.umax, r.elapsed)
end

jldsave(joinpath(OUTDIR, "result.jld2");
        anelastic = a, substepped = s, explicit = e,
        case = CASE, stop_time = STOP_T, cfl = CFL_WIZARD)

io = IOBuffer()
report_case(io, CASE,
            "Long-run dry thermal bubble (25 min), 128×128 GPU, WENO(9). " *
            "Substepper: Ns=12 fixed, ω=0.8, PressureProjectionDamping(0.5), wizard cfl=$(CFL_WIZARD). " *
            "Explicit: Δt=0.1s. Anelastic: wizard cfl=$(CFL_WIZARD).",
            a, s)
println(io, "\n### Explicit-compressible (ground truth)")
println(io, "Δt = 0.1s, ", e.ok ? "ran to t=$(round(Int, e.time_sim))s" : "crashed: $(e.error)")
@printf(io, "elapsed=%.1fs  wmax=%.3g  umax=%.3g\n", e.elapsed, e.wmax, e.umax)
write(joinpath(OUTDIR, "report.md"), take!(io))

#####
##### Visualization: 3-panel mp4/gif from the JLD2 outputs.
##### Wrapped in try/catch so a substepped NaN doesn't kill the whole script.
#####

try
    wa_ts = FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"),  "w")
    we_ts = FieldTimeSeries(joinpath(OUTDIR, "explicit.jld2"),   "w")
    ws_ts = FieldTimeSeries(joinpath(OUTDIR, "substepped.jld2"), "w")

    ta, te, tsub = wa_ts.times, we_ts.times, ws_ts.times
    tmax = min(ta[end], te[end], tsub[end])
    mask_a = findall(t -> t ≤ tmax + 1e-6, ta)
    mask_e = findall(t -> t ≤ tmax + 1e-6, te)
    mask_s = findall(t -> t ≤ tmax + 1e-6, tsub)
    Nt = min(length(mask_a), length(mask_e), length(mask_s))
    @info "[$CASE] building 3-panel animation" Nt tmax

    grid_anim = wa_ts.grid
    x_km = collect(xnodes(grid_anim, Center())) ./ 1e3
    z_km = collect(znodes(grid_anim, Face()))   ./ 1e3

    # Skip any NaN frames in the substepped run (post-NaN snapshots are NaN).
    function _safe_max_abs(field)
        a = Array(interior(field))
        m = 0.0
        @inbounds for x in a
            isfinite(x) && (m = max(m, abs(Float64(x))))
        end
        return m
    end
    vmax = let v = 0.0
        for i in 1:Nt
            v = max(v,
                    _safe_max_abs(wa_ts[mask_a[i]]),
                    _safe_max_abs(we_ts[mask_e[i]]),
                    _safe_max_abs(ws_ts[mask_s[i]]))
        end
        v > 0 ? v : 1.0
    end
    @info "[$CASE] shared color range" vmax

    n = Observable(1)
    function _slice_safe(ts, idx)
        a = Array(interior(ts[idx]))[:, 1, :]
        replace!(x -> isfinite(x) ? x : 0.0, a)
        return a
    end
    wa_slice = @lift _slice_safe(wa_ts, mask_a[$n])
    we_slice = @lift _slice_safe(we_ts, mask_e[$n])
    ws_slice = @lift _slice_safe(ws_ts, mask_s[$n])

    fig = Figure(size = (1800, 560), fontsize = 15)
    title_node = @lift @sprintf("Dry thermal bubble (Ns=12 long run) — t = %5.1f s",
                                ta[mask_a[$n]])
    fig[0, 1:4] = Label(fig, title_node, fontsize = 20, tellwidth = false)

    ax_a = Axis(fig[1, 1]; title = "Anelastic",
                xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
    ax_e = Axis(fig[1, 2]; title = "Compressible explicit (Δt = 0.1 s)",
                xlabel = "x (km)", aspect = DataAspect())
    ax_s = Axis(fig[1, 3]; title = "Compressible substepper (Ns = 12)",
                xlabel = "x (km)", aspect = DataAspect())

    hm = heatmap!(ax_a, x_km, z_km, wa_slice;
                  colormap = :balance, colorrange = (-vmax, vmax))
    heatmap!(     ax_e, x_km, z_km, we_slice;
                  colormap = :balance, colorrange = (-vmax, vmax))
    heatmap!(     ax_s, x_km, z_km, ws_slice;
                  colormap = :balance, colorrange = (-vmax, vmax))
    Colorbar(fig[1, 4], hm; label = "w (m/s)")

    mp4 = joinpath(OUTDIR, "bubble_three_way.mp4")
    record(fig, mp4, 1:Nt; framerate = 15) do i; n[] = i; end
    @info "[$CASE] wrote $mp4"

    gif = joinpath(OUTDIR, "bubble_three_way.gif")
    record(fig, gif, 1:Nt; framerate = 15) do i; n[] = i; end
    @info "[$CASE] wrote $gif"

    # Time series of max|w| (NaN-safe).
    wa_peak = [_safe_max_abs(wa_ts[mask_a[i]]) for i in 1:Nt]
    we_peak = [_safe_max_abs(we_ts[mask_e[i]]) for i in 1:Nt]
    ws_peak = [_safe_max_abs(ws_ts[mask_s[i]]) for i in 1:Nt]
    fig2 = Figure(size = (900, 400))
    ax2 = Axis(fig2[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
               title = "Bubble peak |w|: anelastic vs explicit vs Ns=12 substepper")
    lines!(ax2, ta[mask_a[1:Nt]], wa_peak; label = "anelastic",        linewidth = 2)
    lines!(ax2, te[mask_e[1:Nt]], we_peak; label = "explicit",         linewidth = 2, linestyle = :dash)
    lines!(ax2, tsub[mask_s[1:Nt]], ws_peak; label = "Ns=12 substepper", linewidth = 2, linestyle = :dot)
    axislegend(ax2, position = :rb)
    save(joinpath(OUTDIR, "peak_w.png"), fig2)
    @info "[$CASE] wrote peak_w.png"
catch err
    @warn "[$CASE] animation failed" exception = err
end

@info "[$CASE] done"
