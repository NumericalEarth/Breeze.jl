#####
##### validation/substepping/07_dry_thermal_bubble_wizard.jl
#####
##### Substepping version of examples/dry_thermal_bubble.jl, redone with an
##### adaptive time step (`conjure_time_step_wizard!`, cfl = 0.3) so the outer
##### step tracks the advective CFL as the bubble accelerates past 50 m/s. The
##### stable recipe that emerged from the sweep in §3.A of REPORT.md is:
#####
#####   damping        = PressureProjectionDamping(coefficient = 0.5)
#####   forward_weight = 0.8
#####   cfl (wizard)   = 0.3
#####
##### A brief note on the constants:
#####
#####   - `PressureProjectionDamping(0.5)` is the literal ERF/CM1/WRF form tuned
#####     on the DCMIP2016 BCI (docs/src/appendix/bw_dt_sweep_results.md). The
#####     0.5 coefficient suppresses the 2Δx acoustic mode aggressively — needed
#####     here because Δθ = 10 K in a weakly stratified background produces a
#####     strong pressure response.
#####   - `forward_weight = 0.8` (MPAS ε = 2ω − 1 = 0.6) adds enough implicit
#####     off-centering to the w–ρθ column solve to kill the vertical acoustic
#####     modes the projection damping doesn't catch. The default 0.6 is not
#####     enough here.
#####   - `cfl = 0.3` in the wizard is about half the ARW/MPAS empirical CFL ≈
#####     0.7 at which the Wicker–Skamarock RK3 outer loop is said to be stable.
#####     Here we need the extra headroom because the bubble briefly hits
#####     |w| ≈ 55 m/s over Δz = 78 m (advective CFL ≈ 1.4 if we ran cfl = 0.7).
#####
##### We run both the anelastic baseline and the compressible split-explicit
##### variant with the same outer wizard and same initial state, then save a
##### side-by-side `w` snapshot for visual comparison.
#####
##### Deviation from the docs example: the docs version uses
##### `formulation = :StaticEnergy`. The current MPAS substepper is wired to
##### recover ρθ (LiquidIcePotentialTemperatureFormulation). We run both
##### dynamics with the default LiquidIce formulation so the comparison
##### isolates the dynamics choice.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using Statistics
using Printf
using JLD2

const CASE = "dry_thermal_bubble_wizard"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 25minutes

# Physical setup — mirrors examples/dry_thermal_bubble.jl.
const θ₀_ref = 300.0
const N²     = 1e-6
const r₀     = 2e3
const Δθ     = 10.0
const g      = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g)

function build_grid()
    RectilinearGrid(CPU(); size = (128, 128), halo = (5, 5),
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

function build_compressible_model(grid)
    # Best known-stable config for the dry thermal bubble stress case:
    # - `PressureProjectionDamping(0.5)` — Breeze default (literal ERF/CM1/WRF form at
    #   the BCI-tuned value; aggressive enough to suppress the 2Δx acoustic mode
    #   driven by the Δθ=10K bubble in a weakly-stratified background).
    # - `forward_weight = 0.8` (ε = 2ω − 1 = 0.6) — explicit override of the Breeze
    #   default ω = 0.7. The bubble develops max|w| ≈ 50 m/s via its ring vortex
    #   and the additional vertical-implicit off-centering is needed to bound the
    #   vertical acoustic mode at that amplitude. See validation/substepping/NOTES.md
    #   ("Root cause isolated") for the open investigation of why Breeze needs more
    #   off-centering than ERF/MPAS canonical here.
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(forward_weight = 0.8)
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

const CFL_WIZARD = 0.3

function run_case(label, builder)
    grid = build_grid()
    model = builder(grid)
    θᵢ = θᵢ_builder(grid)
    if label == "anelastic"
        set!(model; θ = θᵢ)
    else
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ, ρ = ref.density)
    end

    sim = Simulation(model; Δt = 0.5, stop_time = STOP_T, verbose = false)
    conjure_time_step_wizard!(sim; cfl = CFL_WIZARD)

    progress_counter = Ref(0)
    function _progress(sim)
        progress_counter[] += 1
        if progress_counter[] % 5 == 0
            @info @sprintf("[%s] iter=%5d t=%7.1fs Δt=%.3fs max|w|=%.2f",
                           label, iteration(sim), sim.model.clock.time, sim.Δt,
                           maximum(abs, interior(sim.model.velocities.w)))
        end
    end
    add_callback!(sim, _progress, IterationInterval(100))

    # Save w (velocity), θ (potential temperature diagnostic), and ρ for the
    # compressible case (prognostic; anelastic's ρ ≡ ρ_ref(z) doesn't vary and
    # isn't needed). ρw is reconstructed from ρ and w post-hoc.
    if label == "anelastic"
        outputs = (; w = model.velocities.w,
                     θ = PotentialTemperature(model))
    else
        outputs = (; w = model.velocities.w,
                     θ = PotentialTemperature(model),
                     ρ = dynamics_density(model.dynamics))
    end
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(10seconds),
                                           overwrite_existing = true)

    res = timed_run!(sim; label)
    return summarize_result(label, res, model)
end

@info "[$CASE] Anelastic run…"
a = run_case("anelastic", build_anelastic_model)
@info "[$CASE] Compressible run…"
c = run_case("compressible", build_compressible_model)

wa_ts = try; FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"), "w"); catch; nothing; end
wc_ts = try; FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"), "w"); catch; nothing; end

try
    if wa_ts !== nothing && wc_ts !== nothing
        wa = wa_ts[end]; wc = wc_ts[end]
        two_column_figure(joinpath(OUTDIR, "summary.png"), wa, wc;
                          title_a = "anelastic w (t = $(round(Int, wa_ts.times[end]))s)",
                          title_b = "compressible w (t = $(round(Int, wc_ts.times[end]))s)",
                          label = "w (m/s)")
        @info "wrote summary.png"

        # Also: time series of max|w| side by side
        ta = wa_ts.times; tc = wc_ts.times
        wa_peak = [maximum(abs, interior(wa_ts[i])) for i in 1:length(ta)]
        wc_peak = [maximum(abs, interior(wc_ts[i])) for i in 1:length(tc)]
        fig = Figure(size = (900, 400))
        ax = Axis(fig[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
                  title = "Bubble peak w: anelastic vs compressible")
        lines!(ax, ta, wa_peak; label = "anelastic", linewidth = 2)
        lines!(ax, tc, wc_peak; label = "compressible", linewidth = 2, linestyle = :dash)
        axislegend(ax, position = :rb)
        save(joinpath(OUTDIR, "peak_w.png"), fig)
    end
catch e
    @warn "plot failed" exception = e
end

jldsave(joinpath(OUTDIR, "result.jld2"); anelastic = a, compressible = c, case = CASE, cfl = CFL_WIZARD, stop_time = STOP_T)

io = IOBuffer()
report_case(io, CASE,
            "2D dry thermal bubble, 128×128, adaptive Δt via wizard (cfl=$(CFL_WIZARD) compressible, 0.3 anelastic for apples-to-apples), stop=$(Int(STOP_T))s, CPU, WENO(9). Compressible uses PressureProjectionDamping(0.5) + forward_weight=0.8.",
            a, c)
write(joinpath(OUTDIR, "report.md"), take!(io))
@info "[$CASE] done"
