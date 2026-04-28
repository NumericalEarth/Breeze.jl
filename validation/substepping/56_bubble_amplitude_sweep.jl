#####
##### validation/substepping/56_bubble_amplitude_sweep.jl
#####
##### Sweep bubble IC amplitude Δθ to determine whether the substepper bug
##### (vs anelastic and explicit-compressible ground truth) is amplitude-
##### dependent. If ratio max|w|_substep / max|w|_explicit is constant
##### across Δθ → structural linear-regime bug. If ratio → 1 as Δθ → 0 →
##### nonlinear / amplitude-dependent bug.
#####
##### Setup: 64×64 stratified atmosphere with N²=1e-4, no background flow.
##### Run anelastic, explicit-compressible, and substepped (ω=0.55) for 300s.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Statistics
using Printf
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()

const CASE   = "bubble_amplitude_sweep"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const Lx = 20e3
const Lz = 10e3
const Nx = 64
const Nz = 64
const θ₀_ref = 300.0
const N²     = 1e-4
const r₀     = 2e3
const g_phys = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g_phys)

const STOP_T = 300.0
const Δt_anel = 1.0
const Δt_subst = 1.0
const Δt_expl = 0.05

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

function θᵢ_builder(grid, Δθ)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
    function θᵢ(x, z)
        r² = (x - x₀)^2 + (z - z₀)^2
        return θᵇᵍ(z) + Δθ * exp(-r² / r₀^2)
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

function run_one(label, builder, Δθ; Δt, stop_time = STOP_T)
    grid = build_grid()
    model = builder(grid)
    if label == "anelastic"
        set!(model; θ = θᵢ_builder(grid, Δθ))
    else
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ_builder(grid, Δθ), ρ = ref.density)
    end

    sim = Simulation(model; Δt, stop_time, verbose = false)

    times = Float64[]; wmax_log = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        push!(times, Float64(sim.model.clock.time))
        push!(wmax_log, wmax)
    end
    add_callback!(sim, _track, IterationInterval(50))

    t0 = time(); status = :ok; err = ""
    try; run!(sim); catch e; status = :crashed; err = sprint(showerror, e); end
    elapsed = time() - t0

    w = model.velocities.w
    wmax_final = Float64(maximum(abs, interior(w)))
    has_nan = any(isnan, parent(w))
    return (; label, Δθ, t = Float64(model.clock.time), wmax_final, has_nan,
              elapsed, status, err, times, wmax_log)
end

results = NamedTuple[]
for Δθ in (1.0, 0.1, 0.01, 0.001, 0.0001)
    @info "================ Δθ = $Δθ K ================"
    @info "    --- anelastic ---"
    push!(results, run_one("anelastic_$(Δθ)", build_anelastic_model, Δθ; Δt = Δt_anel))
    @info "    --- explicit ---"
    push!(results, run_one("explicit_$(Δθ)", build_explicit_model, Δθ; Δt = Δt_expl))
    @info "    --- substepped ---"
    push!(results, run_one("substepped_$(Δθ)", build_substepped_model, Δθ; Δt = Δt_subst))
end

@info "=== SUMMARY ==="
@info @sprintf("  %-10s %-7s %-12s %s", "label", "Δθ", "max|w|", "status")
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-22s Δθ=%s  max|w|=%.6f  (%5.1fs)",
                   mark, r.label, repr(r.Δθ), r.wmax_final, r.elapsed)
end

@info "=== RATIO TABLE (substepped / explicit, scaled by Δθ⁻¹) ==="
for Δθ in (1.0, 0.1, 0.01, 0.001, 0.0001)
    anel = filter(r -> r.label == "anelastic_$(Δθ)" && !r.has_nan, results)
    expl = filter(r -> r.label == "explicit_$(Δθ)" && !r.has_nan, results)
    sub  = filter(r -> r.label == "substepped_$(Δθ)" && !r.has_nan, results)
    if length(anel) == 1 && length(expl) == 1 && length(sub) == 1
        ratio_se = sub[1].wmax_final / expl[1].wmax_final
        ratio_sa = sub[1].wmax_final / anel[1].wmax_final
        ratio_ea = expl[1].wmax_final / anel[1].wmax_final
        @info @sprintf("  Δθ=%s: anel=%.4e expl=%.4e sub=%.4e | sub/expl=%.3f sub/anel=%.3f expl/anel=%.3f",
                       repr(Δθ), anel[1].wmax_final, expl[1].wmax_final, sub[1].wmax_final,
                       ratio_se, ratio_sa, ratio_ea)
    end
end

jldsave(joinpath(OUTDIR, "summary.jld2"); results)
