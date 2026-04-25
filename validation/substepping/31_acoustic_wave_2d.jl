#####
##### validation/substepping/31_acoustic_wave_2d.jl
#####
##### 2D acoustic wave tests with g = 0, centered CN, no damping.
##### Two complementary ICs:
#####
##### A. HORIZONTAL plane wave: (ρθ)′ varies only in x (∂z(ρθ)′ = 0).
#####    Tests the horizontal PGF + horizontal mass continuity. Vertical
#####    motion stays at zero, the wave propagates left/right at c_s.
#####
##### B. VERTICAL plane wave: (ρθ)′ varies only in z (∂x(ρθ)′ = 0).
#####    Tests the vertical implicit Schur tridiagonal. Horizontal motion
#####    stays at zero, the wave propagates up/down at c_s.
#####
##### A clean substepper should give wmax ≈ 0 in test A and umax ≈ 0 in test B,
##### with the resolved-direction velocity matching ρθ′·c_s/(ρ_r θ_r) ≈ 1e-3 m/s
##### for our 0.01 K perturbation.
#####
##### If A passes but B fails (or vice versa), the bug is direction-specific.
##### If both pass, the 2D acoustic core is fully verified — buoyancy testing next.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Statistics
using Printf
using JLD2

CUDA.functional() || error("GPU required")
const arch = GPU()

const CASE = "acoustic_wave_2d"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 60.0
const Δt     = 1.0
const θ₀    = 300.0
const Δθ    = 0.01

# Domain: 20 km wide x 10 km tall.
build_grid() = RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                               x = (-10e3, 10e3), z = (0, 10e3),
                               topology = (Periodic, Flat, Bounded))

# Test A: horizontal plane wave. θ varies cos-like in x, uniform in z.
const k_horiz = 2π / 20e3      # one wavelength = 20 km
θᵢ_horiz(x, z) = θ₀ + Δθ * cos(k_horiz * x)

# Test B: vertical plane wave. θ varies cos-like in z, uniform in x.
const k_vert  = 2π / 10e3      # one wavelength = 10 km
θᵢ_vert(x, z) = θ₀ + Δθ * cos(k_vert * (z - 5e3))

function build_model(; ω, damping)
    grid = build_grid()
    constants = ThermodynamicConstants(eltype(grid); gravitational_acceleration = 0.0)
    td  = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = ω, damping = damping)
    dyn = CompressibleDynamics(td; reference_potential_temperature = z -> θ₀)
    return AtmosphereModel(grid; dynamics = dyn, advection = Centered(order = 2),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label, θᵢ; ω = 0.5, damping = NoDivergenceDamping())
    model = build_model(; ω, damping)
    set!(model; θ = θᵢ, ρ = model.dynamics.reference_state.density)

    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    function _progress(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        umax = Float64(maximum(abs, interior(sim.model.velocities.u)))
        @info @sprintf("[%s] iter=%4d t=%5.1fs umax=%.3g wmax=%.3g",
                       label, iteration(sim), sim.model.clock.time, umax, wmax)
    end
    add_callback!(sim, _progress, IterationInterval(15))

    sim.output_writers[:jld2] = JLD2Writer(model,
        (; w = model.velocities.w, u = model.velocities.u);
        filename = joinpath(OUTDIR, "$(label).jld2"),
        schedule = TimeInterval(2.0),
        overwrite_existing = true)

    t0 = time()
    try
        run!(sim)
    catch e
        @warn "$label CRASH" exception = e
    end
    elapsed = time() - t0
    return (; label,
            t = sim.model.clock.time,
            umax = Float64(maximum(abs, interior(sim.model.velocities.u))),
            wmax = Float64(maximum(abs, interior(sim.model.velocities.w))),
            elapsed)
end

results = NamedTuple[]
@info "=== A: horizontal plane wave (ω=0.5, no damping) ==="
push!(results, run_one("A_horizontal", θᵢ_horiz; ω = 0.5, damping = NoDivergenceDamping()))

@info "=== B: vertical plane wave (ω=0.5, no damping) ==="
push!(results, run_one("B_vertical",   θᵢ_vert;  ω = 0.5, damping = NoDivergenceDamping()))

# Sanity: also try ω=0.8 with some damping (the operational config).
@info "=== A: horizontal plane wave (ω=0.8, β=0.1) ==="
push!(results, run_one("A_horizontal_op", θᵢ_horiz; ω = 0.8,
                       damping = PressureProjectionDamping(coefficient = 0.1)))

@info "=== B: vertical plane wave (ω=0.8, β=0.1) ==="
push!(results, run_one("B_vertical_op",   θᵢ_vert;  ω = 0.8,
                       damping = PressureProjectionDamping(coefficient = 0.1)))

@info "=== SUMMARY ==="
for r in results
    @info @sprintf("  %-22s t=%5.1f umax=%.3g wmax=%.3g  %4.1fs",
                   r.label, r.t, r.umax, r.wmax, r.elapsed)
end
jldsave(joinpath(OUTDIR, "summary.jld2"); results)
