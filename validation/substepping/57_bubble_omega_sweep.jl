#####
##### validation/substepping/57_bubble_omega_sweep.jl
#####
##### For the small bubble (Δθ=0.001, linear regime), sweep ω to see
##### how the sub/explicit ratio depends on off-centering.
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

const CASE   = "bubble_omega_sweep"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const Lx = 20e3
const Lz = 10e3
const Nx = 64
const Nz = 64
const θ₀_ref = 300.0
const N²     = 1e-4
const r₀     = 2e3
const Δθ     = 0.001          # very linear regime
const g_phys = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g_phys)

const STOP_T = 300.0
const Δt_anel = 1.0
const Δt_subst = 1.0
const Δt_expl = 0.05

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

function θᵢ_builder(grid)
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

function build_substepped_model(grid; ω)
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(substeps = 12,
                                         forward_weight = ω,
                                         damping = NoDivergenceDamping())
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder; Δt, stop_time = STOP_T)
    grid = build_grid()
    model = builder(grid)
    if label == "anelastic"
        set!(model; θ = θᵢ_builder(grid))
    else
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ_builder(grid), ρ = ref.density)
    end
    sim = Simulation(model; Δt, stop_time, verbose = false)
    t0 = time(); status = :ok
    try; run!(sim); catch e; status = :crashed; end
    elapsed = time() - t0
    w = model.velocities.w
    wmax_final = Float64(maximum(abs, interior(w)))
    has_nan = any(isnan, parent(w))
    return (; label, wmax_final, has_nan, elapsed, status)
end

@info "=== anelastic ==="
anel = run_one("anelastic", build_anelastic_model; Δt = Δt_anel)
@info "=== explicit ==="
expl = run_one("explicit", build_explicit_model; Δt = Δt_expl)

@info "=== substepped ω sweep ==="
for ω in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
    label = "subst_w$(@sprintf("%.2f", ω))"
    @info "    $label"
    r = run_one(label, grid -> build_substepped_model(grid; ω); Δt = Δt_subst)
    mark = r.has_nan ? "NaN" : "✓"
    ratio = r.wmax_final / expl.wmax_final
    @info @sprintf("    %s ω=%.2f max|w|=%.4e  ratio_sub_expl=%.3f", mark, ω, r.wmax_final, ratio)
end
@info "=== ground truth ==="
@info @sprintf("  anel max|w|=%.4e  expl max|w|=%.4e  expl/anel=%.3f",
               anel.wmax_final, expl.wmax_final, expl.wmax_final / anel.wmax_final)
