#####
##### Sweep Δt with fixed Ns to see if substepper converges to explicit
##### at small Δt.
#####

include("common.jl")

using Breeze, Oceananigans, Oceananigans.Units, CUDA, Statistics, Printf

const arch = CUDA.functional() ? GPU() : CPU()
const Lx, Lz, Nx, Nz = 20e3, 10e3, 64, 64
const θ₀, N², r₀, Δθ, g_phys = 300.0, 1e-4, 2e3, 0.001, 9.80665
θᵇᵍ(z) = θ₀ * exp(N² * z / g_phys)
const STOP_T, Δt_expl = 300.0, 0.05

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center())); z₀ = 0.3 * grid.Lz
    (x, z) -> θᵇᵍ(z) + Δθ * exp(-((x-x₀)^2 + (z-z₀)^2) / r₀^2)
end

function build_explicit_model(grid)
    dyn = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = θᵇᵍ)
    AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 9))
end

function build_substepped_model(grid; Ns)
    td = SplitExplicitTimeDiscretization(substeps = Ns, forward_weight = 0.55,
                                          damping = NoDivergenceDamping())
    dyn = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 9),
                    timestepper = :AcousticRungeKutta3)
end

function run_one(builder; Δt)
    grid = build_grid()
    model = builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ_builder(grid), ρ = ref.density)
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    try; run!(sim); catch; end
    w = model.velocities.w
    (; wmax = Float64(maximum(abs, interior(w))), has_nan = any(isnan, parent(w)))
end

@info "=== explicit (ground truth) ==="
expl = run_one(build_explicit_model; Δt = Δt_expl)
@info @sprintf("  expl max|w|=%.4e", expl.wmax)

@info "=== Δt sweep with Ns=12 ==="
for Δt in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    r = run_one(grid -> build_substepped_model(grid; Ns = 12); Δt = Δt)
    mark = r.has_nan ? "NaN" : "✓"
    ratio = r.wmax / expl.wmax
    @info @sprintf("  %s Δt=%.2f Δτ=%.4f max|w|=%.4e ratio=%.4f",
                   mark, Δt, Δt/12, r.wmax, ratio)
end
