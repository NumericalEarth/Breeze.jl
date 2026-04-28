#####
##### validation/substepping/57b_bubble_ns_sweep.jl
#####
##### Sweep Ns to test the hypothesis that ratio = c_s × Δτ / Δx
##### (i.e., scales as 1/Ns).
#####

include("common.jl")

using Breeze, Oceananigans, Oceananigans.Units, CUDA, Statistics, Printf

const arch = CUDA.functional() ? GPU() : CPU()
const Lx, Lz, Nx, Nz = 20e3, 10e3, 64, 64
const θ₀, N², r₀, Δθ, g_phys = 300.0, 1e-4, 2e3, 0.001, 9.80665
θᵇᵍ(z) = θ₀ * exp(N² * z / g_phys)
const STOP_T, Δt_subst, Δt_expl = 300.0, 1.0, 0.05

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center())); z₀ = 0.3 * grid.Lz
    (x, z) -> θᵇᵍ(z) + Δθ * exp(-((x-x₀)^2 + (z-z₀)^2) / r₀^2)
end

function build_explicit_model(grid)
    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    reference_potential_temperature = θᵇᵍ)
    AtmosphereModel(grid; dynamics, advection = WENO(order = 9))
end

function build_substepped_model(grid; Ns)
    td = SplitExplicitTimeDiscretization(substeps = Ns, forward_weight = 0.55,
                                          damping = NoDivergenceDamping())
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                    timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder; Δt)
    grid = build_grid()
    model = builder(grid)
    if label == "explicit"
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ_builder(grid), ρ = ref.density)
    else
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ_builder(grid), ρ = ref.density)
    end
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    try; run!(sim); catch; end
    w = model.velocities.w
    wmax = Float64(maximum(abs, interior(w)))
    return (; label, wmax, has_nan = any(isnan, parent(w)))
end

@info "=== explicit (ground truth) ==="
expl = run_one("explicit", build_explicit_model; Δt = Δt_expl)
@info @sprintf("  expl max|w|=%.4e", expl.wmax)

c_s = sqrt(1.4 * 287.0 * 300)
Δx = Lx / Nx
@info "=== Ns sweep ==="
for Ns in (6, 12, 24, 48, 96)
    Δτ = Δt_subst / Ns
    cfl = c_s * Δτ / Δx
    r = run_one("Ns$Ns", grid -> build_substepped_model(grid; Ns); Δt = Δt_subst)
    mark = r.has_nan ? "NaN" : "✓"
    ratio = r.wmax / expl.wmax
    @info @sprintf("  %s Ns=%2d Δτ=%.4f CFL=cs·Δτ/Δx=%.4f max|w|=%.4e ratio=%.4f",
                   mark, Ns, Δτ, cfl, r.wmax, ratio)
end
