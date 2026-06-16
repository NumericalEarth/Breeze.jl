# Throughput benchmark: does TerrainFollowingVerticalDiscretization (TFVD) slow
# down compressible split-explicit substepping vs a flat RectilinearGrid?
#
# Three cases on an identical grid / dynamics / Δt / step count — only the
# vertical coordinate differs:
#   :flat       — plain RectilinearGrid (non-terrain dynamics path)
#   :tfvd_flat  — TFVD grid with h = 0 (isolates the terrain-following machinery:
#                 contravariant velocity, slope terms, znode — zero actual terrain)
#   :tfvd_hill  — TFVD grid with a Witch-of-Agnesi hill (realistic)
#
# Reports s/step, Mcell-updates/s, and the slowdown ratio vs :flat.
#
# Env: BENCHMARK_ARCH=gpu|cpu  TF_BENCH_NX/NY/NZ  TF_BENCH_STEPS  TF_BENCH_DT

using Oceananigans
using Oceananigans.Architectures: synchronize
using Oceananigans.TimeSteppers: update_state!, time_step!
using Printf

const ARCH = lowercase(get(ENV, "BENCHMARK_ARCH", "gpu"))
ARCH == "gpu" && @eval using CUDA
arch() = ARCH == "gpu" ? GPU() : CPU()
sync() = synchronize(arch())

using Breeze
using Breeze.CompressibleEquations: CompressibleDynamics, SplitExplicitTimeDiscretization
using Breeze.TerrainFollowingDiscretization: TerrainFollowingVerticalDiscretization,
                                             LinearDecay, TwoLevelDecay, materialize_terrain!

parse_env(T, name, default) = parse(T, get(ENV, name, string(default)))
const Nx = parse_env(Int, "TF_BENCH_NX", 256)
const Ny = parse_env(Int, "TF_BENCH_NY", 256)
const Nz = parse_env(Int, "TF_BENCH_NZ", 128)
const STEPS  = parse_env(Int, "TF_BENCH_STEPS", 40)
const WARMUP = parse_env(Int, "TF_BENCH_WARMUP", 3)
const Δt = parse_env(Float64, "TF_BENCH_DT", 2.0)

const Lx, Ly, Lz = 120e3, 120e3, 25e3
const θ₀, p₀, g, N = 288.0, 1e5, 9.81, 0.011
θ_of_z(z) = θ₀ * exp(N^2 * z / g)
const U = 10.0
hill(x, y) = 1500.0 / (1 + (x / 10e3)^2 + (y / 10e3)^2)^1.5

# LinearDecay (cheap basis) vs TwoLevelDecay/SLEVE (sinh basis — where factoring
# the decay basis out of the discrete slope difference saves transcendentals).
const FORMULATION = lowercase(get(ENV, "TF_BENCH_FORMULATION", "linear"))
const LABEL = get(ENV, "TF_BENCH_LABEL", "")  # e.g. "baseline" / "optimized", for the printout
make_formulation() = FORMULATION == "twolevel" ?
    TwoLevelDecay(large_scale_height = Lz/2, small_scale_height = 3e3) : LinearDecay()

function reference_density(model)
    d = model.dynamics
    d.terrain_reference_density !== nothing && return d.terrain_reference_density
    return d.reference_state.density
end

function build_model(case)
    zfaces = collect(range(0, Lz, length = Nz + 1))
    z = case === :flat ? zfaces :
        TerrainFollowingVerticalDiscretization(zfaces; formulation = make_formulation())
    grid = RectilinearGrid(arch(); size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z,
                           topology = (Periodic, Periodic, Bounded))
    case === :tfvd_flat && materialize_terrain!(grid, (x, y) -> 0.0)
    case === :tfvd_hill && materialize_terrain!(grid, hill)

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(acoustic_cfl = 0.5);
                                    reference_potential_temperature = θ_of_z,
                                    surface_pressure = p₀, standard_pressure = p₀)
    model = AtmosphereModel(grid; dynamics, timestepper = :AcousticRungeKutta3,
                            advection = WENO(order = 9))
    set!(model, ρ = reference_density(model), θ = (x, y, z) -> θ_of_z(z), u = U, v = 0, w = 0)
    update_state!(model)
    return model
end

function benchmark(case)
    model = build_model(case)
    for _ in 1:WARMUP; time_step!(model, Δt); end
    sync()
    t0 = time_ns()
    for _ in 1:STEPS; time_step!(model, Δt); end
    sync()
    s_per_step = 1e-9 * (time_ns() - t0) / STEPS
    wmax = maximum(abs, interior(model.velocities.w))
    return s_per_step, wmax
end

@info @sprintf("Terrain-following throughput benchmark | %s | %s%s | %d×%d×%d | %d steps (Δt=%.1f)",
               ARCH, FORMULATION, isempty(LABEL) ? "" : " [$LABEL]", Nx, Ny, Nz, STEPS, Δt)

cases = (:flat, :tfvd_flat, :tfvd_hill)
results = Dict{Symbol, Tuple{Float64, Float64}}()
for case in cases
    @info "benchmarking $case ..."
    results[case] = benchmark(case)
end

ncells = Nx * Ny * Nz
flat_spt = results[:flat][1]
println("\n", "="^72)
@printf("%-12s %12s %16s %12s %10s\n", "case", "s/step", "Mcell-upd/s", "vs flat", "max|w|")
println("-"^72)
for case in cases
    spt, wmax = results[case]
    @printf("%-12s %12.4f %16.1f %11.2f× %10.3e\n",
            case, spt, ncells / spt / 1e6, spt / flat_spt, wmax)
end
println("="^72)
@printf("TFVD machinery overhead (tfvd_flat / flat): %.1f%%\n", 100 * (results[:tfvd_flat][1]/flat_spt - 1))
@printf("Terrain-following total  (tfvd_hill / flat): %.1f%%\n", 100 * (results[:tfvd_hill][1]/flat_spt - 1))
