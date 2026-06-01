# Sierra-Nevada-style 2D mountain range validation.
#
# A long, multi-scale terrain profile inspired by an east–west cross-section
# of the Sierra Nevada: a broad envelope ~100 km wide with several embedded
# peaks at 10–20 km scale. Tests the dycore's handling of multi-scale topography
# at parameters where both the broad envelope (propagating waves) and the
# narrower peaks (trapped/evanescent) coexist.
#
# Terrain (sum of Witch-of-Agnesi peaks):
#   h(x) = Σₙ hₙ · (1 + ((x - xₙ) / aₙ)²)⁻¹
# Three peaks centred around x = 0: a broad 2.5 km envelope (a = 25 km) plus
# two narrower 1 km peaks (a = 8 km) shifted east and west. The result has
# the multi-scale character of a real mountain range without depending on
# DEM data.
#
# Usage:
#   julia --project=. validation_output/substepper/terrain_sierra_cross_section_validation.jl
#
# Optional env (SIERRA_*):
#   SIERRA_NX, SIERRA_NZ              (default 400, 200)
#   SIERRA_LX, SIERRA_LZ              (default 400km, 25km)
#   SIERRA_STOP_SECONDS               (default 7200, 2h to develop)
#   SIERRA_DT                         (default 4)
#   SIERRA_SNAPSHOT_INTERVAL          (default 120, snapshots every 2 min)
#   SIERRA_U                          background wind (default 10 m/s)
#   SIERRA_N                          Brunt-Väisälä (default 0.011 s⁻¹)
#   SIERRA_OUTPUT_DIR
#   SIERRA_ARCH                       cpu | gpu (default cpu)

using Breeze
using Breeze.CompressibleEquations: CompressibleDynamics,
                                    SplitExplicitTimeDiscretization,
                                    compute_contravariant_velocity!
using Breeze.TerrainFollowingDiscretization: SlopeOutsideInterpolation,
                                              TerrainFollowingVerticalDiscretization,
                                              SLEVE,
                                              materialize_terrain!, build_terrain_metrics
using Oceananigans
using Oceananigans.Grids: znode
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.OutputWriters: JLD2Writer, TimeInterval
using Oceananigans.Architectures: architecture
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑzᵃᵃᶠ
using Oceananigans.Utils: @kernel, @index
using Printf

parse_env(::Type{T}, name, default) where T = parse(T, get(ENV, name, string(default)))

const SIERRA_ARCH = lowercase(get(ENV, "SIERRA_ARCH", "cpu"))
if SIERRA_ARCH == "gpu"
    @eval using CUDA
end
sierra_arch() = SIERRA_ARCH == "gpu" ? GPU() : CPU()

const Nx = parse_env(Int,     "SIERRA_NX", 400)
const Nz = parse_env(Int,     "SIERRA_NZ", 200)
const Lx = parse_env(Float64, "SIERRA_LX", 400e3)
const Lz = parse_env(Float64, "SIERRA_LZ", 25e3)
const stop_seconds      = parse_env(Float64, "SIERRA_STOP_SECONDS", 7200.0)
const Δt                = parse_env(Float64, "SIERRA_DT", 4.0)
const SNAPSHOT_INTERVAL = parse_env(Float64, "SIERRA_SNAPSHOT_INTERVAL", 120.0)
const U                 = parse_env(Float64, "SIERRA_U", 10.0)
const N                 = parse_env(Float64, "SIERRA_N", 0.011)
const OUTPUT_DIR        = get(ENV, "SIERRA_OUTPUT_DIR",
                              "validation_output/substepper/terrain_sierra_cross_section")

const θ₀  = 280.0
const p₀  = 1e5
const pˢᵗ = 1e5
const g   = 9.81
const N²  = N^2

θ_of_z(z) = θ₀ * exp(N² * z / g)

# Sierra-Nevada–inspired cross-section: sum of Witch-of-Agnesi peaks.
#   Broad envelope: 2.5 km high, a = 25 km, centred at x = 0
#   East peak: 1.5 km, a = 8 km, centred at x = +15 km
#   West peak: 1.5 km, a = 10 km, centred at x = -20 km
#   Far foothill: 0.4 km, a = 25 km, x = -70 km
# This gives a multi-scale range ~100 km wide.
@inline _woa(x, x₀, h, a) = h / (1 + ((x - x₀) / a)^2)
hill(x, y) = (
    _woa(x,   0.0, 2500.0, 25e3) +
    _woa(x,  15e3, 1500.0,  8e3) +
    _woa(x, -20e3, 1500.0, 10e3) +
    _woa(x, -70e3,  400.0, 25e3)
)

mkpath(OUTPUT_DIR)

z_faces = TerrainFollowingVerticalDiscretization(
    collect(range(0, Lz, length = Nz + 1));
    formulation = SLEVE(large_scale_height = Lz/2, small_scale_height = 3e3))

grid = RectilinearGrid(sierra_arch();
    size = (Nx, Nz), halo = (5, 5),
    x = (-Lx/2, Lx/2), z = z_faces,
    topology = (Periodic, Flat, Bounded))
materialize_terrain!(grid, hill)
metrics = build_terrain_metrics(grid, SlopeOutsideInterpolation())

time_discretization = SplitExplicitTimeDiscretization(
    acoustic_cfl = 0.5,
    sponge = Breeze.CompressibleEquations.UpperSponge(damping_rate = 1/300,
                                                      depth = 7.5e3))

dynamics = CompressibleDynamics(time_discretization;
    terrain_metrics = metrics,
    reference_potential_temperature = θ_of_z,
    surface_pressure = p₀, standard_pressure = pˢᵗ)

model = AtmosphereModel(grid; dynamics = dynamics,
    timestepper = :AcousticRungeKutta3,
    advection   = WENO(order = 9))

@kernel function _init_terrain_bottom_face_w!(ρw, w, ρ, ρu, grid)
    i, j = @index(Global, NTuple)
    k = 1
    slope_x = ℑxᶜᵃᵃ(i, j, k, grid,
        Breeze.TerrainFollowingDiscretization.∂z∂x, Oceananigans.Face())
    ρu_ccf  = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, ρu)
    @inbounds begin
        ρw_target = slope_x * ρu_ccf
        ρ_ccf     = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρw[i, j, k] = ρw_target
        w[i, j, k]  = ρw_target / ρ_ccf
    end
end

set!(model,
     ρ = model.dynamics.terrain_reference_density,
     θ = (x, z) -> θ_of_z(z),
     u = U, v = 0, w = 0)
let ρu = model.momentum.ρu, ρw = model.momentum.ρw,
    w_field = model.velocities.w, ρ = model.dynamics.density,
    arch = architecture(grid)
    Oceananigans.Utils.launch!(arch, grid, :xy, _init_terrain_bottom_face_w!,
                               ρw, w_field, ρ, ρu, grid)
end
update_state!(model)
compute_contravariant_velocity!(model)

using Oceananigans.Fields: interior
p_int    = interior(model.dynamics.pressure)
pref_int = interior(model.dynamics.terrain_reference_pressure)
@info @sprintf("IC sanity: max|p - p_ref| = %.3e Pa (ratio %.2e)",
               maximum(abs, p_int .- pref_int),
               maximum(abs, p_int .- pref_int) / maximum(abs, p_int))

# Print terrain profile summary
@info "Sierra-inspired terrain peaks:"
peak_x = [-70e3, -20e3, 0.0, 15e3]
for x_p in peak_x
    @info @sprintf("  x = %+6.1f km  h = %5.0f m", x_p/1e3, hill(x_p, 0.0))
end
@info @sprintf("Max terrain height in domain: %.0f m", maximum(hill(x, 0) for x in range(-Lx/2, Lx/2, length=1001)))
@info @sprintf("Fr_h = h_max·N/U = %.2f  La_max = a_max·N/U = %.2f",
               maximum(hill(x, 0) for x in range(-Lx/2, Lx/2, length=1001)) * N / U,
               25e3 * N / U)

simulation = Simulation(model; Δt, stop_time = stop_seconds)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

if SNAPSHOT_INTERVAL > 0
    simulation.output_writers[:snapshots] = JLD2Writer(
        model, (u = model.velocities.u, w = model.velocities.w);
        filename = joinpath(OUTPUT_DIR, "terrain_sierra_cross_section_snapshots.jld2"),
        schedule = TimeInterval(SNAPSHOT_INTERVAL),
        overwrite_existing = true)
end

@info @sprintf("Starting Sierra cross-section: %d×%d, Lx=%.0fkm, %.1fh sim",
               Nx, Nz, Lx/1e3, stop_seconds/3600)

wall0 = time_ns()
run!(simulation)
wall = 1e-9 * (time_ns() - wall0)

u_final = interior(model.velocities.u)
w_final = interior(model.velocities.w)

summary_path = joinpath(OUTPUT_DIR, "terrain_sierra_cross_section_summary.txt")
open(summary_path, "w") do io
    println(io, "Sierra-Nevada-style multi-peak mountain range\n")
    @printf(io, "Configuration\n  Nx, Nz = %d, %d\n  Lx, Lz = %.3e, %.3e m\n",
            Nx, Nz, Lx, Lz)
    @printf(io, "  stop_time = %.3e s, Δt = %.3e s\n", stop_seconds, Δt)
    @printf(io, "  U = %.2f m/s, N = %.4f s⁻¹\n", U, N)
    @printf(io, "  max terrain = %.0f m\n",
            maximum(hill(x, 0) for x in range(-Lx/2, Lx/2, length=1001)))
    @printf(io, "  wall clock = %.3e s\n\n", wall)
    println(io, "Diagnostics")
    @printf(io, "  maximum_w        = %.6e m/s\n", maximum(abs, w_final))
    @printf(io, "  maximum_u_pert   = %.6e m/s\n", maximum(abs, u_final .- U))
    @printf(io, "  IC max|p - p_ref| = %.6e Pa\n",
            maximum(abs, p_int .- pref_int))
    @printf(io, "  nan_count = %d\n", count(isnan, w_final))
end
@info "wrote $summary_path"
