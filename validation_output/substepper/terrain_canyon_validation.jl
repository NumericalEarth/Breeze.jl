# Canyon / valley flow validation.
#
# Two mountains separated by a deep canyon: tests flow into and out of a
# concave terrain feature, lee-side rotor development, and the duality with
# the "obstacle" Witch-of-Agnesi case. Standard reference: Smith (1989) on
# flow over a step, and lee-vortex / wake studies by Smolarkiewicz & Rotunno
# (1989).
#
# Terrain:
#   h(x) =   h_peak · (1 + ((x - x₁)/a)²)⁻¹     (west peak)
#          + h_peak · (1 + ((x - x₂)/a)²)⁻¹     (east peak)
# with x₁ = -d/2, x₂ = +d/2 separated by d. Between the peaks lies a U-shaped
# canyon naturally formed by the two Witch-of-Agnesi tails.
#
# Optional env (CANYON_*):
#   CANYON_NX, CANYON_NZ                (default 400, 200)
#   CANYON_LX, CANYON_LZ                (default 200km, 25km)
#   CANYON_STOP_SECONDS                 (default 3600)
#   CANYON_DT                           (default 4)
#   CANYON_SNAPSHOT_INTERVAL            (default 60)
#   CANYON_H_PEAK                       (default 1500 m)
#   CANYON_A                            (default 8 km)
#   CANYON_SEPARATION                   (default 40 km — about 2.5 a)
#   CANYON_U                            (default 10 m/s)
#   CANYON_N                            (default 0.01 s⁻¹)
#   CANYON_OUTPUT_DIR
#   CANYON_ARCH                         cpu | gpu

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

const CANYON_ARCH = lowercase(get(ENV, "CANYON_ARCH", "cpu"))
if CANYON_ARCH == "gpu"
    @eval using CUDA
end
canyon_arch() = CANYON_ARCH == "gpu" ? GPU() : CPU()

const Nx = parse_env(Int,     "CANYON_NX", 400)
const Nz = parse_env(Int,     "CANYON_NZ", 200)
const Lx = parse_env(Float64, "CANYON_LX", 200e3)
const Lz = parse_env(Float64, "CANYON_LZ", 25e3)
const stop_seconds      = parse_env(Float64, "CANYON_STOP_SECONDS", 3600.0)
const Δt                = parse_env(Float64, "CANYON_DT", 4.0)
const SNAPSHOT_INTERVAL = parse_env(Float64, "CANYON_SNAPSHOT_INTERVAL", 60.0)
const h_peak            = parse_env(Float64, "CANYON_H_PEAK", 1500.0)
const a_peak            = parse_env(Float64, "CANYON_A", 8e3)
const separation        = parse_env(Float64, "CANYON_SEPARATION", 40e3)
const U                 = parse_env(Float64, "CANYON_U", 10.0)
const N                 = parse_env(Float64, "CANYON_N", 0.01)
const OUTPUT_DIR        = get(ENV, "CANYON_OUTPUT_DIR",
                              "validation_output/substepper/terrain_canyon")

const θ₀  = 280.0
const p₀  = 1e5
const pˢᵗ = 1e5
const g   = 9.81
const N²  = N^2

θ_of_z(z) = θ₀ * exp(N² * z / g)

# Two Witch-of-Agnesi peaks creating a canyon between them.
@inline _woa(x, x₀, h, a) = h / (1 + ((x - x₀) / a)^2)
const x_west = -separation / 2
const x_east = +separation / 2
hill(x, y) = _woa(x, x_west, h_peak, a_peak) +
             _woa(x, x_east, h_peak, a_peak)

mkpath(OUTPUT_DIR)

z_faces = TerrainFollowingVerticalDiscretization(
    collect(range(0, Lz, length = Nz + 1));
    formulation = SLEVE(large_scale_height = Lz/2, small_scale_height = 2.5e3))

grid = RectilinearGrid(canyon_arch();
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

# Terrain summary
canyon_bottom_h = hill(0.0, 0.0)
@info @sprintf("Canyon profile: h_peak=%.0fm at x=±%.0fkm, canyon-floor h=%.0fm",
                h_peak, separation/2/1e3, canyon_bottom_h)
@info @sprintf("Fr_h = h_peak·N/U = %.2f", h_peak * N / U)
@info @sprintf("IC sanity: max|p - p_ref| = %.3e Pa (ratio %.2e)",
               maximum(abs, p_int .- pref_int),
               maximum(abs, p_int .- pref_int) / maximum(abs, p_int))

simulation = Simulation(model; Δt, stop_time = stop_seconds)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
if SNAPSHOT_INTERVAL > 0
    simulation.output_writers[:snapshots] = JLD2Writer(
        model, (u = model.velocities.u, w = model.velocities.w);
        filename = joinpath(OUTPUT_DIR, "terrain_canyon_snapshots.jld2"),
        schedule = TimeInterval(SNAPSHOT_INTERVAL),
        overwrite_existing = true)
end

wall0 = time_ns()
run!(simulation)
wall = 1e-9 * (time_ns() - wall0)

u_final = interior(model.velocities.u)
w_final = interior(model.velocities.w)
summary_path = joinpath(OUTPUT_DIR, "terrain_canyon_summary.txt")
open(summary_path, "w") do io
    println(io, "Canyon (two Witch-of-Agnesi peaks)\n")
    @printf(io, "Config: Nx,Nz=%d,%d  Lx,Lz=%.3e,%.3e m\n", Nx, Nz, Lx, Lz)
    @printf(io, "  h_peak=%.0f m  a=%.0f m  separation=%.0f m\n",
            h_peak, a_peak, separation)
    @printf(io, "  U=%.2f m/s  N=%.4f s⁻¹  Fr=%.3f\n", U, N, h_peak*N/U)
    @printf(io, "  wall=%.3e s\n\n", wall)
    println(io, "Diagnostics")
    @printf(io, "  maximum_w        = %.6e m/s\n", maximum(abs, w_final))
    @printf(io, "  maximum_u_pert   = %.6e m/s\n", maximum(abs, u_final .- U))
    @printf(io, "  IC max|p - p_ref| = %.6e Pa\n",
            maximum(abs, p_int .- pref_int))
    @printf(io, "  nan_count = %d\n", count(isnan, w_final))
end
@info "wrote $summary_path"
