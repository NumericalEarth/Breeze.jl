# 3D Sierra-style mountain range — multi-peak elongated terrain.
# Sum of Witch-of-Agnesi peaks arranged as a NE–SW range:
#   - main 3km peak centred at (0,0)
#   - two flanking 1.5km peaks at (±15km, ±10km)
#   - foothill at (-50km, +20km)
# Background wind U from +x. Tests 3D propagation over a multi-scale range.
#
# Optional env (SIERRA3D_*):
#   SIERRA3D_NX, SIERRA3D_NY, SIERRA3D_NZ   (default 200, 200, 100)
#   SIERRA3D_LX, SIERRA3D_LY, SIERRA3D_LZ   (default 120km, 120km, 25km)
#   SIERRA3D_STOP_SECONDS                    (default 3600)
#   SIERRA3D_DT                              (default 2)
#   SIERRA3D_SNAPSHOT_INTERVAL               (default 60)
#   SIERRA3D_U                               (default 10 m/s, +x)
#   SIERRA3D_N                               (default 0.011 s⁻¹)
#   SIERRA3D_OUTPUT_DIR
#   SIERRA3D_ARCH                            cpu | gpu (default cpu)

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
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ
using Oceananigans.Utils: @kernel, @index
using Printf

parse_env(::Type{T}, name, default) where T = parse(T, get(ENV, name, string(default)))

const SIERRA3D_ARCH = lowercase(get(ENV, "SIERRA3D_ARCH", "cpu"))
if SIERRA3D_ARCH == "gpu"
    @eval using CUDA
end
sierra3d_arch() = SIERRA3D_ARCH == "gpu" ? GPU() : CPU()

const Nx = parse_env(Int,     "SIERRA3D_NX", 200)
const Ny = parse_env(Int,     "SIERRA3D_NY", 200)
const Nz = parse_env(Int,     "SIERRA3D_NZ", 100)
const Lx = parse_env(Float64, "SIERRA3D_LX", 120e3)
const Ly = parse_env(Float64, "SIERRA3D_LY", 120e3)
const Lz = parse_env(Float64, "SIERRA3D_LZ", 25e3)
const stop_seconds      = parse_env(Float64, "SIERRA3D_STOP_SECONDS", 3600.0)
const Δt                = parse_env(Float64, "SIERRA3D_DT", 2.0)
const SNAPSHOT_INTERVAL = parse_env(Float64, "SIERRA3D_SNAPSHOT_INTERVAL", 60.0)
const U                 = parse_env(Float64, "SIERRA3D_U", 10.0)
const N                 = parse_env(Float64, "SIERRA3D_N", 0.011)
const OUTPUT_DIR        = get(ENV, "SIERRA3D_OUTPUT_DIR",
                              "validation_output/substepper/terrain_3d_sierra")

const θ₀ = 280.0; const p₀ = 1e5; const pˢᵗ = 1e5; const g = 9.81; const N² = N^2
θ_of_z(z) = θ₀ * exp(N² * z / g)

# 3D Sierra: main peak elongated in y, with flanking foothills.
@inline _bell3d(x, y, x₀, y₀, h, a_x, a_y) = h / (1 + ((x-x₀)/a_x)^2 + ((y-y₀)/a_y)^2)^1.5
hill(x, y) = (
    _bell3d(x, y,    0,    0, 3000.0, 15e3, 30e3) +  # main range NS-elongated
    _bell3d(x, y,  15e3,  20e3, 1500.0, 8e3, 8e3) +  # NE flanker
    _bell3d(x, y,  10e3, -25e3, 1500.0, 8e3, 8e3) +  # SE flanker
    _bell3d(x, y, -30e3,  10e3,  800.0, 12e3, 15e3)  # west foothill
)

mkpath(OUTPUT_DIR)

z_faces = TerrainFollowingVerticalDiscretization(
    collect(range(0, Lz, length = Nz + 1));
    formulation = SLEVE(large_scale_height = Lz/2, small_scale_height = 3e3))
grid = RectilinearGrid(sierra3d_arch();
    size = (Nx, Ny, Nz), halo = (5, 5, 5),
    x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z_faces,
    topology = (Periodic, Periodic, Bounded))
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

@kernel function _init_terrain_bottom_face_w!(ρw, w, ρ, ρu, ρv, grid)
    i, j = @index(Global, NTuple)
    k = 1
    slope_x = ℑxᶜᵃᵃ(i, j, k, grid,
        Breeze.TerrainFollowingDiscretization.∂z∂x, Oceananigans.Face())
    slope_y = ℑyᵃᶜᵃ(i, j, k, grid,
        Breeze.TerrainFollowingDiscretization.∂z∂y, Oceananigans.Face())
    ρu_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, ρu)
    ρv_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, ρv)
    @inbounds begin
        ρw_target  = slope_x * ρu_ccf + slope_y * ρv_ccf
        ρ_ccf      = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρw[i, j, k] = ρw_target
        w[i, j, k]  = ρw_target / ρ_ccf
    end
end

set!(model, ρ = model.dynamics.terrain_reference_density,
     θ = (x, y, z) -> θ_of_z(z),
     u = U, v = 0, w = 0)
let ρu = model.momentum.ρu, ρv = model.momentum.ρv, ρw = model.momentum.ρw,
    w_field = model.velocities.w, ρ = model.dynamics.density,
    arch = architecture(grid)
    Oceananigans.Utils.launch!(arch, grid, :xy, _init_terrain_bottom_face_w!,
                               ρw, w_field, ρ, ρu, ρv, grid)
end
update_state!(model)
compute_contravariant_velocity!(model)

using Oceananigans.Fields: interior
p_int    = interior(model.dynamics.pressure)
pref_int = interior(model.dynamics.terrain_reference_pressure)
@info @sprintf("IC sanity: max|p - p_ref| = %.3e Pa (ratio %.2e)",
               maximum(abs, p_int .- pref_int),
               maximum(abs, p_int .- pref_int) / maximum(abs, p_int))

h_max = maximum(hill(x, y) for x in range(-Lx/2, Lx/2, length=801), y in range(-Ly/2, Ly/2, length=801))
@info @sprintf("3D Sierra: max h = %.0f m, Fr_h = h_max·N/U = %.2f", h_max, h_max*N/U)

simulation = Simulation(model; Δt, stop_time = stop_seconds)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
if SNAPSHOT_INTERVAL > 0
    simulation.output_writers[:snapshots] = JLD2Writer(
        model, (u = model.velocities.u, v = model.velocities.v, w = model.velocities.w);
        filename = joinpath(OUTPUT_DIR, "terrain_3d_sierra_snapshots.jld2"),
        schedule = TimeInterval(SNAPSHOT_INTERVAL),
        overwrite_existing = true)
end

@info @sprintf("Starting 3D Sierra range: %d×%d×%d, %.0fkm×%.0fkm×%.0fkm",
               Nx, Ny, Nz, Lx/1e3, Ly/1e3, Lz/1e3)
wall0 = time_ns()
run!(simulation)
wall = 1e-9 * (time_ns() - wall0)

u_final = interior(model.velocities.u)
v_final = interior(model.velocities.v)
w_final = interior(model.velocities.w)
summary_path = joinpath(OUTPUT_DIR, "terrain_3d_sierra_summary.txt")
open(summary_path, "w") do io
    println(io, "3D Sierra-style mountain range\n")
    @printf(io, "Config: Nx,Ny,Nz=%d,%d,%d  Lx,Ly,Lz=%.3e,%.3e,%.3e m\n",
            Nx, Ny, Nz, Lx, Ly, Lz)
    @printf(io, "  U=%.2f m/s  N=%.4f s⁻¹  h_max=%.0fm  Fr=%.2f\n",
            U, N, h_max, h_max*N/U)
    @printf(io, "  wall=%.3e s\n\n", wall)
    println(io, "Diagnostics")
    @printf(io, "  maximum_w        = %.6e m/s\n", maximum(abs, w_final))
    @printf(io, "  maximum_u_pert   = %.6e m/s\n", maximum(abs, u_final .- U))
    @printf(io, "  maximum_v        = %.6e m/s\n", maximum(abs, v_final))
    @printf(io, "  IC max|p - p_ref| = %.6e Pa\n",
            maximum(abs, p_int .- pref_int))
    @printf(io, "  nan_count = %d\n", count(isnan, w_final))
end
@info "wrote $summary_path"
