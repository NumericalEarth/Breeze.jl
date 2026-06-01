# 3D canyon: two side-by-side 3D bell mountains with a gap between them.
# The gap acts as a 3D canyon — flow accelerates through it. Tests channelling
# and obstacle-pair interaction in 3D.
#
# Optional env (CANYON3D_*):
#   CANYON3D_NX, CANYON3D_NY, CANYON3D_NZ   (default 200, 200, 100)
#   CANYON3D_LX, CANYON3D_LY, CANYON3D_LZ   (default 120km, 120km, 25km)
#   CANYON3D_STOP_SECONDS                    (default 3600)
#   CANYON3D_DT                              (default 2)
#   CANYON3D_SNAPSHOT_INTERVAL               (default 60)
#   CANYON3D_H_PEAK                          (default 1500 m)
#   CANYON3D_A                               (default 8 km)
#   CANYON3D_SEPARATION                      (default 30 km, gap in y)
#   CANYON3D_U                               (default 10 m/s, +x)
#   CANYON3D_N                               (default 0.01 s⁻¹)
#   CANYON3D_OUTPUT_DIR
#   CANYON3D_ARCH                            cpu | gpu (default cpu)

using Breeze
using Breeze.CompressibleEquations: CompressibleDynamics,
                                    SplitExplicitTimeDiscretization,
                                    compute_contravariant_velocity!
using Breeze.TerrainFollowingDiscretization: SlopeOutsideInterpolation,
                                              TerrainFollowingVerticalDiscretization,
                                              SLEVE,
                                              materialize_terrain!, build_terrain_metrics
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.OutputWriters: JLD2Writer, TimeInterval
using Oceananigans.Architectures: architecture
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ
using Oceananigans.Utils: @kernel, @index
using Printf

parse_env(::Type{T}, name, default) where T = parse(T, get(ENV, name, string(default)))

const CANYON3D_ARCH = lowercase(get(ENV, "CANYON3D_ARCH", "cpu"))
if CANYON3D_ARCH == "gpu"
    @eval using CUDA
end
canyon3d_arch() = CANYON3D_ARCH == "gpu" ? GPU() : CPU()

const Nx = parse_env(Int,     "CANYON3D_NX", 200)
const Ny = parse_env(Int,     "CANYON3D_NY", 200)
const Nz = parse_env(Int,     "CANYON3D_NZ", 100)
const Lx = parse_env(Float64, "CANYON3D_LX", 120e3)
const Ly = parse_env(Float64, "CANYON3D_LY", 120e3)
const Lz = parse_env(Float64, "CANYON3D_LZ", 25e3)
const stop_seconds      = parse_env(Float64, "CANYON3D_STOP_SECONDS", 3600.0)
const Δt                = parse_env(Float64, "CANYON3D_DT", 2.0)
const SNAPSHOT_INTERVAL = parse_env(Float64, "CANYON3D_SNAPSHOT_INTERVAL", 60.0)
const h_peak            = parse_env(Float64, "CANYON3D_H_PEAK", 1500.0)
const a_peak            = parse_env(Float64, "CANYON3D_A", 8e3)
const separation        = parse_env(Float64, "CANYON3D_SEPARATION", 30e3)
const U                 = parse_env(Float64, "CANYON3D_U", 10.0)
const N                 = parse_env(Float64, "CANYON3D_N", 0.01)
const OUTPUT_DIR        = get(ENV, "CANYON3D_OUTPUT_DIR",
                              "validation_output/substepper/terrain_3d_canyon")

const θ₀ = 280.0; const p₀ = 1e5; const pˢᵗ = 1e5; const g = 9.81; const N² = N^2
θ_of_z(z) = θ₀ * exp(N² * z / g)

# Two 3D bell mountains side by side in y, both centred at x=0.
@inline _bell3d(x, y, x₀, y₀, h, a) = h / (1 + ((x-x₀)/a)^2 + ((y-y₀)/a)^2)^1.5
const y_south = -separation / 2; const y_north = +separation / 2
hill(x, y) = _bell3d(x, y, 0, y_south, h_peak, a_peak) +
             _bell3d(x, y, 0, y_north, h_peak, a_peak)

mkpath(OUTPUT_DIR)

z_faces = TerrainFollowingVerticalDiscretization(
    collect(range(0, Lz, length = Nz + 1));
    formulation = SLEVE(large_scale_height = Lz/2, small_scale_height = 2.5e3))
grid = RectilinearGrid(canyon3d_arch();
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
     θ = (x, y, z) -> θ_of_z(z), u = U, v = 0, w = 0)
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
@info @sprintf("3D Canyon: h_peak=%.0fm, sep=%.0fkm, Fr=%.2f",
                h_peak, separation/1e3, h_peak*N/U)
@info @sprintf("IC sanity: max|p - p_ref| = %.3e Pa (ratio %.2e)",
               maximum(abs, p_int .- pref_int),
               maximum(abs, p_int .- pref_int) / maximum(abs, p_int))

simulation = Simulation(model; Δt, stop_time = stop_seconds)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
if SNAPSHOT_INTERVAL > 0
    simulation.output_writers[:snapshots] = JLD2Writer(
        model, (u = model.velocities.u, v = model.velocities.v, w = model.velocities.w);
        filename = joinpath(OUTPUT_DIR, "terrain_3d_canyon_snapshots.jld2"),
        schedule = TimeInterval(SNAPSHOT_INTERVAL),
        overwrite_existing = true)
end

wall0 = time_ns()
run!(simulation)
wall = 1e-9 * (time_ns() - wall0)

u_final = interior(model.velocities.u)
v_final = interior(model.velocities.v)
w_final = interior(model.velocities.w)
summary_path = joinpath(OUTPUT_DIR, "terrain_3d_canyon_summary.txt")
open(summary_path, "w") do io
    println(io, "3D canyon (two side-by-side bell mountains)\n")
    @printf(io, "Config: Nx,Ny,Nz=%d,%d,%d\n", Nx, Ny, Nz)
    @printf(io, "  h_peak=%.0fm a=%.0fm separation=%.0fm\n",
            h_peak, a_peak, separation)
    @printf(io, "  U=%.2f m/s N=%.4f s⁻¹ Fr=%.3f\n", U, N, h_peak*N/U)
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
