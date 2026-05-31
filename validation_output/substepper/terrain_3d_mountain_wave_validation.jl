# 3D mountain wave validation for the terrain-following acoustic substepper.
#
# Setup: isolated 3D bell-shaped mountain in a triply-periodic horizontal domain.
# h(x, y) = h₀ · (1 + (r/a)²)^(-3/2),   r² = x² + y²
# Background: uniform U, dry isothermal-θ atmosphere (N = const).
# Standard linear-theory parameters: produces a downstream wave cone
# (Smith 1980).
#
# This stresses TFVD operators in a non-degenerate y-direction (slope_y,
# contravariant momentum cross-terms, 3D divergence over terrain), which the
# 2D Schär test cannot exercise.
#
# Usage:
#   julia --project=. validation_output/substepper/terrain_3d_mountain_wave_validation.jl
#
# Optional env (all SCHAR3D_*):
#   SCHAR3D_NX, SCHAR3D_NY, SCHAR3D_NZ           grid size (default 80, 80, 60)
#   SCHAR3D_LX, SCHAR3D_LY, SCHAR3D_LZ           domain extents (default 80km, 80km, 25km)
#   SCHAR3D_STOP_SECONDS                          sim time (default 3600)
#   SCHAR3D_DT                                    timestep (default 4)
#   SCHAR3D_SNAPSHOT_INTERVAL                     JLD2 snapshot cadence (default 600, 0=off)
#   SCHAR3D_H0                                    mountain height (default 250m)
#   SCHAR3D_A                                     mountain half-width (default 4km)
#   SCHAR3D_U                                     background wind (default 10 m/s, x-direction)
#   SCHAR3D_N                                     Brunt-Väisälä freq (default 0.01 s⁻¹)
#   SCHAR3D_SPONGE_DEPTH                          (default 7.5 km)
#   SCHAR3D_SPONGE_RATE                           (default 0.00333 s⁻¹, CM1-style)
#   SCHAR3D_FORMULATION                           sleve | linear_tfvd  (default sleve)
#   SCHAR3D_ADVECTION                             centered2 | weno9  (default weno9)
#   SCHAR3D_OUTPUT_DIR                            output directory
#   SCHAR3D_ARCH                                  cpu | gpu (default cpu)

using Breeze
using Breeze.CompressibleEquations: CompressibleDynamics,
                                    SplitExplicitTimeDiscretization,
                                    compute_contravariant_velocity!
using Breeze.TerrainFollowingDiscretization: SlopeOutsideInterpolation,
                                              TerrainFollowingVerticalDiscretization,
                                              LinearDecay, SLEVE,
                                              materialize_terrain!, build_terrain_metrics

using Oceananigans
using Oceananigans.Grids: znode, rnode
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.OutputWriters: JLD2Writer, TimeInterval
using Oceananigans.Architectures: architecture
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, ℑxyᶠᶠᵃ
using Oceananigans: @kernel, @index
using Printf

parse_env(::Type{T}, name, default) where T = parse(T, get(ENV, name, string(default)))
parse_env(::Type{Bool}, name, default) =
    lowercase(get(ENV, name, string(default))) in ("true", "1", "yes")

const SCHAR3D_ARCH = lowercase(get(ENV, "SCHAR3D_ARCH", "cpu"))
if SCHAR3D_ARCH == "gpu"
    @eval using CUDA
end
schar_arch() = SCHAR3D_ARCH == "gpu" ? GPU() : CPU()

const Nx  = parse_env(Int,     "SCHAR3D_NX", 80)
const Ny  = parse_env(Int,     "SCHAR3D_NY", 80)
const Nz  = parse_env(Int,     "SCHAR3D_NZ", 60)
const Lx  = parse_env(Float64, "SCHAR3D_LX", 80e3)
const Ly  = parse_env(Float64, "SCHAR3D_LY", 80e3)
const Lz  = parse_env(Float64, "SCHAR3D_LZ", 25e3)
const stop_seconds       = parse_env(Float64, "SCHAR3D_STOP_SECONDS", 3600.0)
const Δt                 = parse_env(Float64, "SCHAR3D_DT", 4.0)
const SNAPSHOT_INTERVAL  = parse_env(Float64, "SCHAR3D_SNAPSHOT_INTERVAL", 600.0)
const h₀                 = parse_env(Float64, "SCHAR3D_H0", 250.0)
const a                  = parse_env(Float64, "SCHAR3D_A", 4e3)
const U                  = parse_env(Float64, "SCHAR3D_U", 10.0)
const N                  = parse_env(Float64, "SCHAR3D_N", 0.01)
const sponge_depth       = parse_env(Float64, "SCHAR3D_SPONGE_DEPTH", 7.5e3)
const sponge_rate        = parse_env(Float64, "SCHAR3D_SPONGE_RATE", 1/300)
const FORMULATION        = lowercase(get(ENV, "SCHAR3D_FORMULATION", "sleve"))
const ADVECTION          = lowercase(get(ENV, "SCHAR3D_ADVECTION", "weno9"))
const OUTPUT_DIR         = get(ENV, "SCHAR3D_OUTPUT_DIR",
                                "validation_output/substepper/terrain_3d_mountain_wave")

const N²  = N^2
const θ₀  = 280.0
const p₀  = 1e5
const pˢᵗ = 1e5
const g   = 9.81

# Bell-shaped 3D mountain (radial Witch-of-Agnesi-like form).
@inline radial(x, y) = sqrt(x^2 + y^2)
hill(x, y) = h₀ * (1 + (radial(x, y) / a)^2)^(-1.5)

θ_of_z(z) = θ₀ * exp(N² * z / g)

advection() = ADVECTION == "weno9" ? WENO(order = 9) : Centered(order = 2)

mkpath(OUTPUT_DIR)

# Build grid
z_faces = TerrainFollowingVerticalDiscretization(
    collect(range(0, Lz, length = Nz + 1));
    formulation = FORMULATION == "sleve" ?
        SLEVE(large_scale_height = Lz/2, small_scale_height = 2.5e3) :
        LinearDecay(),
)

grid = RectilinearGrid(schar_arch();
    size = (Nx, Ny, Nz),
    halo = (5, 5, 5),
    x = (-Lx/2, Lx/2),
    y = (-Ly/2, Ly/2),
    z = z_faces,
    topology = (Periodic, Periodic, Bounded),
)

materialize_terrain!(grid, hill)
metrics = build_terrain_metrics(grid, SlopeOutsideInterpolation())

# Build dynamics
time_discretization = SplitExplicitTimeDiscretization(
    acoustic_cfl = 0.5,
    sponge = Breeze.CompressibleEquations.UpperSponge(damping_rate = sponge_rate,
                                                      depth = sponge_depth),
)

dynamics = CompressibleDynamics(time_discretization;
    terrain_metrics = metrics,
    reference_potential_temperature = θ_of_z,
    surface_pressure = p₀,
    standard_pressure = pˢᵗ,
)

model = AtmosphereModel(grid; dynamics = dynamics,
    timestepper = :AcousticRungeKutta3,
    advection   = advection())

# ─────────────────────────────────────────────────────────────────────────────
# IC: at-rest stratified atmosphere + uniform U, with bottom-face kinematic BC.
# `set!` on TFVD now uses znode (physical altitude) automatically via the
# Breeze override of `Oceananigans.Grids.node`, so the resulting (ρθ, ρ) state
# is in machine-precision discrete hydrostatic balance with the
# terrain_reference_pressure (no spurious column).
# ─────────────────────────────────────────────────────────────────────────────

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

set!(model,
     ρ = model.dynamics.terrain_reference_density,
     θ = (x, y, z) -> θ_of_z(z),
     u = U,
     v = 0,
     w = 0,
)

let ρu = model.momentum.ρu, ρv = model.momentum.ρv, ρw = model.momentum.ρw,
    w_field = model.velocities.w, ρ = model.dynamics.density,
    arch = architecture(grid)
    Oceananigans.Utils.launch!(arch, grid, :xy, _init_terrain_bottom_face_w!,
                               ρw, w_field, ρ, ρu, ρv, grid)
end

update_state!(model)
compute_contravariant_velocity!(model)

# ─────────────────────────────────────────────────────────────────────────────
# Run + IC sanity
# ─────────────────────────────────────────────────────────────────────────────

using Oceananigans.Fields: interior
p_int    = interior(model.dynamics.pressure)
pref_int = interior(model.dynamics.terrain_reference_pressure)
ρu_int   = interior(model.momentum.ρu)
ρv_int   = interior(model.momentum.ρv)
ρw_int   = interior(model.momentum.ρw)
@info @sprintf("IC sanity: max|p - p_ref| = %.3e Pa (ratio %.2e), max|ρw̃ surface| = %.3e",
               maximum(abs, p_int .- pref_int),
               maximum(abs, p_int .- pref_int) / maximum(abs, p_int),
               maximum(abs, interior(model.dynamics.contravariant_vertical_momentum)[:, :, 1]))

simulation = Simulation(model; Δt, stop_time = stop_seconds)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

if SNAPSHOT_INTERVAL > 0
    snapshot_outputs = (
        u = model.velocities.u,
        v = model.velocities.v,
        w = model.velocities.w,
    )
    simulation.output_writers[:snapshots] = JLD2Writer(
        model, snapshot_outputs;
        filename     = joinpath(OUTPUT_DIR, "terrain_3d_mountain_wave_snapshots.jld2"),
        schedule     = TimeInterval(SNAPSHOT_INTERVAL),
        overwrite_existing = true,
    )
end

@info @sprintf("Starting 3D mountain wave: %d×%d×%d, Lx=%.1fkm Ly=%.1fkm Lz=%.1fkm",
               Nx, Ny, Nz, Lx/1e3, Ly/1e3, Lz/1e3)
@info @sprintf("  h₀=%.0fm  a=%.1fkm  U=%.1f m/s  N=%.3f s⁻¹  →  Fr=h₀N/U=%.2f, La=Na/U=%.2f",
               h₀, a/1e3, U, N, h₀*N/U, N*a/U)

wall0 = time_ns()
run!(simulation)
wall = 1e-9 * (time_ns() - wall0)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics summary
# ─────────────────────────────────────────────────────────────────────────────

u_final = interior(model.velocities.u)
v_final = interior(model.velocities.v)
w_final = interior(model.velocities.w)

# Energy in the lee region (downstream of mountain in +x direction)
n_quarter = Nx ÷ 4
n_summit  = Nx ÷ 2
wave_zone = view(w_final, n_summit:n_summit+n_quarter, :, :)

summary_path = joinpath(OUTPUT_DIR, "terrain_3d_mountain_wave_summary.txt")
open(summary_path, "w") do io
    println(io, "3D bell mountain wave validation\n")
    println(io, "Configuration")
    @printf(io, "  Nx, Ny, Nz = %d, %d, %d\n", Nx, Ny, Nz)
    @printf(io, "  Lx, Ly, Lz = %.3e, %.3e, %.3e m\n", Lx, Ly, Lz)
    @printf(io, "  stop_time = %.3e s, Δt = %.3e s\n", stop_seconds, Δt)
    @printf(io, "  h₀=%.1f m, a=%.1f m, U=%.2f m/s, N=%.4f s⁻¹\n",
            h₀, a, U, N)
    @printf(io, "  Fr=h₀N/U = %.3f  (linear: <0.5,  blocking onset: ~1)\n",
            h₀ * N / U)
    @printf(io, "  La=Na/U  = %.3f  (hydrostatic: >>1)\n", N * a / U)
    @printf(io, "  formulation = %s\n", FORMULATION)
    @printf(io, "  advection   = %s\n", ADVECTION)
    @printf(io, "  sponge depth=%.1fm, rate=%.4e /s\n", sponge_depth, sponge_rate)
    @printf(io, "  wall clock  = %.3e s\n", wall)
    println(io)
    println(io, "Diagnostics")
    @printf(io, "  maximum_w_global  = %.6e\n", maximum(abs, w_final))
    @printf(io, "  maximum_w_lee     = %.6e\n", maximum(abs, wave_zone))
    @printf(io, "  maximum_u_pert    = %.6e\n", maximum(abs, u_final .- U))
    @printf(io, "  maximum_v         = %.6e\n", maximum(abs, v_final))
    @printf(io, "  IC max|p - p_ref| = %.6e\n",
            maximum(abs, p_int .- pref_int))
    @printf(io, "  nan_count = %d\n", count(isnan, w_final))
end
@info "wrote $summary_path"
