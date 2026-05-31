# Trapped lee waves over Schär topography with two-layer stratification.
#
# Scorer (1949) showed that when the Scorer parameter ℓ² = N²/U² - (1/U)d²U/dz²
# decreases with altitude, gravity waves can be trapped in the lower atmosphere
# and form a downstream-extending wave train ("lee waves") with finite
# horizontal wavelength. This is one of the most observed mountain-wave
# phenomena (lenticular cloud trains).
#
# This validation uses a two-layer N(z) profile:
#   N(z) = N_lower for z < z_tropopause,
#          N_upper for z ≥ z_tropopause,
# with N_lower > N_upper. The transition is smoothed with tanh to avoid a
# discontinuous derivative.
#
# Setup parallels the standard Schär test (h₀=250m, a=5km, λ=4km, U=10) but
# stratification varies so that the wave is trapped below ~10 km.
#
# Usage:
#   julia --project=. validation_output/substepper/terrain_trapped_lee_waves_validation.jl
#
# Optional env (LEE_*):
#   LEE_NX, LEE_NZ                    grid size (default 400, 200)
#   LEE_LX, LEE_LZ                    extents (default 200km, 30km)
#   LEE_STOP_SECONDS                  sim time (default 7200, 2h to develop)
#   LEE_DT                            (default 2)
#   LEE_SNAPSHOT_INTERVAL             (default 600)
#   LEE_H0, LEE_A, LEE_LAMBDA         hill shape (250m, 5km, 4km)
#   LEE_U                             wind (10 m/s)
#   LEE_N_LOWER                       lower-troposphere N (0.011 s⁻¹)
#   LEE_N_UPPER                       upper-troposphere N (0.005 s⁻¹)
#   LEE_Z_TROPOPAUSE                  transition height (10000 m)
#   LEE_TRANSITION_DEPTH              tanh smoothing (1500 m)
#   LEE_ADVECTION                     centered2 | weno9 (default weno9)
#   LEE_FORMULATION                   sleve | linear_tfvd (default sleve)
#   LEE_OUTPUT_DIR

using Breeze
using Breeze.CompressibleEquations: CompressibleDynamics,
                                    SplitExplicitTimeDiscretization,
                                    compute_contravariant_velocity!
using Breeze.TerrainFollowingDiscretization: SlopeOutsideInterpolation,
                                              TerrainFollowingVerticalDiscretization,
                                              LinearDecay, SLEVE,
                                              materialize_terrain!, build_terrain_metrics
using Oceananigans
using Oceananigans.Grids: znode
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.OutputWriters: JLD2Writer, TimeInterval
using Oceananigans.Architectures: architecture
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑzᵃᵃᶠ
using Oceananigans: @kernel, @index
using Printf

parse_env(::Type{T}, name, default) where T = parse(T, get(ENV, name, string(default)))

const Nx = parse_env(Int,     "LEE_NX", 400)
const Nz = parse_env(Int,     "LEE_NZ", 200)
const Lx = parse_env(Float64, "LEE_LX", 200e3)
const Lz = parse_env(Float64, "LEE_LZ", 30e3)
const stop_seconds      = parse_env(Float64, "LEE_STOP_SECONDS", 7200.0)
const Δt                = parse_env(Float64, "LEE_DT", 2.0)
const SNAPSHOT_INTERVAL = parse_env(Float64, "LEE_SNAPSHOT_INTERVAL", 600.0)
const h₀                = parse_env(Float64, "LEE_H0", 250.0)
const a                 = parse_env(Float64, "LEE_A", 5e3)
const λ                 = parse_env(Float64, "LEE_LAMBDA", 4e3)
const U                 = parse_env(Float64, "LEE_U", 10.0)
const N_LOWER           = parse_env(Float64, "LEE_N_LOWER", 0.011)
const N_UPPER           = parse_env(Float64, "LEE_N_UPPER", 0.005)
const Z_TROPOPAUSE      = parse_env(Float64, "LEE_Z_TROPOPAUSE", 10e3)
const TRANSITION_DEPTH  = parse_env(Float64, "LEE_TRANSITION_DEPTH", 1500.0)
const ADVECTION         = lowercase(get(ENV, "LEE_ADVECTION", "weno9"))
const FORMULATION       = lowercase(get(ENV, "LEE_FORMULATION", "sleve"))
const OUTPUT_DIR        = get(ENV, "LEE_OUTPUT_DIR",
                              "validation_output/substepper/terrain_trapped_lee_waves")

const θ₀  = 280.0
const p₀  = 1e5
const pˢᵗ = 1e5
const g   = 9.81

# Two-layer N(z) profile, smoothed with tanh
@inline N²(z) = let
    s = 0.5 * (1 + tanh((z - Z_TROPOPAUSE) / TRANSITION_DEPTH))
    (N_LOWER^2) * (1 - s) + (N_UPPER^2) * s
end

# θ(z) by integrating dθ/θ = N²/g dz from z=0 with θ(0)=θ₀
# Closed form for two-layer with tanh blend is messy; use numerical integration
# at IC build time.
function θ_of_z(z)
    z_lo = zero(z)
    θ = θ₀
    Δ = (z - z_lo) / 200
    if Δ <= 0
        return θ
    end
    for n in 1:200
        z_mid = z_lo + Δ * (n - 0.5)
        θ *= exp(N²(z_mid) / g * Δ)
    end
    return θ
end

# Schär hill (same as 2D Schär test)
hill(x, y) = h₀ * exp(-(x / a)^2) * cos(π * x / λ)^2

advection() = ADVECTION == "weno9" ? WENO(order = 9) : Centered(order = 2)

mkpath(OUTPUT_DIR)

z_faces = TerrainFollowingVerticalDiscretization(
    collect(range(0, Lz, length = Nz + 1));
    formulation = FORMULATION == "sleve" ?
        SLEVE(large_scale_height = Lz/2, small_scale_height = 2.5e3) :
        LinearDecay(),
)

grid = RectilinearGrid(CPU();
    size = (Nx, Nz),
    halo = (5, 5),
    x = (-Lx/2, Lx/2),
    z = z_faces,
    topology = (Periodic, Flat, Bounded),
)
materialize_terrain!(grid, hill)
metrics = build_terrain_metrics(grid, SlopeOutsideInterpolation())

time_discretization = SplitExplicitTimeDiscretization(
    acoustic_cfl = 0.5,
    sponge = Breeze.CompressibleEquations.UpperSponge(damping_rate = 1/300,
                                                      depth = 7.5e3),
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
     u = U,
     v = 0,
     w = 0,
)

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

# Print N² profile at a few altitudes for diagnostics
@info "Stratification profile (two-layer):"
for z in (0, 2500, 5000, 7500, 10000, 12500, 15000, 20000)
    @info @sprintf("  z = %6.0f m   N = %.4e s⁻¹   ℓ² = N²/U² = %.4e",
                    z, sqrt(N²(z)), N²(z) / U^2)
end

simulation = Simulation(model; Δt, stop_time = stop_seconds)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

if SNAPSHOT_INTERVAL > 0
    simulation.output_writers[:snapshots] = JLD2Writer(
        model,
        (u = model.velocities.u, w = model.velocities.w);
        filename     = joinpath(OUTPUT_DIR, "terrain_trapped_lee_waves_snapshots.jld2"),
        schedule     = TimeInterval(SNAPSHOT_INTERVAL),
        overwrite_existing = true,
    )
end

@info @sprintf("Starting trapped lee waves: %d×%d, Lx=%.1fkm, %.1fh sim",
               Nx, Nz, Lx/1e3, stop_seconds/3600)
@info @sprintf("  Scorer ℓ²: lower=%.3e/m², upper=%.3e/m², drop=%.1fx",
               N_LOWER^2/U^2, N_UPPER^2/U^2, (N_LOWER/N_UPPER)^2)

wall0 = time_ns()
run!(simulation)
wall = 1e-9 * (time_ns() - wall0)

u_final = interior(model.velocities.u)
w_final = interior(model.velocities.w)

# Find the lee wave wavelength: simple DFT (no external FFTW dep) of w at
# the trapping altitude ~ 5 km.
k_z = argmin(abs.(collect(0:Nz-1) .* (Lz/Nz) .- 5e3))
w_slice = w_final[:, 1, k_z+1]  # 1D in x
w_mean = sum(w_slice) / length(w_slice)
anomaly = w_slice .- w_mean
# Power spectrum via O(N²) DFT — fine for N≤1000
function _power(a, m)
    re = zero(eltype(a)); im = zero(eltype(a))
    N = length(a)
    for n in 1:N
        ϕ = 2π * (m - 1) * (n - 1) / N
        re += a[n] * cos(ϕ)
        im -= a[n] * sin(ϕ)
    end
    return re^2 + im^2
end
modes = 1:(Nx ÷ 2)
power = [_power(anomaly, m) for m in modes]
k_axis = collect(modes) .* (2π / Lx)
peak_k = k_axis[argmax(power)]
λ_lee = peak_k > 0 ? 2π / peak_k : NaN
@info @sprintf("Peak lee-wavelength at z=5km: %.2f km (target ~ 7-10 km from theory)",
               λ_lee / 1e3)

summary_path = joinpath(OUTPUT_DIR, "terrain_trapped_lee_waves_summary.txt")
open(summary_path, "w") do io
    println(io, "Trapped lee waves (two-layer N) validation\n")
    println(io, "Configuration")
    @printf(io, "  Nx, Nz = %d, %d\n", Nx, Nz)
    @printf(io, "  Lx, Lz = %.3e, %.3e m\n", Lx, Lz)
    @printf(io, "  stop_time = %.3e s, Δt = %.3e s\n", stop_seconds, Δt)
    @printf(io, "  Schär hill: h₀=%.1fm, a=%.1fm, λ=%.1fm\n", h₀, a, λ)
    @printf(io, "  U=%.2f m/s\n", U)
    @printf(io, "  N_lower=%.4e, N_upper=%.4e s⁻¹\n", N_LOWER, N_UPPER)
    @printf(io, "  Z_tropopause=%.1fm, transition depth=%.1fm\n",
            Z_TROPOPAUSE, TRANSITION_DEPTH)
    @printf(io, "  Scorer ratio (ℓ_lower/ℓ_upper) = %.2f\n",
            N_LOWER / N_UPPER)
    @printf(io, "  formulation = %s\n", FORMULATION)
    @printf(io, "  advection   = %s\n", ADVECTION)
    @printf(io, "  wall clock  = %.3e s\n", wall)
    println(io)
    println(io, "Diagnostics")
    @printf(io, "  maximum_w        = %.6e m/s\n", maximum(abs, w_final))
    @printf(io, "  maximum_u_pert   = %.6e m/s\n", maximum(abs, u_final .- U))
    @printf(io, "  peak_lee_wavelength_z5km = %.6e m\n", λ_lee)
    @printf(io, "  IC max|p - p_ref| = %.6e Pa\n",
            maximum(abs, p_int .- pref_int))
    @printf(io, "  nan_count = %d\n", count(isnan, w_final))
end
@info "wrote $summary_path"
