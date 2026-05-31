## Distributed (multi-GPU) production driver for the YuDidlake2019 tropical-cyclone
## rainband experiment.
##
## This is the distributed sibling of `examples/tropical_cyclone_with_rainband.jl`.
## The science setup (Jordan 1958 sounding, modified-Rankine balanced vortex via
## Picard iteration, MN10 stratiform rainband heating, WRF-style upper sponge) is
## identical; the differences are all infrastructure:
##
##   * MPI + an x-partitioned Distributed/NCCLDistributed architecture.
##   * Configurable resolution for a *production* run (default Δx = 100 m).
##   * No host-side analysis / figures: 3D fields are streamed to disk with a
##     per-rank JLD2Writer. Figures are produced by a separate post-processing job.
##   * A `--benchmark-steps N` calibration mode that times N steps so wall time
##     for the full integration can be estimated by weak scaling before committing
##     a large allocation.
##
## Multi-GPU only works with `CompressibleDynamics(SplitExplicitTimeDiscretization)`
## (no Poisson solve) — which is exactly what this case uses. The split-explicit
## substepping path is also where the single-GPU optimization campaign landed its
## ~19% win, and the topology-aware / halo-fill-elimination work (Phase 2C) pays
## off here specifically because each removed halo fill is an avoided NCCL/MPI
## exchange.
##
## Usage (see examples/distributed_tropical_cyclone.sh for the SLURM recipe):
##   srun ... julia --project=examples examples/distributed_tropical_cyclone.jl \
##       --dx 100 --nz 75 --stop-time 24 --output-interval 1
##   # calibration only (no integration, just per-step timing):
##   srun ... julia ... --benchmark-steps 10

using MPI
MPI.Init()

## Cray MPICH inserts a malformed env entry after MPI_Init on multi-node srun,
## which breaks CUDA.jl → multi-node GPU hangs. Strip it (Oceananigans #5513).
include(joinpath(@__DIR__, "sanitize_environ.jl"))
SanitizeEnviron.sanitize_environ!()

using Breeze
using Breeze: CompressibleDynamics
using Breeze.CompressibleEquations: SplitExplicitTimeDiscretization
using Oceananigans: Oceananigans
using Oceananigans.Units

using CUDA
using NCCL
using Printf

## NCCLDistributed lives in OceananigansNCCLExt (triggered by `using NCCL` +
## `using CUDA`). Package extensions aren't directly `using`-able; fetch through
## Base.get_extension. Mirrors examples/profile_supercell.jl.
const NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
NCCLExt === nothing && error("OceananigansNCCLExt did not load — check NCCL/CUDA install.")
const NCCLDistributed = NCCLExt.NCCLDistributed

## ---------------------------------------------------------------------------
## Command-line arguments
## ---------------------------------------------------------------------------

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

use_nccl        = !("--no-nccl" in ARGS)
FT              = Dict("Float32" => Float32, "Float64" => Float64)[argval("--float-type", "Float32")]
Δx_request      = parse(Float64, argval("--dx", "100"))           # horizontal resolution, m
vertical_choice = argval("--vertical", "gate")                    # "gate" (181-level stretched) or "uniform"
Nz_uniform      = parse(Int,     argval("--nz", "75"))            # vertical levels (uniform mode only)
stop_time_hours = parse(Float64, argval("--stop-time", "24"))    # integration length, hours
output_interval = parse(Float64, argval("--output-interval", "1")) # 3D snapshot cadence, hours
with_heating    = "--heating" in ARGS                             # rainband heating on/off (default: spinup)
## Output goes to $SCRATCH by default — a single 3D snapshot is many GB/rank and
## would blow the home quota. Override with --output-dir.
default_output_dir = haskey(ENV, "SCRATCH") ? joinpath(ENV["SCRATCH"], "tc_distributed") :
                                              joinpath(@__DIR__, "output_tc_distributed")
output_dir      = argval("--output-dir", default_output_dir)
benchmark_steps = parse(Int,     argval("--benchmark-steps", "0")) # >0: time N steps and exit
warmup_steps    = parse(Int,     argval("--warmup-steps", "3"))   # warmup steps before benchmark window
benchmark_dt    = parse(Float64, argval("--benchmark-dt", "0.5")) # fixed Δt during benchmark, s

Oceananigans.defaults.FloatType = FT

Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
rank  = MPI.Comm_rank(MPI.COMM_WORLD)
arch  = use_nccl ?
    NCCLDistributed(GPU(); partition = Partition(Ngpus, 1)) :
    Distributed(GPU(); partition = Partition(Ngpus, 1))

## Diagnostic checkpoint: barrier then a flushed rank-0 log. Because the barrier
## precedes the print, the LAST "CHECKPOINT" line in the log marks the last phase
## that completed on all ranks — so a hang localizes to the span between the last
## printed checkpoint and the next one. The flush defeats srun stderr buffering.
function checkpoint(msg)
    MPI.Barrier(MPI.COMM_WORLD)
    if rank == 0
        @info "CHECKPOINT: $msg"
        flush(stderr)
    end
    return nothing
end

## ---------------------------------------------------------------------------
## Domain & grid (YuDidlake2019 §3a1: 642 km × 642 km box)
## ---------------------------------------------------------------------------
##
## Vertical grid options:
##   * "gate"    — SAM GATE_IDEAL stretched grid (PR #397): 181 levels, 50 m at
##                 the surface → 100 m through the troposphere → 300 m aloft,
##                 model top at 27 km. This is the resolution you actually want
##                 for a 100 m horizontal LES of a TC; the boundary layer and
##                 cloud layer are resolved at 50–100 m.
##   * "uniform" — the example's flat grid: Nz levels over 25 km (Δz ≈ 333 m at
##                 Nz = 75). Cheaper, coarse aloft.

Lx = 642kilometers

## SAM GATE_IDEAL stretched vertical grid, verbatim from PR #397's gate.jl:
## 50 m near the surface, linearly stretched to 100 m through the troposphere,
## then to 300 m aloft, model top at 27 km. Produces 181 levels with a clean
## 50→300 m spacing (no sliver cells). NB: PiecewiseStretchedDiscretization with
## the same breakpoints gives a *different* 259-level grid with 25 m slivers at
## each breakpoint, which would throttle the acoustic substep — so we use the
## canonical GATE construction here.
function gate_vertical_grid(zᵗ; Δz⁰ = 50, Δzᵖ = 100, Δzᵗ = 300)
    z₁, z₂, z₃ = 1275, 5100, 18000   # transition heights
    z_faces = [0.0]
    z = 0.0
    while z < zᵗ
        α = clamp((z - z₁) / (z₂ - z₁), 0, 1)
        β = clamp((z - z₂) / (z₃ - z₂), 0, 1)
        Δz = Δz⁰ + α * (Δzᵖ - Δz⁰) + β * (Δzᵗ - Δzᵖ)
        z = min(z + Δz, zᵗ)
        push!(z_faces, z)
    end
    return z_faces
end

if vertical_choice == "gate"
    z_faces = gate_vertical_grid(27000)
    z_spec = z_faces
    Nz = length(z_faces) - 1
    Lz = z_faces[end]
elseif vertical_choice == "uniform"
    Nz = Nz_uniform
    Lz = 25kilometers
    z_spec = (0, Lz)
else
    error("Unknown --vertical: $vertical_choice (expected 'gate' or 'uniform')")
end

## X-only decomposition requires Nx divisible by Ngpus. Snap Nx to the nearest
## multiple of the rank count so any (--dx, Ngpus) pair works; Δx adjusts slightly.
Nx_request = round(Int, Lx / Δx_request)
Nx = Ny = max(Ngpus, Ngpus * round(Int, Nx_request / Ngpus))
Δx = Lx / Nx
if rank == 0 && Nx != Nx_request
    @info @sprintf("Adjusted Nx %d → %d (multiple of %d ranks) ⇒ Δx = %.1f m",
                   Nx_request, Nx, Ngpus, Δx)
end

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       x = (-Lx / 2, Lx / 2), y = (-Lx / 2, Lx / 2), z = z_spec,
                       topology = (Periodic, Periodic, Bounded))

## True cell-center heights (correct for both uniform and stretched grids).
## z is not partitioned, so this is the full global column on every rank.
z_centers = Array(znodes(grid, Center()))
Δz_min, Δz_max = extrema(diff(Array(znodes(grid, Face()))))

if rank == 0
    backend = use_nccl ? "NCCL" : "MPI"
    @info "Distributed TC rainband (production)" backend Ngpus FT Nx Ny Nz Δx vertical_choice Lz stop_time_hours with_heating benchmark_steps
    @info @sprintf("Vertical: %s, Nz = %d, Δz ∈ [%.0f, %.0f] m, top = %.1f km",
                   vertical_choice, Nz, Δz_min, Δz_max, Lz / 1e3)
    @info @sprintf("Per-GPU grid: %d × %d × %d = %.1f M cells/GPU",
                   Nx ÷ Ngpus, Ny, Nz, (Nx ÷ Ngpus) * Ny * Nz / 1e6)
end
checkpoint("grid built")

## ---------------------------------------------------------------------------
## Jordan (1958) hurricane-season mean sounding (host-side, replicated per rank)
## ---------------------------------------------------------------------------

jordan_z_m = [
    0.0, 132.0, 583.0, 1054.0, 1547.0, 2063.0, 2609.0, 3182.0,
    3792.0, 4442.0, 5138.0, 5888.0, 6703.0, 7595.0, 8581.0, 9682.0,
    10935.0, 12396.0, 13236.0, 14177.0, 15260.0, 16568.0, 17883.0, 19620.0,
    20743.0, 22139.0, 23971.0,
]

jordan_T_K = [
    26.3, 26.0, 23.0, 19.8, 17.3, 14.6, 11.8, 8.6, 5.1, 1.4,
    -2.5, -6.9, -11.9, -17.7, -24.8, -33.2, -43.3, -55.2, -61.5, -67.6,
    -72.2, -73.5, -69.8, -63.9, -60.6, -57.3, -54.0,
] .+ 273.15

jordan_θ_K = [
    298.0, 299.0, 300.0, 302.0, 304.0, 307.0, 309.0, 312.0, 315.0, 318.0,
    321.0, 324.0, 328.0, 332.0, 335.0, 338.0, 342.0, 345.0, 348.0, 354.0,
    364.0, 386.0, 418.0, 468.0, 500.0, 542.0, 597.0,
]

jordan_p_mb = [
    1015.1, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0,
    550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0, 175.0, 150.0,
    125.0, 100.0, 80.0, 60.0, 50.0, 40.0, 30.0,
]

sounding_grid = RectilinearGrid(size = length(jordan_z_m) - 1, z = jordan_z_m,
                                topology = (Flat, Flat, Bounded))

jordan_θ = ZFaceField(sounding_grid)
jordan_T = ZFaceField(sounding_grid)
jordan_p = ZFaceField(sounding_grid)

interior(jordan_θ, 1, 1, :) .= jordan_θ_K
interior(jordan_T, 1, 1, :) .= jordan_T_K
interior(jordan_p, 1, 1, :) .= jordan_p_mb .* 100

## Clamp z into the sounding's range. The GATE grid tops at 27 km but the Jordan
## sounding ends at ~24 km; without clamping, the reference-state Exner integral
## interpolates into jordan_θ's unfilled halo (≈0) above the sounding top, the
## integrand g/(cₚθ) diverges, the Exner function hits -Inf, and `(-Inf)^κ` throws
## a DomainError. Clamping extends the profile with a constant (stable) top value,
## which is fine under the sponge layer.
z_snd_lo, z_snd_hi = first(jordan_z_m), last(jordan_z_m)
clamp_sounding(z) = clamp(z, z_snd_lo, z_snd_hi)
θ_env(z) = Oceananigans.Fields.interpolate(clamp_sounding(z), jordan_θ)
T_env(z) = Oceananigans.Fields.interpolate(clamp_sounding(z), jordan_T)
p_env(z) = Oceananigans.Fields.interpolate(clamp_sounding(z), jordan_p)

## ---------------------------------------------------------------------------
## YD19 vortex parameters
## ---------------------------------------------------------------------------

f             = 5.0e-5
v_max_surface = 43
a_decay       = 0.5
rmw_surface   = 31kilometers
z_vortex_top  = 16kilometers
r_taper_start = 250kilometers
r_taper_end   = 300kilometers

## ---------------------------------------------------------------------------
## Reference state & thermodynamic constants
## ---------------------------------------------------------------------------

constants = ThermodynamicConstants()
Rᵈ  = constants.molar_gas_constant / constants.dry_air.molar_mass
g   = constants.gravitational_acceleration
κ   = Rᵈ / constants.dry_air.heat_capacity
cᵖᵈ = constants.dry_air.heat_capacity

checkpoint("before ReferenceState")
reference_state = ReferenceState(grid, constants;
                                 surface_pressure = p_env(0.0),
                                 potential_temperature = θ_env)
checkpoint("after ReferenceState")

## z is not partitioned, so the local column is the full global column on every rank.
pᵣ = Array(interior(reference_state.pressure, 1, 1, :))
ρᵣ = Array(interior(reference_state.density, 1, 1, :))
Tᵣ = Array(interior(reference_state.temperature, 1, 1, :))
## z_centers was derived from the grid above (correct for stretched grids).

## ---------------------------------------------------------------------------
## Vortex kinematics — RMW(z) and modified-Rankine v(r, z)
## ---------------------------------------------------------------------------

ΔT_floor_default = 0.01

function rmw_analytic(z; ΔT_floor = ΔT_floor_default)
    T_out = T_env(z_vortex_top)
    ΔT_0 = T_env(0.0) - T_out
    ΔT_z = max(T_env(z) - T_out, ΔT_floor)
    return rmw_surface * sqrt(ΔT_0 / ΔT_z)
end

function tangential_wind(r, z; ΔT_floor = ΔT_floor_default)
    r ≥ r_taper_end  && return 0.0
    z ≥ z_vortex_top && return 0.0
    rmw_z = rmw_analytic(z; ΔT_floor)
    v_adj = rmw_surface / rmw_z
    vt = r ≤ rmw_z ?
        v_max_surface * v_adj * r / rmw_z :
        v_max_surface * v_adj * (rmw_z / r)^a_decay
    if r > r_taper_start
        ξ = (r - r_taper_start) / (r_taper_end - r_taper_start)
        vt *= cos(π / 2 * ξ)^2
    end
    return vt
end

## ---------------------------------------------------------------------------
## Balanced-vortex IC (Nolan 2001 / WRF em_tropical_cyclone), host-side
## ---------------------------------------------------------------------------

function dpdz_centered(p, k, i, z_centers, Nz)
    if k == 1
        return (p[i, 2] - p[i, 1]) / (z_centers[2] - z_centers[1])
    elseif k == Nz
        return (p[i, Nz] - p[i, Nz - 1]) / (z_centers[Nz] - z_centers[Nz - 1])
    else
        return (p[i, k + 1] - p[i, k - 1]) / (z_centers[k + 1] - z_centers[k - 1])
    end
end

function solve_balanced_vortex_iterative(r_grid, z_centers, v2d, p_col, T_col, ρ_col;
                                         Rᵈ = 287.04, g = 9.81,
                                         α = 0.5, max_iter = 200, tol = 1.0e-3,
                                         r_safe_min = 100.0)
    Nr = length(r_grid)
    Nz = length(z_centers)
    Δr = r_grid[2] - r_grid[1]
    p = [p_col[k] for i in 1:Nr, k in 1:Nz]
    T = [T_col[k] for i in 1:Nr, k in 1:Nz]
    history = Float64[]
    for iter in 1:max_iter
        T_prev = copy(T)
        ρ = p ./ (Rᵈ .* T)
        for k in 1:Nz
            p[end, k] = p_col[k]
            for i in (Nr - 1):-1:1
                ρ_face = 0.5 * (ρ[i, k] + ρ[i + 1, k])
                v_face = 0.5 * (v2d[i, k] + v2d[i + 1, k])
                r_face = 0.5 * (r_grid[i] + r_grid[i + 1])
                dp_dr = ρ_face * (f * v_face + v_face^2 / max(r_face, r_safe_min))
                p[i, k] = p[i + 1, k] - dp_dr * Δr
            end
        end
        T_new = similar(T)
        for i in 1:Nr
            for k in 1:(Nz - 1)
                dp_dz = dpdz_centered(p, k, i, z_centers, Nz)
                ρ_hyd = max(-dp_dz / g, 1.0e-3)
                T_new[i, k] = p[i, k] / (Rᵈ * ρ_hyd)
            end
            T_new[i, Nz] = T_col[Nz]
        end
        T .= α .* T_new .+ (1 - α) .* T_prev
        maxΔT = maximum(abs.(T .- T_prev))
        push!(history, maxΔT)
        maxΔT < tol && break
    end
    return (; p, T, ρ = p ./ (Rᵈ .* T), history)
end

rmw_profile = [rmw_analytic(z) for z in z_centers]

Nr_pre = 301
r_pre  = collect(range(0.0, r_taper_end, length = Nr_pre))
pˢᵗ    = 1.0e5

rank == 0 && @info "Computing balanced vortex IC (Picard iteration)..."
v2d = [tangential_wind(r_pre[i], z_centers[k]) for i in 1:Nr_pre, k in 1:Nz]
bal = solve_balanced_vortex_iterative(r_pre, z_centers, v2d, pᵣ, Tᵣ, ρᵣ; Rᵈ, g)

θ_pre  = [bal.T[i, k] * (pˢᵗ / pᵣ[k])^κ for i in 1:Nr_pre, k in 1:Nz]
vortex = (; v = v2d, p = bal.p, θ = θ_pre, T = bal.T, ρ = bal.ρ)

rank == 0 && @info @sprintf("IC converged in %d Picard iters", length(bal.history))
checkpoint("vortex IC computed")

@inline function lookup_rz(table, r::Real, z::Real)
    r_c = clamp(r, first(r_pre), last(r_pre))
    z_c = clamp(z, first(z_centers), last(z_centers))
    ir = searchsortedfirst(r_pre, r_c); ir = clamp(ir, 2, length(r_pre))
    iz = searchsortedfirst(z_centers, z_c); iz = clamp(iz, 2, length(z_centers))
    r0, r1 = r_pre[ir - 1], r_pre[ir]
    z0, z1 = z_centers[iz - 1], z_centers[iz]
    fr = (r_c - r0) / (r1 - r0)
    fz = (z_c - z0) / (z1 - z0)
    v00 = table[ir - 1, iz - 1]; v10 = table[ir, iz - 1]
    v01 = table[ir - 1, iz];     v11 = table[ir, iz]
    return (1 - fr) * (1 - fz) * v00 + fr * (1 - fz) * v10 +
        (1 - fr) * fz * v01 + fr * fz * v11
end

uᵢ(x, y, z) = (r = sqrt(x^2 + y^2); r < 1.0 ? 0.0 : -(y / r) * lookup_rz(vortex.v, r, z))
vᵢ(x, y, z) = (r = sqrt(x^2 + y^2); r < 1.0 ? 0.0 : (x / r) * lookup_rz(vortex.v, r, z))
Tᵢ(x, y, z) = lookup_rz(vortex.T, sqrt(x^2 + y^2), z)
ρᵢ(x, y, z) = lookup_rz(vortex.ρ, sqrt(x^2 + y^2), z)

## ---------------------------------------------------------------------------
## Rainband heating (YD19 Eq. 3 / MN10) — device profiles per rank
## ---------------------------------------------------------------------------

Q_max  = 4.24f0 / Float32(hour)
z_bs   = Float32(4kilometers)
σ_r    = Float32(6kilometers)
σ_zs   = Float32(2kilometers)
t_full = Float32(1hour)

ρᵣ_device = CuArray(Float32.(ρᵣ))   # distributed path is always GPU-backed
Πᵣ        = Float32[(pᵣ[k] / pˢᵗ)^κ for k in 1:Nz]
Πᵣ_device = CuArray(Πᵣ)

const π_F32   = Float32(π)
const π_4_F32 = Float32(π / 4)
const π_2_F32 = Float32(π / 2)
const twoπ_F32 = Float32(2π)

@inline function rainband_heating(i, j, k, grid, clock, fields, p)
    x = Oceananigans.Grids.xnode(i, j, k, grid, Center(), Center(), Center())
    y = Oceananigans.Grids.ynode(i, j, k, grid, Center(), Center(), Center())
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    t = clock.time

    ramp = clamp((t - p.t_on) / (p.t_full - p.t_on), 0, 1)
    r = sqrt(x^2 + y^2)
    λ = atan(y, x)
    r_bs = (60 - 10 * (λ / π_4_F32)) * 1000 + z
    G_r = exp(-(r - r_bs)^2 / 2p.σ_r^2)

    z_rel = (z - p.z_bs) / p.σ_zs
    V_z = ifelse(abs(z_rel) < 1, sin(π_F32 * z_rel), 0.0f0)

    λ_c = mod(λ + π_4_F32 + π_F32, twoπ_F32) - π_F32
    A_λ = exp(-(λ_c / π_4_F32)^8)

    Q = p.Q_max * G_r * V_z * A_λ * ramp
    ρᵣ_k = @inbounds p.ρᵣ[k]
    Πᵣ_k = @inbounds p.Πᵣ[k]
    return ρᵣ_k * Q / Πᵣ_k
end

heating_params  = (; Q_max, σ_r, σ_zs, z_bs, t_on = 0.0f0, t_full, ρᵣ = ρᵣ_device, Πᵣ = Πᵣ_device)
heating_forcing = Forcing(rainband_heating; discrete_form = true, parameters = heating_params)

## ---------------------------------------------------------------------------
## Upper-level sponge (WRF damp_opt=2 analog)
## ---------------------------------------------------------------------------

sponge_rate     = 1.0f0 / Float32(333seconds)
## Damping layer occupies the top of the domain: GATE starts at 19 km (model top
## 27 km, per PR #397); the uniform 25 km box uses the example's 20 km start.
sponge_z_bottom = Float32(vertical_choice == "gate" ? 19kilometers : 20kilometers)
sponge_z_top    = Float32(Lz)

ρθᵣ        = Float32[ρᵣ[k] * θ_env(z_centers[k]) for k in 1:Nz]
ρθᵣ_device = CuArray(ρθᵣ)

sponge_vel_params = (z_bot = sponge_z_bottom, z_top = sponge_z_top, rate = sponge_rate)
sponge_ρθ_params  = (z_bot = sponge_z_bottom, z_top = sponge_z_top, rate = sponge_rate, ρθ_bg = ρθᵣ_device)

@inline function sponge_mask(z, z_bot, z_top)
    ξ = (z - z_bot) / (z_top - z_bot)
    return ifelse(ξ ≤ 0, zero(ξ), sin(π_2_F32 * ξ)^2)
end

@inline function sponge_ρu_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Face(), Center(), Center())
    mask = sponge_mask(z, p.z_bot, p.z_top)
    return -p.rate * mask * @inbounds fields.ρu[i, j, k]
end

@inline function sponge_ρv_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Face(), Center())
    mask = sponge_mask(z, p.z_bot, p.z_top)
    return -p.rate * mask * @inbounds fields.ρv[i, j, k]
end

@inline function sponge_ρw_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Face())
    mask = sponge_mask(z, p.z_bot, p.z_top)
    return -p.rate * mask * @inbounds fields.ρw[i, j, k]
end

@inline function sponge_ρθ_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    mask = sponge_mask(z, p.z_bot, p.z_top)
    ρθ_tgt = @inbounds p.ρθ_bg[k]
    return -p.rate * mask * (@inbounds fields.ρθ[i, j, k] - ρθ_tgt)
end

sponge_ρu = Forcing(sponge_ρu_fn; discrete_form = true, parameters = sponge_vel_params)
sponge_ρv = Forcing(sponge_ρv_fn; discrete_form = true, parameters = sponge_vel_params)
sponge_ρw = Forcing(sponge_ρw_fn; discrete_form = true, parameters = sponge_vel_params)
sponge_ρθ = Forcing(sponge_ρθ_fn; discrete_form = true, parameters = sponge_ρθ_params)

## ---------------------------------------------------------------------------
## Model
## ---------------------------------------------------------------------------

coriolis  = FPlane(; f)
dynamics  = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                 surface_pressure = p_env(0.0),
                                 reference_potential_temperature = θ_env)
advection = WENO(order = 5)
forcing = with_heating ?
    (ρu = sponge_ρu, ρv = sponge_ρv, ρw = sponge_ρw, ρθ = (heating_forcing, sponge_ρθ)) :
    (ρu = sponge_ρu, ρv = sponge_ρv, ρw = sponge_ρw, ρθ = sponge_ρθ)

checkpoint("before AtmosphereModel")
model = AtmosphereModel(grid; dynamics, coriolis, advection, forcing)
checkpoint("after AtmosphereModel (before set!)")
set!(model; u = uᵢ, v = vᵢ, T = Tᵢ, ρ = ρᵢ)
checkpoint("after set!")

if rank == 0
    CUDA.reclaim()
    total_GB     = CUDA.total_memory() / 2^30
    available_GB = CUDA.available_memory() / 2^30
    @info @sprintf("GPU memory after model construction (rank 0): %.2f / %.2f GiB used (%.2f free)",
                   total_GB - available_GB, total_GB, available_GB)
end

## ---------------------------------------------------------------------------
## Simulation + CFL wizard (shared by benchmark and production paths)
## ---------------------------------------------------------------------------

simulation = Simulation(model; Δt = benchmark_dt, stop_time = stop_time_hours * hours)
conjure_time_step_wizard!(simulation, cfl = 0.5)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

u, v, w = model.velocities

## ---------------------------------------------------------------------------
## Benchmark / calibration mode: warm up, then time N wizard-driven steps and
## exit. Because this uses the real CFL wizard, the reported Δt is the one the
## production run will actually pick — so the wall-time projection is grounded
## in both the measured ms/step AND the true time step (weak-scaling estimate).
## ---------------------------------------------------------------------------

if benchmark_steps > 0
    MPI.Barrier(MPI.COMM_WORLD)
    simulation.stop_iteration = warmup_steps
    simulation.stop_time = Inf      # let stop_iteration govern during calibration
    run!(simulation)                # compilation + warmup; wizard settles Δt
    MPI.Barrier(MPI.COMM_WORLD)

    Δt_wizard = simulation.Δt
    simulation.stop_iteration = warmup_steps + benchmark_steps
    elapsed = @elapsed run!(simulation)
    MPI.Barrier(MPI.COMM_WORLD)

    if rank == 0
        ms_per_step = 1e3 * elapsed / benchmark_steps
        steps_full  = ceil(Int, stop_time_hours * 3600 / Δt_wizard)
        wall_h      = steps_full * ms_per_step / 1e3 / 3600
        @info @sprintf("Benchmark: %d steps in %.3f s → %.1f ms/step  (wizard Δt = %.3f s, %d GPUs)",
                       benchmark_steps, elapsed, ms_per_step, Δt_wizard, Ngpus)
        @info @sprintf("Projection: %.0f h integration @ Δt=%.3fs → %d steps → %.1f h wall (%.1f node-days)",
                       stop_time_hours, Δt_wizard, steps_full, wall_h,
                       wall_h / 24 * (Ngpus / 4))
    end
    exit(0)
end

## ---------------------------------------------------------------------------
## Production integration
## ---------------------------------------------------------------------------

function progress(sim)
    ## maximum over a distributed Field is collective — all ranks must call it.
    mu = maximum(abs, u)
    mv = maximum(abs, v)
    mw = maximum(abs, w)
    if rank == 0
        @info @sprintf("iter: %d, t: %s, Δt: %s, max|u,v|: %.2f, %.2f m/s, max|w|: %.2e m/s",
                       iteration(sim), prettytime(sim), prettytime(sim.Δt), mu, mv, mw)
    end
    return nothing
end
add_callback!(simulation, progress, TimeInterval(30minutes))

## Per-rank 3D output. Each rank writes its own x-slab; the post-processing job
## reassembles them. Saving u, v, w, T, ρ — the quantities the YD19 azimuthal-mean
## analysis consumes.
T_field = model.temperature
ρ_field = model.dynamics.density
outputs = (; u, v, w, T = T_field, ρ = ρ_field)

stage = with_heating ? "heated" : "spinup"
output_prefix = joinpath(output_dir,
                         @sprintf("tc_%s_dx%dm_%s_nz%d", stage, round(Int, Δx), vertical_choice, Nz))
rank == 0 && mkpath(output_dir)
MPI.Barrier(MPI.COMM_WORLD)

## On a distributed grid JLD2Writer appends the rank to the filename itself, so
## we pass the bare prefix (avoids the doubled `_rankN_rankN` suffix).
simulation.output_writers[:fields] = JLD2Writer(model, outputs;
                                                filename = output_prefix * ".jld2",
                                                schedule = TimeInterval(output_interval * hours),
                                                overwrite_existing = true)

rank == 0 && @info @sprintf("Running %s stage to %.0f h, 3D output every %.1f h → %s_rank*.jld2",
                            stage, stop_time_hours, output_interval, output_prefix)

run!(simulation)

rank == 0 && @info "=== Distributed run complete ==="
