# Compare Breeze advection-limited compressible dynamics with CM1, ERF, and WRF
# for the SK94 inertia-gravity wave test.
#
# Cases:
#   1. Breeze fully explicit compressible (small Δt, acoustic CFL < 1) — reference
#   2. Breeze SSP-RK3 split-explicit (Δt=12, Ns=8) — advection-limited
#   3. CM1 split-explicit (psolver=2, Δt=12) [if data exists]
#   4. ERF split-explicit (Δt=12, 60 substeps) [if data exists]
#   5. WRF split-explicit [if NCDatasets available and data exists]

ENV["CUDA_VISIBLE_DEVICES"] = "-1"

using Breeze
using Breeze.CompressibleEquations: ExplicitTimeStepping, SplitExplicitTimeDiscretization
using Oceananigans.Units
using Oceananigans.Grids: xnodes, znodes
using CairoMakie
using Printf

# ── Problem parameters ──────────────────────────────────────────────────────

p₀ = 100000  # Pa  - surface pressure
θ₀ = 300     # K   - reference potential temperature
U  = 20      # m/s - mean wind
N² = 0.01^2  # s⁻² - Brunt-Väisälä frequency squared

# ── Grid ─────────────────────────────────────────────────────────────────────

Nx, Nz = 300, 10
Lx, Lz = 300kilometers, 10kilometers

grid = RectilinearGrid(CPU(), size=(Nx, Nz), halo=(5, 5),
                       x=(0, Lx), z=(0, Lz),
                       topology=(Periodic, Flat, Bounded))

# ── Initial conditions ────────────────────────────────────────────────────────

Δθ = 0.01    # K - perturbation amplitude
a  = 5000    # m - half-width
x₀ = 100000  # m - perturbation center

constants = ThermodynamicConstants()
g  = constants.gravitational_acceleration
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity

θᵇᵍ(z) = θ₀ * exp(N² * z / g)
θᵢ(x, z) = θᵇᵍ(z) + Δθ * sin(π * z / Lz) / (1 + (x - x₀)^2 / a^2)

# ── Build Breeze models ───────────────────────────────────────────────────────

surface_pressure = p₀

# 1. Breeze fully explicit compressible (reference)
explicit_dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                          surface_pressure,
                                          reference_potential_temperature=θᵇᵍ)

# 2. Breeze SSP-RK3 with advection-limited Δt=12 (Ns=8 acoustic substeps)
ssprk3_dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=8,
                                          divergence_damping_coefficient=0.05);
                                         surface_pressure,
                                         reference_potential_temperature=θᵇᵍ)

# 3. Breeze WS-RK3 with advection-limited Δt=12 (Ns=8 acoustic substeps)
wsrk3_dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=8,
                                         divergence_damping_coefficient=0.10);
                                        surface_pressure,
                                        reference_potential_temperature=θᵇᵍ)

model_explicit = AtmosphereModel(grid; advection=WENO(), dynamics=explicit_dynamics)
model_ssprk3   = AtmosphereModel(grid; advection=WENO(), dynamics=ssprk3_dynamics,
                                 timestepper=:AcousticSSPRungeKutta3)
model_wsrk3    = AtmosphereModel(grid; advection=WENO(), dynamics=wsrk3_dynamics,
                                 timestepper=:AcousticRungeKutta3)

# Set initial conditions
for model in (model_explicit, model_ssprk3, model_wsrk3)
    ref = model.dynamics.reference_state
    set!(model; θ=θᵢ, u=U, qᵗ=0, ρ=ref.density)
end

# ── Time step sizes ───────────────────────────────────────────────────────────

Δx, Δz = Lx / Nx, Lz / Nz
ℂˢ = sqrt(cᵖᵈ / (cᵖᵈ - Rᵈ) * Rᵈ * θ₀)

cfl = 0.5
Δt_explicit = cfl / sqrt(((ℂˢ + U) / Δx)^2 + (ℂˢ / Δz)^2)
Δt_ssprk3   = 12.0
Δt_wsrk3    = 12.0

@info "Time steps" Δt_explicit Δt_ssprk3 Δt_wsrk3

# ── Background θ field ────────────────────────────────────────────────────────

θᵇᵍ_field = CenterField(grid)
set!(θᵇᵍ_field, (x, z) -> θᵇᵍ(z))

# ── Run Breeze simulations ───────────────────────────────────────────────────

stop_time = 3000 # seconds

@info "Running Breeze explicit..." Δt=Δt_explicit
sim_explicit = Simulation(model_explicit; Δt=Δt_explicit, stop_time)
run!(sim_explicit)

@info "Running Breeze SSP-RK3 (Δt=12)..." Δt=Δt_ssprk3
sim_ssprk3 = Simulation(model_ssprk3; Δt=Δt_ssprk3, stop_time)
run!(sim_ssprk3)

@info "Running Breeze WS-RK3 (Δt=12)..." Δt=Δt_wsrk3
sim_wsrk3 = Simulation(model_wsrk3; Δt=Δt_wsrk3, stop_time)
run!(sim_wsrk3)

# ── Extract Breeze results ────────────────────────────────────────────────────

xC = xnodes(grid, Center())
zC = znodes(grid, Center())
zF = znodes(grid, Face())
xkm = xC ./ 1e3
zkm_c = zC ./ 1e3
zkm_f = zF ./ 1e3

function extract_results(model)
    θ_field = Field(PotentialTemperature(model))
    compute!(θ_field)
    θ′ = interior(θ_field, :, 1, :) .- interior(θᵇᵍ_field, :, 1, :)
    w = interior(model.velocities.w, :, 1, :)
    return (; θ′, w)
end

r_explicit = extract_results(model_explicit)
r_ssprk3   = extract_results(model_ssprk3)
r_wsrk3    = extract_results(model_wsrk3)

@info "Breeze explicit"  max_θ′=maximum(abs, r_explicit.θ′) max_w=maximum(abs, r_explicit.w)
@info "Breeze SSP-RK3"   max_θ′=maximum(abs, r_ssprk3.θ′)  max_w=maximum(abs, r_ssprk3.w)
@info "Breeze WS-RK3"    max_θ′=maximum(abs, r_wsrk3.θ′)   max_w=maximum(abs, r_wsrk3.w)

# ── Try reading CM1 output (binary, no special packages needed) ──────────────

function read_cm1_scalar_var(dir, file_num, var_index, nx, ny, nz)
    filename = joinpath(dir, @sprintf("cm1out_%06d_s.dat", file_num))
    skip_records = (var_index - 1) * nz
    recl = nx * ny * 4
    data = zeros(Float32, nx, ny, nz)
    open(filename, "r") do io
        seek(io, skip_records * recl)
        for k in 1:nz
            rec = zeros(Float32, nx * ny)
            read!(io, rec)
            data[:, :, k] = reshape(rec, nx, ny)
        end
    end
    return data
end

function read_cm1_w(dir, file_num, nx, ny, nz_w)
    filename = joinpath(dir, @sprintf("cm1out_%06d_w.dat", file_num))
    data = zeros(Float32, nx, ny, nz_w)
    open(filename, "r") do io
        for k in 1:nz_w
            rec = zeros(Float32, nx * ny)
            read!(io, rec)
            data[:, :, k] = reshape(rec, nx, ny)
        end
    end
    return data
end

cm1_x  = collect(0.5:1.0:299.5)
cm1_zc = collect(0.5:1.0:9.5)

have_cm1_explicit = false
have_cm1_substepped = false
cm1_θ′_explicit = cm1_θ′_substepped = nothing
cm1_w_explicit = cm1_w_substepped = nothing

try
    cm1_dir = "/Users/gregorywagner/Projects/CM1/run/igw_compressible"
    global cm1_θ′_explicit = read_cm1_scalar_var(cm1_dir, 11, 2, Nx, 1, Nz)[:, 1, :]
    global cm1_w_explicit  = read_cm1_w(cm1_dir, 11, Nx, 1, Nz+1)[:, 1, :]
    global have_cm1_explicit = true
    @info "CM1 explicit" max_θ′=maximum(abs, cm1_θ′_explicit) max_w=maximum(abs, cm1_w_explicit)
catch e
    @warn "CM1 explicit data not found" exception=e
end

try
    cm1_dir = "/Users/gregorywagner/Projects/CM1/run/igw_substepped"
    global cm1_θ′_substepped = read_cm1_scalar_var(cm1_dir, 11, 2, Nx, 1, Nz)[:, 1, :]
    global cm1_w_substepped  = read_cm1_w(cm1_dir, 11, Nx, 1, Nz+1)[:, 1, :]
    global have_cm1_substepped = true
    @info "CM1 substepped" max_θ′=maximum(abs, cm1_θ′_substepped) max_w=maximum(abs, cm1_w_substepped)
catch e
    @warn "CM1 substepped data not found" exception=e
end

# ── Try reading ERF output (AMReX plotfile format) ────────────────────────────

have_erf = false
erf_θ′ = erf_w = nothing

function read_amrex_plotfile(plotdir)
    header = readlines(joinpath(plotdir, "Header"))
    ncomp = parse(Int, header[2])
    comp_names = [header[2+i] for i in 1:ncomp]

    m = nothing
    for line in header
        m = match(r"\(\((\d+),(\d+),(\d+)\)\s+\((\d+),(\d+),(\d+)\)", line)
        m !== nothing && break
    end
    lo = (parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
    hi = (parse(Int, m[4]), parse(Int, m[5]), parse(Int, m[6]))
    nx = hi[1] - lo[1] + 1
    ny = hi[2] - lo[2] + 1
    nz = hi[3] - lo[3] + 1

    datafile = joinpath(plotdir, "Level_0", "Cell_D_00000")
    data = open(datafile, "r") do io
        while read(io, Char) != '\n' end
        raw = Vector{Float64}(undef, nx * ny * nz * ncomp)
        read!(io, raw)
        raw
    end
    data_4d = reshape(data, nx, ny, nz, ncomp)
    return (; comp_names, data=data_4d, nx, ny, nz, ncomp)
end

try
    erf_dir = "/Users/gregorywagner/Projects/ERF/run_igw"
    # Read final state (plt00250 for Δt=12, 250 steps)
    erf_final = read_amrex_plotfile(joinpath(erf_dir, "plt00250"))
    erf_init  = read_amrex_plotfile(joinpath(erf_dir, "plt00000"))

    theta_idx = findfirst(==("theta"), erf_final.comp_names)
    w_idx = findfirst(==("z_velocity"), erf_final.comp_names)

    theta_final = erf_final.data[:, 1, :, theta_idx]
    theta_init  = erf_init.data[:, 1, :, theta_idx]

    # Background = horizontal average of initial theta
    theta_bg = dropdims(sum(theta_init, dims=1), dims=1) ./ erf_final.nx
    global erf_θ′ = theta_final .- theta_bg'
    global erf_w  = erf_final.data[:, 1, :, w_idx]
    global have_erf = true
    @info "ERF (Δt=12)" max_θ′=maximum(abs, erf_θ′) max_w=maximum(abs, erf_w)
catch e
    @warn "ERF data not available" exception=e
end

# ── Try reading WRF output (requires NCDatasets) ────────────────────────────

have_wrf = false
wrf_θ′ = wrf_w = wrf_zc = wrf_zf = wrf_x = wrf_zkm = nothing

try
    @eval using NCDatasets
    @eval using Statistics: mean
    wrf_file = "/Users/gregorywagner/Projects/WRF/run/sk94_igw/wrfout_d01_0001-01-01_00:00:00"
    global wrf_θ′, wrf_w, wrf_zc, wrf_zf = NCDataset(wrf_file, "r") do ds
        T_final = ds["T"][:, 1, :, end]
        T_init  = ds["T"][:, 1, :, 1]
        θ_bg = dropdims(mean(T_init, dims=1), dims=1)
        θ′ = T_final .- θ_bg'
        W_final = ds["W"][:, 1, :, end]
        PH  = ds["PH"][:, 1, :, end]
        PHB = ds["PHB"][:, 1, :, end]
        z_stag = (PH .+ PHB) ./ g
        z_cell = (z_stag[:, 1:end-1] .+ z_stag[:, 2:end]) ./ 2
        (θ′, W_final, z_cell[1, :], z_stag[1, :])
    end
    global wrf_x = collect(0.5:1.0:299.5)
    global wrf_zkm = wrf_zc ./ 1e3
    global have_wrf = true
    @info "WRF substepped" max_θ′=maximum(abs, wrf_θ′) max_w=maximum(abs, wrf_w)
catch e
    @warn "WRF data not available" exception=e
end

# ── Determine figure layout ─────────────────────────────────────────────────

# Always: Breeze explicit (reference) + Breeze SSP-RK3 + Breeze WS-RK3
ncols = 3
ncols += have_cm1_substepped
ncols += have_erf
ncols += have_wrf

# ── Create comparison figure ────────────────────────────────────────────────

# Collect all θ′ for common colorbar
all_θ′ = [r_explicit.θ′, r_ssprk3.θ′, r_wsrk3.θ′]
have_cm1_explicit   && push!(all_θ′, cm1_θ′_explicit)
have_cm1_substepped && push!(all_θ′, cm1_θ′_substepped)
have_erf            && push!(all_θ′, erf_θ′)
have_wrf            && push!(all_θ′, wrf_θ′)
θ_max = maximum(maximum.(abs, all_θ′))
θ_lim = (-θ_max, θ_max)

z_mid = Nz ÷ 2

fig = Figure(size=(500 * ncols, 900), fontsize=14)

# ── Row 1: θ' heatmaps ─────────────────────────────────────────────────────

Label(fig[0, 1:ncols], "Potential Temperature Perturbation θ' (K) at t = 3000 s",
      fontsize=18, font=:bold)

col = 1

ax_be = Axis(fig[1, col], title=@sprintf("Breeze Explicit\n(Δt=%.1fs)", Δt_explicit),
             xlabel="x (km)", ylabel="z (km)")
hm = heatmap!(ax_be, xkm, zkm_c, r_explicit.θ′, colormap=:balance, colorrange=θ_lim)
col += 1

ax_ssp = Axis(fig[1, col], title=@sprintf("Breeze SSP-RK3\n(Δt=%.0fs, Ns=8)", Δt_ssprk3),
              xlabel="x (km)", ylabel="z (km)")
heatmap!(ax_ssp, xkm, zkm_c, r_ssprk3.θ′, colormap=:balance, colorrange=θ_lim)
col += 1

ax_ws = Axis(fig[1, col], title=@sprintf("Breeze WS-RK3\n(Δt=%.0fs, Ns=8)", Δt_wsrk3),
             xlabel="x (km)", ylabel="z (km)")
heatmap!(ax_ws, xkm, zkm_c, r_wsrk3.θ′, colormap=:balance, colorrange=θ_lim)
col += 1

if have_cm1_substepped
    ax_cm1 = Axis(fig[1, col], title="CM1 Substepped\n(Δt=12s, Ns=6)",
                  xlabel="x (km)", ylabel="z (km)")
    heatmap!(ax_cm1, cm1_x, cm1_zc, cm1_θ′_substepped, colormap=:balance, colorrange=θ_lim)
    col += 1
end

if have_erf
    ax_erf = Axis(fig[1, col], title="ERF Substepped\n(Δt=12s, Ns=60)",
                  xlabel="x (km)", ylabel="z (km)")
    # ERF uses same grid as Breeze (300x10, Δx=Δz=1km), cell centers at 0.5, 1.5, ...
    erf_xkm = collect(range(0.5, 299.5, length=Nx))
    erf_zkm = collect(range(0.5, 9.5, length=Nz))
    heatmap!(ax_erf, erf_xkm, erf_zkm, erf_θ′, colormap=:balance, colorrange=θ_lim)
    col += 1
end

if have_wrf
    ax_wrf = Axis(fig[1, col], title="WRF Substepped\n(Δt=6s, Ns=6)",
                  xlabel="x (km)", ylabel="z (km)")
    heatmap!(ax_wrf, wrf_x, wrf_zkm, wrf_θ′, colormap=:balance, colorrange=θ_lim)
    col += 1
end

Colorbar(fig[1, ncols + 1], hm, label="θ' (K)")

# ── Row 2: θ' cross-section at mid-height ──────────────────────────────────

Label(fig[2, 1:ncols], @sprintf("θ' cross-section at z ≈ %.1f km", zkm_c[z_mid]),
      fontsize=16, font=:bold)

ax_cs = Axis(fig[3, 1:ncols], xlabel="x (km)", ylabel="θ' (K)")

lines!(ax_cs, xkm, r_explicit.θ′[:, z_mid],
       label=@sprintf("Breeze explicit (Δt=%.1fs)", Δt_explicit),
       linewidth=2.5, color=:royalblue)
lines!(ax_cs, xkm, r_ssprk3.θ′[:, z_mid],
       label=@sprintf("Breeze SSP-RK3 (Δt=%.0fs, Ns=8)", Δt_ssprk3),
       linewidth=2.5, color=:purple)
lines!(ax_cs, xkm, r_wsrk3.θ′[:, z_mid],
       label=@sprintf("Breeze WS-RK3 (Δt=%.0fs, Ns=8)", Δt_wsrk3),
       linewidth=2.5, color=:darkorange, linestyle=:dash)

if have_cm1_explicit
    lines!(ax_cs, cm1_x, cm1_θ′_explicit[:, z_mid],
           label="CM1 explicit (Δt=1s)", linewidth=1.5, color=:royalblue, linestyle=:dash)
end
if have_cm1_substepped
    lines!(ax_cs, cm1_x, cm1_θ′_substepped[:, z_mid],
           label="CM1 substepped (Δt=12s, Ns=6)", linewidth=1.5, color=:orangered, linestyle=:dash)
end
if have_erf
    erf_z_mid = Nz ÷ 2
    lines!(ax_cs, erf_xkm, erf_θ′[:, erf_z_mid],
           label="ERF substepped (Δt=12s)", linewidth=1.5, color=:forestgreen, linestyle=:dash)
end
if have_wrf
    wrf_z_mid = argmin(abs.(wrf_zc .- zC[z_mid]))
    lines!(ax_cs, wrf_x, wrf_θ′[:, wrf_z_mid],
           label="WRF substepped (Δt=6s, Ns=6)", linewidth=1.5, color=:green, linestyle=:dashdot)
end

axislegend(ax_cs, position=:rt)

# ── Row 3: w cross-section at mid-height ────────────────────────────────────

Label(fig[4, 1:ncols], @sprintf("w cross-section at z ≈ %.1f km", zkm_f[z_mid + 1]),
      fontsize=16, font=:bold)

ax_w = Axis(fig[5, 1:ncols], xlabel="x (km)", ylabel="w (m/s)")

lines!(ax_w, xkm, r_explicit.w[:, z_mid + 1],
       label=@sprintf("Breeze explicit (Δt=%.1fs)", Δt_explicit),
       linewidth=2.5, color=:royalblue)
lines!(ax_w, xkm, r_ssprk3.w[:, z_mid + 1],
       label=@sprintf("Breeze SSP-RK3 (Δt=%.0fs, Ns=8)", Δt_ssprk3),
       linewidth=2.5, color=:purple)
lines!(ax_w, xkm, r_wsrk3.w[:, z_mid + 1],
       label=@sprintf("Breeze WS-RK3 (Δt=%.0fs, Ns=8)", Δt_wsrk3),
       linewidth=2.5, color=:darkorange, linestyle=:dash)

if have_cm1_explicit
    cm1_zf = collect(0.0:1.0:10.0)
    cm1_zf_mid = argmin(abs.(cm1_zf .- zkm_f[z_mid + 1]))
    lines!(ax_w, cm1_x, cm1_w_explicit[:, cm1_zf_mid],
           label="CM1 explicit (Δt=1s)", linewidth=1.5, color=:royalblue, linestyle=:dash)
end
if have_cm1_substepped
    cm1_zf = collect(0.0:1.0:10.0)
    cm1_zf_mid = argmin(abs.(cm1_zf .- zkm_f[z_mid + 1]))
    lines!(ax_w, cm1_x, cm1_w_substepped[:, cm1_zf_mid],
           label="CM1 substepped (Δt=12s, Ns=6)", linewidth=1.5, color=:orangered, linestyle=:dash)
end
if have_erf
    # ERF w is at cell centers (not staggered in plotfile output)
    lines!(ax_w, erf_xkm, erf_w[:, erf_z_mid],
           label="ERF substepped (Δt=12s)", linewidth=1.5, color=:forestgreen, linestyle=:dash)
end
if have_wrf
    wrf_zf_km = wrf_zf ./ 1e3
    wrf_zf_mid = argmin(abs.(wrf_zf_km .- zkm_f[z_mid + 1]))
    lines!(ax_w, wrf_x, wrf_w[:, wrf_zf_mid],
           label="WRF substepped (Δt=6s, Ns=6)", linewidth=1.5, color=:green, linestyle=:dashdot)
end

axislegend(ax_w, position=:rt)

# ── Save ────────────────────────────────────────────────────────────────────

outfile = joinpath(@__DIR__, "igw_substepping_comparison.png")
save(outfile, fig, px_per_unit=2)
@info "Figure saved" outfile

# ── Summary statistics ──────────────────────────────────────────────────────

println("\n" * "="^85)
println("  IGW TEST: Advection-Limited Compressible Dynamics Comparison at t=3000s")
println("  Grid: Nx=$Nx, Nz=$Nz  |  Domain: $(Lx/1e3)km × $(Lz/1e3)km")
println("="^85)
@printf("  %-35s  max|θ'|: %.4e K   max|w|: %.4e m/s\n",
        @sprintf("Breeze explicit (Δt=%.1fs)", Δt_explicit),
        maximum(abs, r_explicit.θ′), maximum(abs, r_explicit.w))
@printf("  %-35s  max|θ'|: %.4e K   max|w|: %.4e m/s\n",
        @sprintf("Breeze SSP-RK3 (Δt=%.0fs, Ns=8)", Δt_ssprk3),
        maximum(abs, r_ssprk3.θ′), maximum(abs, r_ssprk3.w))
@printf("  %-35s  max|θ'|: %.4e K   max|w|: %.4e m/s\n",
        @sprintf("Breeze WS-RK3 (Δt=%.0fs, Ns=8)", Δt_wsrk3),
        maximum(abs, r_wsrk3.θ′), maximum(abs, r_wsrk3.w))
if have_cm1_explicit
    @printf("  %-35s  max|θ'|: %.4e K   max|w|: %.4e m/s\n",
            "CM1 explicit (Δt=1s)", maximum(abs, cm1_θ′_explicit), maximum(abs, cm1_w_explicit))
end
if have_cm1_substepped
    @printf("  %-35s  max|θ'|: %.4e K   max|w|: %.4e m/s\n",
            "CM1 substepped (Δt=12s, Ns=6)", maximum(abs, cm1_θ′_substepped), maximum(abs, cm1_w_substepped))
end
if have_erf
    @printf("  %-35s  max|θ'|: %.4e K   max|w|: %.4e m/s\n",
            "ERF substepped (Δt=12s, Ns=60)", maximum(abs, erf_θ′), maximum(abs, erf_w))
end
if have_wrf
    @printf("  %-35s  max|θ'|: %.4e K   max|w|: %.4e m/s\n",
            "WRF substepped (Δt=6s, Ns=6)", maximum(abs, wrf_θ′), maximum(abs, wrf_w))
end
println("="^85)
