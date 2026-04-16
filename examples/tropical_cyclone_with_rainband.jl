# # Tropical Cyclone with Spiral Rainband Heating
#
# Simulates a tropical cyclone with spiral rainband diabatic heating on an f-plane,
# using **fully compressible dynamics** (split-explicit acoustic substepping) and
# **one-moment mixed-phase microphysics** (prognostic cloud liquid, cloud ice, rain,
# and snow).
#
# Physical setup follows Moon & Nolan (2010) with the Yu & Didlake (2019) stratiform
# modification.  The vortex structure uses a modified Rankine profile with a
# height-dependent radius of maximum winds (Stern & Nolan, 2009).
#
# ## References
# - Moon, Y. and Nolan, D. S. (2010). J. Atmos. Sci., 67, 1779–1805.
# - Yu, C.-K. and Didlake, A. C. (2019). J. Atmos. Sci., 76, 3169–3189.
# - Stern, D. P. and Nolan, D. S. (2009). Mon. Wea. Rev., 137, 3825–3852.
# - Dunion, J. P. (2011). J. Atmos. Sci., 68, 1699–1726.

using Breeze
using Breeze: TetensFormula
using Breeze.Thermodynamics: vapor_gas_constant
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, VerticalScalarDiffusivity
using CairoMakie
using Printf
using CUDA

using CloudMicrophysics.Parameters: CloudIce
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

# ## Configuration

Oceananigans.defaults.FloatType = Float32
const FT = Float32
arch = GPU()

Nx = Ny = 750
Nz = 100
Δx = 1500.0               # horizontal resolution (m)
Lx = Ly = Δx * Nx         # 1125 km square domain
Lz = 20_000.0             # 20 km domain height
total_time = 6hours

# ## Dunion (2011) moist tropical sounding
#
# 14 standard levels, reversed to ascending height order.  The surface level is
# pinned to z = 0.  Embedding the data directly avoids CSV/DataFrames dependencies.

const z_snd = Float64[    0,   124,   810,  1541,  3178,  4437,  5887,  7596,  9690, 10949, 12418, 14203, 16590, 20726] # m
const p_snd = Float64[101480, 1e5, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000,  5000] # Pa
const T_snd = 273.15 .+ Float64[26.8, 26.5, 21.9, 17.6, 8.9, 1.6, -6.6, -17.1, -32.3, -42.3, -54.3, -67.2, -74.5, -63.0] # K
const θ_snd = Float64[298.8, 299.6, 301.7, 304.6, 312.3, 317.9, 325.0, 332.7, 339.8, 343.0, 346.6, 354.2, 383.5, 494.5] # K
const qᵗ_snd = Float64[18.65, 18.50, 15.27, 11.96, 6.74, 4.11, 2.41, 1.12, 0.34, 0.14, 0.04, 0.01, 0.01, 0.04] ./ 1000 # kg/kg

function linear_interp(zs, vs, z)
    z = clamp(z, zs[1], zs[end])
    i = clamp(searchsortedfirst(zs, z), 2, length(zs))
    t = (z - zs[i-1]) / (zs[i] - zs[i-1])
    return vs[i-1] + t * (vs[i] - vs[i-1])
end

θ_bg(z) = linear_interp(z_snd, θ_snd, z)
T_bg(z) = linear_interp(z_snd, T_snd, z)
p_bg(z) = linear_interp(z_snd, p_snd, z)
qᵗ_bg(z) = linear_interp(z_snd, qᵗ_snd, z)

# ## Grid

grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       x = (0, Lx), y = (0, Ly), z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

# ## Dynamics
#
# Fully compressible Euler equations with split-explicit acoustic substepping.
# The sounding θ profile is subtracted as a base state for numerical accuracy.

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p_snd[1],
                                reference_potential_temperature = θ_bg)

coriolis = FPlane(f = 5e-5)

# ## Surface fluxes (Emanuel, 1986)

Cᴰ    = 1.229e-3   # momentum drag
Cᵀ    = 1.094e-3   # sensible heat transfer
Cᵛ    = 1.133e-3   # moisture transfer
T_sfc = 300.0       # sea surface temperature (K)

ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ))
ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ))
ρe_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = Cᵀ, surface_temperature = T_sfc))
ρqᵗ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = Cᵛ, surface_temperature = T_sfc))

boundary_conditions = (ρu = ρu_bcs, ρv = ρv_bcs, ρe = ρe_bcs, ρqᵗ = ρqᵗ_bcs)

# ## Sponge layer (18–20 km)
#
# Quadratic Rayleigh damping on momentum and ρθ near the rigid lid.
# The ρθ target is filled from the model's far-field column after `set!`.

sponge_p = (z_start = 18_000f0, z_top = 20_000f0, rate = Float32(1/30))

@inline function sponge_ρu_fn(i, j, k, grid, clock, fields, p)
    z    = znode(i, j, k, grid, Face(), Center(), Center())
    frac = clamp((z - p.z_start) / (p.z_top - p.z_start), 0f0, 1f0)
    return -p.rate * frac * frac * @inbounds fields.ρu[i, j, k]
end

@inline function sponge_ρv_fn(i, j, k, grid, clock, fields, p)
    z    = znode(i, j, k, grid, Center(), Face(), Center())
    frac = clamp((z - p.z_start) / (p.z_top - p.z_start), 0f0, 1f0)
    return -p.rate * frac * frac * @inbounds fields.ρv[i, j, k]
end

@inline function sponge_ρw_fn(i, j, k, grid, clock, fields, p)
    z    = znode(i, j, k, grid, Center(), Center(), Face())
    frac = clamp((z - p.z_start) / (p.z_top - p.z_start), 0f0, 1f0)
    return -p.rate * frac * frac * @inbounds fields.ρw[i, j, k]
end

ρθ_target_cpu = zeros(FT, Nz)
ρθ_target_dev = arch isa GPU ? CuArray(ρθ_target_cpu) : ρθ_target_cpu

sponge_ρθ_p = (z_start = sponge_p.z_start, z_top = sponge_p.z_top,
               rate = sponge_p.rate, ρθ_bg = ρθ_target_dev)

@inline function sponge_ρθ_fn(i, j, k, grid, clock, fields, p)
    z    = znode(i, j, k, grid, Center(), Center(), Center())
    frac = clamp((z - p.z_start) / (p.z_top - p.z_start), 0f0, 1f0)
    mask = p.rate * frac * frac
    return -mask * (@inbounds fields.ρθ[i, j, k] - @inbounds p.ρθ_bg[k])
end

# ## Spiral rainband diabatic heating
#
# Convective component (Moon & Nolan, 2010): sinusoidal heating from 0–7 km.
# Stratiform component (Yu & Didlake, 2019): dipole heating/cooling centered at 6 km.
# Both are localized azimuthally to the lower-right quadrant.

x_center = Lx / 2
y_center = Ly / 2

Rᵈ  = dry_air_gas_constant(constants)
Rᵛ  = vapor_gas_constant(constants)
g   = constants.gravitational_acceleration
cₚ  = 1004.0
κ   = Rᵈ / cₚ
p₀  = 1e5

T_mean = sum(T_snd) / length(T_snd)
ρ_sfc  = p_snd[1] / (Rᵈ * T_snd[1])
H_ρ    = Rᵈ * T_mean / g

r_rb = 70_000.0    # rainband radial centre (m)

con_p = (xc = FT(x_center), yc = FT(y_center),
         Q_max = FT(3.0 / 3600), r_rb = FT(r_rb), σ_r = FT(4000),
         z_bot = 0f0, z_top = 7000f0,
         ϕ_rb = FT(-π/4), σ_ϕ = FT(π/4),
         ρ_sfc = FT(ρ_sfc), H_ρ = FT(H_ρ))

str_p = (xc = FT(x_center), yc = FT(y_center),
         Q_max = FT(4.24 / 3600), r_rb = FT(r_rb), σ_r = FT(8000),
         z_bs = 6000f0, σ_zs = 2000f0,
         ϕ_rb = FT(-π/4), σ_ϕ = FT(π/4),
         ρ_sfc = FT(ρ_sfc), H_ρ = FT(H_ρ))

@inline function convective_heating(x, y, z, t, p)
    r = sqrt((x - p.xc)^2 + (y - p.yc)^2)
    ϕ = atan(y - p.yc, x - p.xc)
    ρ = p.ρ_sfc * exp(-z / p.H_ρ)
    azimuthal = exp(-((ϕ - p.ϕ_rb) / p.σ_ϕ)^8)
    radial    = exp(-((r - p.r_rb) / p.σ_r)^2)
    in_layer  = (z > p.z_bot) & (z < p.z_top)
    vertical  = ifelse(in_layer, sin(oftype(z, π) * (z - p.z_bot) / (p.z_top - p.z_bot)), zero(z))
    return ρ * p.Q_max * radial * azimuthal * vertical
end

@inline function stratiform_heating(x, y, z, t, p)
    r = sqrt((x - p.xc)^2 + (y - p.yc)^2)
    ϕ = atan(y - p.yc, x - p.xc)
    ρ = p.ρ_sfc * exp(-z / p.H_ρ)
    azimuthal = exp(-((ϕ - p.ϕ_rb) / p.σ_ϕ)^8)
    radial    = exp(-((r - p.r_rb) / p.σ_r)^2)
    in_layer  = (z > p.z_bs - p.σ_zs) & (z < p.z_bs + p.σ_zs)
    vertical  = ifelse(in_layer, sin(oftype(z, π) * (z - p.z_bs) / p.σ_zs), zero(z))
    return ρ * p.Q_max * radial * azimuthal * vertical
end

forcing = (ρw = Forcing(sponge_ρw_fn, discrete_form = true, parameters = sponge_p),
           ρu = Forcing(sponge_ρu_fn, discrete_form = true, parameters = sponge_p),
           ρv = Forcing(sponge_ρv_fn, discrete_form = true, parameters = sponge_p),
           ρθ = (Forcing(sponge_ρθ_fn, discrete_form = true, parameters = sponge_ρθ_p),
                 Forcing(convective_heating, parameters = con_p),
                 Forcing(stratiform_heating, parameters = str_p)))

# ## Microphysics and model
#
# One-moment mixed-phase non-equilibrium microphysics: prognostic cloud liquid (ρqᶜˡ),
# cloud ice (ρqᶜⁱ), rain (ρqʳ), and snow (ρqˢ).

cloud_formation = NonEquilibriumCloudFormation(nothing, CloudIce(FT))
microphysics    = OneMomentCloudMicrophysics(FT; cloud_formation)

weno    = WENO(order = 5)
bp_weno = WENO(order = 5, bounds = (0, 1))
scalar_advection = (ρθ = weno, ρqᵗ = bp_weno,
                    ρqᶜˡ = bp_weno, ρqᶜⁱ = bp_weno,
                    ρqʳ = bp_weno, ρqˢ = bp_weno)

vitd    = VerticallyImplicitTimeDiscretization()
closure = VerticalScalarDiffusivity(vitd; ν = 25, κ = 25)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics,
                        momentum_advection = weno, scalar_advection,
                        boundary_conditions, closure, forcing,
                        thermodynamic_constants = constants)

# ## Vortex initial conditions
#
# Modified Rankine profile with height-dependent radius of maximum winds
# (Stern & Nolan, 2009 Eq. 4.4).  Gradient-wind balance determines the
# pressure field; hydrostatic balance then gives a consistent warm-core θ.

RMW       = 31_000.0    # surface radius of maximum winds (m)
V_RMW     = 43.0        # peak tangential wind (m/s)
α_decay   = 0.5         # outer wind decay exponent
z_taper₀  = 13_000.0    # height where wind begins to taper
z_taper₁  = 16_000.0    # height where wind reaches zero
r_zero    = Lx / 2      # outer relaxation radius

z_cpu = Array(znodes(grid, Center()))
Δz    = z_cpu[2] - z_cpu[1]

Rᵐ_col = map(z_cpu) do z
    qᵛ = qᵗ_bg(clamp(z, z_snd[1], z_snd[end]))
    (1 - qᵛ) * Rᵈ + qᵛ * Rᵛ
end

# Height-dependent RMW
T_outflow = T_bg(z_taper₁)
rmw_prof  = zeros(Float64, Nz)
rmw_prof[1] = RMW
for k in 2:Nz
    z_k  = z_cpu[k]
    z_lo = clamp(z_k - Δz/2, z_snd[1], z_snd[end])
    z_hi = clamp(z_k + Δz/2, z_snd[1], z_snd[end])
    dTdz = (T_bg(z_hi) - T_bg(z_lo)) / Δz
    T_k  = T_bg(clamp(z_k, z_snd[1], z_snd[end]))
    denom = 2.0 * (T_k - T_outflow)
    drdz = abs(denom) > 1.0 ? -rmw_prof[k-1] / denom * dTdz : 0.0
    rmw_prof[k] = max(rmw_prof[k-1] + drdz * Δz, RMW)
end

rmw_at(z) = rmw_prof[clamp(searchsortedfirst(z_cpu, z), 1, Nz)]

function tangential_wind(x, y, z)
    r     = sqrt((x - x_center)^2 + (y - y_center)^2)
    rmw_z = rmw_at(z)
    v_adj = RMW / rmw_z
    z >= z_taper₁ && return zero(typeof(r))

    if r <= rmw_z
        vt = V_RMW * v_adj * r / rmw_z
    elseif r >= r_zero
        return zero(typeof(r))
    else
        vt = V_RMW * v_adj * (rmw_z / r)^α_decay * (r_zero - r) / (r_zero - rmw_z)
    end

    if z > z_taper₀
        vt *= 0.5 * (1 + cos(π * (z - z_taper₀) / (z_taper₁ - z_taper₀)))
    end
    return vt
end

# Gradient-wind pressure integration: dp/dr = ρ(fv + v²/r), inward from 1500 km.
∂r_int     = 1_000.0
max_radius = 1_500_000.0
rrange     = collect(0.0:∂r_int:max_radius)
Nr         = length(rrange)
f₀         = coriolis.f

p_vortex = zeros(Float64, Nz, Nr)
for k in 1:Nz
    z_c  = clamp(z_cpu[k], z_snd[1], z_snd[end])
    p_bk = p_bg(z_c)
    ρ_k  = p_bk / (Rᵐ_col[k] * T_bg(z_c))
    p_vortex[k, Nr] = p_bk
    for ri in (Nr-1):-1:1
        r    = rrange[ri]
        v    = tangential_wind(x_center + r, y_center, z_cpu[k])
        dp   = ρ_k * (v * f₀ + v^2 / max(r, 1.0))
        p_vortex[k, ri] = p_vortex[k, ri+1] - dp * ∂r_int
    end
end
p_outer = p_vortex[:, Nr]

# Warm-core θ from hydrostatic balance of the perturbed pressure field.
θ_hydro = zeros(Float64, Nz, Nr)
for ri in 1:Nr, k in 1:Nz
    ρ_h = if k == 1
        -(p_vortex[2, ri] - p_vortex[1, ri]) / (g * Δz)
    elseif k == Nz
        -(p_vortex[Nz, ri] - p_vortex[Nz-1, ri]) / (g * Δz)
    else
        -(p_vortex[k+1, ri] - p_vortex[k-1, ri]) / (2g * Δz)
    end
    ρ_h = max(ρ_h, 1e-3)
    T_loc = p_vortex[k, ri] / (Rᵐ_col[k] * ρ_h)
    θ_hydro[k, ri] = T_loc * (p₀ / p_vortex[k, ri])^κ
end

θ_anom = θ_hydro .- θ_hydro[:, Nr:Nr]

println("  Warm-core anomaly: max Δθ = $(round(maximum(θ_anom), digits=2)) K")

# Pressure at the RMW for secondary circulation
p_at_RMW = zeros(Float64, Nz)
for k in 1:Nz
    ri = clamp(searchsortedfirst(rrange, rmw_prof[k]), 2, Nr)
    tr = clamp((rmw_prof[k] - rrange[ri-1]) / (rrange[ri] - rrange[ri-1]), 0.0, 1.0)
    p_at_RMW[k] = (1 - tr) * p_vortex[k, ri-1] + tr * p_vortex[k, ri]
end

# Bernoulli secondary circulation: boundary-layer inflow + upper-level outflow.
function radial_wind(x, y, z)
    z >= z_taper₁ && return 0.0
    r = sqrt((x - x_center)^2 + (y - y_center)^2)
    k = clamp(searchsortedfirst(z_cpu, z), 1, Nz)
    rmw_z = rmw_prof[k]
    r <= rmw_z && return 0.0

    ri  = clamp(searchsortedfirst(rrange, r), 2, Nr)
    tr  = clamp((r - rrange[ri-1]) / (rrange[ri] - rrange[ri-1]), 0.0, 1.0)
    p_r = (1 - tr) * p_vortex[k, ri-1] + tr * p_vortex[k, ri]
    dp  = p_r - p_at_RMW[k]
    dp <= 0 && return 0.0

    z_c   = clamp(z, z_snd[1], z_snd[end])
    ρ_k   = p_r / (Rᵐ_col[k] * T_bg(z_c))
    speed = sqrt(2 * dp / ρ_k)

    taper_r = clamp((r_zero - r) / (r_zero - rmw_z), 0.0, 1.0)
    taper_z = z > z_taper₀ ? 0.5 * (1 + cos(π * (z - z_taper₀) / (z_taper₁ - z_taper₀))) : 1.0
    sign_z  = -cos(π * clamp(z, 0.0, z_taper₁) / z_taper₁)

    return 0.01 * sign_z * speed * taper_r * taper_z
end

# Bilinear interpolation of θ anomaly from the (k, r) grid.
function θ_at(x, y, z)
    r    = sqrt((x - x_center)^2 + (y - y_center)^2)
    z_c  = clamp(z, z_snd[1], z_snd[end])
    θ_b  = θ_bg(z_c)

    ki = clamp(searchsortedfirst(z_cpu, z), 2, Nz)
    tz = clamp((z - z_cpu[ki-1]) / (z_cpu[ki] - z_cpu[ki-1]), 0.0, 1.0)
    ri = clamp(searchsortedfirst(rrange, r), 2, Nr)
    tr = clamp((r - rrange[ri-1]) / (rrange[ri] - rrange[ri-1]), 0.0, 1.0)

    δθ = (1-tz) * ((1-tr) * θ_anom[ki-1, ri-1] + tr * θ_anom[ki-1, ri]) +
              tz * ((1-tr) * θ_anom[ki,   ri-1] + tr * θ_anom[ki,   ri])
    return θ_b + δθ
end

# Density from gradient-wind pressure: ρ = p / (Rᵐ T)
function ρ_init(x, y, z)
    r   = sqrt((x - x_center)^2 + (y - y_center)^2)
    k   = clamp(searchsortedfirst(z_cpu, z), 1, Nz)
    ri  = clamp(searchsortedfirst(rrange, r), 1, Nr)
    z_c = clamp(z, z_snd[1], z_snd[end])
    return p_vortex[k, ri] / (Rᵐ_col[k] * T_bg(z_c))
end

u_init(x, y, z) = let ϕ = atan(y - y_center, x - x_center)
    -sin(ϕ) * tangential_wind(x, y, z) + cos(ϕ) * radial_wind(x, y, z)
end

v_init(x, y, z) = let ϕ = atan(y - y_center, x - x_center)
     cos(ϕ) * tangential_wind(x, y, z) + sin(ϕ) * radial_wind(x, y, z)
end

θ_init(x, y, z) = θ_at(x, y, z)
qᵗ_init(x, y, z) = qᵗ_bg(clamp(z, z_snd[1], z_snd[end]))

# ## Set initial conditions

println("Setting initial conditions...")
@time set!(model, ρ = ρ_init, θ = θ_init, u = u_init, v = v_init, qᵗ = qᵗ_init)

# Fill the sponge ρθ target from the far-field column (i=1, j=1 is ~500 km from centre).
copyto!(ρθ_target_cpu, Array(interior(model.formulation.potential_temperature_density, 1, 1, :)))
arch isa GPU && copyto!(ρθ_target_dev, ρθ_target_cpu)

u_arr = Array(interior(model.velocities.u))
println("  max |u| = $(round(maximum(abs.(u_arr)), digits=1)) m/s  (expected ≈ $V_RMW)")

# ## Simulation

simulation = Simulation(model; Δt = 8, stop_time = total_time)
conjure_time_step_wizard!(simulation, cfl = 0.7)

u, v, w = model.velocities
θˡⁱ     = liquid_ice_potential_temperature(model)
qᶜˡ     = model.microphysical_fields.qᶜˡ
qᶜⁱ     = model.microphysical_fields.qᶜⁱ
qʳ      = model.microphysical_fields.qʳ
qˢ      = model.microphysical_fields.qˢ

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    wmax = maximum(w)
    umax = maximum(abs, u)
    @info @sprintf("Iter %d  t=%s  Δt=%s  wall=%s  max|u|=%.1f  w=[%.1f,%.1f]  qᶜˡ=%.1e  qʳ=%.1e  qᶜⁱ=%.1e  qˢ=%.1e",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                   umax, minimum(w), wmax,
                   maximum(qᶜˡ), maximum(qʳ), maximum(qᶜⁱ), maximum(qˢ))

    if isnan(wmax) || isnan(umax) || wmax > 60
        @warn "NaN or blowup detected — stopping"
        sim.stop_iteration = iteration(sim)
    end
    wall_clock[] = time_ns()
end

add_callback!(simulation, progress, IterationInterval(25))

max_w_times = Float64[]
max_w_vals  = Float64[]
function collect_max_w(sim)
    push!(max_w_times, time(sim))
    push!(max_w_vals,  maximum(w))
end
add_callback!(simulation, collect_max_w, TimeInterval(1minutes))

# ## Output

z_out = znodes(grid, Center())
k_5km = searchsortedfirst(z_out, 5000)
j_mid = Ny ÷ 2

slice_outputs = (wxy   = view(w,   :, :, k_5km),
                 qᶜˡxy = view(qᶜˡ, :, :, k_5km),
                 qʳxy  = view(qʳ,  :, :, k_5km),
                 uxy   = view(u,   :, :, k_5km),
                 vxy   = view(v,   :, :, k_5km),
                 θxy   = view(θˡⁱ, :, :, k_5km),
                 wxz   = view(w,   :, j_mid, :),
                 qᶜˡxz = view(qᶜˡ, :, j_mid, :),
                 qʳxz  = view(qʳ,  :, j_mid, :))

output_file = "tropical_cyclone_rainband.jld2"

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
    filename = output_file,
    schedule = TimeInterval(2minutes),
    overwrite_existing = true)

# ## Run

@info "Starting simulation" total_time
run!(simulation)

# ## Post-processing
#
# Load saved slices and produce an animated 6-panel figure:
# top row (xy at z ≈ 5 km): vertical velocity, cloud condensate, precipitation;
# bottom row (xz at y = centre): same fields.

wxy_ts   = FieldTimeSeries(output_file, "wxy")
qᶜˡxy_ts = FieldTimeSeries(output_file, "qᶜˡxy")
qʳxy_ts  = FieldTimeSeries(output_file, "qʳxy")
uxy_ts   = FieldTimeSeries(output_file, "uxy")
vxy_ts   = FieldTimeSeries(output_file, "vxy")
wxz_ts   = FieldTimeSeries(output_file, "wxz")
qᶜˡxz_ts = FieldTimeSeries(output_file, "qᶜˡxz")
qʳxz_ts  = FieldTimeSeries(output_file, "qʳxz")

times = wxy_ts.times
Nt    = length(times)

x_km = xnodes(grid, Center()) ./ 1000
y_km = ynodes(grid, Center()) ./ 1000
z_km = znodes(grid, Center()) ./ 1000

wlim   = 10.0
qᶜˡlim = max(Float64(maximum(qᶜˡxy_ts)), Float64(maximum(qᶜˡxz_ts)), 1e-6) / 4
qʳlim  = max(Float64(maximum(qʳxy_ts)),  Float64(maximum(qʳxz_ts)),  1e-6) / 4

_finite(A) = ifelse.(isfinite.(A), A, zero(eltype(A)))

fig = Figure(size = (1500, 800), fontsize = 12)
n   = Observable(1)
fig[0, :] = Label(fig,
    @lift("TC + rainband  —  t = " * prettytime(times[$n])),
    fontsize = 14, tellwidth = false)

axw_xy  = Axis(fig[1,1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="w (m/s)")
axcl_xy = Axis(fig[1,2], aspect=1, xlabel="x (km)", ylabel="y (km)", title="qᶜˡ (kg/kg)")
axr_xy  = Axis(fig[1,3], aspect=1, xlabel="x (km)", ylabel="y (km)", title="qʳ (kg/kg)")

axw_xz  = Axis(fig[2,1], xlabel="x (km)", ylabel="z (km)", title="w (m/s)")
axcl_xz = Axis(fig[2,2], xlabel="x (km)", ylabel="z (km)", title="qᶜˡ (kg/kg)")
axr_xz  = Axis(fig[2,3], xlabel="x (km)", ylabel="z (km)", title="qʳ (kg/kg)")

wxy_n   = @lift _finite(Array(interior(wxy_ts[$n],   :, :, 1)))
qᶜˡxy_n = @lift _finite(Array(interior(qᶜˡxy_ts[$n], :, :, 1)))
qʳxy_n  = @lift _finite(Array(interior(qʳxy_ts[$n],  :, :, 1)))

wxz_n   = @lift _finite(Array(interior(wxz_ts[$n],   :, 1, :)))
qᶜˡxz_n = @lift _finite(Array(interior(qᶜˡxz_ts[$n], :, 1, :)))
qʳxz_n  = @lift _finite(Array(interior(qʳxz_ts[$n],  :, 1, :)))

hmw  = heatmap!(axw_xy,  x_km, y_km, wxy_n;   colormap=:balance, colorrange=(-wlim, wlim))
hmcl = heatmap!(axcl_xy, x_km, y_km, qᶜˡxy_n; colormap=:dense,   colorrange=(0, qᶜˡlim))
hmr  = heatmap!(axr_xy,  x_km, y_km, qʳxy_n;  colormap=:amp,     colorrange=(0, qʳlim))

heatmap!(axw_xz,  x_km, z_km, wxz_n;   colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axcl_xz, x_km, z_km, qᶜˡxz_n; colormap=:dense,   colorrange=(0, qᶜˡlim))
heatmap!(axr_xz,  x_km, z_km, qʳxz_n;  colormap=:amp,     colorrange=(0, qʳlim))

Colorbar(fig[3, 1], hmw,  vertical=false, label="w (m/s)")
Colorbar(fig[3, 2], hmcl, vertical=false, label="qᶜˡ (kg/kg)")
Colorbar(fig[3, 3], hmr,  vertical=false, label="qʳ (kg/kg)")

video_path = "tropical_cyclone_rainband.mp4"
Makie.record(fig, video_path, 1:Nt; framerate = 10) do t_idx
    n[] = t_idx
end

n[] = Nt
Makie.save("tropical_cyclone_rainband_final.png", fig)

# Max vertical velocity time series
fig_ts = Figure(size = (600, 340))
ax_ts  = Axis(fig_ts[1, 1], xlabel = "Time (h)", ylabel = "max w (m/s)",
              title = "Maximum Vertical Velocity")
lines!(ax_ts, max_w_times ./ 3600, max_w_vals)
Makie.save("max_w_timeseries.png", fig_ts)

println("\nDone.  Outputs:")
println("  $video_path")
println("  tropical_cyclone_rainband_final.png")
println("  max_w_timeseries.png")
