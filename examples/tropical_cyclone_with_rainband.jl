# # Tropical cyclone with spiral rainband heating
#
# This example simulates a tropical cyclone with spiral rainband diabatic heating
# following Moon and Nolan (2010). The vortex structure uses the modified Rankine
# profile with a height-dependent radius of maximum winds from Stern and Nolan (2009).
# Surface fluxes use bulk aerodynamic coefficients from Emanuel (1986).
#
# An optional modification following Yu and Didlake (2019) changes the stratiform
# rainband heating profile: the heating centre is raised to 6 km, the amplitude is
# increased, and the radial extent is widened.
#
# ## References
# - Moon, Y. and Nolan, D. S. (2010). The dynamic response of the hurricane wind
#   field to spiral rainband heating. J. Atmos. Sci., 67, 1779-1805.
# - Stern, D. P. and Nolan, D. S. (2009). Reexamining the vertical structure of
#   tangential winds in tropical cyclones. Mon. Wea. Rev., 137, 3825-3852.
# - Emanuel, K. A. (1986). An air-sea interaction theory for tropical cyclones.
#   Part I: Steady-state maintenance. J. Atmos. Sci., 43, 585-604.
# - Yu, C.-K. and Didlake, A. C. (2019). Impact of stratiform rainband heating
#   on the tropical cyclone wind field in idealized simulations. J. Atmos. Sci.,
#   76, 3169-3189.

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Breeze: WENO, DCMIP2016KesslerMicrophysics, TetensFormula
using CairoMakie
using Printf
using Random
using CUDA
using CSV
using DataFrames

Random.seed!(42)
Random.TaskLocalRNG()

###########################
# Configuration flags
###########################

# Set to true to use Yu and Didlake (2019) stratiform rainband modifications:
#   - Higher heating centre (z_bs = 6 km vs 4 km in Moon and Nolan 2010)
#   - Stronger maximum heating amplitude (Q_str_max = 4.24 K/h vs 1.5 K/h)
#   - Broader radial half-width (σ_rs = 8 km vs 6 km)
const use_yu_didlake_2019 = false

###########################
# Domain and grid
###########################

Oceananigans.defaults.FloatType = Float32

arch = GPU()
Nx = Ny = 100
Nz = 128

x = y = (0, 2000Nx)    # 200 km × 200 km domain; 2 km horizontal resolution
z = (0, 20500)          # 0–20.5 km

grid = RectilinearGrid(arch; x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

###########################
# Sounding data  (Dunion 2011 moist tropical)
###########################

println("\n=== Loading Dunion 2011 moist tropical sounding ===")
sounding    = CSV.read("dunion2011_moist_tropical_MT.csv", DataFrame)

# Raw columns from the CSV
Tˢ_data_raw  = 273.15 .+ Float64.(sounding[:, :Temperature_C])   # K
pˢ_data_raw  = 100.0  .* Float64.(sounding[:, :Pressure_hPa])    # Pa
zˢ_data_raw  = Float64.(sounding[:, :GPH_m])                      # m
θˢ_data_raw  = Float64.(sounding[:, :Theta_K])                    # K
# Mixing_ratio_g_kg stores values in g/kg (e.g. 18.65 at surface)
qᵗˢ_gkg_raw  = Float64.(sounding[:, :Mixing_ratio_g_kg])         # g/kg

# CSV is ordered top-to-bottom (highest level first); reverse to ascending-z order
Tˢ_data     = reverse(Tˢ_data_raw)
pˢ_data     = reverse(pˢ_data_raw)
zˢ_data     = reverse(zˢ_data_raw)
θˢ_data     = reverse(θˢ_data_raw)
qᵗˢ_gkg     = reverse(qᵗˢ_gkg_raw)
qᵗˢ_data    = qᵗˢ_gkg ./ 1000.0      # convert g/kg → kg/kg for model initialization

# Force the lowest level to exactly z = 0 m (ground)
zˢ_data[1] = 0.0

println("  $(length(zˢ_data)) levels: z ∈ [$(zˢ_data[1]) m, $(zˢ_data[end]) m]")
println("  Surface: θ = $(round(θˢ_data[1], digits=1)) K, " *
        "qᵗ = $(round(qᵗˢ_data[1]*1000, digits=2)) g/kg, " *
        "T = $(round(Tˢ_data[1]-273.15, digits=1)) °C, " *
        "p = $(round(pˢ_data[1]/100, digits=1)) hPa")

# ---- Piecewise-linear column interpolation (no external packages) ----
# Returns a closure z → linearly interpolated value, clamped to [zs[1], zs[end]].
function make_column_interp(zs::AbstractVector, vs::AbstractVector)
    function interp(z)
        z = clamp(z, zs[1], zs[end])
        i = clamp(searchsortedfirst(zs, z), 2, length(zs))
        t = (z - zs[i-1]) / (zs[i] - zs[i-1])
        return vs[i-1] + t * (vs[i] - vs[i-1])
    end
    return interp
end

# ---- Continuous interpolants (used for IC and pressure integration) ----

θ_sounding_interp  = make_column_interp(zˢ_data, θˢ_data)
qᵗ_sounding_interp = make_column_interp(zˢ_data, qᵗˢ_data)
T_sounding_interp  = make_column_interp(zˢ_data, Tˢ_data)
p_sounding_interp  = make_column_interp(zˢ_data, pˢ_data)

# ---- Sounding Fields (for Oceananigans-style sounding plots) ----
# Build a 1-D vertical grid whose cell centres sit on the sounding levels.
# Face nodes are at the midpoints between successive sounding levels plus
# one extra face at top and bottom.
N_snd  = length(zˢ_data)                                 # number of sounding levels
zˢᶠ_lo = [0.0]                                           # bottom face at z = 0
zˢᶠ_mid = (zˢ_data[1:end-1] .+ zˢ_data[2:end]) ./ 2   # N_snd - 1 interior faces
Δz_top  = zˢ_data[end] - zˢ_data[end-1]
zˢᶠ_hi  = [zˢ_data[end] + Δz_top / 2]                  # top face
zˢᶠ = vcat(zˢᶠ_lo, zˢᶠ_mid, zˢᶠ_hi)                   # N_snd + 1 faces → N_snd cells
sounding_grid = RectilinearGrid(size=N_snd, z=zˢᶠ, topology=(Flat, Flat, Bounded))

θˢ_bcs = FieldBoundaryConditions(sounding_grid, (Center(), Center(), Center()),
                                  bottom=ValueBoundaryCondition(θˢ_data[1]))
Tˢ_bcs = FieldBoundaryConditions(sounding_grid, (Center(), Center(), Center()),
                                  bottom=ValueBoundaryCondition(Tˢ_data[1]))
θˢ_fld = CenterField(sounding_grid, boundary_conditions=θˢ_bcs)
Tˢ_fld = CenterField(sounding_grid, boundary_conditions=Tˢ_bcs)
qᵗˢ_fld = CenterField(sounding_grid)
set!(θˢ_fld,  θˢ_data)
set!(Tˢ_fld,  Tˢ_data)
set!(qᵗˢ_fld, qᵗˢ_gkg)   # store in g/kg for the plot axes

## ---- Sounding plot ----
fig = Figure(size=(1000, 500))
axθ = Axis(fig[1, 1], xlabel="θ (K)",   ylabel="Height (m)", title="Dunion 2011: θ")
axT = Axis(fig[1, 2], xlabel="T (°C)",  ylabel="Height (m)", title="Dunion 2011: T")
axq = Axis(fig[1, 3], xlabel="qᵗ (g/kg)", ylabel="Height (m)", title="Dunion 2011: qᵗ")
axp = Axis(fig[1, 4], xlabel="p (hPa)", ylabel="Height (m)", title="Dunion 2011: p")
lines!(axθ, θˢ_data, zˢ_data)
lines!(axT, Tˢ_data .- 273.15, zˢ_data)
lines!(axq, qᵗˢ_gkg, zˢ_data)
lines!(axp, pˢ_data ./ 100, zˢ_data)
Makie.save("dunion2011_sounding.png", fig)
println("  Saved dunion2011_sounding.png")

###########################
# Reference state
###########################

reference_state = ReferenceState(grid, constants,
                                 surface_pressure     = pˢ_data[1],
                                 potential_temperature = θˢ_data[1])

dynamics = AnelasticDynamics(reference_state)

###########################
# Surface fluxes  (bulk aerodynamic; coefficients from Emanuel 1986)
###########################

Cᴰ = 1.229e-3   # momentum drag coefficient
Cᵀ = 1.094e-3   # sensible heat transfer coefficient
Cᵛ = 1.133e-3   # moisture flux transfer coefficient
T₀ = 300.0      # sea surface temperature (K)

ρe_flux  = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=T₀)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=T₀)

ρe_bcs  = FieldBoundaryConditions(bottom=ρe_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)
ρu_bcs  = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs  = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

boundary_conditions = (ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)

###########################
# Sponge layer  (upper atmosphere — damps spurious reflections near the model top)
###########################

# Gaussian mask centred at 19 km absorbs waves before they reach the rigid lid at 20.5 km.
sponge_rate = 1 / 300.0                                         # s⁻¹  (300 s ≈ 5 min)
sponge_mask = GaussianMask{:z}(center=19_000.0, width=1_500.0)
sponge      = Relaxation(rate=sponge_rate, mask=sponge_mask)

###########################
# Coriolis
###########################

coriolis = FPlane(f=5e-5)    # f-plane at ~20 °N

###########################
# Rainband diabatic heating  (Moon and Nolan 2010 / Yu and Didlake 2019)
###########################
#
# Convective component (MN10):
#   Q_con = Q_con_max · exp[-(r-r_rb)²/σ_rc²] · sin[π(z-z_bc)/σ_zc]  for z_bc < z < z_bc+σ_zc
#   Q_con_max = 3.0 K/h,  σ_rc = 2 km,  z_bc = 0,  σ_zc = 7 km
#
# Stratiform component (MN10):
#   Q_str = Q_str_max · exp[-(r-r_rb)²/σ_rs²] · sin[π(z-z_bs)/σ_zs]  for |z-z_bs| < σ_zs
#   Negative (cooling) below z_bs, positive (heating) above z_bs.
#   Q_str_max = 1.5 K/h,  σ_rs = 6 km,  z_bs = 4 km,  σ_zs = 2 km
#
# Yu and Didlake (2019) modification (use_yu_didlake_2019 = true):
#   Q_str_max = 4.24 K/h,  σ_rs = 8 km,  z_bs = 6 km
#
# Both components are azimuthally localised to the lower-right quadrant using a
# super-Gaussian mask (ϕ ≈ -π/4 from east, half-width π/4).
#
# The tendency for ρθ [kg m⁻³ K s⁻¹] = Q [K/s] × ρ_ref(z).
# An exponential reference density ρ = ρ_sfc·exp(-z/H_ρ) is used inside the
# GPU-compatible forcing kernels.

x_center = Float64(x[1] + x[2]) / 2
y_center = Float64(y[1] + y[2]) / 2

# Reference density for forcing kernels (exponential atmosphere)
Rᵈ    = constants.molar_gas_constant / constants.dry_air.molar_mass  # J kg⁻¹ K⁻¹
T_avg = sum(Tˢ_data) / length(Tˢ_data)                               # mean sounding temperature
ρ_sfc = pˢ_data[1] / (Rᵈ * Tˢ_data[1])                             # surface density  (kg m⁻³)
H_ρ   = Rᵈ * T_avg / constants.gravitational_acceleration            # density scale height (m)

# Rainband radial centre (70 km: midpoint of the 60–80 km range in MN10 Fig. 1)
r_rb = 70_000.0    # m

# ---- Convective parameters ----
Q_con_max = 3.0 / 3600.0   # K s⁻¹  (3.0 K h⁻¹)
σ_rc      = 2_000.0        # radial half-width (m)
z_bc      = 0.0            # base height of convective heating (m)
σ_zc      = 7_000.0        # vertical depth of convective heating (m)

# ---- Stratiform parameters (MN10 or YD19) ----
Q_str_max = use_yu_didlake_2019 ? 4.24 / 3600.0 : 1.5 / 3600.0   # K s⁻¹
σ_rs      = use_yu_didlake_2019 ? 8_000.0 : 6_000.0               # m
z_bs      = use_yu_didlake_2019 ? 6_000.0 : 4_000.0               # melting level (m)
σ_zs      = 2_000.0        # vertical half-width (m)

# ---- Azimuthal localisation ----
ϕ_rb = Float32(-π/4)       # lower-right quadrant (−45° from east)
σ_ϕ  = Float32(π/4)        # super-Gaussian half-width

println("\n=== Rainband configuration ===")
println("  Profile: $(use_yu_didlake_2019 ? "Yu & Didlake (2019)" : "Moon & Nolan (2010)")")
println("  Rainband radius: $(r_rb/1000) km;  azimuth: $(round(rad2deg(ϕ_rb), digits=0))°")
println("  Convective: Q_max = $(Q_con_max*3600) K/h,  σ_r = $(σ_rc/1000) km,  z ∈ [$(z_bc/1000), $((z_bc+σ_zc)/1000)] km")
println("  Stratiform: Q_max = $(Q_str_max*3600) K/h,  σ_r = $(σ_rs/1000) km,  z_bs = $(z_bs/1000) km ± $(σ_zs/1000) km")

# Pack parameters into NamedTuples (all Float32 — GPU-compatible)
con_params = (xc    = Float32(x_center),
              yc    = Float32(y_center),
              Q_max = Float32(Q_con_max),
              r_rb  = Float32(r_rb),
              σ_r   = Float32(σ_rc),
              z_bot = Float32(z_bc),
              z_top = Float32(z_bc + σ_zc),
              ϕ_rb,
              σ_ϕ,
              ρ_sfc = Float32(ρ_sfc),
              H_ρ   = Float32(H_ρ))

str_params = (xc    = Float32(x_center),
              yc    = Float32(y_center),
              Q_max = Float32(Q_str_max),
              r_rb  = Float32(r_rb),
              σ_r   = Float32(σ_rs),
              z_bs  = Float32(z_bs),
              σ_zs  = Float32(σ_zs),
              ϕ_rb,
              σ_ϕ,
              ρ_sfc = Float32(ρ_sfc),
              H_ρ   = Float32(H_ρ))

# GPU-compatible forcing kernels — return  d(ρθ)/dt  [kg m⁻³ K s⁻¹]

@inline function convective_rainband_heating(x, y, z, t, p)
    r  = sqrt((x - p.xc)^2 + (y - p.yc)^2)
    ϕ  = atan(y - p.yc, x - p.xc)
    ρ  = p.ρ_sfc * exp(-z / p.H_ρ)
    azimuthal  = exp(-((ϕ - p.ϕ_rb) / p.σ_ϕ)^8)     # super-Gaussian azimuthal mask
    radial     = exp(-((r - p.r_rb)  / p.σ_r)^2)
    in_layer   = (z > p.z_bot) & (z < p.z_top)
    sinusoidal = sin(oftype(z, π) * (z - p.z_bot) / (p.z_top - p.z_bot))
    vertical   = ifelse(in_layer, sinusoidal, zero(z))
    return ρ * p.Q_max * radial * azimuthal * vertical
end

@inline function stratiform_rainband_heating(x, y, z, t, p)
    r  = sqrt((x - p.xc)^2 + (y - p.yc)^2)
    ϕ  = atan(y - p.yc, x - p.xc)
    ρ  = p.ρ_sfc * exp(-z / p.H_ρ)
    azimuthal  = exp(-((ϕ - p.ϕ_rb) / p.σ_ϕ)^8)
    radial     = exp(-((r - p.r_rb)  / p.σ_r)^2)
    # Full-sine dipole: cooling below z_bs, heating above z_bs (MN10 Fig. 5)
    in_layer   = (z > p.z_bs - p.σ_zs) & (z < p.z_bs + p.σ_zs)
    sinusoidal = sin(oftype(z, π) * (z - p.z_bs) / p.σ_zs)
    vertical   = ifelse(in_layer, sinusoidal, zero(z))
    return ρ * p.Q_max * radial * azimuthal * vertical
end

convective_forcing = Forcing(convective_rainband_heating, parameters=con_params)
stratiform_forcing = Forcing(stratiform_rainband_heating, parameters=str_params)

forcing = (ρθ = (convective_forcing, stratiform_forcing),
           ρw  = sponge)

###########################
# Model
###########################

println("\n=== Creating AtmosphereModel ===")

microphysics           = DCMIP2016KesslerMicrophysics()
weno                   = WENO(order=5)
bounds_preserving_weno = WENO(order=5, bounds=(0, 1))
momentum_advection     = weno
scalar_advection       = (ρθ   = weno,
                          ρqᵗ  = bounds_preserving_weno,
                          ρqᶜˡ = bounds_preserving_weno,
                          ρqʳ  = bounds_preserving_weno)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics,
                        momentum_advection, scalar_advection,
                        forcing, boundary_conditions,
                        thermodynamic_constants=constants)

###########################
# Vortex structure  (Stern and Nolan 2009 / Moon and Nolan 2010)
###########################

# Modified Rankine tangential wind profile with height-dependent RMW.
# Inside  r ≤ RMW(z):  V = V_RMW × (V_RMW/RMW) × (RMW/RMW(z))² × r    (solid body)
# Outside r >  RMW(z):  V = V_RMW × (V_RMW/RMW) × (RMW(z)/r)^a         (power-law decay)
#
# The RMW slope follows Stern & Nolan (2009) Eq. 4.4 (log-pressure coords):
#   dR/dz = -R / [2(T(z) - T_out)] × dT/dz
# where T_out is the outflow temperature at z = 16 km.

RMW    = 31_000.0   # radius of maximum winds at surface (m)
V_RMW  = 43.0       # maximum tangential wind at RMW (m/s)
a      = 0.5        # outer-core wind decay exponent

z_nodes_cpu = Array(znodes(grid, Center()))
Δz_step     = z_nodes_cpu[2] - z_nodes_cpu[1]

T_out_K  = T_sounding_interp(16_000.0)      # outflow temperature

rmw_profile = zeros(Float64, Nz)
rmw_profile[1] = RMW

for k in 2:Nz
    z_k   = z_nodes_cpu[k]
    z_lo  = clamp(z_k - Δz_step/2, zˢ_data[1], zˢ_data[end])
    z_hi  = clamp(z_k + Δz_step/2, zˢ_data[1], zˢ_data[end])
    dTdZ  = (T_sounding_interp(z_hi) - T_sounding_interp(z_lo)) / Δz_step
    T_k   = T_sounding_interp(clamp(z_k, zˢ_data[1], zˢ_data[end]))
    denom = 2.0 * (T_k - T_out_K)
    drdZ  = abs(denom) > 1.0 ? -rmw_profile[k-1] / denom * dTdZ : 0.0
    rmw_profile[k] = max(rmw_profile[k-1] + drdZ * Δz_step, RMW)
end

# CPU-only lookup: safe inside set! (Oceananigans evaluates set! functions on CPU)
function rmw_at_height(z)
    k = clamp(searchsortedfirst(z_nodes_cpu, z), 1, Nz)
    return rmw_profile[k]
end

function tangential_wind(x, y, z)
    r     = sqrt((x - x_center)^2 + (y - y_center)^2)
    rmw_z = rmw_at_height(z)
    v_adj = RMW / rmw_z                       # angular-momentum scaling (Stern & Nolan 2009)
    z >= 16_000.0 && return zero(typeof(r))
    if r <= rmw_z
        return V_RMW * v_adj * r / rmw_z      # solid-body rotation inside eye wall
    else
        return V_RMW * v_adj * (rmw_z / r)^a  # power-law decay outside
    end
end

###########################
# Pressure via gradient-wind balance
###########################
#
# Integrate dp/dr = ρ(fv + v²/r) inward from a far-field radius where p = p_background.
# dp/dr > 0 for a Northern-hemisphere cyclone (pressure increases with radius),
# so moving inward (Δr < 0) reduces pressure → cyclone is a low-pressure centre.

∂r_int     = 1_000.0          # radial step (m)
max_radius = 1_500_000.0      # 1500 km — well into the undisturbed environment
rrange     = collect(0.0:∂r_int:max_radius)   # ascending: rrange[1]=0, rrange[end]=1500 km
Nr         = length(rrange)

p_vortex = zeros(Float64, Nz, Nr)

println("  Computing gradient-wind pressure field ($Nz levels × $Nr radial points)...")

for k in 1:Nz
    z_k  = z_nodes_cpu[k]
    z_c  = clamp(z_k, zˢ_data[1], zˢ_data[end])
    T_k  = T_sounding_interp(z_c)
    p_bg = pˢ_data[1] * exp(-constants.gravitational_acceleration * z_k / (Rᵈ * T_k))
    ρ_k  = p_bg / (Rᵈ * T_k)

    p_vortex[k, Nr] = p_bg    # outer boundary condition: undisturbed background pressure

    # Integrate inward: p(r) = p(r+∂r) − (dp/dr)·∂r
    for r_idx in (Nr-1):-1:1
        r      = rrange[r_idx]
        v_tang = tangential_wind(x_center + r, y_center, z_k)
        dp_dr  = ρ_k * (v_tang * coriolis.f + v_tang^2 / max(r, 1.0))
        p_vortex[k, r_idx] = p_vortex[k, r_idx+1] - dp_dr * ∂r_int
    end
end

p_outer = p_vortex[:, Nr]   # background (outer-edge) pressure profile — used for θ_init

# CPU-only lookup: r-z → pressure
function p_at(x, y, z)
    r     = sqrt((x - x_center)^2 + (y - y_center)^2)
    k     = clamp(searchsortedfirst(z_nodes_cpu, z), 1, Nz)
    r_idx = clamp(searchsortedfirst(rrange, r),      1, Nr)
    return p_vortex[k, r_idx]
end

###########################
# Initial-condition functions  (evaluated on CPU by set!)
###########################

function u_init(x, y, z)
    ϕ = atan(y - y_center, x - x_center)
    return -sin(ϕ) * tangential_wind(x, y, z)
end

function v_init(x, y, z)
    ϕ = atan(y - y_center, x - x_center)
    return  cos(ϕ) * tangential_wind(x, y, z)
end

function θ_init(x, y, z)
    # Background potential temperature from Dunion 2011 sounding
    z_c   = clamp(z, zˢ_data[1], zˢ_data[end])
    θ_bg  = θ_sounding_interp(z_c)
    k     = clamp(searchsortedfirst(z_nodes_cpu, z_c), 1, Nz)
    p_ref = p_outer[k]      # background pressure at this height
    p_loc = p_at(x, y, z)   # pressure at this (x, y, z) including vortex perturbation
    # Warm-core balance: lower p inside vortex ↔ higher θ (Poisson / thermal-wind)
    return θ_bg * (p_ref / p_loc)
end

function qᵗ_init(x, y, z)
    # Sounding moisture profile; no radial variation initially
    z_c = clamp(z, zˢ_data[1], zˢ_data[end])
    return qᵗ_sounding_interp(z_c)   # kg/kg
end

println("  Setting initial conditions...")
set!(model, θ=θ_init, qᵗ=qᵗ_init, u=u_init, v=v_init)
println("  Done.")

###########################
# Visualizations — initial conditions
###########################

println("\n=== Generating initial condition diagnostics ===")

x_km = xnodes(grid, Center()) ./ 1000
y_km = ynodes(grid, Center()) ./ 1000
z_km = znodes(grid, Center()) ./ 1000
r_km = rrange ./ 1000

# ---- 1. Dunion 2011 sounding (already saved above) ----

# ---- 2. RMW vs height ----
fig = Figure(size=(480, 550))
ax  = Axis(fig[1, 1], xlabel="RMW (km)", ylabel="Height (km)",
           title="Radius of Maximum Winds\n(Stern & Nolan 2009 Eq. 4.4)")
lines!(ax, rmw_profile ./ 1000, z_km)
vlines!(ax, [RMW/1000], color=:red, linestyle=:dash, label="Surface RMW")
axislegend(ax)
Makie.save("rmw_profile.png", fig)
println("  Saved rmw_profile.png")

# ---- 3. Tangential wind profile at surface (1-D) ----
vt_sfc = [tangential_wind(x_center + r, y_center, z_nodes_cpu[1]) for r in rrange]
fig = Figure(size=(500, 380))
ax  = Axis(fig[1, 1], xlabel="Radius (km)", ylabel="|V_tan| (m/s)",
           title="Modified Rankine Wind Profile at Surface")
lines!(ax, r_km, vt_sfc)
vlines!(ax, [RMW/1000], color=:red, linestyle=:dash, label="RMW = $(RMW/1000) km")
axislegend(ax)
xlims!(ax, 0, 200)
Makie.save("tangential_wind_profile_1d.png", fig)
println("  Saved tangential_wind_profile_1d.png")

# ---- 4. Pressure deficit cross section (r–z) ----
p_deficit = (p_outer' .- p_vortex) ./ 100   # hPa; shape (Nz, Nr)
fig = Figure(size=(700, 430))
fig[0, :] = Label(fig, "Pressure deficit (hPa)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], xlabel="Radius (km)", ylabel="Height (km)")
cf  = contourf!(ax, r_km, z_km, p_deficit; colormap=:viridis)
Colorbar(fig[1, 2], cf, label="Δp (hPa)")
xlims!(ax, 0, 200)
Makie.save("pressure_deficit_profile.png", fig)
println("  Saved pressure_deficit_profile.png")

# ---- 5. Wind speed plan view at k = 1 (≈ surface) ----
speed_xy = (Array(interior(model.velocities.u, :, :, 1)).^2 .+
            Array(interior(model.velocities.v, :, :, 1)).^2).^0.5
fig = Figure(size=(580, 500))
fig[0, :] = Label(fig, "Wind speed at surface (m/s)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)")
hm  = heatmap!(ax, x_km, y_km, speed_xy; colormap=:amp, colorrange=(0, V_RMW))
Colorbar(fig[1, 2], hm, label="|V| (m/s)")
Makie.save("tangential_wind_profile.png", fig)
println("  Saved tangential_wind_profile.png")

# ---- 6. Wind speed cross section through domain centre (y–z) ----
speed_yz = (Array(interior(model.velocities.u, Nx÷2, :, :)).^2 .+
            Array(interior(model.velocities.v, Nx÷2, :, :)).^2).^0.5
fig = Figure(size=(680, 430))
fig[0, :] = Label(fig, "Wind speed cross section y–z (m/s)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], xlabel="y (km)", ylabel="z (km)")
hm  = heatmap!(ax, y_km, z_km, speed_yz'; colormap=:amp, colorrange=(0, V_RMW))
Colorbar(fig[1, 2], hm, label="|V| (m/s)")
Makie.save("tangential_wind_profile_cross_section.png", fig)
println("  Saved tangential_wind_profile_cross_section.png")

# ---- 7. Potential temperature perturbation plan view (z ≈ 160 m) ----
θ_lip     = liquid_ice_potential_temperature(model)
θ_xy      = Array(interior(θ_lip, :, :, 2))
θ_bg_k2   = θ_sounding_interp(z_km[2] * 1000)
Δθ_xy     = θ_xy .- θ_bg_k2
clim_θ    = maximum(abs.(Δθ_xy))
fig = Figure(size=(580, 500))
fig[0, :] = Label(fig, "Δθ at z ≈ $(round(z_km[2], digits=1)) km (K)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)")
hm  = heatmap!(ax, x_km, y_km, Δθ_xy; colormap=:RdBu_r, colorrange=(-clim_θ, clim_θ))
Colorbar(fig[1, 2], hm, label="Δθ (K)")
Makie.save("theta_init.png", fig)
println("  Saved theta_init.png")

# ---- 8. Potential temperature perturbation cross section (y–z) ----
θ_bg_prof = [θ_sounding_interp(z_km[k] * 1000) for k in 1:Nz]
Δθ_yz     = Array(interior(θ_lip, Nx÷2, :, :)) .- θ_bg_prof'
clim_θyz  = maximum(abs.(Δθ_yz))
fig = Figure(size=(680, 430))
fig[0, :] = Label(fig, "Δθ cross section y–z (K)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], xlabel="y (km)", ylabel="z (km)")
hm  = heatmap!(ax, y_km, z_km, Δθ_yz'; colormap=:RdBu_r, colorrange=(-clim_θyz, clim_θyz))
Colorbar(fig[1, 2], hm, label="Δθ (K)")
Makie.save("theta_init_cross_section.png", fig)
println("  Saved theta_init_cross_section.png")

# ---- 9. Moisture cross section (y–z) ----
# model.tracers.ρqᵗ is the density-weighted field; divide by ρ_ref to get specific humidity
fig = Figure(size=(680, 430))
fig[0, :] = Label(fig, "Initial qᵗ cross section y–z (g/kg)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], xlabel="y (km)", ylabel="z (km)")
qᵗ_yz = Array(interior(model.specific_moisture, Nx÷2, :, :))   # kg/kg
hm  = heatmap!(ax, y_km, z_km, (qᵗ_yz .* 1000)'; colormap=:dense)
Colorbar(fig[1, 2], hm, label="qᵗ (g/kg)")
Makie.save("moisture_init_cross_section.png", fig)
println("  Saved moisture_init_cross_section.png")

# ---- 10. Rainband heating profiles (r–z cross sections) ----
r_vis = range(0.0, 150_000.0, length=150)
z_vis = range(0.0, 15_000.0,  length=150)
Q_con_2d = [convective_rainband_heating(x_center + r, y_center, Float32(z), 0.0f0, con_params)
            for z in z_vis, r in r_vis]
Q_str_2d = [stratiform_rainband_heating(x_center + r, y_center, Float32(z), 0.0f0, str_params)
            for z in z_vis, r in r_vis]

fig = Figure(size=(950, 420))
fig[0, :] = Label(fig, "Rainband heating profiles (K/day equivalent)\n" *
                       "($(use_yu_didlake_2019 ? "Yu & Didlake 2019" : "Moon & Nolan 2010"))",
                  fontsize=14, tellwidth=false)
ax1 = Axis(fig[1, 1], xlabel="Radius (km)", ylabel="Height (km)", title="Convective")
ax2 = Axis(fig[1, 2], xlabel="Radius (km)", ylabel="Height (km)", title="Stratiform")
lim_c = maximum(abs.(Q_con_2d)) * 86400
lim_s = max(maximum(abs.(Q_str_2d)) * 86400, 0.01)
cf1 = contourf!(ax1, collect(r_vis)./1000, collect(z_vis)./1000, Q_con_2d.*86400;
                colormap=:RdBu_r, levels=range(-lim_c, lim_c, length=21))
cf2 = contourf!(ax2, collect(r_vis)./1000, collect(z_vis)./1000, Q_str_2d.*86400;
                colormap=:RdBu_r, levels=range(-lim_s, lim_s, length=21))
Colorbar(fig[1, 3], cf2, label="K/day")
Makie.save("rainband_heating_profiles.png", fig)
println("  Saved rainband_heating_profiles.png")

###########################
# Simulation
###########################

simulation = Simulation(model; Δt=2, stop_time=12hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

θˡⁱ = liquid_ice_potential_temperature(model)
qᶜˡ = model.microphysical_fields.qᶜˡ
qʳ  = model.microphysical_fields.qʳ
qᵛ  = model.microphysical_fields.qᵛ
u, v, w = model.velocities

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info @sprintf("Iter: %d, t: %s, Δt: %s, wall time: %s\n" *
                   "  max|V|: %.2f m/s, max w: %.2f m/s, min w: %.2f m/s\n" *
                   "  max(qᵛ): %.2e  max(qᶜˡ): %.2e  max(qʳ): %.2e",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                   maximum(abs, u), maximum(w), minimum(w),
                   maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ))
    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# Collect time series of max vertical velocity (diagnostic)
max_w_ts    = Float64[]
max_w_times = Float64[]
function collect_max_w(sim)
    push!(max_w_times, time(sim))
    push!(max_w_ts,    maximum(w))
    return nothing
end
add_callback!(simulation, collect_max_w, TimeInterval(1minutes))

# ---- Output writers ----
println("\n=== Setting up output ===")
z_out = znodes(grid, Center())
k_5km = searchsortedfirst(z_out, 5_000)
k_1km = searchsortedfirst(z_out, 1_000)
j_mid = Ny ÷ 2
println("  xy-slices at z ≈ $(z_out[k_5km]) m (k=$k_5km) and z ≈ $(z_out[k_1km]) m (k=$k_1km)")
println("  xz-slice  at y ≈ $(ynodes(grid, Center())[j_mid]) m (j=$j_mid)")

slice_outputs = (
    wxy    = view(w,    :, :, k_5km),
    qʳxy   = view(qʳ,   :, :, k_5km),
    qᶜˡxy  = view(qᶜˡ,  :, :, k_5km),
    uxy    = view(u,    :, :, k_1km),
    vxy    = view(v,    :, :, k_1km),
    wxz    = view(w,    :, j_mid, :),
    qʳxz   = view(qʳ,   :, j_mid, :),
    qᶜˡxz  = view(qᶜˡ,  :, j_mid, :),
)

slices_filename = "tropical_cyclone_and_rainband_slices.jld2"
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = slices_filename,
                                                including = [:grid],
                                                schedule = TimeInterval(2minutes),
                                                overwrite_existing = true)

run!(simulation)

###########################
# Post-run plots
###########################

println("\n=== Plotting output slices ===")
wxy_ts    = FieldTimeSeries(slices_filename, "wxy")
qʳxy_ts   = FieldTimeSeries(slices_filename, "qʳxy")
qᶜˡxy_ts  = FieldTimeSeries(slices_filename, "qᶜˡxy")
wxz_ts    = FieldTimeSeries(slices_filename, "wxz")
qʳxz_ts   = FieldTimeSeries(slices_filename, "qʳxz")
qᶜˡxz_ts  = FieldTimeSeries(slices_filename, "qᶜˡxz")

times = wxy_ts.times
Nt    = length(times)
println("  $Nt snapshots: t ∈ [$(prettytime(times[1])), $(prettytime(times[end]))]")

wlim   = 5.0
qʳlim  = max(maximum(qʳxy_ts),  maximum(qʳxz_ts))  / 4
qᶜˡlim = max(maximum(qᶜˡxy_ts), maximum(qᶜˡxz_ts)) / 4

fig = Figure(size=(1200, 800), fontsize=12)
n   = Observable(1)
fig[0, :] = Label(fig,
    @lift("TC + rainband: w, qᶜˡ, qʳ — t = " * prettytime(times[$n])),
    fontsize=14, tellwidth=false)

axw_xy    = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="w at z≈5 km (m/s)")
axqᶜˡ_xy  = Axis(fig[1, 2], aspect=1, xlabel="x (km)", ylabel="y (km)", title="qᶜˡ at z≈5 km (kg/kg)")
axqʳ_xy   = Axis(fig[1, 3], aspect=1, xlabel="x (km)", ylabel="y (km)", title="qʳ at z≈5 km (kg/kg)")
axw_xz    = Axis(fig[2, 1], xlabel="x (km)", ylabel="z (km)", title="w at y=centre (m/s)")
axqᶜˡ_xz  = Axis(fig[2, 2], xlabel="x (km)", ylabel="z (km)", title="qᶜˡ at y=centre (kg/kg)")
axqʳ_xz   = Axis(fig[2, 3], xlabel="x (km)", ylabel="z (km)", title="qʳ at y=centre (kg/kg)")

wxy_n    = @lift Array(interior(wxy_ts[$n],    :, :, 1))
qᶜˡxy_n  = @lift Array(interior(qᶜˡxy_ts[$n], :, :, 1))
qʳxy_n   = @lift Array(interior(qʳxy_ts[$n],  :, :, 1))
wxz_n    = @lift Array(interior(wxz_ts[$n],    :, 1, :))
qᶜˡxz_n  = @lift Array(interior(qᶜˡxz_ts[$n], :, 1, :))
qʳxz_n   = @lift Array(interior(qʳxz_ts[$n],  :, 1, :))

hmw_xy   = heatmap!(axw_xy,   x_km, y_km, wxy_n;    colormap=:balance, colorrange=(-wlim,   wlim))
hmqᶜˡ_xy = heatmap!(axqᶜˡ_xy, x_km, y_km, qᶜˡxy_n;  colormap=:dense,   colorrange=(0, qᶜˡlim))
hmqʳ_xy  = heatmap!(axqʳ_xy,  x_km, y_km, qʳxy_n;   colormap=:amp,     colorrange=(0, qʳlim))
hmw_xz   = heatmap!(axw_xz,   x_km, z_km, wxz_n;    colormap=:balance, colorrange=(-wlim,   wlim))
hmqᶜˡ_xz = heatmap!(axqᶜˡ_xz, x_km, z_km, qᶜˡxz_n;  colormap=:dense,   colorrange=(0, qᶜˡlim))
hmqʳ_xz  = heatmap!(axqʳ_xz,  x_km, z_km, qʳxz_n;   colormap=:amp,     colorrange=(0, qʳlim))

Colorbar(fig[3, 1], hmw_xy,   vertical=false, label="w (m/s)")
Colorbar(fig[3, 2], hmqᶜˡ_xy, vertical=false, label="qᶜˡ (kg/kg)")
Colorbar(fig[3, 3], hmqʳ_xy,  vertical=false, label="qʳ (kg/kg)")

# Save all frames
for t_idx in 1:Nt
    n[] = t_idx
    Makie.save(@sprintf("tropical_cyclone_slices_t%04d.png", t_idx), fig)
end
n[] = Nt
Makie.save("tropical_cyclone_slices_final.png", fig)
println("  Saved $Nt frames + tropical_cyclone_slices_final.png")

# ---- Max-w time series ----
fig_ts = Figure(size=(600, 340))
ax_ts  = Axis(fig_ts[1, 1], xlabel="Time (h)", ylabel="max w (m/s)",
              title="Maximum vertical velocity")
lines!(ax_ts, max_w_times ./ 3600, max_w_ts)
Makie.save("max_w_timeseries.png", fig_ts)
println("  Saved max_w_timeseries.png")
