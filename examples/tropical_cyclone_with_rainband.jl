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
using Breeze: WENO, TetensFormula, SaturationAdjustment, WarmPhaseEquilibrium, SmagorinskyLilly
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, VerticalScalarDiffusivity
using CloudMicrophysics
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics
using CairoMakie
using Printf
using Random
using CUDA
using CSV
using DataFrames
using DataInterpolations: CubicSpline

Random.seed!(42)
Random.TaskLocalRNG()

# ## Output directory


const figures_dir = "figures"
mkpath(figures_dir)


# ## Configuration flags
# Set to true to use Yu and Didlake (2019) stratiform rainband modifications:
#   - Higher heating centre (z_bs = 6 km vs 4 km in Moon and Nolan 2010)
#   - Stronger maximum heating amplitude (Q_str_max = 4.24 K/h vs 1.5 K/h)
#   - Broader radial half-width (σ_rs = 8 km vs 6 km)
const use_yu_didlake_2019 = true

# Set to true to initialise a Bernoulli secondary circulation (boundary-layer inflow
# at low levels, upper-troposphere outflow at high levels) consistent with the
# gradient-wind pressure field.  Speeds up spin-up by seeding the transverse
# circulation from t = 0.
const use_secondary_circulation = true

# Set to true to initialise a warm-core potential-temperature anomaly derived
# hydrostatically from the gradient-wind pressure field.  When false, θ is
# initialised from the environmental sounding alone (no warm core).
const use_warm_core = true


# ## Domain and grid


Oceananigans.defaults.FloatType = Float64

arch = GPU()
extent = 1000000
Nx = Ny = 512
Nz = 100
resolution=3000
debug = 1
total_time = 6hours

x = y = (0, extent)
z = (0, 20000)          

grid = RectilinearGrid(arch; x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

constants = ThermodynamicConstants()

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

# ---- Cubic spline interpolants (used for IC and pressure integration) ----
# CubicSpline gives C²-continuous profiles, avoiding the kinks that cause spurious gravity wave generation

θ_sounding_interp  = CubicSpline(θˢ_data,  zˢ_data)
qᵗ_sounding_interp = CubicSpline(qᵗˢ_data, zˢ_data)
T_sounding_interp  = CubicSpline(Tˢ_data,  zˢ_data)
p_sounding_interp  = CubicSpline(pˢ_data,  zˢ_data)

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
Makie.save(joinpath(figures_dir, "dunion2011_sounding.png"), fig)
println("  Saved dunion2011_sounding.png")

###########################
# Reference state
###########################

# Reference state from the Dunion 2011 sounding.
#
# Two-step construction:
#
# Step 1 — The ReferenceState constructor integrates the dry hydrostatic equation
#   dp/dz = -gρ  (ρ = p/(Rᵈ T), T = θ(p/pˢᵗ)^κ)
# producing p_ref, ρ_ref, T_ref that are *mutually consistent* with the sounding θ.
# However, the constructor integrates *independently* for each grid level (from z=0
# to z_k with 1000 steps of size z_k/1000), so adjacent levels have slightly different
# integration errors → noisy dp/dz → oscillatory density and temperature at the top.
#
# Step 2 — compute_reference_state! re-integrates p and ρ using the *level-by-level*
# moist hydrostatic kernel (each level's p derived from the one below), which gives
# smooth, consistent profiles.  We feed it the constructor's own T (which satisfies
# T = θ(p/pˢᵗ)^κ) rather than the raw sounding T, preserving the θ-T-p identity.
# Using T_sounding directly would break this because p_ref ≠ p_observed.
reference_state = ReferenceState(grid, constants,
                                 surface_pressure      = pˢ_data[1],
                                 potential_temperature = z -> θ_sounding_interp(clamp(z, zˢ_data[1], zˢ_data[end])),
                                 vapor_mass_fraction   = z -> qᵗ_sounding_interp(clamp(z, zˢ_data[1], zˢ_data[end])))

# Extract the constructor's T (consistent with θ and p_ref) before overwriting
_T_consistent = Array(interior(reference_state.temperature, 1, 1, :))
compute_reference_state!(reference_state,
                         reshape(_T_consistent, 1, 1, Nz),
                         z -> qᵗ_sounding_interp(clamp(z, zˢ_data[1], zˢ_data[end])),
                         constants)


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

# Quadratic ramp from z_start (TC top) to z_top (domain top): mask = ((z-z_start)/(z_top-z_start))².
# Exactly zero at and below z_start so the TC circulation is untouched; increases smoothly
# to full rate at the rigid lid.  Implemented as a discrete forcing for GPU compatibility.
# DO NOT GO BELOW 18 KM ALTITUDE: THE OUTFLOW MUST BE CHARACTERIZED
sponge_params = (z_start = 18_000f0,           # m — base of sponge (TC top)
                 z_top   = 20_000f0,           # m — domain top
                 rate    = Float32(1 / 300.0))  # s⁻¹ — 15 s damping timescale

@inline function sponge_ρw_fn(i, j, k, grid, clock, fields, p)
    z    = znode(i, j, k, grid, Center(), Center(), Face())
    frac = clamp((z - p.z_start) / (p.z_top - p.z_start), 0f0, 1f0)
    mask = frac * frac
    return -p.rate * mask * @inbounds fields.ρw[i, j, k]
end
sponge_ρw = Forcing(sponge_ρw_fn, discrete_form=true, parameters=sponge_params)

# Horizontal wind sponge: relax ρu, ρv toward zero above TC top.

@inline function sponge_ρu_fn(i, j, k, grid, clock, fields, p)
    z    = znode(i, j, k, grid, Face(), Center(), Center())
    frac = clamp((z - p.z_start) / (p.z_top - p.z_start), 0f0, 1f0)
    mask = frac * frac
    return -p.rate * mask * @inbounds fields.ρu[i, j, k]
end

@inline function sponge_ρv_fn(i, j, k, grid, clock, fields, p)
    z    = znode(i, j, k, grid, Center(), Face(), Center())
    frac = clamp((z - p.z_start) / (p.z_top - p.z_start), 0f0, 1f0)
    mask = frac * frac
    return -p.rate * mask * @inbounds fields.ρv[i, j, k]
end

sponge_ρu = Forcing(sponge_ρu_fn, discrete_form=true, parameters=sponge_params)
sponge_ρv = Forcing(sponge_ρv_fn, discrete_form=true, parameters=sponge_params)

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
Rᵈ    = dry_air_gas_constant(constants)                              # J kg⁻¹ K⁻¹ ≈ 287
Rᵛ    = vapor_gas_constant(constants)                                # J kg⁻¹ K⁻¹ ≈ 461
cₚ    = 1004.0                                                       # J kg⁻¹ K⁻¹ (dry air)
κ_exp = Rᵈ / cₚ                                                      # Poisson exponent ≈ 0.286
T_avg = sum(Tˢ_data) / length(Tˢ_data)                               # mean sounding temperature
ρ_sfc = pˢ_data[1] / (Rᵈ * Tˢ_data[1])                             # surface density  (kg m⁻³)
H_ρ   = Rᵈ * T_avg / constants.gravitational_acceleration            # density scale height (m)

# Rainband radial centre (70 km: midpoint of the 60–80 km range in MN10 Fig. 1)
r_rb = 70_000.0    # m

# ---- Convective parameters ----
Q_con_max = 3.0 / 3600.0   # K s⁻¹  (3.0 K h⁻¹)
σ_rc      = 4_000.0        # radial half-width (m)
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

# ---- Sponge for ρθ: relax toward far-field ρθ at t=0 ----
# WRF's Rayleigh damping (damp_opt=2) damps u, v, w, θ toward reference profiles;
# Yu and Didlake (2019) report sponge-induced vortex weakening (43→40 m/s) consistent
# with damping all prognostic fields.
#
# The target ρθ profile is taken from the model's own far-field at t=0 (corner i=j=1,
# ~530 km from the storm centre), which is unperturbed and consistent with both the
# reference state and the initial moisture/temperature.  This avoids any mismatch that
# would arise from re-deriving ρ from the sounding pressure (which is moist-atmosphere
# pressure and differs slightly from the dry reference-state pressure).
#
# ρθ_bg_col is allocated here as a placeholder; filled from the model after set!() below.
ρθ_bg_col    = zeros(Float32, Nz)
ρθ_bg_device = arch isa GPU ? CuArray(ρθ_bg_col) : ρθ_bg_col

sponge_ρθ_params = (z_start = sponge_params.z_start,
                    z_top   = sponge_params.z_top,
                    rate    = sponge_params.rate,
                    ρθ_bg   = ρθ_bg_device)

@inline function sponge_ρθ_fn(i, j, k, grid, clock, fields, p)
    z      = znode(i, j, k, grid, Center(), Center(), Center())
    frac   = clamp((z - p.z_start) / (p.z_top - p.z_start), 0f0, 1f0)
    mask   = frac * frac
    ρθ_tgt = @inbounds p.ρθ_bg[k]
    return -p.rate * mask * (@inbounds fields.ρθ[i, j, k] - ρθ_tgt)
end

sponge_ρθ = Forcing(sponge_ρθ_fn, discrete_form=true, parameters=sponge_ρθ_params)

forcing = ( 
            # ρw = sponge_ρw, 
            # ρu = sponge_ρu, 
            # ρv = sponge_ρv, 
            ρθ = (
                    # sponge_ρθ,
                    convective_forcing,
                    stratiform_forcing
                ),
            )


###########################
# Model
###########################

println("\n=== Creating AtmosphereModel ===")

cloud_formation        = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
microphysics           = OneMomentCloudMicrophysics(; cloud_formation)
weno                   = WENO(order=5)
bounds_preserving_weno = WENO(order=5, bounds=(0, 1))
momentum_advection     = weno
scalar_advection       = (ρθ   = weno,
                          ρqᵗ  = bounds_preserving_weno,
                          ρqᶜˡ = bounds_preserving_weno,
                          ρqʳ  = bounds_preserving_weno)

vitd = VerticallyImplicitTimeDiscretization()
closure = VerticalScalarDiffusivity(vitd; ν=150, κ=150)


model = AtmosphereModel(grid; dynamics, coriolis, microphysics,
                        momentum_advection, scalar_advection,
                        boundary_conditions,
                        closure,
                        forcing,
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
r_zero = extent/2  # target radius where tangential wind is relaxed to zero (m)

z_nodes_cpu = Array(znodes(grid, Center()))
Δz_step     = z_nodes_cpu[2] - z_nodes_cpu[1]

# Level-by-level moist gas constant from the sounding: Rᵐ(z) = qᵈ Rᵈ + qᵛ Rᵛ.
# Used so the gradient-wind pressure integration and the hydrostatic θ recovery both
# use the same gas constant as the moist reference state.
Rᵐ_col = map(z_nodes_cpu) do z
    qᵛ = qᵗ_sounding_interp(clamp(z, zˢ_data[1], zˢ_data[end]))
    (1 - qᵛ) * Rᵈ + qᵛ * Rᵛ
end

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

z_decay_bot = 13_000.0   # height where wind begins to taper (m)
z_decay_top = 16_000.0   # height where wind reaches zero (m)

function tangential_wind(x, y, z)
    r     = sqrt((x - x_center)^2 + (y - y_center)^2)
    rmw_z = rmw_at_height(z)
    v_adj = RMW / rmw_z                       # angular-momentum scaling (Stern & Nolan 2009)
    z >= z_decay_top && return zero(typeof(r))
    if r <= rmw_z
        vt = V_RMW * v_adj * r / rmw_z        # solid-body rotation inside eye wall
    elseif r >= r_zero
        return zero(typeof(r))                 # relaxed to zero by r = Nx*1000
    else
        v_outer = V_RMW * v_adj * (rmw_z / r)^a
        taper   = (r_zero - r) / (r_zero - rmw_z)
        vt = v_outer * taper                   # power-law decay with outer linear taper
    end
    # Smooth cosine taper to zero between z_decay_bot and z_decay_top
    if z > z_decay_bot
        vt *= 0.5 * (1 + cos(π * (z - z_decay_bot) / (z_decay_top - z_decay_bot)))
    end
    return vt
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
    p_bg = p_sounding_interp(z_c)   # use observed sounding pressure (not isothermal formula)
    ρ_k  = p_bg / (Rᵐ_col[k] * T_k)   # moist density consistent with reference state

    p_vortex[k, Nr] = p_bg    # outer boundary condition: undisturbed background pressure

    # Integrate inward: p(r) = p(r+∂r) − (dp/dr)·∂r
    for r_idx in (Nr-1):-1:1
        r      = rrange[r_idx]
        v_tang = tangential_wind(x_center + r, y_center, z_k)
        dp_dr  = ρ_k * (v_tang * coriolis.f + v_tang^2 / max(r, 1.0))
        p_vortex[k, r_idx] = p_vortex[k, r_idx+1] - dp_dr * ∂r_int
    end
end

p_outer = p_vortex[:, Nr]   # background (outer-edge) pressure profile

# ---- Pressure at the height-varying RMW (used for secondary-circulation IC) ----
# Linearly interpolate p_vortex to the RMW radius at each model level.
p_at_RMW = zeros(Float64, Nz)
for k in 1:Nz
    rmw_k = rmw_profile[k]
    ri_hi = clamp(searchsortedfirst(rrange, rmw_k), 2, Nr)
    ri_lo = ri_hi - 1
    tr    = clamp((rmw_k - rrange[ri_lo]) / (rrange[ri_hi] - rrange[ri_lo]), 0.0, 1.0)
    p_at_RMW[k] = (1 - tr) * p_vortex[k, ri_lo] + tr * p_vortex[k, ri_hi]
end

# ---- Hydrostatically-balanced θ from the gradient-wind pressure field ----
# Derive ρ from hydrostatic balance (∂p/∂z = -ρg), then T = p/(Rρ), θ = T(p₀/p)^κ.
# This guarantees θ is consistent with BOTH gradient-wind and hydrostatic balance,
# avoiding the constant-T assumption of the Poisson relation.

g_val  = constants.gravitational_acceleration
p_ref0 = 1e5   # reference pressure for potential temperature (Pa)

θ_vortex = zeros(Float64, Nz, Nr)
Δz_grid  = z_nodes_cpu[2] - z_nodes_cpu[1]   # uniform vertical spacing

for ri in 1:Nr
    for k in 1:Nz
        if k == 1
            # Forward difference at bottom
            ρ_hydro = -(p_vortex[2, ri] - p_vortex[1, ri]) / (g_val * Δz_grid)
        elseif k == Nz
            # Backward difference at top
            ρ_hydro = -(p_vortex[Nz, ri] - p_vortex[Nz-1, ri]) / (g_val * Δz_grid)
        else
            # Centered difference in interior
            ρ_hydro = -(p_vortex[k+1, ri] - p_vortex[k-1, ri]) / (2 * g_val * Δz_grid)
        end
        # Guard against non-physical ρ (should not happen, but be safe)
        ρ_hydro = max(ρ_hydro, 1e-3)
        T_loc = p_vortex[k, ri] / (Rᵐ_col[k] * ρ_hydro)   # moist T recovery
        θ_vortex[k, ri] = T_loc * (p_ref0 / p_vortex[k, ri])^κ_exp
    end
end

# Normalize θ_vortex: subtract the far-field θ_vortex and add the sounding θ.
# This ensures the far field exactly matches the sounding while the warm-core anomaly
# comes from the hydrostatically-consistent pressure structure.
θ_anomaly = zeros(Float64, Nz, Nr)
for k in 1:Nz
    θ_far = θ_vortex[k, Nr]
    for ri in 1:Nr
        θ_anomaly[k, ri] = θ_vortex[k, ri] - θ_far
    end
end


println("  θ warm-core anomaly at surface centre: " *
        "$(round(θ_anomaly[1,1], digits=3)) K  " *
        "(max anomaly: $(round(maximum(θ_anomaly), digits=3)) K)")

# CPU-only lookup: r-z → θ  (sounding + hydrostatic anomaly, bilinear interpolation)
function θ_at(x, y, z)
    r  = sqrt((x - x_center)^2 + (y - y_center)^2)
    z_c = clamp(z, zˢ_data[1], zˢ_data[end])
    θ_bg = θ_sounding_interp(z_c)

    # Vertical index + interpolation weight
    ki = clamp(searchsortedfirst(z_nodes_cpu, z), 2, Nz)
    tz = clamp((z - z_nodes_cpu[ki-1]) / (z_nodes_cpu[ki] - z_nodes_cpu[ki-1]), 0.0, 1.0)

    # Radial index + interpolation weight
    ri = clamp(searchsortedfirst(rrange, r), 2, Nr)
    tr = clamp((r - rrange[ri-1]) / (rrange[ri] - rrange[ri-1]), 0.0, 1.0)

    use_warm_core || return θ_bg

    # Bilinear interpolation of anomaly
    a00 = θ_anomaly[ki-1, ri-1]
    a01 = θ_anomaly[ki-1, ri]
    a10 = θ_anomaly[ki,   ri-1]
    a11 = θ_anomaly[ki,   ri]
    δθ  = (1-tz) * ((1-tr)*a00 + tr*a01) + tz * ((1-tr)*a10 + tr*a11)

    return θ_bg + δθ
end

###########################
# Secondary-circulation radial wind  (Bernoulli, use_secondary_circulation)
###########################
#
# From Bernoulli along a frictionless streamline between RMW and radius r:
#   |u_r|(r, z) = sqrt(2 · (p(r, z) − p(RMW, z)) / ρ(r, z))
#
# Sign (transverse/secondary circulation):
#   z = 0          → sign = −1  (inward / boundary-layer inflow)
#   z = z_decay_top/2 → sign =  0  (level of non-divergence)
#   z = z_decay_top   → sign = +1  (outward / upper-troposphere outflow)
# implemented as  sign(z) = −cos(π z / z_decay_top).
#
# Tapers:
#   Radial : linearly from 1 at RMW to 0 at r_zero (matching tangential wind).
#   Vertical: full strength below z_decay_bot, cosine ramp to 0 at z_decay_top.

function radial_wind(x, y, z)
    # Reduced scaler with warm core to avoid coupling instability between
    # Bernoulli radial wind and warm-core buoyancy (see tests_warm_core_stability/).
    scaler = use_warm_core ? 0.01 : 0.1
    use_secondary_circulation || return 0.0
    z >= z_decay_top && return 0.0         # no radial wind above 16 km

    r     = sqrt((x - x_center)^2 + (y - y_center)^2)
    k     = clamp(searchsortedfirst(z_nodes_cpu, z), 1, Nz)
    rmw_z = rmw_profile[k]

    r <= rmw_z && return 0.0           # calm inside the eye

    # Interpolate p_vortex at radius r
    ri = clamp(searchsortedfirst(rrange, r), 2, Nr)
    tr = clamp((r - rrange[ri-1]) / (rrange[ri] - rrange[ri-1]), 0.0, 1.0)
    p_r = (1 - tr) * p_vortex[k, ri-1] + tr * p_vortex[k, ri]

    dp = p_r - p_at_RMW[k]            # > 0 for r > RMW (pressure increases outward)
    dp <= 0.0 && return 0.0

    # Moist density from isothermal T(z) — consistent with the θ initialisation
    z_c = clamp(z, zˢ_data[1], zˢ_data[end])
    ρ_k = p_r / (Rᵐ_col[k] * T_sounding_interp(z_c))

    speed = sqrt(2.0 * dp / ρ_k)

    # Radial taper: 1 at RMW → 0 at r_zero
    taper_r = clamp((r_zero - r) / (r_zero - rmw_z), 0.0, 1.0)

    # Vertical taper: 1 below z_decay_bot, cosine decay to 0 at z_decay_top
    taper_z = if z >= z_decay_top
        0.0
    elseif z > z_decay_bot
        0.5 * (1.0 + cos(π * (z - z_decay_bot) / (z_decay_top - z_decay_bot)))
    else
        1.0
    end

    # Sign: −1 (inflow) at z = 0  →  0 at z = z_decay_top/2  →  +1 (outflow) at z = z_decay_top
    sign_z = -cos(π * clamp(z, 0.0, z_decay_top) / z_decay_top)

    return sign_z * speed * taper_r * taper_z*scaler
end

###########################
# Initial-condition functions  (evaluated on CPU by set!)
###########################

function u_init(x, y, z)
    ϕ = atan(y - y_center, x - x_center)
    return -sin(ϕ) * tangential_wind(x, y, z) + cos(ϕ) * radial_wind(x, y, z)
end

function v_init(x, y, z)
    ϕ = atan(y - y_center, x - x_center)
    return  cos(ϕ) * tangential_wind(x, y, z) + sin(ϕ) * radial_wind(x, y, z)
end

function θ_init(x, y, z)
    return θ_at(x, y, z)
end

function qᵗ_init(x, y, z)
    # Sounding moisture profile; no radial variation initially
    z_c = clamp(z, zˢ_data[1], zˢ_data[end])
    return qᵗ_sounding_interp(z_c)   # kg/kg
end

println("  Setting initial conditions (threaded array fill)...")
xn = Array(xnodes(grid, Center()))
yn = Array(ynodes(grid, Center()))
zn = Array(znodes(grid, Center()))

u_arr  = zeros(Float32, Nx, Ny, Nz)
v_arr  = zeros(Float32, Nx, Ny, Nz)
θ_arr  = zeros(Float32, Nx, Ny, Nz)
qᵗ_arr = zeros(Float32, Nx, Ny, Nz)

Threads.@threads for i in 1:Nx
    for j in 1:Ny, k in 1:Nz
        u_arr[i,j,k]  = u_init(xn[i], yn[j], zn[k])
        v_arr[i,j,k]  = v_init(xn[i], yn[j], zn[k])
        θ_arr[i,j,k]  = θ_init(xn[i], yn[j], zn[k])
        qᵗ_arr[i,j,k] = qᵗ_init(xn[i], yn[j], zn[k])
    end
end

set!(model, θ=θ_arr, qᵗ=qᵗ_arr, u=u_arr, v=v_arr)
set_to_mean!(model.dynamics.reference_state, model)
println("  Done.")

# Fill the sponge ρθ target from the model's far-field at t=0.
# Corner (i=1, j=1) is ~530 km from the storm centre — fully unperturbed.
copyto!(ρθ_bg_col, Array(interior(model.formulation.potential_temperature_density, 1, 1, :)))
arch isa GPU && copyto!(ρθ_bg_device, ρθ_bg_col)

let z_km_bg = Array(znodes(grid, Center())) ./ 1000
    fig_bg = Figure(size=(400, 600))
    ax_bg  = Axis(fig_bg[1, 1],
        xlabel = "ρθ (kg m⁻³ K)",
        ylabel = "Height (km)",
        title  = "Sponge target: far-field ρθ at t=0")
    lines!(ax_bg, Float64.(ρθ_bg_col), z_km_bg, color=:navy, linewidth=2)
    Makie.save(joinpath(figures_dir, "rho_theta_background.png"), fig_bg)
    println("  Saved rho_theta_background.png")
end

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
ax  = Axis(fig[1, 1], xlabel="Radius of Maximum Winds (km)", ylabel="Height (km)",
           title="Radius of Maximum Winds vs. Height\n(Stern & Nolan 2009 Eq. 4.4)")
lines!(ax, rmw_profile ./ 1000, z_km)
vlines!(ax, [RMW/1000], color=:red, linestyle=:dash, label="Surface RMW = $(RMW/1000) km")
axislegend(ax)
Makie.save(joinpath(figures_dir, "rmw_profile.png"), fig)
println("  Saved rmw_profile.png")

# ---- 3. Tangential wind profile at surface (1-D) ----
vt_sfc = [tangential_wind(x_center + r, y_center, z_nodes_cpu[1]) for r in rrange]
fig = Figure(size=(500, 380))
ax  = Axis(fig[1, 1], xlabel="Radius from storm centre (km)", ylabel="Tangential wind speed (m s⁻¹)",
           title="Initial Tangential Wind Profile at Surface\n(Modified Rankine Vortex)")
lines!(ax, r_km, vt_sfc)
vlines!(ax, [RMW/1000], color=:red, linestyle=:dash, label="RMW = $(RMW/1000) km")
axislegend(ax)
xlims!(ax, 0, Nx/2)
Makie.save(joinpath(figures_dir, "tangential_wind_profile_1d.png"), fig)
println("  Saved tangential_wind_profile_1d.png")

# ---- 4. Pressure deficit cross section (r–z) ----
p_deficit = (p_outer .- p_vortex) ./ 100   # hPa; shape (Nz, Nr)
fig = Figure(size=(700, 430))
fig[0, :] = Label(fig, "Initial Pressure Deficit vs. Background (hPa)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], xlabel="Radius from storm centre (km)", ylabel="Height (km)")
cf  = contourf!(ax, r_km, z_km, p_deficit'; colormap=:viridis)
Colorbar(fig[1, 2], cf, label="Pressure deficit Δp (hPa)")
xlims!(ax, 0, Nx/2)
Makie.save(joinpath(figures_dir, "pressure_deficit_profile.png"), fig)
println("  Saved pressure_deficit_profile.png")

# ---- 5. Wind speed plan view at k = 1 (≈ surface) ----
speed_xy = (Array(interior(model.velocities.u, :, :, 1)).^2 .+
            Array(interior(model.velocities.v, :, :, 1)).^2).^0.5
fig = Figure(size=(580, 500))
fig[0, :] = Label(fig, "Initial Horizontal Wind Speed at Surface (m s⁻¹)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], aspect=1, xlabel="Zonal distance (km)", ylabel="Meridional distance (km)")
hm  = heatmap!(ax, x_km, y_km, speed_xy; colormap=:amp, colorrange=(0, V_RMW))
Colorbar(fig[1, 2], hm, label="Horizontal wind speed |V| (m s⁻¹)")
Makie.save(joinpath(figures_dir, "tangential_wind_profile.png"), fig)
println("  Saved tangential_wind_profile.png")

# ---- 6. Wind speed cross section through domain centre (y–z) ----
speed_yz = (Array(interior(model.velocities.u, Nx÷2, :, :)).^2 .+
            Array(interior(model.velocities.v, Nx÷2, :, :)).^2).^0.5
fig = Figure(size=(680, 430))
fig[0, :] = Label(fig, "Initial Horizontal Wind Speed — North–South Vertical Cross Section (m s⁻¹)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], xlabel="Meridional distance (km)", ylabel="Height (km)")
hm  = heatmap!(ax, y_km, z_km, speed_yz; colormap=:amp, colorrange=(0, V_RMW))
Colorbar(fig[1, 2], hm, label="Horizontal wind speed |V| (m s⁻¹)")
Makie.save(joinpath(figures_dir, "tangential_wind_profile_cross_section.png"), fig)
println("  Saved tangential_wind_profile_cross_section.png")

# ---- 7. Potential temperature perturbation plan view (z ≈ 160 m) ----
θ_lip     = liquid_ice_potential_temperature(model)
θ_xy      = Array(interior(θ_lip, :, :, 2))
θ_bg_k2   = θ_sounding_interp(clamp(z_km[2] * 1000, zˢ_data[1], zˢ_data[end]))
Δθ_xy     = θ_xy .- θ_bg_k2
clim_θ    = max(Float64(maximum(abs.(Δθ_xy))), 0.01)   # Float64 for Makie; guard zero-range
fig = Figure(size=(580, 500))
fig[0, :] = Label(fig, "Initial Potential Temperature Anomaly at z ≈ $(round(z_km[2], digits=1)) km (K)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], aspect=1, xlabel="Zonal distance (km)", ylabel="Meridional distance (km)")
hm  = heatmap!(ax, x_km, y_km, Δθ_xy; colormap=:balance, colorrange=(-clim_θ, clim_θ))
Colorbar(fig[1, 2], colormap=:balance, limits=(-clim_θ, clim_θ), label="Potential temperature anomaly Δθ (K)")
Makie.save(joinpath(figures_dir, "theta_init.png"), fig)
println("  Saved theta_init.png")

# ---- 8. Potential temperature perturbation cross section (y–z) ----
θ_bg_prof = [θ_sounding_interp(clamp(z_km[k] * 1000, zˢ_data[1], zˢ_data[end])) for k in 1:Nz]
Δθ_yz     = Array(interior(θ_lip, Nx÷2, :, :)) .- θ_bg_prof'
clim_θyz  = max(Float64(maximum(abs.(Δθ_yz))), 0.01)
fig = Figure(size=(680, 430))
fig[0, :] = Label(fig, "Initial Potential Temperature Anomaly — North–South Vertical Cross Section (K)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], xlabel="Meridional distance (km)", ylabel="Height (km)")
hm  = heatmap!(ax, y_km, z_km, Δθ_yz; colormap=:balance, colorrange=(-clim_θyz, clim_θyz))
Colorbar(fig[1, 2], colormap=:balance, limits=(-clim_θyz, clim_θyz), label="Potential temperature anomaly Δθ (K)")
Makie.save(joinpath(figures_dir, "theta_init_cross_section.png"), fig)
println("  Saved theta_init_cross_section.png")

# ---- 9. Moisture cross section (y–z) ----
# model.tracers.ρqᵗ is the density-weighted field; divide by ρ_ref to get specific humidity
fig = Figure(size=(680, 430))
fig[0, :] = Label(fig, "Initial Total Water Mixing Ratio — North–South Vertical Cross Section (g kg⁻¹)", fontsize=14, tellwidth=false)
ax  = Axis(fig[1, 1], xlabel="Meridional distance (km)", ylabel="Height (km)")
qᵗ_yz    = Array(interior(specific_prognostic_moisture(model), Nx÷2, :, :))   # kg/kg
qᵗ_gkg   = qᵗ_yz .* 1000f0
qlim     = Float64(maximum(qᵗ_gkg))
hm  = heatmap!(ax, y_km, z_km, qᵗ_gkg; colormap=:dense, colorrange=(0.0, qlim))
Colorbar(fig[1, 2], colormap=:dense, limits=(0.0, qlim), label="Total water mixing ratio qᵗ (g kg⁻¹)")
Makie.save(joinpath(figures_dir, "moisture_init_cross_section.png"), fig)
println("  Saved moisture_init_cross_section.png")

# ---- 10. Radial wind cross section (r–z) ----
if use_secondary_circulation
    r_ur_vis = range(0.0, extent / 2, length=300)   # 0 → domain half-width (m)
    r_ur_km  = collect(r_ur_vis) ./ 1000.0

    ur_2d = [radial_wind(x_center + r, y_center, z) for r in r_ur_vis, z in z_nodes_cpu]
    urlim = max(maximum(abs.(ur_2d)), 0.01)

    fig = Figure(size=(720, 430))
    fig[0, :] = Label(fig, "Initial Secondary Circulation — Radial Wind (m s⁻¹)  [Bernoulli IC]",
                      fontsize=14, tellwidth=false)
    ax = Axis(fig[1, 1], xlabel="Radius from storm centre (km)", ylabel="Height (km)")
    hm = heatmap!(ax, r_ur_km, z_km, ur_2d;
                  colormap=:balance, colorrange=(-urlim, urlim))
    Colorbar(fig[1, 2], colormap=:balance, limits=(-urlim, urlim),
             label="Radial wind uᵣ (m s⁻¹)  [blue = inflow, red = outflow]")
    # Mark the RMW profile
    rmw_km = [rmw_profile[k] / 1000.0 for k in 1:Nz]
    lines!(ax, rmw_km, z_km; color=:black, linestyle=:dash, linewidth=1.5, label="RMW")
    axislegend(ax; position=:rt)
    Makie.save(joinpath(figures_dir, "radial_wind_init_cross_section.png"), fig)
    println("  Saved radial_wind_init_cross_section.png")
end

# ---- 11. Rainband heating profiles (r–z cross sections + plan views) ----
r_vis = range(0.0, 150_000.0, length=150)
z_vis = range(0.0, 15_000.0,  length=150)
Q_con_2d = [convective_rainband_heating(x_center + r, y_center, Float32(z), 0.0f0, con_params) for r in r_vis, z in z_vis]   # (Nr_vis, Nz_vis) — matches contourf!(r_vis, z_vis, M)
Q_str_2d = [stratiform_rainband_heating(x_center + r, y_center, Float32(z), 0.0f0, str_params) for r in r_vis, z in z_vis]

# Plan views at the sin-peak height of each component
xy_vis_km = range(-150.0, 150.0, length=200)   # km, relative to storm centre
z_con_vis = Float32(z_bc + σ_zc / 2)           # convective peak ≈ 3.5 km
z_str_vis = Float32(z_bs + σ_zs / 2)           # stratiform heating peak (σ_zs/2 above z_bs)
Q_con_xy = [convective_rainband_heating(x_center + xr * 1000, y_center + yr * 1000,
                                        z_con_vis, 0.0f0, con_params)
            for xr in xy_vis_km, yr in xy_vis_km]   # (Nxy, Nxy)
Q_str_xy = [stratiform_rainband_heating(x_center + xr * 1000, y_center + yr * 1000,
                                        z_str_vis, 0.0f0, str_params)
            for xr in xy_vis_km, yr in xy_vis_km]

lim_c = maximum(abs.(Q_con_2d)) * 86400
lim_s = max(maximum(abs.(Q_str_2d)) * 86400, 0.01)
z_con_km = round(z_con_vis / 1000, digits=1)
z_str_km = round(z_str_vis / 1000, digits=1)

fig = Figure(size=(950, 840))
fig[0, :] = Label(fig, "Spiral Rainband Diabatic Heating Rate (K day⁻¹)\n" *
                       "($(use_yu_didlake_2019 ? "Yu & Didlake 2019" : "Moon & Nolan 2010"))",
                  fontsize=14, tellwidth=false)

# Row 1: r–z cross sections
ax1 = Axis(fig[1, 1], xlabel="Radius from storm centre (km)", ylabel="Height (km)",
           title="Convective component — r–z cross section")
ax2 = Axis(fig[1, 2], xlabel="Radius from storm centre (km)", ylabel="Height (km)",
           title="Stratiform component — r–z cross section")
cf1 = contourf!(ax1, collect(r_vis)./1000, collect(z_vis)./1000, Q_con_2d.*86400;
                colormap=:balance, levels=range(-lim_c, lim_c, length=21))
cf2 = contourf!(ax2, collect(r_vis)./1000, collect(z_vis)./1000, Q_str_2d.*86400;
                colormap=:balance, levels=range(-lim_s, lim_s, length=21))
Colorbar(fig[1, 3], cf2, label="Heating rate (K day⁻¹)")

# Row 2: plan views at the peak heating height of each component
ax= Axis(fig[2, 1], aspect=1,
           xlabel="Zonal distance from centre (km)", ylabel="Meridional distance from centre (km)",
           title="Convective component — plan view at z ≈ $(z_con_km) km")
ax4 = Axis(fig[2, 2], aspect=1,
           xlabel="Zonal distance from centre (km)", ylabel="Meridional distance from centre (km)",
           title="Stratiform component — plan view at z ≈ $(z_str_km) km")
heatmap!(ax4, collect(xy_vis_km), collect(xy_vis_km), Q_con_xy .* 86400;
         colormap=:balance, colorrange=(-lim_c, lim_c))
heatmap!(ax4, collect(xy_vis_km), collect(xy_vis_km), Q_str_xy .* 86400;
         colormap=:balance, colorrange=(-lim_s, lim_s))
Colorbar(fig[2, 3], colormap=:balance, limits=(-lim_s, lim_s), label="Heating rate (K day⁻¹)")

Makie.save(joinpath(figures_dir, "rainband_heating_profiles.png"), fig)
println("  Saved rainband_heating_profiles.png")

###########################
# Simulation
###########################

simulation = Simulation(model; Δt=8, stop_time=total_time)
conjure_time_step_wizard!(simulation, cfl=0.4)

θˡⁱ = liquid_ice_potential_temperature(model)
qᶜˡ = model.microphysical_fields.qᶜˡ
qʳ  = model.microphysical_fields.qʳ
qᵛ  = model.microphysical_fields.qᵛ
u, v, w = model.velocities

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    wmax = maximum(w)
    wmin = minimum(w)
    umax = maximum(abs, u)
    msg = @sprintf(
        "Iter: %d, t: %s, Δt: %s, wall: %s | max|V|=%.2f w=[%.2f,%.2f] m/s | qᵛ=%.2e qᶜˡ=%.2e qʳ=%.2e",
        iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
        umax, wmin, wmax,
        maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ))
    @info msg

    # Report locations where |w| > 20 m/s (only copy from GPU if threshold exceeded)
    if max(abs(wmax), abs(wmin)) > 20.0 && debug == 1
    w_cpu = Array(interior(w))
    danger = findall(x -> abs(x) > 20.0, w_cpu)
    if !isempty(danger)
        x_nodes = Array(xnodes(grid, Center()))
        y_nodes = Array(ynodes(grid, Center()))
        z_nodes = Array(znodes(grid, Face()))
        # Sort by descending |w| and show the worst 5
        sort!(danger, by=idx -> -abs(w_cpu[idx]))
        n_show = min(5, length(danger))
        @warn "$(length(danger)) points with |w| > 20 m/s — top $n_show:"
        for p in 1:n_show
            ci = danger[p]
            @warn @sprintf("  [%d] w=%.2f m/s at x=%.1f km, y=%.1f km, z=%.1f km",
                           p, w_cpu[ci],
                           x_nodes[ci[1]] / 1000,
                           y_nodes[ci[2]] / 1000,
                           z_nodes[ci[3]] / 1000)
        end
    end
    end  # max(abs(wmax), abs(wmin)) > 20

    if isnan(wmax) || isnan(umax) || wmax > 60.0
        @warn "NaN or blowup detected — stopping simulation"
        sim.stop_iteration = iteration(sim)
    end
    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(25))

# Collect time series of max vertical velocity (diagnostic)
max_w_ts    = Float64[]
max_w_times = Float64[]
function collect_max_w(sim)
    push!(max_w_times, time(sim))
    push!(max_w_ts,    maximum(w))
    return nothing
end
add_callback!(simulation, collect_max_w, TimeInterval(1minutes))

# Collect vertical profile of θˡⁱ at the domain centre (x=Nx÷2, y=Ny÷2)
center_θ_ts    = Vector{Vector{Float32}}()
center_θ_times = Float64[]
const center_θ_txt = joinpath(figures_dir, "center_theta_profiles.txt")
open(center_θ_txt, "w") do io
    # header: time (s) then one column per vertical level (θ in K)
    println(io, "# Center-column θˡⁱ (K) — one row per snapshot")
    println(io, "# Columns: time_s  θ[1]  θ[2]  ...  θ[$Nz]  (levels bottom→top)")
end
function collect_center_θ(sim)
    compute!(θˡⁱ)
    t  = time(sim)
    θv = Array(interior(θˡⁱ, Nx÷2, Ny÷2, :))   # Vector{Float32}, length Nz
    push!(center_θ_times, t)
    push!(center_θ_ts, θv)
    open(center_θ_txt, "a") do io
        print(io, t)
        for val in θv
            print(io, "  ", val)
        end
        println(io)
    end
    return nothing
end
add_callback!(simulation, collect_center_θ, TimeInterval(2minutes))

# ---- Output writers ----
println("\n=== Setting up output ===")
z_out = znodes(grid, Center())
k_5km = searchsortedfirst(z_out, 5_000)
k_1km = searchsortedfirst(z_out, 1_000)
j_mid = Ny ÷ 2
println("  xy-slices at z ≈ $(z_out[k_5km]) m (k=$k_5km) and z ≈ $(z_out[k_1km]) m (k=$k_1km)")
println("  xz-slice  at y ≈ $(ynodes(grid, Center())[j_mid]) m (j=$j_mid)")

slice_outputs = (
    wxy   = view(w,    :, :, k_5km),
    qʳxy  = view(qʳ,   :, :, k_5km),
    qᶜˡxy = view(qᶜˡ,  :, :, k_5km),
    uxy   = view(u,    :, :, k_5km),
    vxy   = view(v,    :, :, k_5km),
    θxy   = view(θˡⁱ,  :, :, k_5km),
    wxz   = view(w,    :, j_mid, :),
    qʳxz  = view(qʳ,   :, j_mid, :),
    qᶜˡxz = view(qᶜˡ,  :, j_mid, :),
    uxz   = view(u,    :, j_mid, :),
    vxz   = view(v,    :, j_mid, :),
    θxz   = view(θˡⁱ,  :, j_mid, :),
)

slices_filename = "tropical_cyclone_and_rainband_slices.jld2"
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = slices_filename,
                                                including = [:grid],
                                                schedule = TimeInterval(2minutes),
                                                overwrite_existing = true)


@info "Starting simulation run..."
run_start = time_ns()
run!(simulation)
run_elapsed = (time_ns() - run_start) * 1e-9
@info @sprintf("Simulation finished in %.2f seconds (%.2f min)", run_elapsed, run_elapsed / 60)

###########################
# Post-run plots
###########################

println("\n=== Plotting output slices ===")
wxy_ts   = FieldTimeSeries(slices_filename, "wxy")
qʳxy_ts  = FieldTimeSeries(slices_filename, "qʳxy")
qᶜˡxy_ts = FieldTimeSeries(slices_filename, "qᶜˡxy")
uxy_ts   = FieldTimeSeries(slices_filename, "uxy")
vxy_ts   = FieldTimeSeries(slices_filename, "vxy")
wxz_ts   = FieldTimeSeries(slices_filename, "wxz")
qʳxz_ts  = FieldTimeSeries(slices_filename, "qʳxz")
qᶜˡxz_ts = FieldTimeSeries(slices_filename, "qᶜˡxz")
uxz_ts   = FieldTimeSeries(slices_filename, "uxz")
vxz_ts   = FieldTimeSeries(slices_filename, "vxz")
θxy_ts   = FieldTimeSeries(slices_filename, "θxy")
θxz_ts   = FieldTimeSeries(slices_filename, "θxz")

times = wxy_ts.times
Nt    = length(times)
println("  $Nt snapshots: t ∈ [$(prettytime(times[1])), $(prettytime(times[end]))]")

wlim   = 10.0
_safemax(fts) = let v = Float64(maximum(fts)); isfinite(v) ? v : 0.0 end
qʳlim  = max(max(_safemax(qʳxy_ts),  _safemax(qʳxz_ts))  / 4, 1e-6)
qᶜˡlim = max(max(_safemax(qᶜˡxy_ts), _safemax(qᶜˡxz_ts)) / 4, 1e-6)
Vrlim  = 20.0              # ±20 m/s  (inflow negative, outflow positive)
Vtlim  = Float64(V_RMW) * 1.5   # max tangential wind scale
θ_bg_k5km = θ_sounding_interp(clamp(z_out[k_5km], zˢ_data[1], zˢ_data[end]))
θ_bg_xz   = reshape(θ_bg_prof, 1, Nz)   # (1, Nz) for broadcasting against (Nx, Nz)
θlim      = 5.0                           # ±5 K anomaly colorscale

# 2-D coordinate arrays (km, relative to storm centre) for wind decomposition
x_center_km = Float32(x_center / 1000)
y_center_km = Float32(y_center / 1000)
dx_km2d = Float32[x_km[i] - x_center_km for i in 1:Nx, j in 1:Ny]   # (Nx, Ny)
dy_km2d = Float32[y_km[j] - y_center_km for i in 1:Nx, j in 1:Ny]   # (Nx, Ny)
r_safe  = max.(sqrt.(dx_km2d.^2 .+ dy_km2d.^2), 1f-3)
# For the xz slice at y = y_center: sin(ϕ)=0, cos(ϕ)=sign(x−xc)
sign_x  = reshape(sign.(x_km .- x_center_km), :, 1)                   # (Nx, 1)

fig = Figure(size=(2400, 800), fontsize=12)
n   = Observable(1)
fig[0, :] = Label(fig,
    @lift("TC + rainband: w, qᶜˡ, qʳ, Vᵣ, Vₜ, Δθˡⁱ — t = " * prettytime(times[$n])),
    fontsize=14, tellwidth=false)

# Row 1: xy plan views at z ≈ 5 km
axw_xy   = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="w at z≈5 km (m/s)")
axqᶜˡ_xy = Axis(fig[1, 2], aspect=1, xlabel="x (km)", ylabel="y (km)", title="qᶜˡ at z≈5 km (kg/kg)")
axqʳ_xy  = Axis(fig[1, 3], aspect=1, xlabel="x (km)", ylabel="y (km)", title="qʳ at z≈5 km (kg/kg)")
axVr_xy  = Axis(fig[1, 4], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Radial wind at z≈5 km (m/s)")
axVt_xy  = Axis(fig[1, 5], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Tangential wind at z≈5 km (m/s)")
axθ_xy   = Axis(fig[1, 6], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Δθˡⁱ at z≈5 km (K)")

# Row 2: x–z cross sections through domain centre
axw_xz   = Axis(fig[2, 1], xlabel="x (km)", ylabel="z (km)", title="w at y=centre (m/s)")
axqᶜˡ_xz = Axis(fig[2, 2], xlabel="x (km)", ylabel="z (km)", title="qᶜˡ at y=centre (kg/kg)")
axqʳ_xz  = Axis(fig[2, 3], xlabel="x (km)", ylabel="z (km)", title="qʳ at y=centre (kg/kg)")
axVr_xz  = Axis(fig[2, 4], xlabel="x (km)", ylabel="z (km)", title="Radial wind at y=centre (m/s)")
axVt_xz  = Axis(fig[2, 5], xlabel="x (km)", ylabel="z (km)", title="Tangential wind at y=centre (m/s)")
axθ_xz   = Axis(fig[2, 6], xlabel="x (km)", ylabel="z (km)", title="Δθˡⁱ at y=centre (K)")

# Observables for saved fields (replace NaN/Inf with zero so Makie's colormap never sees non-finite values)
_finite(A) = ifelse.(isfinite.(A), A, zero(eltype(A)))
wxy_n   = @lift _finite(Array(interior(wxy_ts[$n],   :, :, 1)))
qᶜˡxy_n = @lift _finite(Array(interior(qᶜˡxy_ts[$n], :, :, 1)))
qʳxy_n  = @lift _finite(Array(interior(qʳxy_ts[$n],  :, :, 1)))
uxy_n   = @lift _finite(Array(interior(uxy_ts[$n],   :, :, 1)))
vxy_n   = @lift _finite(Array(interior(vxy_ts[$n],   :, :, 1)))
θxy_n   = @lift _finite(Array(interior(θxy_ts[$n],   :, :, 1)))
wxz_n   = @lift _finite(Array(interior(wxz_ts[$n],   :, 1, :)))
qᶜˡxz_n = @lift _finite(Array(interior(qᶜˡxz_ts[$n], :, 1, :)))
qʳxz_n  = @lift _finite(Array(interior(qʳxz_ts[$n],  :, 1, :)))
uxz_n   = @lift _finite(Array(interior(uxz_ts[$n],   :, 1, :)))
vxz_n   = @lift _finite(Array(interior(vxz_ts[$n],   :, 1, :)))
θxz_n   = @lift _finite(Array(interior(θxz_ts[$n],   :, 1, :)))

# Derived: radial wind Vᵣ = (u·x̂ + v·ŷ)/r,  tangential Vₜ = (−u·ŷ + v·x̂)/r
Vrxy_n  = @lift _finite((($uxy_n) .* dx_km2d .+ ($vxy_n) .* dy_km2d) ./ r_safe)
Vtxy_n  = @lift _finite((-($uxy_n) .* dy_km2d .+ ($vxy_n) .* dx_km2d) ./ r_safe)

# At y=y_centre: sin(ϕ)=0 so Vᵣ=u·sign(x−xc), Vₜ=v·sign(x−xc)
Vrxz_n  = @lift _finite(($uxz_n) .* sign_x)
Vtxz_n  = @lift _finite(($vxz_n) .* sign_x)

# θ perturbation: subtract sounding background
Δθxy_n  = @lift _finite(Float32.($θxy_n) .- Float32(θ_bg_k5km))
Δθxz_n  = @lift _finite(Float32.($θxz_n) .- Float32.(θ_bg_xz))

hmw_xy   = heatmap!(axw_xy,   x_km, y_km, wxy_n;   colormap=:balance, colorrange=(-wlim,   wlim))
hmqᶜˡ_xy = heatmap!(axqᶜˡ_xy, x_km, y_km, qᶜˡxy_n; colormap=:dense,   colorrange=(0, qᶜˡlim))
hmqʳ_xy  = heatmap!(axqʳ_xy,  x_km, y_km, qʳxy_n;  colormap=:amp,     colorrange=(0, qʳlim))
hmVr_xy  = heatmap!(axVr_xy,  x_km, y_km, Vrxy_n;  colormap=:balance, colorrange=(-Vrlim, Vrlim))
hmVt_xy  = heatmap!(axVt_xy,  x_km, y_km, Vtxy_n;  colormap=:amp,     colorrange=(0, Vtlim))
hmθ_xy   = heatmap!(axθ_xy,   x_km, y_km, Δθxy_n;  colormap=:balance, colorrange=(-θlim, θlim))

# |w| > 20 m/s contours on all plan-view panels
absw_xy_n = @lift abs.($wxy_n)
for ax_xy in (axw_xy, axqᶜˡ_xy, axqʳ_xy, axVr_xy, axVt_xy, axθ_xy)
    contour!(ax_xy, x_km, y_km, absw_xy_n; levels=[20.0], color=:black, linewidth=1.5)
end
hmw_xz   = heatmap!(axw_xz,   x_km, z_km, wxz_n;   colormap=:balance, colorrange=(-wlim,   wlim))
hmqᶜˡ_xz = heatmap!(axqᶜˡ_xz, x_km, z_km, qᶜˡxz_n; colormap=:dense,   colorrange=(0, qᶜˡlim))
hmqʳ_xz  = heatmap!(axqʳ_xz,  x_km, z_km, qʳxz_n;  colormap=:amp,     colorrange=(0, qʳlim))
hmVr_xz  = heatmap!(axVr_xz,  x_km, z_km, Vrxz_n;  colormap=:balance, colorrange=(-Vrlim, Vrlim))
hmVt_xz  = heatmap!(axVt_xz,  x_km, z_km, Vtxz_n;  colormap=:amp,     colorrange=(0, Vtlim))
hmθ_xz   = heatmap!(axθ_xz,   x_km, z_km, Δθxz_n;  colormap=:balance, colorrange=(-θlim, θlim))

Colorbar(fig[3, 1], hmw_xy,   vertical=false, label="w (m/s)")
Colorbar(fig[3, 2], hmqᶜˡ_xy, vertical=false, label="qᶜˡ (kg/kg)")
Colorbar(fig[3, 3], hmqʳ_xy,  vertical=false, label="qʳ (kg/kg)")
Colorbar(fig[3, 4], hmVr_xy,  vertical=false, label="Vᵣ (m/s)")
Colorbar(fig[3, 5], hmVt_xy,  vertical=false, label="Vₜ (m/s)")
Colorbar(fig[3, 6], hmθ_xy,   vertical=false, label="Δθˡⁱ (K)")

# Save animation video
video_path = joinpath(figures_dir, "tropical_cyclone_slices.mp4")
Makie.record(fig, video_path, 1:Nt; framerate=10) do t_idx
    n[] = t_idx
end
n[] = Nt
Makie.save(joinpath(figures_dir, "tropical_cyclone_slices_final.png"), fig)
println("  Saved tropical_cyclone_slices.mp4 + tropical_cyclone_slices_final.png")

# ---- Max-w time series ----
fig_ts = Figure(size=(600, 340))
ax_ts  = Axis(fig_ts[1, 1], xlabel="Time (h)", ylabel="max w (m/s)",
              title="Maximum vertical velocity")
lines!(ax_ts, max_w_times ./ 3600, max_w_ts)
Makie.save(joinpath(figures_dir, "max_w_timeseries.png"), fig_ts)
println("  Saved max_w_timeseries.png")

# ---- Center-column θˡⁱ height–time Hovmöller ----
if length(center_θ_ts) >= 2
    θ_matrix  = reduce(hcat, center_θ_ts)          # (Nz, Nt)
    θ_bg_vec  = [θ_sounding_interp(clamp(z * 1000, zˢ_data[1], zˢ_data[end]))
                 for z in Array(z_km)]              # (Nz,) background
    Δθ_matrix = Float64.(θ_matrix) .- θ_bg_vec     # (Nz, Nt) anomaly
    Δθ_safe   = ifelse.(isfinite.(Δθ_matrix), Δθ_matrix, 0.0)  # NaN → 0 for plotting
    t_h       = center_θ_times ./ 3600

    fig_hov = Figure(size=(700, 500))
    fig_hov[0, :] = Label(fig_hov, "θˡⁱ Anomaly at Domain Centre — Height–Time (K)",
                          fontsize=13, tellwidth=false)
    ax_hov  = Axis(fig_hov[1, 1], xlabel="Time (h)", ylabel="Height (km)")
    lim_hov = max(maximum(abs, Δθ_safe), 0.1)
    hm_hov  = heatmap!(ax_hov, t_h, Array(z_km), Δθ_safe';
                        colormap=:balance, colorrange=(-lim_hov, lim_hov))
    Colorbar(fig_hov[1, 2], hm_hov, label="Δθˡⁱ (K)")
    Makie.save(joinpath(figures_dir, "center_theta_hovmoller.png"), fig_hov)
    println("  Saved center_theta_hovmoller.png")
end
