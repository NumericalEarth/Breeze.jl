using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units

using AtmosphericProfilesLibrary
using CairoMakie
using CloudMicrophysics
using Printf
using Random
using CUDA
using CSV
using DataFrames
using Interpolations

Random.seed!(42)

Random.TaskLocalRNG()

## Domain and grid
Oceananigans.defaults.FloatType = Float32

Nx = Ny = 100
Nz = 100

x = y = (0, 100000)
z = (0, 16000)

grid = RectilinearGrid(CPU(); x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

constants = ThermodynamicConstants()

## reference state

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = 101540,
                                 potential_temperature = 300)

dynamics = AnelasticDynamics(reference_state)



## surface fluxes
Cᴰ = 1.229e-3 # Drag coefficient for momentum
Cᵀ = 1.094e-3 # Sensible heat transfer coefficient
Cᵛ = 1.133e-3 # Moisture flux transfer coefficient
T₀ = 300     # Sea surface temperature (K)

ρe_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=T₀)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=T₀)

ρe_bcs = FieldBoundaryConditions(bottom=ρe_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)

ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

## damping 
sponge_rate = 1/8  # s⁻¹ - relaxation rate (8 s timescale)
sponge_mask = GaussianMask{:z}(center=3500, width=500)
sponge = Relaxation(rate=sponge_rate, mask=sponge_mask)

## subsidence
FT = eltype(grid)
wˢ_profile = AtmosphericProfilesLibrary.Rico_subsidence(FT)
wˢ = Field{Nothing, Nothing, Face}(grid)
set!(wˢ, z -> wˢ_profile(z))
subsidence = SubsidenceForcing(wˢ)


## geostrophic forcings
println("\n=== Setting up geostrophic forcings ===")
coriolis = FPlane(f=4.5e-5)
println("  Coriolis parameter: f = $(coriolis.f) s⁻¹")
uᵍ = AtmosphericProfilesLibrary.Rico_geostrophic_ug(FT)
vᵍ = AtmosphericProfilesLibrary.Rico_geostrophic_vg(FT)
geostrophic = geostrophic_forcings(z -> uᵍ(z), z -> vᵍ(z))
println("  Geostrophic wind profiles loaded from RICO")


## moisture tendency
ρᵣ = reference_state.density
∂t_ρqᵗ_large_scale = Field{Nothing, Nothing, Center}(grid)
dqdt_profile = AtmosphericProfilesLibrary.Rico_dqtdt(FT)
set!(∂t_ρqᵗ_large_scale, z -> dqdt_profile(z))
set!(∂t_ρqᵗ_large_scale, ρᵣ * ∂t_ρqᵗ_large_scale)
∂t_ρqᵗ_large_scale_forcing = Forcing(∂t_ρqᵗ_large_scale)

## radiative forcing
∂t_ρθ_large_scale = Field{Nothing, Nothing, Center}(grid)
∂t_θ_large_scale = - 2.5 / day # K / day
set!(∂t_ρθ_large_scale, ρᵣ * ∂t_θ_large_scale)
ρθ_large_scale_forcing = Forcing(∂t_ρθ_large_scale)


## forcing and boundary conditions
Fρu = (subsidence, geostrophic.ρu)
Fρv = (subsidence, geostrophic.ρv)
Fρw = sponge
Fρqᵗ = (subsidence, ∂t_ρqᵗ_large_scale_forcing)
Fρθ = (subsidence, ρθ_large_scale_forcing)

forcing = (ρu=Fρu, ρv=Fρv, ρw=Fρw, ρqᵗ=Fρqᵗ, ρθ=Fρθ)
boundary_conditions = (ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)

## model
println("\n=== Creating AtmosphereModel ===")
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

cloud_formation = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

weno = WENO(order=5)
bounds_preserving_weno = WENO(order=5, bounds=(0, 1))

momentum_advection = weno
scalar_advection = (ρθ = weno,
                    ρqᵗ = bounds_preserving_weno,
                    ρqᶜˡ = bounds_preserving_weno,
                    ρqʳ = bounds_preserving_weno)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics,
                        momentum_advection, scalar_advection, forcing, boundary_conditions)


###########################
# Initial conditions
###########################

RMW = 31000
V_RMW = 43 # m/s
a = 0.5
Deltap = 4500 # Pa
pressure_top = 50000 # Pa (500 mb)
Δz = znodes(grid, Center())[2] - znodes(grid, Center())[1]

# Center of domain for perturbation
x_center = (x[1] + x[2]) / 2
y_center = (y[1] + y[2]) / 2
println("  Perturbation center: ($(x_center/1000), $(y_center/1000)) km")


# =============================================================================
# VORTEX PROFILES (Moon and Nolan 2010 / Emanuel 1986 / Stern and Nolan 2009 / Yu and Didlake 2019)
# =============================================================================

# load moist tropical sounding from Dunion 2011
sounding = CSV.read("dunion2011_moist_tropical_MT.csv", DataFrame)
T_C = sounding[:, :Temperature_C]
T_dew = sounding[:, :Dewpoint_C]
p = Float64.(sounding[1:end-1, :Pressure_hPa])
z = Float64.(sounding[:, :GPH_m])

# Thermodynamic ODE uses absolute temperature (K)
# the radius of the angular momentum surface changes with height, following the Eq. 4.4 in Stern and Nolan 2009
T_K = T_C .+ 273.15
z_asc = reverse(z)
T_asc = reverse(T_K)
T_interp = Interpolations.linear_interpolation(z_asc, T_asc)
T_out_K = T_interp(16000)
radius_of_angular_momentum_surface = Array{Float64}(undef, Nz)
radius_of_angular_momentum_surface[1] = RMW
for k in 2:Nz
    z_k = znodes(grid, Center())[k]
    dTdZ = (T_interp(z_k + Δz) - T_interp(z_k))/Δz
    denom = 2 * (T_interp(z_k) - T_out_K)
    drdZ = -radius_of_angular_momentum_surface[k-1] / denom * dTdZ 
    radius_of_angular_momentum_surface[k] = radius_of_angular_momentum_surface[k-1] + drdZ * Δz
end

function radius_of_angular_momentum_surface(z)
    return radius_of_angular_momentum_surface[searchsortedfirst(znodes(grid, Center()), z)]
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Radius (km)", ylabel="Altitude (km)")
lines!(ax, r./1000, z_sample./1000, linewidth=2)
xlims!(ax, 0, 300)
ax.xticks = (0:50:300, string.(0:50:300))
return fig




function tangential_wind(x, y, z)
    v_adjustment_factor = (RMW) ./ (RMW + radius_of_eyewall_adjustment(z))
    r = sqrt((x-x_center)^2 + (y-y_center)^2)
    if r <= RMW + radius_of_eyewall_adjustment(z)
        return V_RMW * v_adjustment_factor * r/(RMW + radius_of_eyewall_adjustment(z))
    else
        return V_RMW * v_adjustment_factor * ((RMW + radius_of_eyewall_adjustment(z))/r)^a
    end
end


function u_init(x, y, z)
    return -sin(atan(y-y_center, x-x_center))*tangential_wind(x, y, z)
end

function v_init(x, y, z)
    return cos(atan(y-y_center, x-x_center))*tangential_wind(x, y, z)
end




function θ_init(x, y, z)
    # Calculate radius from center
    radius = sqrt((x - x_center)^2 + (y - y_center)^2)
    
    # Get reference pressure at height z using hydrostatic relation
    # For constant potential temperature atmosphere, T = θ (constant with height)
    # Using hydrostatic equation: dp/dz = -ρ*g = -p*g/(R*T)
    # This gives: p(z) = p_surface * exp(-g*z/(R*T))
    # Since T = θ for constant potential temperature
    θ_ref = reference_state.potential_temperature
    p_surface = reference_state.surface_pressure
    g = constants.gravitational_acceleration
    R = constants.molar_gas_constant/constants.dry_air.molar_mass

    linear_reduction_factor = (RMW) ./ (RMW + radius_of_eyewall_adjustment(z))
    
    # Compute reference pressure and density at height z
    # For constant θ, temperature is constant: T = θ_ref
    p_ref = p_surface * exp(-g * z / (R * θ_ref))
    # Compute reference density from ideal gas law: ρ = p/(RT)
    ρ_ref = p_ref / (R * θ_ref)
    
    
    # Apply perturbation if pressure is below 500 mb (50000 Pa)
    # "Below 500 mb" means higher pressure (lower altitude)
    # Perturbation is maximum at surface (SLP) and decreases linearly upward to zero at 500 mb
    # Linear decrease from 1 at surface to 0 at 500 mb
    δp = -Deltap * linear_reduction_factor / (1 + (radius/(RMW+radius_of_eyewall_adjustment(z))*linear_reduction_factor)^2)
    
    # Convert pressure perturbation to potential temperature perturbation
    # For anelastic dynamics with constant reference θ: δp/p_ref ≈ -δθ/θ_ref
    # This relationship comes from the equation of state and hydrostatic balance
    # So: δθ = -θ_ref * (δp/p_ref)
    δθ = -θ_ref * (δp / p_ref)
    
    # Return reference potential temperature plus perturbation
    return θ_ref + δθ
    
end




## plot the wind profiles at the surface



# Apply the initial condition as a potential temperature perturbation
# The pressure perturbation will be generated through the dynamics and equation of state
println("\n=== Applying initial conditions ===")
set!(model, 
    θ=θ_init, 
    u=u_init, 
    v=v_init)





θ_field = liquid_ice_potential_temperature(model)
θ_data = Array(interior(θ_field, :, :, 1))
tangential_wind_data = (Array(interior(model.velocities.u, :, :, 1)).^2 + Array(interior(model.velocities.v, :, :, 1)).^2).^0.5
tangential_wind_slice = (Array(interior(model.velocities.u, :, Nx÷2, :)).^2 + Array(interior(model.velocities.v, :, Nx÷2, :)).^2).^0.5

############## PLOT INITIAL CONDITIONS ##############

# Extract coordinate arrays
x_coords = xnodes(grid, Center())
y_coords = ynodes(grid, Center())
z_coords = znodes(grid, Center())

fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, x_coords, y_coords, θ_data, levels=100)
Colorbar(fig[1, 2], limits=(280, 310), label="Potential Temperature (K)")
Makie.save("theta_init.png", fig)

## cross section at center of domain
fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, y_coords, z_coords, Array(interior(θ_field, Nx÷2, :, :)), levels=100)
Colorbar(fig[1, 2], limits=(280, 310), label="Potential Temperature (K)")
Makie.save("theta_init_cross_section.png", fig)

## tangential wind profile
fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, x_coords, y_coords, tangential_wind_data, levels=100)
Colorbar(fig[1, 2], limits=(0, 33), label="Tangential Wind (m/s)")
Makie.save("tangential_wind_profile.png", fig)

## tangential wind profile at cross section
fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, y_coords, z_coords, tangential_wind_slice, levels=100)
Colorbar(fig[1, 2], limits=(0, 33), label="Tangential Wind (m/s)")
Makie.save("tangential_wind_profile_cross_section.png", fig)




θ_min, θ_max = extrema(θ_data)
println("  Surface potential temperature range: $θ_min - $θ_max K")

## run model
println("\n=== Setting up simulation ===")
simulation = Simulation(model; Δt=2, stop_time=2hours)
println("  Initial time step: Δt = $(simulation.Δt) s")
println("  Stop time: $(simulation.stop_time)")
conjure_time_step_wizard!(simulation, cfl=0.7)
println("  Time step wizard configured with CFL = 0.7")

θˡⁱ = liquid_ice_potential_temperature(model)
qᶜˡ = model.microphysical_fields.qᶜˡ
qʳ = model.microphysical_fields.qʳ
qᵛ = model.microphysical_fields.qᵛ
u, v, w = model.velocities

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, wall time: %s, max|u|: %.2f m/s, max w: %.2f m/s, min w: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                   maximum(abs, u), maximum(w), minimum(w))

    msg *= @sprintf(", max(qᵛ): %.2e, max(qᶜˡ): %.2e, max(qʳ): %.2e",
                    maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ))
    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

max_w_ts = []
max_w_times = []

function collect_max_w(sim)
    push!(max_w_times, time(sim))
    push!(max_w_ts, maximum(w))
    return nothing
end

add_callback!(simulation, collect_max_w, TimeInterval(1minutes))
println("  Callbacks added:")
println("    - Progress report every 100 iterations")
println("    - Max vertical velocity collection every 1 minute")

println("\n=== Setting up output ===")
z = znodes(grid, Center())
k_5km = searchsortedfirst(z, 5000)
println("  Saving xy slices at z = $(z[k_5km]) m (k = $k_5km)")

slice_outputs = (
    wxy = view(w, :, :, k_5km),
    qʳxy = view(qʳ, :, :, k_5km),
    qᶜˡxy = view(qᶜˡ, :, :, k_5km),
)

slices_filename = "splitting_supercell_slices.jld2"
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs; filename=slices_filename,
                                                including = [:grid],
                                                schedule = TimeInterval(2minutes),
                                                overwrite_existing = true)
println("  Output file: $slices_filename")
println("  Output schedule: every 2 minutes")
println("  Output variables: w, qʳ, qᶜˡ at z = $(z[k_5km]) m")

println("\n" * "="^60)
println("Starting simulation...")
println("="^60)
run!(simulation)

println("\n" * "="^60)
println("Simulation completed successfully!")
println("="^60)
println("  Final time: $(simulation.model.clock.time)")
println("  Total iterations: $(simulation.model.clock.iteration)")
if length(max_w_times) > 0
    println("  Maximum vertical velocity recorded: $(maximum(max_w_ts)) m/s")
end

# =============================================================================
# USER-MODIFIABLE CONFIGURATION
# =============================================================================

# Run mode: :setup_only | :plot_tests | :run_simulation
const MODE = :plot_tests

# Domain
const Nx = 100
const Ny = 100
const Nz = 100
const Lx = Ly = 100kilometers
const Lz = 16kilometers

# Basic-state vortex (Modified Rankine, MN10/E86)
const RMW_sfc = 10kilometers      # Radius of maximum wind at surface (m)
const V_max_sfc = 33              # Max tangential wind at surface (m/s)
const vortex_decay = 0.5          # Exponent a in v ~ (RMW/r)^a for r > RMW
const Deltap = 4500               # Central pressure deficit (Pa)
const pressure_top = 50_000       # Pressure at top of perturbation (Pa)

# RMW vertical structure (MN10 Appendix, E86, Stern & Nolan 2009)
# RMW slopes outward linearly with height; slope ∝ RMW_sfc, independent of intensity
# MN10: RMW 31→41 km over 15 km → slope ≈ 2/3. v_max → 0 at z_top=15.9 km
const rmw_slope = 2/3            # dRMW/dz (dimensionless, m/m). ~0.67 from MN10 Fig 3b
const z_top = 15.9kilometers     # Altitude where v_max→0 (tropopause, MN10)

# Stratiform rainband heating (MN10 / Yu-Didlake 2019)
const Q_max = 4.24 / 3600        # Max heating rate (K/h -> K/s)
const z_bs = 4kilometers         # Vertical center (zero heating level, melting level)
const σ_zs = 2kilometers         # Vertical half-wavelength
const σ_rs = 6kilometers         # Radial half-width
const r_band_min = 60kilometers  # Min radius of rainband (downwind end)
const r_band_max = 80kilometers  # Max radius (upwind end)
const rainband_stationary = true # Stationary (Yu-Didlake) vs rotating (MN10)

# Output
const output_dir = "mmn2010_output"
const plot_dpi = 150
const plot_quick = false  # true = skip plan views (faster)
