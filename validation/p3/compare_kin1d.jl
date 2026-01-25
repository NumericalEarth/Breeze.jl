#####
##### P3 kin1d comparison script
#####
##### This script compares Breeze.jl's P3 implementation against the
##### Fortran P3-microphysics kin1d kinematic driver reference data.
#####

using NCDatasets
using CairoMakie
using Statistics

#####
##### Load Fortran reference data
#####

reference_path = joinpath(@__DIR__, "kin1d_reference.nc")
ds = NCDataset(reference_path, "r")

# Extract dimensions
time_seconds = ds["time"][:]
height_meters = ds["z"][:]
nt = length(time_seconds)
nz = length(height_meters)

# Convert time to minutes for plotting
time_minutes = time_seconds ./ 60

# Note: z is ordered from top to bottom in the Fortran output
# z[1] = 12840 m (top), z[end] = 35 m (bottom)
# We'll keep this ordering for consistency

println("=== Fortran P3 kin1d Reference Data ===")
println("Time: $(time_minutes[1]) to $(time_minutes[end]) minutes ($(nt) steps)")
println("Height: $(height_meters[end]) to $(height_meters[1]) m ($(nz) levels)")
println()

# Extract key variables
w = ds["w"][:, :]                # Vertical velocity [m/s]
temperature_C = ds["temperature"][:, :]  # Temperature [°C]
q_cloud = ds["q_cloud"][:, :]    # Cloud liquid [kg/kg]
q_rain = ds["q_rain"][:, :]      # Rain [kg/kg]
q_ice = ds["q_ice"][:, :]        # Total ice [kg/kg]
n_ice = ds["n_ice"][:, :]        # Ice number [1/kg]
rime_fraction = ds["rime_fraction"][:, :]
liquid_fraction = ds["liquid_fraction"][:, :]
reflectivity = ds["reflectivity"][:, :]
prt_liq = ds["prt_liq"][:]       # Liquid precip rate [mm/h]
prt_sol = ds["prt_sol"][:]       # Solid precip rate [mm/h]

# Category 1 ice diagnostics
q_rime_cat1 = ds["q_rime_cat1"][:, :]
q_liquid_on_ice_cat1 = ds["q_liquid_on_ice_cat1"][:, :]
z_ice_cat1 = ds["z_ice_cat1"][:, :]
rho_ice_cat1 = ds["rho_ice_cat1"][:, :]
d_ice_cat1 = ds["d_ice_cat1"][:, :]

close(ds)

# Print statistics
println("=== Reference Data Statistics ===")
println("Max vertical velocity: $(maximum(w)) m/s")
println("Max q_cloud: $(maximum(q_cloud) * 1000) g/kg")
println("Max q_rain: $(maximum(q_rain) * 1000) g/kg")
println("Max q_ice: $(maximum(q_ice) * 1000) g/kg")
println("Max reflectivity: $(maximum(reflectivity)) dBZ")
println("Max liquid precip: $(maximum(prt_liq)) mm/h")
println("Max solid precip: $(maximum(prt_sol)) mm/h")
println()

#####
##### Create visualization of Fortran reference
#####

# Convert height to km for plotting
height_km = height_meters ./ 1000

# Set up figure
fig = Figure(size=(1400, 1000), fontsize=12)

# Note: NCDatasets loads as (time, z), which is exactly what heatmap(x, y, data) expects
# where x=time has 90 points and y=z has 41 points, and data is (90, 41)

# Row 1: Hydrometeor mixing ratios
ax1 = Axis(fig[1, 1], xlabel="Time [min]", ylabel="Height [km]",
           title="Cloud Liquid [g/kg]")
hm1 = heatmap!(ax1, time_minutes, height_km, q_cloud .* 1000,
               colormap=:blues, colorrange=(0, maximum(q_cloud)*1000))
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3], xlabel="Time [min]", ylabel="Height [km]",
           title="Rain [g/kg]")
hm2 = heatmap!(ax2, time_minutes, height_km, q_rain .* 1000,
               colormap=:greens, colorrange=(0, maximum(q_rain)*1000))
Colorbar(fig[1, 4], hm2)

ax3 = Axis(fig[1, 5], xlabel="Time [min]", ylabel="Height [km]",
           title="Ice [g/kg]")
hm3 = heatmap!(ax3, time_minutes, height_km, q_ice .* 1000,
               colormap=:reds, colorrange=(0, maximum(q_ice)*1000))
Colorbar(fig[1, 6], hm3)

# Row 2: Ice properties
ax4 = Axis(fig[2, 1], xlabel="Time [min]", ylabel="Height [km]",
           title="Rime Fraction")
hm4 = heatmap!(ax4, time_minutes, height_km, rime_fraction,
               colormap=:viridis, colorrange=(0, 1))
Colorbar(fig[2, 2], hm4)

ax5 = Axis(fig[2, 3], xlabel="Time [min]", ylabel="Height [km]",
           title="Liquid Fraction on Ice")
hm5 = heatmap!(ax5, time_minutes, height_km, liquid_fraction,
               colormap=:plasma, colorrange=(0, 1))
Colorbar(fig[2, 4], hm5)

ax6 = Axis(fig[2, 5], xlabel="Time [min]", ylabel="Height [km]",
           title="Reflectivity [dBZ]")
hm6 = heatmap!(ax6, time_minutes, height_km, reflectivity,
               colormap=:turbo, colorrange=(-10, 60))
Colorbar(fig[2, 6], hm6)

# Row 3: Dynamics and precip
ax7 = Axis(fig[3, 1], xlabel="Time [min]", ylabel="Height [km]",
           title="Vertical Velocity [m/s]")
hm7 = heatmap!(ax7, time_minutes, height_km, w,
               colormap=:RdBu, colorrange=(-5, 5))
Colorbar(fig[3, 2], hm7)

ax8 = Axis(fig[3, 3], xlabel="Time [min]", ylabel="Height [km]",
           title="Temperature [°C]")
hm8 = heatmap!(ax8, time_minutes, height_km, temperature_C,
               colormap=:thermal, colorrange=(-70, 30))
Colorbar(fig[3, 4], hm8)

ax9 = Axis(fig[3, 5:6], xlabel="Time [min]", ylabel="Precip Rate [mm/h]",
           title="Surface Precipitation")
lines!(ax9, time_minutes, prt_liq, label="Liquid", color=:blue)
lines!(ax9, time_minutes, prt_sol, label="Solid", color=:red)
axislegend(ax9, position=:rt)

# Save figure
save(joinpath(@__DIR__, "kin1d_reference_overview.png"), fig)
println("Saved: kin1d_reference_overview.png")

#####
##### Analyze the simulation physics
#####

println()
println("=== Simulation Analysis ===")
println()

# Find time of maximum ice
t_max_ice = time_minutes[argmax(maximum(q_ice, dims=2)[:, 1])]
z_max_ice = height_km[argmax(maximum(q_ice, dims=1)[1, :])]
println("Peak ice formation: t = $(round(t_max_ice, digits=1)) min, z = $(round(z_max_ice, digits=1)) km")

# Find time of maximum rain at surface (k=end is near surface)
t_max_rain_sfc = time_minutes[argmax(q_rain[:, end])]
println("Peak rain at surface: t = $(round(t_max_rain_sfc, digits=1)) min")

# Find when updraft turns off (w peaks then decreases)
max_w_time = time_minutes[argmax(maximum(w, dims=2)[:, 1])]
println("Peak updraft: t = $(round(max_w_time, digits=1)) min")

# Check ice crystal properties at peak ice time
i_peak = argmax(maximum(q_ice, dims=2)[:, 1])
j_peak = argmax(q_ice[i_peak, :])
println("At peak ice:")
println("  - Mean ice diameter: $(round(d_ice_cat1[i_peak, j_peak] * 1e6, digits=1)) μm")
println("  - Ice bulk density: $(round(rho_ice_cat1[i_peak, j_peak], digits=1)) kg/m³")
println("  - Rime fraction: $(round(rime_fraction[i_peak, j_peak], digits=2))")

#####
##### Discussion of comparison methodology
#####

println()
println("=" ^ 60)
println("COMPARISON METHODOLOGY")
println("=" ^ 60)
println()
println("""
The Fortran kin1d driver is a specialized 1D kinematic cloud model that:

1. PRESCRIBED DYNAMICS: The vertical velocity is prescribed, not computed.
   - Starts at 2 m/s, evolves to 5 m/s peak
   - Profile shape evolves with cloud top height
   - Updraft shuts off after 60 minutes

2. ADVECTION: Uses upstream differencing for vertical advection of all
   hydrometeor species, temperature, and moisture.

3. DIVERGENCE/COMPRESSIBILITY: Applies mass-weighted divergence corrections
   to maintain consistency with the prescribed w-profile.

4. MOISTURE SOURCE: Adds low-level moisture to prevent depletion.

5. P3 MICROPHYSICS: Calls the full P3 scheme at each 10s timestep.

To reproduce this in Breeze.jl would require:

a) Creating a specialized 1D kinematic driver (not the 3D LES framework)
b) Implementing prescribed velocity forcing
c) Matching the exact advection scheme (upstream differencing)
d) Matching the divergence/compressibility corrections

The current Breeze.jl P3 implementation provides the MICROPHYSICS TENDENCIES,
but the kin1d driver tests the complete MICROPHYSICS + ADVECTION + SEDIMENTATION
system integrated together.

For a fair comparison, we should compare:
- Individual process rates (autoconversion, accretion, etc.) in isolation
- Terminal velocities
- Size distribution parameters

The full kin1d comparison requires implementing the kinematic driver framework.
""")

#####
##### Quick process rate comparison (conceptual)
#####

println()
println("=" ^ 60)
println("P3 PROCESS RATE IMPLEMENTATION STATUS")
println("=" ^ 60)
println()

println("""
Breeze.jl P3 implements the following process rates (in process_rates.jl):

WARM RAIN (Khairoutdinov-Kogan 2000):
  ✅ rain_autoconversion_rate    - Cloud → Rain conversion
  ✅ rain_accretion_rate         - Cloud collection by rain
  ✅ rain_self_collection_rate   - Rain number reduction
  ✅ rain_evaporation_rate       - Subsaturated rain evaporation

ICE DEPOSITION/SUBLIMATION:
  ✅ ice_deposition_rate         - Vapor diffusion growth
  ✅ ventilation_enhanced_deposition - Large particle ventilation

MELTING:
  ✅ ice_melting_rate            - Ice → Rain at T > 0°C
  ✅ ice_melting_number_rate     - Number tendency from melting

ICE-ICE INTERACTIONS:
  ✅ ice_aggregation_rate        - Self-collection (number reduction)

RIMING:
  ✅ cloud_riming_rate           - Cloud droplet collection by ice
  ✅ rain_riming_rate            - Rain collection by ice
  ✅ rime_density                - Temperature-dependent rime density

LIQUID FRACTION:
  ✅ shedding_rate               - Liquid coating → rain
  ✅ refreezing_rate             - Liquid coating → ice below 0°C

ICE NUCLEATION:
  ✅ deposition_nucleation_rate  - Cooper (1986) parameterization
  ✅ immersion_freezing_cloud_rate - Bigg (1953) cloud freezing
  ✅ immersion_freezing_rain_rate  - Bigg (1953) rain freezing

SECONDARY ICE:
  ✅ rime_splintering_rate       - Hallett-Mossop (1974)

TERMINAL VELOCITIES:
  ✅ rain_terminal_velocity_mass_weighted
  ✅ rain_terminal_velocity_number_weighted
  ✅ ice_terminal_velocity_mass_weighted
  ✅ ice_terminal_velocity_number_weighted
  ✅ ice_terminal_velocity_reflectivity_weighted

NOT YET IMPLEMENTED:
  ❌ Lookup table integration (uses simplified parameterizations)
  ❌ Full size distribution integrals via tabulated values
  ❌ Complete Z tendencies for all processes
  ❌ Cloud droplet activation (aerosol coupling)
""")

println()
println("Reference data saved. For a complete comparison, implement a")
println("kinematic driver in Breeze.jl or compare isolated process rates.")
