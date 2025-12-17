# # Precipitating shallow cumulus convection (RICO)
#
# This example simulates precipitating shallow cumulus convection following the
# Rain in Cumulus over the Ocean (RICO) intercomparison case [vanZanten2011](@cite).
# RICO is a canonical test case for large eddy simulations of trade-wind cumulus
# with active warm-rain microphysics.
#
# The case is based on observations from the RICO field campaign conducted in the
# winter of 2004-2005 near Antigua and Barbuda in the Caribbean. Unlike BOMEX,
# which is non-precipitating, RICO produces drizzle and light rain from shallow
# cumulus clouds. The intercomparison study by [vanZanten2011](@citet) brought
# together results from multiple large eddy simulation codes to establish benchmark
# statistics for precipitating shallow cumulus.
#
# Initial and boundary conditions for this case are provided by the wonderfully useful
# package [AtmosphericProfilesLibrary.jl](https://github.com/CliMA/AtmosphericProfilesLibrary.jl).
# For precipitation we use the 1-moment scheme from
# [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl), which provides
# prognostic rain mass with autoconversion and accretion processes.

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units

using AtmosphericProfilesLibrary
using CairoMakie
using CloudMicrophysics
using Printf
using Random
using CUDA

Random.seed!(42)

# ## Domain and grid
#
# The RICO domain is 12.8 km × 12.8 km horizontally with a vertical extent of 4 km
# [vanZanten2011](@cite). The intercomparison uses 128 × 128 × 100 grid points
# with 100 m horizontal resolution and 40 m vertical resolution.

Oceananigans.defaults.FloatType = Float32

Nx = Ny = 128
Nz = 100

x = y = (0, 12800)
z = (0, 4000)

grid = RectilinearGrid(GPU(); x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state and formulation
#
# We use the anelastic formulation with a dry adiabatic reference state.
# The surface potential temperature ``θ_0 = 297.9`` K and surface pressure
# ``p_0 = 1015.4`` hPa are taken from [vanZanten2011](@citet).

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = 101540,
                                 potential_temperature = 297.9)

formulation = AnelasticFormulation(reference_state,
                                   thermodynamics = :LiquidIcePotentialTemperature)

# ## Surface fluxes
#
# Unlike the BOMEX protocol, which prescribes momentum, moisture, and thermodynamic fluxes,
# the RICO protocol decrees the computation of fluxes by bulk aerodynamic formulae
# with constant transfer coefficients (see [vanZanten2011](@citet); text surrounding equations 1-4):

Cᴰ = 1.229e-3 # Drag coefficient for momentum
Cᵀ = 1.094e-3 # "Temperature" aka sensible heat transfer coefficient
Cᵛ = 1.133e-3 # Moisture flux transfer coefficient
T₀ = 299.8    # Sea surface temperature (K)

# We implement the specified bulk formula with Breeze utilities whose scope
# currently extends only to constant coefficients (but could expand in the future),

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=T₀)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=T₀)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)

ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

# Within the canon of Monin-Obukhov similarity theory, these transfer
# coefficients should be scaled if the vertical grid spacing is changed.
# Here we can use the values from [vanZanten2011](@citet) verbatim because
# we use the recommended vertical grid spacing of 40 m.

# ## Large-scale subsidence
#
# The RICO protocol includes large-scale subsidence that advects mean profiles downward.
# The subsidence velocity profile increases linearly to ``-0.005`` m/s at 2260 m and
# remains constant above [vanZanten2011](@cite),

FT = eltype(grid)
wˢ_profile = AtmosphericProfilesLibrary.Rico_subsidence(FT)
wˢ = Field{Nothing, Nothing, Face}(grid)
set!(wˢ, z -> wˢ_profile(z))
subsidence = SubsidenceForcing(wˢ)

# This is what it looks like:

lines(wˢ; axis = (xlabel = "wˢ (m/s)",))

# ## Geostrophic forcing
#
# The momentum equations include a Coriolis force with prescribed geostrophic wind.
# The RICO Coriolis parameter corresponds to latitude around 18°N: ``f = 4.5 \times 10^{-5}`` s⁻¹.

coriolis = FPlane(f=4.5e-5)

uᵍ = AtmosphericProfilesLibrary.Rico_geostrophic_ug(FT)
vᵍ = AtmosphericProfilesLibrary.Rico_geostrophic_vg(FT)
geostrophic = geostrophic_forcings(z -> uᵍ(z), z -> vᵍ(z))

# ## Moisture tendency
#
# A prescribed large-scale moisture tendency represents the effects of advection
# by the large-scale circulation [vanZanten2011](@cite).

ρᵣ = formulation.reference_state.density
drying = Field{Nothing, Nothing, Center}(grid)
dqdt_profile = AtmosphericProfilesLibrary.Rico_dqtdt(FT)
set!(drying, z -> dqdt_profile(z))
set!(drying, ρᵣ * drying)
ρqᵗ_drying_forcing = Forcing(drying)

# ## Radiative cooling
#
# A prescribed radiative cooling profile is applied to the thermodynamic equation.
# The RICO case uses a constant radiative cooling rate of ``-2.5`` K/day
# applied uniformly throughout the domain [vanZanten2011](@cite).
# This is the key simplification that allows us to avoid interactive radiation.

cooling = Field{Nothing, Nothing, Center}(grid)
dTdt_rico = AtmosphericProfilesLibrary.Rico_dTdt(FT)
cᵖᵈ = constants.dry_air.heat_capacity
set!(cooling, z -> dTdt_rico(1, z))
set!(cooling, ρᵣ * cᵖᵈ * cooling)
ρe_radiation_forcing = Forcing(cooling)

# ## Assembling forcing and boundary conditions

Fρu = (subsidence, geostrophic.ρu)
Fρv = (subsidence, geostrophic.ρv)
Fρqᵗ = (subsidence, ρqᵗ_drying_forcing)
Fρθ = subsidence
Fρe = ρe_radiation_forcing

forcing = (ρu=Fρu, ρv=Fρv, ρqᵗ=Fρqᵗ, ρe=Fρe)
boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)
nothing #hide

# ## Model setup
#
# We use one-moment bulk microphysics from CloudMicrophysics with warm-phase saturation adjustment
# and 9th-order WENO advection. The one-moment scheme tracks prognostic rain mass (`qʳ`) 
# and includes autoconversion (cloud liquid → rain) and accretion (cloud liquid swept up
# by falling rain) processes. This is a more physically-realistic representation of
# warm-rain precipitation than the zero-moment scheme.

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

microphysics = OneMomentCloudMicrophysics()

ninth_order_weno = WENO(order=9)
bounds_preserving_weno = WENO(order=5, bounds=(0, 1))

momentum_advection = ninth_order_weno
scalar_advection = (ρθ = ninth_order_weno,
                    ρqᵗ = bounds_preserving_weno,
                    ρqᶜˡ = bounds_preserving_weno,
                    ρqʳ = bounds_preserving_weno)

model = AtmosphereModel(grid; formulation, coriolis, microphysics,
                        momentum_advection, scalar_advection, forcing, boundary_conditions)

# ## Initial conditions
#
# Mean profiles are specified as piecewise linear functions by [vanZanten2011](@citet):
#
#    - Liquid-ice potential temperature ``θ^{\ell i}(z)``
#    - Total water specific humidity ``q^t(z)``
#    - Zonal velocity ``u(z)`` and meridional velocity ``v(z)``
#
# The profiles are implemented in the wonderfully useful
# [AtmosphericProfilesLibrary](https://github.com/CliMA/AtmosphericProfilesLibrary.jl)
# package developed by the Climate Modeling Alliance,

θˡⁱ₀ = AtmosphericProfilesLibrary.Rico_θ_liq_ice(FT)
qᵗ₀ = AtmosphericProfilesLibrary.Rico_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.Rico_u(FT)
v₀ = AtmosphericProfilesLibrary.Rico_v(FT)

# We dutifully apply a correction to the Exner function due to the fact that
# Breeze does not currently distinguish between the surface pressure and the
# standard "potential temperature reference pressure" of ``10⁵`` Pa,

using Breeze.Thermodynamics: dry_air_gas_constant

Rᵈ = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
p₀ = reference_state.surface_pressure
χ = (p₀ / 1e5)^(Rᵈ / cᵖᵈ)
zϵ = 1500 # m

θᵢ(x, y, z) = χ * θˡⁱ₀(z) + 1e-2 * (rand() - 0.5) * (z < zϵ)
qᵢ(x, y, z) = qᵗ₀(z)
uᵢ(x, y, z) = u₀(z)
vᵢ(x, y, z) = v₀(z)

set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ, v=vᵢ)

# ## Simulation
#
# We run the simulation for 12 hours with adaptive time-stepping.
# RICO typically requires longer integration times than BOMEX to develop
# a quasi-steady precipitating state.

simulation = Simulation(model; Δt=10, stop_time=6hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Output and progress
#
# We set up a progress callback with hourly messages about interesting
# quantities,

θ = liquid_ice_potential_temperature(model)
qˡ = model.microphysical_fields.qˡ    # total liquid (cloud + rain)
qᶜˡ = model.microphysical_fields.qᶜˡ  # cloud liquid only
qᵛ = model.microphysical_fields.qᵛ
qʳ = model.microphysical_fields.qʳ    # rain mass fraction (diagnostic)
ρqʳ = model.microphysical_fields.ρqʳ 
ρqʳ = model.microphysical_fields.ρqʳ  # rain mass density (prognostic)

## Precipitation rate diagnostic from one-moment microphysics
P = precipitation_rate(model, :liquid)

## Integrals of precipitation rate
∫Pdz = Field(Integral(P, dims=3))
∫PdV = Field(Integral(P))

## For keeping track of the computational expense
wall_clock = Ref(time_ns())

function progress(sim)
    compute!(∫PdV)
    qᶜˡmax = maximum(qᶜˡ)
    qʳmax = maximum(qʳ)
    qʳmin = minimum(qʳ)
    qᵗmax = maximum(sim.model.specific_moisture)
    wmax = maximum(abs, model.velocities.w)
    ∫P = CUDA.@allowscalar ∫PdV[]
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, wall time: %s, max|w|: %.2e m/s \n",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), wmax)

    msg *= @sprintf(" --- max(qᵗ): %.2e, max(qᶜˡ): %.2e, extrema(qʳ): (%.2e, %.2e), ∫PdV: %.2e kg/kg/s",
                    qᵗmax, qᶜˡmax, qʳmin, qʳmax, ∫P)

    @info msg

    return nothing
end

add_callback!(simulation, progress, TimeInterval(1hour))

# In addition to velocities, we output horizontal and time-averages of
# liquid water mass fraction (cloud and rain separately), specific humidity,
# and liquid-ice potential temperature,

outputs = merge(model.velocities, (; θ, qᶜˡ, qʳ, qᵛ))
averaged_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename = "rico.jld2"
simulation.output_writers[:averages] = JLD2Writer(model, averaged_outputs; filename,
                                                  schedule = AveragedTimeInterval(1hour),
                                                  overwrite_existing = true)

# For an animation, we also output slices,
#
# - xz-slices of qˡ and precipitation rate
# - xy-slice of qˡ in cloud layer (z ≈ 1500 m) and vertically-integrated precipitation rate

z = Oceananigans.Grids.znodes(grid, Center())
k_cloud = searchsortedfirst(z, 1500)  # cloud layer height for RICO
@info "Saving xy slices at z = $(z[k_cloud]) m (k = $k_cloud)"

slice_outputs = (
    qˡxz = view(qˡ, :, 1, :),
    Pxz = view(P, :, 1, :),
    qˡxy = view(qˡ, :, :, k_cloud),
    ∫P = ∫Pdz,
)

filename = "rico_slices.jld2"
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs; filename,
                                                schedule = TimeInterval(30seconds),
                                                overwrite_existing = true)

# We're finally ready to run this thing,

run!(simulation)

# ## Results: mean profile evolution
#
# We visualize the evolution of horizontally-averaged profiles every hour.

averages_filename = "rico.jld2"
θt = FieldTimeSeries(averages_filename, "θ")
qᵛt = FieldTimeSeries(averages_filename, "qᵛ")
qᶜˡt = FieldTimeSeries(averages_filename, "qᶜˡ")
qʳt = FieldTimeSeries(averages_filename, "qʳ")
ut = FieldTimeSeries(averages_filename, "u")
vt = FieldTimeSeries(averages_filename, "v")

fig = Figure(size=(900, 800), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="qᵛ (kg/kg)", ylabel="z (m)")
axuv = Axis(fig[2, 1], xlabel="u, v (m/s)", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="qᶜˡ, qʳ (kg/kg)", ylabel="z (m)")

times = θt.times
Nt = length(times)

default_colours = Makie.wong_colors()
colors = [default_colours[mod1(i, length(default_colours))] for i in 1:Nt]

for n in 1:3:Nt
    label = n == 1 ? "initial condition" : "mean over $(Int(times[n-1]/hour))-$(Int(times[n]/hour)) hr"

    lines!(axθ, θt[n], color=colors[n], label=label)
    lines!(axq, qᵛt[n], color=colors[n])
    lines!(axuv, ut[n], color=colors[n], linestyle=:solid)
    lines!(axuv, vt[n], color=colors[n], linestyle=:dash)
    lines!(axqˡ, qᶜˡt[n], color=colors[n], linestyle=:solid)  # cloud liquid
    lines!(axqˡ, qʳt[n], color=colors[n], linestyle=:dash)    # rain
end

# Set axis limits to focus on the boundary layer
for ax in (axθ, axq, axuv, axqˡ)
    ylims!(ax, 0, 3500)
end

xlims!(axθ, 296, 318)
xlims!(axq, 0, 18e-3)
xlims!(axuv, -12, 2)

# Add legends and annotations
axislegend(axθ, position=:rb)
text!(axuv, -10, 3200, text="solid: u\ndashed: v", fontsize=12)
text!(axqˡ, 0.5e-4, 3200, text="solid: qᶜˡ\ndashed: qʳ", fontsize=12)

fig[0, :] = Label(fig, "RICO: Horizontally-averaged profiles", fontsize=18, tellwidth=false)

save("rico_profiles.png", fig)
fig

# The simulation shows the development of a cloudy, precipitating boundary layer with:
# - Deeper cloud layer than BOMEX (tops reaching ~2.5-3 km)
# - Higher moisture content supporting warm-rain processes
# - Trade-wind flow with stronger westerlies
# - Distinct profiles of cloud liquid (qᶜˡ) and rain (qʳ) as in [vanZanten2011](@citet)

# ## Animation: cloud liquid water and precipitation rate
#
# We create a 4-panel animation showing:
# - Top left: xz-slice of cloud liquid water qˡ
# - Top right: xz-slice of precipitation rate P
# - Bottom left: xy-slice of qˡ in the cloud layer
# - Bottom right: vertically-integrated precipitation rate

qˡxz_ts = FieldTimeSeries("rico_slices.jld2", "qˡxz")
Pxz_ts = FieldTimeSeries("rico_slices.jld2", "Pxz")
qˡxy_ts = FieldTimeSeries("rico_slices.jld2", "qˡxy")
∫P_ts = FieldTimeSeries("rico_slices.jld2", "∫P")

z = znodes(qˡxz_ts.grid, Center())

times = qˡxz_ts.times
Nt = length(times)

# Compute color ranges (with fallback to avoid zero range which breaks Makie)
qˡlim = max(maximum(qˡxz_ts), 1e-6) / 4
Plim = max(maximum(Pxz_ts), 1e-10) / 4
∫Plim = max(maximum(∫P_ts), 1e-8) / 4

# Now let's plot the slices and animate them.

fig = Figure(size=(900, 800), fontsize=14)

axqxz = Axis(fig[2, 1], aspect=2, ylabel="z (m)", xaxisposition=:top)
axPxz = Axis(fig[2, 2], aspect=2, ylabel="z (m)", yaxisposition=:right, xaxisposition=:top)
axqxy = Axis(fig[3, 1], aspect=1, xlabel="x (m)", ylabel="y (m)") 
ax∫P  = Axis(fig[3, 2], aspect=1, xlabel="x (m)", ylabel="y (m)", yaxisposition=:right)

hidexdecorations!(axqxz)
hidexdecorations!(axPxz)

n = Observable(1)
qˡxz_n = @lift qˡxz_ts[$n]
Pxz_n = @lift Pxz_ts[$n]
qˡxy_n = @lift qˡxy_ts[$n]
∫P_n = @lift ∫P_ts[$n]
title = @lift "Cloud liquid and precipitation in RICO at t = " * prettytime(times[$n])

hmq1 = heatmap!(axqxz, qˡxz_n, colormap=:dense, colorrange=(0, qˡlim))
hmP1 = heatmap!(axPxz, Pxz_n, colormap=:amp, colorrange=(0, Plim))
hmq2 = heatmap!(axqxy, qˡxy_n, colormap=:dense, colorrange=(0, qˡlim))
hmP2 = heatmap!(ax∫P, ∫P_n, colormap=:amp, colorrange=(0, ∫Plim))

Colorbar(fig[1, 1], hmq1, vertical=false, flipaxis=true, label="Cloud liquid water qˡ (x, y=0, z)")
Colorbar(fig[1, 2], hmP1, vertical=false, flipaxis=true, label="Precipitation rate P (x, y=0, z)")
Colorbar(fig[4, 1], hmq2, vertical=false, flipaxis=false, label="Cloud liquid water qˡ (x, y, z=$(z[k_cloud]))")
Colorbar(fig[4, 2], hmP2, vertical=false, flipaxis=false, label="Column-integrated precipitation rate")

fig[0, :] = Label(fig, title, fontsize=18, tellwidth=false)

rowgap!(fig.layout, 2, -80)
rowgap!(fig.layout, 3, -100)
rowgap!(fig.layout, 4, 0)

CairoMakie.record(fig, "rico_slices.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](rico_slices.mp4)
