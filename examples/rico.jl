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

# ## Sponge layer
#
# To prevent spurious wave reflections from the upper boundary, we add a Rayleigh
# damping sponge layer in the upper 500 m of the domain. The sponge damps vertical
# velocity toward zero using Oceananigans' `Relaxation` forcing with a `GaussianMask`.

sponge_rate = 1/60  # s⁻¹ - relaxation rate (60 s timescale)
sponge_mask = GaussianMask{:z}(center=4000, width=500)
sponge = Relaxation(rate=sponge_rate, mask=sponge_mask)

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


∂t_ρθ_large_scale = Field{Nothing, Nothing, Center}(grid)
∂t_θ_large_scale = - 2.5 / day # K / day
set!(∂t_ρθ_large_scale, ρᵣ * ∂t_θ_large_scale)
ρθ_large_scale_forcing = Forcing(∂t_ρθ_large_scale)

# ## Assembling forcing and boundary conditions

Fρu = (subsidence, geostrophic.ρu)
Fρv = (subsidence, geostrophic.ρv)
Fρw = sponge
Fρqᵗ = (subsidence, ρqᵗ_drying_forcing)
Fρθ = (subsidence, ρθ_large_scale_forcing)

forcing = (ρu=Fρu, ρv=Fρv, ρw=Fρw, ρqᵗ=Fρqᵗ, ρθ=Fρθ)
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

cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

# Default non-equilibrium cloud formation
# cloud_liquid = CloudMicrophysics.Parameters.CloudLiquid{Float32}(τ_relax=2, ρw=1000, r_eff=1.4e-5)
# cloud_formation = NonEquilibriumCloudFormation(cloud_liquid)
# microphysics = OneMomentCloudMicrophysics(; cloud_formation)

ninth_order_weno = WENO(order=5)
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

# We add a small random perturbation below 1500 m to trigger convection.

zϵ = 1500 # m

θᵢ(x, y, z) = θˡⁱ₀(z) + 1e-2 * (rand() - 0.5) * (z < zϵ)
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

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, wall time: %s, max|w|: %.2e m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), wmax)

    msg *= @sprintf(", max(qᵗ): %.2e, max(qᶜˡ): %.2e, extrema(qʳ): (%.2e, %.2e), ∫PdV: %.2e kg/kg/s",
                    qᵗmax, qᶜˡmax, qʳmin, qʳmax, ∫P)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

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
# - xz-slices of qᶜˡ (cloud liquid) and qʳ (rain mass fraction)
# - xy-slice of w (vertical velocity) with qˡ contours overlaid

w = model.velocities.w

z = Oceananigans.Grids.znodes(grid, Center())
k_cloud = searchsortedfirst(z, 1500)  # cloud layer height for RICO
@info "Saving xy slices at z = $(z[k_cloud]) m (k = $k_cloud)"

slice_outputs = (
    qᶜˡxz = view(qᶜˡ, :, 1, :),
    qʳxz = view(qʳ, :, 1, :),
    wxy = view(w, :, :, k_cloud),
    qˡxy = view(qˡ, :, :, k_cloud),
    qʳxy = view(qʳ, :, :, 1),
)

filename = "rico_slices.jld2"
output_interval = 20seconds
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs; filename,
                                                schedule = TimeInterval(output_interval),
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

# ## Animation: cloud structure and dynamics
#
# We create a 4-panel animation showing:
# - Top left: xz-slice of cloud liquid water qᶜˡ
# - Top right: xz-slice of rain mass fraction qʳ
# - Bottom: xy-slice of vertical velocity w with qˡ contours overlaid

qᶜˡxz_ts = FieldTimeSeries("rico_slices.jld2", "qᶜˡxz")
qʳxz_ts = FieldTimeSeries("rico_slices.jld2", "qʳxz")
wxy_ts = FieldTimeSeries("rico_slices.jld2", "wxy")
qˡxy_ts = FieldTimeSeries("rico_slices.jld2", "qˡxy")
qʳxy_ts = FieldTimeSeries("rico_slices.jld2", "qʳxy")

times = qᶜˡxz_ts.times
Nt = length(times)

qᶜˡlim = maximum(qᶜˡxz_ts) / 4
qʳlim = maximum(qʳxz_ts) / 4
wlim = maximum(abs, wxy_ts) / 2
qˡcontour = maximum(qˡxy_ts) / 8  # threshold for cloud contours

# Now let's plot the slices and animate them.

fig = Figure(size=(900, 850), fontsize=14)

axqᶜˡxz = Axis(fig[2, 1], aspect=2, ylabel="z (m)", xaxisposition=:top)
axqʳxz = Axis(fig[2, 2], aspect=2, ylabel="z (m)", yaxisposition=:right, xaxisposition=:top)
axwxy = Axis(fig[3, 1], aspect=1, xlabel="x (m)", ylabel="y (m)")
axqʳxy = Axis(fig[3, 2], aspect=1, xlabel="x (m)", ylabel="y (m)", yaxisposition=:right)

hidexdecorations!(axqᶜˡxz)
hidexdecorations!(axqʳxz)

n = Observable(1)
qᶜˡxz_n = @lift qᶜˡxz_ts[$n]
qʳxz_n = @lift qʳxz_ts[$n]
wxy_n = @lift wxy_ts[$n]
qˡxy_n = @lift qˡxy_ts[$n]
qʳxy_n = @lift qʳxy_ts[$n]
title = @lift @sprintf("Clouds, rain, and updrafts in RICO at t = %16.3f hours", times[$n] / hour)

hmqᶜˡ = heatmap!(axqᶜˡxz, qᶜˡxz_n, colormap=:dense, colorrange=(0, qᶜˡlim))
hmqʳ = heatmap!(axqʳxz, qʳxz_n, colormap=:amp, colorrange=(0, qʳlim))

hmw = heatmap!(axwxy, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
contour!(axwxy, qˡxy_n, levels=[qˡcontour], color=(:black, 0.3), linewidth=3)

hmqʳ = heatmap!(axqʳxy, qʳxy_n, colormap=:amp, colorrange=(0, qʳlim))
contour!(axqʳxy, qˡxy_n, levels=[qˡcontour], color=(:black, 0.3), linewidth=3)

Colorbar(fig[1, 1], hmqᶜˡ, vertical=false, flipaxis=true, label="Cloud liquid qᶜˡ (x, y=0, z)")
Colorbar(fig[1, 2], hmqʳ, vertical=false, flipaxis=true, label="Rain mass fraction qʳ (x, y=0, z)")
Colorbar(fig[4, 1], hmw, vertical=false, flipaxis=false, label="Vertical velocity w (x, y, z=$(z[k_cloud])) with qˡ contours")
Colorbar(fig[4, 2], hmqʳ, vertical=false, flipaxis=false, label="Rain mass fraction qʳ (x, y, z=0)")

fig[0, :] = Label(fig, title, fontsize=18, tellwidth=false)

rowgap!(fig.layout, 2, -60)
rowgap!(fig.layout, 3, -80)

n₁ = floor(Int, 30minutes / output_interval)
n₂ = ceil(Int, 3hours / output_interval)

CairoMakie.record(fig, "rico_slices.mp4", n₁:n₂, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](rico_slices.mp4)
