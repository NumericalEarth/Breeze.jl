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

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units

using AtmosphericProfilesLibrary
using CairoMakie
using CloudMicrophysics
using Printf
using Random

Random.seed!(42)

# ## Domain and grid
#
# The RICO domain is 12.8 km × 12.8 km horizontally with a vertical extent of 4 km
# ([vanZanten2011](@citet)). The intercomparison uses 256 × 256 × 100 grid points
# with 50 m horizontal resolution and 40 m vertical resolution.
#
# For this example, we use a coarser grid (64 × 64 × 100) with 200 m horizontal
# resolution, suitable for development and testing.

Oceananigans.defaults.FloatType = Float32

Nx = Ny = 64
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
                                 base_pressure = 101540,
                                 potential_temperature = 297.9)

formulation = AnelasticFormulation(reference_state,
                                   thermodynamics = :LiquidIcePotentialTemperature)

# ## Surface fluxes
#
# RICO prescribes constant surface sensible and latent heat fluxes
# ([vanZanten2011](@citet)):
# - Sensible heat flux: ``\overline{w'\theta'}|_0 \approx 8 \times 10^{-3}`` K m/s
# - Moisture flux: ``\overline{w'q_t'}|_0 \approx 5.2 \times 10^{-5}`` kg/kg m/s
#
# These values are similar to BOMEX but produce a moister boundary layer
# that supports warm-rain processes.

w′θ′ = 8e-3     # K m/s (sensible heat flux)
w′qᵗ′ = 5.2e-5  # kg/kg m/s (moisture flux)

FT = eltype(grid)
p₀ = reference_state.base_pressure
θ₀ = reference_state.potential_temperature
q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
ρ₀ = Breeze.Thermodynamics.density(p₀, θ₀, q₀, constants)

ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′θ′))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′qᵗ′))

# ## Surface momentum flux (drag)
#
# A bulk drag parameterization is applied with friction velocity
# ``u_* = 0.28`` m/s ([vanZanten2011](@citet)).

u★ = 0.28  # m/s
@inline ρu_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / sqrt(ρu^2 + ρv^2)
@inline ρv_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / sqrt(ρu^2 + ρv^2)

ρu_drag_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρv_drag_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρu_bcs = FieldBoundaryConditions(bottom=ρu_drag_bc)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_drag_bc)

# ## Large-scale subsidence
#
# The RICO case includes large-scale subsidence that advects mean profiles downward.
# The subsidence velocity profile increases linearly to ``-0.005`` m/s at 2260 m and
# remains constant above ([vanZanten2011](@citet)).

wˢ = Field{Nothing, Nothing, Face}(grid)
wˢ_profile = AtmosphericProfilesLibrary.Rico_subsidence(FT)
set!(wˢ, z -> wˢ_profile(z))

# Visualize the subsidence profile:

lines(wˢ; axis = (xlabel = "wˢ (m/s)",))

# Subsidence is implemented as an advection of the horizontally-averaged prognostic variables.

subsidence = SubsidenceForcing(wˢ)

# ## Geostrophic forcing
#
# The momentum equations include a Coriolis force with prescribed geostrophic wind.
# The RICO Coriolis parameter corresponds to latitude ~18°N: ``f = 4.5 \times 10^{-5}`` s⁻¹.

coriolis = FPlane(f=4.5e-5)

uᵍ = AtmosphericProfilesLibrary.Rico_geostrophic_ug(FT)
vᵍ = AtmosphericProfilesLibrary.Rico_geostrophic_vg(FT)
geostrophic = geostrophic_forcings(z -> uᵍ(z), z -> vᵍ(z))

# ## Moisture tendency
#
# A prescribed large-scale moisture tendency represents the effects of advection
# by the large-scale circulation ([vanZanten2011](@citet)).

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
# ([vanZanten2011](@citet)), applied uniformly throughout the domain.
# This is the key simplification that allows us to avoid interactive radiation.

Fρe_field = Field{Nothing, Nothing, Center}(grid)
cᵖᵈ = constants.dry_air.heat_capacity
dTdt_rico = AtmosphericProfilesLibrary.Rico_dTdt(FT)
set!(Fρe_field, z -> dTdt_rico(1, z))
set!(Fρe_field, ρᵣ * cᵖᵈ * Fρe_field)
ρe_radiation_forcing = Forcing(Fρe_field)

# ## Assembling all the forcings

ρu_forcing = (subsidence, geostrophic.ρu)
ρv_forcing = (subsidence, geostrophic.ρv)
ρqᵗ_forcing = (ρqᵗ_drying_forcing, subsidence)
ρθ_forcing = subsidence
ρe_forcing = ρe_radiation_forcing

forcing = (; ρu=ρu_forcing, ρv=ρv_forcing, ρθ=ρθ_forcing,
             ρe=ρe_forcing, ρqᵗ=ρqᵗ_forcing)
nothing #hide

# ## Model setup
#
# We use zero-moment bulk microphysics from CloudMicrophysics with warm-phase saturation adjustment
# and 9th-order WENO advection. The zero-moment scheme allows condensation and evaporation
# and includes instant precipitation removal above a threshold, making it suitable for
# precipitating shallow cumulus simulations like RICO.

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: ZeroMomentCloudMicrophysics

microphysics = ZeroMomentCloudMicrophysics()
advection = WENO(order=9)

model = AtmosphereModel(grid; formulation, coriolis, microphysics, advection, forcing,
                        boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

# ## Initial conditions
#
# Mean profiles are specified as piecewise linear functions by [vanZanten2011](@citet):
#    - Liquid-ice potential temperature ``θ^{\ell i}(z)``
#    - Total water specific humidity ``q^t(z)``
#    - Zonal velocity ``u(z)`` and meridional velocity ``v(z)``

FT = eltype(grid)
θˡⁱ₀ = AtmosphericProfilesLibrary.Rico_θ_liq_ice(FT)
qᵗ₀ = AtmosphericProfilesLibrary.Rico_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.Rico_u(FT)
v₀ = AtmosphericProfilesLibrary.Rico_v(FT)

# Apply Exner function correction for Breeze's reference pressure convention:

using Breeze.Thermodynamics: dry_air_gas_constant

Rᵈ = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
p₀ = reference_state.base_pressure
χ = (p₀ / 1e5)^(Rᵈ / cᵖᵈ)

# The initial profiles are perturbed with random noise below 1600 m to trigger
# convection. Similar perturbation amplitudes as BOMEX are used:
#
# - Potential temperature perturbation: ``δθ = 0.1`` K
# - Moisture perturbation: ``δqᵗ = 2.5 \times 10^{-5}`` kg/kg

δθ = 0.1      # K
δqᵗ = 2.5e-5  # kg/kg
zδ = 1600     # m

ϵ() = rand() - 1/2
θᵢ(x, y, z) = χ * θˡⁱ₀(z) + δθ  * ϵ() * (z < zδ)
qᵢ(x, y, z) = qᵗ₀(z)  + δqᵗ * ϵ() * (z < zδ)
uᵢ(x, y, z) = u₀(z)
vᵢ(x, y, z) = v₀(z)

set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ, v=vᵢ)

# ## Simulation
#
# We run the simulation for 6 hours with adaptive time-stepping.
# RICO typically requires longer integration times than BOMEX to develop
# a quasi-steady precipitating state, but we use 6 hours for consistency with BOMEX.

simulation = Simulation(model; Δt=10, stop_time=6hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Output and progress

θ = liquid_ice_potential_temperature(model)
qˡ = model.microphysical_fields.qˡ
qᵛ = model.microphysical_fields.qᵛ

u_avg = Field(Average(model.velocities.u, dims=(1, 2)))
v_avg = Field(Average(model.velocities.v, dims=(1, 2)))

function progress(sim)
    compute!(u_avg)
    compute!(v_avg)
    qˡmax = maximum(qˡ)
    qᵗmax = maximum(sim.model.specific_moisture)
    umax = maximum(abs, u_avg)
    vmax = maximum(abs, v_avg)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|ū|: (%.2e, %.2e), max(qᵗ): %.2e, max(qˡ): %.2e",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   umax, vmax, qᵗmax, qˡmax)
    @info msg
    return nothing
end

add_callback!(simulation, progress, TimeInterval(1hour))

outputs = merge(model.velocities, model.tracers, (; θ, qˡ, qᵛ))
averaged_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename = "rico.jld2"
simulation.output_writers[:averages] = JLD2Writer(model, averaged_outputs; filename,
                                                  schedule = AveragedTimeInterval(1hour),
                                                  overwrite_existing = true)

# Output horizontal slices at z = 1000 m for animation (near cloud base for RICO)
z = Oceananigans.Grids.znodes(grid, Center())
k = searchsortedfirst(z, 1000)
@info "Saving slices at z = $(z[k]) m (k = $k)"

u, v, w = model.velocities
slice_outputs = (
    wxy = view(w, :, :, k),
    qˡxy = view(qˡ, :, :, k),
    wxz = view(w, :, 1, :),
    qˡxz = view(qˡ, :, 1, :),
)

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = "rico_slices.jld2",
                                                schedule = TimeInterval(30seconds),
                                                overwrite_existing = true)

@info "Running RICO simulation..."
run!(simulation)

# ## Results: mean profile evolution
#
# We visualize the evolution of horizontally-averaged profiles every hour.

θt = FieldTimeSeries(filename, "θ")
qᵛt = FieldTimeSeries(filename, "qᵛ")
qˡt = FieldTimeSeries(filename, "qˡ")
ut = FieldTimeSeries(filename, "u")
vt = FieldTimeSeries(filename, "v")

fig = Figure(size=(900, 800), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="qᵛ (kg/kg)", ylabel="z (m)")
axuv = Axis(fig[2, 1], xlabel="u, v (m/s)", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="qˡ (kg/kg)", ylabel="z (m)")

times = θt.times
Nt = length(times)

default_colours = Makie.wong_colors()
colors = [default_colours[mod1(i, length(default_colours))] for i in 1:Nt]

for n in 1:Nt
    label = n == 1 ? "initial condition" : "mean over $(Int(times[n-1]/hour))-$(Int(times[n]/hour)) hr"

    lines!(axθ, θt[n], color=colors[n], label=label)
    lines!(axq, qᵛt[n], color=colors[n])
    lines!(axuv, ut[n], color=colors[n], linestyle=:solid)
    lines!(axuv, vt[n], color=colors[n], linestyle=:dash)
    lines!(axqˡ, qˡt[n], color=colors[n])
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

fig[0, :] = Label(fig, "RICO: Mean profile evolution (van Zanten et al., 2011)", fontsize=18, tellwidth=false)

save("rico_profiles.png", fig)
fig

# The simulation shows the development of a cloudy, precipitating boundary layer with:
# - Deeper cloud layer than BOMEX (tops reaching ~2.5-3 km)
# - Higher moisture content supporting warm-rain processes
# - Trade-wind flow with stronger westerlies

# ## Animation of horizontal slices

wxz_ts = FieldTimeSeries("rico_slices.jld2", "wxz")
qˡxz_ts = FieldTimeSeries("rico_slices.jld2", "qˡxz")
wxy_ts = FieldTimeSeries("rico_slices.jld2", "wxy")
qˡxy_ts = FieldTimeSeries("rico_slices.jld2", "qˡxy")

times = wxz_ts.times
Nt = length(times)

slices_fig = Figure(size=(1000, 500), fontsize=14)
axwxz = Axis(slices_fig[1, 2], xlabel="x (m)", ylabel="z (m)", title="Vertical velocity w")
axqxz = Axis(slices_fig[1, 3], xlabel="x (m)", ylabel="z (m)", title="Liquid water qˡ")
axwxy = Axis(slices_fig[2, 2], xlabel="x (m)", ylabel="y (m)")
axqxy = Axis(slices_fig[2, 3], xlabel="x (m)", ylabel="y (m)")

wmax = maximum(abs, wxz_ts)
qˡmax = maximum(qˡxz_ts)

n = Observable(1)
wxz_n = @lift wxz_ts[$n]
qˡxz_n = @lift qˡxz_ts[$n]
wxy_n = @lift wxy_ts[$n]
qˡxy_n = @lift qˡxy_ts[$n]
title_text = @lift "RICO slices at t = " * prettytime(times[$n])

hmw = heatmap!(axwxz, wxz_n, colormap=:balance, colorrange=(-wmax, wmax))
hmq = heatmap!(axqxz, qˡxz_n, colormap=:dense, colorrange=(0, qˡmax))
hmw = heatmap!(axwxy, wxy_n, colormap=:balance, colorrange=(-wmax, wmax))
hmq = heatmap!(axqxy, qˡxy_n, colormap=:dense, colorrange=(0, qˡmax))

Colorbar(slices_fig[1:2, 1], hmw, label="w (m/s)", flipaxis=false)
Colorbar(slices_fig[1:2, 4], hmq, label="qˡ (kg/kg)")

slices_fig[0, :] = Label(slices_fig, title_text, fontsize=18, tellwidth=false)

CairoMakie.record(slices_fig, "rico_slices.mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end
nothing #hide

# ![](rico_slices.mp4)
nothing #hide

# ![](rico_slices.mp4)

