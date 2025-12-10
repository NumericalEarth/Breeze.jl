# # Shallow cumulus convection (BOMEX)
#
# This example simulates shallow cumulus convection following the Barbados Oceanographic
# and Meteorological Experiment (BOMEX) intercomparison case [Siebesma2003](@cite).
# BOMEX is a canonical test case for large eddy simulations of shallow cumulus
# convection over a subtropical ocean.
#
# The case is based on observations from the Barbados Oceanographic and Meteorological
# Experiment, which documented the structure and organization of trade-wind cumulus
# clouds. The intercomparison study by [Siebesma2003](@citet) brought together results
# from 10 different large eddy simulation codes to establish benchmark statistics.
#
# Initial and boundary conditions for this case are provided by the wonderfully useful
# package [AtmosphericProfilesLibrary.jl](https://github.com/CliMA/AtmosphericProfilesLibrary.jl).

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans: Oceananigans

using AtmosphericProfilesLibrary
using CairoMakie
using CUDA
using Printf
using Random

Random.seed!(938)

# ## Domain and grid
#
# The BOMEX domain is 6.4 km × 6.4 km horizontally with a vertical extent of 3 km
# ([Siebesma2003](@citet); Section 3a). The intercomparison uses
# 64 × 64 × 75 grid points with 100 m horizontal resolution and 40 m vertical resolution.
#
# For this documentation example, we reduce the numerical precision to Float32.
# This yields a 10x speed up on an NVidia T4 (which is used to build the docs).

Oceananigans.defaults.FloatType = Float32

Nx = Ny = 64
Nz = 75

x = y = (0, 6400)
z = (0, 3000)

grid = RectilinearGrid(GPU(); x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state and formulation
#
# We use the anelastic formulation with a dry adiabatic reference state.
# The surface potential temperature ``θ_0 = 299.1`` K and surface pressure
# ``p_0 = 1015`` hPa are taken from [Siebesma2003](@citet); Appendix B.

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants,
                                 base_pressure = 101500,
                                 potential_temperature = 299.1)

formulation = AnelasticFormulation(reference_state,
                                   thermodynamics = :LiquidIcePotentialTemperature)

# ## Surface fluxes
#
# BOMEX prescribes constant surface sensible and latent heat fluxes
# ([Siebesma2003](@citet), Appendix B, after Eq. B4):
# - Sensible heat flux: ``\overline{w'\theta'}|_0 = 8 \times 10^{-3}`` K m/s
# - Moisture flux: ``\overline{w'q_t'}|_0 = 5.2 \times 10^{-5}`` m/s
#
# ([Siebesma2003](@citet) refers to the moisture flux as the "latent heat flux".
# We convert these kinematic fluxes to mass fluxes by multiplying by surface density,
# which we estimate for a dry state using the pressure and temperature at ``z=0``.

w′θ′ = 8e-3     # K m/s (sensible heat flux)
w′qᵗ′ = 5.2e-5  # m/s (moisture flux)

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
# ``u_* = 0.28`` m/s ([Siebesma2003](@citet); Appendix B, after Eq. B4).

u★ = 0.28  # m/s
@inline ρu_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / sqrt(ρu^2 + ρv^2)
@inline ρv_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / sqrt(ρu^2 + ρv^2)

ρu_drag_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρv_drag_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρu_bcs = FieldBoundaryConditions(bottom=ρu_drag_bc)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_drag_bc)

# ## Large-scale subsidence
#
# The BOMEX case includes large-scale subsidence that advects mean profiles downward.
# The subsidence velocity profile is prescribed by [Siebesma2003](@citet); Appendix B, Eq. B5:
# ```math
# w^s(z) = \begin{cases}
#   W^s \frac{z}{z_1} & z \le z_1 \\
#   W^s \left ( 1 - \frac{z - z_1}{z_2 - z_1} \right ) & z_1 < z \le z_2 \\
#   0 & z > z_2
# \end{cases}
# ```
# where ``W^s = -6.5 \times 10^{-3}`` m/s (note the negative sign for "subisdence"),
# ``z_1 = 1500`` m and ``z_2 = 2100`` m.
#
# The subsidence velocity profile is provided by [AtmosphericProfilesLibrary](https://github.com/CliMA/AtmosphericProfilesLibrary.jl),

wˢ = Field{Nothing, Nothing, Face}(grid)
wˢ_profile = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
set!(wˢ, z -> wˢ_profile(z))

# and looks like:

lines(wˢ; axis = (xlabel = "wˢ (m/s)",))

# Subsidence is implemented as an advection of the horizontally-averaged prognostic variables.
# This implementation --- which requires building `Field`s to represent horizontal averages
# and computing it every time step --- is handled by `SubsidenceForcing`.

subsidence = SubsidenceForcing(wˢ)

# ## Geostrophic forcing
#
# The momentum equations include a Coriolis force with prescribed geostrophic wind.
# The geostrophic wind profiles are given by [Siebesma2003](@citet); Appendix B, Eq. B6.
# Using `geostrophic_forcings`, we specify the geostrophic velocity profiles as functions
# of height, and the forcing is automatically materialized with the model's coriolis
# parameter and reference density.

coriolis = FPlane(f=3.76e-5)

uᵍ = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
geostrophic = geostrophic_forcings(z -> uᵍ(z), z -> vᵍ(z))

# ## Moisture tendency (drying)
#
# A prescribed large-scale drying tendency removes moisture above the cloud layer
# ([Siebesma2003](@citet); Appendix B, Eq. B4). This represents the effects of
# advection by the large-scale circulation.

ρᵣ = formulation.reference_state.density
drying = Field{Nothing, Nothing, Center}(grid)
dqdt_profile = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
set!(drying, z -> dqdt_profile(z))
set!(drying, ρᵣ * drying)
ρqᵗ_drying_forcing = Forcing(drying)

# ## Radiative cooling
#
# A prescribed radiative cooling profile is applied to the thermodynamic equation
# ([Siebesma2003](@citet); Appendix B, Eq. B3). Below the inversion, radiative cooling
# of about 2 K/day counteracts the surface heating. We use an energy forcing for radiation
# to ensure that it is applied to the potential temperature conservation equation
# consistently (see below for some elaboration about that).

Fρe_field = Field{Nothing, Nothing, Center}(grid)
cᵖᵈ = constants.dry_air.heat_capacity
dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
set!(Fρe_field, z -> dTdt_bomex(1, z))
set!(Fρe_field, ρᵣ * cᵖᵈ * Fρe_field)
ρe_radiation_forcing = Forcing(Fρe_field)

# ## Assembling all the forcings
#
# We build tuples of forcings for all the variables. Note that forcing functions
# are provided for both `ρθ` and `ρe`, which both contribute to the tendency of `ρθ`
# in different ways. In particular, the tendency for `ρθ` is written
#
# ```math
# ∂_t (ρ θ) = - \boldsymbol{\nabla \cdot} \, ( ρ \boldsymbol{u} θ ) + F_{ρθ} + \frac{1}{cᵖᵐ Π} F_{ρ e} + \cdots
# ```
#
# where ``F_{ρ e}`` denotes the forcing function provided for `ρe` (e.g. for "energy density"),
# ``F_{ρθ}`` denotes the forcing function provided for `ρθ`, and the ``\cdots`` denote
# additional terms.
#
# The geostrophic forcing provides both `ρu` and `ρv` components, which we merge with
# the subsidence forcing.

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
# We use warm-phase saturation adjustment microphysics and 9th-order WENO advection.

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
advection = WENO(order=9)

model = AtmosphereModel(grid; formulation, coriolis, microphysics, advection, forcing,
                        boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

# ## Initial conditions
#
# ### Profiles from AtmosphericProfilesLibrary
#
# Mean profiles are specified as piecewise linear functions by [Siebesma2003](@citet),
# Appendix B, Tables B1 and B2, and include:
#    - Liquid-ice potential temperature ``θ^{\ell i}(z)`` (Table B1)
#    - Total water specific humidity ``q^t(z)`` (Table B1)
#    - Zonal velocity ``u(z)`` (Table B2)
#
# The amazing and convenient [AtmosphericProfilesLibrary](https://github.com/CliMA/AtmosphericProfilesLibrary.jl)
# implements functions that retrieve these profiles.

FT = eltype(grid)
θˡⁱ₀ = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
qᵗ₀ = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.Bomex_u(FT)

# Breeze's current definition of the Exner function derives its reference
# pressure from the base pressure of the reference profile, rather than using
# the standard ``10^5`` Pa. Because of this, we need to apply a correction to
# the initial condition: without this correction, our results do not match
# [Siebesma2003](@citet) (and note that our outputted potential temperature
# is displaced from [Siebesma2003](@citet)'s by precisely the factor ``χ`` below).

using Breeze.Thermodynamics: dry_air_gas_constant, vapor_gas_constant

Rᵈ = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
p₀ = reference_state.base_pressure
χ = (p₀ / 1e5)^(Rᵈ/  cᵖᵈ)

# The initial profiles are perturbed with random noise below 1600 m to trigger
# convection. The perturbation amplitudes are specified by [Siebesma2003](@citet);
# Appendix B (third paragraph after Eq. B6):
#
# - Potential temperature perturbation: ``δθ = 0.1`` K
# - Moisture perturbation: ``δqᵗ = 2.5 \times 10^{-5}`` kg/kg
#
# Magnitudes for the random perturbations applied to the initial profiles are given by
# [Siebesma2003](@citet), Appendix B, third paragraph after Eq. B6.

δθ = 0.1      # K
δqᵗ = 2.5e-5  # kg/kg
zδ = 1600     # m

ϵ() = rand() - 1/2
θᵢ(x, y, z) = χ * θˡⁱ₀(z) + δθ  * ϵ() * (z < zδ)
qᵢ(x, y, z) = qᵗ₀(z)  + δqᵗ * ϵ() * (z < zδ)
uᵢ(x, y, z) = u₀(z)

set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ)

# ## Simulation
#
# We run the simulation for 6 hours with adaptive time-stepping.

simulation = Simulation(model; Δt=10, stop_time=6hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Output and progress
#
# We add a progress callback and output the hourly time-averages of the horizontally-averaged
# profiles for post-processing.

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

filename = "bomex.jld2"
simulation.output_writers[:averages] = JLD2Writer(model, averaged_outputs; filename,
                                                  schedule = AveragedTimeInterval(1hour),
                                                  overwrite_existing = true)

# Output horizontal slices at z = 600 m for animation
# Find the k-index closest to z = 600 m
z = Oceananigans.Grids.znodes(grid, Center())
k = searchsortedfirst(z, 800)
@info "Saving slices at z = $(z[k]) m (k = $k)"

u, v, w = model.velocities
slice_fields = (; w, qˡ)
slice_outputs = (
    wxy = view(w, :, :, k),
    qˡxy = view(qˡ, :, :, k),
    wxz = view(w, :, 1, :),
    qˡxz = view(qˡ, :, 1, :),
)

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = "bomex_slices.jld2",
                                                schedule = TimeInterval(30seconds),
                                                overwrite_existing = true)

@info "Running BOMEX simulation..."
run!(simulation)

# ## Results: mean profile evolution
#
# We visualize the evolution of horizontally-averaged profiles every hour, similar
# to Figure 3 in the paper by [Siebesma2003](@cite). The intercomparison study shows
# that after spin-up, the boundary layer reaches a quasi-steady state with:
# - A well-mixed layer below cloud base (~500 m)
# - A conditionally unstable cloud layer (~500-1500 m)
# - A stable inversion layer (~1500-2000 m)

θt = FieldTimeSeries(filename, "θ")
qᵛt = FieldTimeSeries(filename, "qᵛ")
qˡt = FieldTimeSeries(filename, "qˡ")
ut = FieldTimeSeries(filename, "u")
vt = FieldTimeSeries(filename, "v")

# Create a 2×2 panel plot showing the evolution of key variables

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
    ylims!(ax, 0, 2500)
end

xlims!(axθ, 298, 310)
xlims!(axq, 3e-3, 18e-3)
xlims!(axuv, -10, 2)

# Add legends and annotations
axislegend(axθ, position=:rb)
text!(axuv, -8.5, 2200, text="solid: u\ndashed: v", fontsize=12)

fig[0, :] = Label(fig, "BOMEX: Mean profile evolution (Siebesma et al., 2003)", fontsize=18, tellwidth=false)

save("bomex_profiles.png", fig)
fig

# The simulation shows the development of a cloudy boundary layer with:
# - Warming of the subcloud layer from surface fluxes
# - Moistening of the lower troposphere
# - Development of cloud water in the conditionally unstable layer
# - Westerly flow throughout the domain with weak meridional winds

# ## Animation of horizontal slices
#
# We create an animation showing the evolution of vertical velocity and liquid
# water at z = 800 m, which is near the cloud base level.

wxz_ts = FieldTimeSeries("bomex_slices.jld2", "wxz")
qˡxz_ts = FieldTimeSeries("bomex_slices.jld2", "qˡxz")
wxy_ts = FieldTimeSeries("bomex_slices.jld2", "wxy")
qˡxy_ts = FieldTimeSeries("bomex_slices.jld2", "qˡxy")

times = wxz_ts.times
Nt = length(times)

# Create animation
slices_fig = Figure(size=(1000, 500), fontsize=14)
axwxz = Axis(slices_fig[1, 2], xlabel="x (m)", ylabel="z (m)", title="Vertical velocity w")
axqxz = Axis(slices_fig[1, 3], xlabel="x (m)", ylabel="z (m)", title="Liquid water qˡ")
axwxy = Axis(slices_fig[2, 2], xlabel="x (m)", ylabel="y (m)")
axqxy = Axis(slices_fig[2, 3], xlabel="x (m)", ylabel="y (m)")

# Determine color limits from the data
wmax = maximum(abs, wxz_ts)
qˡmax = maximum(qˡxz_ts)

n = Observable(1)
wxz_n = @lift wxz_ts[$n]
qˡxz_n = @lift qˡxz_ts[$n]
wxy_n = @lift wxy_ts[$n]
qˡxy_n = @lift qˡxy_ts[$n]
title_text = @lift "BOMEX slices at t = " * prettytime(times[$n])

hmw = heatmap!(axwxz, wxz_n, colormap=:balance, colorrange=(-wmax, wmax))
hmq = heatmap!(axqxz, qˡxz_n, colormap=:dense, colorrange=(0, qˡmax))
hmw = heatmap!(axwxy, wxy_n, colormap=:balance, colorrange=(-wmax, wmax))
hmq = heatmap!(axqxy, qˡxy_n, colormap=:dense, colorrange=(0, qˡmax))

Colorbar(slices_fig[1:2, 1], hmw, label="w (m/s)", flipaxis=false)
Colorbar(slices_fig[1:2, 4], hmq, label="qˡ (kg/kg)")

slices_fig[0, :] = Label(slices_fig, title_text, fontsize=18, tellwidth=false)

# Record animation
CairoMakie.record(slices_fig, "bomex_slices.mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end
nothing #hide

# ![](bomex_slices.mp4)
 
