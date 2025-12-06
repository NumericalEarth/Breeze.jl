# # Shallow cumulus convection (BOMEX)
#
# This example simulates shallow cumulus convection following the Barbados Oceanographic
# and Meteorological Experiment (BOMEX) intercomparison case [Siebesma2003](@cite).
# BOMEX has become a canonical test case for large eddy simulations of shallow cumulus
# convection over a subtropical ocean.
#
# The case is based on observations from the Barbados Oceanographic and Meteorological
# Experiment, which documented the structure and organization of trade-wind cumulus
# clouds. The intercomparison study by Siebesma et al. (2003) brought together results
# from 10 different large eddy simulation codes to establish benchmark statistics.
#
# Initial and boundary conditions for this case are provided by
# [AtmosphericProfilesLibrary.jl](https://github.com/CliMA/AtmosphericProfilesLibrary.jl).

using Breeze
using Oceananigans.Units

using AtmosphericProfilesLibrary
using Printf
using CairoMakie

using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ

# ## Domain and grid
#
# The BOMEX domain is 6.4 km × 6.4 km horizontally with a vertical extent of 3 km
# ([Siebesma2003](@cite), Section 3a). The original intercomparison used
# 64 × 64 × 75 grid points with 100 m horizontal resolution and 40 m vertical resolution.
#
# For this documentation example, we use reduced horizontal resolution (32²) to enable
# fast execution on a CPU. The full resolution case should be run for production simulations.

Nx = Ny = 32
Nz = 75

x = y = (0, 6400)
z = (0, 3000)

stop_time = 1hour

grid = RectilinearGrid(CPU(); x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Initial profiles from AtmosphericProfilesLibrary
#
# The initial thermodynamic profiles are piecewise linear functions defined in
# [Siebesma2003](@cite), Appendix B, Tables B1 and B2. These include:
# - Liquid-ice potential temperature ``θ_{\ell i}(z)`` (Table B1)
# - Total water specific humidity ``q_t(z)`` (Table B1)
# - Zonal velocity ``u(z)`` (Table B2)
#
# The [AtmosphericProfilesLibrary](https://github.com/CliMA/AtmosphericProfilesLibrary.jl)
# provides convenient functions to retrieve these profiles.

FT = eltype(grid)
θ_bomex = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
q_bomex = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
u_bomex = AtmosphericProfilesLibrary.Bomex_u(FT)

# ## Reference state and formulation
#
# We use the anelastic formulation with a dry adiabatic reference state.
# The surface potential temperature ``θ_0 = 299.1`` K and surface pressure
# ``p_0 = 1015`` hPa are taken from [Siebesma2003](@cite), Appendix B.

p₀, θ₀ = 101500, 299.1
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state, thermodynamics=:LiquidIcePotentialTemperature)

# ## Surface fluxes
#
# BOMEX prescribes constant surface sensible and latent heat fluxes
# ([Siebesma2003](@cite), Appendix B, after Eq. B4):
# - Sensible heat flux: ``\overline{w'\theta_v'}|_s = 8 \times 10^{-3}`` K m/s
# - Latent heat flux: ``\overline{w'q_t'}|_s = 5.2 \times 10^{-5}`` m/s
#
# We convert these kinematic fluxes to mass fluxes by multiplying by surface density.

q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
ρ₀ = Breeze.Thermodynamics.density(p₀, θ₀, q₀, constants)

w′θ′ = 8e-3    # K m/s (sensible heat flux)
w′q′ = 5.2e-5  # m/s (latent heat flux)

ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′θ′))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′q′))

# ## Surface momentum flux (drag)
#
# A bulk drag parameterization is applied with friction velocity
# ``u_* = 0.28`` m/s ([Siebesma2003](@cite), Appendix B, after Eq. B4).

u★ = 0.28 # m/s
@inline ρu_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / sqrt(ρu^2 + ρv^2)
@inline ρv_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / sqrt(ρu^2 + ρv^2)

ρu_drag_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρv_drag_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρu_bcs = FieldBoundaryConditions(bottom=ρu_drag_bc)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_drag_bc)

# ## Large-scale subsidence
#
# The BOMEX case includes large-scale subsidence that advects mean profiles downward.
# The subsidence velocity profile is prescribed in [Siebesma2003](@cite), Appendix B, Eq. B5:
# ```math
# w_s(z) = \begin{cases}
#   -0.65 \times 10^{-2} z / z_1 & z \le z_1 \\
#   -0.65 \times 10^{-2} (1 - (z - z_1)/(z_2 - z_1)) & z_1 < z \le z_2 \\
#   0 & z > z_2
# \end{cases}
# ```
# where ``z_1 = 1500`` m and ``z_2 = 2100`` m.
#
# We apply subsidence as a forcing term to the horizontally-averaged prognostic variables.
# This requires computing horizontal averages at each time step and storing them in
# fields that can be accessed by the forcing functions.

@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

@inline function Fρu_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_U = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.u_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_U
end

@inline function Fρv_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_V = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.v_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_V
end

@inline function Fρθ_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_Θ = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.θ_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_Θ
end

@inline function Fρqᵗ_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_Qᵗ = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.qᵗ_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_Qᵗ
end

# Set up horizontally-averaged fields for subsidence (suffix `_f` for "forcing")

u_avg_f = Field{Nothing, Nothing, Center}(grid)
v_avg_f = Field{Nothing, Nothing, Center}(grid)
θ_avg_f = Field{Nothing, Nothing, Center}(grid)
qᵗ_avg_f = Field{Nothing, Nothing, Center}(grid)

# Subsidence velocity profile from AtmosphericProfilesLibrary

wˢ = Field{Nothing, Nothing, Face}(grid)
w_bomex = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
set!(wˢ, z -> w_bomex(z))

ρᵣ = formulation.reference_state.density
ρu_subsidence_forcing = Forcing(Fρu_subsidence, discrete_form=true, parameters=(; u_avg=u_avg_f, wˢ, ρᵣ))
ρv_subsidence_forcing = Forcing(Fρv_subsidence, discrete_form=true, parameters=(; v_avg=v_avg_f, wˢ, ρᵣ))
ρθ_subsidence_forcing = Forcing(Fρθ_subsidence, discrete_form=true, parameters=(; θ_avg=θ_avg_f, wˢ, ρᵣ))
ρqᵗ_subsidence_forcing = Forcing(Fρqᵗ_subsidence, discrete_form=true, parameters=(; qᵗ_avg=qᵗ_avg_f, wˢ, ρᵣ))

# ## Geostrophic forcing
#
# The momentum equations include a Coriolis force with prescribed geostrophic wind.
# The geostrophic wind profiles are given in [Siebesma2003](@cite), Appendix B, Eq. B6.

coriolis = FPlane(f=3.76e-5)

uᵍ = Field{Nothing, Nothing, Center}(grid)
vᵍ = Field{Nothing, Nothing, Center}(grid)
uᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
set!(uᵍ, z -> uᵍ_bomex(z))
set!(vᵍ, z -> vᵍ_bomex(z))
ρuᵍ = Field(ρᵣ * uᵍ)
ρvᵍ = Field(ρᵣ * vᵍ)

@inline Fρu_geostrophic(i, j, k, grid, clock, fields, p) = @inbounds - p.f * p.ρvᵍ[i, j, k]
@inline Fρv_geostrophic(i, j, k, grid, clock, fields, p) = @inbounds + p.f * p.ρuᵍ[i, j, k]

ρu_geostrophic_forcing = Forcing(Fρu_geostrophic, discrete_form=true, parameters=(; f=coriolis.f, ρvᵍ))
ρv_geostrophic_forcing = Forcing(Fρv_geostrophic, discrete_form=true, parameters=(; f=coriolis.f, ρuᵍ))

ρu_forcing = (ρu_subsidence_forcing, ρu_geostrophic_forcing)
ρv_forcing = (ρv_subsidence_forcing, ρv_geostrophic_forcing)

# ## Moisture tendency (drying)
#
# A prescribed large-scale drying tendency removes moisture above the cloud layer
# ([Siebesma2003](@cite), Appendix B, Eq. B4). This represents the effects of
# advection by the large-scale circulation.

drying = Field{Nothing, Nothing, Center}(grid)
dqdt_bomex = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
set!(drying, z -> dqdt_bomex(z))
set!(drying, ρᵣ * drying)
ρqᵗ_drying_forcing = Forcing(drying)

ρqᵗ_forcing = (ρqᵗ_drying_forcing, ρqᵗ_subsidence_forcing)

# ## Radiative cooling
#
# A prescribed radiative cooling profile is applied to the thermodynamic equation
# ([Siebesma2003](@cite), Appendix B, Eq. B3). Below the inversion, radiative cooling
# of about 2 K/day counteracts the surface heating.

Fρθ_field = Field{Nothing, Nothing, Center}(grid)
dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
set!(Fρθ_field, z -> dTdt_bomex(1, z))
set!(Fρθ_field, ρᵣ * Fρθ_field)

ρθ_radiation_forcing = Forcing(Fρθ_field)
ρθ_forcing = (ρθ_radiation_forcing, ρθ_subsidence_forcing)

# ## Model setup
#
# We use warm-phase saturation adjustment microphysics and WENO advection.

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
advection = WENO(order=9)

model = AtmosphereModel(grid; formulation, coriolis, microphysics, advection,
                        forcing = (ρqᵗ=ρqᵗ_forcing, ρu=ρu_forcing, ρv=ρv_forcing, ρθ=ρθ_forcing),
                        boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

# ## Initial conditions
#
# The initial profiles are perturbed with random noise below 1600 m to trigger
# convection. The perturbation amplitudes are specified in [Siebesma2003](@cite),
# Appendix B (third paragraph after Eq. B6):
# - Potential temperature perturbation: ``\delta\theta = 0.1`` K
# - Moisture perturbation: ``\delta q_t = 2.5 \times 10^{-5}`` kg/kg

θϵ = 0.1     # K
qϵ = 2.5e-5  # kg/kg
zϵ = 1600    # m

θᵢ(x, y, z) = θ_bomex(z) + θϵ * rand() * (z < zϵ)
qᵢ(x, y, z) = q_bomex(z) + qϵ * rand() * (z < zϵ)
uᵢ(x, y, z) = u_bomex(z)

set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ)

# ## Simulation
#
# We run the simulation for 1 hour with adaptive time-stepping.

simulation = Simulation(model; Δt=10, stop_time)
conjure_time_step_wizard!(simulation, cfl=0.7)

# Set up horizontal average diagnostics for subsidence forcing.
# These must be computed at each time step via a callback.

θ = liquid_ice_potential_temperature(model)
u_avg = Field(Average(model.velocities.u, dims=(1, 2)))
v_avg = Field(Average(model.velocities.v, dims=(1, 2)))
θ_avg = Field(Average(θ, dims=(1, 2)))
qᵗ_avg = Field(Average(model.specific_moisture, dims=(1, 2)))

function compute_averages!(sim)
    compute!(u_avg)
    compute!(v_avg)
    compute!(θ_avg)
    compute!(qᵗ_avg)
    parent(u_avg_f) .= parent(u_avg)
    parent(v_avg_f) .= parent(v_avg)
    parent(θ_avg_f) .= parent(θ_avg)
    parent(qᵗ_avg_f) .= parent(qᵗ_avg)
    return nothing
end

add_callback!(simulation, compute_averages!)

# ## Output and progress
#
# We output horizontally-averaged profiles for post-processing.

qˡ = model.microphysical_fields.qˡ
qᵛ = model.microphysical_fields.qᵛ

function progress(sim)
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

add_callback!(simulation, progress, IterationInterval(100))

outputs = merge(model.velocities, model.tracers, (; θ, qˡ, qᵛ))
averaged_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename = "bomex.jld2"
simulation.output_writers[:averages] = JLD2Writer(model, averaged_outputs; filename,
                                                  schedule = TimeInterval(20minutes),
                                                  overwrite_existing = true)

@info "Running BOMEX simulation..."
run!(simulation)

# ## Results: mean profile evolution
#
# We visualize the evolution of horizontally-averaged profiles every 20 minutes,
# similar to Figure 3 in [Siebesma2003](@cite). The intercomparison study shows
# that after spin-up, the boundary layer reaches a quasi-steady state with:
# - A well-mixed layer below cloud base (~500 m)
# - A conditionally unstable cloud layer (~500-1500 m)
# - A stable inversion layer (~1500-2000 m)

θt = FieldTimeSeries(filename, "θ")
qᵛt = FieldTimeSeries(filename, "qᵛ")
qˡt = FieldTimeSeries(filename, "qˡ")
ut = FieldTimeSeries(filename, "u")
vt = FieldTimeSeries(filename, "v")

times = θt.times
Nt = length(times)
z = znodes(θt)

# Create a 2×2 panel plot showing the evolution of key variables

fig = Figure(size=(900, 800), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="qᵛ (g/kg)", ylabel="z (m)")
axuv = Axis(fig[2, 1], xlabel="u, v (m/s)", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="qˡ (g/kg)", ylabel="z (m)")

# Plot profiles at each output time (every 20 minutes)
colors = cgrad(:viridis, Nt, categorical=true)

for n in 1:Nt
    t_min = Int(times[n] / 60)
    label = "t = $(t_min) min"

    θn = interior(θt[n], 1, 1, :)
    qᵛn = interior(qᵛt[n], 1, 1, :) .* 1000  # Convert to g/kg
    qˡn = interior(qˡt[n], 1, 1, :) .* 1000  # Convert to g/kg
    un = interior(ut[n], 1, 1, :)
    vn = interior(vt[n], 1, 1, :)

    lines!(axθ, θn, z, color=colors[n], label=label)
    lines!(axq, qᵛn, z, color=colors[n])
    lines!(axuv, un, z, color=colors[n], linestyle=:solid)
    lines!(axuv, vn, z, color=colors[n], linestyle=:dash)
    lines!(axqˡ, qˡn, z, color=colors[n])
end

# Set axis limits to focus on the boundary layer
ylims!(axθ, 0, 2500)
ylims!(axq, 0, 2500)
ylims!(axuv, 0, 2500)
ylims!(axqˡ, 0, 2500)

xlims!(axθ, 298, 312)
xlims!(axq, 4, 18)
xlims!(axuv, -10, 2)

# Add legends and annotations
axislegend(axθ, position=:rt)
text!(axuv, -8.5, 2200, text="solid: u\ndashed: v", fontsize=12)

fig[0, :] = Label(fig, "BOMEX: Mean profile evolution (Siebesma et al., 2003)", fontsize=18, tellwidth=false)

save("bomex_profiles.png", fig)
fig

# The simulation shows the development of a cloudy boundary layer with:
# - Warming of the subcloud layer from surface fluxes
# - Moistening of the lower troposphere
# - Development of cloud water in the conditionally unstable layer
# - Westerly flow throughout the domain with weak meridional winds
#
# Note: This short 1-hour simulation captures the initial spin-up phase.
# For production results comparable to [Siebesma2003](@cite), the simulation
# should be run for 6 hours at full resolution (64² × 75) on a GPU.
