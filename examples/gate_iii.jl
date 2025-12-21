# # Tropical deep convection (GATE III)
#
# This example simulates tropical deep convection following the GATE Phase III case
# from [Khairoutdinov2009](@cite). GATE — the Global Atmospheric Research Program
# Atlantic Tropical Experiment — was nothing short of epic: the largest atmospheric
# field campaign ever conducted. In the summer of 1974, an international armada of
# **39 ships** and **13 aircraft** from 72 countries descended on the tropical Atlantic,
# deploying 5,000 scientists, technicians, and support staff to study tropical convection
# and its role in global weather and climate [Zipser2024](@cite).
#
# Phase III of GATE (30 August - 18 September 1974) featured especially vigorous
# convective activity and has become a benchmark for developing and validating
# cloud-resolving models. The case is characterized by deep cumulus convection
# reaching into the upper troposphere, making it ideal for testing mixed-phase
# microphysics that handles both liquid water and ice.
#
# Initial conditions and large-scale forcings for this case are provided by
# [AtmosphericProfilesLibrary.jl](https://github.com/CliMA/AtmosphericProfilesLibrary.jl).
# For precipitation we use the 0-moment scheme from
# [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) with mixed-phase
# saturation adjustment, which partitions condensate between supercooled liquid and ice
# based on temperature.

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units

using AtmosphericProfilesLibrary
using CairoMakie
using CloudMicrophysics
using Printf
using Random
using CUDA

Random.seed!(123)

# ## Domain and grid
#
# Unlike the shallow cumulus cases BOMEX and RICO, GATE III features deep convection
# extending into the upper troposphere. We use a domain extending to 18 km altitude
# to capture the full depth of convective clouds.
#
# For initial testing, we use a modest resolution. Production runs would use finer
# grids (e.g., 100 m horizontal, 100 m vertical) and larger horizontal domains.

Oceananigans.defaults.FloatType = Float32

Nx = Ny = 2048
Nz = 256
Δx = 100

x = y = (0, Nx * Δx)
z = (0, 27000)       # 18 km vertical extent

grid = RectilinearGrid(GPU(); x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state and formulation
#
# We use the anelastic formulation with a dry adiabatic reference state.
# Surface conditions are estimated from the GATE III profiles.

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = 101500,
                                 potential_temperature = 299.2)

formulation = AnelasticFormulation(reference_state,
                                   thermodynamics = :LiquidIcePotentialTemperature)

# ## Surface fluxes
#
# The GATE III case specifies surface fluxes through prescribed temperature
# and moisture tendencies rather than explicit surface flux boundary conditions.
# For simplicity, we use a bulk formula approach similar to other tropical cases.

FT = eltype(grid)
Cᴰ = 1.2e-3  # Drag coefficient for momentum
Cᵀ = 1.1e-3  # Sensible heat transfer coefficient
Cᵛ = 1.2e-3  # Moisture flux transfer coefficient
T₀ = 300.5   # Sea surface temperature (K)

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=T₀)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=T₀)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)

ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

# ## Large-scale forcing
#
# The GATE III case includes prescribed temperature and moisture tendencies
# that represent radiative cooling and large-scale advective processes.
# These profiles are provided by AtmosphericProfilesLibrary.

ρᵣ = formulation.reference_state.density
cᵖᵈ = constants.dry_air.heat_capacity

# Temperature tendency (radiative + advective)
dTdt_field = Field{Nothing, Nothing, Center}(grid)
dTdt_profile = AtmosphericProfilesLibrary.GATE_III_dTdt(FT)
set!(dTdt_field, z -> dTdt_profile(z))
set!(dTdt_field, ρᵣ * cᵖᵈ * dTdt_field)
ρe_forcing = Forcing(dTdt_field)

# Moisture tendency (large-scale advection)
dqdt_field = Field{Nothing, Nothing, Center}(grid)
dqdt_profile = AtmosphericProfilesLibrary.GATE_III_dqtdt(FT)
set!(dqdt_field, z -> dqdt_profile(z))
set!(dqdt_field, ρᵣ * dqdt_field)
ρqᵗ_forcing = Forcing(dqdt_field)

# ## Coriolis force
#
# GATE III took place in the tropical Atlantic near 8.5°N, where the Coriolis
# parameter is relatively small but non-zero.

coriolis = FPlane(latitude=8.5)

# ## Assembling forcing and boundary conditions

forcing = (ρe=ρe_forcing, ρqᵗ=ρqᵗ_forcing)
boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)
nothing #hide

# ## Model setup
#
# We use zero-moment bulk microphysics with mixed-phase saturation adjustment.
# The mixed-phase equilibrium partitions condensate between supercooled liquid
# and ice as a function of temperature, transitioning from all liquid at 273.15 K
# (freezing point) to all ice at 233.15 K (homogeneous ice nucleation temperature).
#
# This is crucial for deep convection where clouds extend well above the freezing level.

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: ZeroMomentCloudMicrophysics

nucleation = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())
microphysics = ZeroMomentCloudMicrophysics(τ_precip=3minutes, qc_0=1e-4; nucleation)
advection = WENO(order=5)

model = AtmosphereModel(grid; formulation, coriolis, microphysics,
                        advection, forcing, boundary_conditions)

# ## Initial conditions
#
# Initial profiles of temperature, moisture, and wind are specified by
# [Khairoutdinov2009](@citet). The temperature profile can be set directly
# using `set!(model, T=...)`, which internally converts temperature to
# liquid-ice potential temperature using the proper thermodynamic relations.

T₀ = AtmosphericProfilesLibrary.GATE_III_T(FT)
qᵗ₀ = AtmosphericProfilesLibrary.GATE_III_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.GATE_III_u(FT)

# Deep convection needs a strong trigger to get started. We add a warm, moist
# thermal bubble near the surface to initiate convection. The bubble is placed
# at the domain center with a 2 km radius.

Lx, Ly = 25600, 25600
xc, yc, zc = Lx/2, Ly/2, 2000  # bubble center
Rc = 2000  # bubble radius (m)
ΔT = 3     # temperature perturbation (K)
Δq = 4e-3  # moisture perturbation (kg/kg) - enough to saturate the bubble

function Tᵢ(x, y, z)
    r = sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2)
    T_bubble = ΔT * max(0, 1 - r/Rc)
    T_noise = 0.1 * (rand() - 0.5) * (z < 3000)
    return T₀(z) + T_bubble + T_noise
end

function qᵢ(x, y, z)
    r = sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2)
    q_bubble = Δq * max(0, 1 - r/Rc)
    return qᵗ₀(z) + q_bubble
end

uᵢ(x, y, z) = u₀(z)

set!(model, T=Tᵢ, qᵗ=qᵢ, u=uᵢ, v=0)

# ## Simulation
#
# Deep convection develops over several hours. For initial testing, we run
# for 3 hours with adaptive time-stepping. Production runs would extend
# to 24-48 hours to reach statistical equilibrium.

simulation = Simulation(model; Δt=1, stop_time=3hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Output and progress
#
# We set up diagnostics to monitor the development of deep convection.

θ = liquid_ice_potential_temperature(model)
qˡ = model.microphysical_fields.qˡ
qᵛ = model.microphysical_fields.qᵛ
qⁱ = model.microphysical_fields.qⁱ

P = precipitation_rate(model, :liquid)
∫PdV = Field(Integral(P))

wall_clock = Ref(time_ns())
previous_time = Ref(time(simulation))

function progress(sim)
    compute!(∫PdV)
    qᵛmax = maximum(qᵛ)
    qˡmax = maximum(qˡ)
    qⁱmax = maximum(qⁱ)
    wmax = maximum(abs, model.velocities.w)
    ∫P = CUDA.@allowscalar ∫PdV[]
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    SDPD = (time(simulation) - previous_time[]) / elapsed
    previous_time[] = time(simulation)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, wall time: %s, SDPD: %.2f, max|w|: %.2e m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), SDPD, wmax)

    msg *= @sprintf(", max(qᵛ): %.2e, max(qˡ): %.2e, max(qⁱ): %.2e",
                    qᵛmax, qˡmax, qⁱmax)

    @info msg
    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(500))

# Output horizontally-averaged profiles and slices for visualization

outputs = merge(model.velocities, (; θ, qˡ, qᵛ, qⁱ))
averaged_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename = "gate_iii.jld2"
simulation.output_writers[:averages] = JLD2Writer(model, averaged_outputs; filename,
                                                  schedule = AveragedTimeInterval(30minutes),
                                                  overwrite_existing = true)

# xz-slices for animation
slice_outputs = (
    θxz = view(θ, :, 1, :),
    qˡxz = view(qˡ, :, 1, :),
    qⁱxz = view(qⁱ, :, 1, :),
    wxz = view(model.velocities.w, :, 1, :),
)

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = "gate_iii_slices.jld2",
                                                schedule = TimeInterval(1minute),
                                                overwrite_existing = true)

# Run the simulation!
run!(simulation)

# ## Results: mean profile evolution
#
# We visualize the evolution of horizontally-averaged profiles.

θt = FieldTimeSeries(filename, "θ")
qᵛt = FieldTimeSeries(filename, "qᵛ")
qˡt = FieldTimeSeries(filename, "qˡ")
qⁱt = FieldTimeSeries(filename, "qⁱ")
wt = FieldTimeSeries(filename, "w")

fig = Figure(size=(1000, 800), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="qᵛ (kg/kg)", ylabel="z (m)")
axqc = Axis(fig[2, 1], xlabel="Condensate (kg/kg)", ylabel="z (m)")
axw = Axis(fig[2, 2], xlabel="w (m/s)", ylabel="z (m)")

times = θt.times
Nt = length(times)

default_colours = Makie.wong_colors()
colors = [default_colours[mod1(i, length(default_colours))] for i in 1:Nt]

for n in 1:Nt
    label = n == 1 ? "initial" : "$(Int(times[n]/minute)) min avg"
    
    lines!(axθ, θt[n], color=colors[n], label=label)
    lines!(axq, qᵛt[n], color=colors[n])
    lines!(axqc, qˡt[n], color=colors[n], linestyle=:solid)
    lines!(axqc, qⁱt[n], color=colors[n], linestyle=:dash)
    lines!(axw, wt[n], color=colors[n])
end

for ax in (axθ, axq, axqc, axw)
    ylims!(ax, 0, 16000)
end

axislegend(axθ, position=:rb)
text!(axqc, 0.0001, 14000, text="solid: liquid\ndashed: ice", fontsize=12)

fig[0, :] = Label(fig, "GATE III: Deep tropical convection", fontsize=18, tellwidth=false)

save("gate_iii_profiles.png", fig)
fig

# ## Animation: cloud structure
#
# We create an animation showing the evolution of potential temperature,
# cloud liquid water, and cloud ice in an xz-slice.

θxz_ts = FieldTimeSeries("gate_iii_slices.jld2", "θxz")
qˡxz_ts = FieldTimeSeries("gate_iii_slices.jld2", "qˡxz")
qⁱxz_ts = FieldTimeSeries("gate_iii_slices.jld2", "qⁱxz")
wxz_ts = FieldTimeSeries("gate_iii_slices.jld2", "wxz")

times = θxz_ts.times
Nt = length(times)

# Compute color limits
θmin, θmax = extrema(θxz_ts)
qˡmax = max(maximum(qˡxz_ts), 1e-6)
qⁱmax = max(maximum(qⁱxz_ts), 1e-6)
wmax = max(maximum(abs, wxz_ts), 0.1)

# The domain is 25.6 km wide × 18 km tall
domain_aspect = 25600 / 18000

fig = Figure(size=(1200, 900), fontsize=14)

axθ = Axis(fig[2, 1], aspect=domain_aspect, xlabel="x (m)", ylabel="z (m)", title="Potential temperature θ")
axw = Axis(fig[2, 3], aspect=domain_aspect, xlabel="x (m)", ylabel="z (m)", title="Vertical velocity w")
axqˡ = Axis(fig[3, 1], aspect=domain_aspect, xlabel="x (m)", ylabel="z (m)", title="Cloud liquid qˡ")
axqⁱ = Axis(fig[3, 3], aspect=domain_aspect, xlabel="x (m)", ylabel="z (m)", title="Cloud ice qⁱ")

n = Observable(1)
θxz_n = @lift θxz_ts[$n]
qˡxz_n = @lift qˡxz_ts[$n]
qⁱxz_n = @lift qⁱxz_ts[$n]
wxz_n = @lift wxz_ts[$n]
title = @lift "GATE III deep convection at t = " * prettytime(times[$n])

hmθ = heatmap!(axθ, θxz_n, colormap=:thermal, colorrange=(θmin, θmax))
hmw = heatmap!(axw, wxz_n, colormap=:balance, colorrange=(-wmax, wmax))
hmqˡ = heatmap!(axqˡ, qˡxz_n, colormap=:dense, colorrange=(0, qˡmax))
hmqⁱ = heatmap!(axqⁱ, qⁱxz_n, colormap=:ice, colorrange=(0, qⁱmax))

# Each colorbar next to its respective plot
Colorbar(fig[2, 2], hmθ, label="θ (K)")
Colorbar(fig[2, 4], hmw, label="w (m/s)")
Colorbar(fig[3, 2], hmqˡ, label="qˡ (kg/kg)")
Colorbar(fig[3, 4], hmqⁱ, label="qⁱ (kg/kg)")

fig[1, :] = Label(fig, title, fontsize=18, tellwidth=false)

CairoMakie.record(fig, "gate_iii.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](gate_iii.mp4)

