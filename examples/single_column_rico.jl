# # Single column RICO simulation
#
# This example simulates a single-column model (SCM) version of the RICO
# (Rain In Cumulus over the Ocean) intercomparison case [vanZanten2011](@cite).
# This is a direct single-column adaptation of the 3D RICO example.
#
# Single column models are valuable testbeds for turbulence closure development.
# Here we use a simple constant vertical diffusivity for mixing.

using Breeze
using Oceananigans
using Oceananigans.Units
using AtmosphericProfilesLibrary
using CairoMakie
using Printf

# ## Grid setup
#
# Single column grid with same vertical extent and resolution as 3D RICO.

Nz = 100
z = (0, 4000)

grid = RectilinearGrid(size = Nz, halo = 5,
                       z = z,
                       topology = (Flat, Flat, Bounded))

# ## Reference state and formulation
#
# Identical to 3D RICO: surface pressure 1015.4 hPa, surface potential temperature 297.9 K.

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = 101540,
                                 potential_temperature = 297.9)

formulation = AnelasticFormulation(reference_state,
                                   thermodynamics = :LiquidIcePotentialTemperature)

# ## Surface fluxes
#
# Bulk aerodynamic formulas with constant transfer coefficients.

Cᴰ = 1.229e-3 # Drag coefficient for momentum
Cᵀ = 1.094e-3 # Sensible heat transfer coefficient
Cᵛ = 1.133e-3 # Moisture flux transfer coefficient
T₀ = 299.8    # Sea surface temperature (K)

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=T₀)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=T₀)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)

ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

# ## Large-scale subsidence

FT = eltype(grid)
wˢ_profile = AtmosphericProfilesLibrary.Rico_subsidence(FT)
wˢ = Field{Nothing, Nothing, Face}(grid)
set!(wˢ, z -> wˢ_profile(z))
subsidence = SubsidenceForcing(wˢ)

# ## Geostrophic forcing
#
# Coriolis with prescribed geostrophic wind (same as 3D RICO).

coriolis = FPlane(f=4.5e-5)

uᵍ = AtmosphericProfilesLibrary.Rico_geostrophic_ug(FT)
vᵍ = AtmosphericProfilesLibrary.Rico_geostrophic_vg(FT)
geostrophic = geostrophic_forcings(z -> uᵍ(z), z -> vᵍ(z))

# ## Moisture tendency

ρᵣ = formulation.reference_state.density
drying = Field{Nothing, Nothing, Center}(grid)
dqdt_profile = AtmosphericProfilesLibrary.Rico_dqtdt(FT)
set!(drying, z -> dqdt_profile(z))
set!(drying, ρᵣ * drying)
ρqᵗ_drying_forcing = Forcing(drying)

# ## Radiative cooling
#
# Applied to ρθ (potential temperature density) directly.
# Note: The RICO dTdt is already in K/s, and for LiquidIcePotentialTemperatureThermodynamics,
# we force ρθ, so we just multiply by ρᵣ.

cooling = Field{Nothing, Nothing, Center}(grid)
dTdt_rico = AtmosphericProfilesLibrary.Rico_dTdt(FT)
set!(cooling, z -> dTdt_rico(1, z))
set!(cooling, ρᵣ * cooling)  # ρθ forcing = ρ * dθ/dt
ρθ_radiation_forcing = Forcing(cooling)

# ## Assembling forcing and boundary conditions

Fρu = (subsidence, geostrophic.ρu)
Fρv = (subsidence, geostrophic.ρv)
Fρqᵗ = (subsidence, ρqᵗ_drying_forcing)
Fρθ = (subsidence, ρθ_radiation_forcing)

forcing = (ρu=Fρu, ρv=Fρv, ρqᵗ=Fρqᵗ, ρθ=Fρθ)
boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)

# ## Turbulence closure
#
# For SCM we need a turbulence closure (3D RICO uses resolved turbulence).
# We use a simple constant vertical diffusivity for now.
# TODO: implement proper atmospheric TKE or Ri-based closure

using Oceananigans.TurbulenceClosures: VerticalScalarDiffusivity

closure = VerticalScalarDiffusivity(ν=1.0, κ=1.0)  # Moderate diffusivity

# ## Model setup

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())

model = AtmosphereModel(grid; formulation, coriolis, microphysics, closure,
                        timestepper = :QuasiAdamsBashforth2,
                        forcing, boundary_conditions)

# ## Initial conditions
#
# RICO profiles from AtmosphericProfilesLibrary.

θˡⁱ₀ = AtmosphericProfilesLibrary.Rico_θ_liq_ice(FT)
qᵗ₀ = AtmosphericProfilesLibrary.Rico_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.Rico_u(FT)
v₀ = AtmosphericProfilesLibrary.Rico_v(FT)

set!(model, θ = z -> θˡⁱ₀(z),
            qᵗ = z -> qᵗ₀(z),
            u = z -> u₀(z),
            v = z -> v₀(z))

# ## Simulation

simulation = Simulation(model; Δt=5, stop_time=12hours)
conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=30)

# ## Output and progress

θ = liquid_ice_potential_temperature(model)
qˡ = model.microphysical_fields.qˡ
qᵛ = model.microphysical_fields.qᵛ

wall_clock = Ref(time_ns())

function progress(sim)
    qᵛmax = maximum(qᵛ)
    qˡmax = maximum(qˡ)
    qᵗmax = maximum(sim.model.specific_moisture)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, wall time: %s\n",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed))

    msg *= @sprintf(" --- max(qᵗ): %.2e, max(qˡ): %.2e",
                    qᵗmax, qˡmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# ## Output
#
# We save the state variables.

u, v, w = model.velocities

outputs = (; u, v, θ, qˡ, qᵛ)

filename = "single_column_rico.jld2"
simulation.output_writers[:fields] = JLD2Writer(model, outputs; filename,
                                                schedule = TimeInterval(2hours),
                                                overwrite_existing = true)

# ## Run

@info "Running single column RICO simulation..."
run!(simulation)

# ## Results: profile evolution

θt = FieldTimeSeries(filename, "θ")
qᵛt = FieldTimeSeries(filename, "qᵛ")
qˡt = FieldTimeSeries(filename, "qˡ")
ut = FieldTimeSeries(filename, "u")
vt = FieldTimeSeries(filename, "v")

fig = Figure(size=(1000, 600), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="qᵛ (kg/kg)", ylabel="z (m)")
axuv = Axis(fig[2, 1], xlabel="u, v (m/s)", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="qˡ (kg/kg)", ylabel="z (m)")

times = θt.times
Nt = length(times)

default_colours = Makie.wong_colors()
colors = [default_colours[mod1(i, length(default_colours))] for i in 1:Nt]

for n in 1:Nt
    label = n == 1 ? "initial condition" : "t = $(Int(times[n]/hour)) hr"

    lines!(axθ, θt[n], color=colors[n], label=label)
    lines!(axq, qᵛt[n], color=colors[n])
    lines!(axuv, ut[n], color=colors[n], linestyle=:solid)
    lines!(axuv, vt[n], color=colors[n], linestyle=:dash)
    lines!(axqˡ, qˡt[n], color=colors[n])
end

for ax in (axθ, axq, axuv, axqˡ)
    ylims!(ax, 0, 3500)
end

xlims!(axθ, 296, 320)
xlims!(axq, 0, 18e-3)
xlims!(axuv, -12, 2)

axislegend(axθ, position=:rb)
text!(axuv, -10, 3200, text="solid: u\ndashed: v", fontsize=12)

fig[0, :] = Label(fig, "Single Column RICO Simulation", fontsize=18, tellwidth=false)

save("single_column_rico.png", fig)

@info "Saved figure to single_column_rico.png"

fig
