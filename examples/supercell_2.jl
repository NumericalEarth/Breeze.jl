using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie

using Oceananigans.Grids: znode
using Oceananigans: Center, Face
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, ℑzᵃᵃᶜ
using Breeze.Thermodynamics: dry_air_gas_constant
using CUDA

# Supercell simulation
# Reference:
# Klemp et al. (2015): "Idealized global nonhydrostatic atmospheric test cases on a reduced-radius sphere"

#  grid configuration
Nx, Ny, Nz = 336, 336, 40
Lx, Ly, Lz = 168kilometers, 168kilometers, 20kilometers

grid = RectilinearGrid(GPU(),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))


# Problem parameters and initial conditions
pᵣ, θᵣ = 100000, 300
constants  = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants, base_pressure=pᵣ, potential_temperature=θᵣ)
formulation = AnelasticFormulation(reference_state, thermodynamics=:LiquidIcePotentialTemperature)

θₜᵣ = 343           # tropopause potential temperature [K]
zₜᵣ = 12000         # tropopause height [m]
Tₜᵣ = 213           # tropopause temperature [K]
zₛ = 5kilometers    # shear layer height [m]
uₛ = 30             # shear parameter 1 [m/s]
u_c = 15            # shear parameter 2 [m/s]
g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity
Rᵈ = dry_air_gas_constant(constants)
θᵢ₀(x, y, z) = (θᵣ + (θₜᵣ-θᵣ) * (z / zₜᵣ)^(5/4)) * (z <= zₜᵣ) + θₜᵣ * exp(g/(cᵖᵈ*Tₜᵣ) * (z - zₜᵣ)) * (z > zₜᵣ)
RHᵢ(z) = (1 - 3/4 * (z / zₜᵣ)^(5/4)) * (z <= zₜᵣ) + 1/4 * (z > zₜᵣ)
uᵢ(x, y, z) = (uₛ*(z/zₛ)- u_c) * (z < (zₛ-1000))  + ((-4/5 + 3 *(z/zₛ) - 5/4 *(z/zₛ)^2) * uₛ - u_c) * (abs(z - zₛ) < 1000) + (uₛ - u_c) * (z > (zₛ+1000))

# Warm bubble potential temperature perturbation (Eq. 17–18)
Δθ = 3              # perturbation amplitude [K]
r_h = 10kilometers  # bubble horizontal radius [m]
r_z = 1500          # bubble vertical radius [m]
z_c = 1500          # bubble center height [m]
x_c = Lx / 2        # bubble center x-coordinate [m]
y_c = Ly / 2        # bubble center y-coordinate [m]

function θᵢ(x, y, z)
    θ_base = θᵢ₀(x, y, z)
    r = sqrt((x - x_c)^2 + (y - y_c)^2)
    Rθ = sqrt((r / r_h)^2 + ((z - z_c) / r_z)^2)
    θ_pert = ifelse(Rθ < 1, Δθ * cos((π / 2) * Rθ)^2, 0.0)
    return θ_base + θ_pert
end

# Atmosphere model setup
microphysics = KesslerMicrophysics()
advection = WENO(order=5, minimum_buffer_upwind_order=3)
model = AtmosphereModel(grid; formulation, microphysics, advection)
set!(model, θ = θᵢ₀)
θ₀ = Field{Center, Center, Center}(grid)
set!(θ₀, θᵢ₀)

ph = Breeze.AtmosphereModels.compute_hydrostatic_pressure!(CenterField(grid), model)
T = model.temperature

# Saturation mixing ratio (kg/kg) and water vapor initial condition
# Compute specific humidity qᵛ, then convert to total moisture density ρqᵗ
# For Kessler, qᵛ is diagnosed from qᵛ = qᵗ - qᶜˡ - qʳ
# Initially qᶜˡ = qʳ = 0, so qᵗ = qᵛ
ρᵣ = formulation.reference_state.density
qᵛᵢ = Field{Center, Center, Center}(grid)
ph_host = Array(parent(ph)) # bring to CPU to avoid GPU scalar indexing
T_host = Array(parent(T))
qᵛᵢ_host = similar(ph_host)

for k in axes(qᵛᵢ_host, 3), j in axes(qᵛᵢ_host, 2), i in axes(qᵛᵢ_host, 1)
    z = znode(i, j, k, grid, Center(), Center(), Center())
    T_eq = @inbounds T_host[i, j, k]
    p_eq = @inbounds ph_host[i, j, k]
    local qᵛ⁺ = 380 / p_eq * exp(17.27 * ((T_eq - 273) / (T_eq - 36)))
    @inbounds qᵛᵢ_host[i, j, k] = RHᵢ(z) * qᵛ⁺
end

copyto!(parent(qᵛᵢ), qᵛᵢ_host)

# Convert specific humidity to density-weighted total moisture
# Initially qᵗ = qᵛ since cloud and rain are zero
ρqᵗᵢ = ρᵣ * qᵛᵢ

# Set prognostic total moisture ρqᵗ (qᵛ is diagnosed as qᵗ - qᶜˡ - qʳ)
set!(model, ρqᵗ = ρqᵗᵢ, θ = θᵢ, u = uᵢ)
θ = liquid_ice_potential_temperature(model)

qᶜˡ = model.microphysical_fields.qᶜˡ
qʳ = model.microphysical_fields.qʳ
qᵛ = model.microphysical_fields.qᵛ

simulation = Simulation(model; Δt=2, stop_time=2hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

function progress(sim)
    u, v, w = sim.model.velocities
    qᵛ = sim.model.microphysical_fields.qᵛ
    qᶜˡ = sim.model.microphysical_fields.qᶜˡ
    qʳ = sim.model.microphysical_fields.qʳ

    ρe = static_energy_density(sim.model)
    ρemean = mean(ρe)
    msg = @sprintf("Iter: %d, t: %s, Δt: %s, mean(ρe): %.6e J/kg, max|u|: %.5f m/s, max w: %.5f m/s, min w: %.5f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), ρemean, maximum(abs, u), maximum(w), minimum(w))

    msg *= @sprintf(", max(qᵛ): %.5e, max(qᶜˡ): %.5e, max(qʳ): %.5e",
                    maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

outputs = merge(model.velocities, model.tracers, (; θ, qᶜˡ, qʳ, qᵛ))

filename = "supercell.jld2"

ow = JLD2Writer(model, outputs; filename,
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)


simulation.output_writers[:jld2] = ow

run!(simulation)

using CairoMakie
wt  = FieldTimeSeries(filename, "w")

times = wt.times
max_w = [maximum(wt[n]) for n in 1:length(times)]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time [s]", ylabel="Max w [m/s]", title="Maximum Vertical Velocity", xticks = 0:900:maximum(times))
lines!(ax, times, max_w)

save("max_w_timeseries.png", fig)
