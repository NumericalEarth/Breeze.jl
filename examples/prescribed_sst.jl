# # Prescribed sea surface temperature convection
#
# This example simulates moist convection driven by a prescribed sea surface temperature (SST).
# The simulation uses bulk aerodynamic formulas to compute surface fluxes of momentum,
# sensible heat, and latent heat based on bulk transfer coefficients.
# The model uses warm-phase saturation adjustment microphysics with liquid-ice
# potential temperature thermodynamics.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: xnode
using Printf

# ## Grid setup
#
# We use a 2D domain with periodic horizontal boundaries and a bounded vertical domain.
# The domain extends 20 km horizontally and 10 km vertically.

grid = RectilinearGrid(size = (128, 128), halo = (5, 5),
                       x = (-10kilometers, 10kilometers),
                       z = (0, 10kilometers),
                       topology = (Periodic, Flat, Bounded))

# ## Model formulation
#
# We create an AtmosphereModel with warm-phase saturation adjustment microphysics
# and liquid-ice potential temperature thermodynamics. The reference state
# uses a base pressure p₀ = 101325 Pa and reference potential temperature θ₀ = 285 K.

p₀, θ₀ = 101325, 285 # Pa, K
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants; base_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state, thermodynamics = :LiquidIcePotentialTemperature)
microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
momentum_advection = WENO(order=9)
scalar_advection = WENO(order=5)

# ## Surface flux parameters
#
# We define bulk transfer coefficients (Cᴰ for drag, Cᴴ for heat, Cᵛ for vapor)
# and a prescribed sea surface temperature θˢ(x) that varies as a Gaussian with
# a peak in the middle 10% of the domain.

@inline sea_surface_temperature(x, p) = p.T₀ + p.ΔT * sign(x)

parameters = (;
    constants, 
    drag_coefficient = 1e-3,
    heat_transfer_coefficient = 1e-3,
    vapor_transfer_coefficient = 1e-3,
    gust_speed = 1e-2,  # Minimum wind speed (m/s)
    T₀ = θ₀,   # Background SST (K)
    ΔT = 2,   # Maximum SST anomaly (K)
    ρ₀ = Breeze.Thermodynamics.base_density(p₀, θ₀, constants)
)

# ## Boundary condition functions
#
# The boundary conditions compute surface fluxes using bulk aerodynamic formulas.
# For potential temperature thermodynamics, we specify fluxes for ρθ and ρqᵗ.

@inline surface_saturation_specific_humidity(T, ρ, constants) =
    Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, constants, Breeze.Thermodynamics.PlanarLiquidSurface())

using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

@inline function s²ᶠᶜᶜ(i, j, grid, fields)
    u² = @inbounds fields.u[i, j, 1]^2
    v² = ℑxyᶠᶜᵃ(i, j, 1, grid, ϕ², fields.v)
    return u² + v²
end

@inline function s²ᶜᶠᶜ(i, j, grid, fields)
    u² = ℑxyᶜᶠᵃ(i, j, 1, grid, ϕ², fields.u)
    v² = @inbounds fields.v[i, j, 1]^2
    return u² + v²
end

@inline function s²ᶜᶜᶜ(i, j, grid, fields)
    u² = ℑxᶜᵃᵃ(i, j, 1, grid, ϕ², fields.u)
    v² = ℑyᵃᶜᵃ(i, j, 1, grid, ϕ², fields.v)
    return u² + v²
end

# Momentum flux (x-direction)
@inline function x_momentum_flux(i, j, grid, clock, fields, parameters)
    ρu = @inbounds fields.ρu[i, j, 1]
    U = sqrt(s²ᶠᶜᶜ(i, j, grid, fields))
    Uᵍ = parameters.gust_speed
    Ũ² = s²ᶠᶜᶜ(i, j, grid, fields) + Uᵍ^2
    Cᴰ = parameters.drag_coefficient
    return - Cᴰ * Ũ² * ρu / U * (U > 0)
end

# Momentum flux (y-direction)
@inline function y_momentum_flux(i, j, grid, clock, fields, parameters)
    ρv = @inbounds fields.ρv[i, j, 1]
    U = sqrt(s²ᶜᶠᶜ(i, j, grid, fields))
    Uᵍ = parameters.gust_speed
    Ũ² = s²ᶜᶠᶜ(i, j, grid, fields) + Uᵍ^2
    Cᴰ = parameters.drag_coefficient
    return - Cᴰ * Ũ² * ρv / U * (U > 0)
end

# Potential temperature density flux
# The sensible heat flux is ρ w'θ' = -ρ₀ Cᴴ U (θ_air - θ_surface)
# At the surface, θ_surface ≈ T_surface (since Exner function ≈ 1 at p ≈ p₀)
@inline function potential_temperature_flux(i, j, grid, clock, fields, parameters)
    Uᵍ = parameters.gust_speed
    Ũ = sqrt(s²ᶜᶜᶜ(i, j, grid, fields) + Uᵍ^2)
    
    x = xnode(i, j, 1, grid, Center(), Center(), Center())
    θˢ = sea_surface_temperature(x, parameters)
    
    ρ₀ = parameters.ρ₀
    Cᴴ = parameters.heat_transfer_coefficient
    Δθ = @inbounds fields.θ[i, j, 1] - θˢ
    
    return - ρ₀ * Cᴴ * Ũ * Δθ
end

# Moisture density flux
@inline function moisture_density_flux(i, j, grid, clock, fields, parameters)
    constants = parameters.constants
    Cᵛ = parameters.vapor_transfer_coefficient
    Uᵍ = parameters.gust_speed
    Ũ = sqrt(s²ᶜᶜᶜ(i, j, grid, fields) + Uᵍ^2)
    
    x = xnode(i, j, 1, grid, Center(), Center(), Center())
    Tˢ = sea_surface_temperature(x, parameters)
    ρ₀ = parameters.ρ₀
    qᵛ⁺ = surface_saturation_specific_humidity(Tˢ, ρ₀, constants)
    Δq = @inbounds fields.qᵗ[i, j, 1] - qᵛ⁺
    
    return - ρ₀ * Cᵛ * Ũ * Δq
end

# Create boundary conditions
ρu_surface_flux = FluxBoundaryCondition(x_momentum_flux; discrete_form=true, parameters)
ρv_surface_flux = FluxBoundaryCondition(y_momentum_flux; discrete_form=true, parameters)
ρθ_surface_flux = FluxBoundaryCondition(potential_temperature_flux; discrete_form=true, parameters)
ρqᵗ_surface_flux = FluxBoundaryCondition(moisture_density_flux; discrete_form=true, parameters)

ρu_bcs = FieldBoundaryConditions(bottom=ρu_surface_flux)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_surface_flux)
ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_surface_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_surface_flux)

# Create model with boundary conditions
model = AtmosphereModel(grid; momentum_advection, scalar_advection, microphysics, formulation,
                        boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs))

# ## Initial conditions
#
# We initialize the model with a constant potential temperature θ = θ₀ throughout the domain.

set!(model, θ=reference_state.potential_temperature, u=1)

# ## Simulation setup

simulation = Simulation(model, Δt=10, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Diagnostic fields

T = model.temperature
θ = liquid_ice_potential_temperature(model)
qˡ = model.microphysical_fields.qˡ
qᵛ⁺ = Breeze.Microphysics.SaturationSpecificHumidity(model)

# ## Progress callback

function progress(sim)
    qᵗ = sim.model.specific_moisture
    u, v, w = sim.model.velocities

    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    qᵗmin = minimum(qᵗ)
    qᵗmax = maximum(qᵗ)
    qˡmax = maximum(qˡ)

    θmin = minimum(θ)
    θmax = maximum(θ)

    msg = @sprintf("Iter: %d, t = %s, max|u|: (%.2e, %.2e, %.2e)",
                    iteration(sim), prettytime(sim), umax, vmax, wmax)

    msg *= @sprintf(", extrema(qᵗ): (%.2e, %.2e), max(qˡ): %.2e, extrema(T): (%.2e, %.2e)",
                     qᵗmin, qᵗmax, qˡmax, θmin, θmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# ## Output
#
# We write diagnostic fields to a JLD2 file for later analysis.

output_filename = joinpath(@__DIR__, "prescribed_sst_convection.jld2")
outputs = merge(model.velocities, (; T, θ, qˡ, qᵛ⁺, qᵗ=model.specific_moisture))

ow = JLD2Writer(model, outputs;
                filename = output_filename,
                schedule = TimeInterval(2minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

run!(simulation)

# ## Visualization
#
# The plotting code below can be uncommented to visualize the results.
# It creates animations of the temperature, moisture, and condensate fields.

using GLMakie

@assert isfile(output_filename) "Output file $(output_filename) not found."

u_ts = FieldTimeSeries(output_filename, "u")
w_ts = FieldTimeSeries(output_filename, "w")
θ_ts = FieldTimeSeries(output_filename, "θ")
T_ts = FieldTimeSeries(output_filename, "T")
qᵗ_ts = FieldTimeSeries(output_filename, "qᵗ")
qˡ_ts = FieldTimeSeries(output_filename, "qˡ")

times = θ_ts.times
Nt = length(θ_ts)

n = Observable(1)

u_snapshot = @lift u_ts[$n]
w_snapshot = @lift w_ts[$n]
θ_snapshot = @lift θ_ts[$n]
qᵗ_snapshot = @lift qᵗ_ts[$n]
T_snapshot = @lift T_ts[$n]
qˡ_snapshot = @lift qˡ_ts[$n]
title = @lift "t = $(prettytime(times[$n]))"

fig = Figure(size=(800, 800), fontsize=12)
axu = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")
axθ = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)")
axq = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)")
axT = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)")
axqˡ = Axis(fig[3, 2], xlabel="x (m)", ylabel="z (m)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

θ_limits = (minimum(θ_ts), maximum(θ_ts))
T_limits = (minimum(T_ts), maximum(T_ts))
u_limits = (minimum(u_ts), maximum(u_ts))
w_limits = (minimum(w_ts), maximum(w_ts))
qᵗ_max = maximum(qᵗ_ts)
qˡ_max = maximum(qˡ_ts)

hmu = heatmap!(axu, u_snapshot, colorrange=u_limits)
hmw = heatmap!(axw, w_snapshot, colorrange=w_limits)
hmθ = heatmap!(axθ, θ_snapshot, colorrange=θ_limits)
hmq = heatmap!(axq, qᵗ_snapshot, colorrange=(0, qᵗ_max), colormap=:magma)
hmT = heatmap!(axT, T_snapshot, colorrange=T_limits)
hmqˡ = heatmap!(axqˡ, qˡ_snapshot, colorrange=(0, qˡ_max), colormap=:magma)

Colorbar(fig[1, 0], hmu, label = "u [m/s]", vertical=true)
Colorbar(fig[1, 3], hmw, label = "w [m/s]", vertical=true)
Colorbar(fig[2, 0], hmθ, label = "θ [K]", vertical=true)
Colorbar(fig[1, 3], hmq, label = "qᵗ", vertical=true)
Colorbar(fig[2, 0], hmT, label = "T [K]", vertical=true)
Colorbar(fig[2, 3], hmqˡ, label = "qˡ", vertical=true)

fig

record(fig, joinpath(@__DIR__, "prescribed_sst.mp4"), 1:Nt, framerate=12) do nn
    n[] = nn
end

# Potential temperature animation
θ_anim_index = Observable(1)
θ_anim_snapshot = @lift θ_ts[$θ_anim_index]
θ_anim_title = @lift "Potential temperature: t = $(prettytime(times[$θ_anim_index]))"

θ_fig = Figure(size=(500, 400), fontsize=12)
θ_ax = Axis(θ_fig[1, 1], xlabel="x (m)", ylabel="z (m)")
θ_fig[0, :] = Label(θ_fig, θ_anim_title, fontsize=22, tellwidth=false)

θ_heatmap = heatmap!(θ_ax, θ_anim_snapshot, colorrange=θ_limits)
Colorbar(θ_fig[1, 2], θ_heatmap, label = "θ [K]", vertical=true)

record(θ_fig, joinpath(@__DIR__, "prescribed_sst_theta.mp4"), 1:Nt, framerate=12) do nn
    θ_anim_index[] = nn
end

nothing #hide
