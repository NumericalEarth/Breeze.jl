# # Prescribed sea surface temperature convection
#
# This example simulates moist convection driven by a prescribed sea surface temperature (SST).
# The simulation uses bulk aerodynamic formulas to compute surface fluxes of momentum,
# sensible heat, and latent heat based on bulk transfer coefficients and friction velocity.
# The model uses zero-moment bulk microphysics from CloudMicrophysics, which instantly
# removes precipitable condensate above a threshold, providing a simple representation
# of precipitation processes.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: xnode
using Statistics: mean
using Printf
using CloudMicrophysics

# Load CloudMicrophysics extension
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: ZeroMomentCloudMicrophysics

# ## Grid setup
#
# We use a 2D domain with periodic horizontal boundaries and a bounded vertical domain.
# The domain extends 20 km horizontally and 10 km vertically.

grid = RectilinearGrid(size = (128, 128), x = (-10e3, 10e3), z = (0, 10e3),
                       topology = (Periodic, Flat, Bounded))

# ## Model setup
#
# We create an AtmosphereModel with zero-moment bulk microphysics. The reference state
# uses a base pressure p₀ = 101325 Pa and reference potential temperature θ₀ = 285 K,
# typical of the lower atmosphere.

p₀, θ₀ = 101325, 285 # Pa, K
microphysics = ZeroMomentCloudMicrophysics()
reference_state = ReferenceState(grid, base_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state)
thermodynamics = ThermodynamicConstants()

# ## Surface flux parameters
#
# We define bulk transfer coefficients (Cᴰ for drag, Cᴴ for heat, Cᵛ for vapor)
# and a prescribed sea surface temperature θˢ(x) that varies as a Gaussian with
# a peak in the middle 10% of the domain. The fluxes are computed using bulk
# aerodynamic formulas with friction velocity u★.

# Domain extent for Gaussian SST distribution
@inline sea_surface_temperature(x, p) = p.T₀ + p.ΔT * exp(-x^2 / (2 * p.δx^2))

parameters = (;
    thermodynamics, 
    drag_coefficient = 1e-3,      # Cᴰ: drag coefficient
    heat_transfer_coefficient = 1e-3,  # Cᴴ: heat transfer coefficient
    vapor_transfer_coefficient = 1e-3,  # Cᵛ: vapor transfer coefficient
    gust_speed = 1e-2,  # Minimum friction velocity u★ (m/s)
    T₀ = 20,  # Sea surface temperature
    ΔT = 10,  # Maximum SST anomaly (K)
    δx = 1e3,  # Standard deviation of Gaussian SST distribution
    ρ₀ = Breeze.Thermodynamics.base_density(p₀, θ₀, thermodynamics)
)

# ## Boundary condition functions
#
# The boundary conditions compute surface fluxes using bulk aerodynamic formulas.
# We need to convert temperature and moisture fluxes to energy density and moisture
# density fluxes for AtmosphereModel.

# Utility for computing saturation specific humidity at sea surface
@inline surface_saturation_specific_humidity(T, ρ, thermo) =
    Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, Breeze.Thermodynamics.PlanarLiquidSurface())

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
# Returns kinematic momentum flux τₓ/ρ₀ = -u★² u/U for the ρu boundary condition
@inline function x_momentum_flux(i, j, grid, clock, fields, parameters)
    ρu = @inbounds fields.ρu[i, j, 1]
    U = sqrt(s²ᶠᶜᶜ(i, j, grid, fields))
    Uᵍ = parameters.gust_speed
    Ũ² = s²ᶠᶜᶜ(i, j, grid, fields) + Uᵍ^2
    Cᴰ = parameters.drag_coefficient
    return - Cᴰ * Ũ² * ρu / U * (U > 0)
end

# Momentum flux (y-direction)
# Returns kinematic momentum flux τᵧ/ρ₀ = -u★² v/U for the ρv boundary condition
@inline @inbounds function y_momentum_flux(i, j, grid, clock, fields, parameters)
    ρv = fields.ρv[i, j, 1]
    U = sqrt(s²ᶜᶠᶜ(i, j, grid, fields))
    Uᵍ = parameters.gust_speed
    Ũ² = s²ᶜᶠᶜ(i, j, grid, fields) + Uᵍ^2
    Cᴰ = parameters.drag_coefficient
    return - Cᴰ * Ũ² * ρv / U * (U > 0)
end

# Energy density flux (converted from potential temperature flux)
# The sensible heat flux is computed using bulk transfer: Jθ = -u★ θ★
# where θ★ is the temperature scale computed from the bulk transfer coefficient
@inline @inbounds function energy_density_flux(i, j, grid, clock, fields, parameters)
    thermo = parameters.thermodynamics
    Uᵍ = parameters.gust_speed
    Ũ = sqrt(s²ᶜᶜᶜ(i, j, grid, fields) + Uᵍ^2)
    
    # Get x coordinate for spatially varying SST using xnode
    x = xnode(i, j, 1, grid, Center(), Center(), Center())
    Tˢ = sea_surface_temperature(x, parameters)
    ρ₀ = parameters.ρ₀
    qᵛ⁺ = surface_saturation_specific_humidity(Tˢ, ρ₀, thermo)
    qˢ = Breeze.Thermodynamics.MoistureMassFractions(qᵛ⁺)
    cᵖᵐ = mixture_heat_capacity(qˢ, thermo)
    eˢ = cᵖᵐ * Tˢ
    
    # Get temperature from fields (approximate θ ≈ T near surface)
    Cᴴ = parameters.heat_transfer_coefficient
    Δe = fields.e[i, j, 1] - eˢ
    ρcᵖw′T′ = - ρ₀ * Cᴴ * Ũ * Δe

    ρw′q′ = moisture_density_flux(i, j, grid, clock, fields, parameters)
    ℒⁱᵣ = thermo.ice.reference_latent_heat
   
    return ρcᵖw′T′ + ℒⁱᵣ * ρw′q′
end

# Moisture density flux (converted from specific humidity flux)
@inline function moisture_density_flux(i, j, grid, clock, fields, parameters)
    thermo = parameters.thermodynamics
    Cᵛ = parameters.vapor_transfer_coefficient
    Uᵍ = parameters.gust_speed
    Ũ = sqrt(s²ᶜᶠᶜ(i, j, grid, fields) + Uᵍ^2)
    
    # Compute saturation specific humidity at sea surface
    x = xnode(i, j, 1, grid, Center(), Center(), Center())
    Tˢ = sea_surface_temperature(x, parameters)
    ρ₀ = parameters.ρ₀  # Use surface reference density
    qᵛ⁺ = surface_saturation_specific_humidity(Tˢ, ρ₀, thermo)
    Δq = fields.qᵗ[i, j, 1] - qᵛ⁺
    
    return - ρ₀ * Cᵛ * Ũ * Δq
end

# Create boundary conditions
ρu_surface_flux = FluxBoundaryCondition(x_momentum_flux; discrete_form=true, parameters)
ρv_surface_flux = FluxBoundaryCondition(y_momentum_flux; discrete_form=true, parameters)
ρe_surface_flux = FluxBoundaryCondition(energy_density_flux; discrete_form=true, parameters)
ρqᵗ_surface_flux = FluxBoundaryCondition(moisture_density_flux; discrete_form=true, parameters)

ρu_bcs = FieldBoundaryConditions(bottom=ρu_surface_flux)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_surface_flux)
ρe_bcs = FieldBoundaryConditions(bottom=ρe_surface_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_surface_flux)

# Create model with boundary conditions
model = AtmosphereModel(grid; advection = WENO(), microphysics, formulation,
                        boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs))

# ## Initial conditions
#
# We initialize the model with a constant potential temperature θ = θ₀ throughout the domain.
# Then we compute the saturation specific humidity field qᵛ⁺ and use it to construct a
# moisture profile qᵗ that:
# - Starts at 80% of saturation at the surface (z = 0)
# - Ramps linearly to 101% saturation at 1 km (supersaturated to promote cloud formation)
# - Remains constant at 101% from 1-2 km
# - Ramps linearly down to zero by 10 km (top of domain)


set!(model, θ=reference_state.potential_temperature)

# Update model state to compute temperature T from potential temperature θ
Breeze.AtmosphereModels.update_state!(model)

# ## Simulation setup

simulation = Simulation(model, Δt=10, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Diagnostic fields
#
# We compute diagnostic fields for temperature T, saturation specific humidity qᵛ⁺,
# liquid condensate mass fraction qˡ, and supersaturation δ = qᵗ - qᵛ⁺.

T = model.temperature
θ = Breeze.AtmosphereModels.PotentialTemperatureField(model)
qᵛ⁺ = Breeze.AtmosphereModels.SaturationSpecificHumidityField(model)

# Liquid condensate mass fraction from microphysical fields
# For zero-moment microphysics with saturation adjustment, qˡ is stored in microphysical_fields
qˡ = model.microphysical_fields.qˡ

# Supersaturation: positive values indicate supersaturated conditions
σ = Field(model.moisture_mass_fraction - qᵛ⁺)

# ## Progress callback

function progress(sim)
    compute!(T)
    compute!(qˡ)
    compute!(σ)
    
    qᵗ = sim.model.moisture_mass_fraction
    u, v, w = sim.model.velocities

    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    qᵗmin = minimum(qᵗ)
    qᵗmax = maximum(qᵗ)
    qˡmax = maximum(qˡ)
    σmax = maximum(σ)

    Tmin = minimum(T)
    Tmax = maximum(T)

    msg = @sprintf("Iter: %d, t = %s, max|u|: (%.2e, %.2e, %.2e)",
                    iteration(sim), prettytime(sim), umax, vmax, wmax)

    msg *= @sprintf(", extrema(qᵗ): (%.2e, %.2e), max(qˡ): %.2e, max(σ): %.2e, extrema(T): (%.2e, %.2e)",
                     qᵗmin, qᵗmax, qˡmax, σmax, Tmin, Tmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# ## Output
#
# We write diagnostic fields to a JLD2 file for later analysis.

output_filename = joinpath(@__DIR__, "prescribed_sst_convection.jld2")
outputs = merge(model.velocities, (; T, θ, qˡ, qᵛ⁺, σ, qᵗ=model.moisture_mass_fraction))

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

@assert isfile(output_filename) "Output file $(output_filename) not found. Uncomment and run the simulation block above to generate it."

θ_ts = FieldTimeSeries(output_filename, "θ")
T_ts = FieldTimeSeries(output_filename, "T")
qᵗ_ts = FieldTimeSeries(output_filename, "qᵗ")
qˡ_ts = FieldTimeSeries(output_filename, "qˡ")

times = θ_ts.times
Nt = length(θ_ts)

n = Observable(1)

θ_snapshot = @lift θ_ts[$n]
qᵗ_snapshot = @lift qᵗ_ts[$n]
T_snapshot = @lift T_ts[$n]
qˡ_snapshot = @lift qˡ_ts[$n]
title = @lift "t = $(prettytime(times[$n]))"

fig = Figure(size=(800, 400), fontsize=12)
axθ = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")
axT = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

θ_limits = (minimum(θ_ts), maximum(θ_ts))
T_limits = (minimum(T_ts), maximum(T_ts))
qᵗ_max = maximum(qᵗ_ts)
qˡ_max = maximum(qˡ_ts)

hmθ = heatmap!(axθ, θ_snapshot, colorrange=θ_limits)
hmq = heatmap!(axq, qᵗ_snapshot, colorrange=(0, qᵗ_max), colormap=:magma)
hmT = heatmap!(axT, T_snapshot, colorrange=T_limits)
#hmqˡ = heatmap!(axqˡ, qˡ_snapshot, colorrange=(0, qˡ_max), colormap=:magma)

Colorbar(fig[1, 0], hmθ, label = "θ [K]", vertical=true)
Colorbar(fig[1, 3], hmq, label = "qᵗ", vertical=true)
Colorbar(fig[2, 0], hmT, label = "T [K]", vertical=true)
#Colorbar(fig[2, 3], hmqˡ, label = "qˡ", vertical=true)

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