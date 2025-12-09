# # Prescribed sea surface temperature convection
#
# This example simulates moist convection driven by a prescribed sea surface temperature (SST).
# The simulation models the atmospheric response to a horizontally-varying SST pattern,
# a fundamental problem in atmosphere-ocean interaction studies. The setup is representative
# of convection over oceanic fronts or sea surface temperature gradients, where differential
# heating drives organized atmospheric circulations.
#
# The simulation uses bulk aerodynamic formulas to compute surface fluxes of momentum,
# sensible heat, and latent heat based on bulk transfer coefficients. This approach
# parameterizes the complex turbulent exchange processes in the surface layer using
# simple drag law formulations that relate fluxes to the difference between surface
# and near-surface atmospheric properties.
#
# The model uses warm-phase saturation adjustment microphysics with liquid-ice
# potential temperature thermodynamics. Saturation adjustment instantly condenses
# or evaporates water vapor to maintain thermodynamic equilibrium, providing a
# simple yet effective representation of cloud processes in moist convection.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: xnode
using Printf

# ## Grid setup
#
# We use a 2D domain (x-z plane) with periodic horizontal boundaries and a bounded
# vertical domain. The horizontal periodicity allows convective cells to develop
# and interact without artificial boundary effects. The domain extends 20 km
# horizontally to accommodate multiple convective cells, and 10 km vertically
# to capture the full depth of tropospheric convection.
#
# The grid resolution of 128 points in each direction provides approximately
# 156 m horizontal and 78 m vertical resolution, sufficient to resolve the
# energy-containing scales of convective turbulence while remaining computationally
# tractable for this demonstration.

grid = RectilinearGrid(size = (128, 128), halo = (5, 5),
                       x = (-10kilometers, 10kilometers),
                       z = (0, 10kilometers),
                       topology = (Periodic, Flat, Bounded))

# ## Model formulation
#
# We create an AtmosphereModel with warm-phase saturation adjustment microphysics
# and liquid-ice potential temperature thermodynamics. The anelastic formulation
# filters acoustic waves while retaining the essential dynamics of deep convection,
# allowing larger time steps than a fully compressible model.
#
# The reference state defines the background thermodynamic profile against which
# perturbations evolve. We use a base pressure p₀ = 101325 Pa (standard sea level
# pressure) and reference potential temperature θ₀ = 285 K, representing a
# relatively cool maritime atmosphere.

p₀, θ₀ = 101325, 285 # Pa, K
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants; base_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state, thermodynamics = :LiquidIcePotentialTemperature)

# The microphysics scheme uses saturation adjustment to maintain thermodynamic
# equilibrium. The `WarmPhaseEquilibrium` option considers only liquid water
# and vapor, appropriate for warm convection where ice processes are negligible.

microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

# We use high-order WENO advection schemes to accurately represent the sharp
# gradients that develop in convective flows. WENO (Weighted Essentially
# Non-Oscillatory) schemes provide excellent shock-capturing properties while
# maintaining high accuracy in smooth regions.

momentum_advection = WENO(order=9)
scalar_advection = WENO(order=5)

# ## Surface flux parameterization
#
# The surface fluxes are computed using bulk aerodynamic formulas, which relate
# the turbulent fluxes to the difference between surface and atmospheric properties
# multiplied by a transfer coefficient and wind speed. This approach parameterizes
# the complex turbulent exchange processes in the atmospheric surface layer.
#
# The bulk transfer coefficients are:
# - Cᴰ (drag coefficient): relates surface momentum flux to wind speed
# - Cᴴ (heat transfer coefficient): relates sensible heat flux to temperature difference
# - Cᵛ (vapor transfer coefficient): relates latent heat flux to humidity difference
#
# The sea surface temperature varies as a step function across the domain center,
# creating a sharp SST front. This idealized pattern drives a strong circulation
# with rising motion over the warm side and sinking motion over the cold side.

@inline sea_surface_temperature(x, p) = p.T₀ + p.ΔT * sign(cos(2π * x / p.Lx))

parameters = (;
    constants,
    drag_coefficient = 1e-3,
    heat_transfer_coefficient = 1e-3,
    vapor_transfer_coefficient = 1e-3,
    gust_speed = 1e-2,  # Minimum wind speed (m/s)
    T₀ = θ₀,   # Background SST (K)
    ΔT = 2,   # Maximum SST anomaly (K)
    ρ₀ = Breeze.Thermodynamics.base_density(p₀, θ₀, constants),
    Lx = grid.Lx
)

# ## Boundary condition functions
#
# The boundary conditions compute surface fluxes using bulk aerodynamic formulas.
# For potential temperature thermodynamics, we specify fluxes for the potential
# temperature density ρθ and moisture density ρqᵗ.
#
# The flux formulas follow the standard bulk aerodynamic approach:
# ```math
# F_\phi = -\rho_0 C_\phi U (\phi_{air} - \phi_{surface})
# ```
# where φ represents potential temperature or specific humidity, Cᵩ is the
# corresponding transfer coefficient, and U is the near-surface wind speed.

@inline surface_saturation_specific_humidity(T, ρ, constants) =
    Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, constants, Breeze.Thermodynamics.PlanarLiquidSurface())

# We need interpolation operators to compute wind speed at the appropriate
# grid locations for each flux calculation.

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

# The momentum flux (surface stress) uses a quadratic drag law. The stress is
# proportional to the square of the wind speed, directed opposite to the
# near-surface velocity. A small "gust speed" prevents division by zero
# when winds are calm.

@inline function x_momentum_flux(i, j, grid, clock, fields, parameters)
    ρu = @inbounds fields.ρu[i, j, 1]
    U = sqrt(s²ᶠᶜᶜ(i, j, grid, fields))
    Uᵍ = parameters.gust_speed
    Ũ² = s²ᶠᶜᶜ(i, j, grid, fields) + Uᵍ^2
    Cᴰ = parameters.drag_coefficient
    return - Cᴰ * Ũ² * ρu / U * (U > 0)
end

@inline function y_momentum_flux(i, j, grid, clock, fields, parameters)
    ρv = @inbounds fields.ρv[i, j, 1]
    U = sqrt(s²ᶜᶠᶜ(i, j, grid, fields))
    Uᵍ = parameters.gust_speed
    Ũ² = s²ᶜᶠᶜ(i, j, grid, fields) + Uᵍ^2
    Cᴰ = parameters.drag_coefficient
    return - Cᴰ * Ũ² * ρv / U * (U > 0)
end

# The sensible heat flux transfers heat between the ocean surface and atmosphere.
# At the surface, the potential temperature approximately equals the temperature
# since the Exner function is close to unity at surface pressure.

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

# The latent heat flux (moisture flux) transfers water vapor between the ocean
# and atmosphere. The ocean surface is assumed to be saturated at the SST,
# so the flux depends on the difference between the saturation specific humidity
# at the surface and the actual specific humidity in the near-surface air.

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

# Assemble the boundary conditions for all prognostic variables.
# Each flux boundary condition uses `discrete_form=true` to access the
# grid indices directly, enabling efficient computation of spatially-varying fluxes.

ρu_surface_flux = FluxBoundaryCondition(x_momentum_flux; discrete_form=true, parameters)
ρv_surface_flux = FluxBoundaryCondition(y_momentum_flux; discrete_form=true, parameters)
ρθ_surface_flux = FluxBoundaryCondition(potential_temperature_flux; discrete_form=true, parameters)
ρqᵗ_surface_flux = FluxBoundaryCondition(moisture_density_flux; discrete_form=true, parameters)

ρu_bcs = FieldBoundaryConditions(bottom=ρu_surface_flux)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_surface_flux)
ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_surface_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_surface_flux)

# ## Model construction
#
# We assemble the AtmosphereModel with all the components defined above.
# The model will solve the anelastic equations with the specified advection
# schemes, microphysics, and boundary conditions.

model = AtmosphereModel(grid; momentum_advection, scalar_advection, microphysics, formulation,
                        boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs))

# ## Initial conditions
#
# We initialize the model with a uniform potential temperature equal to the
# reference value, creating a neutrally stratified atmosphere. A small
# background wind (1 m/s) in the x-direction provides initial momentum
# for the bulk flux calculations and helps break symmetry.

set!(model, θ=reference_state.potential_temperature, u=1)

# ## Simulation setup
#
# We configure the simulation to run for 4 hours with adaptive time stepping.
# The CFL condition limits the time step to maintain numerical stability,
# with a target CFL number of 0.7 providing a good balance between efficiency
# and accuracy.

simulation = Simulation(model, Δt=10, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Diagnostic fields
#
# We define several diagnostic quantities for analysis and visualization:
# - Temperature T: the actual temperature field
# - Potential temperature θ: conserved in dry adiabatic processes
# - Liquid water content qˡ: mass fraction of cloud liquid water
# - Saturation specific humidity qᵛ⁺: maximum water vapor the air can hold

T = model.temperature
θ = liquid_ice_potential_temperature(model)
qˡ = model.microphysical_fields.qˡ
qᵛ⁺ = Breeze.Microphysics.SaturationSpecificHumidity(model)

# ## Turbulent flux diagnostics
#
# We compute horizontally-averaged vertical turbulent fluxes, which characterize
# the transport of momentum and scalars by convective motions. These fluxes are
# essential diagnostics for understanding the dynamics of convective boundary layers.
#
# The fluxes are computed as horizontal averages of the products of vertical
# velocity with the transported quantity. For 2D simulations, we average along
# the x-direction (dims=1).

u, v, w = model.velocities
qᵗ = model.specific_moisture

# Vertical flux of horizontal momentum (Reynolds stress)
wu = Average(w * u, dims=1)

# Vertical flux of potential temperature (sensible heat flux)
wθ = Average(w * θ, dims=1)

# Vertical flux of total water (moisture flux)
wqᵗ = Average(w * qᵗ, dims=1)

# We also save the mean profiles for computing turbulent perturbations in post-processing
u_avg = Average(u, dims=1)
w_avg = Average(w, dims=1)
θ_avg = Average(θ, dims=1)
qᵗ_avg = Average(qᵗ, dims=1)
qˡ_avg = Average(qˡ, dims=1)

# ## Progress callback
#
# A callback function prints diagnostic information every 10 iterations,
# helping monitor the simulation's progress and detect any numerical issues.

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

    msg *= @sprintf(", extrema(qᵗ): (%.2e, %.2e), max(qˡ): %.2e, extrema(θ): (%.2e, %.2e)",
                     qᵗmin, qᵗmax, qˡmax, θmin, θmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, TimeInterval(20minutes))

# ## Output
#
# We configure two output writers:
# 1. Full 2D fields for visualization and detailed analysis
# 2. Horizontally-averaged profiles and fluxes for bulk statistics
#
# The JLD2 format provides efficient storage with full Julia type preservation.

output_filename = joinpath(@__DIR__, "prescribed_sst_convection.jld2")
outputs = merge(model.velocities, (; T, θ, qˡ, qᵛ⁺, qᵗ=model.specific_moisture))

ow = JLD2Writer(model, outputs;
                filename = output_filename,
                schedule = TimeInterval(2minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

# Horizontally-averaged profiles and fluxes
averages_filename = joinpath(@__DIR__, "prescribed_sst_averages.jld2")
averaged_outputs = (; u_avg, w_avg, θ_avg, qᵗ_avg, qˡ_avg, wu, wθ, wqᵗ)

averages_ow = JLD2Writer(model, averaged_outputs;
                         filename = averages_filename,
                         schedule = TimeInterval(2minutes),
                         overwrite_existing = true)

simulation.output_writers[:averages] = averages_ow

# ## Run the simulation

@info "Running prescribed SST convection simulation..."
run!(simulation)

# ## Visualization
#
# We create animations showing the evolution of the flow fields. The 2×3 panel
# layout displays velocity components (u, w), thermodynamic fields (θ, T),
# and moisture fields (qᵗ, qˡ).

using CairoMakie

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

fig = Figure(size=(800, 800), fontsize=12)

title = @lift "t = $(prettytime(times[$n]))"

axu = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")
axθ = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)")
axq = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)")
axT = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)")
axqˡ = Axis(fig[3, 2], xlabel="x (m)", ylabel="z (m)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

# Compute color limits from the full time series
θ_limits = (minimum(θ_ts), maximum(θ_ts))
T_limits = (minimum(T_ts), maximum(T_ts))
u_limits = (minimum(u_ts), maximum(u_ts))
w_max = max(abs(minimum(w_ts)), abs(maximum(w_ts)))
w_limits = (-w_max, w_max)
qᵗ_max = maximum(qᵗ_ts)
qˡ_max = maximum(qˡ_ts)

hmu = heatmap!(axu, u_snapshot, colorrange=u_limits, colormap=:balance)
hmw = heatmap!(axw, w_snapshot, colorrange=w_limits, colormap=:balance)
hmθ = heatmap!(axθ, θ_snapshot, colorrange=θ_limits)
hmq = heatmap!(axq, qᵗ_snapshot, colorrange=(0, qᵗ_max), colormap = Reverse(:Purples_4))
hmT = heatmap!(axT, T_snapshot, colorrange=T_limits)
hmqˡ = heatmap!(axqˡ, qˡ_snapshot, colorrange=(0, qˡ_max), colormap = Reverse(:Blues_4))

Colorbar(fig[1, 0], hmu, label = "u [m/s]", vertical=true)
Colorbar(fig[1, 3], hmw, label = "w [m/s]", vertical=true)
Colorbar(fig[2, 0], hmθ, label = "θ [K]", vertical=true)
Colorbar(fig[2, 3], hmq, label = "qᵗ", vertical=true)
Colorbar(fig[3, 0], hmT, label = "T [K]", vertical=true)
Colorbar(fig[3, 3], hmqˡ, label = "qˡ", vertical=true)

fig

# And we can also make movies

CairoMakie.record(fig, "prescribed_sst.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](prescribed_sst.mp4)


# Potential temperature animation
n = Observable(1)
θ_snapshot = @lift θ_ts[$n]
title = @lift "Potential temperature: t = $(prettytime(times[$n]))"

fig = Figure(size=(500, 400), fontsize=12)
ax = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

hm = heatmap!(ax, θ_snapshot, colorrange=θ_limits)
Colorbar(fig[1, 2], hm, label = "θ [K]", vertical=true)

fig

CairoMakie.record(fig, "prescribed_sst_theta.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](prescribed_sst_theta.mp4)
