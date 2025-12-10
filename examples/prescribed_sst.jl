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
using Oceananigans.Models: BoundaryConditionOperation
using Printf
using CairoMakie
using Statistics: mean

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
# Non-Oscillatory; [Shu09](@citet)) schemes provide excellent shock-capturing
# properties while maintaining high accuracy in smooth regions.

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
# - Cᵀ (sensible heat transfer coefficient): relates sensible heat flux to temperature difference
# - Cᵛ (vapor transfer coefficient): relates latent heat flux to humidity difference
#
# The sea surface temperature varies as a step function across the domain center,
# creating a sharp SST front. This idealized pattern drives a strong circulation
# with rising motion over the warm side and sinking motion over the cold side.

# Sea surface temperature as a function of x
ΔT = 2 # K
T₀(x) = θ₀ + ΔT * sign(cos(2π * x / grid.Lx))

# We define the transfer coefficients and gustiness parameter.
Cᴰ = 1e-3  # Drag coefficient
Cᵀ = 1e-3  # Sensible heat transfer coefficient
Cᵛ = 1e-3  # Vapor transfer coefficient
Uᵍ = 1e-2  # Minimum wind speed (m/s)

# ## Boundary conditions
#
# Breeze provides convenient abstractions for bulk aerodynamic surface fluxes.
# The `BulkDrag`, `BulkSensibleHeatFlux`, and `BulkVaporFlux` boundary conditions
# compute fluxes using bulk aerodynamic formulas:
#
# ```math
# Jᵘ = - Cᴰ |U| ρu, \\quad Jᵀ = - ρ₀ Cᵀ |U| (θ - θ₀), \\quad Jᵛ = - ρ₀ Cᵛ |U| (qᵗ - qᵛ₀)
# ```
#
# where ``|U|`` is the wind speed (including gustiness), ``Cᴰ, Cᵀ, Cᵛ`` are transfer
# coefficients, and ``θ₀, qᵛ₀`` are the surface temperature and saturation specific humidity.
#
# The reference density ``ρ₀`` is automatically computed from the model's reference state,
# and for `BulkVaporFlux`, the saturation specific humidity is computed from the surface
# temperature if not provided explicitly. Surface temperature can be provided as a
# `Field`, a `Function`, or a `Number`.

ρu_surface_flux = BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ)
ρv_surface_flux = BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ)
ρθ_surface_flux = BulkSensibleHeatFlux(coefficient = Cᵀ, gustiness = Uᵍ, surface_temperature = T₀)
ρqᵗ_surface_flux = BulkVaporFlux(coefficient = Cᵛ, gustiness = Uᵍ, surface_temperature = T₀)

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

ρu, ρv, ρw = model.momentum
u, v, w = model.velocities
qᵗ = model.specific_moisture

# ## Surface flux diagnostics
#
# We use Oceananigans' `BoundaryConditionOperation` to extract the surface flux
# values from the boundary conditions. These 1D fields (varying only in x)
# represent the actual flux values applied at the ocean-atmosphere interface.
#
# The surface fluxes are:
# - τˣ: surface momentum flux (stress), in kg m⁻¹ s⁻²
# - 𝒬ᵀ: sensible heat flux = cᵖᵐ × Jᵀ, in W m⁻²
# - 𝒬ᵛ: latent heat flux = ℒˡ × Jᵛ, in W m⁻²
#
# where Jᵀ is the temperature flux and Jᵛ is the moisture flux.

# Surface momentum flux
τˣ = BoundaryConditionOperation(ρu, :bottom, model)

# Sensible heat flux: 𝒬ᵀ = cᵖᵐ × Jᵀ
ρθ = liquid_ice_potential_temperature_density(model)
cᵖᵈ = constants.dry_air.heat_capacity
cᵖᵛ = constants.vapor.heat_capacity
cᵖᵐ = cᵖᵈ * (1 - qᵛ₀) + cᵖᵛ * qᵛ₀
Jᵀ = BoundaryConditionOperation(ρθ, :bottom, model)
𝒬ᵀ = cᵖᵐ * Jᵀ

# Latent heat flux: 𝒬ᵛ = ℒˡ × Jᵛ
ρqᵗ = model.moisture_density
ℒˡ = Breeze.Thermodynamics.liquid_latent_heat(T₀, constants)
Jᵛ = BoundaryConditionOperation(ρqᵗ, :bottom, model)
𝒬ᵛ = ℒˡ * Jᵛ

# ## Progress callback
#
# A callback function prints diagnostic information every few iterations,
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

add_callback!(simulation, progress, IterationInterval(100))

# ## Output
#
# We save both the full 2D fields and the 1D surface flux fields.
# The JLD2 format provides efficient storage with full Julia type preservation.

output_filename = joinpath(@__DIR__, "prescribed_sst_convection.jld2")
qᵗ = model.specific_moisture
outputs = merge(model.velocities, (; T, θ, qˡ, qᵛ⁺, qᵗ, τˣ, 𝒬ᵀ, 𝒬ᵛ, Σ𝒬=𝒬ᵀ+𝒬ᵛ))

ow = JLD2Writer(model, outputs;
                filename = output_filename,
                schedule = TimeInterval(2minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

# ## Run the simulation

@info "Running prescribed SST convection simulation..."
run!(simulation)

# ## Visualization
#
# We create animations showing the evolution of the flow fields. The figure
# displays velocity components (u, w), thermodynamic fields (θ, T),
# moisture fields (qᵗ, qˡ), and surface fluxes (momentum and heat).

@assert isfile(output_filename) "Output file $(output_filename) not found."

u_ts = FieldTimeSeries(output_filename, "u")
w_ts = FieldTimeSeries(output_filename, "w")
θ_ts = FieldTimeSeries(output_filename, "θ")
T_ts = FieldTimeSeries(output_filename, "T")
qᵗ_ts = FieldTimeSeries(output_filename, "qᵗ")
qˡ_ts = FieldTimeSeries(output_filename, "qˡ")
τˣ_ts = FieldTimeSeries(output_filename, "τˣ")
𝒬ᵀ_ts = FieldTimeSeries(output_filename, "𝒬ᵀ")
𝒬ᵛ_ts = FieldTimeSeries(output_filename, "𝒬ᵛ")
Σ𝒬_ts = FieldTimeSeries(output_filename, "Σ𝒬")

times = θ_ts.times
Nt = length(θ_ts)

n = Observable(Nt)

un = @lift u_ts[$n]
wn = @lift w_ts[$n]
θn = @lift θ_ts[$n]
qᵗn = @lift qᵗ_ts[$n]
Tn = @lift T_ts[$n]
qˡn = @lift qˡ_ts[$n]
τˣn = @lift τˣ_ts[$n]
𝒬ᵀn = @lift 𝒬ᵀ_ts[$n]
𝒬ᵛn = @lift 𝒬ᵛ_ts[$n]
Σ𝒬n = @lift Σ𝒬_ts[$n]

# We compute some extra diagnostics, like the total speed, ``\sqrt{u² + w²}`` and
# the cross-stream vorticity ``∂u/∂z - ∂w/∂x``.

sn = @lift sqrt(u_ts[$n]^2 + w_ts[$n]^2)
ξn = @lift ∂z(u_ts[$n]) - ∂x(w_ts[$n])

# Now we are ready to plot.

fig = Figure(size=(800, 1000), fontsize=13)

title = @lift "t = $(prettytime(times[$n]))"

axs = Axis(fig[1, 1], ylabel="z (m)")
axξ = Axis(fig[1, 2])
axθ = Axis(fig[2, 1], ylabel="z (m)")
axq = Axis(fig[2, 2])
axT = Axis(fig[3, 1], ylabel="z (m)")
axqˡ = Axis(fig[3, 2])

# Surface flux plots at bottom
axτ = Axis(fig[4, 1], xlabel="x (m)", ylabel="τˣ (kg m⁻¹ s⁻²)", title="Surface momentum flux")
ax𝒬 = Axis(fig[4, 2], xlabel="x (m)", ylabel="𝒬 (W m⁻²)", title="Surface heat flux (𝒬ᵀ + 𝒬ᵛ)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

# Compute color limits from the full time series
θ_limits = extrema(θ_ts)
T_limits = extrema(T_ts)
s_limits = (0, mean([maximum(u_ts), maximum(w_ts)]))
ξ_limits = (-0.021, 0.021)

qᵗ_max = maximum(qᵗ_ts)
qˡ_max = maximum(qˡ_ts)

# Flux limits
τˣ_max = max(abs(minimum(τˣ_ts)), abs(maximum(τˣ_ts)))
𝒬_min = min(minimum(𝒬ᵀ_ts), minimum(𝒬ᵛ_ts), minimum(Σ𝒬_ts))
𝒬_max = max(maximum(𝒬ᵀ_ts), maximum(𝒬ᵛ_ts), maximum(Σ𝒬_ts))

hms = heatmap!(axs, sn, colorrange=s_limits, colormap=:speed)
hmξ = heatmap!(axξ, ξn, colorrange=ξ_limits, colormap=:balance)
hmθ = heatmap!(axθ, θn, colorrange=θ_limits, colormap=:thermal)
hmq = heatmap!(axq, qᵗn, colorrange=(0, qᵗ_max), colormap=Reverse(:Purples_4))
hmT = heatmap!(axT, Tn, colorrange=T_limits)
hmqˡ = heatmap!(axqˡ, qˡn, colorrange=(0, qˡ_max), colormap=Reverse(:Blues_4))

# Plot the surface fluxes
lines!(axτ, τˣn, color=:black, linewidth=2)

lines!(ax𝒬, 𝒬ᵀn, color=:firebrick, linewidth=2, label="sensible")
lines!(ax𝒬, 𝒬ᵛn, color=:blue, linewidth=2, label="latent")
lines!(ax𝒬, Σ𝒬n, color=:green, linewidth=4, label="total")
Legend(fig[4, 3], ax𝒬)

# Add zero lines, fix axis limits, and add colorbars.

for ax in (axτ, ax𝒬)
    lines!(ax, [-grid.Lx/2, grid.Lx/2], [0, 0], color=:grey, linestyle=:dash)
end

for ax in (axs, axξ, axθ, axq, axT, axqˡ, axτ, ax𝒬)
    xlims!(ax, -grid.Lx/2, grid.Lx/2)
end

ylims!(axτ, -τˣ_max, τˣ_max)
ylims!(ax𝒬, 𝒬_min, 𝒬_max)

Colorbar(fig[1, 0], hms, label="√(u² + w²) (m/s)", flipaxis=false)
Colorbar(fig[1, 3], hmξ, label="∂u/∂z - ∂w/∂x (s⁻¹)")
Colorbar(fig[2, 0], hmθ, label="θ (K)", flipaxis=false)
Colorbar(fig[2, 3], hmq, label="qᵗ (kg/kg)")
Colorbar(fig[3, 0], hmT, label="T (K)", flipaxis=false)
Colorbar(fig[3, 3], hmqˡ, label="qˡ (kg/kg)")

# Now we are ready to make a cool animation.

CairoMakie.record(fig, "prescribed_sst.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](prescribed_sst.mp4)
