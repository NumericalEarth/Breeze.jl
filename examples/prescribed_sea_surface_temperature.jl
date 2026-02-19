# # Prescribed sea surface temperature convection
#
# This example simulates moist convection driven by a prescribed sea surface temperature (SST).
# The simulation models the atmospheric response to a horizontally-varying SST pattern,
# a fundamental problem in atmosphere-ocean interaction studies. The setup is representative
# of convection over oceanic fronts or sea surface temperature gradients, where differential
# heating drives organized atmospheric circulations.
#
# The simulation uses bulk aerodynamic formulas to compute surface fluxes of momentum,
# sensible heat, and latent heat based on bulk exchange coefficients. This approach
# parameterizes the complex turbulent exchange processes in the surface layer using
# simple drag law formulations that relate fluxes to the difference between surface
# and near-surface atmospheric properties.
#
# The model uses warm-phase saturation adjustment microphysics with liquid-ice
# potential temperature thermodynamics. Saturation adjustment instantly condenses
# or evaporates water vapor to maintain thermodynamic equilibrium, providing a
# simple yet effective representation of cloud processes in moist convection.

using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie

# ## Grid setup
#
# We use a 2D domain (x-z plane) with periodic horizontal boundaries and a bounded
# vertical domain. The horizontal periodicity allows convective cells to develop
# and interact without artificial boundary effects. The domain extends 20 km
# horizontally and 10 km vertically.
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
reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)
dynamics = AnelasticDynamics(reference_state)

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

# ## Boundary conditions
#
# Breeze provides abstractions for specifying bulk surface fluxes.
# The `BulkDrag`, `BulkSensibleHeatFlux`, and `BulkVaporFlux` boundary conditions
# compute fluxes of momentum, potential temperature density, and moisture density
# according to bulk aerodynamic formulae that relate turbulent fluxes to the
# difference between atmosphere properties, surface properties, and the differential
# motion of the air and surface,
#
# ```math
# τˣ = - ρ₀ Cᴰ |ΔU| (u - u₀), \quad Jᶿ = - ρ₀ Cᵀ |ΔU| (θ - θ₀), \quad Jᵛ = - ρ₀ Cᵛ |ΔU| (qᵗ - qᵛ₀),
# ```
#
# where ``Cᴰ, Cᵀ, Cᵛ`` are exchange coefficients, ``|ΔU| = [(u - u₀)^2 + (v - v₀)^2 + {U^g}^2]^{1/2}``
# is the total differential wind speed (including gustiness ``Uᵍ``), and ``u₀, θ₀, qᵛ₀`` are the
# surface velocity, surface temperature, and surface specific humidity respectively. For wet surfaces,
# ``qᵛ₀`` is the saturation specific humidity over a planar liquid surface computed at the surface temperature.
# ``τˣ`` is the surface momentum flux, ``Jᶿ`` is the potential temperature density flux,
# and ``Jᵛ`` is the surface moisture density flux.
# The surface density ``ρ₀`` is computed from the model's reference state.
#
# The potential temperature flux is proportional to the sensible heat flux,
#
# ```math
# 𝒬ᵀ = cᵖᵐ Jᶿ
# ```
#
# where ``cᵖᵐ`` is the mixture heat capacity.
#
# ## Wind and stability-dependent exchange coefficients
#
# Rather than using constant exchange coefficients, we use [`PolynomialCoefficient`](@ref)
# which implements the wind speed and stability-dependent formulation from [LargeYeager2009](@citet).
# This provides a more realistic representation of air-sea exchange processes,
# compared to constant exchange coefficients.
#
# In neutral conditions, the exchange coefficients vary with wind speed according to:
#
# ```math
# C^N_{10}(U_h) = (a₀ + a₁ U_h + a₂ / U_h) × 10⁻³
# ```
#
# and are further modified by atmospheric stability using the bulk Richardson number,
#
# ```math
# Riᴮ = \frac{g}{\overline{θ_v}} \frac{h \, (θ_v - θ_{v0})}{U_h^2}
# ```
#
# where ``h`` is the measurement height (first cell center), ``θ_v`` and ``θ_{v0}``
# are virtual potential temperatures at the measurement height and surface, and
# ``U_h`` is the wind speed at height ``h``.
#
# The default stability correction uses [`FittedStabilityFunction`](@ref), which maps
# ``Riᴮ`` to the Monin-Obukhov stability parameter ``ζ = z/L`` via the non-iterative
# regression of [Li2010](@citet), then evaluates integrated MOST stability functions
# ``Ψᴰ(ζ)`` and ``Ψᵀ(ζ)`` ([hogstrom1996review](@citet) for unstable,
# [beljaars1991flux](@citet) for stable conditions). The stability-corrected transfer
# coefficients are:
#
# ```math
# Cᴰ = Cᴰ_N \left(\frac{α}{α - Ψᴰ}\right)^2, \quad
# Cᵀ = Cᵀ_N \frac{α}{α - Ψᴰ} \frac{β_h}{β_h - Ψᵀ}
# ```
#
# where ``α = \ln(h/ℓ)`` and ``β_h = \ln(h/ℓ_h)`` with roughness lengths ``ℓ``
# (momentum) and ``ℓ_h`` (scalar). This provides structurally correct and different
# corrections for momentum and scalar transfer.
#
# In unstable conditions (over warm and wet surfaces), exchange is enhanced.
# In stable conditions (cold and dry surfaces), exchange is reduced.
#
# We create polynomial coefficients for each flux type. The default coefficients
# come from [LargeYeager2009](@citet) observational fits:

Uᵍ = 1e-2  # Gustiness (m/s)

# Create a polynomial bulk coefficient that will be automatically configured
# for each flux type
coef = PolynomialCoefficient(roughness_length = 1.5e-4)

# ## Surface temperature
#
# The sea surface temperature enters the bulk formulas for sensible heat,
# moisture fluxes, and (when using `PolynomialCoefficient`) the stability
# correction for the exchange coefficients.
#
# In this example, we specify the sea surface temperature as a top hat function
# i.e. representing a pair of ocean fronts in a periodic domain, with a
# difference of 4 degrees K,

ΔT = 4 # K
T₀(x) = θ₀ + ΔT / 2 * sign(cos(2π * x / grid.Lx))

# ## Momentum drag
#
# The `BulkDrag` boundary condition requires `surface_temperature` when using
# `PolynomialCoefficient`, since the stability correction depends on the
# surface virtual potential temperature.

ρu_surface_flux = ρv_surface_flux = BulkDrag(coefficient=coef, gustiness=Uᵍ, surface_temperature=T₀)

# ## Sensible heat flux and vapor fluxes
#
# For `BulkVaporFlux`, the saturation specific humidity is computed from the surface
# temperature. Surface temperature can be provided as a `Field`, a `Function`, or a `Number`.
#
# We complete our specification by using the same polynomial coefficient for
# sensible and latent heat fluxes. The flux type will be automatically inferred:

ρe_surface_flux = BulkSensibleHeatFlux(coefficient=coef, gustiness=Uᵍ, surface_temperature=T₀)
ρqᵗ_surface_flux = BulkVaporFlux(coefficient=coef, gustiness=Uᵍ, surface_temperature=T₀)

# We can visualize how the neutral drag coefficient varies with wind speed,
# and the range of stability-corrected values expected in this simulation.
# The SST ranges from ``θ₀ - ΔT/2`` (cold, stable) to ``θ₀ + ΔT/2`` (warm, unstable),
# so the stability correction spans these two limits.

using Breeze.BoundaryConditions: neutral_coefficient_10m, bulk_richardson_number,
                                 default_neutral_drag_polynomial

h = grid.Lz / grid.Nz / 2  # first cell center height
U_min = 0.1
ℓ = coef.roughness_length
sf = coef.stability_function
α = log(h / ℓ)
β = log(ℓ / sf.scalar_roughness_length)

ΔT_line = 10  # K, temperature difference for stability lines
T_warm = θ₀ + ΔT / 2      # warm SST in this simulation
T_cold = θ₀ - ΔT / 2      # cold SST in this simulation
T_unstable = θ₀ + ΔT_line  # strongly unstable
T_stable   = θ₀ - ΔT_line  # strongly stable

U_range = range(3, 25, length=200)
Cᴰ_neutral  = [neutral_coefficient_10m(default_neutral_drag_polynomial, U, U_min) for U in U_range]
Cᴰ_unstable = [Cᴰ * sf(bulk_richardson_number(h, θ₀, T_unstable, U, U_min), α, β) for (Cᴰ, U) in zip(Cᴰ_neutral, U_range)]
Cᴰ_stable   = [Cᴰ * sf(bulk_richardson_number(h, θ₀, T_stable,   U, U_min), α, β) for (Cᴰ, U) in zip(Cᴰ_neutral, U_range)]
Cᴰ_sim_warm = [Cᴰ * sf(bulk_richardson_number(h, θ₀, T_warm, U, U_min), α, β) for (Cᴰ, U) in zip(Cᴰ_neutral, U_range)]
Cᴰ_sim_cold = [Cᴰ * sf(bulk_richardson_number(h, θ₀, T_cold, U, U_min), α, β) for (Cᴰ, U) in zip(Cᴰ_neutral, U_range)]

fig = Figure(size=(1100, 400))

ax_coef = Axis(fig[1, 1],
               xlabel = "Wind speed (m/s)",
               ylabel = "Cᴰ × 10³",
               title = "Drag coefficient at 10 m")

band!(ax_coef, collect(U_range), Cᴰ_sim_cold .* 1e3, Cᴰ_sim_warm .* 1e3,
      color=(:grey, 0.3), label="Simulation range (ΔT = $ΔT K)")
lines!(ax_coef, U_range, Cᴰ_unstable .* 1e3, color=:firebrick,  linewidth=2, label="Unstable (ΔT = $ΔT_line K)")
lines!(ax_coef, U_range, Cᴰ_neutral  .* 1e3, color=:black,      linewidth=2, label="Neutral")
lines!(ax_coef, U_range, Cᴰ_stable   .* 1e3, color=:dodgerblue, linewidth=2, label="Stable (ΔT = -$ΔT_line K)")

axislegend(ax_coef, position=:rt)

ax_ratio = Axis(fig[1, 2],
                xlabel = "Wind speed (m/s)",
                ylabel = "Cᴰ / Cᴰ_neutral",
                title = "Stability correction factor")

band!(ax_ratio, collect(U_range), Cᴰ_sim_cold ./ Cᴰ_neutral, Cᴰ_sim_warm ./ Cᴰ_neutral,
      color=(:grey, 0.3))
lines!(ax_ratio, U_range, Cᴰ_unstable ./ Cᴰ_neutral, color=:firebrick,  linewidth=2)
lines!(ax_ratio, U_range, ones(length(U_range)),     color=:black,      linewidth=2)
lines!(ax_ratio, U_range, Cᴰ_stable   ./ Cᴰ_neutral, color=:dodgerblue, linewidth=2)

fig

# We finally assemble all of the boundary conditions,

ρu_bcs = FieldBoundaryConditions(bottom=ρu_surface_flux)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_surface_flux)
ρe_bcs = FieldBoundaryConditions(bottom=ρe_surface_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_surface_flux)

# ## Model construction
#
# We assemble the AtmosphereModel with all the components defined above.
# The model will solve the anelastic equations with the specified advection
# schemes, microphysics, and boundary conditions.

model = AtmosphereModel(grid; momentum_advection, scalar_advection, microphysics, dynamics,
                        boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs))

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
# We use Oceananigans' [`BoundaryConditionOperation`](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Models.BoundaryConditionOperation-Tuple{Field,%20Symbol,%20Oceananigans.AbstractModel})
# to extract the surface flux values from the boundary conditions. These 1D fields
# (varying only in x) represent the actual flux values applied at the
# ocean-atmosphere interface.
#
# The surface fluxes are:
#
# - ``τˣ``: momentum flux (stress), in kg m⁻¹ s⁻²
# - ``𝒬ᵀ``: sensible heat flux = cᵖᵐ Jᵀ, in W m⁻²
# - ``𝒬ᵛ``: latent heat flux = ℒˡ Jᵛ, in W m⁻²
#
# where Jᵀ is the temperature density flux and Jᵛ is the moisture density flux.

## Surface momentum flux
τˣ = BoundaryConditionOperation(ρu, :bottom, model)

## Sensible heat flux 𝒬ᵀ
ρe = static_energy_density(model)
𝒬ᵀ = BoundaryConditionOperation(ρe, :bottom, model)

## Latent heat flux: 𝒬ᵛ = ℒˡ Jᵛ (using reference θ₀ for latent heat)
ρqᵗ = model.moisture_density
ℒˡ = Breeze.Thermodynamics.liquid_latent_heat(θ₀, constants)
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
# We include both native model variables and others like, e.g., the total speed,
# ``\sqrt{u² + w²}`` and the cross-stream vorticity ``∂_z u - ∂_x w``.
# The JLD2 format provides efficient storage with full Julia type preservation.

output_filename = "prescribed_sea_surface_temperature_convection.jld2"
qᵗ = model.specific_moisture
u, v, w, = model.velocities
s = sqrt(u^2 + w^2) # speed
ξ = ∂z(u) - ∂x(w)   # cross-stream vorticity
outputs = (; s, ξ, T, θ, qˡ, qᵛ⁺, qᵗ, τˣ, 𝒬ᵀ, 𝒬ᵛ, Σ𝒬=𝒬ᵀ+𝒬ᵛ)

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

s_ts = FieldTimeSeries(output_filename, "s")
ξ_ts = FieldTimeSeries(output_filename, "ξ")
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

sn = @lift s_ts[$n]
ξn = @lift ξ_ts[$n]
θn = @lift θ_ts[$n]
qᵗn = @lift qᵗ_ts[$n]
Tn = @lift T_ts[$n]
qˡn = @lift qˡ_ts[$n]
τˣn = @lift τˣ_ts[$n]
𝒬ᵀn = @lift 𝒬ᵀ_ts[$n]
𝒬ᵛn = @lift 𝒬ᵛ_ts[$n]
Σ𝒬n = @lift Σ𝒬_ts[$n]

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
s_limits = (0, maximum(s_ts))
ξ_lim = 0.8 * maximum(abs, ξ_ts)
ξ_limits = (-ξ_lim, +ξ_lim)

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

CairoMakie.record(fig, "prescribed_sea_surface_temperature.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](prescribed_sea_surface_temperature.mp4)
