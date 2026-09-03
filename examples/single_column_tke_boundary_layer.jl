# # Single-column boundary layers with a prognostic-TKE closure
#
# This example simulates three canonical atmospheric boundary layers — stable, neutral and
# convective — in a single column. A single column resolves no turbulence, so every turbulent
# flux is carried by [`TKEBasedTurbulenceClosure`](@ref): a vertical eddy-diffusivity closure with
# a prognostic equation for turbulent kinetic energy, described in [Turbulence closures](@ref).
# The three columns share a grid, a geostrophic wind, a surface drag law and an initial
# stratification. They differ only in the sign of the surface heat flux, which selects the terms
# of the turbulent kinetic energy budget at work: shear production alone in the neutral column,
# shear production against a stabilizing buoyancy flux in the stable column, and shear and
# buoyancy production together in the convective one.
#
# The example demonstrates
#
#   * How to set up a single-column `AtmosphereModel` on a `(Flat, Flat, Bounded)` grid.
#   * How to drive a boundary layer with a geostrophic wind, bulk surface drag and a surface heat flux.
#   * How to use `TKEBasedTurbulenceClosure` and inspect the diffusivities it computes.

using Breeze
using Oceananigans.Units
using CairoMakie

# ## Grid
#
# The column spans the lowest 2 km of the atmosphere with 20 m resolution. The `Flat` topologies
# in ``x`` and ``y`` leave a single column of cells.

grid = RectilinearGrid(size=100, z=(0, 2000), topology=(Flat, Flat, Bounded))

# ## Reference state
#
# The anelastic dynamics are built around a dry, adiabatic reference state with a potential
# temperature of 300 K.

θ₀ = 300 # K
p₀ = 1e5 # Pa
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
dynamics = AnelasticDynamics(reference_state)

# ## Geostrophic wind and surface drag
#
# A geostrophic wind of 10 m s⁻¹ blows along ``x``. `geostrophic_forcings` supplies the
# pressure-gradient force that holds it in balance with the Coriolis force on an ``f``-plane,
# here at 45°N.

uᵍ = 10 # m s⁻¹
coriolis = FPlane(latitude=45)
forcing = geostrophic_forcings(uᵍ, 0)

# The surface drags on the wind in the first cell through a bulk law, ``𝐉ᵘ = - ρ₀ Cᵈ |𝐮| 𝐮``.
# We take the drag coefficient from the neutral logarithmic wind profile over a roughness
# length of 10 cm, evaluated at the first cell center,

κ = 0.4  # von Kármán constant
ℓʳ = 0.1 # m, roughness length
z₁ = first(znodes(grid, Center()))
Cᵈ = (κ / log(z₁ / ℓʳ))^2

# and apply it to both momentum components,

ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᵈ))
ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᵈ))

# ## Surface heat flux
#
# The surface heat flux sets the regime: a sensible heat flux of -20 W m⁻² cools the stable
# column from below, the neutral column exchanges no heat with the surface, and 100 W m⁻² heats
# the convective column. Energy fluxes are specified as boundary conditions on the static energy
# density `ρs`, and the model converts them into fluxes of its prognostic ``ρθ`` by dividing by
# the heat capacity of the air.

𝒬 = (stable = -20, neutral = 0, convective = 100) # W m⁻²

# ## Model and simulations
#
# Every column starts from the same atmosphere, stably stratified at 3 K km⁻¹ from the ground up
# and moving with the geostrophic wind, and runs for 8 hours. The closure adds the tracer `ρe`,
# the density-weighted turbulent kinetic energy, to the model. We leave it at zero: the closure
# floors the turbulent velocity at `sqrt(minimum_tke)`, which is enough for shear production to
# spin the turbulence up within minutes. Vertical diffusion and the sinks of turbulent kinetic
# energy are treated implicitly, so the columns can take one-minute time steps.

closure = TKEBasedTurbulenceClosure()
θᵢ(z) = θ₀ + 0.003z

function boundary_layer_simulation(𝒬)
    ρs_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(𝒬))
    boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs, ρs=ρs_bcs)
    model = AtmosphereModel(grid; dynamics, closure, coriolis, forcing, boundary_conditions)
    set!(model, θ=θᵢ, u=uᵍ)
    return Simulation(model, Δt=1minute, stop_time=8hours)
end

simulations = map(boundary_layer_simulation, 𝒬)

for simulation in simulations
    run!(simulation)
end

# ## Visualization
#
# We plot the final profiles of potential temperature, wind speed, turbulent kinetic energy and
# the tracer diffusivity ``Kᶜ``, which the closure stores in `model.closure_fields`. The turbulent
# kinetic energy is the tracer `ρe` divided by the reference density.

set_theme!(fontsize=14, linewidth=2.5)

fig = Figure(size=(1000, 800))
ax_θ = Axis(fig[1, 1], xlabel="Potential temperature (K)", ylabel="z (m)")
ax_U = Axis(fig[1, 2], xlabel="Wind speed (m s⁻¹)")
ax_e = Axis(fig[2, 1], xlabel="Turbulent kinetic energy (m² s⁻²)", ylabel="z (m)")
ax_K = Axis(fig[2, 2], xlabel="Tracer diffusivity Kᶜ (m² s⁻¹)")

colors = (stable=:dodgerblue, neutral=:black, convective=:orangered)

for (name, simulation) in pairs(simulations)
    model = simulation.model
    u, v, w = model.velocities
    θ = model.formulation.potential_temperature
    U = Field(sqrt(u^2 + v^2))
    e = Field(model.tracers.ρe / reference_state.density)
    Kᶜ = model.closure_fields.Kᶜ

    color = colors[name]
    lines!(ax_θ, θ; color, label=string(name))
    lines!(ax_U, U; color)
    lines!(ax_e, e; color)
    lines!(ax_K, Kᶜ; color)
end

axislegend(ax_θ, position=:rb)

save("single_column_tke_boundary_layer.png", fig) #src
fig

# The convective column is mixed through a layer 1.5 km deep, capped by the inversion it has
# eroded into the stratification above. Its potential temperature *decreases* with height through
# the lower part of the layer: a downgradient closure needs a gradient to carry the surface heat
# flux upward. The shear-driven columns are shallower, and the stable one shallowest, because the
# surface cooling stratifies the air it is mixed into and the stratification length then cuts the
# mixing length short. In both the stable and the neutral columns the wind at the top of the mixed
# layer overshoots its geostrophic value — a low-level jet, strongest in the stable column. The
# diffusivities grow with height from the surface, where the mixing length is the height itself,
# and collapse where the turbulent kinetic energy runs out.
