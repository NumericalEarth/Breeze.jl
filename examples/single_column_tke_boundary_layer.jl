# # Single-column boundary layer with a prognostic-TKE closure
#
# A conventionally neutral boundary layer in a single column: a geostrophic wind above, a surface
# drag below, and a capping lapse rate that stops the layer from growing forever. Nothing is
# resolved — there is one column, so every eddy is parameterized — which makes this the cleanest
# place to look at what [`TKEBasedTurbulenceClosure`](@ref) actually does.
#
# The closure carries one prognostic equation for the subgrid turbulent kinetic energy ``e``,
#
# ```math
# ∂e/∂t = P + B - ε + \text{transport}, \qquad P = ν S², \qquad B = -K N²,
# ```
#
# and closes the eddy viscosity on it, ``ν = Cᴷ ℓ \sqrt{e}``, with ``ℓ`` blended harmonically from
# a distance-to-the-surface branch, a turbulence branch and a buoyancy branch. We run the same
# column three ways — the default MYNN coefficients, the MY82 set, and a deliberately inconsistent
# pair — and look at the wind, the turbulence and the mixing length.

using Breeze
using Oceananigans
using Oceananigans.Units
using CairoMakie

# ## The column, as a function
#
# Everything that varies between runs is a keyword argument, so the same generator serves the
# closure comparison below and a grid-sensitivity sweep. The vertical grid is built from a first
# spacing and a stretching ratio: `Δz` ramps linearly from `Δz₁` at the surface to
# `stretching * Δz₁` at the model top.

function single_column_simulation(; closure = TKEBasedTurbulenceClosure(),
                                    Δz₁ = 20,
                                    stretching = 4,
                                    Lz = 2000,
                                    latitude = 45,
                                    geostrophic_wind = 10,
                                    friction_velocity = 0.3,
                                    lapse_rate = 0.003,
                                    stop_time = 8hours)

    z = PiecewiseStretchedDiscretization(z = [0, Lz], Δz = [Δz₁, stretching * Δz₁])
    grid = RectilinearGrid(size = length(z) - 1; z, topology = (Flat, Flat, Bounded))

    ## Reference state: a dry, neutral adiabatic atmosphere
    θ₀ = 300
    p₀ = 1e5
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure = p₀,
                                     potential_temperature = θ₀)
    dynamics = AnelasticDynamics(reference_state)

    ## Surface drag with a prescribed friction velocity. A surface-layer scheme would set u★ from
    ## the roughness and the surface fluxes; here we impose it so the column has one fewer moving
    ## part. The closure reads this same stress to floor the near-surface TKE.
    q₀ = zero(Breeze.Thermodynamics.MoistureMassFractions{Float64})
    ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, constants)

    ## A `(Flat, Flat, Bounded)` column has no horizontal coordinates, so a bottom boundary
    ## condition is a function of time and its field dependencies alone.
    @inline ρu_drag(t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / max(sqrt(ρu^2 + ρv^2), 1e-6)
    @inline ρv_drag(t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / max(sqrt(ρu^2 + ρv^2), 1e-6)

    drag_parameters = (; ρ₀, u★ = friction_velocity)
    ρu_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=drag_parameters)
    ρv_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=drag_parameters)
    boundary_conditions = (ρu = FieldBoundaryConditions(bottom=ρu_bc),
                           ρv = FieldBoundaryConditions(bottom=ρv_bc))

    ## A geostrophic wind along x, balanced by Coriolis
    coriolis = FPlane(; latitude)
    forcing = geostrophic_forcings(geostrophic_wind, 0)

    model = AtmosphereModel(grid; dynamics, closure, coriolis, forcing, boundary_conditions,
                            advection = nothing)

    ## A weakly stratified free atmosphere caps the growing layer
    θᵢ(z) = θ₀ + lapse_rate * z
    set!(model; θ = θᵢ, ρu = reference_state.density * geostrophic_wind)

    simulation = Simulation(model; Δt = 20, stop_time)

    return simulation
end

# ## Three coefficient pairs
#
# The closure stores two independent numbers, ``Cᴷ`` and ``Cμ``. In a neutral constant-flux layer
# the logarithmic wind profile constrains only their combination
# ``Cˢ = Cᴷ/(Cμ)^{1/4}``, which must equal one; the published Mellor–Yamada sets all satisfy it.
# The third case below deliberately does not, and that shows up directly in the wind profile.

closures = ("MYNN (default)"    => TKEBasedTurbulenceClosure(),
            "MY82"              => MY82Coefficients(),
            "off the locus"     => TKEBasedTurbulenceClosure(Cᴷ = 0.30, Cμ = 0.0578))

for (name, closure) in closures
    @info name * ":  Cˢ = " * string(round(stress_coefficient(closure), digits=3)) *
          ",  e/u★² = " * string(round(surface_tke_coefficient(closure), digits=2))
end

# Run them.

simulations = [name => single_column_simulation(; closure) for (name, closure) in closures]

for (name, simulation) in simulations
    run!(simulation)
end

# ## Results
#
# The wind speed, the turbulent kinetic energy, the eddy viscosity and the mixing length. Note
# that the off-locus case mixes differently near the surface even though it carries the same
# turbulence level ``e/u_\star² = (Cμ)^{-1/2}`` — that is the point of storing ``Cμ`` separately.

set_theme!(fontsize=14, linewidth=2.5)

fig = Figure(size=(1200, 450))

ax_U = Axis(fig[1, 1]; xlabel="Wind speed (m s⁻¹)", ylabel="z (m)")
ax_e = Axis(fig[1, 2]; xlabel="TKE (m² s⁻²)")
ax_ν = Axis(fig[1, 3]; xlabel="Eddy viscosity (m² s⁻¹)")
ax_ℓ = Axis(fig[1, 4]; xlabel="Mixing length (m)")

[hideydecorations!(ax, grid=false) for ax in (ax_e, ax_ν, ax_ℓ)]

colors = (:black, :dodgerblue, :orangered)

for ((name, simulation), color) in zip(simulations, colors)
    model = simulation.model
    u, v, w = model.velocities
    U = sqrt(u^2 + v^2)

    lines!(ax_U, U; color, label=name)
    lines!(ax_e, model.closure_fields.e; color)
    lines!(ax_ν, model.closure_fields.νₑ; color)
    lines!(ax_ℓ, model.closure_fields.ℓ; color)
end

axislegend(ax_U; position=:rb, framevisible=false)

fig

# ## Grid sensitivity
#
# The same generator, driven over the first grid spacing. A closure whose coefficients sit on the
# log-law locus should place the wind profile in nearly the same place as the near-surface grid is
# refined; drift here is the signature of a surface-layer inconsistency.

Δz₁s = (10, 20, 40)
refinement_simulations = [Δz₁ => single_column_simulation(; Δz₁, stop_time=4hours) for Δz₁ in Δz₁s]

for (Δz₁, simulation) in refinement_simulations
    run!(simulation)
end

fig_grid = Figure(size=(700, 450))
ax_grid = Axis(fig_grid[1, 1]; xlabel="Wind speed (m s⁻¹)", ylabel="z (m)",
               yscale=log10, limits=(nothing, (5, 2000)))

for ((Δz₁, simulation), color) in zip(refinement_simulations, colors)
    u, v, w = simulation.model.velocities
    lines!(ax_grid, sqrt(u^2 + v^2); color, label="Δz₁ = $(Δz₁) m")
end

axislegend(ax_grid; position=:rb, framevisible=false)

fig_grid

save("single_column_tke_boundary_layer.png", fig) #src
nothing #hide
