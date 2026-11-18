#####
##### Bounds-preserving advection of water in moist HydrostaticDynamics
#####
#
# A bounds-preserving WENO scheme, `WENO(order=5, bounds=(0, 1))`, rescales its reconstructions with
# a cell-wise limiter that is computed from the field the scheme reconstructs, before the tendencies
# (`update_advection!`). The water of a moist hydrostatic model is transported as the mixing ratio
# per unit dry mass, r = ρc / ρ, by the dry mass fluxes of continuity (`MixingRatio` in
# `moist_hydrostatic_dynamics.jl`), and not as the mass fraction per unit total mass that the
# generic `AtmosphereModel` pairs with the scheme. The limiter must be that of what is transported:
# a limiter computed from the mass fraction would bound a reconstruction that is never used.
#
# With the limiter of r, the flux through every face is the dry mass flux of continuity times a
# reconstruction of r that keeps the cell within its bounds over a forward Euler step at Courant
# numbers up to the scheme's `maximum_courant_number`, so a strong-stability-preserving time
# stepper, which is a convex combination of such steps, keeps the water non-negative. Nothing is
# clipped: the limiter acts inside the flux, and what leaves a cell enters its neighbor.
#
# This holds for `:SSPRungeKutta3`. It does not hold for `:GravityWaveMultirateInfinitesimalStep`,
# whose scalar update is not a convex combination of forward Euler steps and carries the scalars
# with a centered flux of the perturbation mass flux.

Adapt.adapt_structure(to, r::MixingRatio) =
    MixingRatio(adapt(to, r.specific), adapt(to, r.total_density), adapt(to, r.density))

# The field each scheme of `advection` reconstructs: the mixing ratio per unit dry mass for the
# prognostic densities of water, and what any `AtmosphereModel` reconstructs for the others
function AtmosphereModels.reconstructed_fields(model::MoistHydrostaticModel, advection::NamedTuple{names}) where names
    specific = merge((; momentum = nothing), specific_prognostic_scalar_fields(model))
    water_names = (AtmosphereModels.moisture_prognostic_name(model.microphysics),
                   AtmosphereModels.prognostic_field_names(model.microphysics)...)
    ρ = model.dynamics.density
    ρᵗ = model.dynamics.moisture.total_density
    fields = map(name -> name ∈ water_names ? MixingRatio(specific[name], ρᵗ, ρ) : specific[name], names)
    return NamedTuple{names}(fields)
end

# With a flux limiter (PROVISIONAL, see `flux_limited_div_ρUc`), the fractions of the limiter are
# those of the transport that the tendencies apply: the mixing ratio per unit dry mass, carried by
# the velocities of the state on the layer mass of dry air, over the forward Euler step of the time
# stepper. A time stepper that transports the water with the mass flux of its stages refreshes them
# again for each stage (`moist_stage_scalar_transport.jl`).
function Oceananigans.Advection.update_advection!(scheme::FluxLimitedWENO, model::MoistHydrostaticModel, r::MixingRatio)
    grid = model.grid
    Δt = convert(eltype(grid), flux_limiter_timestep(model.timestepper, model.clock))
    update_bounds_preserving_limiter!(scheme, grid, r, model.velocities, Δt)
    update_flux_limiter!(scheme, grid, r, model.velocities, Δt, model.dynamics.density)
    return nothing
end
