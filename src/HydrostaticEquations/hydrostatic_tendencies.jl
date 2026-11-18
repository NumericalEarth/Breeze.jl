#####
##### Tendencies of the prognostic state (pˢ, u, v, ρθ) of HydrostaticDynamics
#####

#####
##### Surface pressure: column continuity
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Compute the tendency of the surface pressure of `HydrostaticDynamics` from the column integral of
the continuity equation, ``∂_t pˢ = - g Σ D``, with ``D`` the horizontal divergence of the mass
flux of each layer.
"""
function AtmosphereModels.compute_dynamics_tendency!(model::HydrostaticModel)
    grid = model.grid
    arch = grid.architecture
    Gpˢ = model.timestepper.Gⁿ.pˢ
    ρ = model.dynamics.density
    u, v, _ = model.velocities
    g = model.thermodynamic_constants.gravitational_acceleration

    launch!(arch, grid, :xy, _compute_surface_pressure_tendency!, Gpˢ, grid, ρ, u, v, g)

    return nothing
end

@inline surface_pressure_tendency(i, j, grid, ρ, u, v, g) = - g * column_mass_flux_divergence(i, j, grid, ρ, u, v)

@kernel function _compute_surface_pressure_tendency!(Gpˢ, grid, ρ, u, v, g)
    i, j = @index(Global, NTuple)
    @inbounds Gpˢ[i, j, 1] = surface_pressure_tendency(i, j, grid, ρ, u, v, g)
end

#####
##### Potential temperature density
#####

# Potential temperature on the z-face `k`, between layers `k - 1` (below) and `k` (above),
# weighted by each layer's share of the Exner interval between the layers:
#
#   θᶠ = [(Π⁻ - Πᶠ) θ⁻ + (Πᶠ - Π⁺) θ⁺] / (Π⁻ - Π⁺)
#
# This is the interface value for which the hydrostatic integral of `_compute_geopotential!`
# conserves energy. The Exner interval vanishes only on the boundary faces, through which
# there is no mass flux.
@inline function interface_potential_temperature(i, j, k, grid, θ, Π, Πᶠ)
    @inbounds begin
        θ⁻ = θ[i, j, k-1]
        θ⁺ = θ[i, j, k]
        ΔΠ⁻ = Π[i, j, k-1] - Πᶠ[i, j, k]
        ΔΠ⁺ = Πᶠ[i, j, k] - Π[i, j, k]
    end

    ΔΠ = ΔΠ⁻ + ΔΠ⁺
    boundary = ΔΠ == 0
    safe_ΔΠ = ifelse(boundary, one(ΔΠ), ΔΠ)

    return ifelse(boundary, θ⁺, (ΔΠ⁻ * θ⁻ + ΔΠ⁺ * θ⁺) / safe_ΔΠ)
end

# Flux of potential temperature through the z-face `k`, carried by the mass flux ρw of continuity
@inline vertical_potential_temperature_flux(i, j, k, grid, ρw, θ, Π, Πᶠ) =
    @inbounds Azᶜᶜᶠ(i, j, k, grid) * ρw[i, j, k] * interface_potential_temperature(i, j, k, grid, θ, Π, Πᶠ)

# Convergence of the flux of potential temperature. The horizontal fluxes are the mass fluxes
# of continuity times θ reconstructed by `advection`.
@inline potential_temperature_flux_convergence(i, j, k, grid, advection, ρ, velocities, ρw, θ, Π, Πᶠ) =
    ( - horizontal_div_ρUc(i, j, k, grid, advection, ρ, velocities, θ)
      - V⁻¹ᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, vertical_potential_temperature_flux, ρw, θ, Π, Πᶠ))

# An energy source Q (W per unit area and unit of the grid's vertical coordinate) changes ρθ at
# Q / (cᵖᵈ Π) in dry air. The radiative flux divergence is one: the difference of the net flux
# across the layer over the layer's extent Δz in the grid's coordinate. With ρ Δz = Δp / g the
# enthalpy cᵖᵈ T Δp / g of the layer then changes at the difference of the fluxes, whatever the
# coordinate labels are.
@inline function hydrostatic_potential_temperature_tendency(i, j, k, grid, advection, ρ, velocities, ρw, θ, Π, Πᶠ,
                                                            cᵖᵈ, hyperdiffusion, ρθ_forcing, ρE_forcing,
                                                            radiation_flux_divergence_field, clock, model_fields)

    Q = ρE_forcing(i, j, k, grid, clock, model_fields) +
        radiation_flux_divergence(i, j, k, grid, radiation_flux_divergence_field)

    return ( potential_temperature_flux_convergence(i, j, k, grid, advection, ρ, velocities, ρw, θ, Π, Πᶠ)
             + potential_temperature_hyperdiffusion(i, j, k, grid, hyperdiffusion, ρ, θ)
             + ρθ_forcing(i, j, k, grid, clock, model_fields)
             + @inbounds Q / (cᵖᵈ * Π[i, j, k]))
end

@kernel function _compute_hydrostatic_potential_temperature_tendency!(Gρθ, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρθ[i, j, k] = hydrostatic_potential_temperature_tendency(i, j, k, grid, args...)
end

function AtmosphereModels.compute_thermodynamic_tendency!(model::HydrostaticPotentialTemperatureModel,
                                                          common_args, tracer_transport_velocity)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics

    ρθ_args = (model.advection.ρθ,
               dynamics.density,
               model.velocities,
               dynamics.vertical_mass_flux,
               model.formulation.potential_temperature,
               dynamics.exner_function,
               dynamics.interface_exner_function,
               model.thermodynamic_constants.dry_air.heat_capacity,
               dynamics.hyperdiffusion,
               model.forcing.ρθ,
               model.forcing.ρE,
               radiation_flux_divergence(model.radiation),
               model.clock,
               Oceananigans.fields(model))

    Gρθ = model.timestepper.Gⁿ.ρθ
    launch!(arch, grid, :xyz, _compute_hydrostatic_potential_temperature_tendency!, Gρθ, grid, ρθ_args)

    return nothing
end

#####
##### Horizontal velocities: vector-invariant momentum equations
#####

# Pressure-gradient acceleration along coordinate surfaces, cᵖᵈ θ ∇Π + ∇Φ, with θ averaged to
# the velocity point. With the layer Exner function and the geopotential of the column
# diagnostics it vanishes for a resting isentropic atmosphere over any terrain. Its work
# cancels the conversion of internal and potential energy only if the horizontal flux of θ
# carries the same two-point average of θ, which is the second-order `Centered` advection of
# `ρθ`: with any other reconstruction of θ, total energy is conserved only to the order of the
# difference between the two interface values. No energy budget is tested here.
@inline x_pressure_gradient_acceleration(i, j, k, grid, θ, Π, Φ, cᵖᵈ) =
    cᵖᵈ * ℑxᶠᵃᵃ(i, j, k, grid, θ) * ∂xᶠᶜᶜ(i, j, k, grid, Π) + ∂xᶠᶜᶜ(i, j, k, grid, Φ)

@inline y_pressure_gradient_acceleration(i, j, k, grid, θ, Π, Φ, cᵖᵈ) =
    cᵖᵈ * ℑyᵃᶠᵃ(i, j, k, grid, θ) * ∂yᶜᶠᶜ(i, j, k, grid, Π) + ∂yᶜᶠᶜ(i, j, k, grid, Φ)

# Vertical advection w ∂z u, in the form that conserves kinetic energy with the layer
# continuity equation: the mass flux through each z-face times the velocity difference across
# it, averaged over the two faces of the layer and divided by the mass of the velocity cell.
@inline x_mass_flux_δz_u(i, j, k, grid, ρw, u) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, ρw) * δzᵃᵃᶠ(i, j, k, grid, u)
@inline y_mass_flux_δz_v(i, j, k, grid, ρw, v) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, ρw) * δzᵃᵃᶠ(i, j, k, grid, v)

@inline vertical_advection_u(i, j, k, grid, ρ, ρw, u) =
    ℑzᵃᵃᶜ(i, j, k, grid, x_mass_flux_δz_u, ρw, u) / (ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Vᶠᶜᶜ(i, j, k, grid))

@inline vertical_advection_v(i, j, k, grid, ρ, ρw, v) =
    ℑzᵃᵃᶜ(i, j, k, grid, y_mass_flux_δz_v, ρw, v) / (ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Vᶜᶠᶜ(i, j, k, grid))

# The forcing is an acceleration: a force per unit volume supplied under `ρu` or `ρv` arrives
# here divided by the layer mass (see `DensityTendencyForcing`).
@inline function x_velocity_tendency(i, j, k, grid, advection, coriolis, velocities, ρ, ρw, θ, Π, Φ, cᵖᵈ,
                                     hyperdiffusion, u_forcing, clock, model_fields)
    u, v = velocities.u, velocities.v

    return ( - horizontal_advection_U(i, j, k, grid, advection, u, v)
             - bernoulli_head_U(i, j, k, grid, advection, u, v)
             - vertical_advection_u(i, j, k, grid, ρ, ρw, u)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - x_pressure_gradient_acceleration(i, j, k, grid, θ, Π, Φ, cᵖᵈ)
             + x_velocity_hyperdiffusion(i, j, k, grid, hyperdiffusion, ρ, u)
             + u_forcing(i, j, k, grid, clock, model_fields))
end

@inline function y_velocity_tendency(i, j, k, grid, advection, coriolis, velocities, ρ, ρw, θ, Π, Φ, cᵖᵈ,
                                     hyperdiffusion, v_forcing, clock, model_fields)
    u, v = velocities.u, velocities.v

    return ( - horizontal_advection_V(i, j, k, grid, advection, u, v)
             - bernoulli_head_V(i, j, k, grid, advection, u, v)
             - vertical_advection_v(i, j, k, grid, ρ, ρw, v)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - y_pressure_gradient_acceleration(i, j, k, grid, θ, Π, Φ, cᵖᵈ)
             + y_velocity_hyperdiffusion(i, j, k, grid, hyperdiffusion, ρ, v)
             + v_forcing(i, j, k, grid, clock, model_fields))
end

@kernel function _compute_x_velocity_tendency!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = x_velocity_tendency(i, j, k, grid, args...)
end

@kernel function _compute_y_velocity_tendency!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = y_velocity_tendency(i, j, k, grid, args...)
end

function AtmosphereModels.compute_momentum_tendencies!(model::HydrostaticModel, model_fields)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics

    velocity_args = (model.advection.momentum,
                     model.coriolis,
                     model.velocities,
                     dynamics.density,
                     dynamics.vertical_mass_flux,
                     model.formulation.potential_temperature,
                     dynamics.exner_function,
                     dynamics.geopotential,
                     model.thermodynamic_constants.dry_air.heat_capacity,
                     dynamics.hyperdiffusion)

    u_args = tuple(velocity_args..., model.forcing.u, model.clock, model_fields)
    v_args = tuple(velocity_args..., model.forcing.v, model.clock, model_fields)

    launch!(arch, grid, :xyz, _compute_x_velocity_tendency!, model.timestepper.Gⁿ.u, grid, u_args)
    launch!(arch, grid, :xyz, _compute_y_velocity_tendency!, model.timestepper.Gⁿ.v, grid, v_args)

    return nothing
end

#####
##### Dissipation by the hyperdiffusion
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS_NO_RETURN)

Return the rates per unit volume of the grid at which the hyperdiffusion of `model`, an
`AtmosphereModel` with [`HydrostaticDynamics`](@ref), dissipates, as operations that can be
integrated or written to output:

- `θ`: the rate at which it removes the potential temperature variance `ρ θ² / 2`, at cell centers;
- `u` and `v`: the rates at which it removes the kinetic energies `ρ u² / 2` and `ρ v² / 2`, at
  the velocity points.

Each is `(1 / 64 τ₄) m (𝓛ψ)² / V`, with `m` the mass of the cell of `ψ` and `V` its volume, and is
never negative. Integrated over a single grid they are the exact rates of change of the
mass-weighted sums of `ψ² / 2` by the hyperdiffusion at fixed layer mass. Over the panels of a
`YinYangGrid` that holds only up to what the panels exchange through their interpolated halos.
"""
function hyperdiffusion_dissipation(model::HydrostaticModel)
    grid = model.grid
    hyperdiffusion = model.dynamics.hyperdiffusion
    isnothing(hyperdiffusion) && throw(ArgumentError("The HydrostaticDynamics of this model has no hyperdiffusion."))

    ρ = model.dynamics.density
    θ = model.formulation.potential_temperature
    u, v, _ = model.velocities

    return (θ = KernelFunctionOperation{Center, Center, Center}(potential_temperature_variance_dissipation, grid,
                                                                hyperdiffusion, ρ, θ),
            u = KernelFunctionOperation{Face, Center, Center}(x_kinetic_energy_dissipation, grid, hyperdiffusion, ρ, u),
            v = KernelFunctionOperation{Center, Face, Center}(y_kinetic_energy_dissipation, grid, hyperdiffusion, ρ, v))
end
