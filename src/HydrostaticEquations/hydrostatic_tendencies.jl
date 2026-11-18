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

# Flux of potential temperature through the z-face `k`, carried by the mass flux ρw of continuity.
# With any scheme that is not upwind-biased, the interface value is the energy-conserving one.
@inline vertical_potential_temperature_flux(i, j, k, grid, advection, ρw, θ, Π, Πᶠ) =
    @inbounds Azᶜᶜᶠ(i, j, k, grid) * ρw[i, j, k] * interface_potential_temperature(i, j, k, grid, θ, Π, Πᶠ)

# With an upwind-biased scheme, θ is reconstructed on the z-face by the scheme, from the side
# the mass flux comes from, as it is on the x- and y-faces: the same mass flux of continuity
# then carries a reconstruction of the same kind through every face of the cell. The work of the
# pressure-gradient force no longer cancels the conversion of internal and potential energy
# exactly, and the scheme dissipates the variance of θ.
@inline function vertical_potential_temperature_flux(i, j, k, grid, advection::AbstractUpwindBiasedAdvectionScheme,
                                                     ρw, θ, Π, Πᶠ)
    Azρw = Az_qᶜᶜᶠ(i, j, k, grid, ρw)
    θᶠ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, advection, bias(Azρw), θ)
    return Azρw * θᶠ
end

# Convergence of the flux of potential temperature. The fluxes are the mass fluxes of
# continuity times θ reconstructed by `advection`.
@inline potential_temperature_flux_convergence(i, j, k, grid, advection, ρ, velocities, ρw, θ, Π, Πᶠ) =
    ( - horizontal_div_ρUc(i, j, k, grid, advection, ρ, velocities, θ)
      - V⁻¹ᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, vertical_potential_temperature_flux, advection, ρw, θ, Π, Πᶠ))

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
##### Horizontal velocities: momentum equations in vector-invariant or in flux form
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

@inline vertical_advection_u(i, j, k, grid, scheme, ρ, ρw, u) =
    ℑzᵃᵃᶜ(i, j, k, grid, x_mass_flux_δz_u, ρw, u) / (ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Vᶠᶜᶜ(i, j, k, grid))

@inline vertical_advection_v(i, j, k, grid, scheme, ρ, ρw, v) =
    ℑzᵃᵃᶜ(i, j, k, grid, y_mass_flux_δz_v, ρw, v) / (ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Vᶜᶠᶜ(i, j, k, grid))

# Vertical advection w ∂z u with an upwind-biased `scheme`: the divergence of the flux of the
# velocity reconstructed on the z-faces of the velocity cell from the side the mass flux comes
# from, less the velocity times the divergence of that mass flux, so that a velocity that is
# uniform in the column is not changed. This form dissipates kinetic energy.
@inline x_vertical_mass_flux(i, j, k, grid, ρw) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, ρw)
@inline y_vertical_mass_flux(i, j, k, grid, ρw) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, ρw)

@inline function vertical_advection_u(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, ρ, ρw, u)
    δz_ρwu = δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, scheme, ρw, u)
    δz_ρw = δzᵃᵃᶜ(i, j, k, grid, x_vertical_mass_flux, ρw)
    return @inbounds (δz_ρwu - u[i, j, k] * δz_ρw) / (ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Vᶠᶜᶜ(i, j, k, grid))
end

@inline function vertical_advection_v(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, ρ, ρw, v)
    δz_ρwv = δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, scheme, ρw, v)
    δz_ρw = δzᵃᵃᶜ(i, j, k, grid, y_vertical_mass_flux, ρw)
    return @inbounds (δz_ρwv - v[i, j, k] * δz_ρw) / (ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Vᶜᶠᶜ(i, j, k, grid))
end

# Vector-invariant advection (ζ ẑ × 𝐮 + ∇K + w ∂z 𝐮): the vorticity flux and the gradient of the
# kinetic energy of `advection`, and the vertical advection of its vertical scheme
@inline momentum_advection_u(i, j, k, grid, advection::VectorInvariant, ρ, ρw, u, v) =
    ( horizontal_advection_U(i, j, k, grid, advection, u, v)
      + bernoulli_head_U(i, j, k, grid, advection, u, v)
      + vertical_advection_u(i, j, k, grid, advection.vertical_advection_scheme, ρ, ρw, u))

@inline momentum_advection_v(i, j, k, grid, advection::VectorInvariant, ρ, ρw, u, v) =
    ( horizontal_advection_V(i, j, k, grid, advection, u, v)
      + bernoulli_head_V(i, j, k, grid, advection, u, v)
      + vertical_advection_v(i, j, k, grid, advection.vertical_advection_scheme, ρ, ρw, v))

#####
##### Flux-form momentum advection
#####
##### The mass of the cell of u is mᵘ = ℑx(ρ V), and the mass fluxes through its faces are the
##### averages F̃ of the mass fluxes F of continuity, so that ∂ₜmᵘ = - δ·F̃ exactly. With uᴿ the
##### velocity that `advection` reconstructs on each face, from the side the mass flux comes from,
#####
#####     ∂ₜu = - [δ·(F̃ uᴿ) - u δ·F̃] / mᵘ - metric / ℑx(ρ)    ⟺    ∂ₜ(mᵘ u) = - δ·(F̃ uᴿ) - metric terms ,
#####
##### which is the momentum equation in flux form for the prognostic velocity on the moving
##### layers, and leaves a uniform u unchanged. The metric terms are those of the curvature of
##### the horizontal coordinates.
#####

# Mass per unit volume of the grid carried through the x- and y-faces: the density that
# `mass_flux_x` and `mass_flux_y` give the velocities, so that Ax ρu and Ay ρv are those fluxes
struct XMomentumDensity{R, U}
    ρ :: R
    u :: U
end

struct YMomentumDensity{R, V}
    ρ :: R
    v :: V
end

Base.@propagate_inbounds Base.getindex(ρu::XMomentumDensity, i, j, k) = (ρu.ρ[i-1, j, k] + ρu.ρ[i, j, k]) / 2 * ρu.u[i, j, k]
Base.@propagate_inbounds Base.getindex(ρv::YMomentumDensity, i, j, k) = (ρv.ρ[i, j-1, k] + ρv.ρ[i, j, k]) / 2 * ρv.v[i, j, k]

# Divergence of the mass fluxes through the faces of the cells of u and of v, in kg s⁻¹
@inline x_mass_flux_divergence(i, j, k, grid, ρu, ρv, ρw) =
    ( δxᶠᵃᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Ax_qᶠᶜᶜ, ρu)
      + δyᵃᶜᵃ(i, j, k, grid, ℑxᶠᵃᵃ, Ay_qᶜᶠᶜ, ρv)
      + δzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, Az_qᶜᶜᶠ, ρw))

@inline y_mass_flux_divergence(i, j, k, grid, ρu, ρv, ρw) =
    ( δxᶜᵃᵃ(i, j, k, grid, ℑyᵃᶠᵃ, Ax_qᶠᶜᶜ, ρu)
      + δyᵃᶠᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Ay_qᶜᶠᶜ, ρv)
      + δzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, Az_qᶜᶜᶠ, ρw))

@inline function momentum_advection_u(i, j, k, grid, advection, ρ, ρw, u, v)
    ρu = XMomentumDensity(ρ, u)
    ρv = YMomentumDensity(ρ, v)
    momentum = (ρu, ρv, ρw)
    velocities = (u, v)

    δ_ρ𝐮u = Vᶠᶜᶜ(i, j, k, grid) * div_𝐯u(i, j, k, grid, advection, momentum, u)
    δ_ρ𝐮 = x_mass_flux_divergence(i, j, k, grid, ρu, ρv, ρw)
    metric = U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, momentum, velocities)

    return @inbounds ( (δ_ρ𝐮u - u[i, j, k] * δ_ρ𝐮) / ℑxᶠᵃᵃ(i, j, k, grid, cell_mass, ρ)
                       + metric / ℑxᶠᵃᵃ(i, j, k, grid, ρ))
end

@inline function momentum_advection_v(i, j, k, grid, advection, ρ, ρw, u, v)
    ρu = XMomentumDensity(ρ, u)
    ρv = YMomentumDensity(ρ, v)
    momentum = (ρu, ρv, ρw)
    velocities = (u, v)

    δ_ρ𝐮v = Vᶜᶠᶜ(i, j, k, grid) * div_𝐯v(i, j, k, grid, advection, momentum, v)
    δ_ρ𝐮 = y_mass_flux_divergence(i, j, k, grid, ρu, ρv, ρw)
    metric = U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, momentum, velocities)

    return @inbounds ( (δ_ρ𝐮v - v[i, j, k] * δ_ρ𝐮) / ℑyᵃᶠᵃ(i, j, k, grid, cell_mass, ρ)
                       + metric / ℑyᵃᶠᵃ(i, j, k, grid, ρ))
end

# The forcing is an acceleration: a force per unit volume supplied under `ρu` or `ρv` arrives
# here divided by the layer mass (see `DensityTendencyForcing`).
@inline function x_velocity_tendency(i, j, k, grid, advection, coriolis, velocities, ρ, ρw, θ, Π, Φ, cᵖᵈ,
                                     hyperdiffusion, u_forcing, clock, model_fields)
    u, v = velocities.u, velocities.v

    return ( - momentum_advection_u(i, j, k, grid, advection, ρ, ρw, u, v)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - x_pressure_gradient_acceleration(i, j, k, grid, θ, Π, Φ, cᵖᵈ)
             + x_velocity_hyperdiffusion(i, j, k, grid, hyperdiffusion, ρ, u)
             + u_forcing(i, j, k, grid, clock, model_fields))
end

@inline function y_velocity_tendency(i, j, k, grid, advection, coriolis, velocities, ρ, ρw, θ, Π, Φ, cᵖᵈ,
                                     hyperdiffusion, v_forcing, clock, model_fields)
    u, v = velocities.u, velocities.v

    return ( - momentum_advection_v(i, j, k, grid, advection, ρ, ρw, u, v)
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
