using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w,
                              U_dot_∇u_metric, U_dot_∇v_metric, U_dot_∇w_metric
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U
using Oceananigans.Utils: sum_of_velocities

# Fallback kernel functions
@inline ∂ⱼ_𝒯₁ⱼ(i, j, k, grid, args...) = zero(grid)
@inline ∂ⱼ_𝒯₂ⱼ(i, j, k, grid, args...) = zero(grid)
@inline ∂ⱼ_𝒯₃ⱼ(i, j, k, grid, args...) = zero(grid)
@inline div_ρUc(i, j, k, grid, args...) = zero(grid)
@inline c_div_ρU(i, j, k, grid, args...) = zero(grid)

# Split-explicit substepping zeros the vertical PGF and buoyancy from the explicit
# tendency; the fast substep loop re-applies them from the linearized perturbation
# formulas. Other time discretizations (e.g. ExplicitTimeStepping) fall through to
# the full conservation-form PGF and buoyancy.
@inline explicit_z_pressure_gradient(i, j, k, grid, dynamics) = z_pressure_gradient(i, j, k, grid, dynamics)
@inline explicit_buoyancy_forceᶜᶜᶠ(i, j, k, grid, args...) = buoyancy_forceᶜᶜᶠ(i, j, k, grid, args...)

"""
    ∇_dot_Jᶜ(i, j, k, grid, ρ, closure::AbstractTurbulenceClosure, closure_fields,
             id, c, clock, model_fields, buoyancy)

Return the discrete divergence of the dynamic scalar flux `Jᶜ = ρ jᶜ`,
where `jᶜ` is the "kinematic scalar flux", using area-weighted differences divided by cell volume.
Similar to Oceananigans' `∇_dot_qᶜ` signature with the additional density factor `ρ`, where in
Oceananigans `qᶜ` is the kinematic tracer flux.
"""
@inline ∇_dot_Jᶜ(i, j, k, grid, args...) = zero(grid)

#####
##### Buoyancy force interpolation and products
#####

"""
$(TYPEDSIGNATURES)

Interpolate buoyancy force to z-face location.
"""
@inline buoyancy_forceᶜᶜᶠ(i, j, k, grid, args...) = ℑzᵃᵃᶠ(i, j, k, grid, buoyancy_forceᶜᶜᶜ, args...)

"""
$(TYPEDSIGNATURES)

Compute the product of vertical velocity and buoyancy force at z-face location.
Used for the buoyancy flux term in the energy equation.
"""
@inline function w_buoyancy_forceᶜᶜᶠ(i, j, k, grid, w, args...)
    ρ_b = buoyancy_forceᶜᶜᶠ(i, j, k, grid, args...)
    return @inbounds ρ_b * w[i, j, k]
end

# Default: flux-form `∇·(ρ𝐯⊗u)` plus the explicit curvilinear metric correction.
@inline x_momentum_flux_divergence(i, j, k, grid, advection, momentum, velocities, dynamics) =
    div_𝐯u(i, j, k, grid, advection, momentum, velocities.u) +
    U_dot_∇u_metric(i, j, k, grid, advection, momentum, velocities)
@inline y_momentum_flux_divergence(i, j, k, grid, advection, momentum, velocities, dynamics) =
    div_𝐯v(i, j, k, grid, advection, momentum, velocities.v) +
    U_dot_∇v_metric(i, j, k, grid, advection, momentum, velocities)
@inline z_momentum_flux_divergence(i, j, k, grid, advection, momentum, velocities, dynamics) =
    div_𝐯w(i, j, k, grid, advection, momentum, velocities.w) +
    U_dot_∇w_metric(i, j, k, grid, advection, momentum, velocities)

@inline function x_momentum_tendency(i, j, k, grid,
                                     reference_density,
                                     advection,
                                     velocities,
                                     closure,
                                     closure_fields,
                                     momentum,
                                     coriolis,
                                     clock,
                                     model_fields,
                                     ρu_forcing,
                                     dynamics)

    return ( - x_momentum_flux_divergence(i, j, k, grid, advection, momentum, velocities, dynamics)
             - x_pressure_gradient(i, j, k, grid, dynamics)
             - x_f_cross_U(i, j, k, grid, coriolis, momentum)
             - ∂ⱼ_𝒯₁ⱼ(i, j, k, grid, reference_density, closure, closure_fields, clock, model_fields, nothing)
             + ρu_forcing(i, j, k, grid, clock, model_fields))
end

@inline function y_momentum_tendency(i, j, k, grid,
                                     reference_density,
                                     advection,
                                     velocities,
                                     closure,
                                     closure_fields,
                                     momentum,
                                     coriolis,
                                     clock,
                                     model_fields,
                                     ρv_forcing,
                                     dynamics)

    return ( - y_momentum_flux_divergence(i, j, k, grid, advection, momentum, velocities, dynamics)
             - y_pressure_gradient(i, j, k, grid, dynamics)
             - y_f_cross_U(i, j, k, grid, coriolis, momentum)
             - ∂ⱼ_𝒯₂ⱼ(i, j, k, grid, reference_density, closure, closure_fields, clock, model_fields, nothing)
             + ρv_forcing(i, j, k, grid, clock, model_fields))
end

@inline function z_momentum_tendency(i, j, k, grid,
                                     density,
                                     advection,
                                     velocities,
                                     closure,
                                     closure_fields,
                                     momentum,
                                     coriolis,
                                     clock,
                                     model_fields,
                                     ρw_forcing,
                                     dynamics,
                                     formulation,
                                     temperature,
                                     specific_prognostic_moisture,
                                     microphysics,
                                     microphysical_fields,
                                     constants)

    return ( - z_momentum_flux_divergence(i, j, k, grid, advection, momentum, velocities, dynamics)
             - explicit_z_pressure_gradient(i, j, k, grid, dynamics)
             + explicit_buoyancy_forceᶜᶜᶠ(i, j, k, grid, dynamics, temperature,
                                           specific_prognostic_moisture, microphysics, microphysical_fields, constants)
             - z_f_cross_U(i, j, k, grid, coriolis, momentum)
             - ∂ⱼ_𝒯₃ⱼ(i, j, k, grid, density, closure, closure_fields, clock, model_fields, nothing)
             + ρw_forcing(i, j, k, grid, clock, model_fields))
end

@inline function scalar_tendency(i, j, k, grid,
                                 c,
                                 id,
                                 name,
                                 c_forcing,
                                 advection,
                                 dynamics,
                                 formulation,
                                 constants,
                                 specific_prognostic_moisture,
                                 velocities,
                                 microphysics,
                                 microphysical_fields,
                                 closure,
                                 closure_fields,
                                 clock,
                                 model_fields)

    ρ_field = total_density(dynamics)  # total ρ: water/tracer mass-fraction advection & diffusion
    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants, microphysics, microphysical_fields)

    # The sum of `scalar_advective_tendency` and `nonadvective_scalar_tendency` below, in this
    # association order; the terms are spelled out rather than summed from the two parts so the
    # round-off of every path that builds the full tendency at once is unchanged.
    return ( scalar_advective_tendency(i, j, k, grid, c, name, advection, dynamics, velocities, microphysics, microphysical_fields)
             + c_div_ρU(i, j, k, grid, dynamics, velocities, c) # for PrescribedDynamics
             - ∇_dot_Jᶜ(i, j, k, grid, ρ_field, closure, closure_fields, id, c, clock, model_fields, closure_buoyancy)
             + c_forcing(i, j, k, grid, clock, model_fields))
end

"""
$(TYPEDSIGNATURES)

The advective part of [`scalar_tendency`](@ref): the flux divergence ``-∇·(ρ 𝐔ᵗ c)`` of the specific
scalar `c`, weighted by the total density, by the transport velocity plus the species' sedimentation
velocity (`microphysical_velocities`), ``𝐔ᵗ = 𝐔 + 𝐔ᵖ``.
"""
@inline function scalar_advective_tendency(i, j, k, grid, c, name, advection, dynamics, velocities,
                                           microphysics, microphysical_fields)
    Uᵖ = microphysical_velocities(microphysics, microphysical_fields, name)
    Uᵗ = sum_of_velocities(velocities, Uᵖ)
    return - div_ρUc(i, j, k, grid, advection, total_density(dynamics), Uᵗ, c)
end

"""
$(TYPEDSIGNATURES)

The non-advective part of [`scalar_tendency`](@ref), with its signature: closure diffusion, forcing and
the flux-form correction of prescribed dynamics — everything but [`scalar_advective_tendency`](@ref).
The microphysical sources are added separately (`compute_microphysical_tendencies!`). A time stepper
that treats a scalar's advection differently from its sources builds the two parts with one evaluation
of each (`SplitScalarTendencies`) instead of recomputing the advection to separate them.
"""
@inline function nonadvective_scalar_tendency(i, j, k, grid,
                                              c,
                                              id,
                                              name,
                                              c_forcing,
                                              advection,
                                              dynamics,
                                              formulation,
                                              constants,
                                              specific_prognostic_moisture,
                                              velocities,
                                              microphysics,
                                              microphysical_fields,
                                              closure,
                                              closure_fields,
                                              clock,
                                              model_fields)

    ρ_field = total_density(dynamics)
    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants, microphysics, microphysical_fields)

    return ( c_div_ρU(i, j, k, grid, dynamics, velocities, c)
             - ∇_dot_Jᶜ(i, j, k, grid, ρ_field, closure, closure_fields, id, c, clock, model_fields, closure_buoyancy)
             + c_forcing(i, j, k, grid, clock, model_fields))
end
