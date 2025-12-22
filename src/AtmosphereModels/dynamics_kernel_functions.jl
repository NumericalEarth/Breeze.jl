using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U
using Oceananigans.Utils: sum_of_velocities

# Fallback kernel functions
@inline ∂ⱼ_𝒯₁ⱼ(i, j, k, grid, args...) = zero(grid)
@inline ∂ⱼ_𝒯₂ⱼ(i, j, k, grid, args...) = zero(grid)
@inline ∂ⱼ_𝒯₃ⱼ(i, j, k, grid, args...) = zero(grid)
@inline div_ρUc(i, j, k, grid, args...) = zero(grid)

"""
    ∇_dot_Jᶜ(i, j, k, grid, density, closure::AbstractTurbulenceClosure, closure_fields,
             id, c, clock, model_fields, buoyancy)

Return the discrete divergence of the dynamic scalar flux `Jᶜ = ρ jᶜ`,
where `jᶜ` is the "kinematic scalar flux", using area-weighted differences divided by cell volume.
Similar to Oceananigans' `∇_dot_qᶜ` signature with the additional density factor `ρ`, where in
Oceananigans `qᶜ` is the kinematic tracer flux.
"""
@inline ∇_dot_Jᶜ(i, j, k, grid, args...) = zero(grid)

#####
##### Some key functions
#####

@inline function ρ_bᶜᶜᶜ(i, j, k, grid,
                        dynamics::AnelasticDynamics,
                        formulation,
                        reference_density,
                        temperature,
                        specific_moisture,
                        microphysics,
                        microphysical_fields,
                        constants)

    @inbounds begin
        qᵗ = specific_moisture[i, j, k]
        pᵣ = dynamics.reference_state.pressure[i, j, k]
        T = temperature[i, j, k]
        ρᵣ = reference_density[i, j, k]
    end

    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)
    ρ = pᵣ / (Rᵐ * T)
    g = constants.gravitational_acceleration
    ρ′ = ρ - ρᵣ

    return - g * ρ′
end

@inline ρ_bᶜᶜᶠ(i, j, k, grid, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ρ_bᶜᶜᶜ, args...)

@inline function ρ_w_bᶜᶜᶠ(i, j, k, grid, w, args...)
    ρ_b = ρ_bᶜᶜᶠ(i, j, k, grid, args...)
    return @inbounds ρ_b * w[i, j, k]
end

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
                                     ρu_forcing)

    return ( - div_𝐯u(i, j, k, grid, advection, momentum, velocities.u)
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
                                     ρv_forcing)

    return ( - div_𝐯v(i, j, k, grid, advection, momentum, velocities.v)
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
                                     specific_moisture,
                                     microphysics,
                                     microphysical_fields,
                                     constants)

    return ( - div_𝐯w(i, j, k, grid, advection, momentum, velocities.w)
             + ρ_bᶜᶜᶠ(i, j, k, grid, dynamics, formulation, density, temperature,
                      specific_moisture, microphysics, microphysical_fields, constants)
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
                                 specific_moisture,
                                 velocities,
                                 microphysics,
                                 microphysical_fields,
                                 closure,
                                 closure_fields,
                                 clock,
                                 model_fields)

    Uᵖ = microphysical_velocities(microphysics, microphysical_fields, name)
    Uᵗ = sum_of_velocities(velocities, Uᵖ)
    ρ_field = dynamics_density(dynamics)
    @inbounds ρ = ρ_field[i, j, k]
    @inbounds qᵗ = specific_moisture[i, j, k]
    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants)

    # Compute moisture fractions first
    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, microphysical_fields)
    𝒰 = diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)

    return ( - div_ρUc(i, j, k, grid, advection, ρ_field, Uᵗ, c)
             - ∇_dot_Jᶜ(i, j, k, grid, ρ_field, closure, closure_fields, id, c, clock, model_fields, closure_buoyancy)
             + microphysical_tendency(i, j, k, grid, microphysics, name, ρ, microphysical_fields, 𝒰, constants)
             + c_forcing(i, j, k, grid, clock, model_fields))
end
