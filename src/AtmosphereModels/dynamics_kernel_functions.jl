using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
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

@inline @inbounds function ρ_bᶜᶜᶜ(i, j, k, grid,
                                  formulation::AnelasticFormulation,
                                  reference_density,
                                  temperature,
                                  specific_moisture,
                                  microphysics,
                                  microphysical_fields,
                                  thermo)

    qᵗ = specific_moisture[i, j, k]
    pᵣ = formulation.reference_state.pressure[i, j, k]
    T = temperature[i, j, k]
    ρᵣ = reference_density[i, j, k]

    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, thermo)
    ρ = pᵣ / (Rᵐ * T)
    g = thermo.gravitational_acceleration

    return - g * (ρ - ρᵣ)
end

@inline ρ_bᶜᶜᶠ(i, j, k, grid, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ρ_bᶜᶜᶜ, args...)   

@inline @inbounds function ρ_w_bᶜᶜᶠ(i, j, k, grid, w, args...)
    ρ_b = ρ_bᶜᶜᶠ(i, j, k, grid, args...)
    return ρ_b * w[i, j, k]
end

# Note: these are unused currently
hydrostatic_pressure_gradient_x(i, j, k, grid, pₕ′) = ∂xᶠᶜᶜ(i, j, k, grid, pₕ′)
hydrostatic_pressure_gradient_y(i, j, k, grid, pₕ′) = ∂yᶜᶠᶜ(i, j, k, grid, pₕ′)

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
                                     formulation,
                                     temperature,
                                     specific_moisture,
                                     microphysics,
                                     microphysical_fields,
                                     thermo)

    return ( - div_𝐯w(i, j, k, grid, advection, momentum, velocities.w)
             + ρ_bᶜᶜᶠ(i, j, k, grid, formulation, density, temperature,
                      specific_moisture, microphysics, microphysical_fields, thermo)
             - z_f_cross_U(i, j, k, grid, coriolis, momentum)
             - ∂ⱼ_𝒯₃ⱼ(i, j, k, grid, density, closure, closure_fields, clock, model_fields, nothing)
             + ρw_forcing(i, j, k, grid, clock, model_fields))
end

@inline function scalar_tendency(i, j, k, grid,
                                 c,
                                 id,
                                 name,
                                 c_forcing,
                                 formulation,
                                 thermo,
                                 specific_energy,
                                 specific_moisture,
                                 advection,
                                 velocities,
                                 microphysics,
                                 microphysical_fields,
                                 closure,
                                 closure_fields,
                                 clock,
                                 model_fields)

    # TODO fix this
    Uᵖ = microphysical_velocities(microphysics, name)
    Uᵗ = sum_of_velocities(velocities, Uᵖ)
    ρ = formulation.reference_state.density
    diffusive_flux_buoyancy = AtmosphereModelBuoyancy(formulation, thermo)

    𝒰 = diagnose_thermodynamic_state(i, j, k, grid,
                                     formulation,
                                     microphysics,
                                     microphysical_fields,
                                     thermo,
                                     specific_energy,
                                     specific_moisture)

    return ( - div_ρUc(i, j, k, grid, advection, ρ, Uᵗ, c)
             - ∇_dot_Jᶜ(i, j, k, grid, ρ, closure, closure_fields, id, c, clock, model_fields, diffusive_flux_buoyancy)
             + microphysical_tendency(i, j, k, grid, microphysics, name, microphysical_fields, 𝒰, thermo)
             + c_forcing(i, j, k, grid, clock, model_fields))
end

@inline function moist_static_energy_tendency(i, j, k, grid,
                                              id,
                                              ρe_forcing,
                                              formulation,
                                              thermo,
                                              specific_energy,
                                              specific_moisture,
                                              advection,
                                              velocities,
                                              microphysics,
                                              microphysical_fields,
                                              closure,
                                              closure_fields,
                                              clock,
                                              model_fields,
                                              temperature)

    𝒰 = diagnose_thermodynamic_state(i, j, k, grid,
                                     formulation,
                                     microphysics,
                                     microphysical_fields,
                                     thermo,
                                     specific_energy,
                                     specific_moisture)

    ρ = formulation.reference_state.density

    # Compute the buoyancy flux term, ρᵣ w b
    buoyancy_flux = ℑzᵃᵃᶜ(i, j, k, grid, ρ_w_bᶜᶜᶠ,
                          velocities.w, formulation, ρ, temperature, specific_moisture,
                          microphysics, microphysical_fields, thermo)

    closure_buoyancy = AtmosphereModelBuoyancy(formulation, thermo)

    return ( - div_ρUc(i, j, k, grid, advection, ρ, velocities, specific_energy)
             + buoyancy_flux
             - ∇_dot_Jᶜ(i, j, k, grid, ρ, closure, closure_fields, id, specific_energy, clock, model_fields, closure_buoyancy)
             + microphysical_tendency(i, j, k, grid, microphysics, Val(:ρe), microphysical_fields, 𝒰, thermo)
             + ρe_forcing(i, j, k, grid, clock, model_fields))
end
