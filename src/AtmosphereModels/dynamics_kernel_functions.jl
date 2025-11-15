using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w, div_Uc
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
using Oceananigans.Utils: sum_of_velocities

@inline ∂ⱼ_𝒯₁ⱼ(i, j, k, grid, args...) = zero(grid)
@inline ∂ⱼ_𝒯₂ⱼ(i, j, k, grid, args...) = zero(grid)
@inline ∂ⱼ_𝒯₃ⱼ(i, j, k, grid, args...) = zero(grid)

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

@inline function ρ_bᶜᶜᶠ(i, j, k, grid, ρ, T, q, formulation, thermo)
    ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
    bᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, buoyancy, formulation, T, q, thermo)
    return ρᶜᶜᶠ * bᶜᶜᶠ
end

@inline function ρ_w_bᶜᶜᶠ(i, j, k, grid, w, ρ, T, q, formulation, thermo)
    ρ_b = ρ_bᶜᶜᶠ(i, j, k, grid, ρ, T, q, formulation, thermo)
    return @inbounds ρ_b * w[i, j, k]
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

    return ( - div_𝐯u(i, j, k, grid, advection, velocities, momentum.ρu)
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

    return ( - div_𝐯v(i, j, k, grid, advection, velocities, momentum.ρv)
             - y_f_cross_U(i, j, k, grid, coriolis, momentum)
             - ∂ⱼ_𝒯₂ⱼ(i, j, k, grid, reference_density, closure, closure_fields, clock, model_fields, nothing)
             + ρv_forcing(i, j, k, grid, clock, model_fields))
end

@inline function z_momentum_tendency(i, j, k, grid,
                                     reference_density,
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
                                     moisture_mass_fraction,
                                     thermo)

    return ( - div_𝐯w(i, j, k, grid, advection, velocities, momentum.ρw)
             + ρ_bᶜᶜᶠ(i, j, k, grid, reference_density, temperature, moisture_mass_fraction, formulation, thermo)
             - z_f_cross_U(i, j, k, grid, coriolis, momentum)
             - ∂ⱼ_𝒯₃ⱼ(i, j, k, grid, reference_density, closure, closure_fields, clock, model_fields, nothing)
             + ρw_forcing(i, j, k, grid, clock, model_fields))
end

@inline function scalar_tendency(i, j, k, grid,
                                 scalar,
                                 id,
                                 name,
                                 scalar_forcing,
                                 formulation,
                                 thermo,
                                 energy_density,
                                 moisture_density,
                                 advection,
                                 velocities,
                                 microphysics,
                                 microphysical_fields,
                                 closure,
                                 closure_fields,
                                 clock,
                                 model_fields)

    Uᵖ = microphysical_velocities(microphysics, name)
    Uᵗ = sum_of_velocities(velocities, Uᵖ)
    density = formulation.reference_state.density

    𝒰 = diagnose_thermodynamic_state(i, j, k, grid,
                                     formulation,
                                     microphysics,
                                     microphysical_fields,
                                     thermo,
                                     energy_density,
                                     moisture_density)

    return ( - div_Uc(i, j, k, grid, advection, Uᵗ, scalar)
             - ∇_dot_Jᶜ(i, j, k, grid, density, closure, closure_fields, id, scalar, clock, model_fields, nothing)
             + microphysical_tendency(i, j, k, grid, microphysics, name, microphysical_fields, 𝒰, thermo)
             + scalar_forcing(i, j, k, grid, clock, model_fields))
end

@inline function moist_static_energy_tendency(i, j, k, grid,
                                              id,
                                              energy,
                                              ρe_forcing,
                                              formulation,
                                              thermo,
                                              energy_density,
                                              moisture_density,
                                              advection,
                                              velocities,
                                              microphysics,
                                              microphysical_fields,
                                              closure,
                                              closure_fields,
                                              clock,
                                              model_fields,
                                              temperature,
                                              moisture_mass_fraction)

    𝒰 = diagnose_thermodynamic_state(i, j, k, grid, formulation,
                                     microphysics, microphysical_fields,
                                     thermo, energy_density, moisture_density)

    density = formulation.reference_state.density

    # Compute the buoyancy flux term, ρᵣ w b
    buoyancy_flux = ℑzᵃᵃᶜ(i, j, k, grid, ρ_w_bᶜᶜᶠ, velocities.w, density,
                          temperature, moisture_mass_fraction, formulation, thermo)

    return ( - div_Uc(i, j, k, grid, advection, velocities, energy_density)
             + buoyancy_flux
             - ∇_dot_Jᶜ(i, j, k, grid, density, closure, closure_fields, id, energy, clock, model_fields, nothing)
             + microphysical_tendency(i, j, k, grid, microphysics, Val(:ρe), microphysical_fields, 𝒰, thermo)
             + ρe_forcing(i, j, k, grid, clock, model_fields))
end
