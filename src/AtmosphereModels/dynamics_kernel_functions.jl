using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w, div_Uc
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ

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
                                     advection,
                                     velocities,
                                     momentum,
                                     coriolis,
                                     clock,
                                     model_fields,
                                     ρu_forcing)

    return ( - div_𝐯u(i, j, k, grid, advection, velocities, momentum.ρu)
             - x_f_cross_U(i, j, k, grid, coriolis, momentum)
             + ρu_forcing(i, j, k, grid, clock, model_fields))
end

@inline function y_momentum_tendency(i, j, k, grid,
                                     advection,
                                     velocities,
                                     momentum,
                                     coriolis,
                                     clock,
                                     model_fields,
                                     ρv_forcing)

    return ( - div_𝐯v(i, j, k, grid, advection, velocities, momentum.ρv)
             - y_f_cross_U(i, j, k, grid, coriolis, momentum)
             + ρv_forcing(i, j, k, grid, clock, model_fields))
end

@inline function z_momentum_tendency(i, j, k, grid,
                                     advection,
                                     velocities,
                                     momentum,
                                     coriolis,
                                     clock,
                                     model_fields,
                                     ρw_forcing,
                                     reference_density,
                                     formulation,
                                     temperature,
                                     moisture_mass_fraction,
                                     thermo)

    return ( - div_𝐯w(i, j, k, grid, advection, velocities, momentum.ρw)
             - z_f_cross_U(i, j, k, grid, coriolis, momentum)
             + ρ_bᶜᶜᶠ(i, j, k, grid, reference_density, temperature, moisture_mass_fraction, formulation, thermo)
             + ρw_forcing(i, j, k, grid, clock, model_fields))
end

@inline function scalar_tendency(i, j, k, grid,
                                 scalar,
                                 scalar_forcing,
                                 advection,
                                 velocities,
                                 clock,
                                 model_fields)

    return ( - div_Uc(i, j, k, grid, advection, velocities, scalar)
             + scalar_forcing(i, j, k, grid, clock, model_fields))
end

@inline function moist_static_energy_tendency(i, j, k, grid,
                                              moist_static_energy,
                                              ρe_forcing,
                                              advection,
                                              velocities,
                                              clock,
                                              model_fields,
                                              reference_density,
                                              formulation,
                                              temperature,
                                              moisture_mass_fraction,
                                              thermo,
                                              microphysical_fields,
                                              microphysics)

    # Compute the buoyancy flux term, ρᵣ w b
    buoyancy_flux = ℑzᵃᵃᶜ(i, j, k, grid, ρ_w_bᶜᶜᶠ, velocities.w, reference_density,
                          temperature, moisture_mass_fraction, formulation, thermo)

    return ( - div_Uc(i, j, k, grid, advection, velocities, moist_static_energy)
             + buoyancy_flux
             # + microphysical_energy_tendency(i, j, k, grid, formulation, microphysics, microphysical_fields)
             + ρe_forcing(i, j, k, grid, clock, model_fields))
end
