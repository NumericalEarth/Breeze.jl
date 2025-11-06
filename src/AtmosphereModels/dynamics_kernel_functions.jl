using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w, div_Uc
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ

#####
##### Some key functions
#####

@inline function buoyancy(i, j, k, grid, formulation, temperature, specific_humidity, thermo)
    α = specific_volume(i, j, k, grid, formulation, temperature, specific_humidity, thermo)
    αʳ = reference_specific_volume(i, j, k, grid, formulation, thermo)
    g = thermo.gravitational_acceleration
    return g * (α - αʳ) / αʳ
end

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
                                     forcing,
                                     reference_density,
                                     hydrostatic_pressure_anomaly)

    return ( - div_𝐯u(i, j, k, grid, advection, velocities, momentum.ρu)
             - x_f_cross_U(i, j, k, grid, coriolis, momentum)
             # - hydrostatic_pressure_gradient_x(i, j, k, grid, hydrostatic_pressure_anomaly)
             + forcing(i, j, k, grid, clock, model_fields))
end

@inline function y_momentum_tendency(i, j, k, grid,
                                     advection,
                                     velocities,
                                     momentum,
                                     coriolis,
                                     clock,
                                     model_fields,
                                     forcing,
                                     reference_density,
                                     hydrostatic_pressure_anomaly)

    return ( - div_𝐯v(i, j, k, grid, advection, velocities, momentum.ρv)
             - y_f_cross_U(i, j, k, grid, coriolis, momentum)
             # - hydrostatic_pressure_gradient_y(i, j, k, grid, hydrostatic_pressure_anomaly)
             + forcing(i, j, k, grid, clock, model_fields))
end

@inline function z_momentum_tendency(i, j, k, grid,
                                     advection,
                                     velocities,
                                     momentum,
                                     coriolis,
                                     clock,
                                     model_fields,
                                     forcing,
                                     reference_density,
                                     formulation,
                                     temperature,
                                     specific_humidity,
                                     thermo)

    return ( - div_𝐯w(i, j, k, grid, advection, velocities, momentum.ρw)
             - z_f_cross_U(i, j, k, grid, coriolis, momentum)
             + ρ_bᶜᶜᶠ(i, j, k, grid, reference_density, temperature, specific_humidity, formulation, thermo)
             + forcing(i, j, k, grid, clock, model_fields))
end

@inline function scalar_tendency(i, j, k, grid,
                                 scalar,
                                 forcing,
                                 advection,
                                 velocities,
                                 clock,
                                 model_fields)

    return ( - div_Uc(i, j, k, grid, advection, velocities, scalar)
             + forcing(i, j, k, grid, clock, model_fields))
end

@inline function moist_static_energy_tendency(i, j, k, grid,
                                              moist_static_energy,
                                              forcing,
                                              advection,
                                              velocities,
                                              clock,
                                              model_fields,
                                              reference_density,
                                              formulation,
                                              temperature,
                                              specific_humidity,
                                              thermo,
                                              condensates,
                                              microphysics)

    # Compute the buoyancy flux term, ρᵣ w b
    buoyancy_flux = ℑzᵃᵃᶜ(i, j, k, grid, ρ_w_bᶜᶜᶠ, velocities.w, reference_density,
                          temperature, specific_humidity, formulation, thermo)

    return ( - div_Uc(i, j, k, grid, advection, velocities, moist_static_energy)
             + buoyancy_flux
             # + microphysical_energy_tendency(i, j, k, grid, formulation, microphysics, condensates)
             + forcing(i, j, k, grid, clock, model_fields))
end

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function compute_flux_bc_tendencies!(model::AtmosphereModel)

    Gⁿ    = model.timestepper.Gⁿ
    arch  = model.architecture

    # Compute boundary flux contributions
    prognostic_model_fields = prognostic_fields(model)
    args = (arch, model.clock, fields(model))
    field_indices = 1:length(prognostic_model_fields)
    Gⁿ = model.timestepper.Gⁿ
    foreach(q -> compute_x_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)
    foreach(q -> compute_y_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)
    foreach(q -> compute_z_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)

    return nothing
end
