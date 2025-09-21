using ..Thermodynamics:
    saturation_specific_humidity,
    mixture_heat_capacity,
    mixture_gas_constant

using Oceananigans.BoundaryConditions: fill_halo_regions!, compute_x_bcs!, compute_y_bcs!, compute_z_bcs!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: architecture
using Oceananigans.Operators: ℑzᵃᵃᶠ, ℑzᵃᵃᶜ

import Oceananigans.TimeSteppers: update_state!, compute_flux_bc_tendencies!
import Oceananigans: fields, prognostic_fields

const AnelasticModel = AtmosphereModel{<:AnelasticFormulation}

function prognostic_fields(model::AnelasticModel)
    thermodynamic_fields = (ρe=model.energy, ρq=model.absolute_humidity)
    return merge(model.momentum, thermodynamic_fields, model.condensates, model.tracers)
end

fields(model::AnelasticModel) = prognostic_fields(model)

function update_state!(model::AnelasticModel, callbacks=[]; compute_tendencies=true)
    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model), async=true)
    compute_auxiliary_variables!(model)
    update_hydrostatic_pressure!(model)
    compute_tendencies && compute_tendencies!(model)
    return nothing
end

function compute_auxiliary_variables!(model)
    grid = model.grid
    arch = grid.architecture
    velocities = model.velocities
    formulation = model.formulation
    momentum = model.momentum

    launch!(arch, grid, :xyz, _compute_velocities!, velocities, grid, formulation, momentum)
    fill_halo_regions!(velocities)
    foreach(mask_immersed_field!, velocities)

    launch!(arch, grid, :xyz,
            _compute_auxiliary_thermodynamic_variables!,
            model.temperature,
            model.specific_humidity,
            grid,
            model.thermodynamics,
            formulation,
            model.energy,
            model.absolute_humidity)

    fill_halo_regions!(model.temperature)
    fill_halo_regions!(model.specific_humidity)

    return nothing
end

@kernel function _compute_velocities!(velocities, grid, formulation, momentum)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρu = momentum.ρu[i, j, k]
        ρv = momentum.ρv[i, j, k]
        ρw = momentum.ρw[i, j, k]

        ρᵣᵃᵃᶜ = formulation.reference_density[i, j, k]
        ρᵣᵃᵃᶠ = ℑzᵃᵃᶠ(i, j, k, grid, formulation.reference_density)
        velocities.u[i, j, k] = ρu / ρᵣᵃᵃᶜ
        velocities.v[i, j, k] = ρv / ρᵣᵃᵃᶜ
        velocities.w[i, j, k] = ρw / ρᵣᵃᵃᶠ
    end
end

@kernel function _compute_auxiliary_thermodynamic_variables!(temperature,
                                                             specific_humidity,
                                                             grid,
                                                             thermo,
                                                             formulation,
                                                             energy,
                                                             absolute_humidity)
    i, j, k = @index(Global, NTuple)

    𝒰 = thermodynamic_state(i, j, k, grid, formulation, thermo, energy, absolute_humidity)
    @inbounds specific_humidity[i, j, k] = 𝒰.specific_humidity

    # Saturation adjustment
    T = compute_temperature(𝒰, thermo)
    @inbounds temperature[i, j, k] = T
end

#=
@inline function specific_volume(state, ref, thermo)
    T = temperature(state, ref, thermo)
    Rᵐ = mixture_gas_constant(state.q, thermo)
    pᵣ = reference_pressure(state.z, ref, thermo)
    return Rᵐ * T / pᵣ
end
=#

using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w, div_Uc
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ
using Oceananigans.Utils: launch!

function compute_tendencies!(model::AnelasticModel)
    grid = model.grid
    arch = grid.architecture
    Gρu = model.timestepper.Gⁿ.ρu
    Gρv = model.timestepper.Gⁿ.ρv
    Gρw = model.timestepper.Gⁿ.ρw

    common_args = (model.advection,
                   model.velocities,
                   model.momentum,
                   model.coriolis,
                   model.clock,
                   fields(model))    

    pₕ′ = model.hydrostatic_pressure_anomaly
    ρᵣ = model.formulation.reference_density
    u_args = tuple(common_args..., model.forcing.ρu, pₕ′, ρᵣ)
    v_args = tuple(common_args..., model.forcing.ρv, pₕ′, ρᵣ)
    w_args = tuple(common_args..., model.forcing.ρw, ρᵣ,
                   model.formulation, model.temperature,
                   model.specific_humidity, model.thermodynamics)

    launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gρu, grid, u_args)
    launch!(arch, grid, :xyz, compute_y_momentum_tendency!, Gρv, grid, v_args)
    launch!(arch, grid, :xyz, compute_z_momentum_tendency!, Gρw, grid, w_args)

    scalar_args = (model.advection, model.velocities, model.clock, fields(model))
    Gρe = model.timestepper.Gⁿ.ρe
    ρe = model.energy
    Fρe = model.forcing.ρe
    ρe_args = tuple(ρe, Fρe, scalar_args..., ρᵣ,
                    model.formulation, model.temperature,
                    model.specific_humidity, model.thermodynamics, model.condensates, model.microphysics)
    launch!(arch, grid, :xyz, compute_energy_tendency!, Gρe, grid, ρe_args)

    ρq = model.absolute_humidity
    Gρq = model.timestepper.Gⁿ.ρq
    Fρq = model.forcing.ρq
    ρq_args = tuple(ρq, Fρq, scalar_args...)
    launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρq, grid, ρq_args)

    return nothing
end

hydrostatic_pressure_gradient_x(i, j, k, grid, pₕ′) = ∂xᶠᶜᶜ(i, j, k, grid, pₕ′)
hydrostatic_pressure_gradient_y(i, j, k, grid, pₕ′) = ∂yᶜᶠᶜ(i, j, k, grid, pₕ′)

@kernel function compute_scalar_tendency!(Gρc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρc[i, j, k] = scalar_tendency(i, j, k, grid, args...)
end

@kernel function compute_energy_tendency!(Gρe, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρe[i, j, k] = energy_tendency(i, j, k, grid, args...)
end


@kernel function compute_x_momentum_tendency!(Gρu, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρu[i, j, k] = x_momentum_tendency(i, j, k, grid, args...)
end

@kernel function compute_y_momentum_tendency!(Gρv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρv[i, j, k] = y_momentum_tendency(i, j, k, grid, args...)
end

@kernel function compute_z_momentum_tendency!(Gρw, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρw[i, j, k] = z_momentum_tendency(i, j, k, grid, args...)
end

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

    # Note: independent of x
    ρᵣ = @inbounds reference_density[i, j, k]    

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

    # Note: independent of y
    ρᵣ = @inbounds reference_density[i, j, k]    

    return ( - div_𝐯v(i, j, k, grid, advection, velocities, momentum.ρv)
             - y_f_cross_U(i, j, k, grid, coriolis, momentum)
             # - hydrostatic_pressure_gradient_y(i, j, k, grid, hydrostatic_pressure_anomaly)
             + forcing(i, j, k, grid, clock, model_fields))
end

@inline function ρᵣbᶜᶜᶠ(i, j, k, grid, ρᵣ, T, q, formulation, thermo)

    ρᵣᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)
    bᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, buoyancy, formulation, T, q, thermo)

    return ρᵣᶜᶜᶠ * bᶜᶜᶠ
end

@inline function ρᵣwbᶜᶜᶠ(i, j, k, grid, w, ρᵣ, T, q, formulation, thermo)
    ρᵣb = ρᵣbᶜᶜᶠ(i, j, k, grid, ρᵣ, T, q, formulation, thermo)
    return @inbounds ρᵣb * w[i, j, k]
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
             + ρᵣbᶜᶜᶠ(i, j, k, grid, reference_density, temperature, specific_humidity, formulation, thermo)
             + forcing(i, j, k, grid, clock, model_fields)
            )
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

@inline function energy_tendency(i, j, k, grid,
                                 energy,
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


    ρᵣwbᶜᶜᶜ = ℑzᵃᵃᶜ(i, j, k, grid, ρᵣwbᶜᶜᶠ, velocities.w, reference_density,
                    temperature, specific_humidity, formulation, thermo)

    return ( - div_Uc(i, j, k, grid, advection, velocities, energy)
             + ρᵣwbᶜᶜᶜ
             # + microphysical_energy_tendency(i, j, k, grid, formulation, microphysics, condensates)
             + forcing(i, j, k, grid, clock, model_fields))
end
                                        
""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function compute_flux_bc_tendencies!(model::AtmosphereModel)
    
    Gⁿ    = model.timestepper.Gⁿ
    arch  = model.architecture
    clock = model.clock

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
