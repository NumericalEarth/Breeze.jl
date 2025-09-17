using ..Thermodynamics:
    saturation_specific_humidity,
    mixture_heat_capacity,
    mixture_gas_constant

using Oceananigans.BoundaryConditions: fill_halo_regions!, compute_x_bcs!, compute_y_bcs!, compute_z_bcs!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: architecture

import Oceananigans.TimeSteppers: update_state!
import Oceananigans: fields, prognostic_fields


function prognostic_fields(model::RainshaftModel)
    return merge(model.density, model.temperature, model.water_vapor, model.water_condensates)
end

fields(model::RainshaftModel) = prognostic_fields(model)

function update_state!(model::RainshaftModel, callbacks=[]; compute_tendencies=true)
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

@kernel function _compute_auxiliary_thermodynamic_variables!(density,
                                                             temperature,
                                                             pressure,
                                                             thermodynamics)
    i, j, k = @index(Global, NTuple)

    𝒰 = thermodynamic_state(i, j, k, grid, thermodynamics, density, temperature)
    @inbounds pressure[i, j, k] = 𝒰.pressure
end

#=
@inline function specific_volume(state, ref, thermo)
    T = temperature(state, ref, thermo)
    Rᵐ = mixture_gas_constant(state.q, thermo)
    pᵣ = reference_pressure(state.z, ref, thermo)
    return Rᵐ * T / pᵣ
end
=#

using Oceananigans.Advection: div_Uc, with_advective_forcing
using Oceananigans.Utils: launch!

function compute_tendencies!(model::RainshaftModel)
    grid = model.grid
    arch = grid.architecture

    scalar_args = (model.advection, model.clock, fields(model))

    # Density
    Gρ = model.timestepper.Gⁿ.ρ
    ρ = model.density
    Fρ = model.forcing.ρ
    ρ_args = tuple(ρ, Fρ, scalar_args...)
    launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρ, grid, ρ_args)

    # Water vapor
    Gqv = model.timestepper.Gⁿ.qv
    qv = model.water_vapor
    Fqv = model.forcing.qv
    qv_args = tuple(qv, Fqv, scalar_args...)
    launch!(arch, grid, :xyz, compute_scalar_tendency!, Gqv, grid, qv_args)

    # Temperature
    Gthermo = model.timestepper.Gⁿ.T
    thermo_var = model.temperature
    Fthermo_var = model.forcing.T
    thermo_var_args = tuple(thermo_var, Fthermo_var, scalar_args...)
    launch!(arch, grid, :xyz, compute_scalar_tendency!, Gthermo, grid, thermo_var_args)

    # Condensates
    ρq = model.absolute_humidity
    Gρq = model.timestepper.Gⁿ.ρq
    Fρq = model.forcing.ρq
    ρq_args = tuple(ρq, Fρq, scalar_args...)
    launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρq, grid, ρq_args)

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

@kernel function compute_scalar_tendency!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = scalar_tendency(i, j, k, grid, args...)
end

@inline function scalar_tendency(i, j, k, grid,
                                 scalar,
                                 forcing,
                                 advection,
                                 velocities,
                                 clock,
                                 model_fields)

    # Figure this out                            
    total_velocities = with_advective_forcing(forcing, velocities)
    
    return ( - div_Uc(i, j, k, grid, advection, total_velocities, scalar)
             + forcing(i, j, k, grid, clock, model_fields))
end
