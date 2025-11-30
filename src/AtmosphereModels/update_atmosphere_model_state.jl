using ..Thermodynamics:
    Thermodynamics,
    mixture_heat_capacity,
    mixture_gas_constant

using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!, compute_x_bcs!, compute_y_bcs!, compute_z_bcs!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Utils: launch!

import Oceananigans.TimeSteppers: update_state!, compute_flux_bc_tendencies!

const AnelasticModel = AtmosphereModel{<:AnelasticFormulation}

function update_state!(model::AnelasticModel, callbacks=[]; compute_tendencies=true)
    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model), async=true)
    compute_auxiliary_variables!(model)
    compute_tendencies && compute_tendencies!(model)
    return nothing
end

"""
    $(TYPEDSIGNATURES)

Compute auxiliary model variables:

- velocities from momentum and density (eg ``u = ρu / ρ``)

- thermodynamic variables from the prognostic thermodynamic state,
    * temperature ``T``, possibly involving saturation adjustment
    * moist static energy ``e = ρe / ρ``
    * moisture mass fraction ``qᵗ = ρqᵗ / ρ``


"""
function compute_auxiliary_variables!(model)
    grid = model.grid
    arch = grid.architecture

    launch!(arch, grid, :xyz,
            _compute_velocities!,
            model.velocities,
            grid,
            model.formulation,
            model.momentum)

    fill_halo_regions!(model.velocities)
    foreach(mask_immersed_field!, model.velocities)

    launch!(arch, grid, :xyz,
            _compute_auxiliary_thermodynamic_variables!,
            model.temperature,
            model.specific_energy,
            model.specific_moisture,
            grid,
            model.thermodynamics,
            model.formulation,
            model.microphysics,
            model.microphysical_fields,
            model.energy_density,
            model.moisture_density)

    # TODO: Can we compute the thermodynamic variable within halos as well, and avoid
    # halo filling later on?
    fill_halo_regions!(model.temperature)
    fill_halo_regions!(model.specific_energy)
    fill_halo_regions!(model.specific_moisture)
    fill_halo_regions!(model.microphysical_fields)

    # Compute diffusivities
    compute_diffusivities!(model.closure_fields, model.closure, model)

    # TODO: should we mask the auxiliary variables? They can also be masked in the kernel

    return nothing
end

@kernel function _compute_velocities!(velocities, grid, formulation, momentum)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρu = momentum.ρu[i, j, k]
        ρv = momentum.ρv[i, j, k]
        ρw = momentum.ρw[i, j, k]

        ρᶜ = formulation.reference_state.density[i, j, k]
        ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, formulation.reference_state.density)

        velocities.u[i, j, k] = ρu / ρᶜ
        velocities.v[i, j, k] = ρv / ρᶜ
        velocities.w[i, j, k] = ρw / ρᶠ
    end
end

@kernel function _compute_auxiliary_thermodynamic_variables!(temperature,
                                                             specific_energy,
                                                             specific_moisture,
                                                             grid,
                                                             thermo,
                                                             formulation,
                                                             microphysics,
                                                             microphysical_fields,
                                                             energy_density,
                                                             moisture_density)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρe = energy_density[i, j, k]
        ρqᵗ = moisture_density[i, j, k]
        ρ = formulation.reference_state.density[i, j, k]

        e = ρe / ρ
        qᵗ = ρqᵗ / ρ
        specific_energy[i, j, k] = e
        specific_moisture[i, j, k] = qᵗ
    end

    𝒰₀ = diagnose_thermodynamic_state(i, j, k, grid,
                                      formulation,
                                      microphysics,
                                      microphysical_fields,
                                      thermo,
                                      specific_energy,
                                      specific_moisture)

    # Adjust the thermodynamic state if using a microphysics scheme
    # that invokes saturation adjustment
    𝒰₁ = maybe_adjust_thermodynamic_state(𝒰₀, microphysics, microphysical_fields, qᵗ, thermo)

    update_microphysical_fields!(microphysical_fields, microphysics,
                                 i, j, k, grid,
                                 ρ, 𝒰₁, thermo)
                                 
    T = Thermodynamics.temperature(𝒰₁, thermo)
    @inbounds temperature[i, j, k] = T
end

function compute_tendencies!(model::AnelasticModel)
    grid = model.grid
    arch = grid.architecture
    Gρu = model.timestepper.Gⁿ.ρu
    Gρv = model.timestepper.Gⁿ.ρv
    Gρw = model.timestepper.Gⁿ.ρw

    model_fields = fields(model)

    #####
    ##### Momentum tendencies
    #####

    momentum_args = (
        model.formulation.reference_state.density,
        model.advection,
        model.velocities,
        model.closure,
        model.closure_fields,
        model.momentum,
        model.coriolis,
        model.clock,
        model_fields)

    u_args = tuple(momentum_args..., model.forcing.ρu)
    v_args = tuple(momentum_args..., model.forcing.ρv)

    # Extra arguments for vertical velocity are required to compute
    # buoyancy:
    w_args = tuple(momentum_args..., model.forcing.ρw,
                   model.formulation,
                   model.temperature,
                   model.specific_moisture,
                   model.microphysics,
                   model.microphysical_fields,
                   model.thermodynamics)

    launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gρu, grid, u_args)
    launch!(arch, grid, :xyz, compute_y_momentum_tendency!, Gρv, grid, v_args)
    launch!(arch, grid, :xyz, compute_z_momentum_tendency!, Gρw, grid, w_args)

    # Arguments common to energy density, moisture density, and tracer density tendencies:
    common_args = (
        model.formulation,
        model.thermodynamics,
        model.specific_energy,
        model.specific_moisture,
        model.advection,
        model.velocities,
        model.microphysics,
        model.microphysical_fields,
        model.closure,
        model.closure_fields,
        model.clock,
        model_fields)

    #####
    ##### Energy density tendency
    #####

    ρe_args = (
        Val(1),
        model.forcing.ρe,
        common_args...,
        model.temperature)

    Gρe = model.timestepper.Gⁿ.ρe
    launch!(arch, grid, :xyz, compute_moist_static_energy_tendency!, Gρe, grid, ρe_args)

    #####
    ##### Moisture density tendency
    #####

    ρq_args = (
        model.specific_moisture,
        Val(2),
        Val(:ρqᵗ),
        model.forcing.ρqᵗ,
        common_args...)

    Gρqᵗ = model.timestepper.Gⁿ.ρqᵗ
    launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρqᵗ, grid, ρq_args)

    #####
    ##### Tracer density tendencies
    #####

    for (i, name) in enumerate(keys(model.tracers))
        scalar_args = (
            model.tracers[name],
            Val(i + 2),
            Val(name),
            model.forcing[name],
            common_args...)

        Gρc = getproperty(model.timestepper.Gⁿ, name)
        launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρc, grid, scalar_args)
    end

    return nothing
end

# See dynamics_kernel_functions.jl
@kernel function compute_scalar_tendency!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = scalar_tendency(i, j, k, grid, args...)
end

@kernel function compute_moist_static_energy_tendency!(Gρe, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρe[i, j, k] = moist_static_energy_tendency(i, j, k, grid, args...)
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

"""
$(TYPEDSIGNATURES)

Apply boundary conditions by adding flux divergences to the right-hand-side.
"""
function compute_flux_bc_tendencies!(model::AtmosphereModel)

    Gⁿ = model.timestepper.Gⁿ
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
