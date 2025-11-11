using ..Thermodynamics:
    Thermodynamics,
    total_moisture_mass_fraction,
    mixture_heat_capacity,
    mixture_gas_constant

using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!, compute_x_bcs!, compute_y_bcs!, compute_z_bcs!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Utils: launch!

import Oceananigans: fields, prognostic_fields
import Oceananigans.TimeSteppers: update_state!, compute_flux_bc_tendencies!

const AnelasticModel = AtmosphereModel{<:AnelasticFormulation}

function prognostic_fields(model::AnelasticModel)
    thermodynamic_fields = (ρe=model.energy_density, ρqᵗ=model.moisture_density)
    μphys = model.microphysics
    μfields = model.microphysical_fields
    prognostic_microphysical_fields = NamedTuple(μfields[name] for name in prognostic_field_names(μphys))
    return merge(model.momentum, thermodynamic_fields, prognostic_microphysical_fields, model.tracers)
end

fields(model::AnelasticModel) = prognostic_fields(model)

function update_state!(model::AnelasticModel, callbacks=[]; compute_tendencies=true)
    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model), async=true)
    compute_auxiliary_variables!(model)
    # update_hydrostatic_pressure!(model)
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
            model.moist_static_energy,
            model.moisture_mass_fraction,
            grid,
            model.thermodynamics,
            formulation,
            model.microphysics,
            model.microphysical_fields,
            model.energy_density,
            model.moisture_density)

    fill_halo_regions!(model.temperature)
    fill_halo_regions!(model.moisture_mass_fraction)

    return nothing
end

@kernel function _compute_velocities!(velocities, grid, formulation, momentum)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρu = momentum.ρu[i, j, k]
        ρv = momentum.ρv[i, j, k]
        ρw = momentum.ρw[i, j, k]

        ρᵣᵃᵃᶜ = formulation.reference_state.density[i, j, k]
        ρᵣᵃᵃᶠ = ℑzᵃᵃᶠ(i, j, k, grid, formulation.reference_state.density)
        velocities.u[i, j, k] = ρu / ρᵣᵃᵃᶜ
        velocities.v[i, j, k] = ρv / ρᵣᵃᵃᶜ
        velocities.w[i, j, k] = ρw / ρᵣᵃᵃᶠ
    end
end

@kernel function _compute_auxiliary_thermodynamic_variables!(temperature,
                                                             moist_static_energy,
                                                             moisture_mass_fraction,
                                                             grid,
                                                             thermo,
                                                             formulation,
                                                             microphysics,
                                                             microphysical_fields,
                                                             energy_density,
                                                             moisture_density)
    i, j, k = @index(Global, NTuple)

    𝒰₀ = diagnose_thermodynamic_state(i, j, k, grid, formulation, thermo, energy_density, moisture_density)
    𝒰₁ = compute_thermodynamic_state(𝒰₀, microphysics, thermo)
    update_microphysical_fields!(microphysical_fields, microphysics, i, j, k, grid, 𝒰₁, thermo)

    @inbounds begin
        @inbounds temperature[i, j, k] = Thermodynamics.temperature(𝒰₁, thermo)
        moisture_mass_fraction[i, j, k] = total_moisture_mass_fraction(𝒰₁)
        ρe = energy_density[i, j, k]
        ρᵣ = formulation.reference_state.density[i, j, k]
        moist_static_energy[i, j, k] = ρe / ρᵣ
    end
end

function compute_tendencies!(model::AnelasticModel)
    grid = model.grid
    arch = grid.architecture
    Gρu = model.timestepper.Gⁿ.ρu
    Gρv = model.timestepper.Gⁿ.ρv
    Gρw = model.timestepper.Gⁿ.ρw

    model_fields = merge(fields(model), model.velocities, model.microphysical_fields,
                         (e = model.moist_static_energy, qᵗ = model.moisture_mass_fraction)) 

    common_args = (model.advection,
                   model.velocities,
                   model.closure,
                   model.diffusivity_fields,
                   model.momentum,
                   model.coriolis,
                   model.clock,
                   model_fields)

    pₕ′ = model.hydrostatic_pressure_anomaly
    ρᵣ = model.formulation.reference_state.density
    u_args = tuple(common_args..., model.forcing.ρu, pₕ′, ρᵣ)
    v_args = tuple(common_args..., model.forcing.ρv, pₕ′, ρᵣ)
    w_args = tuple(common_args..., model.forcing.ρw, ρᵣ,
                   model.formulation, model.temperature,
                   model.moisture_mass_fraction, model.thermodynamics)

    launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gρu, grid, u_args)
    launch!(arch, grid, :xyz, compute_y_momentum_tendency!, Gρv, grid, v_args)
    launch!(arch, grid, :xyz, compute_z_momentum_tendency!, Gρw, grid, w_args)

    scalar_args = (ρᵣ, model.advection, model.velocities, model.closure, model.diffusivity_fields, model.clock, model_fields)
    Gρe = model.timestepper.Gⁿ.ρe
    ρe = model.energy_density
    e = model.moist_static_energy
    Fρe = model.forcing.ρe
    ρe_args = tuple(ρe, Val(1), e, Fρe, scalar_args...,
                    model.formulation, model.temperature,
                    model.moisture_mass_fraction, model.thermodynamics, model.microphysical_fields, model.microphysics)
    launch!(arch, grid, :xyz, compute_moist_static_energy_tendency!, Gρe, grid, ρe_args)

    ρqᵗ = model.moisture_density
    Gρqᵗ = model.timestepper.Gⁿ.ρqᵗ
    Fρqᵗ = model.forcing.ρqᵗ
    ρq_args = tuple(ρqᵗ, Val(2), Fρqᵗ, scalar_args...)
    launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρqᵗ, grid, ρq_args)

    # Generic tracer tendencies (if any)
    for (i, name) in enumerate(keys(model.tracers))
        id = Val(i + 2)
        c = getproperty(model.tracers, name)
        Gc = getproperty(model.timestepper.Gⁿ, name)
        Fc = getproperty(model.forcing, name)
        args = tuple(c, id, Fc, scalar_args...)
        launch!(arch, grid, :xyz, compute_scalar_tendency!, Gc, grid, args)
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
