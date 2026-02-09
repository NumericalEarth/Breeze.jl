using ..Thermodynamics: Thermodynamics, mixture_heat_capacity, mixture_gas_constant

using Oceananigans.BoundaryConditions: fill_halo_regions!, compute_x_bcs!, compute_y_bcs!, compute_z_bcs!
using Oceananigans.Grids: Bounded, Periodic, Flat, topology, halo_size
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ

function TimeSteppers.update_state!(model::AtmosphereModel, callbacks=[]; compute_tendencies=true)
    tracer_density_to_specific!(model) # convert tracer density to specific tracer distribution

    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model), async=true)
    compute_auxiliary_variables!(model)
    update_radiation!(model.radiation, model)
    compute_forcings!(model)
    microphysics_model_update!(model.microphysics, model)
    compute_tendencies && compute_tendencies!(model)

    tracer_specific_to_density!(model) # convert specific tracer distribution to tracer density

    return nothing
end

#####
##### Compute forcing-specific quantities (e.g., horizontal averages for subsidence)
#####

"""
$(TYPEDSIGNATURES)

Compute forcing-specific quantities needed before tendency calculation.
For example, [`SubsidenceForcing`](@ref Breeze.Forcings.SubsidenceForcing)
requires horizontal averages of the fields being advected.
"""
function compute_forcings!(model)
    for forcing in model.forcing
        compute_forcing!(forcing)
    end
    return nothing
end

tracer_density_to_specific!(model) = tracer_density_to_specific!(model.tracers, dynamics_density(model.dynamics))
tracer_specific_to_density!(model) = tracer_specific_to_density!(model.tracers, dynamics_density(model.dynamics))

function tracer_density_to_specific!(tracers, density)
    # TODO: do all tracers a single kernel
    for ρc in tracers
        parent(ρc) ./= parent(density)
    end
    return nothing
end

function tracer_specific_to_density!(tracers, density)
    # TODO: do all tracers a single kernel
    for c in tracers
        parent(c) .*= parent(density)
    end
    return nothing
end

diagnostic_indices(::Bounded, N, H) = 1:N+1
diagnostic_indices(::Periodic, N, H) = -H+1:N+H
diagnostic_indices(::Flat, N, H) = 1:N

#####
##### Velocity and momentum computation
#####

"""
$(TYPEDSIGNATURES)

Compute velocities from momentum: `u = ρu / ρ` for each velocity component.
"""
function compute_velocities!(model::AtmosphereModel)
    grid = model.grid
    arch = grid.architecture

    #TODO: Better support OffsetStaticSize in KernalAbstractions
    # For now, just use :xyz instead of KernelParameters

    # TX, TY, TZ = topology(grid)
    # Nx, Ny, Nz = size(grid)
    # Hx, Hy, Hz = halo_size(grid)

    # ii = diagnostic_indices(TX(), Nx, Hx)
    # jj = diagnostic_indices(TY(), Ny, Hy)
    # kk = diagnostic_indices(TZ(), Nz, Hz)

    # kp = KernelParameters(ii, jj, kk)

    # Ensure halos are filled before velocity computation
    # (prognostic field halo fill in update_state! is async)
    density = dynamics_density(model.dynamics)
    fill_halo_regions!(density)
    fill_halo_regions!(model.momentum)

    launch!(arch, grid, :xyz,
            _compute_velocities!,
            model.velocities,
            grid,
            model.dynamics,
            model.momentum)

    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities)

    return nothing
end

function compute_momentum_tendencies!(model::AtmosphereModel, model_fields)
    grid = model.grid
    arch = grid.architecture
    Gρu = model.timestepper.Gⁿ.ρu
    Gρv = model.timestepper.Gⁿ.ρv
    Gρw = model.timestepper.Gⁿ.ρw

    momentum_args = (
        dynamics_density(model.dynamics),
        model.advection.momentum,
        model.velocities,
        model.closure,
        model.closure_fields,
        model.momentum,
        model.coriolis,
        model.clock,
        model_fields)

    u_args = tuple(momentum_args..., model.forcing.ρu, model.dynamics)
    v_args = tuple(momentum_args..., model.forcing.ρv, model.dynamics)

    # Extra arguments for vertical velocity are required to compute buoyancy
    w_args = tuple(momentum_args..., model.forcing.ρw,
                   model.dynamics,
                   model.formulation,
                   model.temperature,
                   model.specific_moisture,
                   model.microphysics,
                   model.microphysical_fields,
                   model.thermodynamic_constants)

    launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gρu, grid, u_args)
    launch!(arch, grid, :xyz, compute_y_momentum_tendency!, Gρv, grid, v_args)
    launch!(arch, grid, :xyz, compute_z_momentum_tendency!, Gρw, grid, w_args)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute auxiliary model variables:

- velocities from momentum and density (eg ``u = ρu / ρ``)

- thermodynamic variables from the prognostic thermodynamic state,
    * temperature ``T``, possibly involving saturation adjustment
    * specific thermodynamic variable (``e = ρe / ρ`` or ``θ = ρθ / ρ``)
    * moisture mass fraction ``qᵗ = ρqᵗ / ρ``
"""
function compute_auxiliary_variables!(model)
    # Compute velocities from momentum (skip for kinematic dynamics with prescribed velocities)
    compute_velocities!(model)

    # Dispatch on thermodynamic formulation type
    compute_auxiliary_thermodynamic_variables!(model)

    # Dispatch on dynamics type (computes pressure for compressible dynamics)
    compute_auxiliary_dynamics_variables!(model)

    # Compute diffusivities
    compute_diffusivities!(model.closure_fields, model.closure, model)

    # TODO: should we mask the auxiliary variables? They can also be masked in the kernel

    return nothing
end

function compute_auxiliary_thermodynamic_variables!(model::AtmosphereModel)
    grid = model.grid
    arch = grid.architecture

    launch!(arch, grid, :xyz,
            _compute_auxiliary_thermodynamic_variables!,
            model.temperature,
            model.specific_moisture,
            model.formulation,
            model.dynamics,
            grid,
            model.thermodynamic_constants,
            model.microphysics,
            model.microphysical_fields,
            model.moisture_density)

    fill_halo_regions!(model.temperature)
    fill_halo_regions!(model.specific_moisture)
    fill_halo_regions!(model.microphysical_fields)
    fill_halo_regions!(model.formulation)

    return nothing
end

@kernel function _compute_velocities!(velocities, grid, dynamics, momentum)
    i, j, k = @index(Global, NTuple)

    ρ = dynamics_density(dynamics)

    @inbounds begin
        ρu = momentum.ρu[i, j, k]
        ρv = momentum.ρv[i, j, k]
        ρw = momentum.ρw[i, j, k]

        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)

        velocities.u[i, j, k] = ρu / ρᶠᶜᶜ
        velocities.v[i, j, k] = ρv / ρᶜᶠᶜ
        velocities.w[i, j, k] = ρw / ρᶜᶜᶠ
    end
end

@kernel function _compute_auxiliary_thermodynamic_variables!(temperature,
                                                             specific_moisture,
                                                             formulation,
                                                             dynamics,
                                                             grid,
                                                             constants,
                                                             microphysics,
                                                             microphysical_fields,
                                                             moisture_density)
    i, j, k = @index(Global, NTuple)

    compute_auxiliary_thermodynamic_variables!(formulation, dynamics, i, j, k, grid)

    ρ_field = dynamics_density(dynamics)
    @inbounds begin
        ρ = ρ_field[i, j, k]
        ρqᵗ = moisture_density[i, j, k]
        qᵗ = ρqᵗ / ρ
        specific_moisture[i, j, k] = qᵗ
    end

    # Compute moisture fractions first (needed by diagnose_thermodynamic_state)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, microphysical_fields)

    𝒰₀ = diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)

    # Adjust the thermodynamic state if using a microphysics scheme
    # that invokes saturation adjustment
    𝒰₁ = maybe_adjust_thermodynamic_state(𝒰₀, microphysics, qᵗ, constants)

    update_microphysical_fields!(microphysical_fields, i, j, k, grid,
                                 microphysics, ρ, 𝒰₁, constants)

    T = Thermodynamics.temperature(𝒰₁, constants)
    @inbounds temperature[i, j, k] = T
end

function compute_tendencies!(model::AtmosphereModel)
    grid = model.grid
    arch = grid.architecture

    model_fields = fields(model)

    #####
    ##### Momentum tendencies (skip for kinematic dynamics)
    #####

    compute_momentum_tendencies!(model, model_fields)

    # Arguments common to energy density, moisture density, and tracer density tendencies:
    common_args = (
        model.dynamics,
        model.formulation,
        model.thermodynamic_constants,
        model.specific_moisture,
        model.velocities,
        model.microphysics,
        model.microphysical_fields,
        model.closure,
        model.closure_fields,
        model.clock,
        model_fields)

    #####
    ##### Thermodynamic density tendency (dispatches on thermodynamic formulation type)
    #####

    compute_thermodynamic_tendency!(model, common_args)

    #####
    ##### Moisture density tendency
    #####

    ρq_args = (
        model.specific_moisture,
        Val(2),
        Val(:ρqᵗ),
        model.forcing.ρqᵗ,
        model.advection.ρqᵗ,
        common_args...)

    Gρqᵗ = model.timestepper.Gⁿ.ρqᵗ
    launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρqᵗ, grid, ρq_args)

    #####
    ##### Tracer density tendencies
    #####

    prognostic_microphysical_fields = NamedTuple(name => model.microphysical_fields[name]
                                                 for name in prognostic_field_names(model.microphysics))

    scalars = merge(prognostic_microphysical_fields, model.tracers)
    for (i, name) in enumerate(keys(scalars))
        ρc = scalars[name]

        scalar_args = (
            ρc,
            Val(i + 2),
            Val(name),
            model.forcing[name],
            model.advection[name],
            common_args...)

        Gρc = getproperty(model.timestepper.Gⁿ, name)
        launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρc, grid, scalar_args)
    end

    #####
    ##### Dynamics-specific tendencies (e.g., density for compressible dynamics)
    #####

    compute_dynamics_tendency!(model)

    return nothing
end

# See dynamics_kernel_functions.jl
@kernel function compute_scalar_tendency!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = scalar_tendency(i, j, k, grid, args...)
end

# Note: compute_static_energy_tendency! and compute_potential_temperature_tendency!
# are now defined in their respective formulation submodules

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
function TimeSteppers.compute_flux_bc_tendencies!(model::AtmosphereModel)

    arch  = model.architecture

    # Compute boundary flux contributions
    prognostic_model_fields = prognostic_fields(model)
    args = (arch, model.clock, fields(model))
    field_indices = 1:length(prognostic_model_fields)
    Gⁿ = model.timestepper.Gⁿ

    # TODO: should we call tracer_density_to_specific!(model) here?
    foreach(q -> compute_x_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)
    foreach(q -> compute_y_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)
    foreach(q -> compute_z_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)

    return nothing
end
