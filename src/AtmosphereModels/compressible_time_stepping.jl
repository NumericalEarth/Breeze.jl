#####
##### Explicit time stepping for CompressibleDynamics
#####
##### For compressible dynamics, there is no pressure correction step.
##### Instead, pressure is computed diagnostically from the equation of state.
#####

using Oceananigans: prognostic_fields, fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

using Breeze.Thermodynamics: mixture_gas_constant, mixture_heat_capacity, dry_air_gas_constant

"""
$(TYPEDSIGNATURES)

For compressible dynamics, there is no pressure correction step.
This function is a no-op.
"""
function TimeSteppers.compute_pressure_correction!(model::CompressibleModel, Δt)
    # No pressure correction for compressible dynamics
    # Mask immersed velocities and fill halo regions for momentum
    foreach(mask_immersed_field!, model.momentum)
    fill_halo_regions!(model.momentum, model.clock, fields(model))
    return nothing
end

"""
$(TYPEDSIGNATURES)

For compressible dynamics, there is no pressure correction to apply.
Pressure is computed diagnostically from the equation of state.
"""
function TimeSteppers.make_pressure_correction!(model::CompressibleModel, Δt)
    # No pressure correction for compressible dynamics
    return nothing
end

#####
##### Update state for compressible dynamics
#####
##### This computes diagnostic quantities from prognostic fields,
##### including pressure from the equation of state.
#####
##### For compressible dynamics, we must compute pressure BEFORE temperature
##### to break the circular dependency in the thermodynamic state.
#####

function TimeSteppers.update_state!(model::CompressibleModel, callbacks=[]; compute_tendencies=true)
    tracer_density_to_specific!(model)
    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model), async=true)

    # First compute θ = ρθ / ρ (doesn't need pressure)
    compute_specific_thermodynamic_variable!(model)

    # Then compute pressure from the prognostic fields (ρ, θ)
    # This uses the Poisson equation: p = p₀ (ρ Rᵐ θ / p₀)^γ
    compute_pressure_from_prognostics!(model)

    # Now compute remaining auxiliary variables (velocities, temperature, etc.)
    # Temperature can now be computed because pressure is available
    compute_auxiliary_variables!(model)

    update_radiation!(model.radiation, model)
    compute_forcings!(model)
    compute_tendencies && compute_tendencies!(model)
    tracer_specific_to_density!(model)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute the specific thermodynamic variable (θ or e) from the prognostic density forms.
This doesn't require pressure.
"""
function compute_specific_thermodynamic_variable!(model::CompressibleModel)
    grid = model.grid
    arch = grid.architecture

    launch!(arch, grid, :xyz,
            _compute_specific_thermodynamic_variable!,
            model.formulation,
            model.dynamics)

    fill_halo_regions!(model.formulation)
    return nothing
end

@kernel function _compute_specific_thermodynamic_variable!(formulation, dynamics)
    i, j, k = @index(Global, NTuple)
    ρ = dynamics_density(dynamics)
    @inbounds ρᵢ = ρ[i, j, k]
    compute_specific_thermodynamic_variable!(formulation, dynamics, ρᵢ, i, j, k)
end

# Dispatch for potential temperature formulation
@inline function compute_specific_thermodynamic_variable!(formulation::LiquidIcePotentialTemperatureFormulation,
                                                          dynamics, ρ, i, j, k)
    @inbounds begin
        ρθ = formulation.potential_temperature_density[i, j, k]
        formulation.potential_temperature[i, j, k] = ρθ / ρ
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute pressure from the prognostic fields for compressible dynamics.

For potential temperature formulation, uses the Poisson equation:
```math
p = p₀ \\left( \\frac{ρ R^m θ}{p₀} \\right)^{γ}
```

where `γ = cᵖ/cᵛ` is the heat capacity ratio.
"""
function compute_pressure_from_prognostics!(model::CompressibleModel)
    grid = model.grid
    arch = grid.architecture

    p₀ = dynamics_surface_pressure(model.dynamics)

    launch!(arch, grid, :xyz,
            _compute_pressure_from_potential_temperature!,
            model.dynamics.pressure,
            grid,
            model.dynamics.density,
            model.formulation,
            model.specific_moisture,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants,
            p₀)

    fill_halo_regions!(model.dynamics.pressure)
    return nothing
end

@kernel function _compute_pressure_from_potential_temperature!(pressure,
                                                                grid,
                                                                density,
                                                                formulation::LiquidIcePotentialTemperatureFormulation,
                                                                specific_moisture,
                                                                microphysics,
                                                                microphysical_fields,
                                                                constants,
                                                                p₀)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = density[i, j, k]
        θ = formulation.potential_temperature[i, j, k]
        qᵗ = specific_moisture[i, j, k]
    end

    # Compute moisture fractions for mixture properties
    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    cᵛᵐ = cᵖᵐ - Rᵐ  # Heat capacity at constant volume

    # Poisson equation: p = p₀ (ρ Rᵐ θ / p₀)^γ where γ = cᵖ/cᵛ
    γ = cᵖᵐ / cᵛᵐ
    @inbounds pressure[i, j, k] = p₀ * (ρ * Rᵐ * θ / p₀)^γ
end

#####
##### Compute tendencies for compressible dynamics
#####
##### The main difference from anelastic is that we also compute
##### a tendency for the prognostic density field.
#####

function compute_tendencies!(model::CompressibleModel)
    grid = model.grid
    arch = grid.architecture
    Gρu = model.timestepper.Gⁿ.ρu
    Gρv = model.timestepper.Gⁿ.ρv
    Gρw = model.timestepper.Gⁿ.ρw
    Gρ = model.timestepper.Gⁿ.ρ

    model_fields = fields(model)

    #####
    ##### Density tendency (mass conservation)
    #####

    density_args = (
        model.advection.momentum,  # Use same advection as momentum for now
        model.momentum,
        model.dynamics.density)

    launch!(arch, grid, :xyz, compute_density_tendency!, Gρ, grid, density_args)

    #####
    ##### Momentum tendencies
    #####

    momentum_args = (
        model.dynamics.density,
        model.advection.momentum,
        model.velocities,
        model.closure,
        model.closure_fields,
        model.momentum,
        model.coriolis,
        model.clock,
        model_fields)

    u_args = tuple(momentum_args..., model.forcing.ρu)
    v_args = tuple(momentum_args..., model.forcing.ρv)

    # For compressible dynamics, vertical momentum includes pressure gradient
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

    # Add pressure gradient to momentum tendencies
    launch!(arch, grid, :xyz, add_pressure_gradient!, Gρu, Gρv, Gρw, grid, model.dynamics.pressure)

    # Common arguments for scalar tendencies
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
    ##### Thermodynamic density tendency
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

    return nothing
end

#####
##### Density tendency (continuity equation)
#####

@kernel function compute_density_tendency!(Gρ, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρ[i, j, k] = density_tendency(i, j, k, grid, args...)
end

using Oceananigans.Advection: div_𝐯u  # Mass flux divergence

"""
    density_tendency(i, j, k, grid, advection, momentum, density)

Compute the tendency for the density equation (mass conservation):

```math
\\partial_t \\rho = -\\nabla \\cdot (\\rho \\mathbf{u}) = -\\nabla \\cdot \\mathbf{M}
```

where `M = (ρu, ρv, ρw)` is the momentum.
"""
@inline function density_tendency(i, j, k, grid, advection, momentum, density)
    ρu = momentum.ρu
    ρv = momentum.ρv
    ρw = momentum.ρw

    # Mass flux divergence: ∇⋅(ρu) = ∂x(ρu) + ∂y(ρv) + ∂z(ρw)
    return -divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)
end

using Oceananigans.Operators: divᶜᶜᶜ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ

#####
##### Pressure gradient contribution to momentum
#####

@kernel function add_pressure_gradient!(Gρu, Gρv, Gρw, grid, pressure)
    i, j, k = @index(Global, NTuple)

    # Add pressure gradient contributions: -∂p/∂x, -∂p/∂y, -∂p/∂z
    @inbounds Gρu[i, j, k] -= ∂xᶠᶜᶜ(i, j, k, grid, pressure)
    @inbounds Gρv[i, j, k] -= ∂yᶜᶠᶜ(i, j, k, grid, pressure)
    @inbounds Gρw[i, j, k] -= ∂zᶜᶜᶠ(i, j, k, grid, pressure)
end

