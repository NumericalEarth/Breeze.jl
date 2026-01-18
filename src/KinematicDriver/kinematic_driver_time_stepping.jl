#####
##### Time stepping for PrescribedDynamics (kinematic dynamics)
#####

using KernelAbstractions: @kernel, @index

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: set!, FunctionField
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Oceananigans.Utils: launch!

#####
##### Model initialization
#####

AtmosphereModels.initialize_model_thermodynamics!(::KinematicModel) = nothing

#####
##### Velocity and momentum: no-ops (no momentum, velocities may be FunctionFields)
#####

AtmosphereModels.compute_velocities!(::KinematicModel) = nothing
AtmosphereModels.compute_momentum_tendencies!(::KinematicModel, model_fields) = nothing

#####
##### Setting velocities for kinematic models
#####

# Dispatch on velocity field type
AtmosphereModels.set_velocity!(model::KinematicModel, name::Symbol, value) =
    set_velocity!(model.velocities[name], value)

# Regular velocity fields: just set directly
set_velocity!(velocity::AbstractField, value) = set!(velocity, value)

# FunctionFields (from PrescribedVelocityFields): cannot be set
set_velocity!(::FunctionField, value) =
    throw(ArgumentError("Cannot set velocity component of PrescribedVelocityFields."))

# No momentum in kinematic models
AtmosphereModels.set_momentum!(::KinematicModel, name::Symbol, value) =
    throw(ArgumentError("Cannot set momentum component '$name' of a KinematicModel."))

#####
##### Pressure correction: no-op for kinematic dynamics
#####

TimeSteppers.compute_pressure_correction!(::KinematicModel, Δt) = nothing
TimeSteppers.make_pressure_correction!(::KinematicModel, Δt) = nothing

#####
##### Density tendency (prognostic density only)
#####

@inline mass_flux_x(i, j, k, grid, ρ, u) = ℑxᶠᵃᵃ(i, j, k, grid, ρ) * u[i, j, k]
@inline mass_flux_y(i, j, k, grid, ρ, v) = ℑyᵃᶠᵃ(i, j, k, grid, ρ) * v[i, j, k]
@inline mass_flux_z(i, j, k, grid, ρ, w) = ℑzᵃᵃᶠ(i, j, k, grid, ρ) * w[i, j, k]

@inline function div_ρU(i, j, k, grid, ρ, velocities)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (
        δxᶜᵃᵃ(i, j, k, grid, mass_flux_x, ρ, velocities.u) +
        δyᵃᶜᵃ(i, j, k, grid, mass_flux_y, ρ, velocities.v) +
        δzᵃᵃᶜ(i, j, k, grid, mass_flux_z, ρ, velocities.w)
    )
end

# Default: no divergence correction
@inline AtmosphereModels.c_div_ρU(i, j, k, grid, ::PrescribedDynamics, velocities, c) = zero(grid)

# With divergence correction: c * ∇·(ρU)
@inline function AtmosphereModels.c_div_ρU(i, j, k, grid, dynamics::PrescribedDynamics{true}, velocities, c)
    return @inbounds c[i, j, k] * div_ρU(i, j, k, grid, dynamics_density(dynamics), velocities)
end

# Fixed density: no tendency to compute
AtmosphereModels.compute_dynamics_tendency!(::AtmosphereModel{<:PrescribedDynamics{<:Any, <:PrescribedDensity}}) = nothing

# Prognostic density: compute tendency from continuity equation
function AtmosphereModels.compute_dynamics_tendency!(model::KinematicModel)
    grid = model.grid
    arch = grid.architecture
    ρ = dynamics_density(model.dynamics)
    Gρ = model.timestepper.Gⁿ.ρ

    launch!(arch, grid, :xyz, _compute_density_tendency!, Gρ, grid, ρ, model.velocities)

    return nothing
end

@kernel function _compute_density_tendency!(Gρ, grid, ρ, velocities)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρ[i, j, k] = -div_ρU(i, j, k, grid, ρ, velocities)
end

#####
##### Pressure: always prescribed in kinematic models
#####

AtmosphereModels.compute_auxiliary_dynamics_variables!(::KinematicModel) = nothing
