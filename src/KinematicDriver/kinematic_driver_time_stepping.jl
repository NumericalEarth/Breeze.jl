#####
##### Time stepping for PrescribedDynamics (kinematic dynamics)
#####

using Oceananigans.Fields: set!, FunctionField

#####
##### Model initialization
#####

AtmosphereModels.initialize_model_thermodynamics!(m::KinematicModel) = set!(m, θ=m.dynamics.reference_state.potential_temperature)

#####
##### Velocity and momentum: no-ops (no momentum, velocities may be FunctionFields)
#####

AtmosphereModels.compute_velocities!(::KinematicModel) = nothing
AtmosphereModels.compute_momentum_tendencies!(::KinematicModel, model_fields) = nothing

#####
##### Setting velocities for kinematic models
#####

# Dispatch on velocity specification stored in dynamics
AtmosphereModels.set_velocity!(model::KinematicModel, name::Symbol, value) =
    set_velocity!(model.dynamics.velocity_specification, model.velocities, name, value)

# Regular velocity fields (velocity_specification is nothing): just set directly
set_velocity!(::Nothing, velocities, name, value) = set!(velocities[name], value)

# PrescribedVelocityFields: cannot be set
set_velocity!(::PrescribedVelocityFields, velocities, name, value) = 
    throw(ArgumentError("Cannot set velocity component '$name' of PrescribedVelocityFields."))

# No momentum in kinematic models
AtmosphereModels.set_momentum!(::KinematicModel, name::Symbol, value) = 
    throw(ArgumentError("Cannot set momentum component '$name' of a KinematicModel."))

#####
##### Pressure correction: no-op for kinematic dynamics
#####

TimeSteppers.compute_pressure_correction!(::KinematicModel, Δt) = nothing
TimeSteppers.make_pressure_correction!(::KinematicModel, Δt) = nothing
