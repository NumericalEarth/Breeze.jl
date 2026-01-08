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

# For regular fields: just set the velocity (no momentum to update)
function AtmosphereModels.set_velocity!(model::KinematicModel, name::Symbol, value)
    u = model.velocities[name]
    if u isa FunctionField
        throw(ArgumentError("Cannot set velocity '$name': it is a FunctionField (prescribed). Use regular velocity fields if you need to set velocities."))
    end
    set!(u, value)
    return nothing
end

# No momentum in kinematic models
AtmosphereModels.set_momentum!(::KinematicModel, name::Symbol, value) = 
    throw(ArgumentError("KinematicModel has no momentum fields. Set velocities directly."))

#####
##### Pressure correction: no-op for kinematic dynamics
#####

TimeSteppers.compute_pressure_correction!(::KinematicModel, Δt) = nothing
TimeSteppers.make_pressure_correction!(::KinematicModel, Δt) = nothing
