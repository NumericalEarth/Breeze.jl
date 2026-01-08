#####
##### Time stepping for PrescribedDynamics (kinematic dynamics)
#####

using Oceananigans.Fields: set!

#####
##### Model initialization
#####

AtmosphereModels.initialize_model_thermodynamics!(m::KinematicModel) = set!(m, θ=m.dynamics.reference_state.potential_temperature)

#####
##### Velocity and momentum: no-ops (velocities are FunctionFields, no momentum)
#####

AtmosphereModels.compute_velocities!(::KinematicModel) = nothing
AtmosphereModels.compute_momentum_tendencies!(::KinematicModel, model_fields) = nothing

#####
##### Setting velocities/momentum throws for kinematic models
#####

AtmosphereModels.set_velocity!(::KinematicModel, name, value) = throw(ArgumentError("Cannot set velocities for KinematicModel. Velocities are prescribed."))
AtmosphereModels.set_momentum!(::KinematicModel, name, value) = throw(ArgumentError("Cannot set momentum for KinematicModel. Velocities are prescribed."))

#####
##### Pressure correction: no-op for kinematic dynamics
#####

TimeSteppers.compute_pressure_correction!(::KinematicModel, Δt) = nothing
TimeSteppers.make_pressure_correction!(::KinematicModel, Δt) = nothing
