#####
##### Time stepping for PrescribedVelocityFields (kinematic dynamics)
#####

using Oceananigans.Fields: set!

#####
##### Model initialization
#####

"""
$(TYPEDSIGNATURES)

Initialize thermodynamic state for kinematic models.
Sets the initial potential temperature to the reference state value.
"""
function AtmosphereModels.initialize_model_thermodynamics!(model::KinematicModel)
    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model, θ=θ₀)
    return nothing
end

#####
##### Pressure correction: no-op for kinematic dynamics
#####

"""
$(TYPEDSIGNATURES)

No pressure correction needed for kinematic dynamics (velocities are prescribed).
"""
TimeSteppers.compute_pressure_correction!(model::KinematicModel, Δt) = nothing

"""
$(TYPEDSIGNATURES)

No pressure correction needed for kinematic dynamics (velocities are prescribed).
"""
TimeSteppers.make_pressure_correction!(model::KinematicModel, Δt) = nothing

