#####
##### Time stepping for CompressibleDynamics
#####

#####
##### Model initialization
#####

# No default initialization for compressible models
initialize_model_thermodynamics!(model::CompressibleModel) = nothing

#####
##### Pressure correction (no-op for compressible dynamics)
#####

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - pressure is computed diagnostically from the equation of state.
"""
TimeSteppers.compute_pressure_correction!(model::CompressibleModel, Δt) = nothing

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - no pressure projection is needed.
"""
TimeSteppers.make_pressure_correction!(model::CompressibleModel, Δt) = nothing

#####
##### Pressure solver (no-op)
#####

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - pressure is computed from the equation of state, not solved.
"""
solve_for_pressure!(model::CompressibleModel) = nothing

