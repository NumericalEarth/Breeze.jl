#####
##### PrescribedDynamics definition
#####

"""
$(TYPEDSIGNATURES)

Dynamics specification for kinematic (prescribed velocity) atmosphere models.

In kinematic mode, the velocity field is not solved from the momentum equations.
This enables isolated testing of microphysics, thermodynamics, and other physics
without solving the pressure Poisson equation.

By default, velocities are regular fields that can be set via `set!(model, u=..., w=...)`.
For time-dependent velocities, pass `velocities = PrescribedVelocityFields(...)` to the model.

# Example

```julia
reference_state = ReferenceState(grid, ThermodynamicConstants())

# Constant velocity (settable)
model = AtmosphereModel(grid; dynamics = PrescribedDynamics(reference_state))
set!(model, θ=300, qᵗ=0.01, w=1)

# Time-dependent velocity
w_func(x, y, z, t) = sin(π * z / 2000) * (1 - exp(-t / 100))
model = AtmosphereModel(grid;
    dynamics = PrescribedDynamics(reference_state),
    velocities = PrescribedVelocityFields(w = w_func))
```
"""
struct PrescribedDynamics{R}
    reference_state :: R
end

Base.summary(::PrescribedDynamics) = "PrescribedDynamics"

#####
##### PrescribedVelocityFields definition
#####

"""
$(TYPEDSIGNATURES)

Velocity specification using functions for time-dependent prescribed velocities.

When `parameters === nothing`, velocity functions have signature `f(x, y, z, t)`.
When `parameters !== nothing`, velocity functions have signature `f(x, y, z, t, parameters)`.

# Example

```julia
# Simple time-dependent updraft
w_func(x, y, z, t) = 2 * sin(π * z / 2000) * min(1, t / 100)
velocities = PrescribedVelocityFields(w = w_func)

# With parameters
w_param(x, y, z, t, p) = p.w_max * sin(π * z / p.H)
velocities = PrescribedVelocityFields(w = w_param, parameters = (; w_max=5, H=2000))
```
"""
struct PrescribedVelocityFields{U, V, W, P}
    u :: U
    v :: V
    w :: W
    parameters :: P
end

function PrescribedVelocityFields(; u = ZeroField(),
                                    v = ZeroField(),
                                    w = ZeroField(),
                                    parameters = nothing)
    return PrescribedVelocityFields(u, v, w, parameters)
end

Base.summary(::PrescribedVelocityFields) = "PrescribedVelocityFields"

#####
##### Dynamics interface implementation
#####

AtmosphereModels.prognostic_momentum_field_names(::PrescribedDynamics) = ()
AtmosphereModels.prognostic_dynamics_field_names(::PrescribedDynamics) = ()
AtmosphereModels.additional_dynamics_field_names(::PrescribedDynamics) = ()
AtmosphereModels.materialize_dynamics(d::PrescribedDynamics, grid, bcs) = d
AtmosphereModels.dynamics_pressure_solver(::PrescribedDynamics, grid) = nothing
AtmosphereModels.dynamics_density(d::PrescribedDynamics) = d.reference_state.density
AtmosphereModels.dynamics_pressure(d::PrescribedDynamics) = d.reference_state.pressure
AtmosphereModels.mean_pressure(d::PrescribedDynamics) = d.reference_state.pressure
AtmosphereModels.pressure_anomaly(::PrescribedDynamics) = ZeroField()
AtmosphereModels.total_pressure(d::PrescribedDynamics) = mean_pressure(d)
AtmosphereModels.surface_pressure(d::PrescribedDynamics) = d.reference_state.surface_pressure
AtmosphereModels.standard_pressure(d::PrescribedDynamics) = d.reference_state.standard_pressure
AtmosphereModels.dynamics_prognostic_fields(::PrescribedDynamics) = NamedTuple()

#####
##### Momentum and velocity materialization
#####

# For PrescribedDynamics with default velocities (regular fields)
function AtmosphereModels.materialize_momentum_and_velocities(::PrescribedDynamics, grid, bcs)
    # No momentum for kinematic dynamics
    momentum = (;)
    
    # Regular velocity fields (settable)
    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZFaceField(grid)
    velocities = (; u, v, w)
    
    return momentum, velocities
end

# For PrescribedVelocityFields (function-based velocities)
function AtmosphereModels.materialize_velocities(velocities::PrescribedVelocityFields, grid)
    clock = Clock{eltype(grid)}(time=0)
    parameters = velocities.parameters

    u = wrap_prescribed_field(Face, Center, Center, velocities.u, grid; clock, parameters)
    v = wrap_prescribed_field(Center, Face, Center, velocities.v, grid; clock, parameters)
    w = wrap_prescribed_field(Center, Center, Face, velocities.w, grid; clock, parameters)

    return (; u, v, w)
end

# Utility for wrapping functions/fields
wrap_prescribed_field(X, Y, Z, f::Function, grid; kwargs...) = FunctionField{X, Y, Z}(f, grid; kwargs...)
wrap_prescribed_field(X, Y, Z, f, grid; kwargs...) = field((X, Y, Z), f, grid)

#####
##### Adapt and on_architecture
#####

Adapt.adapt_structure(to, d::PrescribedDynamics) = PrescribedDynamics(adapt(to, d.reference_state))

Oceananigans.Architectures.on_architecture(to, d::PrescribedDynamics) = 
    PrescribedDynamics(on_architecture(to, d.reference_state))

Adapt.adapt_structure(to, v::PrescribedVelocityFields) =
    PrescribedVelocityFields(adapt(to, v.u), adapt(to, v.v), adapt(to, v.w), adapt(to, v.parameters))

Oceananigans.Architectures.on_architecture(to, v::PrescribedVelocityFields) =
    PrescribedVelocityFields(on_architecture(to, v.u), on_architecture(to, v.v),
                             on_architecture(to, v.w), on_architecture(to, v.parameters))
