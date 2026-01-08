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

```jldoctest
using Oceananigans
using Breeze

grid = RectilinearGrid(size=(4, 4, 8), extent=(1000, 1000, 2000))
reference_state = ReferenceState(grid, ThermodynamicConstants())
dynamics = PrescribedDynamics(reference_state)

# output
PrescribedDynamics
└── reference_state: ReferenceState{Float64}
```
"""
struct PrescribedDynamics{R}
    reference_state :: R
end

Base.summary(::PrescribedDynamics) = "PrescribedDynamics"

function Base.show(io::IO, d::PrescribedDynamics)
    print(io, "PrescribedDynamics\n")
    print(io, "└── reference_state: ", summary(d.reference_state))
end

#####
##### PrescribedVelocityFields definition
#####

"""
$(TYPEDSIGNATURES)

Velocity specification using functions for time-dependent prescribed velocities.

When `parameters === nothing`, velocity functions have signature `f(x, y, z, t)`.
When `parameters !== nothing`, velocity functions have signature `f(x, y, z, t, parameters)`.

# Example

```jldoctest
using Oceananigans.Fields: ZeroField
using Breeze

w_func(x, y, z, t) = 2 * sin(π * z / 2000)
velocities = PrescribedVelocityFields(w = w_func)

# output
PrescribedVelocityFields
├── u: ZeroField
├── v: ZeroField
├── w: w_func
└── parameters: nothing
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

function Base.show(io::IO, v::PrescribedVelocityFields)
    print(io, "PrescribedVelocityFields\n")
    print(io, "├── u: ", prettysummary(v.u), "\n")
    print(io, "├── v: ", prettysummary(v.v), "\n")
    print(io, "├── w: ", prettysummary(v.w), "\n")
    print(io, "└── parameters: ", prettysummary(v.parameters))
end

prettysummary(::Nothing) = "nothing"
prettysummary(::ZeroField) = "ZeroField"
prettysummary(f::Function) = nameof(f)
prettysummary(x) = summary(x)

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
