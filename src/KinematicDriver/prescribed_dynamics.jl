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
└── reference_state: ReferenceState{Float64}(p₀=101325.0, θ₀=288.0, pˢᵗ=100000.0)
```
"""
struct PrescribedDynamics{R, V}
    reference_state :: R
    velocity_specification :: V
end

# Default constructor: no prescribed velocity functions
PrescribedDynamics(reference_state) = PrescribedDynamics(reference_state, nothing)

Base.summary(::PrescribedDynamics) = "PrescribedDynamics"

function Base.show(io::IO, d::PrescribedDynamics)
    print(io, "PrescribedDynamics\n")
    print(io, "└── reference_state: ", summary(d.reference_state))
end

#####
##### Dynamics interface implementation
#####

AtmosphereModels.prognostic_momentum_field_names(::PrescribedDynamics) = ()
AtmosphereModels.prognostic_dynamics_field_names(::PrescribedDynamics) = ()
AtmosphereModels.additional_dynamics_field_names(::PrescribedDynamics) = ()
AtmosphereModels.materialize_dynamics(d::PrescribedDynamics, grid, bcs) = d

# Store velocity specification in dynamics for dispatch
AtmosphereModels.update_dynamics_with_velocities(d::PrescribedDynamics, v::PrescribedVelocityFields) =
    PrescribedDynamics(d.reference_state, v)
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
    momentum = (;)
    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZFaceField(grid)
    velocities = (; u, v, w)
    return momentum, velocities
end

# For PrescribedVelocityFields (from Oceananigans)
function AtmosphereModels.materialize_velocities(velocities::PrescribedVelocityFields, grid)
    clock = Clock{eltype(grid)}(time=0)
    parameters = velocities.parameters

    u = wrap_prescribed_field(Face, Center, Center, velocities.u, grid; clock, parameters)
    v = wrap_prescribed_field(Center, Face, Center, velocities.v, grid; clock, parameters)
    w = wrap_prescribed_field(Center, Center, Face, velocities.w, grid; clock, parameters)

    return (; u, v, w)
end

wrap_prescribed_field(X, Y, Z, f::Function, grid; kwargs...) = FunctionField{X, Y, Z}(f, grid; kwargs...)
wrap_prescribed_field(X, Y, Z, f, grid; kwargs...) = field((X, Y, Z), f, grid)

#####
##### Adapt and on_architecture
#####

Adapt.adapt_structure(to, d::PrescribedDynamics) = 
    PrescribedDynamics(adapt(to, d.reference_state), adapt(to, d.velocity_specification))

Oceananigans.Architectures.on_architecture(to, d::PrescribedDynamics) = 
    PrescribedDynamics(on_architecture(to, d.reference_state), on_architecture(to, d.velocity_specification))
