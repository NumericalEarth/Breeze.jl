#####
##### PrescribedDynamics definition
#####

"""
$(TYPEDSIGNATURES)

Dynamics specification for kinematic (prescribed velocity) atmosphere models.

In kinematic mode, the velocity field is prescribed by functions `u`, `v`, `w` rather than
solved from the momentum equations. This enables isolated testing of microphysics,
thermodynamics, and other physics without solving the pressure Poisson equation.

Fields
======

- `reference_state`: The reference state providing density and pressure profiles
- `u, v, w`: Velocity functions or `ZeroField`s
- `parameters`: Optional parameters passed to velocity functions

When `parameters === nothing`, velocity functions have signature `u(x, y, z, t)`.
When `parameters !== nothing`, velocity functions have signature `u(x, y, z, t, parameters)`.
"""
struct PrescribedDynamics{R, U, V, W, P}
    reference_state :: R
    u :: U
    v :: V
    w :: W
    parameters :: P
end

"""
    PrescribedDynamics(reference_state;
                       u = ZeroField(),
                       v = ZeroField(),
                       w = ZeroField(),
                       parameters = nothing)

Construct `PrescribedDynamics` with prescribed velocity functions `u`, `v`, and `w`.

Arguments
=========

- `reference_state`: A `ReferenceState` providing background density and pressure profiles.

Keyword Arguments
=================

- `u, v, w`: Functions specifying the velocity field. Default is `ZeroField()` (no motion).
- `parameters`: Optional parameters passed to velocity functions.

Example
=======

```julia
reference_state = ReferenceState(grid, constants)

# Bell-shaped updraft
w_profile(x, y, z, t) = 2 * sin(π * z / 10000)
dynamics = PrescribedDynamics(reference_state; w=w_profile)

# With parameters
w_param(x, y, z, t, p) = p.w_max * sin(π * z / p.H)
dynamics = PrescribedDynamics(reference_state; w=w_param, parameters=(; w_max=5, H=10000))
```
"""
function PrescribedDynamics(reference_state;
                            u = ZeroField(),
                            v = ZeroField(),
                            w = ZeroField(),
                            parameters = nothing)
    return PrescribedDynamics(reference_state, u, v, w, parameters)
end

Adapt.adapt_structure(to, d::PrescribedDynamics) =
    PrescribedDynamics(adapt(to, d.reference_state),
                       adapt(to, d.u),
                       adapt(to, d.v),
                       adapt(to, d.w),
                       adapt(to, d.parameters))

#####
##### Dynamics interface implementation
#####

AtmosphereModels.materialize_dynamics(d::PrescribedDynamics, grid, bcs) = d

#####
##### Pressure interface (no pressure solve)
#####

AtmosphereModels.mean_pressure(d::PrescribedDynamics) = d.reference_state.pressure
AtmosphereModels.pressure_anomaly(::PrescribedDynamics) = ZeroField()
AtmosphereModels.total_pressure(d::PrescribedDynamics) = d.reference_state.pressure
AtmosphereModels.dynamics_density(d::PrescribedDynamics) = d.reference_state.density
AtmosphereModels.dynamics_pressure(d::PrescribedDynamics) = d.reference_state.pressure
AtmosphereModels.surface_pressure(d::PrescribedDynamics) = d.reference_state.surface_pressure
AtmosphereModels.standard_pressure(d::PrescribedDynamics) = d.reference_state.standard_pressure
AtmosphereModels.dynamics_pressure_solver(::PrescribedDynamics, grid) = nothing

#####
##### Prognostic fields: none for kinematic dynamics
#####

AtmosphereModels.prognostic_dynamics_field_names(::PrescribedDynamics) = ()
AtmosphereModels.additional_dynamics_field_names(::PrescribedDynamics) = ()
AtmosphereModels.prognostic_momentum_field_names(::PrescribedDynamics) = ()

#####
##### Momentum and velocity materialization
#####

wrap_prescribed_field(LX, LY, LZ, f::Function, grid; kwargs...) = FunctionField{LX, LY, LZ}(f, grid; kwargs...)
wrap_prescribed_field(LX, LY, LZ, ::ZeroField, grid; kwargs...) = ZeroField()
wrap_prescribed_field(LX, LY, LZ, f, grid; kwargs...) = field((LX, LY, LZ), f, grid)

function AtmosphereModels.materialize_momentum_and_velocities(dynamics::PrescribedDynamics, grid, bcs)
    clock = Clock{eltype(grid)}(time=0)
    parameters = dynamics.parameters

    u = wrap_prescribed_field(Face, Center, Center, dynamics.u, grid; clock, parameters)
    v = wrap_prescribed_field(Center, Face, Center, dynamics.v, grid; clock, parameters)
    w = wrap_prescribed_field(Center, Center, Face, dynamics.w, grid; clock, parameters)

    velocities = (; u, v, w)
    momentum = (;)  # Empty NamedTuple - no prognostic momentum

    return momentum, velocities
end

#####
##### Show methods
#####

function Base.summary(d::PrescribedDynamics)
    p₀ = prettysummary(d.reference_state.surface_pressure)
    θ₀ = prettysummary(d.reference_state.potential_temperature)
    return "PrescribedDynamics(p₀=$p₀, θ₀=$θ₀)"
end

function Base.show(io::IO, d::PrescribedDynamics)
    print(io, summary(d), '\n')
    print(io, "├── u: ", summary(d.u), '\n')
    print(io, "├── v: ", summary(d.v), '\n')
    print(io, "├── w: ", summary(d.w), '\n')
    print(io, "└── parameters: ", prettysummary(d.parameters))
end
