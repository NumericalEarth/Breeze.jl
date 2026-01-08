#####
##### PrescribedVelocityFields definition
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
struct PrescribedVelocityFields{R, U, V, W, P}
    reference_state :: R
    u :: U
    v :: V
    w :: W
    parameters :: P
end

"""
    PrescribedVelocityFields(reference_state;
                             u = ZeroField(),
                             v = ZeroField(),
                             w = ZeroField(),
                             parameters = nothing)

Construct `PrescribedVelocityFields` with prescribed velocity functions `u`, `v`, and `w`.

Arguments
=========

- `reference_state`: A `ReferenceState` providing background density and pressure profiles.
  This is required because the anelastic formulation needs a reference density for
  converting between specific and density-weighted variables.

Keyword Arguments
=================

- `u, v, w`: Functions specifying the velocity field. Default is `ZeroField()` (no motion).
- `parameters`: Optional parameters passed to velocity functions.

If `isnothing(parameters)`, then `u, v, w` are called with the signature:

```julia
u(x, y, z, t) = # velocity component
```

If `!isnothing(parameters)`, then `u, v, w` are called with the signature:

```julia
u(x, y, z, t, parameters) = # parameterized velocity component
```

Example
=======

```julia
# Define reference state
reference_state = ReferenceState(grid, constants)

# Bell-shaped updraft
w_profile(x, y, z, t) = 2.0 * sin(π * z / 10000)
dynamics = PrescribedVelocityFields(reference_state; w=w_profile)

# With parameters
w_param(x, y, z, t, p) = p.w_max * sin(π * z / p.H)
params = (; w_max=5.0, H=10000.0)
dynamics = PrescribedVelocityFields(reference_state; w=w_param, parameters=params)
```
"""
function PrescribedVelocityFields(reference_state;
                                  u = ZeroField(),
                                  v = ZeroField(),
                                  w = ZeroField(),
                                  parameters = nothing)
    return PrescribedVelocityFields(reference_state, u, v, w, parameters)
end

Adapt.adapt_structure(to, dynamics::PrescribedVelocityFields) =
    PrescribedVelocityFields(adapt(to, dynamics.reference_state),
                             adapt(to, dynamics.u),
                             adapt(to, dynamics.v),
                             adapt(to, dynamics.w),
                             adapt(to, dynamics.parameters))

#####
##### Dynamics interface implementation
#####

# Materialize dynamics: already complete (unlike AnelasticDynamics which needs to create pressure_anomaly)
AtmosphereModels.materialize_dynamics(dynamics::PrescribedVelocityFields, grid, boundary_conditions) = dynamics

#####
##### Pressure interface (no pressure solve for kinematic dynamics)
#####

AtmosphereModels.mean_pressure(dynamics::PrescribedVelocityFields) = dynamics.reference_state.pressure
AtmosphereModels.pressure_anomaly(dynamics::PrescribedVelocityFields) = ZeroField()
AtmosphereModels.total_pressure(dynamics::PrescribedVelocityFields) = dynamics.reference_state.pressure

#####
##### Density and pressure access interface
#####

AtmosphereModels.dynamics_density(dynamics::PrescribedVelocityFields) = dynamics.reference_state.density
AtmosphereModels.dynamics_pressure(dynamics::PrescribedVelocityFields) = dynamics.reference_state.pressure

#####
##### Boundary condition interface
#####

AtmosphereModels.surface_pressure(dynamics::PrescribedVelocityFields) = dynamics.reference_state.surface_pressure
AtmosphereModels.standard_pressure(dynamics::PrescribedVelocityFields) = dynamics.reference_state.standard_pressure

#####
##### Prognostic fields interface
#####

AtmosphereModels.prognostic_dynamics_field_names(::PrescribedVelocityFields) = ()
AtmosphereModels.additional_dynamics_field_names(::PrescribedVelocityFields) = ()

#####
##### Pressure solver: none needed
#####

AtmosphereModels.dynamics_pressure_solver(::PrescribedVelocityFields, grid) = nothing

#####
##### Prescribed velocities check
#####

AtmosphereModels.has_prescribed_velocities(::PrescribedVelocityFields) = true

#####
##### Momentum and velocity materialization
#####

# Wrap velocity functions in FunctionField
wrap_prescribed_field(LX, LY, LZ, f::Function, grid; kwargs...) = FunctionField{LX, LY, LZ}(f, grid; kwargs...)
wrap_prescribed_field(LX, LY, LZ, ::ZeroField, grid; kwargs...) = ZeroField()
wrap_prescribed_field(LX, LY, LZ, f, grid; kwargs...) = field((LX, LY, LZ), f, grid)

function AtmosphereModels.materialize_momentum_and_velocities(dynamics::PrescribedVelocityFields, grid, boundary_conditions)
    clock = Clock{eltype(grid)}(time=0)
    parameters = dynamics.parameters

    # Create velocity fields from prescribed functions
    u = wrap_prescribed_field(Face, Center, Center, dynamics.u, grid; clock, parameters)
    v = wrap_prescribed_field(Center, Face, Center, dynamics.v, grid; clock, parameters)
    w = wrap_prescribed_field(Center, Center, Face, dynamics.w, grid; clock, parameters)

    velocities = (; u, v, w)

    # Create dummy momentum fields for interface compatibility
    # These are never updated during time stepping
    ρu = XFaceField(grid)
    ρv = YFaceField(grid)
    ρw = ZFaceField(grid)
    momentum = (; ρu, ρv, ρw)

    return momentum, velocities
end

#####
##### Show methods
#####

function Base.summary(dynamics::PrescribedVelocityFields)
    p₀_str = prettysummary(dynamics.reference_state.surface_pressure)
    θ₀_str = prettysummary(dynamics.reference_state.potential_temperature)
    return string("PrescribedVelocityFields(p₀=", p₀_str, ", θ₀=", θ₀_str, ")")
end

function Base.show(io::IO, dynamics::PrescribedVelocityFields)
    print(io, summary(dynamics), '\n')
    print(io, "├── u: ", summary(dynamics.u), '\n')
    print(io, "├── v: ", summary(dynamics.v), '\n')
    print(io, "├── w: ", summary(dynamics.w), '\n')
    print(io, "└── parameters: ", prettysummary(dynamics.parameters))
end

