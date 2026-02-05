#####
##### CompressibleDynamics definition
#####

"""
$(TYPEDEF)

Fully compressible dynamics with prognostic density and diagnostic pressure.

Fields
======

- `density`: Prognostic density field ρ
- `pressure`: Diagnostic pressure field p = ρ Rᵐ T
- `standard_pressure`: Reference pressure pˢᵗ for potential temperature (default 10⁵ Pa)
- `surface_pressure`: Mean pressure at the bottom of the atmosphere p₀
- `time_discretization`: Time discretization scheme ([`SplitExplicitTimeDiscretization`](@ref) or [`ExplicitTimeStepping`](@ref))

The `time_discretization` determines how tendencies are computed and which
time-stepper is used:
- [`SplitExplicitTimeDiscretization`](@ref): Acoustic substepping with separate slow/fast tendencies
- [`ExplicitTimeStepping`](@ref): All tendencies computed together (small Δt required)
"""
struct CompressibleDynamics{D, P, FT, TD}
    density :: D              # ρ (prognostic)
    pressure :: P             # p = ρ R^m T (diagnostic)
    standard_pressure :: FT   # pˢᵗ (reference pressure for potential temperature)
    surface_pressure :: FT    # p₀ (mean pressure at the bottom of the atmosphere)
    time_discretization :: TD # SplitExplicitTimeDiscretization or ExplicitTimeStepping
end

"""
$(TYPEDSIGNATURES)

Construct `CompressibleDynamics` with optional `standard_pressure` (default 10⁵ Pa).
The density and pressure fields are materialized later in the model constructor.

Keyword Arguments
=================

- `standard_pressure`: Reference pressure for potential temperature (default: 10⁵ Pa)
- `surface_pressure`: Mean surface pressure (default: 101325.0 Pa)
- `time_discretization`: Time discretization scheme. Default: [`SplitExplicitTimeDiscretization`](@ref)
  for acoustic substepping. Use [`ExplicitTimeStepping`](@ref) for standard explicit.
"""
function CompressibleDynamics(; standard_pressure = 1e5,
                                surface_pressure = 101325.0,
                                time_discretization::TD = SplitExplicitTimeDiscretization()) where TD
    FT = promote_type(typeof(standard_pressure), typeof(surface_pressure))
    pˢᵗ = convert(FT, standard_pressure)
    p₀ = convert(FT, surface_pressure)
    return CompressibleDynamics(nothing, nothing, pˢᵗ, p₀, time_discretization)
end

Adapt.adapt_structure(to, dynamics::CompressibleDynamics) =
    CompressibleDynamics(adapt(to, dynamics.density),
                         adapt(to, dynamics.pressure),
                         dynamics.standard_pressure,
                         dynamics.surface_pressure,
                         dynamics.time_discretization)

#####
##### Materialization
#####

"""
$(TYPEDSIGNATURES)

Materialize a stub `CompressibleDynamics` into a full dynamics object with density and pressure fields.
"""
function AtmosphereModels.materialize_dynamics(dynamics::CompressibleDynamics, grid, boundary_conditions, thermodynamic_constants)
    # Get density boundary conditions if provided
    if haskey(boundary_conditions, :ρ)
        density = CenterField(grid, boundary_conditions=boundary_conditions.ρ)
    else
        density = CenterField(grid)  # Use default for grid topology
    end

    pressure = CenterField(grid)  # Diagnostic pressure from equation of state

    FT = eltype(grid)
    standard_pressure = convert(FT, dynamics.standard_pressure)
    surface_pressure = convert(FT, dynamics.surface_pressure)

    return CompressibleDynamics(density, pressure, standard_pressure, surface_pressure, dynamics.time_discretization)
end

#####
##### Pressure interface
#####

"""
$(TYPEDSIGNATURES)

Return the mean (reference) pressure for `CompressibleDynamics`.
For compressible dynamics, there is no separate mean pressure - returns the full pressure field.
"""
AtmosphereModels.mean_pressure(dynamics::CompressibleDynamics) = dynamics.pressure

"""
$(TYPEDSIGNATURES)

Return the pressure anomaly for `CompressibleDynamics`.
For compressible dynamics, there is no decomposition - returns zero.
"""
AtmosphereModels.pressure_anomaly(dynamics::CompressibleDynamics) = 0

"""
$(TYPEDSIGNATURES)

Return the total pressure for `CompressibleDynamics`, in Pa.
"""
AtmosphereModels.total_pressure(dynamics::CompressibleDynamics) = dynamics.pressure

#####
##### Density and pressure access interface
#####

"""
$(TYPEDSIGNATURES)

Return the prognostic density field for `CompressibleDynamics`.
"""
AtmosphereModels.dynamics_density(dynamics::CompressibleDynamics) = dynamics.density

"""
$(TYPEDSIGNATURES)

Return the pressure field for `CompressibleDynamics`.
Pressure is computed diagnostically from the equation of state.
"""
AtmosphereModels.dynamics_pressure(dynamics::CompressibleDynamics) = dynamics.pressure

#####
##### Prognostic fields
#####

# Compressible dynamics has prognostic density
AtmosphereModels.prognostic_dynamics_field_names(::CompressibleDynamics) = (:ρ,)
AtmosphereModels.additional_dynamics_field_names(::CompressibleDynamics) = ()

"""
$(TYPEDSIGNATURES)

Return prognostic fields specific to compressible dynamics.
Returns the density field as a prognostic variable.
"""
AtmosphereModels.dynamics_prognostic_fields(dynamics::CompressibleDynamics) = (; ρ=dynamics.density)

"""
$(TYPEDSIGNATURES)

Return a standard surface pressure for boundary condition regularization.
For compressible dynamics, uses the standard atmospheric pressure (101325 Pa).
"""
AtmosphereModels.surface_pressure(dynamics::CompressibleDynamics) = dynamics.surface_pressure

"""
$(TYPEDSIGNATURES)

Return the standard pressure for potential temperature calculations.
"""
AtmosphereModels.standard_pressure(dynamics::CompressibleDynamics) = dynamics.standard_pressure

#####
##### Pressure solver (none needed for compressible dynamics)
#####

"""
$(TYPEDSIGNATURES)

Return `nothing` for `CompressibleDynamics` - no pressure solver is needed.
Pressure is computed directly from the equation of state.
"""
AtmosphereModels.dynamics_pressure_solver(dynamics::CompressibleDynamics, grid) = nothing

"""
$(TYPEDSIGNATURES)

Return the default timestepper for `CompressibleDynamics` based on its `time_discretization`.

- [`SplitExplicitTimeDiscretization`](@ref): Returns `:AcousticSSPRungeKutta3` for acoustic substepping
- [`ExplicitTimeStepping`](@ref): Returns `:SSPRungeKutta3` for standard explicit time-stepping
"""
AtmosphereModels.default_timestepper(dynamics::CompressibleDynamics) =
    default_timestepper(dynamics.time_discretization)

default_timestepper(::SplitExplicitTimeDiscretization) = :AcousticSSPRungeKutta3
default_timestepper(::ExplicitTimeStepping) = :SSPRungeKutta3

function AtmosphereModels.augment_timestepper_kwargs(dynamics::CompressibleDynamics, timestepper_kwargs)
    td = dynamics.time_discretization
    if td isa SplitExplicitTimeDiscretization
        return merge((; split_explicit = td), timestepper_kwargs)
    else
        return timestepper_kwargs
    end
end

#####
##### Show methods
#####

Base.summary(::SplitExplicitTimeDiscretization) = "SplitExplicitTimeDiscretization"
Base.summary(::ExplicitTimeStepping) = "ExplicitTimeStepping"

function Base.summary(dynamics::CompressibleDynamics)
    td = summary(dynamics.time_discretization)
    return "CompressibleDynamics($td)"
end

function Base.show(io::IO, dynamics::CompressibleDynamics)
    print(io, summary(dynamics), '\n')
    if dynamics.density === nothing
        print(io, "├── density: not materialized\n")
        print(io, "├── pressure: not materialized\n")
        print(io, "└── time_discretization: ", summary(dynamics.time_discretization))
    else
        print(io, "├── density: ", prettysummary(dynamics.density), '\n')
        print(io, "├── pressure: ", prettysummary(dynamics.pressure), '\n')
        print(io, "└── time_discretization: ", summary(dynamics.time_discretization))
    end
end

#####
##### Momentum and velocity materialization
#####

function AtmosphereModels.materialize_momentum_and_velocities(::CompressibleDynamics, grid, boundary_conditions)
    ρu = XFaceField(grid, boundary_conditions=boundary_conditions.ρu)
    ρv = YFaceField(grid, boundary_conditions=boundary_conditions.ρv)
    ρw = ZFaceField(grid, boundary_conditions=boundary_conditions.ρw)
    momentum = (; ρu, ρv, ρw)

    velocity_bcs = NamedTuple(name => FieldBoundaryConditions() for name in (:u, :v, :w))
    velocity_bcs = regularize_field_boundary_conditions(velocity_bcs, grid, (:u, :v, :w))
    u = XFaceField(grid, boundary_conditions=velocity_bcs.u)
    v = YFaceField(grid, boundary_conditions=velocity_bcs.v)
    w = ZFaceField(grid, boundary_conditions=velocity_bcs.w)
    velocities = (; u, v, w)

    return momentum, velocities
end

#####
##### Potential temperature diagnostics interface
#####

"""
$(TYPEDSIGNATURES)

Return the pressure field for potential temperature diagnostics.
For compressible dynamics, uses the actual pressure field.
"""
AtmosphereModels.Diagnostics.dynamics_pressure_for_potential_temperature(dynamics::CompressibleDynamics) = dynamics.pressure

"""
$(TYPEDSIGNATURES)

Return the density field for potential temperature diagnostics.
For compressible dynamics, uses the actual density field.
"""
AtmosphereModels.Diagnostics.dynamics_density_for_potential_temperature(dynamics::CompressibleDynamics) = dynamics.density

"""
$(TYPEDSIGNATURES)

Return the standard pressure for potential temperature diagnostics.
"""
AtmosphereModels.Diagnostics.dynamics_standard_pressure(dynamics::CompressibleDynamics) = dynamics.standard_pressure
