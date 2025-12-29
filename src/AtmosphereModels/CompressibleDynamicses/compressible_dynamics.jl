#####
##### CompressibleDynamics definition
#####

"""
    CompressibleDynamics{D, P, FT}

Represents fully compressible dynamics with prognostic density.

The compressible formulation directly time-steps density and computes pressure
from the ideal gas law. This formulation resolves acoustic waves and requires
explicit time-stepping with appropriate CFL conditions.

# Fields
- `density`: Prognostic density field `ρ`
- `pressure`: Pressure field computed from ideal gas law `p = ρ R^m T`
- `standard_pressure`: Reference pressure for potential temperature calculations (pˢᵗ, default 10⁵ Pa)

# Key differences from AnelasticDynamics
- Density `ρ` is prognostic (time-stepped), not a fixed reference state
- Pressure is computed from the equation of state, not from a projection
- No pressure solver is required
- Acoustic waves are resolved (not filtered)
"""
struct CompressibleDynamics{D, P, FT}
    density :: D              # ρ (prognostic)
    pressure :: P             # p = ρ R^m T (diagnostic)
    standard_pressure :: FT   # pˢᵗ (reference pressure for potential temperature)
end

"""
$(TYPEDSIGNATURES)

Construct `CompressibleDynamics` with optional `standard_pressure` (default 10⁵ Pa).
The density and pressure fields are materialized later in the model constructor.
"""
CompressibleDynamics(; standard_pressure=1e5) = CompressibleDynamics(nothing, nothing, standard_pressure)

Adapt.adapt_structure(to, dynamics::CompressibleDynamics) =
    CompressibleDynamics(adapt(to, dynamics.density),
                         adapt(to, dynamics.pressure),
                         dynamics.standard_pressure)

#####
##### Materialization
#####

"""
$(TYPEDSIGNATURES)

Materialize a stub `CompressibleDynamics` into a full dynamics object with density and pressure fields.
"""
function materialize_dynamics(dynamics::CompressibleDynamics, grid, boundary_conditions)
    # Get density boundary conditions if provided
    if haskey(boundary_conditions, :ρ)
        density = CenterField(grid, boundary_conditions=boundary_conditions.ρ)
    else
        density = CenterField(grid)  # Use default for grid topology
    end
    pressure = CenterField(grid)  # Diagnostic pressure from equation of state
    return CompressibleDynamics(density, pressure, dynamics.standard_pressure)
end

#####
##### Pressure interface
#####

"""
$(TYPEDSIGNATURES)

Return the mean (reference) pressure for `CompressibleDynamics`.
For compressible dynamics, there is no separate mean pressure - returns the full pressure field.
"""
mean_pressure(dynamics::CompressibleDynamics) = dynamics.pressure

"""
$(TYPEDSIGNATURES)

Return the pressure anomaly for `CompressibleDynamics`.
For compressible dynamics, there is no decomposition - returns zero.
"""
pressure_anomaly(dynamics::CompressibleDynamics) = 0

"""
$(TYPEDSIGNATURES)

Return the total pressure for `CompressibleDynamics`, in Pa.
"""
total_pressure(dynamics::CompressibleDynamics) = dynamics.pressure

#####
##### Density and pressure access interface
#####

"""
$(TYPEDSIGNATURES)

Return the prognostic density field for `CompressibleDynamics`.
"""
dynamics_density(dynamics::CompressibleDynamics) = dynamics.density

"""
$(TYPEDSIGNATURES)

Return the pressure field for `CompressibleDynamics`.
Pressure is computed diagnostically from the equation of state.
"""
dynamics_pressure(dynamics::CompressibleDynamics) = dynamics.pressure

#####
##### Prognostic fields
#####

# Compressible dynamics has prognostic density
prognostic_dynamics_field_names(::CompressibleDynamics) = (:ρ,)
additional_dynamics_field_names(::CompressibleDynamics) = ()

"""
$(TYPEDSIGNATURES)

Return prognostic fields specific to compressible dynamics.
Returns the density field as a prognostic variable.
"""
dynamics_prognostic_fields(dynamics::CompressibleDynamics) = (; ρ=dynamics.density)

"""
$(TYPEDSIGNATURES)

Return a standard surface pressure for boundary condition regularization.
For compressible dynamics, uses the standard atmospheric pressure (101325 Pa).
"""
dynamics_surface_pressure(dynamics::CompressibleDynamics) = 101325.0

"""
$(TYPEDSIGNATURES)

Return the standard pressure for potential temperature calculations.
"""
standard_pressure(dynamics::CompressibleDynamics) = dynamics.standard_pressure

#####
##### Pressure solver (none needed for compressible dynamics)
#####

"""
$(TYPEDSIGNATURES)

Return `nothing` for `CompressibleDynamics` - no pressure solver is needed.
Pressure is computed directly from the equation of state.
"""
dynamics_pressure_solver(dynamics::CompressibleDynamics, grid) = nothing

#####
##### Show methods
#####

function Base.summary(dynamics::CompressibleDynamics)
    return "CompressibleDynamics"
end

function Base.show(io::IO, dynamics::CompressibleDynamics)
    print(io, summary(dynamics), '\n')
    if dynamics.density === nothing
        print(io, "├── density: not materialized\n")
        print(io, "└── pressure: not materialized")
    else
        print(io, "├── density: ", prettysummary(dynamics.density), '\n')
        print(io, "└── pressure: ", prettysummary(dynamics.pressure))
    end
end

#####
##### Momentum and velocity materialization
#####

function materialize_momentum_and_velocities(dynamics::CompressibleDynamics, grid, boundary_conditions)
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

