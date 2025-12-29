#####
##### AnelasticDynamics definition
#####

"""
    AnelasticDynamics{R, P}

Represents anelastic dynamics expanded about a hydrostatic reference state.

The anelastic approximation filters acoustic waves by assuming density and pressure
are small perturbations from a reference state `(ρᵣ(z), pᵣ(z))` with constant
reference potential temperature `θᵣ`.

# Fields
- `reference_state`: The hydrostatic, adiabatic reference state
- `pressure_anomaly`: The non-hydrostatic pressure perturbation field (kinematic, `p'/ρᵣ`)
"""
struct AnelasticDynamics{R, P}
    reference_state :: R
    pressure_anomaly :: P
end

"""
$(TYPEDSIGNATURES)

Return `AnelasticDynamics` representing incompressible fluid dynamics expanded about `reference_state`.
"""
AnelasticDynamics(reference_state) = AnelasticDynamics(reference_state, nothing)

Adapt.adapt_structure(to, dynamics::AnelasticDynamics) =
    AnelasticDynamics(adapt(to, dynamics.reference_state),
                      adapt(to, dynamics.pressure_anomaly))

#####
##### Default dynamics and materialization
#####

"""
$(TYPEDSIGNATURES)

Construct a "stub" `AnelasticDynamics` with just the `reference_state`.
The pressure anomaly field is materialized later in the model constructor.
"""
function default_dynamics(grid, constants)
    reference_state = ReferenceState(grid, constants)
    return AnelasticDynamics(reference_state)
end

"""
$(TYPEDSIGNATURES)

Materialize a stub `AnelasticDynamics` into a full dynamics object with the pressure anomaly field.
"""
function materialize_dynamics(dynamics::AnelasticDynamics, grid, boundary_conditions)
    pressure_anomaly = CenterField(grid)
    return AnelasticDynamics(dynamics.reference_state, pressure_anomaly)
end

#####
##### Pressure interface
#####

"""
$(TYPEDSIGNATURES)

Return the mean (reference) pressure field for `AnelasticDynamics`, in Pa.
"""
mean_pressure(dynamics::AnelasticDynamics) = dynamics.reference_state.pressure

"""
$(TYPEDSIGNATURES)

Return the non-hydrostatic pressure anomaly for `AnelasticDynamics`, in Pa.
Note: the internal field stores the kinematic pressure `p'/ρᵣ`; this function
returns `ρᵣ * p'/ρᵣ = p'` in Pa.
"""
function pressure_anomaly(dynamics::AnelasticDynamics)
    ρᵣ = dynamics.reference_state.density
    p′_over_ρᵣ = dynamics.pressure_anomaly
    return ρᵣ * p′_over_ρᵣ
end

"""
$(TYPEDSIGNATURES)

Return the total pressure for `AnelasticDynamics`, in Pa.
This is `p = p̄ + p'`, where `p̄` is the hydrostatic reference pressure
and `p'` is the non-hydrostatic pressure anomaly.
"""
function total_pressure(dynamics::AnelasticDynamics)
    p̄ = mean_pressure(dynamics)
    p′ = pressure_anomaly(dynamics)
    return p̄ + p′
end

#####
##### Density and pressure access interface
#####

"""
$(TYPEDSIGNATURES)

Return the reference density field for `AnelasticDynamics`.

For anelastic models, the dynamics density is the time-independent
reference state density `ρᵣ(z)`.
"""
dynamics_density(dynamics::AnelasticDynamics) = dynamics.reference_state.density

"""
$(TYPEDSIGNATURES)

Return the reference pressure field for `AnelasticDynamics`.

For anelastic models, the dynamics pressure is the time-independent
hydrostatic reference state pressure `pᵣ(z)`.
"""
dynamics_pressure(dynamics::AnelasticDynamics) = dynamics.reference_state.pressure

#####
##### Prognostic fields
#####

# Anelastic dynamics has no prognostic density - the density is the fixed reference state
prognostic_dynamics_field_names(::AnelasticDynamics) = ()
additional_dynamics_field_names(::AnelasticDynamics) = ()

"""
$(TYPEDSIGNATURES)

Return the surface pressure from the reference state for boundary condition regularization.
"""
dynamics_surface_pressure(dynamics::AnelasticDynamics) = dynamics.reference_state.surface_pressure

"""
$(TYPEDSIGNATURES)

Return the standard pressure from the reference state for potential temperature calculations.
"""
standard_pressure(dynamics::AnelasticDynamics) = dynamics.reference_state.standard_pressure

#####
##### Show methods
#####

function Base.summary(dynamics::AnelasticDynamics)
    p₀_str = prettysummary(dynamics.reference_state.surface_pressure)
    θ₀_str = prettysummary(dynamics.reference_state.potential_temperature)
    return string("AnelasticDynamics(p₀=", p₀_str, ", θ₀=", θ₀_str, ")")
end

function Base.show(io::IO, dynamics::AnelasticDynamics)
    print(io, summary(dynamics), '\n')
    if dynamics.pressure_anomaly === nothing
        print(io, "└── pressure_anomaly: not materialized")
    else
        print(io, "└── pressure_anomaly: ", prettysummary(dynamics.pressure_anomaly))
    end
end

#####
##### Momentum and velocity materialization
#####

function materialize_momentum_and_velocities(dynamics::AnelasticDynamics, grid, boundary_conditions)
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

