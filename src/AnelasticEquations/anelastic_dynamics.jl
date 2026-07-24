#####
##### AnelasticDynamics definition
#####

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
function AtmosphereModels.default_dynamics(grid, constants)
    reference_state = ReferenceState(grid, constants)
    return AnelasticDynamics(reference_state)
end

"""
$(TYPEDSIGNATURES)

Materialize a stub `AnelasticDynamics` into a full dynamics object with the pressure anomaly field.
"""
function AtmosphereModels.materialize_dynamics(dynamics::AnelasticDynamics, grid, boundary_conditions, thermodynamic_constants)
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
AtmosphereModels.mean_pressure(dynamics::AnelasticDynamics) = dynamics.reference_state.pressure

"""
$(TYPEDSIGNATURES)

Return the non-hydrostatic pressure anomaly for `AnelasticDynamics`, in Pa.

!!! note "Kinematic pressure versus pressure"

    The internal field stores the kinematic pressure anomaly, i.e., ``p' / ρᵣ``
    (in m²/s²); this function returns ``p'`` in Pa.
"""
function AtmosphereModels.pressure_anomaly(dynamics::AnelasticDynamics)
    ρᵣ = dynamics.reference_state.density
    p′_over_ρᵣ = dynamics.pressure_anomaly
    return ρᵣ * p′_over_ρᵣ
end

"""
$(TYPEDSIGNATURES)

Return the total pressure for `AnelasticDynamics`, in Pa.
That is ``p = p̄ + p'``, where ``p̄`` is the hydrostatic reference pressure
and ``p'`` is the non-hydrostatic pressure anomaly.
"""
function AtmosphereModels.total_pressure(dynamics::AnelasticDynamics)
    p̄ = mean_pressure(dynamics)
    p′ = pressure_anomaly(dynamics)
    return p̄ + p′
end

"""
$(TYPEDSIGNATURES)

Default surface temperature for `BulkDrag` under `AnelasticDynamics`: the
reference-state surface temperature, recovered from the reference potential
temperature via the surface Exner function ``T₀ = (p₀/pˢᵗ)^{Rᵈ/cᵖᵈ}\\,θ₀``.

Used only when the user constructs `BulkDrag` without an explicit
`surface_temperature`. The result is a horizontally uniform scalar.
"""
function AtmosphereModels.default_drag_surface_temperature(dynamics::AnelasticDynamics, grid, constants)
    ref = dynamics.reference_state
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    Π₀ = (ref.surface_pressure / ref.standard_pressure)^(Rᵈ / cᵖᵈ)
    return Π₀ * ref.potential_temperature
end

#####
##### Density and pressure access interface
#####

"""
$(TYPEDSIGNATURES)

Return the reference density field for `AnelasticDynamics`.

For anelastic models, the dynamics density is the time-independent
reference state density ``ρᵣ(z)``.
"""
AtmosphereModels.dynamics_density(dynamics::AnelasticDynamics) = dynamics.reference_state.density

@inline function AtmosphereModels.humidity_density(i, j, k, dynamics::AnelasticDynamics,
                                                    T, q, constants)
    @inbounds p = dynamics.reference_state.pressure[i, j, k]
    return density(T, p, q, constants)
end

"""
$(TYPEDSIGNATURES)

Return the reference pressure field for `AnelasticDynamics`.

For anelastic models, the dynamics pressure is the time-independent
hydrostatic reference state pressure ``pᵣ(z)``.
"""
AtmosphereModels.dynamics_pressure(dynamics::AnelasticDynamics) = dynamics.reference_state.pressure

#####
##### Prognostic fields
#####

# Anelastic dynamics has no prognostic density - the density is the fixed reference state
AtmosphereModels.prognostic_dynamics_field_names(::AnelasticDynamics) = ()
AtmosphereModels.additional_dynamics_field_names(::AnelasticDynamics) = ()

"""
$(TYPEDSIGNATURES)

Return the surface pressure from the reference state for boundary condition regularization.
"""
AtmosphereModels.surface_pressure(dynamics::AnelasticDynamics) = dynamics.reference_state.surface_pressure

"""
$(TYPEDSIGNATURES)

Return the standard pressure from the reference state for potential temperature calculations.
"""
AtmosphereModels.standard_pressure(dynamics::AnelasticDynamics) = dynamics.reference_state.standard_pressure

AtmosphereModels.dynamics_reference_state(dynamics::AnelasticDynamics) = dynamics.reference_state

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

function AtmosphereModels.materialize_momentum_and_velocities(dynamics::AnelasticDynamics, grid, boundary_conditions)
    ρu = XFaceField(grid, boundary_conditions=boundary_conditions.ρu)
    ρv = YFaceField(grid, boundary_conditions=boundary_conditions.ρv)
    ρw = ZFaceField(grid, boundary_conditions=boundary_conditions.ρw)
    momentum = (; ρu, ρv, ρw)

    # Velocity is diagnostic (u = ρu/ρ via compute_velocities!). Use the auxiliary-field
    # default BCs (`nothing` on Bounded-Face sides, Periodic on Periodic sides), which
    # is what XFaceField gives us when constructed with no `boundary_conditions=` kwarg.
    # `nothing` on Bounded-Face prevents `fill_halo_regions!(velocities)` from clobbering
    # the kernel-computed boundary face — momentum carries the wall BC.
    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZFaceField(grid)
    velocities = (; u, v, w)

    return momentum, velocities
end
