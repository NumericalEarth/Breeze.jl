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

Return the dynamics pressure field for `AnelasticDynamics`, in Pa.

For anelastic models, this is the time-independent hydrostatic reference state
pressure ``pᵣ(z)``.
"""
AtmosphereModels.dynamics_pressure(dynamics::AnelasticDynamics) = dynamics.reference_state.pressure

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
    p̄ = dynamics_pressure(dynamics)
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

# The anelastic reference density ρᵣ(z) is a dry reference state evaluated at the reference
# temperature, so it is not the local moist density that mass fractions are referenced to.
# Rediagnose that at the reference pressure: ρ = pᵣ(z) / (Rᵐ(q) T).
@inline function AtmosphereModels.gas_phase_density(i, j, k, dynamics::AnelasticDynamics, T, q, constants)
    @inbounds p = dynamics.reference_state.pressure[i, j, k]
    return density(T, p, q, constants)
end

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

    # Velocity is diagnostic (u = ρu/ρ via compute_velocities!). Its own normal-direction
    # faces are computed by that kernel, so those sides carry `nothing` and are never
    # clobbered by `fill_halo_regions!(velocities)` — momentum carries the wall BC there.
    # The tangential sides carry the velocity's own boundary conditions, which default to
    # the mirror fill (free slip) and accept `ValueBoundaryCondition(0)` for no slip: the
    # closure's viscous stress differentiates the diagnostic velocity, so a no-slip wall in
    # a wall-resolving simulation is set here rather than on the momentum.
    u = velocity_field(XFaceField, grid, get(boundary_conditions, :u, nothing), Val(:x))
    v = velocity_field(YFaceField, grid, get(boundary_conditions, :v, nothing), Val(:y))
    w = velocity_field(ZFaceField, grid, get(boundary_conditions, :w, nothing), Val(:z))
    velocities = (; u, v, w)

    return momentum, velocities
end

# The anelastic velocity is diagnostic, but the closure's viscous stress differentiates it, so its
# tangential wall values are the no-slip / free-slip condition of a wall-resolving simulation and
# are honoured (see `materialize_momentum_and_velocities`). The sides normal to a velocity are
# computed from the momentum and any condition there would be ignored, so those are rejected.
function AtmosphereModels.validate_velocity_boundary_conditions(::AnelasticDynamics, user_boundary_conditions)
    normal_sides = (u = (:west, :east), v = (:south, :north), w = (:bottom, :top))
    for name in (:u, :v, :w)
        haskey(user_boundary_conditions, name) || continue
        bcs = getproperty(user_boundary_conditions, name)
        for side in normal_sides[name]
            hasproperty(bcs, side) || continue
            isnothing(getproperty(bcs, side)) && continue
            getproperty(bcs, side) isa DefaultBoundaryCondition && continue
            throw(ArgumentError(string("A boundary condition was given for the velocity ", name, " on the ", side,
                                       " boundary, which is normal to it. The normal velocity at a wall is computed ",
                                       "from the momentum, so set that condition on ρ", name, " instead; conditions on ",
                                       "the sides tangential to a velocity set the wall stress seen by the closure.")))
        end
    end
    return nothing
end

# Without boundary conditions of its own the velocity keeps the auxiliary-field defaults, which
# mirror across a wall: free slip
velocity_field(FieldType, grid, ::Nothing, direction) = FieldType(grid)
velocity_field(FieldType, grid, bcs, direction) =
    FieldType(grid, boundary_conditions=tangential_velocity_boundary_conditions(bcs, direction))

# The velocity's boundary conditions with the two sides normal to it removed
tangential_velocity_boundary_conditions(bcs, ::Val{:x}) =
    FieldBoundaryConditions(; west=nothing, east=nothing, south=bcs.south, north=bcs.north,
                              bottom=bcs.bottom, top=bcs.top, immersed=bcs.immersed)

tangential_velocity_boundary_conditions(bcs, ::Val{:y}) =
    FieldBoundaryConditions(; west=bcs.west, east=bcs.east, south=nothing, north=nothing,
                              bottom=bcs.bottom, top=bcs.top, immersed=bcs.immersed)

tangential_velocity_boundary_conditions(bcs, ::Val{:z}) =
    FieldBoundaryConditions(; west=bcs.west, east=bcs.east, south=bcs.south, north=bcs.north,
                              bottom=nothing, top=nothing, immersed=bcs.immersed)
