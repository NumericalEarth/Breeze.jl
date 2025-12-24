#####
##### Formulation definition
#####

"""
$(TYPEDSIGNATURES)

`AnelasticFormulation` is a dynamical formulation wherein the density and pressure are
small perturbations from a dry, hydrostatic, adiabatic `reference_state`.
The prognostic energy variable is the moist static energy density.
The energy density equation includes a buoyancy flux term, following
[Pauluis (2008)](@cite Pauluis2008).
"""
struct AnelasticFormulation{T, R, P}
    thermodynamics :: T
    reference_state :: R
    pressure_anomaly :: P
end

const valid_thermodynamics_types = (:LiquidIcePotentialTemperature, :StaticEnergy)

"""
$(TYPEDSIGNATURES)

Construct an un-materialized "stub" `AnelasticFormulation` with `reference_state` and `thermodynamics`.
The thermodynamics and pressure fields are materialized later in the model constructor.
"""
AnelasticFormulation(reference_state; thermodynamics=:LiquidIcePotentialTemperature) =
    AnelasticFormulation(thermodynamics, reference_state, nothing)

Adapt.adapt_structure(to, formulation::AnelasticFormulation) =
    AnelasticFormulation(adapt(to, formulation.thermodynamics),
                         adapt(to, formulation.reference_state),
                         adapt(to, formulation.pressure_anomaly))

# Note: AnelasticModel = AtmosphereModel{<:AnelasticFormulation} is defined in AtmosphereModels.jl

#####
##### Prognostic and additional field names
#####

function prognostic_field_names(formulation::AnelasticFormulation{<:Symbol})
    if formulation.thermodynamics == :StaticEnergy
        return tuple(:ρe)
    elseif formulation.thermodynamics == :LiquidIcePotentialTemperature
        return tuple(:ρθ)
    else
        throw(ArgumentError("Got $(formulation.thermodynamics) thermodynamics, which is not one of \
                             the valid types $valid_thermodynamics_types."))
    end
end

function additional_field_names(formulation::AnelasticFormulation{<:Symbol})
    if formulation.thermodynamics == :StaticEnergy
        return tuple(:e)
    elseif formulation.thermodynamics == :LiquidIcePotentialTemperature
        return tuple(:θ)
    end
end

#####
##### Default formulation and materialization
#####

"""
$(TYPEDSIGNATURES)

Construct a "stub" `AnelasticFormulation` with just the `reference_state`.
The thermodynamics and pressure fields are materialized later in the model constructor.
"""
function default_formulation(grid, constants)
    reference_state = ReferenceState(grid, constants)
    return AnelasticFormulation(reference_state)
end

"""
$(TYPEDSIGNATURES)

Materialize a stub `AnelasticFormulation` into a full formulation with thermodynamic fields
and the pressure anomaly field. The thermodynamic fields depend on the type of thermodynamics
specified in the stub (`:StaticEnergy` or `:LiquidIcePotentialTemperature`).
"""
function materialize_formulation(stub::AnelasticFormulation, grid, boundary_conditions)
    thermo_type = stub.thermodynamics
    pressure_anomaly = CenterField(grid)
    thermodynamics = materialize_thermodynamics(Val(thermo_type), grid, boundary_conditions)
    return AnelasticFormulation(thermodynamics, stub.reference_state, pressure_anomaly)
end

function materialize_thermodynamics(::Val{T}, grid, boundary_conditions) where T
    throw(ArgumentError("Got $T thermodynamics, which is not one of \
                         the valid types $valid_thermodynamics_types."))
    return nothing
end

#####
##### Pressure interface
#####

"""
$(TYPEDSIGNATURES)

Return the mean (reference) pressure field for `AnelasticFormulation`, in Pa.
"""
mean_pressure(formulation::AnelasticFormulation) = formulation.reference_state.pressure

"""
$(TYPEDSIGNATURES)

Return the non-hydrostatic pressure anomaly for `AnelasticFormulation`, in Pa.
Note: the internal field stores the kinematic pressure `p'/ρᵣ`; this function
returns `ρᵣ * p'/ρᵣ = p'` in Pa.
"""
function pressure_anomaly(formulation::AnelasticFormulation)
    ρᵣ = formulation.reference_state.density
    p′_over_ρᵣ = formulation.pressure_anomaly
    return ρᵣ * p′_over_ρᵣ
end

"""
$(TYPEDSIGNATURES)

Return the total pressure for `AnelasticFormulation`, in Pa.
This is `p = p̄ + p'`, where `p̄` is the hydrostatic reference pressure
and `p'` is the non-hydrostatic pressure anomaly.
"""
function total_pressure(formulation::AnelasticFormulation)
    p̄ = mean_pressure(formulation)
    p′ = pressure_anomaly(formulation)
    return p̄ + p′
end

#####
##### Density interface
#####

"""
$(TYPEDSIGNATURES)

Return the reference density field for `AnelasticFormulation`.

For anelastic models, the formulation density is the time-independent
reference state density `ρᵣ(z)`.
"""
formulation_density(formulation::AnelasticFormulation) = formulation.reference_state.density

"""
$(TYPEDSIGNATURES)

Return the reference pressure field for `AnelasticFormulation`.

For anelastic models, the formulation pressure is the time-independent
hydrostatic reference state pressure `pᵣ(z)`.
"""
formulation_pressure(formulation::AnelasticFormulation) = formulation.reference_state.pressure

#####
##### Show methods
#####

function Base.summary(formulation::AnelasticFormulation)
    p₀_str = prettysummary(formulation.reference_state.surface_pressure)
    θ₀_str = prettysummary(formulation.reference_state.potential_temperature)
    return string("AnelasticFormulation(p₀=", p₀_str, ", θ₀=", θ₀_str, ")")
end

function Base.show(io::IO, formulation::AnelasticFormulation)
    print(io, summary(formulation), '\n')

    if formulation.thermodynamics isa Symbol
        print(io, "└── thermodynamics: ", formulation.thermodynamics, '\n')
    else
        print(io, "├── pressure_anomaly: ", prettysummary(formulation.pressure_anomaly), '\n')
        print(io, "└── thermodynamics: ", prettysummary(formulation.thermodynamics))
    end
end

#####
##### Momentum and velocity materialization
#####

function materialize_momentum_and_velocities(formulation::AnelasticFormulation, grid, boundary_conditions)
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
