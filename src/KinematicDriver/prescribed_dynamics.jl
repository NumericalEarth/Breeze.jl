#####
##### PrescribedDensity definition
#####

"""
$(TYPEDSIGNATURES)

Wrapper for a fixed (prescribed) density profile in kinematic dynamics.

The wrapped `density` may be a `Field`, a function of position, or a constant.
When wrapped, the density is *not* treated as a prognostic variable.
"""
struct PrescribedDensity{D}
    density :: D
end

Base.summary(::PrescribedDensity) = "PrescribedDensity"

function Base.show(io::IO, d::PrescribedDensity)
    print(io, "PrescribedDensity(", summary(d.density), ")")
end

#####
##### PrescribedDynamics definition
#####

struct PrescribedDynamics{Div, D, P, FT}
    density :: D
    pressure :: P
    surface_pressure :: FT
    standard_pressure :: FT

    function PrescribedDynamics{Div}(ρ::D, p::P, p₀::FT, pˢᵗ::FT) where {Div, D, P, FT}
        return new{Div, D, P, FT}(ρ, p, p₀, pˢᵗ)
    end
end

const DivergenceCorrectedPrescribedDynamics = PrescribedDynamics{true}


"""
$(TYPEDSIGNATURES)

Construct `PrescribedDynamics` from a [`ReferenceState`](@ref).

Dynamics specification for kinematic (prescribed velocity) atmosphere models.
In kinematic mode, the velocity field is not solved from the momentum equations.
This enables isolated testing of microphysics, thermodynamics, and other physics
without solving the pressure Poisson equation.

The density is wrapped in [`PrescribedDensity`](@ref) (fixed in time), and the pressure
and standard pressure are taken from the reference state.

If `divergence_correction=true`, scalar tendencies include an additional ``+ c ∇·(ρU)``
term so that the implied velocity field conserves mass (KiD-style "separate divergence"
correction). This effectively switches scalars to advective form under the assumption
that missing velocity components satisfy ``∇·(ρU)=0``.

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
├── density: PrescribedDensity
├── pressure: 2000×1 Field{Nothing, Nothing, Center} on RectilinearGrid on CPU
└── standard_pressure: 100000.0
```
"""
function PrescribedDynamics(reference_state::ReferenceState; divergence_correction=false)
    density = PrescribedDensity(reference_state.density)
    pressure = reference_state.pressure
    p₀ = reference_state.surface_pressure
    pˢᵗ = reference_state.standard_pressure
    return PrescribedDynamics{divergence_correction}(density, pressure, p₀, pˢᵗ)
end

"""
$(TYPEDSIGNATURES)

Construct `PrescribedDynamics` from a density field or [`PrescribedDensity`](@ref).

Density can be either a prognostic field (evolves via continuity), or wrapped in
[`PrescribedDensity`](@ref) to remain fixed in time.

Keyword Arguments
=================
- `pressure`: Pressure field. If `nothing`, pressure is diagnosed from the ideal gas law.
- `standard_pressure`: Reference pressure for potential temperature (default 1e5 Pa).
- `divergence_correction`: If `true`, apply divergence correction for prescribed density (default `false`).
"""
# Helper to extract the underlying field from PrescribedDensity or regular density
@inline _density_field(d::PrescribedDensity) = d.density
@inline _density_field(d) = d

function PrescribedDynamics(density;
                            pressure = nothing,
                            surface_pressure = 101325,
                            standard_pressure = 1e5,
                            divergence_correction = false)
    FT = eltype(_density_field(density))
    p₀ = convert(FT, surface_pressure)
    pˢᵗ = convert(FT, standard_pressure)
    return PrescribedDynamics{divergence_correction}(density, pressure, p₀, pˢᵗ)
end

Base.summary(::PrescribedDynamics) = "PrescribedDynamics"

function Base.show(io::IO, d::PrescribedDynamics{Div}) where Div
    print(io, "PrescribedDynamics\n")
    print(io, "├── density: ", summary(d.density), '\n')
    print(io, "├── pressure: ", prettysummary(d.pressure), '\n')
    print(io, "├── surface_pressure: ", prettysummary(d.surface_pressure), '\n')
    print(io, "└── standard_pressure: ", prettysummary(d.standard_pressure))
    Div && print(io, '\n', "└── divergence_correction: true")
end

#####
##### Helper utilities
#####

# Type alias for PrescribedDynamics with prescribed (fixed) density
const FixedDensityPrescribedDynamics{Div} = PrescribedDynamics{Div, <:PrescribedDensity}

@inline AtmosphereModels.dynamics_density(d::PrescribedDynamics{<:Any, <:PrescribedDensity}) = d.density.density
@inline AtmosphereModels.dynamics_density(d::PrescribedDynamics) = d.density

#####
##### Dynamics interface implementation
#####

AtmosphereModels.prognostic_momentum_field_names(::PrescribedDynamics) = ()
AtmosphereModels.additional_dynamics_field_names(::PrescribedDynamics) = ()

# Prescribed density means no prognostic density field
AtmosphereModels.prognostic_dynamics_field_names(::FixedDensityPrescribedDynamics) = ()
AtmosphereModels.prognostic_dynamics_field_names(::PrescribedDynamics) = (:ρ,)

# PrescribedDynamics allows velocity boundary conditions (velocities are regular fields)
AtmosphereModels.validate_velocity_boundary_conditions(::PrescribedDynamics, user_boundary_conditions) = nothing

# PrescribedDynamics needs default boundary conditions for velocities
AtmosphereModels.velocity_boundary_condition_names(::PrescribedDynamics) = (:u, :v, :w)

function AtmosphereModels.materialize_dynamics(d::PrescribedDynamics{Div}, grid, bcs) where Div
    FT = eltype(grid)
    p₀ = convert(FT, d.surface_pressure)
    pˢᵗ = convert(FT, d.standard_pressure)
    density = materialize_density(d.density, grid, bcs)
    pressure = materialize_pressure(d.pressure, grid)
    return PrescribedDynamics{Div}(density, pressure, p₀, pˢᵗ)
end

AtmosphereModels.dynamics_pressure_solver(::PrescribedDynamics, grid) = nothing
AtmosphereModels.dynamics_pressure(d::PrescribedDynamics) = d.pressure
AtmosphereModels.mean_pressure(d::PrescribedDynamics) = d.pressure
AtmosphereModels.pressure_anomaly(::PrescribedDynamics) = ZeroField()
AtmosphereModels.total_pressure(d::PrescribedDynamics) = mean_pressure(d)
AtmosphereModels.surface_pressure(d::PrescribedDynamics) = d.surface_pressure
AtmosphereModels.standard_pressure(d::PrescribedDynamics) = d.standard_pressure

# Prescribed density means no prognostic density field
AtmosphereModels.dynamics_prognostic_fields(::FixedDensityPrescribedDynamics) = NamedTuple()
AtmosphereModels.dynamics_prognostic_fields(d::PrescribedDynamics) = (; ρ=dynamics_density(d))

#####
##### Density and pressure materialization
#####

function materialize_density(density, grid, bcs)
    if density isa PrescribedDensity
        field = materialize_density_field(density.density, grid, bcs)
        return PrescribedDensity(field)
    end
    return materialize_density_field(density, grid, bcs)
end

function materialize_density_field(density, grid, bcs)
    if density isa AbstractField
        return density
    end

    density_bcs = haskey(bcs, :ρ) ? bcs.ρ : FieldBoundaryConditions()
    ρ = CenterField(grid, boundary_conditions=density_bcs)

    if !isnothing(density)
        set!(ρ, density)
        fill_halo_regions!(ρ)
    end

    return ρ
end

function materialize_pressure(pressure, grid)
    if pressure isa AbstractField
        return pressure
    end

    p = CenterField(grid)
    if !isnothing(pressure)
        set!(p, pressure)
        fill_halo_regions!(p)
    end

    return p
end

#####
##### Momentum and velocity materialization
#####

# For PrescribedDynamics with default velocities (regular fields)
function AtmosphereModels.materialize_momentum_and_velocities(::PrescribedDynamics, grid, bcs)
    momentum = (;)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZFaceField(grid, boundary_conditions=bcs.w)
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

Adapt.adapt_structure(to, d::PrescribedDynamics{Div}) where Div =
    PrescribedDynamics{Div}(adapt(to, d.density),
                            adapt(to, d.pressure),
                            d.surface_pressure,
                            d.standard_pressure)

Oceananigans.Architectures.on_architecture(to, d::PrescribedDynamics{Div}) where Div =
    PrescribedDynamics{Div}(on_architecture(to, d.density),
                            on_architecture(to, d.pressure),
                            d.surface_pressure,
                            d.standard_pressure)
