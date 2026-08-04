#####
##### PrescribedDensity: wrapper for fixed density
#####

"""
$(TYPEDSIGNATURES)

Wrapper indicating that density is fixed (not prognostic).
"""
struct PrescribedDensity{D}
    density :: D
end

Base.summary(::PrescribedDensity) = "PrescribedDensity"
Base.eltype(pd::PrescribedDensity) = eltype(pd.density)
Base.show(io::IO, d::PrescribedDensity) = print(io, "PrescribedDensity(", summary(d.density), ")")

Adapt.adapt_structure(to, pd::PrescribedDensity) = PrescribedDensity(adapt(to, pd.density))

Oceananigans.Architectures.on_architecture(to, pd::PrescribedDensity) =
    PrescribedDensity(on_architecture(to, pd.density))

#####
##### PrescribedDynamics: kinematic model dynamics
#####

"""
$(TYPEDEF)

Dynamics for kinematic atmosphere models where velocity is prescribed.
The type parameter `Div` indicates whether divergence correction is applied.
"""
struct PrescribedDynamics{Div, D, P, SP, FT}
    density :: D
    pressure :: P
    surface_pressure :: SP
    base_pressure :: FT
    standard_pressure :: FT
end

# Convenient method for letting the user specify only the divergence parameter
# and infer the others.
function PrescribedDynamics{Div}(density::D,
                                 pressure::P,
                                 surface_pressure::SP,
                                 base_pressure::FT,
                                 standard_pressure::FT) where {Div, D, P, SP, FT}
    return PrescribedDynamics{Div, D, P, SP, FT}(density, pressure, surface_pressure,
                                                 base_pressure, standard_pressure)
end

"""
$(TYPEDSIGNATURES)

Construct `PrescribedDynamics` from a [`ReferenceState`](@ref).
Wraps density in `PrescribedDensity` (fixed in time).

If `divergence_correction=true`, scalar tendencies include `+c∇·(ρU)` to
account for the non-divergent velocity field.

# Example

The grid's bottom face is the ground, so give it as `z = (0, Lz)`; on a domain that starts at
``z = 0`` the bottom-face `surface_pressure` and the ``z = 0`` `base_pressure` datum coincide.

```jldoctest
using Oceananigans
using Breeze

grid = RectilinearGrid(size=(4, 4, 8), x=(0, 1000), y=(0, 1000), z=(0, 2000))
reference_state = ReferenceState(grid, ThermodynamicConstants())
dynamics = PrescribedDynamics(reference_state)

(dynamics.surface_pressure[1, 1, 1], dynamics.base_pressure)

# output
(101325.0, 101325.0)
```
"""
function PrescribedDynamics(reference_state::ReferenceState; divergence_correction=false)
    density = PrescribedDensity(reference_state.density)
    pressure = reference_state.pressure
    pˢ = reference_state.surface_pressure
    p₀ = reference_state.base_pressure
    pˢᵗ = reference_state.standard_pressure
    return PrescribedDynamics{divergence_correction}(density, pressure, pˢ, p₀, pˢᵗ)
end

"""
$(TYPEDSIGNATURES)

Construct `PrescribedDynamics` from a density field or `PrescribedDensity`.
If `pressure=nothing`, hydrostatic pressure is computed during materialization. `base_pressure`
is the pressure datum at `z = 0`. On a raised domain, the default bottom-face pressure extends
the lowest prescribed density down to the datum; pass `surface_pressure` to prescribe a different
hydrostatic anchor explicitly.
"""
function PrescribedDynamics(density;
                            pressure = nothing,
                            surface_pressure = nothing,
                            base_pressure = 101325,
                            standard_pressure = 1e5,
                            divergence_correction = false)

    FT = eltype(density)
    return PrescribedDynamics{divergence_correction}(density, pressure, surface_pressure,
                                                     convert(FT, base_pressure),
                                                     convert(FT, standard_pressure))
end

Base.summary(::PrescribedDynamics) = "PrescribedDynamics"

function Base.show(io::IO, d::PrescribedDynamics)
    print(io, "PrescribedDynamics\n")
    print(io, "├── density: ", summary(d.density), '\n')
    print(io, "├── pressure: ", prettysummary(d.pressure), '\n')
    print(io, "├── surface_pressure: ", prettysummary(d.surface_pressure), '\n')
    print(io, "├── base_pressure: ", prettysummary(d.base_pressure), '\n')
    print(io, "└── standard_pressure: ", prettysummary(d.standard_pressure))
end

#####
##### Dynamics interface
#####

# Extract the underlying density field
@inline unwrap_density(pd::PrescribedDensity) = pd.density
@inline unwrap_density(ρ) = ρ  # pass-through for regular fields

@inline AtmosphereModels.dynamics_density(d::PrescribedDynamics) = unwrap_density(d.density)

AtmosphereModels.prognostic_momentum_field_names(::PrescribedDynamics) = ()
AtmosphereModels.additional_dynamics_field_names(::PrescribedDynamics) = ()
AtmosphereModels.validate_velocity_boundary_conditions(::PrescribedDynamics, bcs) = nothing
AtmosphereModels.velocity_boundary_condition_names(::PrescribedDynamics) = (:u, :v, :w)

# Prescribed density → no prognostic density; otherwise ρ is prognostic
AtmosphereModels.prognostic_dynamics_field_names(::PrescribedDynamics{<:Any, <:PrescribedDensity}) = ()
AtmosphereModels.prognostic_dynamics_field_names(::PrescribedDynamics) = tuple(:ρ)

AtmosphereModels.dynamics_prognostic_fields(::PrescribedDynamics{<:Any, <:PrescribedDensity}) = NamedTuple()
AtmosphereModels.dynamics_prognostic_fields(d::PrescribedDynamics) = (; ρ=dynamics_density(d))

# Pressure accessors
AtmosphereModels.dynamics_pressure_solver(::PrescribedDynamics, grid) = nothing
AtmosphereModels.dynamics_pressure(d::PrescribedDynamics) = d.pressure
AtmosphereModels.pressure_anomaly(::PrescribedDynamics) = ZeroField()
AtmosphereModels.total_pressure(d::PrescribedDynamics) = d.pressure
AtmosphereModels.surface_pressure(d::PrescribedDynamics) = d.surface_pressure
AtmosphereModels.base_pressure(d::PrescribedDynamics) = d.base_pressure
AtmosphereModels.standard_pressure(d::PrescribedDynamics) = d.standard_pressure

#####
##### Materialization
#####

function AtmosphereModels.materialize_dynamics(d::PrescribedDynamics{Div}, grid, bcs, constants) where Div
    FT = eltype(grid)
    p₀ = convert(FT, d.base_pressure)
    pˢᵗ = convert(FT, d.standard_pressure)
    g = constants.gravitational_acceleration
    density = materialize_density(d.density, grid, bcs)
    pressure, surface_pressure = materialize_pressure(d.pressure, d.surface_pressure,
                                                       density, p₀, g, grid)
    return PrescribedDynamics{Div}(density, pressure, surface_pressure, p₀, pˢᵗ)
end

materialize_density(density::AbstractField, grid, bcs) = density

function materialize_density(density::PrescribedDensity, grid, bcs)
    ρ = materialize_density(density.density, grid, bcs)
    return PrescribedDensity(ρ)
end

function materialize_density(density, grid, bcs)
    ρ_bcs = haskey(bcs, :ρ) ? bcs.ρ : FieldBoundaryConditions()
    ρ = CenterField(grid, boundary_conditions=ρ_bcs)
    if !isnothing(density)
        set!(ρ, density)
        fill_halo_regions!(ρ)
    end
    return ρ
end

# A user-prescribed anchor is normalized to a 2D field, so `surface_pressure(dynamics)` reports the
# same type however the dynamics was built — including when it came from a `ReferenceState`, whose
# own `surface_pressure` is passed straight through.
materialize_surface_pressure(surface_pressure::AbstractField, grid) = surface_pressure
materialize_surface_pressure(surface_pressure, grid) = surface_state_field(grid, surface_pressure)

# Allocate a bottom-face field and fill it from `kernel!`. The halo fill is what lets the result be
# read by the column kernels and aliased into a boundary condition, so it belongs here rather than
# at each call site.
function surface_pressure_field(grid, kernel!, args...)
    pˢ = Field{Center, Center, Nothing}(grid)
    launch!(grid.architecture, grid, :xy, kernel!, pˢ, args..., grid)
    fill_halo_regions!(pˢ)
    return pˢ
end

# A pressure field we allocate carries the derived bottom-face pressure as its bottom boundary
# value; one the user supplied is used exactly as given, boundary conditions included.
function pressure_field_with_surface_value(grid, pˢ)
    loc = (Center(), Center(), Center())
    p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(surface_boundary_value(pˢ)))
    return CenterField(grid, boundary_conditions=p_bcs)
end

# `materialize_pressure` dispatches on the pressure spec and returns `(pressure, bottom-face
# pressure)`. Each method derives its own default anchor; a user-supplied `surface_pressure` always
# wins over it.
function materialize_pressure(::Nothing, surface_pressure, density, p₀, g, grid)
    ρ = unwrap_density(density)
    pˢ = isnothing(surface_pressure) ?
         surface_pressure_field(grid, _surface_pressure_from_base!, ρ, p₀, g) :
         materialize_surface_pressure(surface_pressure, grid)
    p = pressure_field_with_surface_value(grid, pˢ)

    # Compute hydrostatic pressure: ∂p/∂z = -ρg
    launch!(grid.architecture, grid, :xy, _hydrostatic_pressure!, p, ρ, pˢ, g, grid)
    fill_halo_regions!(p)
    return p, pˢ
end

# A field the user supplied is used exactly as given, halos and boundary conditions included.
function materialize_pressure(pressure::AbstractField, surface_pressure, density, p₀, g, grid)
    pˢ = isnothing(surface_pressure) ?
         surface_pressure_field(grid, _surface_pressure_from_pressure_field!,
                                pressure, unwrap_density(density), g) :
         materialize_surface_pressure(surface_pressure, grid)
    return pressure, pˢ
end

# A number or function: fill a scratch field so the bottom-face pressure can be diagnosed from it,
# then build the real field carrying that value as its bottom boundary condition.
function materialize_pressure(pressure, surface_pressure, density, p₀, g, grid)
    scratch = CenterField(grid)
    set!(scratch, pressure)
    fill_halo_regions!(scratch)

    pˢ = isnothing(surface_pressure) ?
         surface_pressure_field(grid, _surface_pressure_from_pressure_field!,
                                scratch, unwrap_density(density), g) :
         materialize_surface_pressure(surface_pressure, grid)
    p = pressure_field_with_surface_value(grid, pˢ)
    set!(p, pressure)
    fill_halo_regions!(p)
    return p, pˢ
end

@kernel function _surface_pressure_from_base!(pˢ, ρ, p₀, g, grid)
    i, j = @index(Global, NTuple)
    zˢ = znode(i, j, 1, grid, Center(), Center(), Face())
    @inbounds pˢ[i, j, 1] = p₀ - ρ[i, j, 1] * g * zˢ
end

@kernel function _surface_pressure_from_pressure_field!(pˢ, p, ρ, g, grid)
    i, j = @index(Global, NTuple)
    @inbounds pˢ[i, j, 1] = surface_pressure_from_cell_center(i, j, 1, grid, p, ρ, g)
end

@kernel function _hydrostatic_pressure!(p, ρ, pˢ, g, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        pₖ = column_surface_pressure(pˢ, i, j)
        for k in 1:grid.Nz
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            p[i, j, k] = pₖ - ρ[i, j, k] * g * Δz / 2
            pₖ = pₖ - ρ[i, j, k] * g * Δz
        end
    end
end

#####
##### Velocity materialization
#####

function AtmosphereModels.materialize_momentum_and_velocities(::PrescribedDynamics, grid, bcs)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZFaceField(grid, boundary_conditions=bcs.w)
    return NamedTuple(), (; u, v, w)
end

function AtmosphereModels.materialize_velocities(velocities::PrescribedVelocityFields, grid)
    clock = Clock{eltype(grid)}(time=0)
    params = velocities.parameters
    u = wrap_velocity(Face, Center, Center, velocities.u, grid; clock, parameters=params)
    v = wrap_velocity(Center, Face, Center, velocities.v, grid; clock, parameters=params)
    w = wrap_velocity(Center, Center, Face, velocities.w, grid; clock, parameters=params)
    return (; u, v, w)
end

wrap_velocity(X, Y, Z, f::Function, grid; kwargs...) = FunctionField{X, Y, Z}(f, grid; kwargs...)
wrap_velocity(X, Y, Z, f, grid; kwargs...) = field((X, Y, Z), f, grid)

#####
##### Adapt and architecture transfer
#####

Adapt.adapt_structure(to, d::PrescribedDynamics{Div}) where Div =
    PrescribedDynamics{Div}(adapt(to, d.density), adapt(to, d.pressure),
                            adapt(to, d.surface_pressure),
                            d.base_pressure, d.standard_pressure)

Oceananigans.Architectures.on_architecture(to, d::PrescribedDynamics{Div}) where Div =
    PrescribedDynamics{Div}(on_architecture(to, d.density), on_architecture(to, d.pressure),
                            on_architecture(to, d.surface_pressure),
                            d.base_pressure, d.standard_pressure)
