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

```jldoctest
using Oceananigans
using Breeze

grid = RectilinearGrid(size=(4, 4, 8), extent=(1000, 1000, 2000))
reference_state = ReferenceState(grid, ThermodynamicConstants())
dynamics = PrescribedDynamics(reference_state)

# output
PrescribedDynamics
├── density: PrescribedDensity
├── pressure: 1×1×8 Field{Nothing, Nothing, Center} reduced over dims = (1, 2) on RectilinearGrid on CPU
├── surface_pressure: 127379.0
├── base_pressure: 101325.0
└── standard_pressure: 100000.0
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

function materialize_surface_pressure(surface_pressure::Number, density, p₀, g, grid)
    return convert(eltype(grid), surface_pressure)
end

materialize_surface_pressure(surface_pressure::AbstractField, density, p₀, g, grid) = surface_pressure

function materialize_surface_pressure(surface_pressure, density, p₀, g, grid)
    pˢ = Field{Center, Center, Nothing}(grid)
    set!(pˢ, surface_pressure)
    fill_halo_regions!(pˢ)
    return pˢ
end

function surface_pressure_from_base(density, p₀, g, grid)
    pˢ = Field{Center, Center, Nothing}(grid)
    arch = grid.architecture
    launch!(arch, grid, :xy, _surface_pressure_from_base!, pˢ, density, p₀, g, grid)
    fill_halo_regions!(pˢ)
    return pˢ
end

function surface_pressure_from_pressure_field(pressure, density, g, grid)
    pˢ = Field{Center, Center, Nothing}(grid)
    arch = grid.architecture
    launch!(arch, grid, :xy, _surface_pressure_from_pressure_field!, pˢ, pressure, density, g, grid)
    fill_halo_regions!(pˢ)
    return pˢ
end

pressure_with_bottom_value(pressure::Field, surface_pressure) =
    field_with_bottom_value(pressure, surface_pressure)

pressure_with_bottom_value(pressure::AbstractField, surface_pressure) = pressure

fill_pressure_halos!(pressure::Field) = fill_halo_regions!(pressure)
fill_pressure_halos!(pressure::AbstractField) = nothing

function materialize_pressure(pressure, surface_pressure, density, p₀, g, grid)
    ρ = unwrap_density(density)

    if isnothing(pressure)
        pˢ = isnothing(surface_pressure) ? surface_pressure_from_base(ρ, p₀, g, grid) :
                                          materialize_surface_pressure(surface_pressure, ρ, p₀, g, grid)
        loc = (Center(), Center(), Center())
        p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(pˢ))
        p = CenterField(grid, boundary_conditions=p_bcs)

        # Compute hydrostatic pressure: ∂p/∂z = -ρg
        arch = grid.architecture
        launch!(arch, grid, :xy, _hydrostatic_pressure!, p, ρ, pˢ, g, grid)
    else
        p = pressure isa AbstractField ? pressure : CenterField(grid)
        pressure isa AbstractField || set!(p, pressure)
        fill_pressure_halos!(p)

        pˢ = isnothing(surface_pressure) ? surface_pressure_from_pressure_field(p, ρ, g, grid) :
                                          materialize_surface_pressure(surface_pressure, ρ, p₀, g, grid)
        p = pressure_with_bottom_value(p, pˢ)
    end

    fill_pressure_halos!(p)
    return p, pˢ
end

@kernel function _surface_pressure_from_base!(pˢ, ρ, p₀, g, grid)
    i, j = @index(Global, NTuple)
    zˢ = znode(i, j, 1, grid, Center(), Center(), Face())
    @inbounds pˢ[i, j, 1] = p₀ - ρ[i, j, 1] * g * zˢ
end

@kernel function _surface_pressure_from_pressure_field!(pˢ, p, ρ, g, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        p¹ = p[i, j, 1]
        ρ¹ = ρ[i, j, 1]
    end
    Δz = Δzᶜᶜᶜ(i, j, 1, grid)
    @inbounds pˢ[i, j, 1] = surface_pressure_from_cell_center(p¹, ρ¹, Δz, g)
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
