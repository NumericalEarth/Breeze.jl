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
struct PrescribedDynamics{Div, D, P, FT}
    density :: D
    pressure :: P
    surface_pressure :: FT
    standard_pressure :: FT
end

# Convenient method for letting the user specify only the divergence parameter
# and infer the others.
function PrescribedDynamics{Div}(density::D,
                                 pressure::P,
                                 surface_pressure::FT,
                                 standard_pressure::FT) where {Div, D, P, FT}
    return PrescribedDynamics{Div, D, P, FT}(density, pressure, surface_pressure, standard_pressure)
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
├── surface_pressure: 101325.0
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

Construct `PrescribedDynamics` from a density field or `PrescribedDensity`.
If `pressure=nothing`, hydrostatic pressure is computed during materialization.
"""
function PrescribedDynamics(density;
                            pressure = nothing,
                            surface_pressure = 101325,
                            standard_pressure = 1e5,
                            divergence_correction = false)

    FT = eltype(density)
    return PrescribedDynamics{divergence_correction}(density, pressure,
                                                     convert(FT, surface_pressure),
                                                     convert(FT, standard_pressure))
end

Base.summary(::PrescribedDynamics) = "PrescribedDynamics"

function Base.show(io::IO, d::PrescribedDynamics)
    print(io, "PrescribedDynamics\n")
    print(io, "├── density: ", summary(d.density), '\n')
    print(io, "├── pressure: ", prettysummary(d.pressure), '\n')
    print(io, "├── surface_pressure: ", prettysummary(d.surface_pressure), '\n')
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
AtmosphereModels.validate_velocity_boundary_conditions(::PrescribedDynamics, _bcs) = nothing
AtmosphereModels.velocity_boundary_condition_names(::PrescribedDynamics) = (:u, :v, :w)

# Prescribed density → no prognostic density; otherwise ρ is prognostic
AtmosphereModels.prognostic_dynamics_field_names(::PrescribedDynamics{<:Any, <:PrescribedDensity}) = ()
AtmosphereModels.prognostic_dynamics_field_names(::PrescribedDynamics) = tuple(:ρ)

AtmosphereModels.dynamics_prognostic_fields(::PrescribedDynamics{<:Any, <:PrescribedDensity}) = NamedTuple()
AtmosphereModels.dynamics_prognostic_fields(d::PrescribedDynamics) = (; ρ=dynamics_density(d))

# Pressure accessors
AtmosphereModels.dynamics_pressure_solver(::PrescribedDynamics, _grid) = nothing
AtmosphereModels.dynamics_pressure(d::PrescribedDynamics) = d.pressure
AtmosphereModels.mean_pressure(d::PrescribedDynamics) = d.pressure
AtmosphereModels.pressure_anomaly(::PrescribedDynamics) = ZeroField()
AtmosphereModels.total_pressure(d::PrescribedDynamics) = d.pressure
AtmosphereModels.surface_pressure(d::PrescribedDynamics) = d.surface_pressure
AtmosphereModels.standard_pressure(d::PrescribedDynamics) = d.standard_pressure

#####
##### Materialization
#####

function AtmosphereModels.materialize_dynamics(d::PrescribedDynamics{Div}, grid, bcs, constants) where Div
    FT = eltype(grid)
    p₀ = convert(FT, d.surface_pressure)
    pˢᵗ = convert(FT, d.standard_pressure)
    g = constants.gravitational_acceleration
    density = materialize_density(d.density, grid, bcs)
    pressure = materialize_pressure(d.pressure, density, p₀, g, grid)
    return PrescribedDynamics{Div}(density, pressure, p₀, pˢᵗ)
end

materialize_density(density::AbstractField, _grid, _bcs) = density
materialize_pressure(pressure::AbstractField, _args...) = pressure

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

function materialize_pressure(pressure, density, p₀, g, grid)
    loc = (Center(), Center(), Center())
    p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
    p = CenterField(grid, boundary_conditions=p_bcs)

    if isnothing(pressure)
        # Compute hydrostatic pressure: ∂p/∂z = -ρg
        ρ = unwrap_density(density)
        arch = grid.architecture
        launch!(arch, grid, :xy, _hydrostatic_pressure!, p, ρ, p₀, g, grid)
    else
        set!(p, pressure)
    end

    fill_halo_regions!(p)
    return p
end

@kernel function _hydrostatic_pressure!(p, ρ, p₀, g, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        pₖ = p₀
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
wrap_velocity(X, Y, Z, f, grid; _kwargs...) = field((X, Y, Z), f, grid)

#####
##### Adapt and architecture transfer
#####

Adapt.adapt_structure(to, d::PrescribedDynamics{Div}) where Div =
    PrescribedDynamics{Div}(adapt(to, d.density), adapt(to, d.pressure),
                            d.surface_pressure, d.standard_pressure)

Oceananigans.Architectures.on_architecture(to, d::PrescribedDynamics{Div}) where Div =
    PrescribedDynamics{Div}(on_architecture(to, d.density), on_architecture(to, d.pressure),
                            d.surface_pressure, d.standard_pressure)
