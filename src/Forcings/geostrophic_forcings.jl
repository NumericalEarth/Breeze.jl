using Oceananigans: Field, set!, compute!
using Oceananigans.Grids: Center

#####
##### Geostrophic forcing types
#####

"""
    UGeostrophicForcing{F, V}

Forcing on the x-momentum equation representing the Coriolis force
acting on the deviation from geostrophic balance:

```math
F_{\\rho u} = - f \\rho_r v^g
```

where ``f`` is the Coriolis parameter, ``\\rho_r`` is the reference density,
and ``v^g`` is the y-component of the geostrophic velocity.

Before materialization, `vᵍ` is a function of `z`.
After materialization, `vᵍ` contains the computed field `ρᵣ * vᵍ(z)`.
"""
struct UGeostrophicForcing{F, V}
    f :: F
    vᵍ :: V
end

"""
    VGeostrophicForcing{F, U}

Forcing on the y-momentum equation representing the Coriolis force
acting on the deviation from geostrophic balance:

```math
F_{\\rho v} = + f \\rho_r u^g
```

where ``f`` is the Coriolis parameter, ``\\rho_r`` is the reference density,
and ``u^g`` is the x-component of the geostrophic velocity.

Before materialization, `uᵍ` is a function of `z`.
After materialization, `uᵍ` contains the computed field `ρᵣ * uᵍ(z)`.
"""
struct VGeostrophicForcing{F, U}
    f :: F
    uᵍ :: U
end

# Callable interface (for materialized forcings where vᵍ/uᵍ contain ρᵣ * velocity)
@inline function (forcing::UGeostrophicForcing)(i, j, k, grid, clock, fields)
    f = forcing.f
    ρvᵍ = forcing.vᵍ
    return @inbounds - f * ρvᵍ[i, j, k]
end

@inline function (forcing::VGeostrophicForcing)(i, j, k, grid, clock, fields)
    f = forcing.f
    ρuᵍ = forcing.uᵍ
    return @inbounds + f * ρuᵍ[i, j, k]
end

"""
    $(TYPEDSIGNATURES)

Create a pair of geostrophic forcings for the x- and y-momentum equations.

Arguments
=========

- `uᵍ`: Function of `z` specifying the x-component of the geostrophic velocity.
- `vᵍ`: Function of `z` specifying the y-component of the geostrophic velocity.
- `f`: Coriolis parameter (keyword argument).

Returns a `NamedTuple` with `ρu` and `ρv` forcing entries that can be merged
into the model forcing.

Example
=======

```julia
uᵍ(z) = -10 + 0.001z
vᵍ(z) = 0.0

coriolis = FPlane(f=1e-4)
forcing = geostrophic_forcings(uᵍ, vᵍ; f=coriolis.f)
model = AtmosphereModel(grid; coriolis, forcing)
```
"""
function geostrophic_forcings(uᵍ, vᵍ; f)
    Fρu = UGeostrophicForcing(f, vᵍ)
    Fρv = VGeostrophicForcing(f, uᵍ)
    return (; ρu=Fρu, ρv=Fρv)
end

#####
##### Materialization functions for geostrophic forcings
#####
# Geostrophic forcings

function materialize_atmosphere_model_forcing(forcing::UGeostrophicForcing, field, name, model_field_names, context)
    grid = field.grid
    reference_density = context.reference_density
    vᵍ_func = forcing.vᵍ
    FT = eltype(grid)
    f = FT(forcing.f)

    # Create a column velocity field to hold vᵍ
    vᵍ = Field{Nothing, Nothing, Center}(grid)
    set!(vᵍ, vᵍ_func)

    # Compute the geostrophic momentum density field ρᵣ * vᵍ
    ρvᵍ = Field(reference_density * vᵍ)
    compute!(ρvᵍ)

    return UGeostrophicForcing(f, ρvᵍ)
end

function materialize_atmosphere_model_forcing(forcing::VGeostrophicForcing, field, name, model_field_names, context)
    grid = field.grid
    reference_density = context.reference_density
    uᵍ_func = forcing.uᵍ
    FT = eltype(grid)
    f = FT(forcing.f)

    # Create a column velocity field to hold uᵍ
    uᵍ = Field{Nothing, Nothing, Center}(grid)
    set!(uᵍ, uᵍ_func)

    # Compute the geostrophic momentum density field ρᵣ * uᵍ
    ρuᵍ = Field(reference_density * uᵍ)
    compute!(ρuᵍ)

    return VGeostrophicForcing(f, ρuᵍ)
end
