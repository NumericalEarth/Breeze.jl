using Oceananigans: Field, set!, compute!
using Oceananigans.Grids: Center

#####
##### Geostrophic forcing types
#####

"""
    UGeostrophicForcing{V}

Forcing on the x-momentum equation representing the Coriolis force
acting on the deviation from geostrophic balance:

```math
F_{\\rho u} = - f \\rho_r v^g
```

where ``f`` is the Coriolis parameter, ``\\rho_r`` is the reference density,
and ``v^g`` is the y-component of the geostrophic velocity.

Before materialization, `vᵍ` is a function of `z`.
After materialization, this struct holds `f` and the computed field `ρᵣ * vᵍ(z)`.
"""
struct UGeostrophicForcing{V}
    vᵍ :: V
end

"""
    VGeostrophicForcing{U}

Forcing on the y-momentum equation representing the Coriolis force
acting on the deviation from geostrophic balance:

```math
F_{\\rho v} = + f \\rho_r u^g
```

where ``f`` is the Coriolis parameter, ``\\rho_r`` is the reference density,
and ``u^g`` is the x-component of the geostrophic velocity.

Before materialization, `uᵍ` is a function of `z`.
After materialization, this struct holds `f` and the computed field `ρᵣ * uᵍ(z)`.
"""
struct VGeostrophicForcing{U}
    uᵍ :: U
end

"""
    MaterializedGeostrophicForcing{F, V}

Materialized geostrophic forcing containing the Coriolis parameter `f`
and the precomputed geostrophic momentum density field.
"""
struct MaterializedGeostrophicForcing{S, F, V}
    sign :: S  # +1 for v-forcing, -1 for u-forcing
    f :: F
    ρϕᵍ :: V
end

@inline function (forcing::MaterializedGeostrophicForcing)(i, j, k, grid, clock, fields)
    return @inbounds forcing.sign * forcing.f * forcing.ρϕᵍ[i, j, k]
end

"""
    $(TYPEDSIGNATURES)

Create a pair of geostrophic forcings for the x- and y-momentum equations.

The Coriolis parameter is extracted from the model's `coriolis` during
model construction.

Arguments
=========

- `uᵍ`: Function of `z` specifying the x-component of the geostrophic velocity.
- `vᵍ`: Function of `z` specifying the y-component of the geostrophic velocity.

Returns a `NamedTuple` with `ρu` and `ρv` forcing entries that can be merged
into the model forcing.

Example
=======

```julia
uᵍ(z) = -10 + 0.001z
vᵍ(z) = 0.0

coriolis = FPlane(f=1e-4)
forcing = geostrophic_forcings(uᵍ, vᵍ)
model = AtmosphereModel(grid; coriolis, forcing)
```
"""
function geostrophic_forcings(uᵍ, vᵍ)
    Fρu = UGeostrophicForcing(vᵍ)
    Fρv = VGeostrophicForcing(uᵍ)
    return (; ρu=Fρu, ρv=Fρv)
end

#####
##### Materialization functions for geostrophic forcings
#####

function materialize_atmosphere_model_forcing(forcing::UGeostrophicForcing, field, name, model_field_names, context)
    grid = field.grid
    reference_density = context.reference_density
    coriolis = context.coriolis
    vᵍ_func = forcing.vᵍ
    FT = eltype(grid)
    f = FT(coriolis.f)

    # Create a column velocity field to hold vᵍ
    vᵍ = Field{Nothing, Nothing, Center}(grid)
    set!(vᵍ, vᵍ_func)

    # Compute the geostrophic momentum density field ρᵣ * vᵍ
    ρvᵍ = Field(reference_density * vᵍ)
    compute!(ρvᵍ)

    # Fρu = -f * ρᵣ * vᵍ
    return MaterializedGeostrophicForcing(-one(FT), f, ρvᵍ)
end

function materialize_atmosphere_model_forcing(forcing::VGeostrophicForcing, field, name, model_field_names, context)
    grid = field.grid
    reference_density = context.reference_density
    coriolis = context.coriolis
    uᵍ_func = forcing.uᵍ
    FT = eltype(grid)
    f = FT(coriolis.f)

    # Create a column velocity field to hold uᵍ
    uᵍ = Field{Nothing, Nothing, Center}(grid)
    set!(uᵍ, uᵍ_func)

    # Compute the geostrophic momentum density field ρᵣ * uᵍ
    ρuᵍ = Field(reference_density * uᵍ)
    compute!(ρuᵍ)

    # Fρv = +f * ρᵣ * uᵍ
    return MaterializedGeostrophicForcing(one(FT), f, ρuᵍ)
end
