using Oceananigans: Field, set!, compute!
using Oceananigans.Grids: Center

#####
##### Geostrophic forcing types (unmaterialized stubs)
#####

"""
    UGeostrophicForcing{V}

Stub for a forcing on the x-momentum equation representing the Coriolis force
acting on the deviation from geostrophic balance:

```math
F_{\\rho u} = - f \\rho_r v^g
```

where ``f`` is the Coriolis parameter, ``\\rho_r`` is the reference density,
and ``v^g`` is the y-component of the geostrophic velocity.

This stub is materialized during model construction with the model's
coriolis parameter and reference density.
"""
struct UGeostrophicForcing{V}
    vᵍ :: V
end

"""
    VGeostrophicForcing{U}

Stub for a forcing on the y-momentum equation representing the Coriolis force
acting on the deviation from geostrophic balance:

```math
F_{\\rho v} = + f \\rho_r u^g
```

where ``f`` is the Coriolis parameter, ``\\rho_r`` is the reference density,
and ``u^g`` is the x-component of the geostrophic velocity.

This stub is materialized during model construction with the model's
coriolis parameter and reference density.
"""
struct VGeostrophicForcing{U}
    uᵍ :: U
end

"""
    $(TYPEDSIGNATURES)

Create a pair of geostrophic forcings for the x- and y-momentum equations.

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

forcing = geostrophic_forcings(uᵍ, vᵍ)
model = AtmosphereModel(grid; coriolis=FPlane(f=1e-4), forcing)
```
"""
function geostrophic_forcings(uᵍ, vᵍ)
    Fρu = UGeostrophicForcing(vᵍ)
    Fρv = VGeostrophicForcing(uᵍ)
    return (; ρu=Fρu, ρv=Fρv)
end

#####
##### Materialized geostrophic forcings
#####

struct MaterializedUGeostrophicForcing{F, V}
    f :: F
    ρvᵍ :: V
end

struct MaterializedVGeostrophicForcing{F, U}
    f :: F
    ρuᵍ :: U
end

@inline function (forcing::MaterializedUGeostrophicForcing)(i, j, k, grid, clock, fields)
    f = forcing.f
    ρvᵍ = forcing.ρvᵍ
    return @inbounds - f * ρvᵍ[i, j, k]
end

@inline function (forcing::MaterializedVGeostrophicForcing)(i, j, k, grid, clock, fields)
    f = forcing.f
    ρuᵍ = forcing.ρuᵍ
    return @inbounds + f * ρuᵍ[i, j, k]
end

#####
##### Materialization functions for geostrophic forcings
#####

# These are called from AtmosphereModels.atmosphere_model_forcing

function materialize_geostrophic_forcing(forcing::UGeostrophicForcing,
                                         grid,
                                         coriolis,
                                         reference_density)
    vᵍ_func = forcing.vᵍ
    FT = eltype(grid)
    f = FT(coriolis.f)

    # Create a column velocity field to hold vᵍ
    # Note: vᵍ is a 1D column field (Nothing, Nothing, Center), so set! takes a function of z only
    vᵍ = Field{Nothing, Nothing, Center}(grid)
    set!(vᵍ, vᵍ_func)

    # Compute the geostrophic momentum density field ρᵣ * vᵍ
    ρvᵍ = Field(reference_density * vᵍ)
    compute!(ρvᵍ)

    return MaterializedUGeostrophicForcing(f, ρvᵍ)
end

function materialize_geostrophic_forcing(forcing::VGeostrophicForcing,
                                         grid,
                                         coriolis,
                                         reference_density)
    uᵍ_func = forcing.uᵍ
    FT = eltype(grid)
    f = FT(coriolis.f)

    # Create a column velocity field to hold uᵍ
    # Note: uᵍ is a 1D column field (Nothing, Nothing, Center), so set! takes a function of z only
    uᵍ = Field{Nothing, Nothing, Center}(grid)
    set!(uᵍ, uᵍ_func)

    # Compute the geostrophic momentum density field ρᵣ * uᵍ
    ρuᵍ = Field(reference_density * uᵍ)
    compute!(ρuᵍ)

    return MaterializedVGeostrophicForcing(f, ρuᵍ)
end

