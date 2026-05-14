using ..AtmosphereModels: AtmosphereModels
using Oceananigans: Field, set!, compute!
using Oceananigans.Grids: Center, XDirection, YDirection
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### Geostrophic forcing types
#####

struct GeostrophicForcing{S, V, R, F}
    geostrophic_velocity :: V
    density :: R
    direction :: S  # +1 for v-forcing, -1 for u-forcing
    coriolis_parameter :: F
end

Adapt.adapt_structure(to,gf::GeostrophicForcing) =
    GeostrophicForcing(Adapt.adapt(to, gf.geostrophic_velocity),
                       Adapt.adapt(to, gf.density),
                       Adapt.adapt(to, gf.direction),
                       Adapt.adapt(to, gf.coriolis_parameter))

GeostrophicForcing(u, dir) = GeostrophicForcing(u, nothing, dir, nothing)

const XGeostrophicForcing = GeostrophicForcing{XDirection}
const YGeostrophicForcing = GeostrophicForcing{YDirection}

#####
##### Show methods
#####

direction_str(::XDirection) = "XDirection"
direction_str(::YDirection) = "YDirection"

function Base.summary(forcing::GeostrophicForcing)
    dir = direction_str(forcing.direction)
    f = forcing.coriolis_parameter
    f_str = isnothing(f) ? "" : "(f=$(prettysummary(f)))"
    return string("GeostrophicForcing{", dir, "}", f_str)
end

function Base.show(io::IO, forcing::GeostrophicForcing)
    print(io, summary(forcing))
    print(io, '\n')
    print(io, "└── geostrophic_velocity: ", prettysummary(forcing.geostrophic_velocity))
end

const GeostrophicForcingTuple = Tuple{GeostrophicForcing, Vararg{GeostrophicForcing}}
const NamedGeostrophicForcingTuple = NamedTuple{S, <:GeostrophicForcingTuple} where S

function Base.show(io::IO, ft::NamedGeostrophicForcingTuple)
    names = keys(ft)
    N = length(ft)

    print(io, "NamedTuple with ", N, " GeostrophicForcings:\n")

    for name in names[1:end-1]
        forcing = ft[name]
        print(io, "├── $name: ", summary(forcing), "\n")
        print(io, "│   └── geostrophic_velocity: ", prettysummary(forcing.geostrophic_velocity), "\n")
    end

    name = names[end]
    forcing = ft[name]
    print(io, "└── $name: ", summary(forcing), "\n")
    print(io, "    └── geostrophic_velocity: ", prettysummary(forcing.geostrophic_velocity))
end

@inline geostrophic_momentum(i, j, k, grid, ρ, uᵍ) = @inbounds ρ[i, j, k] * uᵍ[i, j, k]

@inline function (forcing::XGeostrophicForcing)(i, j, k, grid, clock, fields)
    f = forcing.coriolis_parameter
    ρvᵍ = ℑxᶠᵃᵃ(i, j, k, grid, geostrophic_momentum, forcing.density, forcing.geostrophic_velocity)
    return - f * ρvᵍ
end

@inline function (forcing::YGeostrophicForcing)(i, j, k, grid, clock, fields)
    f = forcing.coriolis_parameter
    ρuᵍ = ℑyᵃᶠᵃ(i, j, k, grid, geostrophic_momentum, forcing.density, forcing.geostrophic_velocity)
    return + f * ρuᵍ
end

"""
$(TYPEDSIGNATURES)

Create a pair of geostrophic forcings for the x- and y-momentum equations.

The Coriolis parameter is extracted from the model's `coriolis` during
model construction. The geostrophic velocity is stored separately from the
model density, so compressible models form ``ρ uᵍ`` from the live prognostic
density after initialization.

Arguments
=========

- `uᵍ`: Function of `z` specifying the x-component of the geostrophic velocity.
- `vᵍ`: Function of `z` specifying the y-component of the geostrophic velocity.

Returns a `NamedTuple` with `ρu` and `ρv` forcing entries that can be merged
into the model forcing.

Example
=======

```jldoctest
using Breeze

uᵍ(z) = -10 + 0.001z
vᵍ(z) = 0.0

coriolis = FPlane(f=1e-4)
forcing = geostrophic_forcings(uᵍ, vᵍ)

# output
NamedTuple with 2 GeostrophicForcings:
├── ρu: GeostrophicForcing{XDirection}
│   └── geostrophic_velocity: vᵍ (generic function with 1 method)
└── ρv: GeostrophicForcing{YDirection}
    └── geostrophic_velocity: uᵍ (generic function with 1 method)
```
"""
function geostrophic_forcings(uᵍ, vᵍ)
    Fρu = GeostrophicForcing(vᵍ, XDirection())
    Fρv = GeostrophicForcing(uᵍ, YDirection())
    return (; ρu=Fρu, ρv=Fρv)
end

#####
##### Materialization functions for geostrophic forcings
#####

function AtmosphereModels.materialize_atmosphere_model_forcing(forcing::GeostrophicForcing, field, name, model_field_names, context)
    grid = field.grid

    forcing_uᵍ = forcing.geostrophic_velocity

    uᵍ = if forcing_uᵍ isa Field
        forcing_uᵍ
    else
        uᵍ = Field{Nothing, Nothing, Center}(grid)
        set!(uᵍ, forcing_uᵍ)
    end

    FT = eltype(grid)
    f = context.coriolis.f |> FT
    ρ = context.density

    return GeostrophicForcing(uᵍ, ρ, forcing.direction, f)
end
