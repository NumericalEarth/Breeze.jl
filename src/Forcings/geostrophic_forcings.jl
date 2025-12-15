using Oceananigans: Field, set!, compute!
using Oceananigans.Grids: Center, XDirection, YDirection
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### Geostrophic forcing types
#####

struct GeostrophicForcing{S, V, F}
    geostrophic_momentum :: V
    direction :: S  # +1 for v-forcing, -1 for u-forcing
    coriolis_parameter :: F
end

Adapt.adapt_structure(to,gf::GeostrophicForcing) =
    GeostrophicForcing(Adapt.adapt(to, gf.geostrophic_momentum),
                       Adapt.adapt(to, gf.direction),
                       Adapt.adapt(to, gf.coriolis_parameter))

GeostrophicForcing(u, dir) = GeostrophicForcing(u, dir, nothing)

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
    print(io, "└── geostrophic_momentum: ", prettysummary(forcing.geostrophic_momentum))
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
        print(io, "│   └── geostrophic_momentum: ", prettysummary(forcing.geostrophic_momentum), "\n")
    end

    name = names[end]
    forcing = ft[name]
    print(io, "└── $name: ", summary(forcing), "\n")
    print(io, "    └── geostrophic_momentum: ", prettysummary(forcing.geostrophic_momentum))
end

@inline function (forcing::XGeostrophicForcing)(i, j, k, grid, clock, fields)
    f = forcing.coriolis_parameter
    ρvᵍ = @inbounds forcing.geostrophic_momentum[i, j, k]
    return - f * ρvᵍ
end

@inline function (forcing::YGeostrophicForcing)(i, j, k, grid, clock, fields)
    f = forcing.coriolis_parameter
    ρuᵍ = @inbounds forcing.geostrophic_momentum[i, j, k]
    return + f * ρuᵍ
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

```jldoctest
using Breeze

uᵍ(z) = -10 + 0.001z
vᵍ(z) = 0.0

coriolis = FPlane(f=1e-4)
forcing = geostrophic_forcings(uᵍ, vᵍ)

# output
NamedTuple with 2 GeostrophicForcings:
├── ρu: GeostrophicForcing{XDirection}
│   └── geostrophic_momentum: vᵍ (generic function with 1 method)
└── ρv: GeostrophicForcing{YDirection}
    └── geostrophic_momentum: uᵍ (generic function with 1 method)
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

function materialize_atmosphere_model_forcing(forcing::GeostrophicForcing, field, name, model_field_names, context)
    grid = field.grid

    forcing_uᵍ = forcing.geostrophic_momentum

    uᵍ = if forcing_uᵍ isa Field
        forcing_uᵍ
    else
        uᵍ = Field{Nothing, Nothing, Center}(grid)
        set!(uᵍ, forcing_uᵍ)
    end

    # Compute the geostrophic momentum density field ρᵣ * vᵍ
    ρᵣ = context.reference_density
    set!(uᵍ, ρᵣ * uᵍ)

    FT = eltype(grid)
    f = context.coriolis.f |> FT

    return GeostrophicForcing(uᵍ, forcing.direction, f)
end
