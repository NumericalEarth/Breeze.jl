using ..AtmosphereModels: AtmosphereModels
using Oceananigans: Field, set!, compute!
using Oceananigans.Grids: Center, XDirection, YDirection
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### Geostrophic forcing
#####

struct GeostrophicForcing{S, V, F}
    geostrophic_velocity :: V
    direction :: S  # XDirection for u-forcing, YDirection for v-forcing
    coriolis_parameter :: F
end

Adapt.adapt_structure(to, gf::GeostrophicForcing) =
    GeostrophicForcing(Adapt.adapt(to, gf.geostrophic_velocity),
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

#####
##### Kernel: returns the specific geostrophic-adjustment tendency
##### (the ρ-multiply and horizontal interpolation happen in SpecificForcing).
##### The 1×1×Center storage is horizontally uniform, so [i, j, k] is well-defined.
#####

@inline function (forcing::XGeostrophicForcing)(i, j, k, grid, clock, fields)
    f = forcing.coriolis_parameter
    vᵍ = @inbounds forcing.geostrophic_velocity[i, j, k]
    return - f * vᵍ
end

@inline function (forcing::YGeostrophicForcing)(i, j, k, grid, clock, fields)
    f = forcing.coriolis_parameter
    uᵍ = @inbounds forcing.geostrophic_velocity[i, j, k]
    return + f * uᵍ
end

"""
$(TYPEDSIGNATURES)

Create a pair of geostrophic forcings for the x- and y-momentum equations,
keyed under specific names `u` and `v`. Each `GeostrophicForcing` returns a
specific tendency; the model's density factor `ρ` is applied automatically via
[`SpecificForcing`](@ref Breeze.Forcings.SpecificForcing) when the forcing is
dispatched under a specific key, with the correct horizontal interpolation of
ρ to the appropriate cell face.

The Coriolis parameter is extracted from the model's `coriolis` during
model construction.

Arguments
=========

- `uᵍ`: Function of `z` specifying the x-component of the geostrophic velocity.
- `vᵍ`: Function of `z` specifying the y-component of the geostrophic velocity.

Returns a `NamedTuple` with `u` and `v` forcing entries that can be merged
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
├── u: GeostrophicForcing{XDirection}
│   └── geostrophic_velocity: vᵍ (generic function with 1 method)
└── v: GeostrophicForcing{YDirection}
    └── geostrophic_velocity: uᵍ (generic function with 1 method)
```
"""
function geostrophic_forcings(uᵍ, vᵍ)
    Fu = GeostrophicForcing(vᵍ, XDirection())
    Fv = GeostrophicForcing(uᵍ, YDirection())
    return (; u=Fu, v=Fv)
end

#####
##### Materialization: store geostrophic velocity profile and the Coriolis parameter
#####

function AtmosphereModels.materialize_atmosphere_model_forcing(forcing::GeostrophicForcing,
                                                               field, name, model_field_names,
                                                               context::NamedTuple)
    if startswith(string(name), "ρ")
        msg = string("GeostrophicForcing now returns a specific tendency (e.g. -f vᵍ for u) and ",
                     "must be supplied under the specific prognostic name (`u` or `v`). ",
                     "Use `geostrophic_forcings(uᵍ, vᵍ)` to build the `(; u, v)` pair; ",
                     "Breeze applies the density factor ρ automatically via SpecificForcing.")
        throw(ArgumentError(msg))
    end

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

    return GeostrophicForcing(uᵍ, forcing.direction, f)
end
