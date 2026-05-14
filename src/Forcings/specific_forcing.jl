using ..AtmosphereModels: AtmosphereModels
using Oceananigans.Fields: location
using Oceananigans.Forcings: materialize_forcing
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### SpecificForcing
#####

struct SpecificForcing{F, D, LZ}
    forcing :: F            # materialized inner forcing (kernel callable)
    density :: D            # ρᵣ(z) for anelastic, ρ(x, y, z, t) for compressible
    target_z_location :: LZ # Center for u, v, scalars; Face for w
end

"""
$(TYPEDSIGNATURES)

Wrap a user-supplied forcing that produces a *specific* (per-unit-mass) tendency so that
Breeze applies the density multiply ``ρ`` at kernel time. After materialization, the
kernel callable returns

```math
ρ(i, j, k) \\, F_ϕ(i, j, k, t)
```

for fields located at `Center`, and interpolates ``ρ`` to the target vertical face via
``ℑzᵃᵃᶠ`` for forcings on `w` (or any `Face`-located target). `ρ` is `ρᵣ(z)` under
[`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) and the prognostic
`ρ(x, y, z, t)` under [`CompressibleDynamics`](@ref Breeze.CompressibleEquations.CompressibleDynamics);
the same wrapper handles both.

Users typically supply specific forcings directly through specific-named keys
(`u`, `v`, `w`, `θ`, `e`, `qᵉ`, `qᵛ`, …) in the `forcing` `NamedTuple` passed to
[`AtmosphereModel`](@ref), and the dispatch wraps each entry in `SpecificForcing`
automatically. The wrapper can also be constructed directly when finer control is needed.

The inner `forcing` can be anything accepted by Oceananigans' `materialize_forcing`:
a function `(x, y, z, t)`, a [`Returns`](@ref) callable, a `Field`, an
`Oceananigans.Forcing`, or a tuple of these.
"""
SpecificForcing(forcing) = SpecificForcing(forcing, nothing, nothing)

Adapt.adapt_structure(to, sf::SpecificForcing) =
    SpecificForcing(Adapt.adapt(to, sf.forcing),
                    Adapt.adapt(to, sf.density),
                    sf.target_z_location)

#####
##### Kernel callables: dispatch on vertical location of target field
#####

@inline function (sf::SpecificForcing{<:Any, <:Any, <:Center})(i, j, k, grid, clock, fields)
    Fϕ = sf.forcing(i, j, k, grid, clock, fields)
    return @inbounds sf.density[i, j, k] * Fϕ
end

@inline function (sf::SpecificForcing{<:Any, <:Any, <:Face})(i, j, k, grid, clock, fields)
    ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, sf.density)
    Fϕ = sf.forcing(i, j, k, grid, clock, fields)
    return ρᶠ * Fϕ
end

#####
##### Materialization: resolve density and vertical location from context + target field
#####

function AtmosphereModels.materialize_atmosphere_model_forcing(forcing::SpecificForcing,
                                                               field, name, model_field_names,
                                                               context::NamedTuple)
    inner = materialize_forcing(forcing.forcing, field, name, model_field_names)
    ρ = context.density
    _, _, LZ = location(field)
    return SpecificForcing(inner, ρ, LZ())
end

#####
##### Show
#####

Base.summary(sf::SpecificForcing) = string("SpecificForcing with forcing: ", prettysummary(sf.forcing))

Base.show(io::IO, sf::SpecificForcing) = print(io, summary(sf))
