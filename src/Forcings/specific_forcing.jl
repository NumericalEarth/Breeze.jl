using ..AtmosphereModels: AtmosphereModels, materialize_atmosphere_model_forcing
using Oceananigans: instantiated_location
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### SpecificForcing
#####

struct SpecificForcing{F, D, L}
    forcing :: F          # materialized inner forcing (kernel callable)
    density :: D          # ρᵣ(z) for anelastic, ρ(x, y, z, t) for compressible
    target_location :: L  # (LX, LY, LZ) tuple of instances for the target prognostic field
end

"""
$(TYPEDSIGNATURES)

Wrap a user-supplied forcing that produces a *specific* (per-unit-mass) tendency so that
Breeze applies the density multiply ``ρ`` at kernel time. After materialization, the
kernel callable returns

```math
ρ(i, j, k) \\, F_ϕ(i, j, k, t)
```

interpolating ``ρ`` to the appropriate cell face for fields whose target prognostic
lives at `Face` in any direction (e.g. ``ρ`` is interpolated to x-Face for `u`-forcings
via ``ℑxᶠᵃᵃ``, to z-Face for `w` via ``ℑzᵃᵃᶠ``). `ρ` is `ρᵣ(z)` under
[`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) and the prognostic
`ρ(x, y, z, t)` under [`CompressibleDynamics`](@ref Breeze.CompressibleEquations.CompressibleDynamics);
the same wrapper handles both.

Users typically supply specific forcings directly through specific-named keys
(`u`, `v`, `w`, `θ`, `e`, `qᵉ`, `qᵛ`, …) in the `forcing` `NamedTuple` passed to
[`AtmosphereModel`](@ref Breeze.AtmosphereModels.AtmosphereModel), and the dispatch wraps each entry in `SpecificForcing`
automatically. The wrapper can also be constructed directly when finer control is needed.

The inner `forcing` can be anything accepted by Breeze's
`materialize_atmosphere_model_forcing`: a function `(x, y, z, t)`, a `Returns`
callable, a `Field`, an `Oceananigans.Forcing`, a Breeze forcing such as
[`SubsidenceForcing`](@ref Breeze.Forcings.SubsidenceForcing) or one produced by
[`geostrophic_forcings`](@ref Breeze.Forcings.geostrophic_forcings), or a tuple of these.
"""
SpecificForcing(forcing) = SpecificForcing(forcing, nothing, nothing)

Adapt.adapt_structure(to, sf::SpecificForcing) =
    SpecificForcing(Adapt.adapt(to, sf.forcing),
                    Adapt.adapt(to, sf.density),
                    sf.target_location)

#####
##### Density at the target prognostic location: dispatch on (LX, LY, LZ).
##### Only the four staggered locations that appear in the Breeze prognostic state are
##### handled (Center,Center,Center for scalars; Face,Center,Center for u;
##### Center,Face,Center for v; Center,Center,Face for w).
#####

@inline density_at_target(::Tuple{Center, Center, Center}, ρ, i, j, k, grid) = @inbounds ρ[i, j, k]
@inline density_at_target(::Tuple{Face,   Center, Center}, ρ, i, j, k, grid) = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
@inline density_at_target(::Tuple{Center, Face,   Center}, ρ, i, j, k, grid) = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
@inline density_at_target(::Tuple{Center, Center, Face},   ρ, i, j, k, grid) = ℑzᵃᵃᶠ(i, j, k, grid, ρ)

#####
##### Kernel callable
#####

@inline function (sf::SpecificForcing)(i, j, k, grid, clock, fields)
    Fϕ = sf.forcing(i, j, k, grid, clock, fields)
    ρ = density_at_target(sf.target_location, sf.density, i, j, k, grid)
    return ρ * Fϕ
end

#####
##### Materialization: resolve density and target field location from context + target field
#####

function AtmosphereModels.materialize_atmosphere_model_forcing(forcing::SpecificForcing,
                                                               field, name, model_field_names,
                                                               context::NamedTuple)
    # SpecificForcing wraps a forcing that produces a specific tendency, so propagate
    # the specific (un-prefixed) prognostic name to the inner materializer — Breeze
    # forcing types like SubsidenceForcing and GeostrophicForcing expect the specific
    # name to look up the field they advect or apply at.
    specific_name = startswith(string(name), "ρ") ?
                    Symbol(string(name)[nextind(string(name), 1):end]) : name
    # Materialize the inner forcing against the *specific* field, not the ρ-weighted prognostic.
    specific_field = get(context.specific_fields, specific_name, field)
    inner = materialize_atmosphere_model_forcing(forcing.forcing, specific_field, specific_name,
                                                 model_field_names, context)
    ρ = context.density
    target_location = instantiated_location(field)
    return SpecificForcing(inner, ρ, target_location)
end

#####
##### compute_forcing! recurses into the inner so wrapped forcings (e.g. SubsidenceForcing,
##### which precomputes a horizontal average each step) refresh their auxiliary fields.
#####

AtmosphereModels.compute_forcing!(sf::SpecificForcing) = AtmosphereModels.compute_forcing!(sf.forcing)

#####
##### Show
#####

Base.summary(sf::SpecificForcing) = string("SpecificForcing with forcing: ", prettysummary(sf.forcing))

Base.show(io::IO, sf::SpecificForcing) = print(io, summary(sf))
