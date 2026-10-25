using Breeze.AtmosphereModels: AtmosphereModels, grid_moisture_fractions, precipitating_mass_fraction
using Oceananigans.Forcings: Relaxation, evaluate_target
using Oceananigans.Grids: node
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### NonprecipitatingWaterRelaxation
#####

struct NonprecipitatingWaterRelaxation{R, M, T, L, W, MP, MF, Q, D}
    rate :: R
    mask :: M
    target :: T
    location :: L
    density_weighted :: W # Val(true) under `ρqᵗ`, Val(false) under `qᵗ`
    microphysics :: MP
    microphysical_fields :: MF
    specific_prognostic_moisture :: Q
    density :: D
end

"""
$(TYPEDSIGNATURES)

The form that a `Relaxation` supplied under the total-moisture key `qᵗ` takes: it restores the
*non-precipitating* water

```math
qⁿᵖ = qᵛ + qˡ + qⁱ - qᵖ ,
```

vapor and cloud condensate, to `relaxation.target` at `relaxation.rate` within
`relaxation.mask`, and applies the tendency ``r (qⁿᵖ★ - qⁿᵖ)`` to the prognostic moisture
(``qᵛ`` or ``qᵉ``; Breeze multiplies by the density ``ρ``). Under the density key `ρqᵗ` the
target is a density and the tendency is ``r (ρqⁿᵖ★ - ρ qⁿᵖ)``. Here ``qᵖ`` is the precipitating mass
fraction of the microphysics scheme (rain and snow, say), so precipitation, which falls out of
the large-scale state that a relaxation target describes, is neither relaxed nor counted.

This is the "qt" of large-eddy simulation forcing protocols, and the same quantity under every
scheme: under saturation adjustment ``qⁿᵖ = qᵉ``, the prognostic moisture itself; under
non-equilibrium cloud formation it is ``qᵛ + qᶜˡ + qᶜⁱ``. Relaxing the prognostic vapor alone toward
a total-water target would instead leave the column's water too high by its cloud condensate.

Users do not construct this directly; `AtmosphereModel` builds it from a `Relaxation` under `qᵗ`
or `ρqᵗ` (alone or in a tuple of forcings), and other forcings under those keys are water sources
applied to the prognostic moisture unchanged. A `Relaxation` under the scheme's own moisture name
(`qᵛ` or `qᵉ`) still relaxes that variable alone.

```jldoctest
using Breeze

grid = RectilinearGrid(size=4, z=(0, 1000), topology=(Flat, Flat, Bounded))
relaxation = Relaxation(rate=1/3600, target=0.01)
model = AtmosphereModel(grid; forcing=(; qᵗ=relaxation))
model.forcing.ρqᵛ.forcing

# output
NonprecipitatingWaterRelaxation(rate=0.0002777777777777778, mask=1, target=0.01)
```
"""
function NonprecipitatingWaterRelaxation(relaxation::Relaxation)
    isnothing(relaxation.transform) ||
        throw(ArgumentError("a Relaxation under the total-moisture key qᵗ relaxes the non-precipitating " *
                            "water and does not support `transform`."))
    return NonprecipitatingWaterRelaxation(relaxation.rate, relaxation.mask, relaxation.target,
                                           nothing, nothing, nothing, nothing, nothing, nothing)
end

AtmosphereModels.total_moisture_forcing(relaxation::Relaxation) = NonprecipitatingWaterRelaxation(relaxation)

Adapt.adapt_structure(to, r::NonprecipitatingWaterRelaxation) =
    NonprecipitatingWaterRelaxation(Adapt.adapt(to, r.rate),
                                    Adapt.adapt(to, r.mask),
                                    Adapt.adapt(to, r.target),
                                    Adapt.adapt(to, r.location),
                                    r.density_weighted,
                                    Adapt.adapt(to, r.microphysics),
                                    Adapt.adapt(to, r.microphysical_fields),
                                    Adapt.adapt(to, r.specific_prognostic_moisture),
                                    Adapt.adapt(to, r.density))

Base.summary(r::NonprecipitatingWaterRelaxation) =
    string("NonprecipitatingWaterRelaxation(rate=", r.rate, ", mask=", summary(r.mask),
           ", target=", prettysummary(r.target), ")")

Base.show(io::IO, r::NonprecipitatingWaterRelaxation) = print(io, summary(r))

#####
##### Kernel: the specific tendency r (qⁿᵖ★ - qⁿᵖ) under `qᵗ` (the ρ-multiply happens in
##### SpecificForcing), or the density tendency r (ρqⁿᵖ★ - ρ qⁿᵖ) under `ρqᵗ`
#####

@inline relaxed_water(::Val{false}, ρ, qⁿᵖ) = qⁿᵖ
@inline relaxed_water(::Val{true}, ρ, qⁿᵖ) = ρ * qⁿᵖ

# The moisture mass fractions are those of the tendency kernels (cf. `TemperatureForcing`):
# total density, prognostic moisture and microphysical fields at (i, j, k).
@inline function (r::NonprecipitatingWaterRelaxation)(i, j, k, grid, clock, fields)
    X = node(i, j, k, grid, r.location...)
    @inbounds ρ = r.density[i, j, k]
    @inbounds qᵛᵉ = r.specific_prognostic_moisture[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, r.microphysics, ρ, qᵛᵉ, r.microphysical_fields)
    qᵖ = precipitating_mass_fraction(i, j, k, r.microphysics, r.microphysical_fields)
    qⁿᵖ = q.vapor + q.liquid + q.ice - qᵖ
    qⁿᵖ★ = evaluate_target(r.target, i, j, k, X, clock.time)
    return r.rate * r.mask(X...) * (qⁿᵖ★ - relaxed_water(r.density_weighted, ρ, qⁿᵖ))
end

#####
##### Materialization: the target and location as Oceananigans materializes them, and the moisture state
#####

function AtmosphereModels.materialize_atmosphere_model_forcing(r::NonprecipitatingWaterRelaxation,
                                                               field, name, model_field_names,
                                                               context::NamedTuple)
    relaxation = Relaxation(; rate=r.rate, mask=r.mask, target=r.target)
    materialized = materialize_forcing(relaxation, field, name, model_field_names)

    return NonprecipitatingWaterRelaxation(materialized.rate,
                                           materialized.mask,
                                           materialized.target,
                                           materialized.location,
                                           Val(startswith(string(name), "ρ")),
                                           context.microphysics,
                                           context.microphysical_fields,
                                           context.specific_prognostic_moisture,
                                           context.total_density)
end
