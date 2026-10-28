using ..AtmosphereModels: AtmosphereModels
using Oceananigans: Average, Field, set!, compute!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: AbstractField
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators: ∂zᶜᶜᶠ
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### Subsidence forcing
#####

struct SubsidenceForcing{W, A}
    subsidence_vertical_velocity :: W
    averaged_field :: A
end

Adapt.adapt_structure(to, sf::SubsidenceForcing) =
    SubsidenceForcing(Adapt.adapt(to, sf.subsidence_vertical_velocity),
                      Adapt.adapt(to, sf.averaged_field))

"""
$(TYPEDSIGNATURES)

Forcing that represents large-scale subsidence advecting horizontally-averaged
fields downward. The kernel returns the *specific* tendency

```math
F_ϕ = - w^s \\, ∂_z \\overline{ϕ}
```

where ``w^s`` is the `subsidence_vertical_velocity` and ``\\overline{ϕ}`` is the
horizontal average of the field being forced. Supply `SubsidenceForcing` under
the specific prognostic name (e.g. `θ`, `qᵉ`, `u`); the `AtmosphereModel` dispatch
wraps it in [`SpecificForcing`](@ref) so the density factor ``ρ`` is applied
automatically at kernel time.

# Fields
- `wˢ`: Either a function of `z` specifying the subsidence velocity profile,
        or a `Field` containing the subsidence velocity.

The horizontal average is computed automatically during `update_state!`.

# Example

```jldoctest
using Breeze

grid = RectilinearGrid(size=(64, 64, 75), x=(0, 6400), y=(0, 6400), z=(0, 3000))

wˢ(z) = z < 1500 ? -0.0065 * z / 1500 : -0.0065 * (1 - (z - 1500) / 600)
subsidence = SubsidenceForcing(wˢ)
forcing = (; θ=subsidence, qᵛ=subsidence)

model = AtmosphereModel(grid; forcing)

model.forcing.ρθ.forcing

# output
SubsidenceForcing with wˢ: 1×1×76 Field{Nothing, Nothing, Face} reduced over dims = (1, 2) on RectilinearGrid on CPU
└── averaged_field: 1×1×75 Field{Nothing, Nothing, Center} reduced over dims = (1, 2) on RectilinearGrid on CPU
```
"""
SubsidenceForcing(wˢ) = SubsidenceForcing(wˢ, nothing)

function Base.summary(forcing::SubsidenceForcing)
    wˢ = forcing.subsidence_vertical_velocity
    return string("SubsidenceForcing with wˢ: ", prettysummary(wˢ))
end

function Base.show(io::IO, forcing::SubsidenceForcing)
    print(io, summary(forcing))
    if !isnothing(forcing.averaged_field)
        print(io, '\n')
        print(io, "└── averaged_field: ", prettysummary(forcing.averaged_field))
    end
end

#####
##### Kernel: returns the specific subsidence tendency (the ρ-multiply happens in SpecificForcing)
#####

@inline w_dz_ϕᵃᵃᶠ(i, j, k, grid, w, ϕ) = @inbounds w[1, 1, k] * ∂zᶜᶜᶠ(1, 1, k, grid, ϕ)

@inline function ℑzbᵃᵃᶜ(i, j, k, grid, w_dz_ϕᵃᵃᶠ, wˢ, ϕ_avg)
    w_dz_ϕ⁺ = w_dz_ϕᵃᵃᶠ(i, j, k+1, grid, wˢ, ϕ_avg)
    w_dz_ϕᵏ = w_dz_ϕᵃᵃᶠ(i, j, k, grid, wˢ, ϕ_avg)
    ℑz_w_dz_ϕ = (w_dz_ϕ⁺ + w_dz_ϕᵏ) / 2
    top = k == grid.Nz
    bottom = k == 1
    return ifelse(top, w_dz_ϕᵏ, ifelse(bottom, w_dz_ϕ⁺, ℑz_w_dz_ϕ))
end

@inline function (forcing::SubsidenceForcing)(i, j, k, grid, clock, fields)
    wˢ = forcing.subsidence_vertical_velocity
    ϕ_avg = forcing.averaged_field
    w_dz_ϕ_avg = ℑzbᵃᵃᶜ(i, j, k, grid, w_dz_ϕᵃᵃᶠ, wˢ, ϕ_avg)
    return - w_dz_ϕ_avg
end

#####
##### Materialization: build the horizontal average of the specific field
#####

function AtmosphereModels.materialize_atmosphere_model_forcing(forcing::SubsidenceForcing,
                                                               field, name, model_field_names,
                                                               context::NamedTuple)
    if startswith(string(name), "ρ")
        msg = string("SubsidenceForcing now returns a specific tendency F_ϕ = -wˢ ∂_z ϕ̄ and ",
                     "must be supplied under the specific prognostic name (e.g. `θ` instead of `ρθ`). ",
                     "Breeze applies the density factor ρ automatically via SpecificForcing.")
        throw(ArgumentError(msg))
    end

    grid = field.grid

    if forcing.subsidence_vertical_velocity isa AbstractField
        wˢ = forcing.subsidence_vertical_velocity
    else
        wˢ = Field{Nothing, Nothing, Face}(grid)
        set!(wˢ, forcing.subsidence_vertical_velocity)
        fill_halo_regions!(wˢ)
    end

    # `name` is the specific prognostic name (e.g. :θ); look up the matching field directly.
    specific_field = haskey(context.specific_fields, name) ? context.specific_fields[name] : field
    averaged_field = Average(specific_field, dims=(1, 2)) |> Field

    return SubsidenceForcing(wˢ, averaged_field)
end

#####
##### compute_forcing! for subsidence forcing
#####

function AtmosphereModels.compute_forcing!(forcing::SubsidenceForcing)
    compute!(forcing.subsidence_vertical_velocity)
    compute!(forcing.averaged_field)
    return nothing
end
