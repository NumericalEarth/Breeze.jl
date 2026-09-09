using ..AtmosphereModels: AtmosphereModels
using Oceananigans: Average, Field, set!, compute!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: AbstractField
using Oceananigans.Grids: AbstractGrid, Center, Face, Flat
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

# `w` and `ϕ` are reduced (x, y)-averaged fields on an ordinary grid, which ignore `i, j`, and full
# fields in a column ensemble, where every column carries its own subsidence and its own profile.
@inline w_dz_ϕᵃᵃᶠ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

# The face values of wˢ ∂z ϕ̄ reconstructed to the cell centers. Interior cells average the faces
# above and below. The boundary cells cannot use the boundary face, whose gradient would reach into
# the halo, and must not use the adjacent interior face alone: with subsidence into the lid that is
# a downwind difference, ∂ₜ(ϕ̄ₙ - ϕ̄ₙ₋₁) ∝ +|wˢ| (ϕ̄ₙ - ϕ̄ₙ₋₁), which doubles the gradient at the
# lid every 2Δz/|wˢ| — an hour for 1 cm s⁻¹ and 20 m — and drives the top cell away from the
# column. Averaging the two interior faces nearest the boundary gives the boundary cell the same
# combination of gradients its neighbour sees, so the boundary gradient has no tendency of its own
# and is carried neutrally, as an upwind scheme with a linearly extrapolated ghost value would.
@inline function ℑzbᵃᵃᶜ(i, j, k, grid, w_dz_ϕᵃᵃᶠ, wˢ, ϕ_avg)
    w_dz_ϕ⁺⁺ = w_dz_ϕᵃᵃᶠ(i, j, k+2, grid, wˢ, ϕ_avg)
    w_dz_ϕ⁺ = w_dz_ϕᵃᵃᶠ(i, j, k+1, grid, wˢ, ϕ_avg)
    w_dz_ϕᵏ = w_dz_ϕᵃᵃᶠ(i, j, k, grid, wˢ, ϕ_avg)
    w_dz_ϕ⁻ = w_dz_ϕᵃᵃᶠ(i, j, k-1, grid, wˢ, ϕ_avg)
    interior = (w_dz_ϕ⁺ + w_dz_ϕᵏ) / 2
    at_top = (w_dz_ϕᵏ + w_dz_ϕ⁻) / 2
    at_bottom = (w_dz_ϕ⁺ + w_dz_ϕ⁺⁺) / 2
    top = k == grid.Nz
    bottom = k == 1
    return ifelse(top, at_top, ifelse(bottom, at_bottom, interior))
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

horizontally_averaged(specific_field, grid) = Average(specific_field, dims=(1, 2)) |> Field
horizontally_averaged(specific_field, ::AbstractGrid{<:Any, <:Flat, <:Flat}) = specific_field

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

    # The horizontal mean of a single column is the column itself, and in a column ensemble the
    # columns are independent, so the subsidence acts on each column's own profile there.
    averaged_field = horizontally_averaged(specific_field, grid)

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
