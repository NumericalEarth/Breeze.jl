using ..AtmosphereModels: AtmosphereModels
using Oceananigans: Average, Field, set!, compute!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: AbstractField
using Oceananigans.Grids: AbstractGrid, Center, Face, Flat
using Oceananigans.Operators: ∂zᶜᶜᶠ, Δzᶜᶜᶜ
using Oceananigans.Advection: Centered, AbstractUpwindBiasedAdvectionScheme, _biased_interpolate_zᵃᵃᶠ, bias
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### Subsidence forcing
#####

struct SubsidenceForcing{W, A, S}
    subsidence_vertical_velocity :: W
    averaged_field :: A
    advection :: S
end

Adapt.adapt_structure(to, sf::SubsidenceForcing) =
    SubsidenceForcing(Adapt.adapt(to, sf.subsidence_vertical_velocity),
                      Adapt.adapt(to, sf.averaged_field),
                      Adapt.adapt(to, sf.advection))

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

# Arguments
- `wˢ`: Either a function of `z` specifying the subsidence velocity profile,
        or a `Field` containing the subsidence velocity.

# Keyword arguments
- `advection`: the discretization of ``w^s ∂_z \\overline{ϕ}``. The default `Centered()` averages the
  centered face gradients ``w^s δ_z \\overline{ϕ}`` to the cell — a centered difference that does not see a
  two-cell (``2Δz``) mode, which subsidence therefore cannot remove. An upwind-biased scheme such as
  `UpwindBiased(order = 1)` or `WENO(order = 5)` instead reconstructs ``\\overline{ϕ}`` at the faces from the
  upwind side and forms ``w^s ∂_z \\overline{ϕ} = ∂_z(w^s \\overline{ϕ}) - \\overline{ϕ} \\, ∂_z w^s``, which damps
  that mode and does not undershoot at sharp gradients; first-order upwind is the discretization most
  large-eddy simulation codes use for prescribed subsidence.

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
├── averaged_field: 1×1×75 Field{Nothing, Nothing, Center} reduced over dims = (1, 2) on RectilinearGrid on CPU
└── advection: Centered(order=2)
```
"""
SubsidenceForcing(wˢ; advection = Centered()) = SubsidenceForcing(wˢ, nothing, advection)

function Base.summary(forcing::SubsidenceForcing)
    wˢ = forcing.subsidence_vertical_velocity
    return string("SubsidenceForcing with wˢ: ", prettysummary(wˢ))
end

function Base.show(io::IO, forcing::SubsidenceForcing)
    print(io, summary(forcing))
    if !isnothing(forcing.averaged_field)
        print(io, '\n')
        print(io, "├── averaged_field: ", prettysummary(forcing.averaged_field))
    end
    print(io, '\n')
    print(io, "└── advection: ", summary(forcing.advection))
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

# wˢ ∂z ϕ̄ at the cell center with the centered discretization above
@inline w_dz_ϕᶜᶜᶜ(i, j, k, grid, ::Centered, wˢ, ϕ_avg) = ℑzbᵃᵃᶜ(i, j, k, grid, w_dz_ϕᵃᵃᶠ, wˢ, ϕ_avg)

# The upwind-biased flux wˢ ϕ̃ at a face, with ϕ̄ reconstructed from the upwind side of the face
@inline function w_ϕᵃᵃᶠ(i, j, k, grid, scheme, w, ϕ)
    @inbounds wᵏ = w[i, j, k]
    return wᵏ * _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, bias(wᵏ), ϕ)
end

# wˢ ∂z ϕ̄ = ∂z(wˢ ϕ̄) - ϕ̄ ∂z wˢ with upwind-biased face values: the advective form, so that a
# prescribed wˢ with ∂z wˢ ≠ 0 does not act as a source, built from the flux form so that the
# reconstruction is the one the scheme defines. The boundary faces use the boundary-adjacent
# reconstruction the scheme provides for `Bounded` directions.
@inline function w_dz_ϕᶜᶜᶜ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, wˢ, ϕ_avg)
    F⁺ = w_ϕᵃᵃᶠ(i, j, k+1, grid, scheme, wˢ, ϕ_avg)
    F⁻ = w_ϕᵃᵃᶠ(i, j, k, grid, scheme, wˢ, ϕ_avg)
    @inbounds ϕᵏ = ϕ_avg[i, j, k]
    @inbounds δw = wˢ[i, j, k+1] - wˢ[i, j, k]
    return (F⁺ - F⁻ - ϕᵏ * δw) / Δzᶜᶜᶜ(i, j, k, grid)
end

@inline function (forcing::SubsidenceForcing)(i, j, k, grid, clock, fields)
    wˢ = forcing.subsidence_vertical_velocity
    ϕ_avg = forcing.averaged_field
    return - w_dz_ϕᶜᶜᶜ(i, j, k, grid, forcing.advection, wˢ, ϕ_avg)
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

    return SubsidenceForcing(wˢ, averaged_field, forcing.advection)
end

#####
##### compute_forcing! for subsidence forcing
#####

function AtmosphereModels.compute_forcing!(forcing::SubsidenceForcing)
    compute!(forcing.subsidence_vertical_velocity)
    compute!(forcing.averaged_field)
    return nothing
end
