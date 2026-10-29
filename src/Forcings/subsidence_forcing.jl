using ..AtmosphereModels: AtmosphereModels
using Oceananigans: Average, Field, set!, compute!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: AbstractField
using Oceananigans.Grids: AbstractGrid, Center, Face, Flat
using Oceananigans.Operators: ∂zᶜᶜᶠ, Δzᶜᶜᶜ
using Oceananigans.Advection: Centered, UpwindBiased, AbstractUpwindBiasedAdvectionScheme, _biased_interpolate_zᵃᵃᶠ, bias
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### Subsidence forcing
#####

struct SubsidenceForcing{W, A, S, B}
    subsidence_vertical_velocity :: W
    averaged_field :: A
    advection :: S
    inflow_lower_bound :: B
end

Adapt.adapt_structure(to, sf::SubsidenceForcing) =
    SubsidenceForcing(Adapt.adapt(to, sf.subsidence_vertical_velocity),
                      Adapt.adapt(to, sf.averaged_field),
                      Adapt.adapt(to, sf.advection),
                      Adapt.adapt(to, sf.inflow_lower_bound))

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
- `advection`: the discretization of ``w^s ∂_z \\overline{ϕ}``. The default `UpwindBiased(order = 1)`
  damps ``2Δz`` modes and does not undershoot at sharp gradients. `Centered()` cannot see a ``2Δz`` mode,
  so it persists; higher-order upwind schemes damp it but may over- or undershoot.

With an upwind scheme every cell, including the two boundary cells, is updated from its own face
fluxes. Air leaving the column through a boundary face carries the boundary cell's value. Air
entering it carries the value extrapolated linearly from the two cells inside, so a linear
profile crosses the boundary undistorted; for mass fractions such as moisture that value is
bounded below by zero, so subsidence never brings negative moisture into the column.

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
└── advection: UpwindBiased(order=1)
```
"""
SubsidenceForcing(wˢ; advection = UpwindBiased(order = 1)) = SubsidenceForcing(wˢ, nothing, advection, nothing)

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
# Only the interior faces 2:Nz are used: a two-cell column has the single face 2, and a one-cell
# column has no gradient at all.
@inline function ℑzbᵃᵃᶜ(i, j, k, grid, w_dz_ϕᵃᵃᶠ, wˢ, ϕ_avg)
    Nz = grid.Nz
    kᵉ = max(2, min(k, Nz - 1))
    k⁻ = clamp(kᵉ, 2, Nz)
    k⁺ = clamp(kᵉ + 1, 2, Nz)
    w_dz_ϕ = (w_dz_ϕᵃᵃᶠ(i, j, k⁻, grid, wˢ, ϕ_avg) + w_dz_ϕᵃᵃᶠ(i, j, k⁺, grid, wˢ, ϕ_avg)) / 2
    return ifelse(Nz < 2, zero(w_dz_ϕ), w_dz_ϕ)
end

# wˢ ∂z ϕ̄ at the cell center with the centered discretization above
@inline w_dz_ϕᶜᶜᶜ(i, j, k, grid, ::Centered, wˢ, ϕ_avg, lower_bound) = ℑzbᵃᵃᶜ(i, j, k, grid, w_dz_ϕᵃᵃᶠ, wˢ, ϕ_avg)

# The value of ϕ̄ carried through a boundary face. Outflow carries the boundary cell's own value.
# Inflow carries the value extrapolated linearly from the two cells inside, bounded below for mass
# fractions; a one-cell column has no gradient to extrapolate.
@inline function boundary_upwind_value(i, j, k, grid, wᵏ, ϕ, lower_bound)
    Nz = grid.Nz
    top = k == Nz + 1
    boundary_cell = ifelse(top, Nz, 1)
    inner_cell = clamp(ifelse(top, Nz - 1, 2), 1, Nz)
    @inbounds ϕᵇ = ϕ[i, j, boundary_cell]
    @inbounds Δϕ = ϕᵇ - ϕ[i, j, inner_cell]
    inflow = ifelse(top, wᵏ < 0, wᵏ > 0)
    return ifelse(inflow, bounded_below(ϕᵇ + Δϕ, lower_bound), ϕᵇ)
end

@inline bounded_below(ϕ, ::Nothing) = ϕ
@inline bounded_below(ϕ, lower_bound) = max(lower_bound, ϕ)

# The upwind-biased flux wˢ ϕ̃ at a face, with ϕ̄ reconstructed from the upwind side of an interior
# face and given explicitly at the two boundary faces
@inline function w_ϕᵃᵃᶠ(i, j, k, grid, scheme, w, ϕ, lower_bound)
    @inbounds wᵏ = w[i, j, k]
    ϕ̃ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, bias(wᵏ), ϕ)
    ϕᵇ = boundary_upwind_value(i, j, k, grid, wᵏ, ϕ, lower_bound)
    boundary = (k == 1) | (k == grid.Nz + 1)
    return wᵏ * ifelse(boundary, ϕᵇ, ϕ̃)
end

# wˢ ∂z ϕ̄ = ∂z(wˢ ϕ̄) - ϕ̄ ∂z wˢ in flux form, so a divergent wˢ is not a source. Every cell,
# the boundary cells included, uses its own two face fluxes.
@inline function w_dz_ϕᶜᶜᶜ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, wˢ, ϕ_avg, lower_bound)
    F⁺ = w_ϕᵃᵃᶠ(i, j, k+1, grid, scheme, wˢ, ϕ_avg, lower_bound)
    F⁻ = w_ϕᵃᵃᶠ(i, j, k, grid, scheme, wˢ, ϕ_avg, lower_bound)
    @inbounds ϕᵏ = ϕ_avg[i, j, k]
    @inbounds δw = wˢ[i, j, k+1] - wˢ[i, j, k]
    return (F⁺ - F⁻ - ϕᵏ * δw) / Δzᶜᶜᶜ(i, j, k, grid)
end

@inline function (forcing::SubsidenceForcing)(i, j, k, grid, clock, fields)
    wˢ = forcing.subsidence_vertical_velocity
    ϕ_avg = forcing.averaged_field
    return - w_dz_ϕᶜᶜᶜ(i, j, k, grid, forcing.advection, wˢ, ϕ_avg, forcing.inflow_lower_bound)
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

    # Fields weighted by the total density are mass fractions (moisture, microphysical moments,
    # tracers), which must not be brought in negative through an inflow boundary
    mass_fraction = Symbol(:ρ, name) ∉ context.continuity_density_names
    inflow_lower_bound = mass_fraction ? 0 : nothing

    return SubsidenceForcing(wˢ, averaged_field, forcing.advection, inflow_lower_bound)
end

#####
##### compute_forcing! for subsidence forcing
#####

function AtmosphereModels.compute_forcing!(forcing::SubsidenceForcing)
    compute!(forcing.subsidence_vertical_velocity)
    compute!(forcing.averaged_field)
    return nothing
end
