using Oceananigans: Average, Field, set!, compute!
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: AbstractField
using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### Subsidence forcing types (unmaterialized stubs)
#####

struct SubsidenceForcing{W, R, A}
    subsidence_vertical_velocity :: W
    reference_density :: R
    averaged_field :: A
end

Adapt.adapt_structure(to, sf::SubsidenceForcing) =
    SubsidenceForcing(Adapt.adapt(to, sf.subsidence_vertical_velocity),
                      Adapt.adapt(to, sf.reference_density),
                      Adapt.adapt(to, sf.averaged_field))

"""
    $(TYPEDSIGNATURES)

Forcing that represents large-scale subsidence advecting horizontally-averaged
fields downward:

```math
F_{ρ ϕ} = - ρᵣ wˢ ∂_z \\overline{ϕ}
```

where ``wˢ`` is the `subsidence_vertical_velocity`, ``ρᵣ`` is the reference density,
and ``\\overline{ϕ}`` is the horizontal average of the field being forced.

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
forcing = (; ρe=subsidence, ρqᵗ=subsidence)

model = AtmosphereModel(grid; forcing)

model.forcing.ρe

# output
SubsidenceForcing with wˢ: 1×1×76 Field{Nothing, Nothing, Face} reduced over dims = (1, 2) on RectilinearGrid on CPU
└── averaged_field: 1×1×75 Field{Nothing, Nothing, Center} reduced over dims = (1, 2) on RectilinearGrid on CPU
```
"""
SubsidenceForcing(wˢ) = SubsidenceForcing(wˢ, nothing, nothing)

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
##### Materialized subsidence forcing
#####

# Kernel function for subsidence forcing
@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

@inline function (forcing::SubsidenceForcing)(i, j, k, grid, clock, fields)
    wˢ = forcing.subsidence_vertical_velocity
    ϕ_avg = forcing.averaged_field
    ρᵣ = forcing.reference_density
    w_dz_ϕ_avg = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, ϕ_avg)
    return @inbounds - ρᵣ[i, j, k] * w_dz_ϕ_avg
end

#####
##### Materialization function for subsidence forcing
#####

# This is called from AtmosphereModels.atmosphere_model_forcing
# The `averaged_field` is determined by the field name (e.g., :ρu → u, :ρθ → θ)
# and passed in from atmosphere_model_forcing

# Strip the ρ prefix from density variable names
# e.g., :ρu → :u, :ρθ → :θ, :ρe → :e
strip_density_prefix(name::Symbol) = Symbol(collect(string(name))[2:end]...)

function materialize_atmosphere_model_forcing(forcing::SubsidenceForcing, field, name, model_field_names, context)
    grid = field.grid

    if forcing.subsidence_vertical_velocity isa AbstractField
        wˢ = forcing.subsidence_vertical_velocity
    else
        ibc = ImpenetrableBoundaryCondition()
        loc = (nothing, nothing, Face())
        bcs = FieldBoundaryConditions(grid, loc, bottom=ibc, top=ibc)
        wˢ = Field{Nothing, Nothing, Face}(grid, boundary_conditions=bcs)
        set!(wˢ, forcing.subsidence_vertical_velocity)
        fill_halo_regions!(wˢ)
    end

    ρᵣ = context.reference_density

    if name ∈ (:ρu, :ρv, :ρw, :ρθ, :ρe)
        specific_name = strip_density_prefix(name)
        specific_field = context.specific_fields[specific_name]
    else
        specific_field = field / ρᵣ |> Field
    end

    averaged_field = Average(specific_field, dims=(1, 2)) |> Field

    return SubsidenceForcing(wˢ, ρᵣ, averaged_field)
end
    
#####
##### compute_forcing! for subsidence forcing
#####

function compute_forcing!(forcing::SubsidenceForcing)
    compute!(forcing.subsidence_vertical_velocity)
    compute!(forcing.averaged_field)
    return nothing
end

