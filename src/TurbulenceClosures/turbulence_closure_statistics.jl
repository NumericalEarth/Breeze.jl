#####
##### The statistics of the vapor excess from the turbulence closure
#####

"""
$(TYPEDEF)

The statistics of the vapor excess over liquid from the model's turbulence closure (see
[`AtmosphereModels.supersaturation_statistics`](@ref)): the components of the double-Gaussian
subgrid distribution of a `TKEBasedTurbulenceClosure` with a `subgrid_distribution` (such as a
`SimplifiedHigherOrderClosure`), as Gaussian components `(; a, μ, σ)` of the vapor excess. A
closure without a subgrid distribution supplies no spread (`σ = 0`): the grid-mean rate.

```jldoctest
julia> using Breeze

julia> TurbulenceClosureStatistics()
TurbulenceClosureStatistics()
```
"""
struct TurbulenceClosureStatistics <: AtmosphereModels.AbstractSupersaturationStatistics end

# A closure without subgrid statistics: no spread
@inline function AtmosphereModels.supersaturation_statistics(::TurbulenceClosureStatistics, i, j, k, grid,
                                                             closure, closure_fields, model_fields)
    FT = grid_float_type(grid)
    return (; μ = zero(FT), σ = zero(FT), skewness = zero(FT), f⁺ = zero(FT))
end
