#####
##### Subgrid statistics: the closure → microphysics interface
#####
#
# A turbulence closure may know more about a cell than its mean: SHOC, for example, diagnoses a
# cloud fraction and a cloud liquid from an assumed subgrid distribution. The microphysics never
# reads closure fields. It declares, through `subgrid_correlation(microphysics)`, which subgrid
# information it uses, and receives it per cell as an isbits `AbstractSubgridStatistics` built by
# `subgrid_statisticsᶜᶜᶜ`, which closures extend. `NoSubgridCorrelation` and its
# `DiracStatistics` (no subgrid variability) select the grid-mean code of the microphysics,
# unchanged.

"""
$(TYPEDEF)

Abstract supertype for the per-cell subgrid statistics that a turbulence closure supplies to the
microphysics. See [`DiracStatistics`](@ref).
"""
abstract type AbstractSubgridStatistics end

"""
$(TYPEDEF)

No subgrid variability: the microphysics evaluates its rates at the grid-mean state, exactly as
without a subgrid model.

```jldoctest
julia> using Breeze.AtmosphereModels: DiracStatistics

julia> DiracStatistics()
DiracStatistics()
```
"""
struct DiracStatistics <: AbstractSubgridStatistics end


"""
$(TYPEDEF)

Abstract supertype for the choice a microphysics scheme makes of the subgrid information it uses.
See [`NoSubgridCorrelation`](@ref).
"""
abstract type AbstractSubgridCorrelation end

"""
$(TYPEDEF)

No subgrid information: [`subgrid_statisticsᶜᶜᶜ`](@ref) returns [`DiracStatistics`](@ref) and the
microphysics runs its grid-mean code unchanged.

```jldoctest
julia> using Breeze.AtmosphereModels: NoSubgridCorrelation

julia> NoSubgridCorrelation()
NoSubgridCorrelation()
```
"""
struct NoSubgridCorrelation <: AbstractSubgridCorrelation end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid information `microphysics` uses; [`NoSubgridCorrelation`](@ref) by default.
"""
subgrid_correlation(microphysics) = NoSubgridCorrelation()

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the [`AbstractSubgridStatistics`](@ref) of cell `(i, j, k)` that the subgrid choice
`correlation` of the microphysics asks of the turbulence `closure` and its `closure_fields`.
Closures that supply subgrid statistics add methods for their correlation type; any other closure
supplies no cloud.
"""
@inline subgrid_statisticsᶜᶜᶜ(i, j, k, grid, ::NoSubgridCorrelation, closure, closure_fields) = DiracStatistics()

