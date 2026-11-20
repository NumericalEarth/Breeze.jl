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
microphysics. See [`DiracStatistics`](@ref) and [`ClosureCloudStatistics`](@ref).
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

The subgrid cloud of one cell as a turbulence closure diagnosed it: the closure, not the
microphysics, owns the cloud fraction and the cloud liquid. See [`ClosureCloud`](@ref).

$(TYPEDFIELDS)

```jldoctest
julia> using Breeze.AtmosphereModels: ClosureCloudStatistics

julia> ClosureCloudStatistics(0.2, 1e-5)
ClosureCloudStatistics{Float64}(0.2, 1.0e-5)
```
"""
struct ClosureCloudStatistics{FT} <: AbstractSubgridStatistics
    "Cloud fraction"
    cloud_fraction :: FT
    "Cell-mean cloud liquid mass fraction that the closure diagnosed, kg kg⁻¹"
    cloud_liquid :: FT
end

ClosureCloudStatistics(cloud_fraction, cloud_liquid) =
    ClosureCloudStatistics(promote(cloud_fraction, cloud_liquid)...)

"""
$(TYPEDEF)

Abstract supertype for the choice a microphysics scheme makes of the subgrid information it uses.
See [`NoSubgridCorrelation`](@ref) and [`ClosureCloud`](@ref).
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
$(TYPEDEF)

Subgrid cloud owned by the turbulence closure. A closure that diagnoses its own subgrid cloud
(the assumed-distribution cloud of `SimplifiedHigherOrderClosure`) supplies the cloud fraction
and sets the cloud liquid of the model once per time step; the microphysics forms and evaporates
no cloud liquid of its own and owns precipitation only. Autoconversion and accretion act on the
in-cloud liquid, the cloud liquid over the cloud fraction, and are weighted by the cloud
fraction. A closure without subgrid cloud supplies a cloud fraction of zero.

$(TYPEDFIELDS)

```jldoctest
julia> using Breeze.AtmosphereModels: ClosureCloud

julia> ClosureCloud(Float64)
ClosureCloud{Float64}(1.0, 0.01)
```
"""
struct ClosureCloud{FT} <: AbstractSubgridCorrelation
    "Factor multiplying the in-cloud autoconversion rate, for the variability of the liquid within the cloud"
    autoconversion_enhancement :: FT
    "Floor on the cloud fraction used to form in-cloud means"
    minimum_cloud_fraction :: FT
end

# Walk the fields so that traced numbers (`Reactant.to_rarray`) are converted for the kernel.
Adapt.adapt_structure(to, correlation::ClosureCloud) =
    ClosureCloud(adapt(to, correlation.autoconversion_enhancement),
                 adapt(to, correlation.minimum_cloud_fraction))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`ClosureCloud`](@ref) with float type `FT`. The default `autoconversion_enhancement = 1`
assumes uniform liquid within the cloud; a gamma distribution of in-cloud liquid with shape ν
and an autoconversion rate ∝ qᵖ gives Γ(ν + p) / (Γ(ν) νᵖ), about 3.02 for ν = 1 and p = 2.47.
"""
function ClosureCloud(FT::DataType = Oceananigans.defaults.FloatType;
                      autoconversion_enhancement = 1,
                      minimum_cloud_fraction = 0.01)
    return ClosureCloud(convert(FT, autoconversion_enhancement), convert(FT, minimum_cloud_fraction))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid information `microphysics` uses; [`NoSubgridCorrelation`](@ref) by default.
"""
subgrid_correlation(microphysics) = NoSubgridCorrelation()

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the [`AbstractSubgridStatistics`](@ref) of cell `(i, j, k)` that the subgrid choice
`correlation` of the microphysics asks of the turbulence `closure` and its `closure_fields`.
Closures that diagnose subgrid cloud add a method for [`ClosureCloud`](@ref); any other closure
supplies no cloud.
"""
@inline subgrid_statisticsᶜᶜᶜ(i, j, k, grid, ::NoSubgridCorrelation, closure, closure_fields) = DiracStatistics()

@inline subgrid_statisticsᶜᶜᶜ(i, j, k, grid, ::ClosureCloud, closure, closure_fields) =
    ClosureCloudStatistics(zero(grid), zero(grid))
