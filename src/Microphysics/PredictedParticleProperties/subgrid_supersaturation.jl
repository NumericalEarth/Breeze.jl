#####
##### Subgrid supersaturation: a hook in P3's cloud condensation and evaporation
#####
#
# P3 condenses and evaporates cloud liquid at a rate proportional to the grid-mean vapor excess
# over liquid, qᵛ - qᵛ⁺ˡ. The cell average of the local rate differs from the rate at the cell
# mean by the correlation between the fluctuations of the vapor excess and the presence of
# liquid, so that the cell-averaged rate is driven by
#
#   qᵛ - qᵛ⁺ˡ + Γ,
#
# with Γ the subgrid supersaturation correlation [kg kg⁻¹]. P3 carries no model for Γ. An
# external subgrid model, stored in the `subgrid` slot of P3 (`P3Microphysics(; subgrid = model)`),
# supplies it by extending the three functions below. The subgrid choices of Breeze
# (`AbstractSubgridCorrelation`: the default `NoSubgridCorrelation()`, and `ClosureCloud`, under
# which P3 does not condense) supply no Γ: no value is added and no code path changes.
#
# Γ enters the vapor exchange of cloud liquid only: its diffusional growth and decay, the seed
# mass with which prescribed-number droplets start that growth, and the saturation-adjustment
# caps on those two rates. Rain, ice, aerosol activation and every collection process keep
# their grid-mean form.

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return what the subgrid supersaturation model `subgrid` reads on the grid (for example closure
fields and the conserved variables), as an object that is passed to the P3 kernels and then to
[`subgrid_supersaturation_statistics`](@ref Breeze.Microphysics.PredictedParticleProperties.subgrid_supersaturation_statistics). `nothing` without a subgrid model.
"""
subgrid_supersaturation_fields(::AbstractSubgridCorrelation, model) = nothing

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid statistics of cell `(i, j, k)` that the subgrid supersaturation model
`subgrid` forms from `fields`, the result of [`subgrid_supersaturation_fields`](@ref Breeze.Microphysics.PredictedParticleProperties.subgrid_supersaturation_fields). The
statistics are passed, with the thermodynamic state of the cell, to
[`subgrid_supersaturation`](@ref Breeze.Microphysics.PredictedParticleProperties.subgrid_supersaturation). `nothing` without a subgrid model.
"""
@inline subgrid_supersaturation_statistics(i, j, k, grid, ::AbstractSubgridCorrelation, fields) = nothing

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid supersaturation correlation Γ [kg kg⁻¹] of a cell: the amount added to the
grid-mean vapor excess `qᵛ - qᵛ⁺ˡ` in the condensation and evaporation of cloud liquid. A
subgrid model extends this function for its own type and `statistics`; the arguments are the
vapor, saturation and cloud liquid mass fractions, the temperature `T`, pressure `P` and density
`ρ` that P3 evaluates its rates with, and the thermodynamic `constants`. Without a subgrid model
the result is `nothing` and P3 uses the grid-mean vapor excess unchanged.

```jldoctest
julia> using Breeze.Microphysics.PredictedParticleProperties: subgrid_supersaturation, cloud_vapor_excess

julia> using Breeze.AtmosphereModels: NoSubgridCorrelation

julia> Γ = subgrid_supersaturation(NoSubgridCorrelation(), nothing, 1e-2, 1.1e-2, 0.0, 285.0, 9e4, 1.1, nothing)

julia> cloud_vapor_excess(Γ, -1e-3)
-0.001

julia> cloud_vapor_excess(0.5, -1.0)
-0.5
```
"""
@inline subgrid_supersaturation(::AbstractSubgridCorrelation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ, constants) = nothing

# The vapor excess, and the vapor, that cloud liquid sees
@inline cloud_vapor_excess(::Nothing, vapor_excess) = vapor_excess
@inline cloud_vapor_excess(Γ, vapor_excess) = vapor_excess + Γ

# The seed mass of prescribed-number droplets starts cloud condensation, so it sees Γ; aerosol
# activation, which predicts the droplet number, keeps the grid-mean vapor.
@inline seed_vapor(aerosol, Γ, qᵛ) = qᵛ
@inline seed_vapor(::Nothing, ::Nothing, qᵛ) = qᵛ
@inline seed_vapor(::Nothing, Γ, qᵛ) = qᵛ + Γ
