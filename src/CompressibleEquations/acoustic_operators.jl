#####
##### Topology-aware horizontal derivative wrappers used by the acoustic
##### substep loop.
#####
##### Oceananigans exports topology-aware *difference* operators
##### (`δxTᶠᵃᵃ`, `δyTᵃᶠᵃ`, `δxTᶜᵃᵃ`, `δyTᵃᶜᵃ` in
##### `Operators/topology_aware_operators.jl`) that handle `Periodic`
##### wrap-around and `Bounded` walls without reading halos. The
##### acoustic substep kernels also need the matching center → face
##### *derivatives* (∂x at face-x / center-y and the transposed
##### variant). Those derivative variants are not yet upstream, so
##### they are defined locally here as thin wrappers.
#####
##### TODO: upstream `∂xTᶠᶜᶜ`, `∂yTᶜᶠᶜ` to
##### `Oceananigans.Operators.topology_aware_operators`. Once these
##### exist upstream, this file can be deleted.
#####

using Oceananigans.Operators: δxTᶠᵃᵃ, δyTᵃᶠᵃ, Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ

@inline ∂xTᶠᶜᶜ(i, j, k, grid, f, args...) =
    δxTᶠᵃᵃ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline ∂yTᶜᶠᶜ(i, j, k, grid, f, args...) =
    δyTᵃᶠᵃ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
