#####
##### Topology-aware horizontal derivative wrappers used by the acoustic
##### substep loop.
#####
##### Oceananigans exports topology-aware *difference* operators
##### (`δxTᶠᵃᵃ`, `δyTᵃᶠᵃ`, `δxTᶜᵃᵃ`, `δyTᵃᶜᵃ` in
##### `Operators/topology_aware_operators.jl`) that handle `Periodic`
##### wrap-around and `Bounded` walls without reading halos. The
##### acoustic substep kernels also need the matching center → face
##### *derivatives* (∂x at face-x / center-y / center-z and the
##### transposed y variant). Those derivative variants are not yet
##### upstream, so they are defined locally here as thin wrappers.
#####
##### The Oceananigans difference operators with center result
##### (`δxTᶜᵃᵃ`, `δyTᵃᶜᵃ`) are used directly inside the predictor
##### kernel — no extra wrapper is needed.
#####
##### TODO: upstream `∂xTᶠᶜᶜ`, `∂yTᶜᶠᶜ` to
##### `Oceananigans.Operators.topology_aware_operators`. Once these
##### exist upstream, this file can be deleted.
#####

using Oceananigans.Operators: δxTᶠᵃᵃ, δyTᵃᶠᵃ, Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ

# Topology-aware center → face derivatives. At `Periodic` boundaries
# they read the wrap-around value directly; at `Bounded` boundaries they
# return zero (NoFlux), matching the boundary condition for the
# linearised perturbation pressure gradient and the Klemp-2018
# divergence-damping increment. Connected-topology dispatch comes
# automatically through the underlying `δxTᶠᵃᵃ` / `δyTᵃᶠᵃ`.
#
# Single method per direction matches the Oceananigans topology-aware
# operator convention (no `::Function` narrowing): the underlying
# `δxTᶠᵃᵃ` / `δyTᵃᶠᵃ` already dispatch internally on whether the
# first argument is an `AbstractArray` or a callable, so the wrapper
# does not need to discriminate.
@inline ∂xTᶠᶜᶜ(i, j, k, grid, f, args...) =
    δxTᶠᵃᵃ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline ∂yTᶜᶠᶜ(i, j, k, grid, f, args...) =
    δyTᵃᶠᵃ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
