module Advection

export div_ρUc

using Oceananigans.Advection:
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z,
    BoundsPreservingWENO,
    bounded_tracer_flux_divergence_x,
    bounded_tracer_flux_divergence_y,
    bounded_tracer_flux_divergence_z

using Adapt: Adapt
using Oceananigans.Advection: WENO, explicit_velocity_scaleᶜᶜᶠ
using Oceananigans.Utils: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.Fields: ZeroField
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Oceananigans.TimeSteppers: time_discretization

using ..AtmosphereModels: AtmosphereModels, div_ρUc

# Simple wrappers: interpolate ρ to face, multiply existing flux
@inline tracer_mass_flux_x(i, j, k, grid, ρ, args...) =
    ℑxᶠᵃᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_x(i, j, k, grid, args...)

@inline tracer_mass_flux_y(i, j, k, grid, ρ, args...) =
    ℑyᵃᶠᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_y(i, j, k, grid, args...)

@inline tracer_mass_flux_z(i, j, k, grid, ρ, args...) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ) * _advective_tracer_flux_z(i, j, k, grid, args...)

# Main operator
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection, ρ, U, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (
        δxᶜᵃᵃ(i, j, k, grid, tracer_mass_flux_x, ρ, advection, U.u, c) +
        δyᵃᶜᵃ(i, j, k, grid, tracer_mass_flux_y, ρ, advection, U.v, c) +
        δzᵃᵃᶜ(i, j, k, grid, tracer_mass_flux_z, ρ, advection, U.w, c))
end

# Fallback for nothing advection
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::Nothing, ρ, U, c) = zero(grid)
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::BoundsPreservingWENO, ρ, U, ::ZeroField) = zero(grid)

# Is this immersed-boundary safe without having to extend it in ImmersedBoundaries.jl? I think so... (velocity on immmersed boundaries is masked to 0)
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection::BoundsPreservingWENO, ρ, U, c)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, U.w, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# A bounds-preserving WENO whose vertical time discretization is adaptive-implicit.
const BoundsPreservingAVIDWENO = WENO{<:Any, <:Any, <:Any, <:AdaptiveVerticallyImplicitDiscretization, <:Tuple}

# Lazily s-scaled face velocity: indexing yields wᵉ = s·w, the explicit fraction of the IMEX
# vertical-advection split, so the bounds-preserving flux functions consume the explicit
# velocity without duplicating their reconstruction (issue #913).
struct ExplicitVerticalVelocity{G, S, T, W}
    grid :: G
    scheme :: S
    td :: T
    w :: W
end

Adapt.adapt_structure(to, v::ExplicitVerticalVelocity) =
    ExplicitVerticalVelocity(Adapt.adapt(to, v.grid), Adapt.adapt(to, v.scheme),
                             Adapt.adapt(to, v.td), Adapt.adapt(to, v.w))

@inline Base.getindex(v::ExplicitVerticalVelocity, i, j, k) =
    @inbounds explicit_velocity_scaleᶜᶜᶠ(i, j, k, v.grid, v.scheme, v.td, v.w) * v.w[i, j, k]

# Disambiguates against the `ZeroField` shortcut above: a zero tracer advects to zero
# regardless of the vertical time discretization.
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::BoundsPreservingAVIDWENO, ρ, U, ::ZeroField) = zero(grid)

# The bounded path never routed through the AVID flux scaling, so it transported the full
# explicit flux AND received the (1 - s) implicit remainder — 1 + (1 - s) total (issue #913).
# The horizontal fluxes stay fully explicit under AVID, matching the plain-WENO path.
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection::BoundsPreservingAVIDWENO, ρ, U, c)
    wᵉ = ExplicitVerticalVelocity(grid, advection, time_discretization(advection), U.w)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, wᵉ, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

end # module
