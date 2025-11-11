module TurbulenceClosures

using Oceananigans
using Oceananigans.Operators:
    # Face-centered difference operators with area metrics
    δxᶠᵃᵃ, δxᶜᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ,
    # Cell volumes (inverse)
    V⁻¹ᶠᶜᶜ, V⁻¹ᶜᶠᶜ, V⁻¹ᶜᶜᶠ, V⁻¹ᶜᶜᶜ,
    # Face areas for q-located fields
    Ax_qᶜᶜᶜ, Ax_qᶠᶠᶜ, Ax_qᶠᶜᶠ,
    Ay_qᶠᶠᶜ, Ay_qᶜᶜᶜ, Ay_qᶜᶠᶠ,
    Az_qᶠᶜᶠ, Az_qᶜᶠᶠ, Az_qᶜᶜᶜ,
    Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ,
    # Interpolator used for ρᵣ at z-faces
    ℑzᵃᵃᶠ

using Oceananigans.TurbulenceClosures:
    AbstractTurbulenceClosure,
    time_discretization,
    _viscous_flux_ux, _viscous_flux_uy, _viscous_flux_uz,
    _viscous_flux_vx, _viscous_flux_vy, _viscous_flux_vz,
    _viscous_flux_wx, _viscous_flux_wy, _viscous_flux_wz,
    _diffusive_flux_x, _diffusive_flux_y, _diffusive_flux_z

import ..AtmosphereModels: ∂ⱼ_𝒯₁ⱼ, ∂ⱼ_𝒯₂ⱼ, ∂ⱼ_𝒯₃ⱼ, ∇_dot_Jᶜ

@inline ∂ⱼ_𝒯₁ⱼ(i, j, k, grid, ρᵣ, ::Nothing, args...) = zero(grid)
@inline ∂ⱼ_𝒯₂ⱼ(i, j, k, grid, ρᵣ, ::Nothing, args...) = zero(grid)
@inline ∂ⱼ_𝒯₃ⱼ(i, j, k, grid, ρᵣ, ::Nothing, args...) = zero(grid)
@inline ∇_dot_Jᶜ(i, j, k, grid, ρᵣ, ::Nothing, args...) = zero(grid)

#####
##### Scalar (tracer) dynamic fluxes: J = ρᵣ τ
#####

# Face flux wrappers that call Oceananigans' kinematic diffusive fluxes and
# multiply by ρᵣ at the appropriate face.

@inline function Jx(i, j, k, grid, ρᵣ, disc, closure, K, id, c, clock, fields, buoyancy)
    return @inbounds ρᵣ[i, j, k] * _diffusive_flux_x(i, j, k, grid, disc, closure, K, id, c, clock, fields, buoyancy)
end

@inline function Jy(i, j, k, grid, ρᵣ, disc, closure, K, id, c, clock, fields, buoyancy)
    return @inbounds ρᵣ[i, j, k] * _diffusive_flux_y(i, j, k, grid, disc, closure, K, id, c, clock, fields, buoyancy)
end

@inline function Jz(i, j, k, grid, ρᵣ, disc, closure, K, id, c, clock, fields, buoyancy)
    ρᵣᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)
    return ρᵣᶠ * _diffusive_flux_z(i, j, k, grid, disc, closure, K, id, c, clock, fields, buoyancy)
end

"""
    ∇_dot_Jᶜ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, K, id, c, clock, fields, buoyancy)

Return the discrete divergence of the dynamic scalar flux `J = ρᵣ τ` at cell
centers, using area-weighted differences divided by cell volume.
Matches Oceananigans' `∇_dot_qᶜ` signature with the additional `ρᵣ`.
"""
@inline function ∇_dot_Jᶜ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, K, id, c, clock, fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (
          δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, Jx, ρᵣ, disc, closure, K, id, c, clock, fields, buoyancy)
        + δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, Jy, ρᵣ, disc, closure, K, id, c, clock, fields, buoyancy)
        + δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, Jz, ρᵣ, disc, closure, K, id, c, clock, fields, buoyancy))
end

#####
##### Momentum dynamic stresses: 𝒯 = ρᵣ τ
#####

# Face stress wrappers for u-momentum
@inline _𝒯ᵤˣ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = @inbounds ρᵣ[i, j, k] * _viscous_flux_ux(i, j, k, grid, disc, closure, K, clock, fields, buoyancy)
@inline _𝒯ᵤʸ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = @inbounds ρᵣ[i, j, k] * _viscous_flux_uy(i, j, k, grid, disc, closure, K, clock, fields, buoyancy)
@inline function _𝒯ᵤᶻ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)
    ρᵣᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)
    return ρᵣᶠ * _viscous_flux_uz(i, j, k, grid, disc, closure, K, clock, fields, buoyancy)
end

@inline function ∂ⱼ_𝒯₁ⱼ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, K, clock, fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (
          δxᶠᵃᵃ(i, j, k, grid, Ax_qᶜᶜᶜ, _𝒯ᵤˣ, ρᵣ, disc, closure, K, clock, fields, buoyancy)
        + δyᵃᶜᵃ(i, j, k, grid, Ay_qᶠᶠᶜ, _𝒯ᵤʸ, ρᵣ, disc, closure, K, clock, fields, buoyancy)
        + δzᵃᵃᶜ(i, j, k, grid, Az_qᶠᶜᶠ, _𝒯ᵤᶻ, ρᵣ, disc, closure, K, clock, fields, buoyancy))
end

# Face stress wrappers for v-momentum
@inline _𝒯ᵥˣ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = @inbounds ρᵣ[i, j, k] * _viscous_flux_vx(i, j, k, grid, disc, closure, K, clock, fields, buoyancy)
@inline _𝒯ᵥʸ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = @inbounds ρᵣ[i, j, k] * _viscous_flux_vy(i, j, k, grid, disc, closure, K, clock, fields, buoyancy)
@inline function _𝒯ᵥᶻ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)
    ρᵣᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)
    return ρᵣᶠ * _viscous_flux_vz(i, j, k, grid, disc, closure, K, clock, fields, buoyancy)
end

@inline function ∂ⱼ_𝒯₂ⱼ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, K, clock, fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (
          δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, _𝒯ᵥˣ, ρᵣ, disc, closure, K, clock, fields, buoyancy)
        + δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, _𝒯ᵥʸ, ρᵣ, disc, closure, K, clock, fields, buoyancy)
        + δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶠᶠ, _𝒯ᵥᶻ, ρᵣ, disc, closure, K, clock, fields, buoyancy))
end

# Face stress wrappers for w-momentum
@inline _𝒯ʷˣ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = @inbounds ρᵣ[i, j, k] * _viscous_flux_wx(i, j, k, grid, disc, closure, K, clock, fields, buoyancy)
@inline _𝒯ʷʸ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = @inbounds ρᵣ[i, j, k] * _viscous_flux_wy(i, j, k, grid, disc, closure, K, clock, fields, buoyancy)
@inline function _𝒯ʷᶻ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)
    ρᵣᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)
    return ρᵣᶠ * _viscous_flux_wz(i, j, k, grid, disc, closure, K, clock, fields, buoyancy)
end

@inline function ∂ⱼ_𝒯₃ⱼ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, K, clock, fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (
          δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶠ, _𝒯ʷˣ, ρᵣ, disc, closure, K, clock, fields, buoyancy)
        + δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶠ, _𝒯ʷʸ, ρᵣ, disc, closure, K, clock, fields, buoyancy)
        + δzᵃᵃᶠ(i, j, k, grid, Az_qᶜᶜᶜ, _𝒯ʷᶻ, ρᵣ, disc, closure, K, clock, fields, buoyancy))
end

# Public aliases for face stress functions (9 components)
@inline 𝒯ᵤˣ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = _𝒯ᵤˣ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)
@inline 𝒯ᵤʸ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = _𝒯ᵤʸ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)
@inline 𝒯ᵤᶻ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = _𝒯ᵤᶻ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)

@inline 𝒯ᵥˣ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = _𝒯ᵥˣ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)
@inline 𝒯ᵥʸ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = _𝒯ᵥʸ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)
@inline 𝒯ᵥᶻ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = _𝒯ᵥᶻ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)

@inline 𝒯ʷˣ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = _𝒯ʷˣ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)
@inline 𝒯ʷʸ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = _𝒯ʷʸ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)
@inline 𝒯ʷᶻ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy) = _𝒯ʷᶻ(i, j, k, grid, ρᵣ, disc, closure, K, clock, fields, buoyancy)

"""
    div_𝒯ᵤ(i, j, k, grid, ρᵣ, τᵤˣ, τᵤʸ, τᵤᶻ, args...)
    div_𝒯ᵥ(i, j, k, grid, ρᵣ, τᵥˣ, τᵥʸ, τᵥᶻ, args...)
    div_𝒯ʷ(i, j, k, grid, ρᵣ, τʷˣ, τʷʸ, τʷᶻ, args...)

Divergence of dynamic stresses for u-, v-, and w-momentum, respectively.
Each takes the corresponding kinematic face-flux functions and multiplies by
`ρᵣ` at the face prior to differencing.

Arguments
- `i, j, k`: Index where divergence is evaluated
- `grid`: Oceananigans grid
- `ρᵣ`: reference density field (centered)
- `τ•ˣ, τ•ʸ, τ•ᶻ`: kinematic stress components at x-, y-, z-faces
- `args...`: Additional arguments passed to the kinematic flux functions

These mirror Oceananigans' closure kernel operators but return per-volume
stress divergences appropriate for Breeze's anelastic equations (i.e., using the
dynamic stress `𝒯 = ρᵣ τ`).
"""

end # module TurbulenceClosures
