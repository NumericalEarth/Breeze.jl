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

@inline ∂ⱼ_𝒯₁ⱼ(i, j, k, grid, ρ, ::Nothing, args...) = zero(grid)
@inline ∂ⱼ_𝒯₂ⱼ(i, j, k, grid, ρ, ::Nothing, args...) = zero(grid)
@inline ∂ⱼ_𝒯₃ⱼ(i, j, k, grid, ρ, ::Nothing, args...) = zero(grid)
@inline ∇_dot_Jᶜ(i, j, k, grid, ρ, ::Nothing, args...) = zero(grid)

#####
##### Scalar (tracer) dynamic fluxes: J = ρᵣ τ
#####

# Face flux wrappers that call Oceananigans' kinematic diffusive fluxes and
# multiply by ρᵣ at the appropriate face.

@inline Jᶜx(i, j, k, grid, ρ, args...) = ℑxᶠᶜᶜ(i, j, k, grid, ρ) * _diffusive_flux_x(i, j, k, grid, args...)
@inline Jᶜy(i, j, k, grid, ρ, args...) = ℑyᶜᶠᶜ(i, j, k, grid, ρ) * _diffusive_flux_y(i, j, k, grid, args...)
@inline Jᶜz(i, j, k, grid, ρ, args...) = ℑzᶜᶜᶠ(i, j, k, grid, ρ) * _diffusive_flux_z(i, j, k, grid, args...)

"""
    ∇_dot_Jᶜ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, K, id, c, clock, fields, buoyancy)

Return the discrete divergence of the dynamic scalar flux `J = ρᵣ τ` at cell
centers, using area-weighted differences divided by cell volume.
Matches Oceananigans' `∇_dot_qᶜ` signature with the additional `ρᵣ`.
"""
@inline function ∇_dot_Jᶜ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, closure_fields, clock, fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (
          δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, Jᶜx, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, Jᶜy, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, Jᶜz, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy))
end

#####
##### Momentum dynamic stresses: 𝒯 = ρᵣ τ
#####

# Face stress wrappers for u-momentum
@inline 𝒯_ux(i, j, k, grid, ρ, args...) = @inbounds ρ[i, j, k]     * _viscous_flux_ux(i, j, k, grid, args...)
@inline 𝒯_uy(i, j, k, grid, ρ, args...) = ℑxyᶠᶠᶜ(i, j, k, grid, ρ) * _viscous_flux_uy(i, j, k, grid, args...)
@inline 𝒯_uz(i, j, k, grid, ρ, args...) = ℑxzᶠᶜᶠ(i, j, k, grid, ρ) * _viscous_flux_uz(i, j, k, grid, args...)

@inline 𝒯_vx(i, j, k, grid, ρ, args...) = ℑxyᶠᶠᶜ(i, j, k, grid, ρ) * _viscous_flux_vx(i, j, k, grid, args...)
@inline 𝒯_vy(i, j, k, grid, ρ, args...) = @inbounds ρ[i, j, k]     * _viscous_flux_vy(i, j, k, grid, args...)
@inline 𝒯_vz(i, j, k, grid, ρ, args...) = ℑyzᶜᶠᶠ(i, j, k, grid, ρ) * _viscous_flux_vz(i, j, k, grid, args...)

@inline 𝒯_wx(i, j, k, grid, ρ, args...) = ℑxzᶠᶜᶠ(i, j, k, grid, ρ) * _viscous_flux_wx(i, j, k, grid, args...)
@inline 𝒯_wy(i, j, k, grid, ρ, args...) = ℑyzᶜᶠᶠ(i, j, k, grid, ρ) * _viscous_flux_wy(i, j, k, grid, args...)
@inline 𝒯_wz(i, j, k, grid, ρ, args...) = @inbounds ρ[i, j, k]     * _viscous_flux_wz(i, j, k, grid, args...)

@inline function ∂ⱼ_𝒯₁ⱼ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, closure_fields, clock, model_fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (
          δxᶠᵃᵃ(i, j, k, grid, Ax_qᶜᶜᶜ, 𝒯_ux, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δyᵃᶜᵃ(i, j, k, grid, Ay_qᶠᶠᶜ, 𝒯_uy, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δzᵃᵃᶜ(i, j, k, grid, Az_qᶠᶜᶠ, 𝒯_uz, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy))
end

@inline function ∂ⱼ_𝒯₂ⱼ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, closure_fields, clock, model_fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (
          δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, 𝒯_vx, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, 𝒯_vy, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶠᶠ, 𝒯_vz, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy))
end

# Face stress wrappers for w-momentum

@inline function ∂ⱼ_𝒯₃ⱼ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, closure_fields, clock, model_fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (
          δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶠ, 𝒯_wx, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶠ, 𝒯_wy, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δzᵃᵃᶠ(i, j, k, grid, Az_qᶜᶜᶜ, 𝒯_wz, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy))
end

end # module TurbulenceClosures
