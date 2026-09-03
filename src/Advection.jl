module Advection

export div_ρUc,
       surface_advective_tracer_flux

using Oceananigans.Advection:
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z,
    AdaptiveImplicitVerticalAdvection,
    _biased_interpolate_zᵃᵃᶠ,
    BoundsPreservingWENO,
    LeftBias,
    RightBias,
    upwind_biased_product,
    implicit_vertical_velocityᶜᶜᶠ,
    bounded_tracer_flux_divergence_x,
    bounded_tracer_flux_divergence_y,
    bounded_tracer_flux_divergence_z,
    explicit_velocity_scaleᶜᶜᶠ,
    vertical_scheme,
    _ω̂₁, _ω̂ₙ, _ε₂

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Grids: Center
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, Azᶜᶜᶠ
using Oceananigans.Utils: SumOfArrays
using Oceananigans.TimeSteppers: time_discretization
using DocStringExtensions: TYPEDSIGNATURES

using ..AtmosphereModels:
    AtmosphereModels,
    div_ρUc,
    total_density,
    transport_velocities

const AIVA = AdaptiveImplicitVerticalAdvection

# TODO: upstream AIVA support for the bounds-preserving path to Oceananigans.
# `bounded_tracer_flux_divergence_z` lacks the `::AIVA` dispatch that
# `advective_tracer_flux_z` has, so this wrapper applies the explicit CFL scaling from
# below. It must be deleted the moment upstream learns to scale that path itself, or the
# velocity would be scaled twice.
struct CFLScaledVerticalVelocity{A, G, W}
    advection :: A
    grid :: G
    velocity :: W
end

@inline function Base.getindex(w::CFLScaledVerticalVelocity, i, j, k)
    scheme = vertical_scheme(w.advection)
    td = time_discretization(scheme)
    scale = explicit_velocity_scaleᶜᶜᶠ(i, j, k, w.grid, scheme, td, w.velocity)
    @inbounds velocity = w.velocity[i, j, k]
    return scale * velocity
end

@inline explicit_vertical_velocity(advection, grid, w) = w
@inline explicit_vertical_velocity(advection::AIVA, grid, w) =
    CFLScaledVerticalVelocity(advection, grid, w)

# Simple wrappers: interpolate ρ to face, multiply existing flux
@inline tracer_mass_flux_x(i, j, k, grid, ρ, args...) =
    ℑxᶠᵃᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_x(i, j, k, grid, args...)

@inline tracer_mass_flux_y(i, j, k, grid, ρ, args...) =
    ℑyᵃᶠᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_y(i, j, k, grid, args...)

@inline tracer_mass_flux_z(i, j, k, grid, ρ, args...) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ) * _advective_tracer_flux_z(i, j, k, grid, args...)

# Main operator
# `tracer_mass_flux_z` reaches Oceananigans' `advective_tracer_flux_z(..., ::AVID, ...)`,
# which applies `explicit_velocity_scaleᶜᶜᶠ` itself, so `U.w` is passed unscaled here.
# The bounds-preserving path below has no such dispatch and does need the wrapper.
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
    explicit_w = explicit_vertical_velocity(advection, grid, U.w)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, explicit_w, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

#####
##### Advective mass fluxes for the sedimentation of condensate content
#####
#
# General method of `AtmosphereModels.sedimentation_mass_fluxes`, consumed by the
# thermodynamic-variable tendencies through `condensate_content_fluxes`: through the two faces
# of cell (i, j, k), the vertical advective flux of `q` at the combined velocity `wᵗ + wˢ` and at
# the transport velocity `wᵗ` alone, with the tracer's own advection scheme, so that their
# difference is by construction the sedimentation part of the mass flux the tracer tendency
# applies to the cell through `div_ρUc`. For `advection === nothing` every flux vanishes
# (Oceananigans returns zero), so no mass and no latent heat move, consistently.
#
# Under adaptive implicit vertical advection these are the CFL-scaled explicit fluxes only,
# the part of the transport the tendency applies; the first-order remainder the implicit solve
# applies is moved with its content after the solve, from the solved state
# (`AtmosphereModels.implicit_sedimentation_mass_fluxes`).

@inline function AtmosphereModels.sedimentation_mass_fluxes(i, j, k, grid, advection, wᵗ, wˢ, q)
    w = SumOfArrays{2}(wᵗ, wˢ)
    F⁻ = (sedimentation_mass_flux(i, j, k,   grid, advection, w,  q),
          sedimentation_mass_flux(i, j, k,   grid, advection, wᵗ, q))
    F⁺ = (sedimentation_mass_flux(i, j, k+1, grid, advection, w,  q),
          sedimentation_mass_flux(i, j, k+1, grid, advection, wᵗ, q))
    return F⁻, F⁺
end

@inline sedimentation_mass_flux(i, j, k, grid, advection, w, q) =
    _advective_tracer_flux_z(i, j, k, grid, advection, w, q)

# Bounds-preserving WENO limits, cell by cell, the two face reconstructions that draw on the
# cell itself, so the flux through a face depends on which cell's tendency is being formed:
# the two cells sharing a face do not see the same flux through it once the limiter engages,
# and `div_ρUc` is not the difference of two face-local fluxes. The fluxes of cell k are
# therefore rebuilt from the same limited reconstructions `bounded_tracer_flux_divergence_z`
# forms its mass fluxes from, so the latent heat stays with the mass at cloud and precipitation
# edges, where the limiter acts and the unlimited WENO fluxes would move heat the tracer
# tendency does not move.
@inline function AtmosphereModels.sedimentation_mass_fluxes(i, j, k, grid, advection::BoundsPreservingWENO, wᵗ, wˢ, q)
    w = SumOfArrays{2}(wᵗ, wˢ)
    c₋ᴸ, c₋ᴿ, c₊ᴸ, c₊ᴿ = bounded_face_reconstructions(i, j, k, grid, advection, q)
    F⁻ = (bounded_sedimentation_mass_flux(i, j, k,   grid, advection, w,  q, c₋ᴸ, c₋ᴿ),
          bounded_sedimentation_mass_flux(i, j, k,   grid, advection, wᵗ, q, c₋ᴸ, c₋ᴿ))
    F⁺ = (bounded_sedimentation_mass_flux(i, j, k+1, grid, advection, w,  q, c₊ᴸ, c₊ᴿ),
          bounded_sedimentation_mass_flux(i, j, k+1, grid, advection, wᵗ, q, c₊ᴸ, c₊ᴿ))
    return F⁻, F⁺
end

# Face flux from the limited reconstructions `cᴸ`, `cᴿ`, with the explicit CFL scaling that
# `div_ρUc` applies under adaptive implicit vertical advection.
@inline function bounded_sedimentation_mass_flux(i, j, k, grid, advection, w, q, cᴸ, cᴿ)
    explicit_w = explicit_vertical_velocity(advection, grid, w)
    @inbounds wₑ = explicit_w[i, j, k]
    return Azᶜᶜᶠ(i, j, k, grid) * upwind_biased_product(wₑ, cᴸ, cᴿ)
end

# TODO: move `bounded_face_reconstructions` upstream. It reproduces the reconstruction and
# limiting half of `Oceananigans.Advection.bounded_tracer_flux_divergence_z`, reaching into the
# private `_ω̂₁`, `_ω̂ₙ`, `_ε₂` constants to do it, so that `sedimentation_mass_fluxes` and
# `surface_advective_tracer_flux` form the same face fluxes the tracer tendency applies. The
# clean fix is for Oceananigans to factor this helper out of its divergence and export it;
# until then this copy has to be kept in sync by hand.

# Reconstructions of `c` at the lower (`k`) and upper (`k + 1`) faces of cell (i, j, k), returned
# as `(c₋ᴸ, c₋ᴿ, c₊ᴸ, c₊ᴿ)`, with the cell's bounds-preserving limiter θ applied to the two that
# draw on the cell itself (`c₋ᴿ` and `c₊ᴸ`), exactly as `bounded_tracer_flux_divergence_z` does.
@inline function bounded_face_reconstructions(i, j, k, grid, advection::BoundsPreservingWENO, c)
    c_min = @inbounds advection.bounds[1]
    c_max = @inbounds advection.bounds[2]

    c₊ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, RightBias, c)

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    @inbounds cᵢⱼ = c[i, j, k]
    p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M = max(p̃, c₊ᴸ, c₋ᴿ)
    m = min(p̃, c₊ᴸ, c₋ᴿ)

    θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
    θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
    θ = min(θ_max, θ_min, one(grid))

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    return c₋ᴸ, c₋ᴿ, c₊ᴸ, c₊ᴿ
end

#####
##### Advection-consistent surface flux
#####

"""
$(TYPEDSIGNATURES)

Compute the downward advective mass flux per unit area at the bottom face (`k = 1`)
for tracer `c` advected by vertical velocity `w` through density field `ρ`, using the
given `advection` scheme.

For an explicit scheme this evaluates the same face flux that `div_ρUc` uses at
the bottom boundary. For adaptive implicit vertical advection it returns the
instantaneous split-operator flux at the supplied tracer state; the step-integrated
implicit outflow depends on the post-solve tracer and must be accumulated separately.

For adaptive implicit vertical advection, the result includes both the CFL-scaled
high-order explicit flux and its first-order implicit remainder, evaluated at `c`.

Returns a positive value for downward (out-of-domain) flux.
"""
@inline function surface_advective_tracer_flux(i, j, grid, advection, ρ, w, c)
    flux_Az = _advective_tracer_flux_z(i, j, 1, grid, advection, w, c)
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * flux_Az / Azᶜᶜᶠ(i, j, 1, grid)
    return explicit_flux + implicit_surface_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c)
end

@inline implicit_surface_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c) = 0

@inline function implicit_surface_advective_tracer_flux(i, j, grid, advection::AIVA, ρ_face, w, c)
    scheme = vertical_scheme(advection)
    td = time_discretization(scheme)
    wⁱ = implicit_vertical_velocityᶜᶜᶠ(i, j, 1, grid, scheme, td, w)
    @inbounds c_above = c[i, j, 1]
    return -ρ_face * min(wⁱ, 0) * c_above
end

# Bounds-preserving WENO: the bottom face flux of cell 1 as `bounded_tracer_flux_divergence_z`
# forms it, from the reconstructions limited by that cell.
@inline function surface_advective_tracer_flux(i, j, grid, advection::BoundsPreservingWENO, ρ, w, c)
    c₋ᴸ, c₋ᴿ, _, _ = bounded_face_reconstructions(i, j, 1, grid, advection, c)
    explicit_w = explicit_vertical_velocity(advection, grid, w)
    @inbounds w⁻ = explicit_w[i, j, 1]
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * upwind_biased_product(w⁻, c₋ᴸ, c₋ᴿ)
    return explicit_flux + implicit_surface_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c)
end

#####
##### Surface precipitation flux (flux out of the bottom boundary)
#####
#
# Scheme-independent implementation of `AtmosphereModels.surface_precipitation_flux`: the
# bottom-face flux of every sedimentation constituent, summed inside one kernel function. It
# lives here rather than in `AtmosphereModels` because it builds on
# `surface_advective_tracer_flux`, and `AtmosphereModels` is loaded before this module.
# Reusing the `(; w, q, ρq, phase, advection)` constituents the model resolved once means the
# diagnostic can never disagree with the thermodynamic tendencies about which masses fall, with
# which humidity field and advection scheme. The tuple recursion keeps the kernel type-stable
# across constituents that carry different advection schemes.
@inline surface_precipitation_flux_kernel(i, j, k, grid, constituents, ρ, wᵗ) =
    sedimenting_surface_flux(i, j, grid, constituents, ρ, wᵗ)

@inline sedimenting_surface_flux(i, j, grid, ::Tuple{}, ρ, wᵗ) = zero(grid)

@inline function sedimenting_surface_flux(i, j, grid, constituents, ρ, wᵗ)
    (; w, q, advection) = first(constituents)
    flux = surface_advective_tracer_flux(i, j, grid, advection, ρ, SumOfArrays{2}(wᵗ, w), q)
    return flux + sedimenting_surface_flux(i, j, grid, Base.tail(constituents), ρ, wᵗ)
end

# Any scheme that declares its sedimenting condensate through `sedimentation_velocity` and
# `condensate_phase` gets the advection-consistent diagnostic for free; with nothing sedimenting
# (including `Nothing` microphysics) the sum is empty and the flux is zero. Schemes that move
# precipitation by internal means (such as `DCMIP2016KM`) override this method instead.
function AtmosphereModels.surface_precipitation_flux(model, microphysics)
    operation = KernelFunctionOperation{Center, Center, Nothing}(surface_precipitation_flux_kernel, model.grid,
                                                                 model.sedimentation_constituents,
                                                                 total_density(model.dynamics),
                                                                 transport_velocities(model).w)
    return Field(operation)
end

end # module
