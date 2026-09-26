module Advection

export div_ρUc,
       bottom_advective_tracer_flux

using Oceananigans.Advection:
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z,
    AdaptiveImplicitVerticalAdvection,
    _biased_interpolate_zᵃᵃᶠ,
    BoundsPreservation,
    BoundsPreservingWENO,
    LeftBias,
    RightBias,
    upwind_biased_product,
    rescaled_reconstruction,
    implicit_vertical_velocityᶜᶜᶠ,
    bounded_tracer_flux_divergence_x,
    bounded_tracer_flux_divergence_y,
    bounded_tracer_flux_divergence_z,
    explicit_velocity_scaleᶜᶜᶠ,
    vertical_scheme,
    WENO

using Adapt: Adapt
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Grids: Center
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, Azᶜᶜᶠ
using Oceananigans.Utils: SumOfArrays, AdaptiveVerticallyImplicitDiscretization
using Oceananigans.TimeSteppers: time_discretization
using DocStringExtensions: TYPEDSIGNATURES

using ..AtmosphereModels:
    AtmosphereModels,
    div_ρUc,
    total_density,
    transport_velocities

const AIVA = AdaptiveImplicitVerticalAdvection

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
const BoundsPreservingAVIDWENO = WENO{<:Any, <:Any, <:Any, <:AdaptiveVerticallyImplicitDiscretization, <:BoundsPreservation}

# Indexing yields wᵉ = s·w, the explicit fraction of the IMEX split, so the bounded flux
# functions consume it without duplicating their reconstruction (issue #913).
struct ExplicitVerticalVelocity{G, S, T, W}
    grid :: G
    advection_scheme :: S
    time_discretization :: T
    vertical_velocity :: W
end

Adapt.adapt_structure(to, v::ExplicitVerticalVelocity) =
    ExplicitVerticalVelocity(Adapt.adapt(to, v.grid), Adapt.adapt(to, v.advection_scheme),
                             Adapt.adapt(to, v.time_discretization), Adapt.adapt(to, v.vertical_velocity))

@inline Base.getindex(v::ExplicitVerticalVelocity, i, j, k) =
    @inbounds explicit_velocity_scaleᶜᶜᶠ(i, j, k, v.grid, v.advection_scheme, v.time_discretization,
                                         v.vertical_velocity) * v.vertical_velocity[i, j, k]

# Disambiguates against the `ZeroField` shortcut above.
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::BoundsPreservingAVIDWENO, ρ, U, ::ZeroField) = zero(grid)

# Without the s-scaled velocity the bounded path transported 1 + (1 - s) times: a full
# explicit flux plus the implicit remainder (issue #913). Horizontal fluxes stay explicit.
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection::BoundsPreservingAVIDWENO, ρ, U, c)
    wᵉ = ExplicitVerticalVelocity(grid, advection, time_discretization(advection), U.w)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, wᵉ, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# The vertical velocity the bounded flux functions below transport with: the velocity itself
# for an explicit scheme, its s-scaled explicit fraction under adaptive implicit vertical
# advection (the same `ExplicitVerticalVelocity` the bounded divergence uses, so the scaling is
# applied exactly once).
@inline explicit_vertical_velocity(advection, grid, w) = w
@inline explicit_vertical_velocity(advection::BoundsPreservingAVIDWENO, grid, w) =
    ExplicitVerticalVelocity(grid, advection, time_discretization(advection), w)

#####
##### Advective mass fluxes for the sedimentation of condensate content
#####
#
# General method of `AtmosphereModels.sedimentation_mass_fluxes`, consumed by
# `AtmosphereModels.sedimentation_tendency`: through the two faces
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

# Bounds-preserving WENO rescales its face reconstructions by the cached limiter of the cell
# each one draws on, so the fluxes of cell k are rebuilt from the same limited reconstructions
# `bounded_tracer_flux_divergence_z` forms its mass fluxes from: the latent heat stays with the
# mass at cloud and precipitation edges, where the limiter acts and the unlimited WENO fluxes
# would move heat the tracer tendency does not move.
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

# Reconstructions of `c` at the lower (`k`) and upper (`k + 1`) faces of cell (i, j, k), returned
# as `(c₋ᴸ, c₋ᴿ, c₊ᴸ, c₊ᴿ)`, each rescaled by the cached bounds-preserving limiter θ of the cell it
# draws on (k − 1, k, k and k + 1), exactly as `bounded_tracer_flux_divergence_z` forms its face
# states, so that the two cells sharing a face see the same limited flux through it and the
# latent heat moves with exactly the mass the tracer tendency moves. The limiter is refreshed by
# `update_advection!` before the tendencies that consume it (see `update_state!`).
@inline function bounded_face_reconstructions(i, j, k, grid, advection::BoundsPreservingWENO, c)
    θ = advection.bounds.limiter

    c₊ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, RightBias, c)

    c₊ᴸ = rescaled_reconstruction(c₊ᴸ, i, j, k,   grid, θ, c)
    c₊ᴿ = rescaled_reconstruction(c₊ᴿ, i, j, k+1, grid, θ, c)
    c₋ᴸ = rescaled_reconstruction(c₋ᴸ, i, j, k-1, grid, θ, c)
    c₋ᴿ = rescaled_reconstruction(c₋ᴿ, i, j, k,   grid, θ, c)

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
@inline function bottom_advective_tracer_flux(i, j, grid, advection, ρ, w, c)
    flux_Az = _advective_tracer_flux_z(i, j, 1, grid, advection, w, c)
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * flux_Az / Azᶜᶜᶠ(i, j, 1, grid)
    return explicit_flux + implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c)
end

@inline implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c) = 0

@inline function implicit_bottom_advective_tracer_flux(i, j, grid, advection::AIVA, ρ_face, w, c)
    scheme = vertical_scheme(advection)
    td = time_discretization(scheme)
    wⁱ = implicit_vertical_velocityᶜᶜᶠ(i, j, 1, grid, scheme, td, w)
    @inbounds c_above = c[i, j, 1]
    return -ρ_face * min(wⁱ, 0) * c_above
end

# Bounds-preserving WENO: the bottom face flux of cell 1 as `bounded_tracer_flux_divergence_z`
# forms it, from the reconstructions rescaled by the cached limiter of each donor cell.
@inline function bottom_advective_tracer_flux(i, j, grid, advection::BoundsPreservingWENO, ρ, w, c)
    c₋ᴸ, c₋ᴿ, _, _ = bounded_face_reconstructions(i, j, 1, grid, advection, c)
    explicit_w = explicit_vertical_velocity(advection, grid, w)
    @inbounds w⁻ = explicit_w[i, j, 1]
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * upwind_biased_product(w⁻, c₋ᴸ, c₋ᴿ)
    return explicit_flux + implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c)
end

#####
##### Precipitation flux through the bottom boundary
#####
#
# Scheme-independent implementation of `AtmosphereModels.bottom_precipitation_flux`: the
# bottom-face flux of every sedimenting condensate, summed inside one kernel function. It
# lives here rather than in `AtmosphereModels` because it builds on
# `bottom_advective_tracer_flux`, and `AtmosphereModels` is loaded before this module.
# Reusing the `SedimentingCondensate`s the model resolved once means the diagnostic can never
# disagree with the thermodynamic tendencies about which masses fall, with which humidity field
# and advection scheme. The tuple recursion keeps the kernel type-stable across condensates that
# carry different advection schemes.
@inline bottom_precipitation_flux_kernel(i, j, k, grid, condensates, ρ, wᵗ) =
    sedimenting_bottom_flux(i, j, grid, condensates, ρ, wᵗ)

@inline sedimenting_bottom_flux(i, j, grid, ::Tuple{}, ρ, wᵗ) = zero(grid)

@inline function sedimenting_bottom_flux(i, j, grid, condensates, ρ, wᵗ)
    condensate = first(condensates)
    w = SumOfArrays{2}(wᵗ, condensate.velocity)
    flux = bottom_advective_tracer_flux(i, j, grid, condensate.advection, ρ, w, condensate.specific_humidity)
    return flux + sedimenting_bottom_flux(i, j, grid, Base.tail(condensates), ρ, wᵗ)
end

# Any scheme that declares its sedimenting condensate through `sedimentation_velocity` and
# `condensate_phase` gets the advection-consistent diagnostic for free; with nothing sedimenting
# (including `Nothing` microphysics) the sum is empty and the flux is zero. Schemes that move
# precipitation by internal means (such as `DCMIP2016KM`) override this method instead.
function AtmosphereModels.bottom_precipitation_flux(model, microphysics)
    operation = KernelFunctionOperation{Center, Center, Nothing}(bottom_precipitation_flux_kernel, model.grid,
                                                                 values(model.sedimentation),
                                                                 total_density(model.dynamics),
                                                                 transport_velocities(model).w)
    return Field(operation)
end

end # module
