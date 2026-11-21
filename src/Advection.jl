module Advection

export div_ρUc,
       bottom_advective_tracer_flux,
       FrozenWeightsWENO

using Oceananigans.Advection:
    Advection as OceananigansAdvection,
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z,
    AdaptiveImplicitVerticalAdvection,
    AbstractUpwindBiasedAdvectionScheme,
    _biased_interpolate_xᶠᵃᵃ,
    _biased_interpolate_yᵃᶠᵃ,
    _biased_interpolate_zᵃᵃᶠ,
    _symmetric_interpolate_xᶠᵃᵃ,
    _symmetric_interpolate_yᵃᶠᵃ,
    AbstractCenteredAdvectionScheme,
    FluxFormAdvection,
    bias,
    BoundsPreservation,
    BoundsPreservingWENO,
    FunctionStencil,
    implicit_vertical_velocityᶜᶜᶠ,
    explicit_velocity_scaleᶜᶜᶠ,
    materialize_advection,
    adapt_advection_order,
    update_adaptive_timestep!,
    vertical_scheme,
    WENO,
    FluxLimitedWENO,
    FluxLimitedBounds,
    upwind_face_value_x,
    upwind_face_value_y,
    upwind_face_value_z,
    upwind_face_flux_x,
    upwind_face_flux_y,
    upwind_face_flux_z,
    face_factor,
    transport_x,
    transport_y

using Adapt: Adapt, adapt
using Oceananigans: Oceananigans, CenterField
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: Architectures, on_architecture
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Grids: Center, AbstractGrid, Flat
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, Az⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, Azᶜᶜᶠ
using Oceananigans.Utils: SumOfArrays, AdaptiveVerticallyImplicitDiscretization, ExplicitTimeDiscretization
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, time_discretization
using DocStringExtensions: TYPEDEF
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS
using DocStringExtensions: TYPEDSIGNATURES

using Breeze.TimeSteppers: tendencies_feed_first_stage, MassConsistentVelocity

using ..AtmosphereModels:
    AtmosphereModels,
    div_ρUc,
    total_density,
    transport_velocities

using ..HydrostaticEquations: HydrostaticEquations, XMomentumDensity, YMomentumDensity, mass_flux_x, mass_flux_y,
                              CoordinateMassFlux
using ..PressureLevelDiscretization: PressureLevelGrid

const AIVA = AdaptiveImplicitVerticalAdvection

# Simple wrappers: interpolate ρ to face, multiply existing flux
@inline tracer_mass_flux_x(i, j, k, grid, ρ, args...) =
    ℑxᶠᵃᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_x(i, j, k, grid, args...)

@inline tracer_mass_flux_y(i, j, k, grid, ρ, args...) =
    ℑyᵃᶠᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_y(i, j, k, grid, args...)

@inline tracer_mass_flux_z(i, j, k, grid, ρ, args...) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ) * _advective_tracer_flux_z(i, j, k, grid, args...)

# On a pressure-level grid the horizontal tracer fluxes are the mass fluxes of continuity, written in the
# layer mass Δp / g, times the face value that `advection` reconstructs: ℑx(ρ) Ax, with
# Ax = Δy ℑx(Δp / (g ρ)), differs from that mass flux at second order. (Oceananigans' upwind fluxes
# multiply by Axᶠᶜᶜ directly, so the face value is reconstructed here rather than through a transport.)
@inline x_face_value(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, u, c) =
    @inbounds _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias(u[i, j, k]), c)
@inline y_face_value(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, v, c) =
    @inbounds _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(v[i, j, k]), c)
@inline x_face_value(i, j, k, grid, scheme::AbstractCenteredAdvectionScheme, u, c) = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
@inline y_face_value(i, j, k, grid, scheme::AbstractCenteredAdvectionScheme, v, c) = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
@inline x_face_value(i, j, k, grid, scheme::FluxFormAdvection, u, c) = x_face_value(i, j, k, grid, scheme.x, u, c)
@inline y_face_value(i, j, k, grid, scheme::FluxFormAdvection, v, c) = y_face_value(i, j, k, grid, scheme.y, v, c)

@inline tracer_mass_flux_x(i, j, k, grid::PressureLevelGrid, ρ, advection, u, c) =
    mass_flux_x(i, j, k, grid, ρ, u) * x_face_value(i, j, k, grid, advection, u, c)
@inline tracer_mass_flux_y(i, j, k, grid::PressureLevelGrid, ρ, advection, v, c) =
    mass_flux_y(i, j, k, grid, ρ, v) * y_face_value(i, j, k, grid, advection, v, c)
@inline tracer_mass_flux_x(i, j, k, grid::PressureLevelGrid, ρ, ::Nothing, u, c) = zero(grid)

# On a pressure-level grid the vertical transport is the mass flux of continuity through the
# coordinate surfaces, carried with the coordinate velocity in a `CoordinateMassFlux`: indexed, it is
# the velocity w (what the explicit scale of adaptive implicit vertical advection, the Courant
# numbers and the sedimentation read); the fluxes take its mass flux ρw per unit area, Az ρw ĉ,
# without ℑz(ρ). The scale of the explicit part is that of the velocity.
@inline explicit_mass_flux_scale(i, j, k, grid, advection, W) =
    explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, advection, time_discretization(advection), W)
@inline explicit_mass_flux_scale(i, j, k, grid, advection::FluxFormAdvection, W) =
    explicit_mass_flux_scale(i, j, k, grid, advection.z, W)

@inline tracer_mass_flux_z(i, j, k, grid, ρ, ::Nothing, W::CoordinateMassFlux, c) = zero(grid)
@inline tracer_mass_flux_z(i, j, k, grid, ρ, advection, W::CoordinateMassFlux, c) =
    explicit_mass_flux_scale(i, j, k, grid, advection, W) *
    _advective_tracer_flux_z(i, j, k, grid, explicit_scheme(advection), W.mass_flux, c)

# The scheme with an explicit time discretization: the explicit part is scaled above
@inline explicit_scheme(advection) = advection
@inline explicit_scheme(advection::FluxFormAdvection) = FluxFormAdvection(advection.x, advection.y, explicit_scheme(advection.z))
@inline explicit_scheme(advection::WENO{N, FT, WCT, <:AdaptiveVerticallyImplicitDiscretization}) where {N, FT, WCT} =
    WENO{N, FT, WCT}(advection.bounds, advection.buffer_scheme, advection.advecting_velocity_scheme, ExplicitTimeDiscretization())

@inline function OceananigansAdvection.transport_z(i, j, k, grid, advection, ρ, W::CoordinateMassFlux)
    s = explicit_mass_flux_scale(i, j, k, grid, advection, W)
    return @inbounds Azᶜᶜᶠ(i, j, k, grid) * s * W.mass_flux[i, j, k]
end
@inline tracer_mass_flux_y(i, j, k, grid::PressureLevelGrid, ρ, ::Nothing, v, c) = zero(grid)

# The transport of the flux limiter of Oceananigans: the same mass fluxes
@inline OceananigansAdvection.transport_x(i, j, k, grid::PressureLevelGrid, advection, ρ, u) = mass_flux_x(i, j, k, grid, ρ, u)
@inline OceananigansAdvection.transport_y(i, j, k, grid::PressureLevelGrid, advection, ρ, v) = mass_flux_y(i, j, k, grid, ρ, v)

# Main operator: the volume-weighted divergence of the face fluxes. Every method of `div_ρUc` is
# V⁻¹ times the sum of the fluxes out of the cell, `ρUc_flux_divergence`, which a mass-weighted
# stage of a pressure-level grid takes without V⁻¹ (`layer_mass_flux_divergence`).
@inline ρUc_flux_divergence(i, j, k, grid, advection, ρ, U, c) =
    (δxᶜᵃᵃ(i, j, k, grid, tracer_mass_flux_x, ρ, advection, U.u, c) +
     δyᵃᶜᵃ(i, j, k, grid, tracer_mass_flux_y, ρ, advection, U.v, c) +
     δzᵃᵃᶜ(i, j, k, grid, tracer_mass_flux_z, ρ, advection, U.w, c))

@inline AtmosphereModels.div_ρUc(i, j, k, grid, advection, ρ, U, c) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * ρUc_flux_divergence(i, j, k, grid, advection, ρ, U, c)

# Fallback for nothing advection
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::Nothing, ρ, U, c) = zero(grid)
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::BoundsPreservingWENO, ρ, U, ::ZeroField) = zero(grid)
@inline ρUc_flux_divergence(i, j, k, grid, ::Nothing, ρ, U, c) = zero(grid)
@inline ρUc_flux_divergence(i, j, k, grid, ::BoundsPreservingWENO, ρ, U, ::ZeroField) = zero(grid)

# The transport tendency of a scalar in the units in which a time stepper advances it: `div_ρUc`
# per unit volume on most grids, and on a pressure-level grid the divergence of the same face
# fluxes per unit area, (δx F + δy F + δz F) / Az: the rate of change of the scalar's mass per unit
# area M = (Δp / g) c, exactly, without the rounding of Δz V⁻¹.
@inline HydrostaticEquations.layer_mass_flux_divergence(i, j, k, grid, advection, ρ, U, c) = AtmosphereModels.div_ρUc(i, j, k, grid, advection, ρ, U, c)
@inline HydrostaticEquations.layer_mass_flux_divergence(i, j, k, grid::PressureLevelGrid, advection, ρ, U, c) =
    Az⁻¹ᶜᶜᶜ(i, j, k, grid) * ρUc_flux_divergence(i, j, k, grid, advection, ρ, U, c)

# The upwind face values that the θ kernel of Oceananigans #6534 caches are written only on the
# faces where the velocities it was refreshed with (those of the model) carry flow, in their
# direction. Breeze transports with other velocities (the sedimentation velocity of condensate,
# which crosses the bottom face; the transport velocities of the stages), for which the cache is
# stale: a uniform rain column then exported nothing through the ground (cloud_microphysics_1M,
# "Bounds-preserving WENO bottom precipitation flux"). Breeze's bounded fluxes therefore form the
# rescaled upwind face values from the reconstruction and the cached θ, without the face cache.
# TODO: an Oceananigans option to skip the face cache, or a cache keyed by velocity (#6534).
@inline without_face_cache(advection) = advection

@inline function without_face_cache(advection::WENO{N, FT, WCT, <:Any, <:BoundsPreservation}) where {N, FT, WCT}
    b = advection.bounds
    bounds = BoundsPreservation(b.minimum_value, b.maximum_value, b.maximum_cfl, b.minimum_weight,
                                b.reconstruction_limiter, b.flux_limiter, nothing)
    return WENO{N, FT, WCT}(bounds, advection.buffer_scheme, advection.advecting_velocity_scheme,
                            advection.time_discretization)
end

#####
##### Bounded-by-construction face fluxes of bounds-preserving and flux-limited WENO (Breeze-owned)
#####
#
# Two rounding defects of Oceananigans #6534 let Float32 rain go below zero by ~1e-17 (see
# derivations/hydro_current_stack/OA6534_FCT_CANCELLATION.md):
#   (B) the rescaled upwind face value θ (ĉ - c) + c can land an ulp outside the bounds;
#   (A) the limited flux Jᵘ + ϑ (Jᴴ - Jᵘ) leaves an ulp-level residue where Jᵘ and Jᴴ - Jᵘ cancel.
# Breeze forms its bounded fluxes with its own functions: the face value is clamped to the bounds
# (`bounded_face_value`, as on the nonhydrostatic campaign branch, 64f5f49c), and the limited flux is
# the convex combination ϑ Jᴴ + (1 - ϑ) Jᵘ of two bounded fluxes, with the face factor ϑ chosen from
# the cached antidiffusive flux as in #6534.
# TODO: both belong in Oceananigans' `bounded_face_flux_{x,y,z}` / `limited_face_flux_{x,y,z}` (#6534).

# The limited value θ (ĉ - c) + c lies within the bounds in exact arithmetic. In floating point it
# can land a unit in the last place of c outside them where ĉ is far outside (θ ≪ 1), and a wind
# then carries it into an empty neighbour. Bounding the face value before the flux is formed keeps
# the flux single-valued (it depends on the donor cell alone), hence conservative.
#
# The clamp acts on a scalar only where its donor value lies within the bounds of its scheme: the
# interval is widened to include the donor value cᵢ, so a scalar outside the bounds it was given (a
# uniform tracer of 2 with bounds (0, 1), a potential temperature advected with the bounds of water)
# is transported as before. On the nonhydrostatic branch (64f5f49c) the clamp is to the bounds alone.
@inline function bounded_face_value(advection, ĉ, cᵢ)
    b = advection.bounds
    return clamp(ĉ, min(b.minimum_value, cᵢ), max(b.maximum_value, cᵢ))
end

# The donor value of the face (i, j, k) for a face velocity ũ: the cell the flow comes from
@inline x_donor(i, j, k, c, ũ) = @inbounds ifelse(ũ > 0, c[i-1, j, k], c[i, j, k])
@inline y_donor(i, j, k, c, ṽ) = @inbounds ifelse(ṽ > 0, c[i, j-1, k], c[i, j, k])
@inline z_donor(i, j, k, c, w̃) = @inbounds ifelse(w̃ > 0, c[i, j, k-1], c[i, j, k])

@inline function clamped_face_flux_x(i, j, k, grid, advection, ρ, u, c)
    b = advection.bounds
    ũ = @inbounds u[i, j, k]
    ĉ = bounded_face_value(advection, upwind_face_value_x(i, j, k, grid, advection, nothing, b.reconstruction_limiter, ũ, c), x_donor(i, j, k, c, ũ))
    return transport_x(i, j, k, grid, advection, ρ, u) * ĉ
end

@inline function clamped_face_flux_y(i, j, k, grid, advection, ρ, v, c)
    b = advection.bounds
    ṽ = @inbounds v[i, j, k]
    ĉ = bounded_face_value(advection, upwind_face_value_y(i, j, k, grid, advection, nothing, b.reconstruction_limiter, ṽ, c), y_donor(i, j, k, c, ṽ))
    return transport_y(i, j, k, grid, advection, ρ, v) * ĉ
end

@inline function clamped_face_flux_z(i, j, k, grid, advection, ρ, w, c)
    b = advection.bounds
    td = advection.time_discretization
    w̃ = @inbounds explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, advection, td, w) * w[i, j, k]
    ĉ = bounded_face_value(advection, upwind_face_value_z(i, j, k, grid, advection, nothing, b.reconstruction_limiter, w̃, c), z_donor(i, j, k, c, w̃))
    return ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Azᶜᶜᶠ(i, j, k, grid) * w̃ * ĉ
end

# The bounded fluxes of Oceananigans' flux limiter (its antidiffusive fluxes are these less the upwind
# fluxes, which take `transport_x/y/z`): on a pressure-level grid, the same mass fluxes times the
# bounded face value, so that the antidiffusive flux of a uniform scalar vanishes
@inline function OceananigansAdvection.bounded_face_flux_x(i, j, k, grid::PressureLevelGrid, advection::BoundsPreservingWENO, ρ, u, c)
    ũ = @inbounds u[i, j, k]
    ĉ = upwind_face_value_x(i, j, k, grid, advection, advection.bounds.upwind_face_values, advection.bounds.reconstruction_limiter, ũ, c)
    return transport_x(i, j, k, grid, advection, ρ, u) * ĉ
end

@inline function OceananigansAdvection.bounded_face_flux_y(i, j, k, grid::PressureLevelGrid, advection::BoundsPreservingWENO, ρ, v, c)
    ṽ = @inbounds v[i, j, k]
    ĉ = upwind_face_value_y(i, j, k, grid, advection, advection.bounds.upwind_face_values, advection.bounds.reconstruction_limiter, ṽ, c)
    return transport_y(i, j, k, grid, advection, ρ, v) * ĉ
end

@inline function OceananigansAdvection.bounded_face_flux_z(i, j, k, grid::PressureLevelGrid, advection::BoundsPreservingWENO, ρ, W::CoordinateMassFlux, c)
    ρw̃ = @inbounds explicit_mass_flux_scale(i, j, k, grid, advection, W) * W.mass_flux[i, j, k]
    ĉ = upwind_face_value_z(i, j, k, grid, advection, advection.bounds.upwind_face_values, advection.bounds.reconstruction_limiter, ρw̃, c)
    return Azᶜᶜᶠ(i, j, k, grid) * ρw̃ * ĉ
end

# The bounded flux through a coordinate surface of a pressure-level grid: the mass flux, scaled to
# its explicit part with the velocity, times the bounded face value
@inline function clamped_face_flux_z(i, j, k, grid, advection, ρ, W::CoordinateMassFlux, c)
    b = advection.bounds
    ρw̃ = @inbounds explicit_mass_flux_scale(i, j, k, grid, advection, W) * W.mass_flux[i, j, k]
    ĉ = bounded_face_value(advection, upwind_face_value_z(i, j, k, grid, advection, nothing, b.reconstruction_limiter, ρw̃, c), z_donor(i, j, k, c, ρw̃))
    return Azᶜᶜᶠ(i, j, k, grid) * ρw̃ * ĉ
end

@inline function convex_limited_face_flux_x(i, j, k, grid, advection, ρ, u, c)
    L = advection.bounds.flux_limiter
    A = @inbounds L.antidiffusive_fluxes.x[i, j, k]
    ϑ = face_factor(A, L.outgoing_fraction, L.incoming_fraction, i-1, j, k, i, j, k)
    return ϑ * clamped_face_flux_x(i, j, k, grid, advection, ρ, u, c) + (1 - ϑ) * upwind_face_flux_x(i, j, k, grid, advection, ρ, u, c)
end

@inline function convex_limited_face_flux_y(i, j, k, grid, advection, ρ, v, c)
    L = advection.bounds.flux_limiter
    A = @inbounds L.antidiffusive_fluxes.y[i, j, k]
    ϑ = face_factor(A, L.outgoing_fraction, L.incoming_fraction, i, j-1, k, i, j, k)
    return ϑ * clamped_face_flux_y(i, j, k, grid, advection, ρ, v, c) + (1 - ϑ) * upwind_face_flux_y(i, j, k, grid, advection, ρ, v, c)
end

@inline function convex_limited_face_flux_z(i, j, k, grid, advection, ρ, w, c)
    L = advection.bounds.flux_limiter
    A = @inbounds L.antidiffusive_fluxes.z[i, j, k]
    ϑ = face_factor(A, L.outgoing_fraction, L.incoming_fraction, i, j, k-1, i, j, k)
    return ϑ * clamped_face_flux_z(i, j, k, grid, advection, ρ, w, c) + (1 - ϑ) * upwind_face_flux_z(i, j, k, grid, advection, ρ, w, c)
end

# Divergences of these fluxes; a Flat direction has no faces
for (dir, F) in ((:x, :clamped_face_flux_x), (:y, :clamped_face_flux_y), (:z, :clamped_face_flux_z),
                 (:x, :convex_limited_face_flux_x), (:y, :convex_limited_face_flux_y), (:z, :convex_limited_face_flux_z))
    D = Symbol(F, :_divergence)
    if dir == :x
        @eval @inline $D(i, j, k, grid, args...) = $F(i+1, j, k, grid, args...) - $F(i, j, k, grid, args...)
        @eval @inline $D(i, j, k, grid::AbstractGrid{<:Any, Flat}, args...) = zero(grid)
    elseif dir == :y
        @eval @inline $D(i, j, k, grid, args...) = $F(i, j+1, k, grid, args...) - $F(i, j, k, grid, args...)
        @eval @inline $D(i, j, k, grid::AbstractGrid{<:Any, <:Any, Flat}, args...) = zero(grid)
    else
        @eval @inline $D(i, j, k, grid, args...) = $F(i, j, k+1, grid, args...) - $F(i, j, k, grid, args...)
        @eval @inline $D(i, j, k, grid::AbstractGrid{<:Any, <:Any, <:Any, Flat}, args...) = zero(grid)
    end
end

# Is this immersed-boundary safe without having to extend it in ImmersedBoundaries.jl? I think so... (velocity on immmersed boundaries is masked to 0)
@inline function ρUc_flux_divergence(i, j, k, grid, advection::BoundsPreservingWENO, ρ, U, c)
    div_x = clamped_face_flux_x_divergence(i, j, k, grid, advection, ρ, U.u, c)
    div_y = clamped_face_flux_y_divergence(i, j, k, grid, advection, ρ, U.v, c)
    div_z = clamped_face_flux_z_divergence(i, j, k, grid, advection, ρ, U.w, c)
    return div_x + div_y + div_z
end

@inline AtmosphereModels.div_ρUc(i, j, k, grid, advection::BoundsPreservingWENO, ρ, U, c) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * ρUc_flux_divergence(i, j, k, grid, advection, ρ, U, c)

# PROVISIONAL (hydrostatic convex-MIS work; to be superseded by the flux-corrected transport of the
# CompressibleDynamics port): WENO with a flux limiter (Oceananigans PR 6534). Every face carries the
# first-order upwind mass flux plus the fraction of the antidiffusive flux that the fractions of
# `update_flux_limiter!` afford, which must have been computed with the same `ρ`, `U` and `c` and
# with the time step of the forward Euler update. The limited fluxes scale the vertical velocity to
# its explicit part themselves; that path is not tested under adaptive implicit vertical advection.
@inline flux_limited_flux_divergence(i, j, k, grid, advection, ρ, U, c) =
    (convex_limited_face_flux_x_divergence(i, j, k, grid, advection, ρ, U.u, c) +
     convex_limited_face_flux_y_divergence(i, j, k, grid, advection, ρ, U.v, c) +
     convex_limited_face_flux_z_divergence(i, j, k, grid, advection, ρ, U.w, c))

@inline flux_limited_div_ρUc(i, j, k, grid, advection, ρ, U, c) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * flux_limited_flux_divergence(i, j, k, grid, advection, ρ, U, c)

const FluxLimitedAVIDWENO = WENO{<:Any, <:Any, <:Any, <:AdaptiveVerticallyImplicitDiscretization, <:FluxLimitedBounds}

@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::FluxLimitedWENO, ρ, U, ::ZeroField) = zero(grid)
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::FluxLimitedAVIDWENO, ρ, U, ::ZeroField) = zero(grid)

@inline AtmosphereModels.div_ρUc(i, j, k, grid, advection::FluxLimitedWENO, ρ, U, c) =
    flux_limited_div_ρUc(i, j, k, grid, advection, ρ, U, c)

@inline AtmosphereModels.div_ρUc(i, j, k, grid, advection::FluxLimitedAVIDWENO, ρ, U, c) =
    flux_limited_div_ρUc(i, j, k, grid, advection, ρ, U, c)

@inline ρUc_flux_divergence(i, j, k, grid, ::FluxLimitedWENO, ρ, U, ::ZeroField) = zero(grid)
@inline ρUc_flux_divergence(i, j, k, grid, ::FluxLimitedAVIDWENO, ρ, U, ::ZeroField) = zero(grid)
@inline ρUc_flux_divergence(i, j, k, grid, advection::FluxLimitedWENO, ρ, U, c) =
    flux_limited_flux_divergence(i, j, k, grid, advection, ρ, U, c)
@inline ρUc_flux_divergence(i, j, k, grid, advection::FluxLimitedAVIDWENO, ρ, U, c) =
    flux_limited_flux_divergence(i, j, k, grid, advection, ρ, U, c)

# The horizontal part of `div_ρUc`, from the same face fluxes
@inline AtmosphereModels.horizontal_div_ρUc(i, j, k, grid, advection, ρ, U, c) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, tracer_mass_flux_x, ρ, advection, U.u, c) +
                              δyᵃᶜᵃ(i, j, k, grid, tracer_mass_flux_y, ρ, advection, U.v, c))

@inline AtmosphereModels.horizontal_div_ρUc(i, j, k, grid, ::Nothing, ρ, U, c) = zero(grid)
@inline AtmosphereModels.horizontal_div_ρUc(i, j, k, grid, ::BoundsPreservingWENO, ρ, U, ::ZeroField) = zero(grid)

@inline function AtmosphereModels.horizontal_div_ρUc(i, j, k, grid, advection::BoundsPreservingWENO, ρ, U, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (clamped_face_flux_x_divergence(i, j, k, grid, advection, ρ, U.u, c) +
                                     clamped_face_flux_y_divergence(i, j, k, grid, advection, ρ, U.v, c))
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

# A velocity that is the explicitly transported one already is taken as it is: where the explicit
# fraction is limited, scaling it again would take off what it adds to the limited velocity.
@inline OceananigansAdvection.explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, ::AdaptiveVerticallyImplicitDiscretization,
                                                        ::ExplicitVerticalVelocity{<:Any, <:Any, <:Any, <:MassConsistentVelocity}) = one(grid)

# Disambiguates against the `ZeroField` shortcut above.
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::BoundsPreservingAVIDWENO, ρ, U, ::ZeroField) = zero(grid)
@inline ρUc_flux_divergence(i, j, k, grid, ::BoundsPreservingAVIDWENO, ρ, U, ::ZeroField) = zero(grid)

# Without the s-scaled velocity the bounded path transported 1 + (1 - s) times: a full
# explicit flux plus the implicit remainder (issue #913). Horizontal fluxes stay explicit.
@inline function ρUc_flux_divergence(i, j, k, grid, advection::BoundsPreservingAVIDWENO, ρ, U, c)
    wᵉ = ExplicitVerticalVelocity(grid, advection, time_discretization(advection), U.w)
    div_x = clamped_face_flux_x_divergence(i, j, k, grid, advection, ρ, U.u, c)
    div_y = clamped_face_flux_y_divergence(i, j, k, grid, advection, ρ, U.v, c)
    div_z = clamped_face_flux_z_divergence(i, j, k, grid, advection, ρ, wᵉ, c)
    return div_x + div_y + div_z
end

@inline AtmosphereModels.div_ρUc(i, j, k, grid, advection::BoundsPreservingAVIDWENO, ρ, U, c) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * ρUc_flux_divergence(i, j, k, grid, advection, ρ, U, c)

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
    F⁻ = (_advective_tracer_flux_z(i, j, k,   grid, advection, w,  q),
          _advective_tracer_flux_z(i, j, k,   grid, advection, wᵗ, q))
    F⁺ = (_advective_tracer_flux_z(i, j, k+1, grid, advection, w,  q),
          _advective_tracer_flux_z(i, j, k+1, grid, advection, wᵗ, q))
    return F⁻, F⁺
end

# Bounds-preserving WENO (Oceananigans #6534 semantics) forms each face flux from the upwind
# reconstruction alone, rescaled by the limiter θ of the upwind cell (`reconstruction_limiter`, or
# none with `NoRescaling`), read from the face values cached by the θ kernel where the scheme keeps
# them. Like the tracer tendency (`div_ρUc`), they are formed without the face cache (see
# `without_face_cache`), so the latent heat moves with exactly the mass the tracer tendency moves.
@inline function AtmosphereModels.sedimentation_mass_fluxes(i, j, k, grid, advection::BoundsPreservingWENO, wᵗ, wˢ, q)
    w = SumOfArrays{2}(wᵗ, wˢ)
    F⁻ = (bounded_sedimentation_mass_flux(i, j, k,   grid, advection, w,  q),
          bounded_sedimentation_mass_flux(i, j, k,   grid, advection, wᵗ, q))
    F⁺ = (bounded_sedimentation_mass_flux(i, j, k+1, grid, advection, w,  q),
          bounded_sedimentation_mass_flux(i, j, k+1, grid, advection, wᵗ, q))
    return F⁻, F⁺
end

# Face flux from the rescaled upwind face value, with the explicit CFL scaling that `div_ρUc`
# applies under adaptive implicit vertical advection.
@inline function bounded_sedimentation_mass_flux(i, j, k, grid, advection, w, q)
    bounds = advection.bounds
    explicit_w = explicit_vertical_velocity(advection, grid, w)
    @inbounds wₑ = explicit_w[i, j, k]
    ĉ = bounded_face_value(advection, upwind_face_value_z(i, j, k, grid, advection, nothing, bounds.reconstruction_limiter, wₑ, q), z_donor(i, j, k, q, wₑ))
    return Azᶜᶜᶠ(i, j, k, grid) * wₑ * ĉ
end

# With a flux limiter the tracer tendency applies the upwind flux plus the limited antidiffusive
# flux of each face (`limited_face_flux_z`). The antidiffusive part is the same at `wᵗ + wˢ` and at
# `wᵗ`, so the sedimentation part of the mass flux, their difference, is the upwind flux of the
# sedimentation velocity, and the latent heat moves with exactly what the tendency moves.
@inline function AtmosphereModels.sedimentation_mass_fluxes(i, j, k, grid, advection::FluxLimitedWENO, wᵗ, wˢ, q)
    w = SumOfArrays{2}(wᵗ, wˢ)
    F⁻ = (convex_limited_face_flux_z(i, j, k,   grid, advection, 1, w,  q),
          convex_limited_face_flux_z(i, j, k,   grid, advection, 1, wᵗ, q))
    F⁺ = (convex_limited_face_flux_z(i, j, k+1, grid, advection, 1, w,  q),
          convex_limited_face_flux_z(i, j, k+1, grid, advection, 1, wᵗ, q))
    return F⁻, F⁺
end

#####
##### Advection-consistent surface flux
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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

# Bounds-preserving WENO: the bottom face flux of cell 1 as `bounded_face_flux_z` forms it, from the
# rescaled upwind face value, without the face cache (see `without_face_cache`).
@inline function bottom_advective_tracer_flux(i, j, grid, advection::BoundsPreservingWENO, ρ, w, c)
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * bounded_sedimentation_mass_flux(i, j, 1, grid, advection, w, c) / Azᶜᶜᶠ(i, j, 1, grid)
    return explicit_flux + implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c)
end

# With a flux limiter: the limited bottom face flux the tracer tendency applies.
@inline function bottom_advective_tracer_flux(i, j, grid, advection::FluxLimitedWENO, ρ, w, c)
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * convex_limited_face_flux_z(i, j, 1, grid, advection, 1, w, c) / Azᶜᶜᶠ(i, j, 1, grid)
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
# `condensate_liquid_fraction` gets the advection-consistent diagnostic for free; with nothing sedimenting
# (including `Nothing` microphysics) the sum is empty and the flux is zero. Schemes that move
# precipitation by internal means (such as `DCMIP2016KM`) override this method instead.
function AtmosphereModels.bottom_precipitation_flux(model, microphysics)
    operation = KernelFunctionOperation{Center, Center, Nothing}(bottom_precipitation_flux_kernel, model.grid,
                                                                 values(model.sedimentation),
                                                                 total_density(model.dynamics),
                                                                 transport_velocities(model).w)
    return Field(operation)
end

include("compressible_vector_invariant.jl")
include("hydrostatic_compressible_vector_invariant.jl")

#####
##### WENO with nonlinear weights frozen at the outer-step start
#####

# Reads the stored step-start tracer. A `Function` subtype, so WENO's `FunctionStencil` path
# loads its smoothness stencil from this field in place of the reconstructed one.
struct StepStartTracer{F} <: Function
    field :: F
end

@inline (c::StepStartTracer)(i, j, k, grid) = @inbounds c.field[i, j, k]

Adapt.adapt_structure(to, c::StepStartTracer) = StepStartTracer(adapt(to, c.field))
Architectures.on_architecture(to, c::StepStartTracer) = StepStartTracer(on_architecture(to, c.field))

Adapt.adapt_structure(to, VI::FunctionStencil{<:StepStartTracer}) = FunctionStencil(adapt(to, VI.func))
Architectures.on_architecture(to, VI::FunctionStencil{<:StepStartTracer}) = FunctionStencil(on_architecture(to, VI.func))

"""
$(TYPEDEF)

A scalar advection scheme that reconstructs with the stencils and optimal weights of a `WENO`
scheme, but with the nonlinear smoothness weights computed from the tracer at the start of the
outer time step rather than from the stage state being reconstructed. The fluxes stay live: only
the weights are frozen, so the stage reconstruction is linear in the stage tracer and the
[`MultirateInfinitesimalStep`](@ref Breeze.TimeSteppers.MultirateInfinitesimalStep) stage recombination sees one consistent advection operator
for the whole step, as it does with `UpwindBiased`. Under any other time stepper the frozen
tracer is refreshed before every tendency evaluation, and the scheme reduces to its `WENO`.

The scheme carries one `CenterField` per tracer, the tracer copy at the step start, which
`update_advection!` refreshes before the tendency evaluations that feed the first stage of a
step (`Breeze.TimeSteppers.tendencies_feed_first_stage`). It is a scalar
scheme: momentum has no step-start copy, so it cannot be given as `momentum_advection`.
Bounds preservation is not supported.

```jldoctest
using Breeze

FrozenWeightsWENO(order=9)

# output
FrozenWeightsWENO(order=9)
└── weno: WENO{5, Float64, Nothing}(order=9)
```
"""
struct FrozenWeightsWENO{N, FT, TD, W, S} <: AbstractUpwindBiasedAdvectionScheme{N, FT, TD}
    weno :: W                # stencils, smoothness indicators and optimal weights
    smoothness_stencil :: S  # `nothing` until materialized; then reads the step-start tracer

    function FrozenWeightsWENO(weno::WENO{N, FT, WCT, TD}, smoothness_stencil::S) where {N, FT, WCT, TD, S}
        isnothing(weno.bounds) ||
            throw(ArgumentError("FrozenWeightsWENO does not support bounds-preserving WENO."))
        return new{N, FT, TD, typeof(weno), S}(weno, smoothness_stencil)
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`FrozenWeightsWENO`](@ref) from `weno`, or from the `WENO` built with `FT` and the
keyword arguments of `WENO`.
"""
FrozenWeightsWENO(weno::WENO) = FrozenWeightsWENO(weno, nothing)
FrozenWeightsWENO(FT::DataType=Oceananigans.defaults.FloatType; kwargs...) = FrozenWeightsWENO(WENO(FT; kwargs...))

step_start_tracer(scheme::FrozenWeightsWENO) = scheme.smoothness_stencil.func.field

Base.summary(::FrozenWeightsWENO{N}) where N = string("FrozenWeightsWENO(order=", 2N-1, ")")

function Base.show(io::IO, scheme::FrozenWeightsWENO)
    print(io, summary(scheme), '\n')
    print(io, "└── weno: ", summary(scheme.weno))
end

Adapt.adapt_structure(to, scheme::FrozenWeightsWENO) =
    FrozenWeightsWENO(adapt(to, scheme.weno), adapt(to, scheme.smoothness_stencil))

Architectures.on_architecture(to, scheme::FrozenWeightsWENO) =
    FrozenWeightsWENO(on_architecture(to, scheme.weno), on_architecture(to, scheme.smoothness_stencil))

@inline OceananigansTimeSteppers.time_discretization(scheme::FrozenWeightsWENO) = time_discretization(scheme.weno)

# The step-start copy is allocated here, once per tracer, like the bounds-preserving limiter.
OceananigansAdvection.materialize_advection(scheme::FrozenWeightsWENO, grid) =
    FrozenWeightsWENO(materialize_advection(scheme.weno, grid), FunctionStencil(StepStartTracer(CenterField(grid))))

# Order adaptation on small grids happens after materialization in `AtmosphereModel`, so the
# lowered-order WENO is materialized here and the step-start copy is kept.
function OceananigansAdvection.adapt_advection_order(scheme::FrozenWeightsWENO{B}, N::Int, grid) where B
    N >= B && return scheme
    return FrozenWeightsWENO(materialize_advection(adapt_advection_order(scheme.weno, N, grid), grid), scheme.smoothness_stencil)
end

AtmosphereModels.validate_momentum_advection(::FrozenWeightsWENO, grid) =
    throw(ArgumentError("FrozenWeightsWENO is a scalar advection scheme; pass it as `scalar_advection` " *
                        "and give `momentum_advection` a scheme of its own."))

# The reconstruction: WENO's `FunctionStencil` path loads the stencil of `ψ` for the anchor and
# differences, and the stencil of the step-start tracer for the smoothness indicators. Routing
# through the topologically conditional `_biased_interpolate` of the inner scheme keeps the
# buffer-scheme fallback near boundaries, which passes the stencil on to the lower-order WENOs.
@inline OceananigansAdvection._biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::FrozenWeightsWENO, bias, ψ) =
    _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.weno, bias, ψ, scheme.smoothness_stencil)

@inline OceananigansAdvection._biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::FrozenWeightsWENO, bias, ψ) =
    _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.weno, bias, ψ, scheme.smoothness_stencil)

@inline OceananigansAdvection._biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::FrozenWeightsWENO, bias, ψ) =
    _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.weno, bias, ψ, scheme.smoothness_stencil)

# Refresh the step-start tracer copy only when the tendencies about to be built feed the first stage
# of an outer step. Halos are copied with the interior, so no halo fill follows.
function OceananigansAdvection.update_advection!(scheme::FrozenWeightsWENO, model, tracer)
    update_adaptive_timestep!(scheme, model)
    if tendencies_feed_first_stage(model.timestepper)
        copyto!(parent(step_start_tracer(scheme)), parent(tracer))
    end
    return nothing
end

end # module
