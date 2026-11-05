module Advection

export div_ρUc,
       bottom_advective_tracer_flux

using Oceananigans.Advection:
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z,
    AdaptiveImplicitVerticalAdvection,
    _biased_interpolate_xᶠᵃᵃ,
    _biased_interpolate_yᵃᶠᵃ,
    _biased_interpolate_zᵃᵃᶠ,
    BoundsPreservation,
    BoundsPreservingWENO,
    LeftBias,
    RightBias,
    upwind_biased_product,
    rescaled_reconstruction,
    bounded_tracer_flux_divergence_x,
    bounded_tracer_flux_divergence_y,
    bounded_tracer_flux_divergence_z,
    WENO

using Oceananigans: Oceananigans
using Adapt: Adapt
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Grids: Center
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, Axᶠᶜᶜ, Ayᶜᶠᶜ, Azᶜᶜᶠ
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: AbstractGrid, Flat
using Oceananigans.Utils: launch!, sum_of_velocities
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: SumOfArrays, AdaptiveVerticallyImplicitDiscretization
using DocStringExtensions: TYPEDSIGNATURES

using ..AtmosphereModels:
    AtmosphereModels,
    div_ρUc,
    explicit_vertical_fraction,
    split_implicit_vertical_velocity,
    total_density,
    transport_velocities,
    microphysical_velocities,
    reconstructed_fields,
    OutflowLimitedWENO,
    outflow_budget_from_initial_state,
    stage_time_step

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
# functions consume it without duplicating their reconstruction (issue #913). The fraction also
# keeps the donor cell's outflow Courant number within the limiter's bound (see
# `explicit_vertical_fraction`), which depends on the density `ρ` the flux is weighted with.
struct ExplicitVerticalVelocity{G, S, W, R}
    grid :: G
    advection_scheme :: S
    vertical_velocity :: W
    density :: R
end

Adapt.adapt_structure(to, v::ExplicitVerticalVelocity) =
    ExplicitVerticalVelocity(Adapt.adapt(to, v.grid), Adapt.adapt(to, v.advection_scheme),
                             Adapt.adapt(to, v.vertical_velocity), Adapt.adapt(to, v.density))

@inline Base.getindex(v::ExplicitVerticalVelocity, i, j, k) =
    @inbounds explicit_vertical_fraction(i, j, k, v.grid, v.advection_scheme, v.vertical_velocity,
                                         v.density) * v.vertical_velocity[i, j, k]

# Disambiguates against the `ZeroField` shortcut above.
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::BoundsPreservingAVIDWENO, ρ, U, ::ZeroField) = zero(grid)

# Without the s-scaled velocity the bounded path transported 1 + (1 - s) times: a full
# explicit flux plus the implicit remainder (issue #913). Horizontal fluxes stay explicit.
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection::BoundsPreservingAVIDWENO, ρ, U, c)
    wᵉ = ExplicitVerticalVelocity(grid, advection, U.w, ρ)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, wᵉ, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# The vertical velocity the bounded flux functions below transport with: the velocity itself
# for an explicit scheme, its s-scaled explicit fraction under adaptive implicit vertical
# advection (the same `ExplicitVerticalVelocity` the bounded divergence uses, so the scaling is
# applied exactly once).
@inline explicit_vertical_velocity(advection, grid, w, ρ) = w
@inline explicit_vertical_velocity(advection::BoundsPreservingAVIDWENO, grid, w, ρ) =
    ExplicitVerticalVelocity(grid, advection, w, ρ)

#####
##### Outflow-limited bounded fluxes (see `AtmosphereModels.OutflowLimiter`)
#####
#
# Face fluxes formed exactly as `bounded_tracer_flux_divergence_{x,y,z}` form theirs (each
# reconstruction rescaled by θ of the cell it draws on), so with every outflow fraction equal
# to one the divergence below is the bounded divergence above, term by term.

@inline bounded_face_flux_x(i, j, k, ::AbstractGrid{FT, Flat}, advection, ρ, u, c) where FT = zero(FT)
@inline bounded_face_flux_y(i, j, k, ::AbstractGrid{FT, <:Any, Flat}, advection, ρ, v, c) where FT = zero(FT)

@inline function bounded_face_flux_x(i, j, k, grid, advection, ρ, u, c)
    θ = advection.bounds.limiter
    cᴸ = rescaled_reconstruction(_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, advection, LeftBias,  c), i-1, j, k, grid, θ, c)
    cᴿ = rescaled_reconstruction(_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, advection, RightBias, c), i,   j, k, grid, θ, c)
    @inbounds ũ = u[i, j, k]
    return ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Axᶠᶜᶜ(i, j, k, grid) * upwind_biased_product(ũ, cᴸ, cᴿ)
end

@inline function bounded_face_flux_y(i, j, k, grid, advection, ρ, v, c)
    θ = advection.bounds.limiter
    cᴸ = rescaled_reconstruction(_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, advection, LeftBias,  c), i, j-1, k, grid, θ, c)
    cᴿ = rescaled_reconstruction(_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, advection, RightBias, c), i, j,   k, grid, θ, c)
    @inbounds ṽ = v[i, j, k]
    return ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Ayᶜᶠᶜ(i, j, k, grid) * upwind_biased_product(ṽ, cᴸ, cᴿ)
end

@inline function bounded_face_flux_z(i, j, k, grid, advection, ρ, w, c)
    θ = advection.bounds.limiter
    cᴸ = rescaled_reconstruction(_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, advection, LeftBias,  c), i, j, k-1, grid, θ, c)
    cᴿ = rescaled_reconstruction(_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, advection, RightBias, c), i, j, k,   grid, θ, c)
    @inbounds w̃ = w[i, j, k]
    return ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Azᶜᶜᶠ(i, j, k, grid) * upwind_biased_product(w̃, cᴸ, cᴿ)
end

# Scale a face flux by the outflow fraction of its upwind cell: the one below/west/south of
# the face for a non-negative velocity, the one above/east/north otherwise.
@inline function limited_face_flux_x(i, j, k, grid, advection, ρ, u, c)
    r = advection.bounds.limiter.outflow_fraction
    @inbounds rᵘ = ifelse(u[i, j, k] ≥ 0, r[i-1, j, k], r[i, j, k])
    return rᵘ * bounded_face_flux_x(i, j, k, grid, advection, ρ, u, c)
end

@inline function limited_face_flux_y(i, j, k, grid, advection, ρ, v, c)
    r = advection.bounds.limiter.outflow_fraction
    @inbounds rᵘ = ifelse(v[i, j, k] ≥ 0, r[i, j-1, k], r[i, j, k])
    return rᵘ * bounded_face_flux_y(i, j, k, grid, advection, ρ, v, c)
end

@inline upwind_outflow_fraction_z(i, j, k, advection, w) =
    @inbounds ifelse(w[i, j, k] ≥ 0, advection.bounds.limiter.outflow_fraction[i, j, k-1],
                                     advection.bounds.limiter.outflow_fraction[i, j, k])

@inline limited_face_flux_z(i, j, k, grid, advection, ρ, w, c) =
    upwind_outflow_fraction_z(i, j, k, advection, w) * bounded_face_flux_z(i, j, k, grid, advection, ρ, w, c)

@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::OutflowLimitedWENO, ρ, U, ::ZeroField) = zero(grid)

@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection::OutflowLimitedWENO, ρ, U, c)
    wᵉ = ExplicitVerticalVelocity(grid, advection, U.w, ρ)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, limited_face_flux_x, advection, ρ, U.u, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, limited_face_flux_y, advection, ρ, U.v, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, limited_face_flux_z, advection, ρ, wᵉ, c))
end

# Mass the explicit fluxes of the coming stage remove from cell (i, j, k), per unit volume.
@inline function explicit_outflow(i, j, k, grid, advection, ρ, U, wᵉ, c)
    Fx⁻ = bounded_face_flux_x(i,   j, k, grid, advection, ρ, U.u, c)
    Fx⁺ = bounded_face_flux_x(i+1, j, k, grid, advection, ρ, U.u, c)
    Fy⁻ = bounded_face_flux_y(i, j,   k, grid, advection, ρ, U.v, c)
    Fy⁺ = bounded_face_flux_y(i, j+1, k, grid, advection, ρ, U.v, c)
    Fz⁻ = bounded_face_flux_z(i, j, k,   grid, advection, ρ, wᵉ, c)
    Fz⁺ = bounded_face_flux_z(i, j, k+1, grid, advection, ρ, wᵉ, c)
    outflow = max(0, Fx⁺) - min(0, Fx⁻) + max(0, Fy⁺) - min(0, Fy⁻) + max(0, Fz⁺) - min(0, Fz⁻)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * outflow
end

@kernel function _compute_outflow_fraction!(r, grid, advection, ρ, U, c, q⁰, from_initial_state, Δt)
    i, j, k = @index(Global, NTuple)
    td = time_discretization(advection)
    wᵉ = ExplicitVerticalVelocity(grid, advection, U.w, ρ)
    removed = Δt * explicit_outflow(i, j, k, grid, advection, ρ, U, wᵉ, c)
    @inbounds budget = max(0, ifelse(from_initial_state, q⁰[i, j, k], ρ[i, j, k] * c[i, j, k]))
    @inbounds r[i, j, k] = ifelse(isfinite(removed) & (removed > budget), budget / removed, one(budget))
end

# The velocity the scalar tendency of `name` advects with (see `scalar_tendency`).
@inline scalar_transport_velocities(model, name) =
    sum_of_velocities(transport_velocities(model),
                      microphysical_velocities(model.microphysics, model.microphysical_fields, Val(name)))

# Oceananigans fills θ through the `Field` interface, so refresh it on a twin of the scheme that
# carries the bare rescaling field.
function Oceananigans.Advection.update_bounds_preserving_limiter!(scheme::OutflowLimitedWENO, grid, c)
    bounds = scheme.bounds
    rescaling_bounds = BoundsPreservation(bounds.minimum_value, bounds.maximum_value,
                                          bounds.maximum_courant_number, bounds.limiter.rescaling)
    twin = WENO{weno_parameters(scheme)...}(rescaling_bounds, scheme.buffer_scheme,
                                            scheme.advecting_velocity_scheme, scheme.time_discretization)
    return Oceananigans.Advection.update_bounds_preserving_limiter!(twin, grid, c)
end

@inline weno_parameters(::WENO{N, FT, WCT}) where {N, FT, WCT} = (N, FT, WCT)
function AtmosphereModels.update_outflow_limiter!(advection::OutflowLimitedWENO, model, name)
    grid = model.grid
    r = advection.bounds.limiter.outflow_fraction
    c = reconstructed_fields(model, model.advection)[name]
    ρ = total_density(model.dynamics)
    U = scalar_transport_velocities(model, name)
    q⁰ = getproperty(model.timestepper.U⁰, name)
    from_initial_state = outflow_budget_from_initial_state(model.timestepper, model.clock)
    Δt = stage_time_step(model.timestepper, model.clock)
    launch!(architecture(grid), grid, :xyz, _compute_outflow_fraction!, r, grid, advection, ρ, U, c, q⁰, from_initial_state, Δt)
    fill_halo_regions!(r)
    return nothing
end

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

@inline function AtmosphereModels.sedimentation_mass_fluxes(i, j, k, grid, advection, wᵗ, wˢ, q, ρ)
    w = SumOfArrays{2}(wᵗ, wˢ)
    F⁻ = (_advective_tracer_flux_z(i, j, k,   grid, advection, w,  q),
          _advective_tracer_flux_z(i, j, k,   grid, advection, wᵗ, q))
    F⁺ = (_advective_tracer_flux_z(i, j, k+1, grid, advection, w,  q),
          _advective_tracer_flux_z(i, j, k+1, grid, advection, wᵗ, q))
    return F⁻, F⁺
end

# Bounds-preserving WENO rescales its face reconstructions by the cached limiter of the cell
# each one draws on, so the fluxes of cell k are rebuilt from the same limited reconstructions
# `bounded_tracer_flux_divergence_z` forms its mass fluxes from: the latent heat stays with the
# mass at cloud and precipitation edges, where the limiter acts and the unlimited WENO fluxes
# would move heat the tracer tendency does not move.
@inline AtmosphereModels.sedimentation_mass_fluxes(i, j, k, grid, advection::BoundsPreservingWENO, wᵗ, wˢ, q, ρ) =
    bounded_sedimentation_mass_fluxes(i, j, k, grid, advection, wᵗ, wˢ, q, ρ)

@inline function bounded_sedimentation_mass_fluxes(i, j, k, grid, advection, wᵗ, wˢ, q, ρ)
    w = SumOfArrays{2}(wᵗ, wˢ)
    c₋ᴸ, c₋ᴿ, c₊ᴸ, c₊ᴿ = bounded_face_reconstructions(i, j, k, grid, advection, q)
    F⁻ = (bounded_sedimentation_mass_flux(i, j, k,   grid, advection, w,  q, ρ, c₋ᴸ, c₋ᴿ),
          bounded_sedimentation_mass_flux(i, j, k,   grid, advection, wᵗ, q, ρ, c₋ᴸ, c₋ᴿ))
    F⁺ = (bounded_sedimentation_mass_flux(i, j, k+1, grid, advection, w,  q, ρ, c₊ᴸ, c₊ᴿ),
          bounded_sedimentation_mass_flux(i, j, k+1, grid, advection, wᵗ, q, ρ, c₊ᴸ, c₊ᴿ))
    return F⁻, F⁺
end

# Outflow-limited bounded WENO: the tracer tendency moves each face flux scaled by the outflow
# fraction of the upwind cell of the combined velocity, so both fluxes of a face take that
# factor and their difference, the sedimentation flux, stays the part the tendency moves.
@inline function AtmosphereModels.sedimentation_mass_fluxes(i, j, k, grid, advection::OutflowLimitedWENO, wᵗ, wˢ, q, ρ)
    w = SumOfArrays{2}(wᵗ, wˢ)
    F⁻, F⁺ = bounded_sedimentation_mass_fluxes(i, j, k, grid, advection, wᵗ, wˢ, q, ρ)
    r⁻ = upwind_outflow_fraction_z(i, j, k,   advection, w)
    r⁺ = upwind_outflow_fraction_z(i, j, k+1, advection, w)
    return r⁻ .* F⁻, r⁺ .* F⁺
end

# Face flux from the limited reconstructions `cᴸ`, `cᴿ`, with the explicit CFL scaling that
# `div_ρUc` applies under adaptive implicit vertical advection.
@inline function bounded_sedimentation_mass_flux(i, j, k, grid, advection, w, q, ρ, cᴸ, cᴿ)
    explicit_w = explicit_vertical_velocity(advection, grid, w, ρ)
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
    return explicit_flux + implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ, ρ_face, w, c)
end

@inline implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ, ρ_face, w, c) = 0

@inline function implicit_bottom_advective_tracer_flux(i, j, grid, advection::AIVA, ρ, ρ_face, w, c)
    wⁱ = split_implicit_vertical_velocity(i, j, 1, grid, advection, w, ρ)
    @inbounds c_above = c[i, j, 1]
    return -ρ_face * min(wⁱ, 0) * c_above
end

# Bounds-preserving WENO: the bottom face flux of cell 1 as `bounded_tracer_flux_divergence_z`
# forms it, from the reconstructions rescaled by the cached limiter of each donor cell.
@inline function bottom_advective_tracer_flux(i, j, grid, advection::BoundsPreservingWENO, ρ, w, c)
    c₋ᴸ, c₋ᴿ, _, _ = bounded_face_reconstructions(i, j, 1, grid, advection, c)
    explicit_w = explicit_vertical_velocity(advection, grid, w, ρ)
    @inbounds w⁻ = explicit_w[i, j, 1]
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * upwind_biased_product(w⁻, c₋ᴸ, c₋ᴿ)
    return explicit_flux + implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ, ρ_face, w, c)
end

# Outflow-limited: the explicit bottom-face flux is scaled as `div_ρUc` scales it.
@inline function bottom_advective_tracer_flux(i, j, grid, advection::OutflowLimitedWENO, ρ, w, c)
    explicit_w = explicit_vertical_velocity(advection, grid, w, ρ)
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = - limited_face_flux_z(i, j, 1, grid, advection, ρ, explicit_w, c) / Azᶜᶜᶠ(i, j, 1, grid)
    return explicit_flux + implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ, ρ_face, w, c)
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

end # module
