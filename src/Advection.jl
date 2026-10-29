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
    _ε₂,
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

using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: Architectures, architecture, on_architecture, device
using Oceananigans.BoundaryConditions: BoundaryCondition, Flux
using Oceananigans.Fields: Field, ZeroField, XFaceField, YFaceField, ZFaceField
using Oceananigans.Grids: Center, Periodic, Bounded, topology, horizontally_extended_interior_kernel_parameters
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, Axᶠᶜᶜ, Ayᶜᶠᶜ, Azᶜᶜᶠ
using Oceananigans.Utils: SumOfArrays, AdaptiveVerticallyImplicitDiscretization, ExplicitTimeDiscretization, launch!
using Oceananigans.TimeSteppers: time_discretization

using Adapt: Adapt
using DocStringExtensions: TYPEDEF, TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index
using KernelAbstractions.NDIteration: StaticSize

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
##### Bounds-preserving WENO with a face-reconstruction cache (T17 unit (e): port of handoff R3-T4 cd937b6d / B1
##### 9bc82a0a, ED-1 f63e27d6, ED-2 5d0b2022 + fd03ec02, and the k-first (32, 1, 4) launch of the edges pass, f3b942f
##### `edges_order`)
#####
##### Oceananigans' bounded flux divergence evaluates, per cell and direction, both one-sided reconstructions at both
##### faces (12 WENO reconstructions per cell) and rescales each with the limiter θ of its donor cell; the limiter pass
##### has already evaluated exactly these six per cell. So the limiter pass below (the "edges pass") stores, per cell,
##### its six one-sided reconstructions rescaled with its own θ, and the flux loads them:
#####
#####     xᴿ[i]   = θᵢ (c⁻ᵢ − cᵢ) + cᵢ    (the flux's c₋ᴿ of cell i and c₊ᴿ of cell i−1)
#####     xᴸ[i+1] = θᵢ (c⁺ᵢ − cᵢ) + cᵢ    (the flux's c₊ᴸ of cell i and c₋ᴸ of cell i+1)
#####
##### Every stored value is the same `_biased_interpolate` call on the same field, then `rescaled_reconstruction`'s
##### arithmetic with the θ Oceananigans' limiter would have stored, so the flux (`bounded_tracer_flux_divergence_*`'s
##### expressions verbatim, the reconstructions replaced by loads) is the old one bit for bit.
#####
##### Difference from the handoff's form: e9's Oceananigans computes θ also in the first horizontal halo layer
##### (`horizontally_extended_interior_kernel_parameters`), so the boundary faces of a bounded x/y direction are
##### rescaled with θ of the halo cell, not θ[0] = θ[1]. The edges pass therefore runs over the same extended range and
##### each cell writes only the faces it owns (no periodic images or boundary special cases in x and y). In z the
##### limiter's halo is a no-flux fill (θ[0] = θ[1], θ[Nz+1] = θ[Nz]; checked in `caches_edges`), so levels 1 and Nz
##### write the boundary faces as the handoff does. θ itself is not stored. The sedimentation readers
##### (`bounded_face_reconstructions`) load the z faces too; each scheme has its own cache, refreshed by
##### `update_advection!` as the limiter was (the handoff shares one cache and runs the pass at the tendency launch,
##### which e9's sedimentation and θ-tendency readers do not allow).
#####

"""
Whether bounds-preserving WENO schemes get a [`BoundsPreservingEdgeCache`](@ref) at model construction (`true` by
default; `BREEZE_EDGE_CACHE=false` at load turns it off; consulted when a model is built).
"""
const BOUNDS_PRESERVING_EDGE_CACHE = Ref(true)

"""
Whether the edge cache is also built on non-GPU architectures (`false` by default; `BREEZE_EDGE_CACHE_CPU=true` at load;
consulted when a model is built). On x86 CPUs Oceananigans' uncached bounded flux is not face-consistent at the bit
level: the `@muladd` (contract-flagged) WENO arithmetic is fused differently at the different code positions where the
two cells sharing a face reconstruct it (probe `benchmarks/gcp_t17scalar_face_consistency.jl`, job 5336: 37 of 1656
faces in Float32, 38 in Float64, differ between the two cells). A cache evaluates each face once, so it cannot
reproduce those bits; CPU models therefore keep Oceananigans' path, bit for bit. On GPUs the uncached path is
face-consistent and the cache is bit-identical.
"""
const EDGE_CACHE_ON_CPU = Ref(false)

env_switch(name, default) = lowercase(get(ENV, name, default)) in ("1", "true")

function __init__()
    BOUNDS_PRESERVING_EDGE_CACHE[] = env_switch("BREEZE_EDGE_CACHE", "true")
    EDGE_CACHE_ON_CPU[] = env_switch("BREEZE_EDGE_CACHE_CPU", "false")
    return nothing
end

"""
$(TYPEDEF)

θ quotients of the edges pass by IEEE division as written (Oceananigans' `bounds_preserving_limiter`).
"""
struct IEEELimiterQuotient end

limiter_quotient() = IEEELimiterQuotient()

# θ = min(|(cᵐᵃˣ − cᵢ) / (M − cᵢ + ε₂)|, |(cᵐⁱⁿ − cᵢ) / (m − cᵢ − ε₂)|, 1), the expressions of
# `bounds_preserving_limiter` in its statement order (fd03ec02)
@inline function limiter_theta(cᵢ, cᵐⁱⁿ, cᵐᵃˣ, m, M, ε₂, ::IEEELimiterQuotient)
    θᵐᵃˣ = abs((cᵐᵃˣ - cᵢ) / (M - cᵢ + ε₂))
    θᵐⁱⁿ = abs((cᵐⁱⁿ - cᵢ) / (m - cᵢ - ε₂))
    return min(θᵐᵃˣ, θᵐⁱⁿ, one(cᵢ))
end

"""
$(TYPEDEF)

The one-sided face reconstructions of a bounds-preserving WENO scheme, rescaled with the limiter of their donor cell:
`xᴸ[i, j, k]` holds the left-biased reconstruction at x-face `i` rescaled with the limiter of cell `i - 1`, `xᴿ[i, j, k]`
the right-biased one rescaled with the limiter of cell `i`; likewise in y and z. Filled by the edges pass, read by the
flux divergence and the sedimentation fluxes. It takes the place of the limiter field in `scheme.bounds.limiter`.
"""
struct BoundsPreservingEdgeCache{X, Y, Z, Q}
    xᴸ :: X
    xᴿ :: X
    yᴸ :: Y
    yᴿ :: Y
    zᴸ :: Z
    zᴿ :: Z
    quotient :: Q
end

function BoundsPreservingEdgeCache(grid, quotient)
    FT = eltype(grid)
    poisoned(field) = (fill!(parent(field), convert(FT, NaN)); field)
    return BoundsPreservingEdgeCache(poisoned(XFaceField(grid)), poisoned(XFaceField(grid)),
                                     poisoned(YFaceField(grid)), poisoned(YFaceField(grid)),
                                     poisoned(ZFaceField(grid)), poisoned(ZFaceField(grid)), quotient)
end

Adapt.adapt_structure(to, e::BoundsPreservingEdgeCache) =
    BoundsPreservingEdgeCache(Adapt.adapt(to, e.xᴸ), Adapt.adapt(to, e.xᴿ), Adapt.adapt(to, e.yᴸ),
                              Adapt.adapt(to, e.yᴿ), Adapt.adapt(to, e.zᴸ), Adapt.adapt(to, e.zᴿ), e.quotient)

Architectures.on_architecture(to, e::BoundsPreservingEdgeCache) =
    BoundsPreservingEdgeCache(on_architecture(to, e.xᴸ), on_architecture(to, e.xᴿ), on_architecture(to, e.yᴸ),
                              on_architecture(to, e.yᴿ), on_architecture(to, e.zᴸ), on_architecture(to, e.zᴿ),
                              e.quotient)

const EdgeCachedBoundsPreservingWENO =
    WENO{<:Any, <:Any, <:Any, <:ExplicitTimeDiscretization, <:BoundsPreservation{<:Any, <:BoundsPreservingEdgeCache}}

# The cached path covers explicit bounds-preserving WENO on a non-immersed grid, periodic or bounded in x and y and
# bounded in z, whose materialized limiter field fills its bottom and top halos with a no-flux (zero-gradient)
# condition, in the grid's float type.
caches_edges(scheme, grid) = false

periodic_or_bounded(T) = T === Periodic || T === Bounded

function caches_edges(scheme::BoundsPreservingWENO, grid)
    architecture(grid) isa Architectures.GPU || EDGE_CACHE_ON_CPU[] || return false
    θ = scheme.bounds.limiter
    θ isa Field || return false
    scheme.time_discretization isa ExplicitTimeDiscretization || return false
    grid isa ImmersedBoundaryGrid && return false
    TX, TY, TZ = topology(grid)
    (periodic_or_bounded(TX) && periodic_or_bounded(TY) && TZ === Bounded) || return false
    bcs = θ.boundary_conditions
    (bcs.bottom isa BoundaryCondition{<:Flux} && bcs.top isa BoundaryCondition{<:Flux}) || return false
    return eltype(scheme) === eltype(grid)
end

# `scheme` with an edge cache in place of its limiter field.
function with_edge_cache(scheme::WENO{N, FT, WCT}, grid) where {N, FT, WCT}
    bounds = scheme.bounds
    cache = BoundsPreservingEdgeCache(grid, limiter_quotient())
    cached_bounds = BoundsPreservation(bounds.minimum_value, bounds.maximum_value, bounds.maximum_courant_number, cache)
    return WENO{N, FT, WCT}(cached_bounds, scheme.buffer_scheme, scheme.advecting_velocity_scheme,
                            scheme.time_discretization)
end

function AtmosphereModels.with_reconstruction_caches(advection::NamedTuple, grid)
    BOUNDS_PRESERVING_EDGE_CACHE[] || return advection
    return map(scheme -> caches_edges(scheme, grid) ? with_edge_cache(scheme, grid) : scheme, advection)
end

# The edges pass replaces Oceananigans' limiter pass and its halo fill (θ is not stored).
function Oceananigans.Advection.update_bounds_preserving_limiter!(scheme::EdgeCachedBoundsPreservingWENO, grid, c)
    isnothing(c) && return nothing
    edges = scheme.bounds.limiter
    parameters = horizontally_extended_interior_kernel_parameters(grid)
    launch!(architecture(grid), grid, parameters, _compute_bounds_preserving_edges!, edges, grid, scheme, c)
    return nothing
end

@kernel function _compute_bounds_preserving_edges!(edges, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    bounds_preserving_edges!(edges, i, j, k, grid, scheme, c)
end

@inline rescaled_edge(ĉ, θᵢ, cᵢ) = θᵢ * (ĉ - cᵢ) + cᵢ

# `bounds_preserving_limiter`, keeping the six one-sided reconstructions it reduces and storing them rescaled
# (`reconstruction_extrema_{x,y,z}`'s calls and arithmetic, in order).
@inline function bounds_preserving_edges!(edges, i, j, k, grid, scheme, c)
    FT = eltype(c)
    ε₂ = convert(FT, _ε₂)
    Nz = size(grid, 3)

    cᵐⁱⁿ = scheme.bounds.minimum_value
    cᵐᵃˣ = scheme.bounds.maximum_value
    ω̂₁ = scheme.bounds.maximum_courant_number

    cᵢ = @inbounds c[i, j, k]

    cˣ⁻ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, scheme, RightBias, c)
    cˣ⁺ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, LeftBias,  c)
    cʸ⁻ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, scheme, RightBias, c)
    cʸ⁺ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, LeftBias,  c)
    cᶻ⁻ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, scheme, RightBias, c)
    cᶻ⁺ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, LeftBias,  c)

    m, M = cᵢ, cᵢ
    m, M = edge_extrema(cᵢ, cˣ⁻, cˣ⁺, m, M, ω̂₁)
    m, M = edge_extrema(cᵢ, cʸ⁻, cʸ⁺, m, M, ω̂₁)
    m, M = edge_extrema(cᵢ, cᶻ⁻, cᶻ⁺, m, M, ω̂₁)

    θᵢ = limiter_theta(cᵢ, cᵐⁱⁿ, cᵐᵃˣ, m, M, ε₂, edges.quotient)

    @inbounds begin
        edges.xᴿ[i,   j, k] = rescaled_edge(cˣ⁻, θᵢ, cᵢ)
        edges.xᴸ[i+1, j, k] = rescaled_edge(cˣ⁺, θᵢ, cᵢ)
        edges.yᴿ[i, j,   k] = rescaled_edge(cʸ⁻, θᵢ, cᵢ)
        edges.yᴸ[i, j+1, k] = rescaled_edge(cʸ⁺, θᵢ, cᵢ)
        edges.zᴿ[i, j, k]   = rescaled_edge(cᶻ⁻, θᵢ, cᵢ)
        edges.zᴸ[i, j, k+1] = rescaled_edge(cᶻ⁺, θᵢ, cᵢ)
    end

    # The bounded-z boundary faces, rescaled with θ[0] = θ[1] about c[0] and θ[Nz+1] = θ[Nz] about c[Nz+1]. A branch
    # around a store: its condition depends on k alone.
    if k == 1
        ĉ = _biased_interpolate_zᵃᵃᶠ(i, j, 1, grid, scheme, LeftBias, c)
        c₀ = @inbounds c[i, j, 0]
        @inbounds edges.zᴸ[i, j, 1] = rescaled_edge(ĉ, θᵢ, c₀)
    end
    if k == Nz
        ĉ = _biased_interpolate_zᵃᵃᶠ(i, j, Nz+1, grid, scheme, RightBias, c)
        c₊ = @inbounds c[i, j, Nz+1]
        @inbounds edges.zᴿ[i, j, Nz+1] = rescaled_edge(ĉ, θᵢ, c₊)
    end

    return nothing
end

# `reconstruction_extrema_{x,y,z}`'s arithmetic on given one-sided reconstructions c⁻, c⁺ of the cell value cᵢ.
@inline function edge_extrema(cᵢ, c⁻, c⁺, m, M, ω̂₁)
    p̃ = (cᵢ - ω̂₁ * (c⁻ + c⁺)) / (1 - 2ω̂₁)
    return min(m, p̃, c⁻, c⁺), max(M, p̃, c⁻, c⁺)
end

@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::EdgeCachedBoundsPreservingWENO, ρ, U, ::ZeroField) = zero(grid)

# `bounded_tracer_flux_divergence_{x,y,z}` with the four rescaled reconstructions loaded from the cache.
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection::EdgeCachedBoundsPreservingWENO, ρ, U, c)
    edges = advection.bounds.limiter
    div_x = cached_bounded_flux_divergence_x(i, j, k, grid, edges, ρ, U.u)
    div_y = cached_bounded_flux_divergence_y(i, j, k, grid, edges, ρ, U.v)
    div_z = cached_bounded_flux_divergence_z(i, j, k, grid, edges, ρ, U.w)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

@inline function cached_bounded_flux_divergence_x(i, j, k, grid, edges, ρ, u)
    c₊ᴸ = @inbounds edges.xᴸ[i+1, j, k]
    c₊ᴿ = @inbounds edges.xᴿ[i+1, j, k]
    c₋ᴸ = @inbounds edges.xᴸ[i,   j, k]
    c₋ᴿ = @inbounds edges.xᴿ[i,   j, k]

    u⁺ = @inbounds u[i+1, j, k]
    u⁻ = @inbounds u[i,   j, k]
    Ax_ρuc⁺ = ℑxᶠᵃᵃ(i+1, j, k, grid, ρ) * Axᶠᶜᶜ(i+1, j, k, grid) * upwind_biased_product(u⁺, c₊ᴸ, c₊ᴿ)
    Ax_ρuc⁻ = ℑxᶠᵃᵃ(i,   j, k, grid, ρ) * Axᶠᶜᶜ(i,   j, k, grid) * upwind_biased_product(u⁻, c₋ᴸ, c₋ᴿ)

    return Ax_ρuc⁺ - Ax_ρuc⁻
end

@inline function cached_bounded_flux_divergence_y(i, j, k, grid, edges, ρ, v)
    c₊ᴸ = @inbounds edges.yᴸ[i, j+1, k]
    c₊ᴿ = @inbounds edges.yᴿ[i, j+1, k]
    c₋ᴸ = @inbounds edges.yᴸ[i, j,   k]
    c₋ᴿ = @inbounds edges.yᴿ[i, j,   k]

    v⁺ = @inbounds v[i, j+1, k]
    v⁻ = @inbounds v[i, j,   k]
    Ay_ρvc⁺ = ℑyᵃᶠᵃ(i, j+1, k, grid, ρ) * Ayᶜᶠᶜ(i, j+1, k, grid) * upwind_biased_product(v⁺, c₊ᴸ, c₊ᴿ)
    Ay_ρvc⁻ = ℑyᵃᶠᵃ(i, j,   k, grid, ρ) * Ayᶜᶠᶜ(i, j,   k, grid) * upwind_biased_product(v⁻, c₋ᴸ, c₋ᴿ)

    return Ay_ρvc⁺ - Ay_ρvc⁻
end

@inline function cached_bounded_flux_divergence_z(i, j, k, grid, edges, ρ, w)
    c₊ᴸ = @inbounds edges.zᴸ[i, j, k+1]
    c₊ᴿ = @inbounds edges.zᴿ[i, j, k+1]
    c₋ᴸ = @inbounds edges.zᴸ[i, j, k]
    c₋ᴿ = @inbounds edges.zᴿ[i, j, k]

    w⁺ = @inbounds w[i, j, k+1]
    w⁻ = @inbounds w[i, j, k]
    Az_ρwc⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) * Azᶜᶜᶠ(i, j, k+1, grid) * upwind_biased_product(w⁺, c₊ᴸ, c₊ᴿ)
    Az_ρwc⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ) * Azᶜᶜᶠ(i, j, k,   grid) * upwind_biased_product(w⁻, c₋ᴸ, c₋ᴿ)

    return Az_ρwc⁺ - Az_ρwc⁻
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

@inline function AtmosphereModels.sedimentation_mass_fluxes(i, j, k, grid, advection, wᵗ, wˢ, q)
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

# The same four limited reconstructions, loaded from the edge cache (stored by the edges pass of the same
# `update_advection!`).
@inline function bounded_face_reconstructions(i, j, k, grid, advection::EdgeCachedBoundsPreservingWENO, c)
    edges = advection.bounds.limiter
    c₋ᴸ = @inbounds edges.zᴸ[i, j, k]
    c₋ᴿ = @inbounds edges.zᴿ[i, j, k]
    c₊ᴸ = @inbounds edges.zᴸ[i, j, k+1]
    c₊ᴿ = @inbounds edges.zᴿ[i, j, k+1]
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
