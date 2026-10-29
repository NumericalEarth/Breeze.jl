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

using Oceananigans: Oceananigans
using Oceananigans.Advection:
    _biased_interpolate_xᶠᵃᵃ,
    _biased_interpolate_yᵃᶠᵃ,
    bounds_preserving_limiter
using Oceananigans.Architectures: architecture, CPU, GPU
using Oceananigans.BoundaryConditions: FBC, fill_bottom_and_top_halo!
using Oceananigans.Fields: AbstractField
using Oceananigans.Grids: Bounded, Flat, topology, horizontally_extended_interior_kernel_parameters
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Operators: Axᶠᶜᶜ, Ayᶜᶠᶜ
using Oceananigans.Utils: ExplicitTimeDiscretization, launch!
using KernelAbstractions: @index, @kernel

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
    transport_velocities,
    AtmosphereModel

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


#####
##### Bounds-preserving WENO edge cache (port of the handoff's R3-T4/B1 idea onto Oceananigans' limiter field)
#####
#
# Oceananigans' bounds-preserving WENO computes, per cell, the six one-sided face reconstructions in the
# limiter pass (to form θ) and then recomputes all twelve reconstructions a cell's flux divergence needs
# in the tendency kernel, each rescaled by the θ of its donor cell. With the edge cache, the limiter pass
# stores the six *rescaled* reconstructions each cell donates, θᵢ (ĉ − cᵢ) + cᵢ, at the faces they belong
# to, and the scalar tendency only loads them. The rescaled values are formed by the same operations in
# the same order as `rescaled_reconstruction` in Oceananigans, from the same reconstruction calls, with
# θᵢ from Oceananigans' own `bounds_preserving_limiter`; the flux and divergence are Oceananigans'/Breeze's
# expressions verbatim. The pass runs on the limiter's range (interior plus the first horizontal halo
# layer), so the horizontal boundary faces are written by the halo cells exactly as their θ was formed;
# the vertical boundary faces (whose donor is the halo cell k = 0 or Nz + 1, with θ from the limiter
# field's `Flux` halo fill, θ[0] = θ[1], θ[Nz+1] = θ[Nz]) are written by the cells k = 1 and k = Nz.
# θ itself is still written and its vertical halos filled, so every other consumer (sedimentation
# mass fluxes, the bottom precipitation flux) is unchanged.
#
# The cache is held per scheme, in a host-side registry keyed by the limiter array (released when that
# array is finalized), so no model or scheme type changes. Switch: `BREEZE_EDGE_CACHE=false`.

const EDGE_CACHE = Ref(get(ENV, "BREEZE_EDGE_CACHE", "true") != "false")

const ExplicitBoundsPreservingWENO =
    WENO{<:Any, <:Any, <:Any, <:ExplicitTimeDiscretization, <:BoundsPreservation}

struct BoundsPreservingEdges{A}
    xᴸ :: A
    xᴿ :: A
    yᴸ :: A
    yᴿ :: A
    zᴸ :: A
    zᴿ :: A
end

Adapt.adapt_structure(to, e::BoundsPreservingEdges) =
    BoundsPreservingEdges(Adapt.adapt(to, e.xᴸ), Adapt.adapt(to, e.xᴿ), Adapt.adapt(to, e.yᴸ),
                          Adapt.adapt(to, e.yᴿ), Adapt.adapt(to, e.zᴸ), Adapt.adapt(to, e.zᴿ))

# Host-side per-scheme state.
mutable struct EdgeCacheState{E}
    edges :: Union{Nothing, E}  # nothing once the limiter array is finalized
    active :: Bool     # true once the limiter pass of this stage wrote the cache
    candidate :: Bool  # the scalar has been +0 in every entry at every scan so far (zero-scalar skip)
    zero :: Bool       # this stage's scan found it +0 everywhere: transport is +0, no limiter pass
end

#####
##### Whole-field zero-scalar skip (handoff T2, #201–#203/#227, in whole-field form)
#####
#
# While a scalar's specific field holds +0 in every entry of its parent array (halos included) and has
# done so at every scan since the model was built, its limiter pass is skipped and its tendency launch
# receives `ZeroTransport`, whose `div_ρUc` is the literal +0. That is the full path's bits for finite
# density and velocities: every rescaled reconstruction is θ(±0) + (+0) = +0, every upwind product
# ((u + |u|)(+0) + (u − |u|)(+0))/2 = +0, ρA(+0) = +0, and V⁻¹(+0 − +0 + …) = +0. The rest of the
# tendency (closure, forcing) is evaluated as before. θ is never written while skipped, and the full
# pass would write +0 into exactly the entries it writes (min(|1/ε|, |0/(−ε)|, 1) = +0; Flux halo fill
# copies it), so θ stays bitwise equal to its zero initialization, and θ's other consumers (sedimentation
# mass fluxes, bottom precipitation flux) see the same θ. The first scan that finds any entry whose bits
# are not +0 retires the scalar for good (no re-arm), and from then on the full pass runs, so the state
# when the field becomes nonzero is bitwise the full path's. Non-finite ρ or velocities at a zero cell
# would give NaN on the full path and +0 here (tier A for finite states, as in the handoff).
# Switch: `BREEZE_ZERO_SCALAR_SKIP=false`. Cost: one reduction and one device→host read per candidate
# scalar per stage, none once every scalar is retired.

const ZERO_SCALAR_SKIP = Ref(get(ENV, "BREEZE_ZERO_SCALAR_SKIP", "true") != "false")
const ZERO_SCALAR_SKIPS = Ref(0)

struct ZeroTransport end

@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::ZeroTransport, ρ, U, c) = zero(grid)

@inline is_plus_zero(x::Float32) = reinterpret(UInt32, x) == 0x00000000
@inline is_plus_zero(x::Float64) = reinterpret(UInt64, x) == 0x0000000000000000

all_plus_zero(c) = mapreduce(is_plus_zero, &, parent(c); init=true)

const EDGE_CACHE_STATES = Dict{UInt, Any}()
const EDGE_CACHE_LOCK = ReentrantLock()

limiter_key(θ) = objectid(parent(θ))

edge_cache_state(scheme) = lock(() -> get(EDGE_CACHE_STATES, limiter_key(scheme.bounds.limiter), nothing), EDGE_CACHE_LOCK)

edge_cache_eligible(scheme, grid) = false

function edge_cache_eligible(scheme::ExplicitBoundsPreservingWENO, grid)
    EDGE_CACHE[] || return false
    grid isa ImmersedBoundaryGrid && return false
    arch = architecture(grid)
    (arch isa CPU || arch isa GPU) || return false
    TX, TY, TZ = topology(grid)
    (TX === Flat || TY === Flat || TZ !== Bounded) && return false
    θ = scheme.bounds.limiter
    θ isa Field || return false
    bcs = θ.boundary_conditions
    return bcs.bottom isa FBC && bcs.top isa FBC
end

function new_edge_cache_state(scheme, grid)
    θ = scheme.bounds.limiter
    key = limiter_key(θ)
    make() = similar(θ.data)  # same axes as θ: x faces up to Nx + 2, z faces up to Nz + 1
    state = EdgeCacheState(BoundsPreservingEdges(make(), make(), make(), make(), make(), make()), false,
                           ZERO_SCALAR_SKIP[], false)
    lock(EDGE_CACHE_LOCK) do
        EDGE_CACHE_STATES[key] = state
    end
    finalizer(_ -> (state.edges = nothing; nothing), parent(θ))
    return state
end

# The limiter pass of an AtmosphereModel scalar. Falls back to Oceananigans' pass when ineligible.
function Oceananigans.Advection.update_advection!(scheme::ExplicitBoundsPreservingWENO, model::AtmosphereModel, c::AbstractField)
    grid = model.grid
    edge_cache_eligible(scheme, grid) ||
        return Oceananigans.Advection.update_bounds_preserving_limiter!(scheme, grid, c)
    state = edge_cache_state(scheme)
    if isnothing(state) || isnothing(state.edges)
        state = new_edge_cache_state(scheme, grid)
    end
    if state.candidate && all_plus_zero(c)
        state.zero = true
        state.active = false
        ZERO_SCALAR_SKIPS[] += 1
        return nothing
    end
    state.candidate = false
    state.zero = false
    update_edge_cache!(state, scheme, grid, c)
    return nothing
end

function update_edge_cache!(state, scheme, grid, c)
    θ = scheme.bounds.limiter
    parameters = horizontally_extended_interior_kernel_parameters(grid)
    launch!(architecture(grid), grid, parameters, _compute_bounds_preserving_edges!, state.edges, θ, grid, scheme, c)
    fill_bottom_and_top_halo!(θ.data, θ.boundary_conditions, θ.indices, (Center(), Center(), Center()), grid)
    state.active = true
    return nothing
end

@inline rescaled_edge(ĉ, θᵢ, cᵢ) = θᵢ * (ĉ - cᵢ) + cᵢ

@kernel function _compute_bounds_preserving_edges!(edges, θ, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    θᵢ = bounds_preserving_limiter(i, j, k, grid, scheme, c)

    cˣ⁻ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, scheme, RightBias, c)
    cˣ⁺ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, LeftBias,  c)
    cʸ⁻ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, scheme, RightBias, c)
    cʸ⁺ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, LeftBias,  c)
    cᶻ⁻ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, scheme, RightBias, c)
    cᶻ⁺ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, LeftBias,  c)

    @inbounds begin
        cᵢ = c[i, j, k]
        θ[i, j, k] = θᵢ
        edges.xᴿ[i,   j, k] = rescaled_edge(cˣ⁻, θᵢ, cᵢ)
        edges.xᴸ[i+1, j, k] = rescaled_edge(cˣ⁺, θᵢ, cᵢ)
        edges.yᴿ[i, j,   k] = rescaled_edge(cʸ⁻, θᵢ, cᵢ)
        edges.yᴸ[i, j+1, k] = rescaled_edge(cʸ⁺, θᵢ, cᵢ)
        edges.zᴿ[i, j, k  ] = rescaled_edge(cᶻ⁻, θᵢ, cᵢ)
        edges.zᴸ[i, j, k+1] = rescaled_edge(cᶻ⁺, θᵢ, cᵢ)
    end

    # Vertical boundary faces: donors are the halo cells, whose θ the `Flux` fill sets to θ[1] / θ[Nz].
    Nz = size(grid, 3)
    if k == 1
        ĉᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, 1, grid, scheme, LeftBias, c)
        @inbounds edges.zᴸ[i, j, 1] = rescaled_edge(ĉᴸ, θᵢ, c[i, j, 0])
    end
    if k == Nz
        ĉᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, Nz+1, grid, scheme, RightBias, c)
        @inbounds edges.zᴿ[i, j, Nz+1] = rescaled_edge(ĉᴿ, θᵢ, c[i, j, Nz+1])
    end
end

# What the scalar tendency launch receives in place of the scheme when the cache is current.
struct EdgeCachedAdvection{E}
    edges :: E
end

Adapt.adapt_structure(to, a::EdgeCachedAdvection) = EdgeCachedAdvection(Adapt.adapt(to, a.edges))

function AtmosphereModels.tendency_advection(scheme::ExplicitBoundsPreservingWENO)
    state = edge_cache_state(scheme)
    isnothing(state) && return scheme
    state.zero && return ZeroTransport()
    (isnothing(state.edges) || !state.active) && return scheme
    return EdgeCachedAdvection(state.edges)
end

@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::EdgeCachedAdvection, ρ, U, ::ZeroField) = zero(grid)

# Breeze's bounded `div_ρUc` with the face states loaded instead of reconstructed: same products, same order.
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, a::EdgeCachedAdvection, ρ, U, c)
    e = a.edges
    u, v, w = U.u, U.v, U.w
    @inbounds begin
        Ax⁺ = ℑxᶠᵃᵃ(i+1, j, k, grid, ρ) * Axᶠᶜᶜ(i+1, j, k, grid) * upwind_biased_product(u[i+1, j, k], e.xᴸ[i+1, j, k], e.xᴿ[i+1, j, k])
        Ax⁻ = ℑxᶠᵃᵃ(i,   j, k, grid, ρ) * Axᶠᶜᶜ(i,   j, k, grid) * upwind_biased_product(u[i,   j, k], e.xᴸ[i,   j, k], e.xᴿ[i,   j, k])
        Ay⁺ = ℑyᵃᶠᵃ(i, j+1, k, grid, ρ) * Ayᶜᶠᶜ(i, j+1, k, grid) * upwind_biased_product(v[i, j+1, k], e.yᴸ[i, j+1, k], e.yᴿ[i, j+1, k])
        Ay⁻ = ℑyᵃᶠᵃ(i, j,   k, grid, ρ) * Ayᶜᶠᶜ(i, j,   k, grid) * upwind_biased_product(v[i, j,   k], e.yᴸ[i, j,   k], e.yᴿ[i, j,   k])
        Az⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) * Azᶜᶜᶠ(i, j, k+1, grid) * upwind_biased_product(w[i, j, k+1], e.zᴸ[i, j, k+1], e.zᴿ[i, j, k+1])
        Az⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ) * Azᶜᶜᶠ(i, j, k,   grid) * upwind_biased_product(w[i, j, k],   e.zᴸ[i, j, k],   e.zᴿ[i, j, k])
    end
    div_x = Ax⁺ - Ax⁻
    div_y = Ay⁺ - Ay⁻
    div_z = Az⁺ - Az⁻
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

end # module
