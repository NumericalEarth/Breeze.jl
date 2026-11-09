#####
##### Outflow limiting for bounds-preserving WENO under adaptive implicit vertical advection
#####
#
# Bounds-preserving WENO stays in bounds only for a forward-Euler update from the state its fluxes
# are built on. Wicker–Skamarock stages update from the step-start state `qⁿ` with fluxes built on
# the previous stage, so a cell filled by the previous implicit solve can export more than `qⁿ`
# holds. Each cell's outgoing face fluxes are therefore scaled by `r = min(1, available / outflow)`,
# a conservative zeroth-order flux-corrected transport limiter.

"""
$(TYPEDEF)

The `limiter` of a `BoundsPreservation` for bounds-preserving WENO under adaptive implicit vertical
advection: Oceananigans' reconstruction rescaling `θ`, which indexing forwards to, and the fraction
`r ∈ [0, 1]` of each cell's explicit outflow the coming stage may apply.
"""
struct OutflowLimiter{Θ, R, F}
    rescaling :: Θ             # Oceananigans' bounds-preserving reconstruction rescaling θ
    outflow_fraction :: R      # r, the fraction of each cell's explicit outflow the stage applies
    provisional_fraction :: R  # r from the previous pass, which scales the credited inflow
    face_fluxes :: F           # the stage's face fluxes `(; x, y, z)`, before the fractions scale them
end

OutflowLimiter(rescaling, r, r₀) = OutflowLimiter(rescaling, r, r₀, nothing)

@inline Base.getindex(l::OutflowLimiter, i, j, k) = @inbounds l.rescaling[i, j, k]
@inline Base.setindex!(l::OutflowLimiter, v, i, j, k) = @inbounds l.rescaling[i, j, k] = v

Adapt.adapt_structure(to, l::OutflowLimiter) =
    OutflowLimiter(adapt(to, l.rescaling), adapt(to, l.outflow_fraction), adapt(to, l.provisional_fraction), adapt(to, l.face_fluxes))

Oceananigans.Architectures.on_architecture(to, l::OutflowLimiter) =
    OutflowLimiter(Oceananigans.Architectures.on_architecture(to, l.rescaling),
                   Oceananigans.Architectures.on_architecture(to, l.outflow_fraction),
                   Oceananigans.Architectures.on_architecture(to, l.provisional_fraction),
                   Oceananigans.Architectures.on_architecture(to, l.face_fluxes))

BoundaryConditions.fill_halo_regions!(l::OutflowLimiter, args...; kwargs...) =
    fill_halo_regions!(l.rescaling, args...; kwargs...)

Base.summary(l::OutflowLimiter) = "OutflowLimiter"

const AVID = Oceananigans.Utils.AdaptiveVerticallyImplicitDiscretization

"A bounds-preserving WENO whose vertical time discretization is adaptive-implicit and whose outflow is limited."
const OutflowLimitedWENO = Oceananigans.Advection.WENO{<:Any, <:Any, <:Any, <:AVID, <:Oceananigans.Advection.BoundsPreservation{<:Any, <:OutflowLimiter}}

"""
$(TYPEDSIGNATURES)

Return `scheme` with an [`OutflowLimiter`](@ref) in its bounds-preservation slot if it is a
materialized bounds-preserving WENO with adaptive implicit vertical advection, otherwise `scheme`
unchanged.
"""
with_outflow_limiter(scheme, grid) = scheme


"""
$(TYPEDSIGNATURES)

Face fields `(; x, y, z)` in which `update_outflow_limiter!` stores a scalar's face fluxes once per
stage, for its passes and for the scalar's tendency to read instead of rebuilding them.
"""
outflow_limiter_face_fluxes(grid) = (; x = XFaceField(grid), y = YFaceField(grid), z = ZFaceField(grid))

function with_outflow_limiter(scheme::Oceananigans.Advection.WENO{N, FT, WCT, <:AVID, <:Oceananigans.Advection.BoundsPreservation}, grid) where {N, FT, WCT}
    bounds = scheme.bounds
    bounds.limiter isa OutflowLimiter && return scheme
    r = CenterField(grid)
    r₀ = CenterField(grid)
    fill!(r, 1)
    fill!(r₀, 1)
    limiter = OutflowLimiter(bounds.limiter, r, r₀, outflow_limiter_face_fluxes(grid))
    limited_bounds = Oceananigans.Advection.BoundsPreservation(bounds.minimum_value, bounds.maximum_value, bounds.maximum_courant_number, limiter)
    return Oceananigans.Advection.WENO{N, FT, WCT}(limited_bounds, scheme.buffer_scheme,
                                                   scheme.advecting_velocity_scheme,
                                                   scheme.time_discretization)
end

"""
$(TYPEDSIGNATURES)

Refresh the outflow fractions of `scheme` (see [`OutflowLimiter`](@ref)) for the scalar `name`,
before the tendencies that consume them. A no-op for every scheme without an `OutflowLimiter`.
"""
update_outflow_limiter!(scheme, model, name) = nothing

"""
$(TYPEDSIGNATURES)

Whether the coming stage of `timestepper` updates from the step-start state `timestepper.U⁰`
(`true`) or from the current state (`false`, the forward-Euler default).
"""
stage_starts_from_initial_state(timestepper, clock) = false
