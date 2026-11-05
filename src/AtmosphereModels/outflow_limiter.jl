#####
##### Outflow limiting for bounds-preserving WENO under adaptive implicit vertical advection
#####
#
# A bounds-preserving WENO (`WENO(; bounds)`) keeps a tracer in its bounds only for a forward-Euler
# update *from the state its fluxes are built on*, at Courant number ≤ `maximum_courant_number`
# (5/18 by default). Breeze's scalar stages break both conditions:
#
#   1. Wicker–Skamarock RK3 updates every stage from the step-start state,
#      `qₖ = qⁿ + βₖ Δt G(qₖ₋₁)`, so the explicit outflow of a cell is built on `qₖ₋₁` but drawn
#      from `qⁿ`. Under adaptive implicit vertical advection the implicit solve of the previous
#      stage can carry mass into a cell that was empty at `tⁿ`; the explicit outflow of that mass
#      is then subtracted from `qⁿ ≈ 0` and the cell goes negative. This happens at any explicit
#      CFL target, and grows with the vertical Courant number.
#   2. The adaptive split caps the explicit vertical Courant number at `cfl` (0.5 by default),
#      not at `maximum_courant_number`.
#
# The fix is a flux-corrected (zero-th order FCT) outflow limiter: before the tendencies are
# built, each cell's total explicit outflow over the coming stage, `βΔt V⁻¹ Σ max(F_out, 0)`, is
# compared with the mass the stage update will subtract it from, and every outgoing face flux is
# scaled by `r = min(1, budget / outflow)` of its upwind cell. Each face flux leaves exactly one
# cell, so the scaling is conservative, and since the limited reconstructions are themselves
# non-negative the inflows are too: the stage update stays non-negative for any Courant number.
# The implicit remainder (first-order upwind backward Euler, an M-matrix) then keeps it so.
#
# The fractions `r` are cached next to the bounds-preserving rescaling `θ` in the scheme's
# `BoundsPreservation.limiter` slot, so the tendency kernels, which receive the scheme, read them
# without new kernel arguments.

"""
$(TYPEDEF)

The `limiter` of a `BoundsPreservation` carried by a bounds-preserving WENO under adaptive implicit
vertical advection. It holds Oceananigans' cell-wise reconstruction rescaling `θ` (`rescaling`,
which it forwards indexing to, so Oceananigans' bounded reconstructions read it unchanged) and the
cell-wise fraction `outflow_fraction` ``∈ [0, 1]`` by which the explicit outgoing face fluxes of
each cell are scaled so that a Runge–Kutta stage cannot remove more mass than the cell holds.

$(TYPEDFIELDS)
"""
struct OutflowLimiter{Θ, R}
    "Oceananigans' bounds-preserving reconstruction rescaling ``θ``"
    rescaling :: Θ
    "fraction of each cell's explicit outflow the coming stage may apply"
    outflow_fraction :: R
    "first-pass fraction from the outflow alone, a lower bound on `outflow_fraction` that bounds the inflow"
    provisional_fraction :: R
end

@inline Base.getindex(l::OutflowLimiter, i, j, k) = @inbounds l.rescaling[i, j, k]
@inline Base.setindex!(l::OutflowLimiter, v, i, j, k) = @inbounds l.rescaling[i, j, k] = v

Adapt.adapt_structure(to, l::OutflowLimiter) =
    OutflowLimiter(adapt(to, l.rescaling), adapt(to, l.outflow_fraction), adapt(to, l.provisional_fraction))

Oceananigans.Architectures.on_architecture(to, l::OutflowLimiter) =
    OutflowLimiter(Oceananigans.Architectures.on_architecture(to, l.rescaling),
                   Oceananigans.Architectures.on_architecture(to, l.outflow_fraction),
                   Oceananigans.Architectures.on_architecture(to, l.provisional_fraction))

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

function with_outflow_limiter(scheme::Oceananigans.Advection.WENO{N, FT, WCT, <:AVID, <:Oceananigans.Advection.BoundsPreservation}, grid) where {N, FT, WCT}
    bounds = scheme.bounds
    bounds.limiter isa OutflowLimiter && return scheme
    r = CenterField(grid)
    r₀ = CenterField(grid)
    fill!(r, 1)
    fill!(r₀, 1)
    limiter = OutflowLimiter(bounds.limiter, r, r₀)
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

Whether the coming stage of `timestepper` subtracts the scalar tendencies from the step-start state
`timestepper.U⁰` (`true`) or from the current state (`false`, the forward-Euler default). This
selects the mass budget of the outflow limiter.
"""
outflow_budget_from_initial_state(timestepper, clock) = false
