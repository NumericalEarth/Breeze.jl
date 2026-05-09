#####
##### CompressibleVectorInvariant: vector-invariant momentum advection
##### for the conservative compressible formulation.
#####
##### The flux-form conservative momentum tendency expands as
#####
#####   ∇·(ρ𝐯⊗u) = ρ (𝐯·∇)u + u (∇·ρ𝐯)
#####
##### where (𝐯·∇)u is computed in vector-invariant form (vorticity flux +
##### Bernoulli + vertical advection) by an inner Oceananigans `VectorInvariant`
##### scheme. The vector-invariant decomposition absorbs the spherical
##### curvature metric into the vorticity flux, so the explicit
##### `U_dot_∇u_metric` correction is *not* applied.
#####
##### This is a fallback test scheme for diagnosing whether mid-latitude
##### spurious modes in the substepper are an interaction between flux-form
##### `div_𝐯u` discretization and the curvilinear grid.
#####

using Oceananigans.Advection: AbstractAdvectionScheme, U_dot_∇u, U_dot_∇v, WENOVectorInvariant, WENO,
                              materialize_advection
import Oceananigans.Advection: materialize_advection
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ∂zᶜᶜᶜ, div_xyᶜᶜᶜ
using Oceananigans.Advection: div_𝐯w,
                              U_dot_∇u_metric, U_dot_∇v_metric, U_dot_∇w_metric

"""
    CompressibleVectorInvariant(; vector_invariant=WENOVectorInvariant())

Conservative-form momentum advection that uses vector-invariant `(𝐯·∇)u`
for the rotational/Bernoulli decomposition. Computes

    ∇·(ρ𝐯⊗u) = ρ (𝐯·∇)u + u (∇·ρ𝐯)

at velocity faces, with the inner `(𝐯·∇)` provided by `vector_invariant`.

Use in place of a flux-form scheme on a conservative `CompressibleDynamics`
when investigating curvilinear-grid interactions with the substepper.
"""
struct CompressibleVectorInvariant{N, FT, VI, WS} <: AbstractAdvectionScheme{N, FT}
    vector_invariant :: VI    # for horizontal (u, v): vorticity flux + Bernoulli + vertical
    w_scheme         :: WS    # for w equation: flux-form (we still need ∇·(ρ𝐯⊗w))
end

function CompressibleVectorInvariant(; vector_invariant=WENOVectorInvariant(),
                                     w_scheme=WENO())
    N = max(Oceananigans.Grids.required_halo_size_x(vector_invariant),
            Oceananigans.Grids.required_halo_size_x(w_scheme))
    FT = eltype(vector_invariant)
    return CompressibleVectorInvariant{N, FT, typeof(vector_invariant), typeof(w_scheme)}(
        vector_invariant, w_scheme)
end

# Recurse into sub-schemes so WENOVectorInvariant's deferred WENO weight
# computation gets bound to the grid (otherwise `newton_div(::Type{Nothing},…)`
# fails inside the WENO smoothness indicators).
materialize_advection(scheme::CompressibleVectorInvariant, grid) =
    CompressibleVectorInvariant(
        vector_invariant = materialize_advection(scheme.vector_invariant, grid),
        w_scheme         = materialize_advection(scheme.w_scheme, grid))


# Mass-flux divergence at cell centers, used for the `u·(∇·ρ𝐯)` correction.
@inline _mass_flux_div_ccc(i, j, k, grid, ρu, ρv, ρw) =
    div_xyᶜᶜᶜ(i, j, k, grid, ρu, ρv) + ∂zᶜᶜᶜ(i, j, k, grid, ρw)

#####
##### Override `div_𝐯u`, `div_𝐯v`, `div_𝐯w` for the conservative-VI scheme.
#####
##### Signature matches Breeze's call: `div_𝐯u(i, j, k, grid, advection,
##### momentum, velocities.u)` where `momentum` is a NamedTuple of mass
##### fluxes (ρu, ρv, ρw). To call the inner `VectorInvariant` we need the
##### velocities — these are recovered from `momentum / ℑ(ρ_face)`. Since
##### Breeze passes `momentum` as a NamedTuple, we extract the velocities
##### from a sibling NamedTuple `velocities` injected via `kernel_args`.
#####
##### To keep the override local and avoid threading more arguments through
##### the existing tendency call sites, we adopt the convention that
##### `momentum` here is actually the **velocities** NamedTuple — Breeze's
##### `x_momentum_tendency` is patched to pass `velocities` instead of
##### `momentum` when the advection scheme is `CompressibleVectorInvariant`.
##### See `dynamics_kernel_functions.jl` for the dispatch logic.
#####

# ρ at the velocity-face is provided via the mass-flux divergence call,
# but the VI advection itself only needs velocities. The "u·∇·M" correction
# uses the mass fluxes directly. So we need BOTH momentum and velocities.
# We pack both via a small helper NamedTuple.

# Helper: full conservative VI tendency at u-face.
@inline function _compressible_vi_x(i, j, k, grid, scheme::CompressibleVectorInvariant,
                                    velocities, momentum, ρ_field)
    inner = scheme.vector_invariant
    vi_du = U_dot_∇u(i, j, k, grid, inner, velocities)        # at (f, c, c)
    ρ_x   = ℑxᶠᵃᵃ(i, j, k, grid, ρ_field)
    u_face = @inbounds velocities.u[i, j, k]
    div_M  = ℑxᶠᵃᵃ(i, j, k, grid, _mass_flux_div_ccc,
                   momentum.ρu, momentum.ρv, momentum.ρw)
    return ρ_x * vi_du + u_face * div_M
end

@inline function _compressible_vi_y(i, j, k, grid, scheme::CompressibleVectorInvariant,
                                    velocities, momentum, ρ_field)
    inner = scheme.vector_invariant
    vi_dv = U_dot_∇v(i, j, k, grid, inner, velocities)        # at (c, f, c)
    ρ_y   = ℑyᵃᶠᵃ(i, j, k, grid, ρ_field)
    v_face = @inbounds velocities.v[i, j, k]
    div_M  = ℑyᵃᶠᵃ(i, j, k, grid, _mass_flux_div_ccc,
                   momentum.ρu, momentum.ρv, momentum.ρw)
    return ρ_y * vi_dv + v_face * div_M
end

# Vertical: keep flux-form for w. Use the dedicated `w_scheme` (default WENO).
@inline function _compressible_vi_z(i, j, k, grid, scheme::CompressibleVectorInvariant,
                                    velocities, momentum, ρ_field)
    return div_𝐯w(i, j, k, grid, scheme.w_scheme, momentum, velocities.w)
end

# Metric-correction stubs: VI absorbs the spherical metric into the vorticity flux,
# so explicit corrections are zeroed. Two methods per direction: a generic one
# for non-curvilinear grids, plus a `grid::HCGOrIBG`-specific method that
# disambiguates against Oceananigans' curvilinear-grid `U_dot_∇u_metric` (Aqua
# flags the ambiguity otherwise).
import Oceananigans.Advection: U_dot_∇u_metric, U_dot_∇v_metric, U_dot_∇w_metric
using Oceananigans.Grids: AbstractHorizontallyCurvilinearGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
const _CVI_HCGOrIBG = Union{AbstractHorizontallyCurvilinearGrid,
                             ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any,
                                                   <:AbstractHorizontallyCurvilinearGrid}}

@inline U_dot_∇u_metric(i, j, k, grid, ::CompressibleVectorInvariant, U, V) = zero(grid)
@inline U_dot_∇v_metric(i, j, k, grid, ::CompressibleVectorInvariant, U, V) = zero(grid)
@inline U_dot_∇w_metric(i, j, k, grid, ::CompressibleVectorInvariant, U, V) = zero(grid)

@inline U_dot_∇u_metric(i, j, k, grid::_CVI_HCGOrIBG, ::CompressibleVectorInvariant, U, V) = zero(grid)
@inline U_dot_∇v_metric(i, j, k, grid::_CVI_HCGOrIBG, ::CompressibleVectorInvariant, U, V) = zero(grid)
@inline U_dot_∇w_metric(i, j, k, grid::_CVI_HCGOrIBG, ::CompressibleVectorInvariant, U, V) = zero(grid)

# Dispatch for the AtmosphereModel's `*_momentum_flux_divergence` indirection:
# replace `div_𝐯u + U_dot_∇u_metric` with the conservative-VI tendency.
@inline function AtmosphereModels.x_momentum_flux_divergence(
        i, j, k, grid, scheme::CompressibleVectorInvariant, momentum, velocities, dynamics)
    ρ_field = dynamics_density(dynamics)
    return _compressible_vi_x(i, j, k, grid, scheme, velocities, momentum, ρ_field)
end

@inline function AtmosphereModels.y_momentum_flux_divergence(
        i, j, k, grid, scheme::CompressibleVectorInvariant, momentum, velocities, dynamics)
    ρ_field = dynamics_density(dynamics)
    return _compressible_vi_y(i, j, k, grid, scheme, velocities, momentum, ρ_field)
end

@inline function AtmosphereModels.z_momentum_flux_divergence(
        i, j, k, grid, scheme::CompressibleVectorInvariant, momentum, velocities, dynamics)
    ρ_field = dynamics_density(dynamics)
    return _compressible_vi_z(i, j, k, grid, scheme, velocities, momentum, ρ_field)
end
