#####
##### Vector-invariant momentum advection for the compressible core (MPAS-style)
#####
#
# Breeze prognoses coupled momentum ρ𝐮. Verified against the MPAS source
# (`atm_compute_dyn_tend_work`), the horizontal momentum tendency is the split
#
#     −∇·(ρ𝐮⊗u) = −ρ(𝐮·∇)u − u ∇·(ρ𝐮),
#
# with the advective part ρ(𝐮·∇)u = ρ(ζ×𝐮 + ∇K) discretized in vector-invariant
# form. We expose this through `CompressibleVectorInvariant`, a thin wrapper around
# an Oceananigans `VectorInvariant`/`WENOVectorInvariant` carrying a `divergence`
# trait that selects the consistent pairing of vertical-advection treatment and
# mass-divergence correction (see design/vector_invariant_momentum.md):
#
#   * `HorizontalDivergence` (MPAS-faithful): mass-flux ζ₃ vorticity flux (the
#     prognostic ρ𝐮ₕ transports the vorticity, as in MPAS) + ρ-weighted horizontal
#     KE gradient + flux-form vertical advection ∂z(ρw·u) + horizontal-divergence
#     correction −u ∇ₕ·(ρ𝐮ₕ). Vertical momentum ρw stays flux form, matching MPAS.
#
#   * `ThreeDimensionalDivergence` (unsplit, design/compressible_orthogonal_cgrid_
#     weno_vi_with_hollingsworth.md): the mass-flux Lamb vector form
#
#         ∂t(ρ𝐮) = 𝐔×ζ − ρ∇K − 𝐮 ∇·𝐔 + ⋯,   𝐔 = ρ𝐮,  K = |𝐮|²/2,
#
#     with the *prognostic mass fluxes* transporting the vorticity (the same
#     fluxes continuity uses), the full 3D kinetic energy, the full 3D mass-flux
#     divergence, and a symmetric vector-invariant ρw equation. The density-
#     weighted (mass-flux) structure is the Hollingsworth control: weighting a
#     velocity-form operator by an outer interpolated ρ carries an indefinite
#     kinetic-energy residual where ρ varies.
#
#     NOTE: this flavor must NOT be assembled from Oceananigans' `U_dot_∇u`: the
#     upwind-vertical `vertical_advection_U` there rewrites w∂z u = ∂z(wu) + u∇ₕ·𝐮
#     using incompressible continuity, so `U_dot_∇u = (𝐮·∇)𝐮 + 𝐮(∇·𝐮)` for
#     WENOVectorInvariant — adding a divergence correction on top double-counts
#     an O(1) compressible term.
#
# Both flavors are exact discretizations of −∇·(ρ𝐮⊗u) in the continuum; they
# differ in discrete conservation/upwinding. Coriolis is left as the separate
# `f×U` term in the momentum tendency, transported by the same mass flux
# (continuum-equivalent to folding f into the absolute vorticity ζ+f, and it
# keeps planetary vorticity out of the WENO reconstruction).

using Oceananigans.Advection: VectorInvariant, WENOVectorInvariant,
                              VectorInvariantUpwindVorticity,
                              VectorInvariantSelfVerticalUpwinding,
                              VectorInvariantCrossVerticalUpwinding,
                              bernoulli_head_U, bernoulli_head_V,
                              _advective_momentum_flux_Wu, _advective_momentum_flux_Wv,
                              _biased_interpolate_xᶜᵃᵃ, _biased_interpolate_xᶠᵃᵃ,
                              _biased_interpolate_yᵃᶜᵃ, _biased_interpolate_yᵃᶠᵃ,
                              _symmetric_interpolate_xᶠᵃᵃ, _symmetric_interpolate_yᵃᶠᵃ,
                              bias, FunctionStencil, Khᶜᶜᶜ, ϕ²
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ,
                              V⁻¹ᶠᶜᶜ, V⁻¹ᶜᶠᶜ, V⁻¹ᶜᶜᶜ, δzᵃᵃᶜ, div_xyᶜᶜᶜ, divᶜᶜᶜ,
                              ζ₃ᶠᶠᶜ, Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Az_qᶜᶜᶠ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ,
                              Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, Az⁻¹ᶠᶜᶜ, Az⁻¹ᶜᶠᶜ,
                              δxᶜᶜᶜ, δyᶜᶜᶜ, δzᶜᶜᶜ,
                              ∂xᶠᶜᶠ, ∂zᶠᶜᶠ, ∂yᶜᶠᶠ, ∂zᶜᶠᶠ,
                              ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ
using Oceananigans.Grids: AbstractGrid

#####
##### Divergence-treatment traits and the scheme wrapper
#####

"""
    HorizontalDivergence()

Vector-invariant flavor that advects the horizontal momentum with the
vector-invariant form in the horizontal only — the ζ₃ vorticity flux transported
by the prognostic horizontal mass fluxes, plus the ρ-weighted horizontal
kinetic-energy gradient — treats vertical advection of horizontal momentum in
flux form, and corrects with the horizontal mass-flux divergence ``∇ₕ·(ρ𝐮ₕ)``.
This is the MPAS-Atmosphere choice (MPAS transports its rotational tendency
with the ρ𝐮 fluxes).
"""
struct HorizontalDivergence end

"""
    ThreeDimensionalDivergence()

Unsplit compressible vector-invariant flavor: the advective part of the coupled
momentum tendency is the mass-flux Lamb vector plus the density-weighted kinetic
energy gradient plus the full three-dimensional mass-flux divergence,

```math
∂_t(ρ𝐮) = 𝐔×ζ − ρ∇K − 𝐮 (∇·𝐔) + ⋯, \\qquad 𝐔 = ρ𝐮, \\quad K = |𝐮|²/2 ,
```

where the transporting mass fluxes are the prognostic momenta (the same fluxes
used by continuity) and all three momentum components, including ``ρw``, are
advected in vector-invariant form. See
`design/compressible_orthogonal_cgrid_weno_vi_with_hollingsworth.md`.

Reconstruction staging: the vertical-vorticity (ζ₃) fluxes and the mass-flux
divergence are upwinded when the wrapped scheme upwinds them; the horizontal-
vorticity (ζ₁, ζ₂) fluxes and the kinetic-energy gradient are centered.
"""
struct ThreeDimensionalDivergence end

"""
    CompressibleVectorInvariant{S, D, V}

Vector-invariant momentum advection for the compressible coupled-momentum (ρ𝐮)
equations. Holds an underlying Oceananigans vector-invariant `scheme` (supplying
the vorticity-flux reconstruction operators), a `vertical_scheme` for the flux-form
vertical advection of the [`HorizontalDivergence`](@ref) flavor (stored as a
materialized field, *not* built in-kernel, so the GPU kernel reads a concrete
scheme), and a `divergence` trait selecting the [`HorizontalDivergence`](@ref)
(MPAS-faithful) or [`ThreeDimensionalDivergence`](@ref) (unsplit mass-flux Lamb
vector) flavor. Construct with [`CompressibleVectorInvariant(FT; ...)`](@ref) or
[`CompressibleWENOVectorInvariant`](@ref).
"""
# NOTE: `divergence` stays the SECOND type parameter — the advective-part and divergence-correction
# methods dispatch on `CompressibleVectorInvariant{<:Any, <:HorizontalDivergence/ThreeDimensionalDivergence}`.
# `vertical_scheme` is appended LAST so it doesn't shift that positional dispatch.
struct CompressibleVectorInvariant{S, D, V}
    scheme :: S
    divergence :: D
    vertical_scheme :: V
end

"""
    CompressibleVectorInvariant(FT = Oceananigans.defaults.FloatType;
                                divergence = HorizontalDivergence(), kwargs...)

Construct a [`CompressibleVectorInvariant`](@ref) momentum advection scheme. The
`divergence` keyword selects the [`HorizontalDivergence`](@ref) (MPAS-faithful) or
[`ThreeDimensionalDivergence`](@ref) flavor. The remaining keyword arguments
configure the underlying vector-invariant reconstruction operators and are
forwarded to [`VectorInvariant`](@ref) (e.g. `vorticity_scheme`,
`vertical_advection_scheme`, `divergence_scheme`, `upwinding`).
"""
CompressibleVectorInvariant(FT::DataType = Oceananigans.defaults.FloatType;
                            divergence = HorizontalDivergence(),
                            vertical_scheme = Centered(FT),
                            kwargs...) =
    CompressibleVectorInvariant(VectorInvariant(FT; kwargs...), divergence, vertical_scheme)

"""
    CompressibleWENOVectorInvariant(FT = Oceananigans.defaults.FloatType;
                                    divergence = ThreeDimensionalDivergence(), kwargs...)

WENO-based [`CompressibleVectorInvariant`](@ref). Mirrors Oceananigans'
[`WENOVectorInvariant`](@ref) and defaults to the total-divergence
([`ThreeDimensionalDivergence`](@ref)) flavor. Keyword arguments other than
`divergence` are forwarded to [`WENOVectorInvariant`](@ref).
"""
CompressibleWENOVectorInvariant(FT::DataType = Oceananigans.defaults.FloatType;
                                divergence = ThreeDimensionalDivergence(),
                                vertical_scheme = Centered(FT),
                                kwargs...) =
    CompressibleVectorInvariant(WENOVectorInvariant(FT; kwargs...), divergence, vertical_scheme)

Base.summary(::HorizontalDivergence) = "HorizontalDivergence"
Base.summary(::ThreeDimensionalDivergence) = "ThreeDimensionalDivergence"
Base.summary(a::CompressibleVectorInvariant) =
    string("CompressibleVectorInvariant(", summary(a.scheme), ", divergence=", summary(a.divergence), ")")
Base.show(io::IO, a::CompressibleVectorInvariant) = print(io, summary(a))

# Delegate halo/order queries to the underlying scheme. The generic
# `required_halo_size_*` fallback returns 1, which would under-size the halo for the
# wide vector-invariant stencil, so we forward to `scheme`. Order adaptation is a
# no-op (as for a bare `VectorInvariant`).
@inline Oceananigans.Grids.required_halo_size_x(a::CompressibleVectorInvariant) = Oceananigans.Grids.required_halo_size_x(a.scheme)
@inline Oceananigans.Grids.required_halo_size_y(a::CompressibleVectorInvariant) = Oceananigans.Grids.required_halo_size_y(a.scheme)
@inline Oceananigans.Grids.required_halo_size_z(a::CompressibleVectorInvariant) = Oceananigans.Grids.required_halo_size_z(a.scheme)

Advection.adapt_advection_order(a::CompressibleVectorInvariant, grid::AbstractGrid) = a

# Materialize the underlying scheme (e.g. resolve a WENO scheme's `Nothing`
# weight-computation type to a concrete, architecture-dependent type). Without this
# the generic `materialize_advection` passthrough leaves the inner scheme
# unmaterialized and WENO reconstruction fails (`newton_div(::Type{Nothing}, …)`).
Advection.materialize_advection(a::CompressibleVectorInvariant, grid) =
    CompressibleVectorInvariant(materialize_advection(a.scheme, grid),
                                a.divergence,
                                materialize_advection(a.vertical_scheme, grid))

#####
##### Momentum flux divergence: horizontal components dispatch on the flavor
#####

# Vertical reconstruction for the flux-form vertical advection of horizontal momentum and the
# z-momentum equation of the HorizontalDivergence flavor (the 3D flavor advects vertically through
# its ζ₁/ζ₂ rotational fluxes instead). Carried as a materialized struct field (default `Centered`)
# — NOT constructed in-kernel: building a scheme object inside the GPU kernel is illegal IR
# (`apply_type`/`new_structv`).
# TODO (vertical reconstruction): default to a deliberate momentum-flux reconstruction (e.g. WENO).
@inline compressible_vi_vertical_scheme(a::CompressibleVectorInvariant) = a.vertical_scheme

const HorizontalCVI = CompressibleVectorInvariant{<:Any, <:HorizontalDivergence}
const ThreeDimensionalCVI = CompressibleVectorInvariant{<:Any, <:ThreeDimensionalDivergence}

# The ζ₃ vorticity flux is transported by the prognostic mass flux (Vζ₃ᶠᶜᶜ /
# Uζ₃ᶜᶠᶜ below) — the density-weighted structure that controls the Hollingsworth
# energy residual, and what MPAS actually does (its rotational tendency uses the
# ρu fluxes). The kinetic-energy gradient is ρ-weighted outside, which is the
# correct placement for ρ∇K.
@inline function x_momentum_flux_divergence(i, j, k, grid, advection::HorizontalCVI,
                                            momentum, velocities, dynamics)
    ρ = dynamics_density(dynamics)
    scheme = advection.scheme
    u, v = velocities.u, velocities.v
    return - Vζ₃ᶠᶜᶜ(i, j, k, grid, scheme, u, v, momentum.ρv) +
           ℑxᶠᵃᵃ(i, j, k, grid, ρ) * bernoulli_head_U(i, j, k, grid, scheme, u, v) +
           x_vertical_momentum_flux_divergence(i, j, k, grid, advection, momentum, velocities) +
           @inbounds(velocities.u[i, j, k]) *
               x_divergence_correction(i, j, k, grid, advection.divergence, momentum) +
           U_dot_∇u_metric(i, j, k, grid, scheme, momentum, velocities)
end

@inline function y_momentum_flux_divergence(i, j, k, grid, advection::HorizontalCVI,
                                            momentum, velocities, dynamics)
    ρ = dynamics_density(dynamics)
    scheme = advection.scheme
    u, v = velocities.u, velocities.v
    return + Uζ₃ᶜᶠᶜ(i, j, k, grid, scheme, u, v, momentum.ρu) +
           ℑyᵃᶠᵃ(i, j, k, grid, ρ) * bernoulli_head_V(i, j, k, grid, scheme, u, v) +
           y_vertical_momentum_flux_divergence(i, j, k, grid, advection, momentum, velocities) +
           @inbounds(velocities.v[i, j, k]) *
               y_divergence_correction(i, j, k, grid, advection.divergence, momentum) +
           U_dot_∇v_metric(i, j, k, grid, scheme, momentum, velocities)
end

# Vertical momentum stays flux form for the MPAS-faithful horizontal flavor,
# plus the nonhydrostatic w-curvature metric term (zero on rectilinear grids).
@inline z_momentum_flux_divergence(i, j, k, grid, advection::HorizontalCVI,
                                   momentum, velocities, dynamics) =
    div_𝐯w(i, j, k, grid, compressible_vi_vertical_scheme(advection), momentum, velocities.w) +
    U_dot_∇w_metric(i, j, k, grid, advection.scheme, momentum, velocities)

#####
##### Flux-form vertical advection of horizontal momentum (horizontal flavor only)
#####

@inline x_vertical_momentum_flux_divergence(i, j, k, grid, a::HorizontalCVI, momentum, velocities) =
    V⁻¹ᶠᶜᶜ(i, j, k, grid) *
        δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, compressible_vi_vertical_scheme(a), momentum.ρw, velocities.u)

@inline y_vertical_momentum_flux_divergence(i, j, k, grid, a::HorizontalCVI, momentum, velocities) =
    V⁻¹ᶜᶠᶜ(i, j, k, grid) *
        δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, compressible_vi_vertical_scheme(a), momentum.ρw, velocities.v)

#####
##### Mass-divergence correction −u ∇ₕ·(ρ𝐮ₕ) of the horizontal flavor
#####

@inline x_divergence_correction(i, j, k, grid, ::HorizontalDivergence, momentum) =
    ℑxᶠᵃᵃ(i, j, k, grid, div_xyᶜᶜᶜ, momentum.ρu, momentum.ρv)
@inline y_divergence_correction(i, j, k, grid, ::HorizontalDivergence, momentum) =
    ℑyᵃᶠᵃ(i, j, k, grid, div_xyᶜᶜᶜ, momentum.ρu, momentum.ρv)

#####
##### Three-dimensional (unsplit) compressible vector-invariant flavor
#####
##### Discretizes the flux divergence ∇·(ρ𝐮⊗𝐮) (which the tendency subtracts) as
#####
#####   x: −(Vζ₃)ᶠᶜᶜ + (Wζ₂)ᶠᶜᶜ + ρᶠᶜᶜ ∂x K + u ⟨∇·𝐔⟩ᶠᶜᶜ + metric
#####   y: +(Uζ₃)ᶜᶠᶜ − (Wζ₁)ᶜᶠᶜ + ρᶜᶠᶜ ∂y K + v ⟨∇·𝐔⟩ᶜᶠᶜ + metric
#####   z: +(Vζ₁)ᶜᶜᶠ − (Uζ₂)ᶜᶜᶠ + ρᶜᶜᶠ ∂z K + w ⟨∇·𝐔⟩ᶜᶜᶠ + metric
#####
##### with relative vorticity ζ₁ = ∂y w − ∂z v (cff), ζ₂ = ∂z u − ∂x w (fcf),
##### ζ₃ = ∂x v − ∂y u (ffc), full 3D kinetic energy K = |𝐮|²/2 (ccc), and the
##### full 3D mass-flux divergence ∇·𝐔 of the prognostic momenta — the same mass
##### fluxes used by continuity (design principle of
##### design/compressible_orthogonal_cgrid_weno_vi_with_hollingsworth.md).
#####
##### Only ζ₃ and ∇·𝐔 are upwinded (when the wrapped scheme upwinds them); the
##### ζ₁/ζ₂ fluxes and ∇K are centered. TODO (staged, per the design note):
##### WENO reconstruction of the ζ₁/ζ₂ fluxes biased by the vertical mass flux,
##### and of the kinetic-energy gradient.
#####

# Horizontal-axis relative vorticity components
@inline ζ₁ᶜᶠᶠ(i, j, k, grid, v, w) = ∂yᶜᶠᶠ(i, j, k, grid, w) - ∂zᶜᶠᶠ(i, j, k, grid, v)
@inline ζ₂ᶠᶜᶠ(i, j, k, grid, u, w) = ∂zᶠᶜᶠ(i, j, k, grid, u) - ∂xᶠᶜᶠ(i, j, k, grid, w)

# Vertical and full 3D kinetic energy at cell centers. The horizontal part of
# the KE gradient in the u/v equations goes through `bernoulli_head_U/V` (which
# upwinds it for WENO schemes — Stage 4 of the design note); only the w²/2
# contribution and the z-gradient (w equation) are centered.
@inline Kwᶜᶜᶜ(i, j, k, grid, w) = ℑzᵃᵃᶜ(i, j, k, grid, ϕ², w) / 2
@inline K³ᶜᶜᶜ(i, j, k, grid, u, v, w) = Khᶜᶜᶜ(i, j, k, grid, u, v) + Kwᶜᶜᶜ(i, j, k, grid, w)

# ζ₃ fluxes transported by the horizontal mass fluxes. The centered form mirrors
# Oceananigans' enstrophy-conserving stencil; the upwind form mirrors its
# vorticity upwinding, with the reconstruction biased by the mass-flux sign.
@inline function Vζ₃ᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariant, u, v, ρv)
    V̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, ρv) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    return V̂ * ℑyᵃᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v)
end

@inline function Vζ₃ᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantUpwindVorticity, u, v, ρv)
    Sζ = scheme.vorticity_stencil
    V̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, ρv) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ζᴿ = _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vorticity_scheme, bias(V̂), ζ₃ᶠᶠᶜ, Sζ, u, v)
    return V̂ * ζᴿ
end

@inline function Uζ₃ᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariant, u, v, ρu)
    Û = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, ρu) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    return Û * ℑxᶜᵃᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v)
end

@inline function Uζ₃ᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantUpwindVorticity, u, v, ρu)
    Sζ = scheme.vorticity_stencil
    Û = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, ρu) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ζᴿ = _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vorticity_scheme, bias(Û), ζ₃ᶠᶠᶜ, Sζ, u, v)
    return Û * ζᴿ
end

# ζ₂/ζ₁ fluxes transported by the vertical mass flux (centered, Az-weighted,
# mirroring Oceananigans' energy-conserving vertical stencil)
@inline Az_ρw_ζ₂ᶠᶜᶠ(i, j, k, grid, u, w, ρw) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, ρw) * ζ₂ᶠᶜᶠ(i, j, k, grid, u, w)
@inline Az_ρw_ζ₁ᶜᶠᶠ(i, j, k, grid, v, w, ρw) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, ρw) * ζ₁ᶜᶠᶠ(i, j, k, grid, v, w)

@inline Wζ₂ᶠᶜᶜ(i, j, k, grid, u, w, ρw) = ℑzᵃᵃᶜ(i, j, k, grid, Az_ρw_ζ₂ᶠᶜᶠ, u, w, ρw) * Az⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline Wζ₁ᶜᶠᶜ(i, j, k, grid, v, w, ρw) = ℑzᵃᵃᶜ(i, j, k, grid, Az_ρw_ζ₁ᶜᶠᶠ, v, w, ρw) * Az⁻¹ᶜᶠᶜ(i, j, k, grid)

# ζ₂/ζ₁ fluxes for the w equation, transported by the horizontal mass fluxes (centered)
@inline ρv_ζ₁ᶜᶠᶠ(i, j, k, grid, v, w, ρv) = ℑzᵃᵃᶠ(i, j, k, grid, ρv) * ζ₁ᶜᶠᶠ(i, j, k, grid, v, w)
@inline ρu_ζ₂ᶠᶜᶠ(i, j, k, grid, u, w, ρu) = ℑzᵃᵃᶠ(i, j, k, grid, ρu) * ζ₂ᶠᶜᶠ(i, j, k, grid, u, w)

@inline Vζ₁ᶜᶜᶠ(i, j, k, grid, v, w, ρv) = ℑyᵃᶜᵃ(i, j, k, grid, ρv_ζ₁ᶜᶠᶠ, v, w, ρv)
@inline Uζ₂ᶜᶜᶠ(i, j, k, grid, u, w, ρu) = ℑxᶜᵃᵃ(i, j, k, grid, ρu_ζ₂ᶠᶜᶠ, u, w, ρu)

# Full 3D mass-flux divergence interpolated to the momentum locations. For
# schemes with divergence-upwinding machinery, the own-direction flux difference
# is upwinded by the advected velocity component (self-upwinding, mirroring
# Oceananigans' `upwinded_divergence_flux_Uᶠᶜᶜ`) and the cross-direction
# differences are interpolated with the symmetric cross scheme; otherwise the
# whole divergence is interpolated centered. The vertical (ccf) interpolation is
# centered (TODO: staged z-upwinding per the design note).
const VectorInvariantUpwindDivergence = Union{VectorInvariantSelfVerticalUpwinding,
                                              VectorInvariantCrossVerticalUpwinding}

# Volume-normalized mass-flux divergence contributions (kg m⁻³ s⁻¹). Normalizing
# per cell BEFORE reconstruction keeps the WENO stencil values O(∇·𝐔) ~ 10⁻⁵;
# reconstructing the raw area-weighted differences (~10⁹ kg/s on a global grid)
# overflows Float32 in the squared smoothness indicators.
@inline δx_ρU(i, j, k, grid, ρu, ρv, ρw) = δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, ρu) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline δy_ρV(i, j, k, grid, ρu, ρv, ρw) = δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, ρv) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline δz_ρW(i, j, k, grid, ρu, ρv, ρw) = δzᶜᶜᶜ(i, j, k, grid, Az_qᶜᶜᶠ, ρw) * V⁻¹ᶜᶜᶜ(i, j, k, grid)

@inline δyz_ρU(i, j, k, grid, ρu, ρv, ρw) = δy_ρV(i, j, k, grid, ρu, ρv, ρw) + δz_ρW(i, j, k, grid, ρu, ρv, ρw)
@inline δxz_ρV(i, j, k, grid, ρu, ρv, ρw) = δx_ρU(i, j, k, grid, ρu, ρv, ρw) + δz_ρW(i, j, k, grid, ρu, ρv, ρw)

# Smoothness indicator for the divergence upwinding: the full 3D divergence.
@inline mass_divergence_smoothness(i, j, k, grid, ρu, ρv, ρw) =
    δx_ρU(i, j, k, grid, ρu, ρv, ρw) + δyz_ρU(i, j, k, grid, ρu, ρv, ρw)

const mass_divergence_stencil = FunctionStencil(mass_divergence_smoothness)

@inline mass_divergenceᶠᶜᶜ(i, j, k, grid, scheme, û, ρu, ρv, ρw) =
    ℑxᶠᵃᵃ(i, j, k, grid, divᶜᶜᶜ, ρu, ρv, ρw)
@inline mass_divergenceᶜᶠᶜ(i, j, k, grid, scheme, v̂, ρu, ρv, ρw) =
    ℑyᵃᶠᵃ(i, j, k, grid, divᶜᶜᶜ, ρu, ρv, ρw)
@inline mass_divergenceᶜᶜᶠ(i, j, k, grid, scheme, ŵ, ρu, ρv, ρw) =
    ℑzᵃᵃᶠ(i, j, k, grid, divᶜᶜᶜ, ρu, ρv, ρw)

@inline function mass_divergenceᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantUpwindDivergence, û, ρu, ρv, ρw)
    cross_scheme = scheme.upwinding.cross_scheme
    δˢ = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, cross_scheme, δyz_ρU, ρu, ρv, ρw)
    δᴿ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, bias(û),
                                   δx_ρU, mass_divergence_stencil, ρu, ρv, ρw)
    return δˢ + δᴿ
end

@inline function mass_divergenceᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantUpwindDivergence, v̂, ρu, ρv, ρw)
    cross_scheme = scheme.upwinding.cross_scheme
    δˢ = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, cross_scheme, δxz_ρV, ρu, ρv, ρw)
    δᴿ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, bias(v̂),
                                   δy_ρV, mass_divergence_stencil, ρu, ρv, ρw)
    return δˢ + δᴿ
end

@inline function x_momentum_flux_divergence(i, j, k, grid, advection::ThreeDimensionalCVI,
                                            momentum, velocities, dynamics)
    ρ = dynamics_density(dynamics)
    scheme = advection.scheme
    u, v, w = velocities.u, velocities.v, velocities.w
    ρu, ρv, ρw = momentum.ρu, momentum.ρv, momentum.ρw
    @inbounds û = u[i, j, k]
    ∂xK = bernoulli_head_U(i, j, k, grid, scheme, u, v) + ∂xᶠᶜᶜ(i, j, k, grid, Kwᶜᶜᶜ, w)
    return - Vζ₃ᶠᶜᶜ(i, j, k, grid, scheme, u, v, ρv) +
             Wζ₂ᶠᶜᶜ(i, j, k, grid, u, w, ρw) +
             ℑxᶠᵃᵃ(i, j, k, grid, ρ) * ∂xK +
             û * mass_divergenceᶠᶜᶜ(i, j, k, grid, scheme, û, ρu, ρv, ρw) +
             U_dot_∇u_metric(i, j, k, grid, scheme, momentum, velocities)
end

@inline function y_momentum_flux_divergence(i, j, k, grid, advection::ThreeDimensionalCVI,
                                            momentum, velocities, dynamics)
    ρ = dynamics_density(dynamics)
    scheme = advection.scheme
    u, v, w = velocities.u, velocities.v, velocities.w
    ρu, ρv, ρw = momentum.ρu, momentum.ρv, momentum.ρw
    @inbounds v̂ = v[i, j, k]
    ∂yK = bernoulli_head_V(i, j, k, grid, scheme, u, v) + ∂yᶜᶠᶜ(i, j, k, grid, Kwᶜᶜᶜ, w)
    return + Uζ₃ᶜᶠᶜ(i, j, k, grid, scheme, u, v, ρu) -
             Wζ₁ᶜᶠᶜ(i, j, k, grid, v, w, ρw) +
             ℑyᵃᶠᵃ(i, j, k, grid, ρ) * ∂yK +
             v̂ * mass_divergenceᶜᶠᶜ(i, j, k, grid, scheme, v̂, ρu, ρv, ρw) +
             U_dot_∇v_metric(i, j, k, grid, scheme, momentum, velocities)
end

@inline function z_momentum_flux_divergence(i, j, k, grid, advection::ThreeDimensionalCVI,
                                            momentum, velocities, dynamics)
    ρ = dynamics_density(dynamics)
    scheme = advection.scheme
    u, v, w = velocities.u, velocities.v, velocities.w
    ρu, ρv, ρw = momentum.ρu, momentum.ρv, momentum.ρw
    @inbounds ŵ = w[i, j, k]
    return + Vζ₁ᶜᶜᶠ(i, j, k, grid, v, w, ρv) -
             Uζ₂ᶜᶜᶠ(i, j, k, grid, u, w, ρu) +
             ℑzᵃᵃᶠ(i, j, k, grid, ρ) * ∂zᶜᶜᶠ(i, j, k, grid, K³ᶜᶜᶜ, u, v, w) +
             ŵ * mass_divergenceᶜᶜᶠ(i, j, k, grid, scheme, ŵ, ρu, ρv, ρw) +
             U_dot_∇w_metric(i, j, k, grid, scheme, momentum, velocities)
end
