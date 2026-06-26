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
#   * `HorizontalDivergence` (MPAS-faithful): horizontal VI advective part
#     ρ(ζ×𝐮 + ∇K)ₕ + flux-form vertical advection ∂z(ρw·u) + horizontal-divergence
#     correction −u ∇ₕ·(ρ𝐮).
#
#   * `ThreeDimensionalDivergence`: full velocity-form VI advective part ρ·U_dot_∇u
#     (includes vertical advection w∂z u) + total-divergence correction −u ∇·(ρ𝐮).
#
# Both are exact discretizations of −∇·(ρ𝐮⊗u) in the continuum; they differ in
# discrete conservation/upwinding. Coriolis is left as the separate `f×U` term in
# the momentum tendency (continuum-equivalent to folding f into the absolute
# vorticity ζ+f). Vertical momentum ρw stays flux form for both flavors, matching
# MPAS.

using Oceananigans.Advection: VectorInvariant, WENOVectorInvariant, U_dot_∇u, U_dot_∇v,
                              horizontal_advection_U, horizontal_advection_V,
                              bernoulli_head_U, bernoulli_head_V,
                              _advective_momentum_flux_Wu, _advective_momentum_flux_Wv
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, V⁻¹ᶠᶜᶜ, V⁻¹ᶜᶠᶜ, δzᵃᵃᶜ, div_xyᶜᶜᶜ, divᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid

#####
##### Divergence-treatment traits and the scheme wrapper
#####

"""
    HorizontalDivergence()

Vector-invariant flavor that advects the horizontal momentum with the
vector-invariant form in the horizontal only, treats vertical advection of
horizontal momentum in flux form, and corrects with the horizontal mass-flux
divergence ``∇ₕ·(ρ𝐮)``. This is the MPAS-Atmosphere choice.
"""
struct HorizontalDivergence end

"""
    ThreeDimensionalDivergence()

Vector-invariant flavor that uses the full velocity-form advection ``ρ(𝐮·∇)u``
(vertical advection included) and corrects with the total mass-flux divergence
``∇·(ρ𝐮)``.
"""
struct ThreeDimensionalDivergence end

"""
    CompressibleVectorInvariant{S, D}

Vector-invariant momentum advection for the compressible coupled-momentum (ρ𝐮)
equations. Holds an underlying Oceananigans vector-invariant `scheme` (supplying
the vorticity-flux and kinetic-energy-gradient operators) and a `divergence` trait
selecting the [`HorizontalDivergence`](@ref) (MPAS-faithful) or
[`ThreeDimensionalDivergence`](@ref) flavor. Construct with
[`CompressibleVectorInvariant(FT; ...)`](@ref) or
[`CompressibleWENOVectorInvariant`](@ref).
"""
struct CompressibleVectorInvariant{S, D}
    scheme :: S
    divergence :: D
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
                            divergence = HorizontalDivergence(), kwargs...) =
    CompressibleVectorInvariant(VectorInvariant(FT; kwargs...), divergence)

"""
    CompressibleWENOVectorInvariant(FT = Oceananigans.defaults.FloatType;
                                    divergence = ThreeDimensionalDivergence(), kwargs...)

WENO-based [`CompressibleVectorInvariant`](@ref). Mirrors Oceananigans'
[`WENOVectorInvariant`](@ref) and defaults to the total-divergence
([`ThreeDimensionalDivergence`](@ref)) flavor. Keyword arguments other than
`divergence` are forwarded to [`WENOVectorInvariant`](@ref).
"""
CompressibleWENOVectorInvariant(FT::DataType = Oceananigans.defaults.FloatType;
                                divergence = ThreeDimensionalDivergence(), kwargs...) =
    CompressibleVectorInvariant(WENOVectorInvariant(FT; kwargs...), divergence)

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
    CompressibleVectorInvariant(materialize_advection(a.scheme, grid), a.divergence)

#####
##### Momentum flux divergence: horizontal components dispatch on the flavor
#####

# TODO (vertical reconstruction): both the flux-form vertical advection of horizontal
# momentum and the z-momentum equation reconstruct with `Centered()` for now. The
# scheme should carry a deliberate momentum-flux reconstruction (e.g. matching WENO).
@inline compressible_vi_vertical_scheme(::CompressibleVectorInvariant) = Centered()

@inline function x_momentum_flux_divergence(i, j, k, grid, advection::CompressibleVectorInvariant,
                                            momentum, velocities, dynamics)
    ρ = dynamics_density(dynamics)
    return ℑxᶠᵃᵃ(i, j, k, grid, ρ) *
               x_vector_invariant_advection(i, j, k, grid, advection, velocities) +
           x_vertical_momentum_flux_divergence(i, j, k, grid, advection, momentum, velocities) +
           @inbounds(velocities.u[i, j, k]) *
               x_divergence_correction(i, j, k, grid, advection.divergence, momentum) +
           U_dot_∇u_metric(i, j, k, grid, advection.scheme, momentum, velocities)
end

@inline function y_momentum_flux_divergence(i, j, k, grid, advection::CompressibleVectorInvariant,
                                            momentum, velocities, dynamics)
    ρ = dynamics_density(dynamics)
    return ℑyᵃᶠᵃ(i, j, k, grid, ρ) *
               y_vector_invariant_advection(i, j, k, grid, advection, velocities) +
           y_vertical_momentum_flux_divergence(i, j, k, grid, advection, momentum, velocities) +
           @inbounds(velocities.v[i, j, k]) *
               y_divergence_correction(i, j, k, grid, advection.divergence, momentum) +
           U_dot_∇v_metric(i, j, k, grid, advection.scheme, momentum, velocities)
end

# Vertical momentum stays flux form (MPAS treats ρw in flux form) for both flavors,
# plus the nonhydrostatic w-curvature metric term (zero on rectilinear grids).
@inline z_momentum_flux_divergence(i, j, k, grid, advection::CompressibleVectorInvariant,
                                   momentum, velocities, dynamics) =
    div_𝐯w(i, j, k, grid, compressible_vi_vertical_scheme(advection), momentum, velocities.w) +
    U_dot_∇w_metric(i, j, k, grid, advection.scheme, momentum, velocities)

#####
##### Advective part: horizontal (MPAS) vs full velocity-form (3D)
#####

@inline x_vector_invariant_advection(i, j, k, grid, a::CompressibleVectorInvariant{<:Any, <:HorizontalDivergence}, U) =
    horizontal_advection_U(i, j, k, grid, a.scheme, U.u, U.v) +
    bernoulli_head_U(i, j, k, grid, a.scheme, U.u, U.v)

@inline y_vector_invariant_advection(i, j, k, grid, a::CompressibleVectorInvariant{<:Any, <:HorizontalDivergence}, U) =
    horizontal_advection_V(i, j, k, grid, a.scheme, U.u, U.v) +
    bernoulli_head_V(i, j, k, grid, a.scheme, U.u, U.v)

@inline x_vector_invariant_advection(i, j, k, grid, a::CompressibleVectorInvariant{<:Any, <:ThreeDimensionalDivergence}, U) =
    U_dot_∇u(i, j, k, grid, a.scheme, U)

@inline y_vector_invariant_advection(i, j, k, grid, a::CompressibleVectorInvariant{<:Any, <:ThreeDimensionalDivergence}, U) =
    U_dot_∇v(i, j, k, grid, a.scheme, U)

#####
##### Flux-form vertical advection of horizontal momentum (horizontal flavor only)
#####
##### For the 3D flavor vertical advection is already inside `U_dot_∇u`, so the
##### separate vertical term is zero.
#####

@inline x_vertical_momentum_flux_divergence(i, j, k, grid, a::CompressibleVectorInvariant{<:Any, <:HorizontalDivergence},
                                            momentum, velocities) =
    V⁻¹ᶠᶜᶜ(i, j, k, grid) *
        δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, compressible_vi_vertical_scheme(a), momentum.ρw, velocities.u)

@inline y_vertical_momentum_flux_divergence(i, j, k, grid, a::CompressibleVectorInvariant{<:Any, <:HorizontalDivergence},
                                            momentum, velocities) =
    V⁻¹ᶜᶠᶜ(i, j, k, grid) *
        δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, compressible_vi_vertical_scheme(a), momentum.ρw, velocities.v)

@inline x_vertical_momentum_flux_divergence(i, j, k, grid, ::CompressibleVectorInvariant{<:Any, <:ThreeDimensionalDivergence}, momentum, velocities) = zero(grid)
@inline y_vertical_momentum_flux_divergence(i, j, k, grid, ::CompressibleVectorInvariant{<:Any, <:ThreeDimensionalDivergence}, momentum, velocities) = zero(grid)

#####
##### Mass-divergence correction −u ∇·(ρ𝐮), interpolated to the velocity face
#####
##### Horizontal flavor uses ∇ₕ·(ρ𝐮); 3D flavor uses the total divergence ∇·(ρ𝐮).
##### TODO (3D flavor): replace the centered total divergence with a WENO-upwinded
##### reconstruction (the `CompressibleWENOVectorInvariant` `divergence_scheme`).
#####

@inline x_divergence_correction(i, j, k, grid, ::HorizontalDivergence, momentum) =
    ℑxᶠᵃᵃ(i, j, k, grid, div_xyᶜᶜᶜ, momentum.ρu, momentum.ρv)
@inline y_divergence_correction(i, j, k, grid, ::HorizontalDivergence, momentum) =
    ℑyᵃᶠᵃ(i, j, k, grid, div_xyᶜᶜᶜ, momentum.ρu, momentum.ρv)

@inline x_divergence_correction(i, j, k, grid, ::ThreeDimensionalDivergence, momentum) =
    ℑxᶠᵃᵃ(i, j, k, grid, divᶜᶜᶜ, momentum.ρu, momentum.ρv, momentum.ρw)
@inline y_divergence_correction(i, j, k, grid, ::ThreeDimensionalDivergence, momentum) =
    ℑyᵃᶠᵃ(i, j, k, grid, divᶜᶜᶜ, momentum.ρu, momentum.ρv, momentum.ρw)
