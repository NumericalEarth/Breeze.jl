#####
##### Adaptive implicit vertical advection (AIVA) for the anelastic, mass-flux formulation
#####
##### Oceananigans ≥ 0.110.8 provides the AIVA machinery for fields at z-Centers — scalars
##### (Center, Center, Center) and horizontal momentum (Face, Center, Center) / (Center, Face,
##### Center): CFL-scaled explicit fluxes, density-weighted implicit first-order-upwind
##### tridiagonal coefficients, and an `implicit_step!` that threads `(advection, velocities,
##### density)` into the vertically-implicit solve. Breeze passes the anelastic reference
##### density `ρ` so the implicit coefficients match the mass-flux form `∇·(ρ 𝐯 c)` used by
##### `div_ρUc`; with a horizontally-uniform reference density they are exact at all three
##### locations.
#####
##### Two mass-flux-specific pieces live in this file because upstream has no consumer for them:
#####
##### 1. Breeze advects momentum with the *mass flux* `(ρu, ρv, ρw)` as the advecting field
#####    (`div_𝐯u(advection, momentum, u)`), so Oceananigans' adaptive-implicit explicit-flux
#####    scaling — which computes the vertical CFL from the advecting field it is handed —
#####    would split on `|ρw|` instead of `|w|`, inconsistently with the implicit solve. The
#####    AIVA methods of `x/y/z_momentum_flux_divergence` below scale the vertical momentum
#####    flux with the *velocity* CFL instead, using the same interpolations as the implicit
#####    velocities so the explicit/implicit split is consistent.
#####
##### 2. Vertical momentum `ρw` lives at (Center, Center, Face). Oceananigans keeps the `Ww`
#####    flux fully explicit and has no implicit-advection coefficients for z-Face fields, so
#####    AIVA alone does not remove the `w ∂z w` CFL restriction — the limiting term in a deep
#####    convective updraft. The z-Face, density-weighted coefficients below fill that gap:
#####    the advecting velocity is `w` interpolated to cell centers, and the upwind flux at
#####    center k reconstructs the specific velocity `w = ρw / ℑz(ρ)`,
#####
#####      Fⁱ_k = Azᶜᶜᶜ_k ρ_k [ max(w̄ⁱ_k, 0) (ρw)_k / ℑz(ρ)_k + min(w̄ⁱ_k, 0) (ρw)_{k+1} / ℑz(ρ)_{k+1} ]
#####
#####    The tridiagonal row for face k spans centers k-1 and k (volume `Vᶜᶜᶠ`), mirroring the
#####    z-Face convention of the vertically-implicit diffusion solve.
#####
##### TODO (upstream): contribute the z-Face implicit-advection path (the coefficients,
##### `explicit_velocity_scaleᶜᶜᶜ`, and `get_coefficient` dispatch on `ℓz`) to Oceananigans,
##### so `NonhydrostaticModel` momentum AIVA treats `w` consistently too.

using Oceananigans.Advection:
    AdaptiveImplicitVerticalAdvection,
    vertical_scheme,
    densityᶜᶜᶜ,
    densityᶜᶜᶠ,
    explicit_velocity_scaleᶠᶜᶠ,
    explicit_velocity_scaleᶜᶠᶠ,
    advective_momentum_flux_Wu,
    advective_momentum_flux_Wv,
    advective_momentum_flux_Ww,
    _advective_momentum_flux_Uu,
    _advective_momentum_flux_Vu,
    _advective_momentum_flux_Uv,
    _advective_momentum_flux_Vv,
    _advective_momentum_flux_Uw,
    _advective_momentum_flux_Vw,
    _symmetric_interpolate_zᵃᵃᶜ

using Oceananigans.BoundaryConditions: _unwrap_for_gpu
using Oceananigans.Grids: Center, Face, ZDirection, peripheral_node
using Oceananigans.Operators: Az, volume,
                              V⁻¹ᶠᶜᶜ, V⁻¹ᶜᶠᶜ, V⁻¹ᶜᶜᶠ,
                              δxᶠᵃᵃ, δxᶜᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ
using Oceananigans.TimeSteppers: ExplicitTimeDiscretization, time_discretization
using Oceananigans.TurbulenceClosures:
    VerticallyImplicitDiffusionLowerDiagonal,
    VerticallyImplicitDiffusionDiagonal,
    VerticallyImplicitDiffusionUpperDiagonal,
    _ivd_lower_diagonal,
    _ivd_upper_diagonal,
    ivd_diagonal

const AIVA = AdaptiveImplicitVerticalAdvection

#####
##### Per-field advection lookup for the implicit step
#####

# Momentum prognostics share the single `:momentum` scheme; scalars are keyed by name.
@inline function field_advection_scheme(advection, name::Symbol)
    (name === :ρu || name === :ρv || name === :ρw) && return advection.momentum
    return haskey(advection, name) ? advection[name] : nothing
end

# `ρw` lives at (Center, Center, Face) and needs the Breeze-owned z-Face implicit-advection
# coefficients below; every other prognostic is at z-Centers and uses Oceananigans' coefficients.
implicit_step_advection(advection, name::Symbol) =
    name === :ρw ? VerticalMomentumImplicitAdvection(advection) : advection

# Density weighting the advective flux of each prognostic. Momentum and the thermodynamic
# variable are carried by the coupling density (`ρu = ρᵈ u`, `ρθ = ρᵈ θ`; see `dynamics_density`),
# while water species and tracers advect as mass fractions of the total density ρ = ρᵈ + Σρˣ
# (see `scalar_tendency`). The implicit solve must weight its upwind coefficients with the same
# density the explicit flux divergence uses; on the anelastic core the two densities coincide.
function implicit_advection_density(dynamics, formulation, name::Symbol)
    coupling = name === :ρu || name === :ρv || name === :ρw ||
               name === thermodynamic_density_name(formulation)
    return coupling ? dynamics_density(dynamics) : total_density(dynamics)
end

#####
##### Explicit vertical momentum fluxes scaled by the velocity CFL
#####
##### These mirror Oceananigans' adaptive-implicit flux scaling but compute the scale from the
##### velocity `w` rather than from the advecting mass flux `ρw`, matching the implicit
##### velocities `implicit_vertical_velocityᶠᶜᶠ/ᶜᶠᶠ` (and `implicit_vertical_velocityᶜᶜᶜ` below)
##### used by the tridiagonal solve.
#####

@inline function scaled_momentum_flux_Wu(i, j, k, grid, advection, W, u, w)
    scheme = vertical_scheme(advection)
    td = time_discretization(scheme)
    s = explicit_velocity_scaleᶠᶜᶠ(i, j, k, grid, scheme, td, w)
    return s * advective_momentum_flux_Wu(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, u)
end

@inline function scaled_momentum_flux_Wv(i, j, k, grid, advection, W, v, w)
    scheme = vertical_scheme(advection)
    td = time_discretization(scheme)
    s = explicit_velocity_scaleᶜᶠᶠ(i, j, k, grid, scheme, td, w)
    return s * advective_momentum_flux_Wv(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, v)
end

@inline function scaled_momentum_flux_Ww(i, j, k, grid, advection, W, w)
    scheme = vertical_scheme(advection)
    td = time_discretization(scheme)
    s = explicit_velocity_scaleᶜᶜᶜ(i, j, k, grid, scheme, td, w)
    return s * advective_momentum_flux_Ww(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, w)
end

# The AIVA methods reproduce `div_𝐯u/v/w` with the vertical flux routed through the
# velocity-CFL scaling above. Horizontal fluxes dispatch to the fully-explicit methods
# Oceananigans defines for the adaptive-implicit time discretization.
@inline function x_momentum_flux_divergence(i, j, k, grid, advection::AIVA, momentum, velocities, dynamics)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uu, advection, momentum[1], velocities.u) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vu, advection, momentum[2], velocities.u) +
                                    δzᵃᵃᶜ(i, j, k, grid, scaled_momentum_flux_Wu, advection, momentum[3], velocities.u, velocities.w)) +
           U_dot_∇u_metric(i, j, k, grid, advection, momentum, velocities)
end

@inline function y_momentum_flux_divergence(i, j, k, grid, advection::AIVA, momentum, velocities, dynamics)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uv, advection, momentum[1], velocities.v) +
                                    δyᵃᶠᵃ(i, j, k, grid, _advective_momentum_flux_Vv, advection, momentum[2], velocities.v) +
                                    δzᵃᵃᶜ(i, j, k, grid, scaled_momentum_flux_Wv, advection, momentum[3], velocities.v, velocities.w)) +
           U_dot_∇v_metric(i, j, k, grid, advection, momentum, velocities)
end

@inline function z_momentum_flux_divergence(i, j, k, grid, advection::AIVA, momentum, velocities, dynamics)
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uw, advection, momentum[1], velocities.w) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vw, advection, momentum[2], velocities.w) +
                                    δzᵃᵃᶠ(i, j, k, grid, scaled_momentum_flux_Ww, advection, momentum[3], velocities.w)) +
           U_dot_∇w_metric(i, j, k, grid, advection, momentum, velocities)
end

#####
##### Implicit vertical advection of vertical momentum (z-Face fields)
#####

# Breeze-owned wrapper selecting the z-Face implicit-advection coefficients in the
# `get_coefficient` methods below. Oceananigans' own adaptive-implicit `get_coefficient`
# methods do not dispatch on `ℓz` and would apply z-Center coefficients to `ρw`; wrapping
# the scheme (a Breeze-owned type in the signature) routes the `ρw` solve here without
# committing type piracy.
struct VerticalMomentumImplicitAdvection{A}
    scheme :: A
end

Adapt.adapt_structure(to, a::VerticalMomentumImplicitAdvection) =
    VerticalMomentumImplicitAdvection(adapt(to, a.scheme))

# Advecting velocity for `w ∂z w`: `w` interpolated to cell centers with the scheme's symmetric
# reconstruction (the same interpolation used by the explicit `Ww` flux), split by the target CFL
# with `Δzᶜᶜᶜ` — the hop between the faces where `w` lives.
@inline function explicit_velocity_scaleᶜᶜᶜ(i, j, k, grid, scheme, td, w)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    w̄  = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)
    α  = abs(w̄) * Δt / Δz
    return ifelse(α > td.cfl, td.cfl / α, one(α))
end

@inline function implicit_vertical_velocityᶜᶜᶜ(i, j, k, grid, scheme, td, w)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    w̄  = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)
    α  = abs(w̄) * Δt / Δz
    return w̄ * (1 - ifelse(α > td.cfl, td.cfl / α, one(α)))
end

#####
##### Tridiagonal coefficients for z-Face fields (rows are faces; fluxes live at cell centers)
#####
##### Row k is the control volume `Vᶜᶜᶠ(k)` around face k, bounded by the centers k-1 and k.
##### With the density-weighted upwind flux Fⁱ_k defined at the top of this file, the system
##### (I - Δt L) (ρw)ⁿ⁺¹ = (ρw)★ has row k
#####
#####   upper(k)  =   Δt / Vᶜᶜᶠ(k) Azᶜᶜᶜ(k)   ρ(k)   min(w̄ⁱ_k, 0)   / ℑz(ρ)_{k+1}
#####   lower(k′) = - Δt / Vᶜᶜᶠ(k) Azᶜᶜᶜ(k-1) ρ(k-1) max(w̄ⁱ_{k-1}, 0) / ℑz(ρ)_{k-1},  k = k′ + 1
#####   diag(k)   =   Δt / Vᶜᶜᶠ(k) [Azᶜᶜᶜ(k) ρ(k) max(w̄ⁱ_k, 0) - Azᶜᶜᶜ(k-1) ρ(k-1) min(w̄ⁱ_{k-1}, 0)] / ℑz(ρ)_k
#####
##### Boundary faces (`peripheral_node` at (Center, Center, Face)) reduce to identity rows, and
##### the coupling of row Nz to the untouched boundary face Nz+1 vanishes with `w` there.
#####

@inline function implicit_w_advection_upper_diagonal(i, j, k, grid, advection, w, ρ, Δt)
    scheme = vertical_scheme(advection)
    td     = time_discretization(scheme)
    w̄ⁱ     = implicit_vertical_velocityᶜᶜᶜ(i, j, k, grid, scheme, td, w)
    Azᵏ    = Az(i, j, k, grid, Center(), Center(), Center())
    ρᵏ     = densityᶜᶜᶜ(i, j, k, grid, ρ)
    ρᶠᵏ⁺¹  = densityᶜᶜᶠ(i, j, k+1, grid, ρ)
    V⁻¹    = 1 / volume(i, j, k, grid, Center(), Center(), Face())
    active = !peripheral_node(i, j, k, grid, Center(), Center(), Face()) &
             !peripheral_node(i, j, k, grid, Center(), Center(), Center())
    return Δt * V⁻¹ * Azᵏ * ρᵏ * min(w̄ⁱ, 0) / ρᶠᵏ⁺¹ * active
end

# Uses k′ = k - 1 (LinearAlgebra.Tridiagonal convention): the coefficient of (ρw)_{k′} in row k.
@inline function implicit_w_advection_lower_diagonal(i, j, k′, grid, advection, w, ρ, Δt)
    scheme = vertical_scheme(advection)
    td     = time_discretization(scheme)
    k      = k′ + 1
    w̄ⁱ     = implicit_vertical_velocityᶜᶜᶜ(i, j, k′, grid, scheme, td, w)
    Azᵏ⁻¹  = Az(i, j, k′, grid, Center(), Center(), Center())
    ρᵏ⁻¹   = densityᶜᶜᶜ(i, j, k′, grid, ρ)
    ρᶠᵏ⁻¹  = densityᶜᶜᶠ(i, j, k′, grid, ρ)
    V⁻¹    = 1 / volume(i, j, k, grid, Center(), Center(), Face())
    active = !peripheral_node(i, j, k, grid, Center(), Center(), Face()) &
             !peripheral_node(i, j, k′, grid, Center(), Center(), Center())
    return - Δt * V⁻¹ * Azᵏ⁻¹ * ρᵏ⁻¹ * max(w̄ⁱ, 0) / ρᶠᵏ⁻¹ * active
end

@inline function implicit_w_advection_diagonal(i, j, k, grid, advection, w, ρ, Δt)
    scheme = vertical_scheme(advection)
    td     = time_discretization(scheme)
    w̄ⁱ⁺    = implicit_vertical_velocityᶜᶜᶜ(i, j, k,   grid, scheme, td, w)
    w̄ⁱ⁻    = implicit_vertical_velocityᶜᶜᶜ(i, j, k-1, grid, scheme, td, w)

    Az⁺ = Az(i, j, k,   grid, Center(), Center(), Center())
    Az⁻ = Az(i, j, k-1, grid, Center(), Center(), Center())
    ρ⁺  = densityᶜᶜᶜ(i, j, k,   grid, ρ)
    ρ⁻  = densityᶜᶜᶜ(i, j, k-1, grid, ρ)
    ρᶠᵏ = densityᶜᶜᶠ(i, j, k, grid, ρ)

    active⁺ = !peripheral_node(i, j, k,   grid, Center(), Center(), Center())
    active⁻ = !peripheral_node(i, j, k-1, grid, Center(), Center(), Center())
    active  = !peripheral_node(i, j, k,   grid, Center(), Center(), Face())

    V⁻¹ = 1 / volume(i, j, k, grid, Center(), Center(), Face())

    return Δt * V⁻¹ / ρᶠᵏ * (Az⁺ * ρ⁺ * max(w̄ⁱ⁺, 0) * active⁺ -
                             Az⁻ * ρ⁻ * min(w̄ⁱ⁻, 0) * active⁻) * active
end

#####
##### get_coefficient seam for the ρw solve: diffusion (z-Face) + z-Face implicit advection
#####

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionUpperDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                         advection::VerticalMomentumImplicitAdvection, w, ρ)
    du_diff = _ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    du_adv  = implicit_w_advection_upper_diagonal(i, j, k, grid, advection.scheme, w, ρ, Δt)
    return du_diff + du_adv
end

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionLowerDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                         advection::VerticalMomentumImplicitAdvection, w, ρ)
    dl_diff = _ivd_lower_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    dl_adv  = implicit_w_advection_lower_diagonal(i, j, k, grid, advection.scheme, w, ρ, Δt)
    return dl_diff + dl_adv
end

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                         advection::VerticalMomentumImplicitAdvection, w, ρ)
    d_diff = ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    d_adv  = implicit_w_advection_diagonal(i, j, k, grid, advection.scheme, w, ρ, Δt)
    return d_diff + d_adv
end
