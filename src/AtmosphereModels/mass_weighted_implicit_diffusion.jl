#####
##### Mass-flux weighting of the vertically-implicit diffusion coefficients
#####
##### The prognostic is `q = ρ c`, but the flux the explicit path forms is `ρ κ ∂z c`
##### (`Jᶜz`/`𝒯_uz` in src/TurbulenceClosures/TurbulenceClosures.jl weight the kinematic flux by
##### `ℑz(ρ)`). Backward Euler on `∂t q = ∂z(ρ κ ∂z (q/ρ))` gives row k
#####
#####   du(k)  = - Δt κₖ₊₁ (ρᶠₖ₊₁ / ρᶜₖ₊₁) Δz⁻¹ᶜₖ Δz⁻¹ᶠₖ₊₁
#####   dl(k′) = - Δt κₖ   (ρᶠₖ   / ρᶜₖ₋₁) Δz⁻¹ᶜₖ Δz⁻¹ᶠₖ,          k = k′ + 1
#####   d(k)   =   1 + Δt Δz⁻¹ᶜₖ [κₖ₊₁ ρᶠₖ₊₁ Δz⁻¹ᶠₖ₊₁ + κₖ ρᶠₖ Δz⁻¹ᶠₖ] / ρᶜₖ
#####
##### The density ratio is the *only* difference from upstream, and it multiplies the whole
##### κ-sum — so each coefficient is the upstream one times a ratio, and closure tuples (whose
##### κ-sum happens inside `_ivd_*_diagonal`, see Oceananigans' closure_tuples.jl) and
##### `closure === nothing` keep working unchanged.
#####
##### The `ρᶜ` factor belongs to the *column*, not the row: `du(k)` divides by `ρᶜₖ₊₁` while the
##### diagonal divides by `ρᶜₖ`. So the diagonal must be written out explicitly rather than as
##### `1 - du - dl`, which is only conservative when the off-diagonals carry no location-dependent
##### prefactor.
#####
##### `ρw` (z-Face) takes the same rule with the locations exchanged. Its specific variable is
##### `w = ρw / ρᶠ`, reconstructed at faces, and its stress `ρ ν ∂z w` lives at centers — the
##### opposite of a tracer — so where the z-Center rows carry `ρᶠ / ρᶜ`, the z-Face rows carry
##### `ρᶜ / ρᶠ`. Backward Euler on `∂t q = ∂z(ρ ν ∂z (q/ρ))` at face k gives
#####
#####   du(k)  = - Δt νₖ (ρᶜₖ   / ρᶠₖ₊₁) Δz⁻¹ᶜₖ Δz⁻¹ᶠₖ
#####   dl(m)  = - Δt νₘ (ρᶜₘ   / ρᶠₘ)   Δz⁻¹ᶜₘ Δz⁻¹ᶠₘ₊₁,        k = m + 1
#####   d(k)   =   1 + Δt Δz⁻¹ᶠₖ [νₖ ρᶜₖ Δz⁻¹ᶜₖ + νₖ₋₁ ρᶜₖ₋₁ Δz⁻¹ᶜₖ₋₁] / ρᶠₖ
#####
##### which is the same `ρᶜ / ρᶠ` placement upstream's z-Face *advection* coefficients already use.
##### The diagonal is written out for the same reason as at z-Centers, with `ρᶠ` in place of `ρᶜ`.
#####
##### The row also carries the implicit-advection contribution for adaptive-implicit schemes, so
##### this file is included after `implicit_vertical_advection.jl`, whose `AIVA` alias it uses.
#####

using Oceananigans.Advection:
    densityᶜᶜᶜ,
    densityᶜᶜᶠ,
    implicit_advection_upper_diagonal,
    implicit_advection_lower_diagonal,
    implicit_advection_diagonal

using Oceananigans.Grids: Center, Face, ZDirection
using Oceananigans.TurbulenceClosures:
    VerticallyImplicitDiffusionLowerDiagonal,
    VerticallyImplicitDiffusionDiagonal,
    VerticallyImplicitDiffusionUpperDiagonal,
    _implicit_linear_coefficient,
    _ivd_lower_diagonal,
    _ivd_upper_diagonal,
    boundary_flux_diagonal

# Breeze-owned wrapper routing a z-Center prognostic's implicit solve to the mass-flux-weighted
# coefficients below. Wrapping the scheme puts a Breeze-owned type in the `get_coefficient`
# signature, so these methods are neither type piracy nor ambiguous with Oceananigans' own.
#
# `diffusion_density` weights the diffusion half of the row, leaving the density the solve is
# called with to the advection half. The two are the same field everywhere except in
# `implicit_substep!`, which freezes the stage-entry `(w, ρᵈ)` so that the explicit and implicit
# halves of the *advective* flux partition one transport (see `cache_advecting_state!`). The
# diffusion half has no such pairing to keep: it reconstructs the specific variable as `q / ρᵈ`
# from the field it is solving for, which the acoustic loop has already advanced, so it needs the
# live `ρᵈ`. `nothing` means "use the density the solve was called with".
struct MassWeightedImplicitDiffusion{A, D}
    scheme :: A
    diffusion_density :: D
end

MassWeightedImplicitDiffusion(scheme) = MassWeightedImplicitDiffusion(scheme, nothing)

Adapt.adapt_structure(to, a::MassWeightedImplicitDiffusion) =
    MassWeightedImplicitDiffusion(adapt(to, a.scheme), adapt(to, a.diffusion_density))

# `implicit_step!` sees the wrapper, not the scheme, when it decides whether to solve at all.
BoundaryConditions.needs_implicit_solver(a::MassWeightedImplicitDiffusion) =
    BoundaryConditions.needs_implicit_solver(a.scheme)

# Dispatch rather than a branch: `nothing` defers to the density the solve was called with.
@inline mass_weighting_density(::Nothing, ρ) = ρ
@inline mass_weighting_density(ρᵈ, ρ) = ρᵈ

# The implicit-advection contribution, which is present only for adaptive-implicit schemes.
@inline mass_weighted_advection_upper_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ) = zero(grid)
@inline mass_weighted_advection_lower_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ) = zero(grid)
@inline mass_weighted_advection_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ) = zero(grid)

@inline mass_weighted_advection_upper_diagonal(i, j, k, grid, scheme::AIVA, w, Δt, ℓx, ℓy, ℓz, ρ) =
    implicit_advection_upper_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ)
@inline mass_weighted_advection_lower_diagonal(i, j, k, grid, scheme::AIVA, w, Δt, ℓx, ℓy, ℓz, ρ) =
    implicit_advection_lower_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ)
@inline mass_weighted_advection_diagonal(i, j, k, grid, scheme::AIVA, w, Δt, ℓx, ℓy, ℓz, ρ) =
    implicit_advection_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ)

# As with the advection coefficients, `ρ` is interpolated in z only, so these are exact for
# a horizontally-uniform density (the anelastic reference state) at all three z-Center locations.
# The batched solver evaluates the off-diagonals only for k = 1 … Nz-1, so the divisions below
# never reach a halo value of `ρ`; the diagonal's `ρᶠ⁺` at k = Nz+1 does, but it only multiplies
# a `du` that the peripheral-node mask has already zeroed.
@inline function mass_weighted_ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz::Center, Δt, clk, fields, ρ)
    du = _ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    ρᶠ = densityᶜᶜᶠ(i, j, k+1, grid, ρ)   # where κₖ₊₁ weights the flux
    ρᶜ = densityᶜᶜᶜ(i, j, k+1, grid, ρ)   # where qₖ₊₁ is reconstructed as cₖ₊₁ = qₖ₊₁/ρᶜₖ₊₁
    return du * ρᶠ / ρᶜ
end

# `k′ = k - 1` (LinearAlgebra.Tridiagonal convention): the coefficient of `q(k′)` in row `k′ + 1`.
@inline function mass_weighted_ivd_lower_diagonal(i, j, k′, grid, clo, K, id, ℓx, ℓy, ℓz::Center, Δt, clk, fields, ρ)
    dl = _ivd_lower_diagonal(i, j, k′, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    ρᶠ = densityᶜᶜᶠ(i, j, k′+1, grid, ρ)
    ρᶜ = densityᶜᶜᶜ(i, j, k′,   grid, ρ)
    return dl * ρᶠ / ρᶜ
end

@inline function mass_weighted_ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz::Center, Δt, clk, fields, ρ)
    du  = _ivd_upper_diagonal(i, j, k,   grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    dl  = _ivd_lower_diagonal(i, j, k-1, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    lin = _implicit_linear_coefficient(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)

    ρᶜ  = densityᶜᶜᶜ(i, j, k,   grid, ρ)
    ρᶠ⁺ = densityᶜᶜᶠ(i, j, k+1, grid, ρ)
    ρᶠ⁻ = densityᶜᶜᶠ(i, j, k,   grid, ρ)

    # Both off-diagonal fluxes act on qₖ here, so both divide by ρᶜₖ — not by the neighbors'
    # ρᶜ, which is why this is not `1 - du - dl`. A linear coefficient damps q and c at the
    # same rate, so it is unweighted.
    return one(grid) - Δt * lin - (du * ρᶠ⁺ + dl * ρᶠ⁻) / ρᶜ
end

#####
##### z-Face mirrors (`ρw`). Locations exchange: `ρᶜ` where the stress lives, `ρᶠ` where `w` is
##### reconstructed. Row 1's `ρᶜ` at k-1 = 0 multiplies a `dl` the peripheral-node mask has already
##### zeroed, but its `ρᶠ₁` divides, so that row reads `ρ`'s bottom halo unmasked — as upstream's
##### z-Face advection diagonal already does.
#####

@inline function mass_weighted_ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz::Face, Δt, clk, fields, ρ)
    du = _ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    ρᶜ = densityᶜᶜᶜ(i, j, k,   grid, ρ)   # where νₖ weights the stress
    ρᶠ = densityᶜᶜᶠ(i, j, k+1, grid, ρ)   # where qₖ₊₁ is reconstructed as wₖ₊₁ = qₖ₊₁/ρᶠₖ₊₁
    return du * ρᶜ / ρᶠ
end

# `m = k - 1` (LinearAlgebra.Tridiagonal convention): the stress between faces `m` and `m + 1`
# sits at center `m`, which is also where `qₘ`'s reconstruction divides by `ρᶠₘ`.
@inline function mass_weighted_ivd_lower_diagonal(i, j, m, grid, clo, K, id, ℓx, ℓy, ℓz::Face, Δt, clk, fields, ρ)
    dl = _ivd_lower_diagonal(i, j, m, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    ρᶜ = densityᶜᶜᶜ(i, j, m, grid, ρ)
    ρᶠ = densityᶜᶜᶠ(i, j, m, grid, ρ)
    return dl * ρᶜ / ρᶠ
end

@inline function mass_weighted_ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz::Face, Δt, clk, fields, ρ)
    du  = _ivd_upper_diagonal(i, j, k,   grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    dl  = _ivd_lower_diagonal(i, j, k-1, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    lin = _implicit_linear_coefficient(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)

    ρᶠ  = densityᶜᶜᶠ(i, j, k,   grid, ρ)
    ρᶜ⁺ = densityᶜᶜᶜ(i, j, k,   grid, ρ)
    ρᶜ⁻ = densityᶜᶜᶜ(i, j, k-1, grid, ρ)

    # Both stresses act on qₖ here, so both divide by this row's ρᶠₖ — not by the neighbors'.
    return one(grid) - Δt * lin - (du * ρᶜ⁺ + dl * ρᶜ⁻) / ρᶠ
end

#####
##### get_coefficient seam: mass-weighted diffusion + implicit advection, at both z-locations
#####
##### Oceananigans ≥ 0.110.20 appends the field's top, bottom and immersed boundary conditions after
##### `(advection, w, density)`, and its own fallbacks absorb them with `args...`. A method that
##### fixes the trailing count at three stops matching, and the solve silently falls through to the
##### *unweighted* diffusion-only fallback — the weighting vanishes with no error. These signatures
##### mirror upstream's AIVA methods, including the implicit-explicit boundary-flux diagonal.
#####

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionUpperDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                         advection::MassWeightedImplicitDiffusion, w, ρ, args...)
    ρᵈ = mass_weighting_density(advection.diffusion_density, ρ)
    du_diff = mass_weighted_ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, ρᵈ)
    du_adv  = mass_weighted_advection_upper_diagonal(i, j, k, grid, advection.scheme, w, Δt, ℓx, ℓy, ℓz, ρ)
    return du_diff + du_adv
end

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionLowerDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                         advection::MassWeightedImplicitDiffusion, w, ρ, args...)
    ρᵈ = mass_weighting_density(advection.diffusion_density, ρ)
    dl_diff = mass_weighted_ivd_lower_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, ρᵈ)
    dl_adv  = mass_weighted_advection_lower_diagonal(i, j, k, grid, advection.scheme, w, Δt, ℓx, ℓy, ℓz, ρ)
    return dl_diff + dl_adv
end

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                         advection::MassWeightedImplicitDiffusion, w, ρ,
                                         top_bc, bottom_bc, immersed_bc, args...)
    ρᵈ = mass_weighting_density(advection.diffusion_density, ρ)
    d_diff = mass_weighted_ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, ρᵈ)
    d_adv  = mass_weighted_advection_diagonal(i, j, k, grid, advection.scheme, w, Δt, ℓx, ℓy, ℓz, ρ)
    d_bc   = boundary_flux_diagonal(i, j, k, grid, ℓx, ℓy, ℓz, Δt, clk, fields, top_bc, bottom_bc, immersed_bc)
    return d_diff + d_adv + d_bc
end
