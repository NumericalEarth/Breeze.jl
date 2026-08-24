#####
##### Mass-flux weighting of the vertically-implicit diffusion coefficients (z-Center fields)
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
##### `ρw` (z-Face) is deliberately *not* weighted here. Upstream's z-Face coefficients evaluate
##### `ν` at a single center for both off-diagonals of a row (`ivd_upper_diagonal` at center k and
##### `ivd_lower_diagonal(k-1)` also at center k, vertically_implicit_diffusion_solver.jl:95-112),
##### so there is no consistent center at which to evaluate `ρ` for the lower diagonal. That
##### stencil is exact on uniform grids, which is where it is currently exercised; weighting it
##### would require fixing the index convention upstream first.
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

using Oceananigans.Grids: ZDirection
using Oceananigans.TurbulenceClosures:
    VerticallyImplicitDiffusionLowerDiagonal,
    VerticallyImplicitDiffusionDiagonal,
    VerticallyImplicitDiffusionUpperDiagonal,
    _implicit_linear_coefficient,
    _ivd_lower_diagonal,
    _ivd_upper_diagonal

# Breeze-owned wrapper routing a z-Center prognostic's implicit solve to the mass-flux-weighted
# coefficients below. Like `VerticalMomentumImplicitAdvection`, wrapping the scheme puts a
# Breeze-owned type in the `get_coefficient` signature, so these methods are neither type piracy
# nor ambiguous with Oceananigans' own `AIVA` and fallback methods.
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

# Dispatch rather than a branch: `nothing` defers to the density the solve was called with.
@inline mass_weighting_density(::Nothing, ρ) = ρ
@inline mass_weighting_density(ρᵈ, ρ) = ρᵈ

# The implicit-advection contribution, which is present only for adaptive-implicit schemes.
@inline mass_weighted_advection_upper_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ρ) = zero(grid)
@inline mass_weighted_advection_lower_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ρ) = zero(grid)
@inline mass_weighted_advection_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ρ) = zero(grid)

@inline mass_weighted_advection_upper_diagonal(i, j, k, grid, scheme::AIVA, w, Δt, ℓx, ℓy, ρ) =
    implicit_advection_upper_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ρ)
@inline mass_weighted_advection_lower_diagonal(i, j, k, grid, scheme::AIVA, w, Δt, ℓx, ℓy, ρ) =
    implicit_advection_lower_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ρ)
@inline mass_weighted_advection_diagonal(i, j, k, grid, scheme::AIVA, w, Δt, ℓx, ℓy, ρ) =
    implicit_advection_diagonal(i, j, k, grid, scheme, w, Δt, ℓx, ℓy, ρ)

# As with the advection coefficients, `ρ` is interpolated in z only, so these are exact for
# a horizontally-uniform density (the anelastic reference state) at all three z-Center locations.
# The batched solver evaluates the off-diagonals only for k = 1 … Nz-1, so the divisions below
# never reach a halo value of `ρ`; the diagonal's `ρᶠ⁺` at k = Nz+1 does, but it only multiplies
# a `du` that the peripheral-node mask has already zeroed.
@inline function mass_weighted_ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, ρ)
    du = _ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    ρᶠ = densityᶜᶜᶠ(i, j, k+1, grid, ρ)   # where κₖ₊₁ weights the flux
    ρᶜ = densityᶜᶜᶜ(i, j, k+1, grid, ρ)   # where qₖ₊₁ is reconstructed as cₖ₊₁ = qₖ₊₁/ρᶜₖ₊₁
    return du * ρᶠ / ρᶜ
end

# `k′ = k - 1` (LinearAlgebra.Tridiagonal convention): the coefficient of `q(k′)` in row `k′ + 1`.
@inline function mass_weighted_ivd_lower_diagonal(i, j, k′, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, ρ)
    dl = _ivd_lower_diagonal(i, j, k′, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    ρᶠ = densityᶜᶜᶠ(i, j, k′+1, grid, ρ)
    ρᶜ = densityᶜᶜᶜ(i, j, k′,   grid, ρ)
    return dl * ρᶠ / ρᶜ
end

@inline function mass_weighted_ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, ρ)
    du  = _ivd_upper_diagonal(i, j, k,   grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    dl  = _ivd_lower_diagonal(i, j, k-1, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    lin = _implicit_linear_coefficient(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)

    ρᶜ  = densityᶜᶜᶜ(i, j, k,   grid, ρ)
    ρᶠ⁺ = densityᶜᶜᶠ(i, j, k+1, grid, ρ)
    ρᶠ⁻ = densityᶜᶜᶠ(i, j, k,   grid, ρ)

    # Both off-diagonal fluxes act on qₖ here, so both divide by ρᶜₖ — not by the neighbours'
    # ρᶜ, which is why this is not `1 - du - dl`. A linear coefficient damps q and c at the
    # same rate, so it is unweighted.
    return one(grid) - Δt * lin - (du * ρᶠ⁺ + dl * ρᶠ⁻) / ρᶜ
end

#####
##### get_coefficient seam for z-Center prognostics: mass-weighted diffusion + implicit advection
#####

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionUpperDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                         advection::MassWeightedImplicitDiffusion, w, ρ)
    ρᵈ = mass_weighting_density(advection.diffusion_density, ρ)
    du_diff = mass_weighted_ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, ρᵈ)
    du_adv  = mass_weighted_advection_upper_diagonal(i, j, k, grid, advection.scheme, w, Δt, ℓx, ℓy, ρ)
    return du_diff + du_adv
end

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionLowerDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                         advection::MassWeightedImplicitDiffusion, w, ρ)
    ρᵈ = mass_weighting_density(advection.diffusion_density, ρ)
    dl_diff = mass_weighted_ivd_lower_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, ρᵈ)
    dl_adv  = mass_weighted_advection_lower_diagonal(i, j, k, grid, advection.scheme, w, Δt, ℓx, ℓy, ρ)
    return dl_diff + dl_adv
end

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                         advection::MassWeightedImplicitDiffusion, w, ρ)
    ρᵈ = mass_weighting_density(advection.diffusion_density, ρ)
    d_diff = mass_weighted_ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, ρᵈ)
    d_adv  = mass_weighted_advection_diagonal(i, j, k, grid, advection.scheme, w, Δt, ℓx, ℓy, ρ)
    return d_diff + d_adv
end
