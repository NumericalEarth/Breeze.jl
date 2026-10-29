#####
##### Density-weighted implicit-diffusion rows carried from level to level
#####
##### The z sweep of `BatchedTridiagonalSolver` evaluates, at level k, `c(k-1)`, `b(k)` and `a(k-1)` through
##### `get_coefficient`. For a `DensityWeightedImplicitOperator` over a single scalar-diffusivity closure, those three
##### rows (see density_weighted_implicit_diffusion.jl) read: at a z-Center location the diffusivity of faces k and
##### k+1, the inverse spacings of center k and of faces k and k+1, the densities of faces k and k+1 and of cells
##### k-1 and k; and `du(k-1)` again, which row k-1's diagonal already formed. Every one of those except the values of
##### face k+1 and cell k was evaluated at level k-1.
#####
##### The sweep below is Oceananigans' `solve_batched_tridiagonal_system_z!` (same operations on values, same order,
##### same types) with the diffusion half of the rows formed from carried values: per level one diffusivity, two
##### inverse spacings, one face and one cell density, and the products `du ρ` and `dl ρ` that the off-diagonals and
##### the diagonal share. Each value is the expression of Oceananigans' `ivd_upper_diagonal` / `ivd_lower_diagonal`
##### and of `density_weighted_ivd_*`, on the same operands, so every coefficient is the same number: it is
##### computed once instead of up to three times. The implicit-advection parts, the linear coefficient and the
##### boundary-flux diagonal are evaluated per row exactly as before.
#####
##### Only closures whose `_ivd_*_diagonal` are Oceananigans' generic single-closure rows carry
##### (`AbstractScalarDiffusivity`: `ScalarDiffusivity`, `TKEBasedTurbulenceClosure`); closure tuples, `nothing` and
##### everything else keep the `get_coefficient` sweep.
#####

using Oceananigans.Grids: peripheral_node
using Oceananigans.Advection: densityᶜᶜᶜ, densityᶜᶜᶠ
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, getclosure, ivd_diffusivity, Δz⁻¹
using Oceananigans.Solvers: masked_multiply, back_substitute_z!, right_hand_side_ahead, float_eltype


# The trailing arguments `implicit_step!` hands `solve!`:
# (closure, closure_fields, id, ℓx, ℓy, ℓz, Δt, clock, fields, advection, w, density, top_bc, bottom_bc, immersed_bc, ϕ)
const CarriedRowArguments = Tuple{<:AbstractScalarDiffusivity, Any, Any, Any, Any, Union{Center, Face}, Any, Any, Any,
                                  <:DensityWeightedImplicitOperator, Any, Any, Any, Any, Any, Any}

#####
##### The diffusion half of row k from the carried values. Each returns
#####
#####   Pu = du(k) ρ⁺        (the product `du * ρ` of the upper diagonal at k and of the diagonal at k)
#####   Pl = dl(k-1) ρ⁻      (the product `dl * ρ` of the lower diagonal at k-1 and of the diagonal at k)
#####   D  = the density the diagonal at k divides by (ρᶜₖ at z-Centers, ρᶠₖ at z-Faces)
#####   S  = the values to carry to level k+1
#####

# z-Center: S holds face k's diffusivity, inverse spacing, density and activity (row k's lower face).
@inline function carried_face_values(i, j, k, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)
    κ    = ivd_diffusivity(i, j, k, grid, ℓx, ℓy, Face(), closure_ij, K, id, clk, fields)
    Δz⁻¹ᶠ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, Face())
    ρᶠ   = densityᶜᶜᶠ(i, j, k, grid, ρ)
    p    = !peripheral_node(i, j, k, grid, ℓx, ℓy, Face())
    return (κ, Δz⁻¹ᶠ, ρᶠ, p)
end

@inline carried_initial_state(i, j, grid, ::Center, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ) =
    carried_face_values(i, j, 1, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)

@inline function carried_diffusion_row(i, j, k, grid, ::Center, S, closure_ij, K, id, ℓx, ℓy, Δt, clk, fields, ρ)
    κᵏ, Δz⁻¹ᶠₖ, ρᶠₖ, pₖ = S
    κᵏ⁺¹, Δz⁻¹ᶠₖ₊₁, ρᶠₖ₊₁, pₖ₊₁ = S⁺ = carried_face_values(i, j, k+1, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)
    Δz⁻¹ᶜₖ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, Center())
    ρᶜₖ = densityᶜᶜᶜ(i, j, k, grid, ρ)

    du = - Δt * κᵏ⁺¹ * (Δz⁻¹ᶜₖ * Δz⁻¹ᶠₖ₊₁)   # ivd_upper_diagonal(k)
    du = du * pₖ₊₁
    dl = - Δt * κᵏ * (Δz⁻¹ᶜₖ * Δz⁻¹ᶠₖ)       # ivd_lower_diagonal(k-1)
    dl = dl * pₖ

    return du * ρᶠₖ₊₁, dl * ρᶠₖ, ρᶜₖ, S⁺
end

# z-Face: S holds center k-1's diffusivity, inverse spacing and density (row k's lower stress).
@inline function carried_center_values(i, j, k, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)
    ν    = ivd_diffusivity(i, j, k, grid, ℓx, ℓy, Center(), closure_ij, K, id, clk, fields)
    Δz⁻¹ᶜ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, Center())
    ρᶜ   = densityᶜᶜᶜ(i, j, k, grid, ρ)
    return (ν, Δz⁻¹ᶜ, ρᶜ)
end

@inline carried_initial_state(i, j, grid, ::Face, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ) =
    carried_center_values(i, j, 0, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)

@inline function carried_diffusion_row(i, j, k, grid, ::Face, S, closure_ij, K, id, ℓx, ℓy, Δt, clk, fields, ρ)
    νᵏ⁻¹, Δz⁻¹ᶜₖ₋₁, ρᶜₖ₋₁ = S
    νᵏ, Δz⁻¹ᶜₖ, ρᶜₖ = S⁺ = carried_center_values(i, j, k, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)
    Δz⁻¹ᶠₖ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, Face())
    ρᶠₖ = densityᶜᶜᶠ(i, j, k, grid, ρ)
    pₖ = !peripheral_node(i, j, k, grid, ℓx, ℓy, Face())

    du = - Δt * νᵏ * (Δz⁻¹ᶜₖ * Δz⁻¹ᶠₖ)          # ivd_upper_diagonal(k)
    du = du * pₖ
    dl = - Δt * νᵏ⁻¹ * (Δz⁻¹ᶜₖ₋₁ * Δz⁻¹ᶠₖ)      # ivd_lower_diagonal(k-1)
    dl = dl * pₖ

    return du * ρᶜₖ, dl * ρᶜₖ₋₁, ρᶠₖ, S⁺
end

# The diagonal at k: `density_weighted_ivd_diagonal + density_weighted_advection_diagonal + boundary_flux_diagonal`.
@inline function carried_diagonal(i, j, k, grid, Pu, Pl, D, args)
    clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, operator, w, ρ, top_bc, bottom_bc, immersed_bc, ϕ = args
    lin    = _implicit_linear_coefficient(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    d_diff = one(grid) - Δt * lin - (Pu + Pl) / D
    d_adv  = density_weighted_advection_diagonal(i, j, k, grid, operator.scheme, w, Δt, ℓx, ℓy, ℓz, ρ)
    d_bc   = boundary_flux_diagonal(i, j, k, grid, ℓx, ℓy, ℓz, Δt, clk, fields, top_bc, bottom_bc, immersed_bc, ϕ)
    return d_diff + d_adv + d_bc
end

@inline function Solvers.solve_batched_tridiagonal_system_z!(i, j, Nz, ϕ,
                                                     a::VerticallyImplicitDiffusionLowerDiagonal,
                                                     b::VerticallyImplicitDiffusionDiagonal,
                                                     c::VerticallyImplicitDiffusionUpperDiagonal,
                                                     f, t, grid, p, args::CarriedRowArguments, tridiagonal_direction::ZDirection)
    FT = eltype(ϕ)
    TT = eltype(t)
    ahead = right_hand_side_ahead(f)
    ε = 10 * eps(float_eltype(ϕ))

    clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, operator, w, ρ = args
    scheme = operator.scheme
    ρ_diff = weighting_density(operator.density, ρ)
    closure_ij = getclosure(i, j, clo)

    @inbounds begin
        S = carried_initial_state(i, j, grid, ℓz, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ_diff)
        Puᵏ, Plᵏ, Dᵏ, S = carried_diffusion_row(i, j, 1, grid, ℓz, S, closure_ij, K, id, ℓx, ℓy, Δt, clk, fields, ρ_diff)

        β  = carried_diagonal(i, j, 1, grid, Puᵏ, Plᵏ, Dᵏ, args)
        f₁ = Solvers.get_coefficient(i, j, 1, grid, f, p, tridiagonal_direction, args...)
        ϕᵏ⁻¹ = convert(FT, f₁ / β)
        ϕ[i, j, 1] = ϕᵏ⁻¹

        fᵏ = ahead ? Solvers.get_coefficient(i, j, min(2, Nz), grid, f, p, tridiagonal_direction, args...) : f₁

        for k = 2:Nz
            fᵏ⁺¹ = ahead ? Solvers.get_coefficient(i, j, ifelse(k < Nz, k+1, k), grid, f, p, tridiagonal_direction, args...) : fᵏ

            Puᵏ⁻¹ = Puᵏ
            Dᵏ⁻¹  = Dᵏ
            Puᵏ, Plᵏ, Dᵏ, S = carried_diffusion_row(i, j, k, grid, ℓz, S, closure_ij, K, id, ℓx, ℓy, Δt, clk, fields, ρ_diff)

            # density_weighted_ivd_upper_diagonal(k-1) = du(k-1) ρ / D(k), lower_diagonal(k-1) = dl(k-1) ρ / D(k-1)
            cᵏ⁻¹ = Puᵏ⁻¹ / Dᵏ + density_weighted_advection_upper_diagonal(i, j, k-1, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ)
            bᵏ   = carried_diagonal(i, j, k, grid, Puᵏ, Plᵏ, Dᵏ, args)
            aᵏ⁻¹ = Plᵏ / Dᵏ⁻¹ + density_weighted_advection_lower_diagonal(i, j, k-1, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ)

            tᵏ = convert(TT, cᵏ⁻¹ / β)
            t[i, j, k] = tᵏ
            β = bᵏ - masked_multiply(aᵏ⁻¹, tᵏ)
            fᵏ = ahead ? fᵏ : Solvers.get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            definitely_diagonally_dominant = abs(β) > ε
            ϕ★ = (fᵏ - masked_multiply(aᵏ⁻¹, ϕᵏ⁻¹)) / β
            if definitely_diagonally_dominant
                ϕᵏ = convert(FT, ϕ★)
                ϕ[i, j, k] = ϕᵏ
            else
                ϕᵏ = ϕ[i, j, k]
            end

            ϕᵏ⁻¹ = ϕᵏ
            fᵏ = fᵏ⁺¹
        end

        back_substitute_z!(ϕ, t, i, j, Nz, ϕᵏ⁻¹)
    end

    return nothing
end
