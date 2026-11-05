#####
##### Mixing-length subgrid statistics of the TKE closure
#####
#
# The TKE closure does not prognose second moments of the scalars, so it supplies them with the
# down-gradient mixing-length form (Bechtold et al. 1995; Cuijpers and Bechtold 1995)
#
#   ⟨φ'ψ'⟩ = c ℓ² ∂zφ ∂zψ,      φ, ψ ∈ {qᵗ, θˡⁱ},
#
# with ℓ the closure's mixing length `ℓᵐ` and c the correlation's `variance_coefficient`, plus the
# within-cell variance of the resolved gradient, c_Δ Δz² ∂zφ ∂zψ (c_Δ = 1/12 for a linear profile).
# ℓ and the gradients live at the cell's two faces; the products are averaged to the centre.
# The bottom and top boundary faces contribute nothing (no gradient across the boundary).

@inline function face_mixing_length_moments(i, j, k, grid, correlation, ℓ, θ, qᵗ)
    L² = correlation.variance_coefficient * @inbounds(ℓ[i, j, k]^2) +
         correlation.grid_variance_coefficient * Δzᶜᶜᶠ(i, j, k, grid)^2
    ∂qᵗ = ∂zᶜᶜᶠ(i, j, k, grid, qᵗ)
    ∂θ = ∂zᶜᶜᶠ(i, j, k, grid, θ)
    return L² * ∂qᵗ^2, L² * ∂θ^2, L² * ∂qᵗ * ∂θ
end

@inline function AtmosphereModels.subgrid_statisticsᶜᶜᶜ(i, j, k, grid, correlation::GaussianSubgridCorrelation,
                                                         ::FlavorOfTKEClosure, closure_fields, state)
    Nz = size(grid, 3)
    ℓ = closure_fields.ℓᵐ
    tᵇ, θᵇ, cᵇ = face_mixing_length_moments(i, j, k, grid, correlation, ℓ, state.θ, state.qᵗ)
    tᵗ, θᵗ, cᵗ = face_mixing_length_moments(i, j, k + 1, grid, correlation, ℓ, state.θ, state.qᵗ)
    bottom = k == 1
    top = k == Nz
    σ²ᵗ = (ifelse(bottom, zero(tᵇ), tᵇ) + ifelse(top, zero(tᵗ), tᵗ)) / 2
    σ²ᶿ = (ifelse(bottom, zero(θᵇ), θᵇ) + ifelse(top, zero(θᵗ), θᵗ)) / 2
    cᵗᶿ = (ifelse(bottom, zero(cᵇ), cᵇ) + ifelse(top, zero(cᵗ), cᵗ)) / 2
    return GaussianStatistics(σ²ᵗ, σ²ᶿ, cᵗᶿ)
end
