"""
$(TYPEDEF)

The second moments of E3SM `shoc.F90` (`diag_second_shoc_moments`), diagnosed down the gradient: the fluxes
are `-K ∂z` of the mean, the variances and the covariance of θˡⁱ and qᵗ are `ℂᵛ τ Kᶜ ∂z a ∂z b`, and the
vertical velocity variance is `ℂʷ₁ (2/3) e`. The surface values of `shoc.F90` (`diag_second_moments_srf`) are
not used: the surface face takes the moments of the face above.
"""
struct DowngradientSecondMoments{FT} <: AbstractSubgridDistributionComponent
    ℂᵛ₁ :: FT
    ℂᵛ₂ :: FT
    ℂᵛ₃ :: FT
    ℂʷ₁ :: FT
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `DowngradientSecondMoments{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword argument.
"""
DowngradientSecondMoments(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(DowngradientSecondMoments, FT, (ℂᵛ₁ = 1, ℂᵛ₂ = 1, ℂᵛ₃ = 1, ℂʷ₁ = 1), kw)

Adapt.@adapt_structure DowngradientSecondMoments

@inline vertical_velocity_variance(sm::DowngradientSecondMoments, e) =
    (sm.ℂʷ₁ * (convert(typeof(e), 2) / 3)) * e

@inline function interface_second_moments(sm::DowngradientSecondMoments, τ, Kᶜ, Kᵘ, Δz⁻¹, Δθˡⁱ, Δqᵗ, Δe, Δu, Δv)
    Δz⁻² = Δz⁻¹ * Δz⁻¹
    τKᶜ = τ * Kᶜ

    θˡⁱ′² = sm.ℂᵛ₁ * τKᶜ * Δz⁻² * Δθˡⁱ * Δθˡⁱ
    qᵗ′² = sm.ℂᵛ₂ * τKᶜ * Δz⁻² * Δqᵗ * Δqᵗ
    θˡⁱ′qᵗ′ = sm.ℂᵛ₃ * τKᶜ * Δz⁻² * Δθˡⁱ * Δqᵗ

    w′θˡⁱ′ = -1 * Kᶜ * Δz⁻¹ * Δθˡⁱ
    w′qᵗ′ = -1 * Kᶜ * Δz⁻¹ * Δqᵗ
    w′e′ = -1 * Kᶜ * Δz⁻¹ * Δe
    w′u′ = -1 * Kᵘ * Δz⁻¹ * Δu
    w′v′ = -1 * Kᵘ * Δz⁻¹ * Δv

    return θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′, w′θˡⁱ′, w′qᵗ′, w′e′, w′u′, w′v′
end
