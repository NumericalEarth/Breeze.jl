"""
$(TYPEDEF)

The third moment of vertical velocity diagnosed algebraically from the second moments and their vertical
gradients, as E3SM `shoc.F90` computes it (`compute_diag_third_shoc_moment`, after Canuto et al. 2001).
"""
struct AlgebraicThirdMoment{FT} <: AbstractSubgridDistributionComponent
    ℂʷ₂ :: FT
    ℂʷ₃ :: FT
    ℂʷ₄ :: FT
    ℂʷ₅ :: FT
    ℂʷ₆ :: FT
    ℂʷ₇ :: FT
    ℂʷ₈ :: FT
    ℂʷ₉ :: FT
    ℂʷ₁₀ :: FT
    ℂʷ₁₁ :: FT
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `AlgebraicThirdMoment{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword argument.
"""
function AlgebraicThirdMoment(FT::DataType = Oceananigans.defaults.FloatType; kw...)
    defaults = (ℂʷ₂ = 7, ℂʷ₃ = 0.52, ℂʷ₄ = 0.87, ℂʷ₅ = 0.5, ℂʷ₆ = 0.6, ℂʷ₇ = 2.4, ℂʷ₈ = 0.6, ℂʷ₉ = 1.2, ℂʷ₁₀ = 1.5, ℂʷ₁₁ = 1.25)
    names = keys(defaults)
    all(name -> name ∈ names, keys(kw)) ||
        throw(ArgumentError("$(setdiff(keys(kw), names)) are not fields of AlgebraicThirdMoment; its fields are $(names)."))
    values = map(value -> convert(FT, value), Tuple(merge(defaults, NamedTuple(kw))))
    return AlgebraicThirdMoment(values...)
end

Adapt.@adapt_structure AlgebraicThirdMoment

@inline function third_moment(tm::AlgebraicThirdMoment, stratification_number_bound, g, τ, N², w′², θˡⁱ,
                              compact_inverse_spacing, wide_inverse_spacing, Δθˡⁱ′², Δw′θˡⁱ′, Δw′², Δe, w′θˡⁱ′)
    FT = typeof(w′θˡⁱ′)
    c = tm.ℂʷ₂

    a₀ = (tm.ℂʷ₃ * (1 / (c * c))) / (c - 2)
    a₁ = tm.ℂʷ₄ / (c * c)
    a₂ = tm.ℂʷ₅ / c
    a₃ = tm.ℂʷ₆ / (c * (c - 2))
    first_pole = 1 / (a₁ + a₃)

    τ² = τ * τ
    αᴺ = bound_stratification_number(stratification_number_bound, τ² * N², first_pole)
    β = g / θˡⁱ

    f₀ = wide_inverse_spacing * (β * β * β) * ((τ * τ) * (τ * τ)) * w′θˡⁱ′ * Δθˡⁱ′²
    f₁ = wide_inverse_spacing * (β * β) * (τ * τ * τ) * (w′θˡⁱ′ * Δw′θˡⁱ′ + convert(FT, 0.5) * w′² * Δθˡⁱ′²)
    f₂ = compact_inverse_spacing * β * τ² * w′θˡⁱ′ * Δw′² + 2 * wide_inverse_spacing * β * τ² * w′² * Δw′θˡⁱ′
    f₃ = wide_inverse_spacing * β * τ² * w′² * Δw′θˡⁱ′ + compact_inverse_spacing * β * τ² * (w′θˡⁱ′ * Δe)
    f₄ = compact_inverse_spacing * τ * w′² * (Δw′² + Δe)
    f₅ = compact_inverse_spacing * τ * w′² * Δw′²

    a₄ = tm.ℂʷ₇ / (3 * c + 5)
    a₅ = tm.ℂʷ₈ / (c * (3 + 5 * c))
    ω₀ = a₄ / (1 - a₅ * αᴺ)
    ω₁ = ω₀ / (2 * c)
    ω₂ = ω₁ * f₃ + tm.ℂʷ₁₁ * ω₀ * f₄

    X₀ = (a₂ * αᴺ * (1 - a₃ * αᴺ)) / (1 - (a₁ + a₃) * αᴺ)
    Y₀ = (2 * a₂ * αᴺ * X₀) / (1 - a₃ * αᴺ)
    X₁ = (a₀ * f₀ + a₁ * f₁ + a₂ * (1 - a₃ * αᴺ) * f₂) / (1 - (a₁ + a₃) * αᴺ)
    Y₁ = (2 * a₂ * (αᴺ * X₁ + (a₀ / a₁) * f₀ + f₁)) / (1 - a₃ * αᴺ)

    A₀ = ω₀ * X₀ + ω₁ * Y₀
    A₁ = ω₀ * X₁ + ω₁ * Y₁ + ω₂

    return (A₁ - tm.ℂʷ₉ * X₁ - tm.ℂʷ₁₀ * f₅) / (c - tm.ℂʷ₉ * X₀ + A₀)
end
