"""
$(TYPEDEF)

Stability functions of the local diagnosis of the scalar second moments of the [`DoubleGaussian`](@ref)
of [`TKEBasedTurbulenceClosure`](@ref), in place of the ratio ``𝕊ᶜ / 𝕊ᴰ`` that the diagnosis
``ℂᵛ τ Kᶜ ∂_z a ∂_z b`` inherits from the time scale ``τ`` and the diffusivity ``Kᶜ``:
"""
struct VarianceStabilityFunctions{FT}
    ℂᵛ₁⁻ :: FT
    ℂᵛ₁⁰ :: FT
    ℂᵛ₁⁺ :: FT
    ℂᵛ₂⁻ :: FT
    ℂᵛ₂⁰ :: FT
    ℂᵛ₂⁺ :: FT
    ℂᵛ₃⁻ :: FT
    ℂᵛ₃ᵐ :: FT
    ℂᵛ₃⁺ :: FT
end

Adapt.@adapt_structure VarianceStabilityFunctions

convert_eltype(::Type{FT}, vs::VarianceStabilityFunctions) where FT =
    VarianceStabilityFunctions{FT}(map(name -> convert(FT, getfield(vs, name)), fieldnames(VarianceStabilityFunctions))...)

Base.summary(vs::VarianceStabilityFunctions{FT}) where FT = "VarianceStabilityFunctions{$FT}"

function Base.show(io::IO, vs::VarianceStabilityFunctions)
    print(io, summary(vs), '\n',
          "├── 𝕊ᵛ₁ (ℂᵛ₁⁻, ℂᵛ₁⁰, ℂᵛ₁⁺): ", join(prettysummary.((vs.ℂᵛ₁⁻, vs.ℂᵛ₁⁰, vs.ℂᵛ₁⁺)), ", "), '\n',
          "├── 𝕊ᵛ₂ (ℂᵛ₂⁻, ℂᵛ₂⁰, ℂᵛ₂⁺): ", join(prettysummary.((vs.ℂᵛ₂⁻, vs.ℂᵛ₂⁰, vs.ℂᵛ₂⁺)), ", "), '\n',
          "└── 𝕏ᵛ  (ℂᵛ₃⁻, ℂᵛ₃ᵐ, ℂᵛ₃⁺): ", join(prettysummary.((vs.ℂᵛ₃⁻, vs.ℂᵛ₃ᵐ, vs.ℂᵛ₃⁺)), ", "))
end

function validate_variance_stability_coefficients(coefficients)
    ℂᵛ₁⁻, ℂᵛ₁⁰, ℂᵛ₁⁺, ℂᵛ₂⁻, ℂᵛ₂⁰, ℂᵛ₂⁺, ℂᵛ₃⁻, ℂᵛ₃ᵐ, ℂᵛ₃⁺ = coefficients
    all(C -> C > 0, (ℂᵛ₁⁻, ℂᵛ₁⁰, ℂᵛ₁⁺, ℂᵛ₂⁻, ℂᵛ₂⁰, ℂᵛ₂⁺)) ||
        throw(ArgumentError("The variance coefficients ℂᵛ₁ and ℂᵛ₂ of VarianceStabilityFunctions must be positive, " *
                            "but they are $((ℂᵛ₁⁻, ℂᵛ₁⁰, ℂᵛ₁⁺, ℂᵛ₂⁻, ℂᵛ₂⁰, ℂᵛ₂⁺))"))
    all(C -> -1 ≤ C ≤ 1, (ℂᵛ₃⁻, ℂᵛ₃ᵐ, ℂᵛ₃⁺)) ||
        throw(ArgumentError("The correlation coefficients ℂᵛ₃ of VarianceStabilityFunctions must be within [-1, 1], " *
                            "but they are $((ℂᵛ₃⁻, ℂᵛ₃ᵐ, ℂᵛ₃⁺))"))
    return nothing
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct [`VarianceStabilityFunctions`](@ref). The defaults reproduce the down-gradient diagnosis of
the default [`DoubleGaussian`](@ref) with the default [`TurbulentRichardsonStabilityFunction`](@ref):
``ℂᵛ_i^{-,0,+} = (2 ℂᵛ_i / ℂᵠ) ℂᶜ^{-,0,+} / ℂᴰ^{-,0,+}`` with ``ℂᵛ_i = 1`` and ``ℂᵠ = 1.3``, and the
perfect correlation ``ℂᵛ₃ = 1``. The variance coefficients must be positive and the correlation
coefficients within ``[-1, 1]``.
"""
function VarianceStabilityFunctions(FT::DataType = Oceananigans.defaults.FloatType;
                                    ℂᵛ₁⁻ = 1.5480121696035578, ℂᵛ₁⁰ = 0.6836078066514023, ℂᵛ₁⁺ = 0.3096024339207115,
                                    ℂᵛ₂⁻ = 1.5480121696035578, ℂᵛ₂⁰ = 0.6836078066514023, ℂᵛ₂⁺ = 0.3096024339207115,
                                    ℂᵛ₃⁻ = 1, ℂᵛ₃ᵐ = 1, ℂᵛ₃⁺ = 1)
    coefficients = (ℂᵛ₁⁻, ℂᵛ₁⁰, ℂᵛ₁⁺, ℂᵛ₂⁻, ℂᵛ₂⁰, ℂᵛ₂⁺, ℂᵛ₃⁻, ℂᵛ₃ᵐ, ℂᵛ₃⁺)
    validate_variance_stability_coefficients(coefficients)
    return VarianceStabilityFunctions{FT}(map(C -> convert(FT, C), coefficients)...)
end

variance_time_scale_dissipation(::DissipationTimeScale, sf) = (sf.ℂᴰ⁻, sf.ℂᴰ⁰, sf.ℂᴰ⁺)
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The [`VarianceStabilityFunctions`](@ref) that reproduce the down-gradient diagnosis
``ℂᵛ τ Kᶜ ∂_z a ∂_z b`` of a [`DoubleGaussian`](@ref) with the time scale `time_scale`
([`DissipationTimeScale`](@ref) or [`EddyTurnoverTimeScale`](@ref)) and the second moments
`second_moments` ([`DowngradientSecondMoments`](@ref)) in a closure with the stability functions `sf`:
"""
function VarianceStabilityFunctions(sf::TurbulentRichardsonStabilityFunction, time_scale,
                                    second_moments::DowngradientSecondMoments)
    FT = typeof(sf.ℂᶜ⁰)
    ℂᴰ⁻, ℂᴰ⁰, ℂᴰ⁺ = variance_time_scale_dissipation(time_scale, sf)
    ℂᵠ = time_scale.ℂᵠ
    sm = second_moments
    a₁ = 2 * sm.ℂᵛ₁ / ℂᵠ
    a₂ = 2 * sm.ℂᵛ₂ / ℂᵠ
    r = sm.ℂᵛ₃ / sqrt(sm.ℂᵛ₁ * sm.ℂᵛ₂) # the correlation of θˡⁱ and qᵗ
    return VarianceStabilityFunctions(FT;
                                      ℂᵛ₁⁻ = a₁ * sf.ℂᶜ⁻ / ℂᴰ⁻, ℂᵛ₁⁰ = a₁ * sf.ℂᶜ⁰ / ℂᴰ⁰, ℂᵛ₁⁺ = a₁ * sf.ℂᶜ⁺ / ℂᴰ⁺,
                                      ℂᵛ₂⁻ = a₂ * sf.ℂᶜ⁻ / ℂᴰ⁻, ℂᵛ₂⁰ = a₂ * sf.ℂᶜ⁰ / ℂᴰ⁰, ℂᵛ₂⁺ = a₂ * sf.ℂᶜ⁺ / ℂᴰ⁺,
                                      ℂᵛ₃⁻ = r, ℂᵛ₃ᵐ = r, ℂᵛ₃⁺ = r)
end

#####
##### Evaluation
#####

@inline function variance_stability_functions(vs::VarianceStabilityFunctions, v)
    𝕊ᵛ₁ = evaluate_log_quadratic(log_quadratic_coefficients(vs.ℂᵛ₁⁻, vs.ℂᵛ₁⁰, vs.ℂᵛ₁⁺), v)
    𝕊ᵛ₂ = evaluate_log_quadratic(log_quadratic_coefficients(vs.ℂᵛ₂⁻, vs.ℂᵛ₂⁰, vs.ℂᵛ₂⁺), v)
    t = (1 + v) / 2
    s = 1 - t
    𝕏ᵛ = s * s * vs.ℂᵛ₃⁻ + 2 * t * s * vs.ℂᵛ₃ᵐ + t * t * vs.ℂᵛ₃⁺
    return 𝕊ᵛ₁, 𝕊ᵛ₂, 𝕏ᵛ
end

@inline local_scalar_second_moments(::Nothing, i, j, k, grid, closure, velocities, closure_fields, e, Δθ, Δq, Δz,
                                    θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′) = (θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′)

@inline function local_scalar_second_moments(vs::VarianceStabilityFunctions, i, j, k, grid, closure, velocities,
                                             closure_fields, e, Δθ, Δq, Δz, θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′)
    ℓᵐ = closure_fields.ℓᵐ
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, closure_fields.N², e, ℓᵐ)
    v = variance_richardson_coordinate(closure.stability_functions, αᴺ, αˢ)
    𝕊ᵛ₁, 𝕊ᵛ₂, 𝕏ᵛ = variance_stability_functions(vs, v)
    ℓ = @inbounds ℓᵐ[i, j, k]
    ∂zθ = Δθ / Δz
    ∂zq = Δq / Δz
    ℓ² = ℓ * ℓ
    return (𝕊ᵛ₁ * ℓ² * ∂zθ * ∂zθ,
            𝕊ᵛ₂ * ℓ² * ∂zq * ∂zq,
            𝕏ᵛ * sqrt(𝕊ᵛ₁ * 𝕊ᵛ₂) * ℓ² * ∂zθ * ∂zq)
end

@inline variance_richardson_coordinate(sf::TurbulentRichardsonStabilityFunction, αᴺ, αˢ) =
    bounded_richardson_coordinate(sf, αᴺ, αˢ)

validate_variance_stability_functions(sf, ::Nothing) = nothing
validate_variance_stability_functions(::TurbulentRichardsonStabilityFunction, ::VarianceStabilityFunctions) = nothing
validate_variance_stability_functions(sf, ::VarianceStabilityFunctions) =
    throw(ArgumentError("VarianceStabilityFunctions are functions of the turbulent Richardson number of a " *
                        "TurbulentRichardsonStabilityFunction, but the closure's stability functions are $(summary(sf))."))
