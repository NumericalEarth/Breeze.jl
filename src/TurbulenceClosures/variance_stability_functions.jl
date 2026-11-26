#####
##### Stability functions of the scalar variances and covariance of the subgrid distribution
#####
##### The local diagnosis of the second moments of θˡⁱ and qᵗ is ℂᵛ τ Kᶜ ∂z a ∂z b. With the time scale of
##### the scalar-variance budget, τ = 2ℓ / (ℂᵠ 𝕊ᴰ √e), and Kᶜ = 𝕊ᶜ ℓ √e, it is
#####
#####     θˡⁱ′² = (2ℂᵛ₁ / ℂᵠ) (𝕊ᶜ / 𝕊ᴰ) ℓ² (∂z θˡⁱ)²,
#####
##### a stability function borrowed from those of the flux and of the dissipation. `VarianceStabilityFunctions`
##### gives the variances and the covariance stability functions of their own,
#####
#####     θˡⁱ′² = 𝕊ᵛ₁ ℓ² (∂z θˡⁱ)²,   qᵗ′² = 𝕊ᵛ₂ ℓ² (∂z qᵗ)²,   θˡⁱ′qᵗ′ = 𝕊ᵛ₃ ℓ² ∂z θˡⁱ ∂z qᵗ,
#####     𝕊ᵛ₃ = 𝕏ᵛ √(𝕊ᵛ₁ 𝕊ᵛ₂),
#####
##### of the turbulent Richardson number of the closure's `TurbulentRichardsonStabilityFunction`.
#####

"""
$(TYPEDEF)

Stability functions of the local diagnosis of the scalar second moments of the [`DoubleGaussian`](@ref)
of [`TKEBasedTurbulenceClosure`](@ref), in place of the ratio ``𝕊ᶜ / 𝕊ᴰ`` that the diagnosis
``ℂᵛ τ Kᶜ ∂_z a ∂_z b`` inherits from the time scale ``τ`` and the diffusivity ``Kᶜ``:

```math
θˡⁱ′² = 𝕊ᵛ₁ ℓ² (∂_z θˡⁱ)², \\qquad qᵗ′² = 𝕊ᵛ₂ ℓ² (∂_z qᵗ)², \\qquad
θˡⁱ′qᵗ′ = 𝕊ᵛ₃ ℓ² ∂_z θˡⁱ ∂_z qᵗ, \\qquad 𝕊ᵛ₃ = 𝕏ᵛ \\sqrt{𝕊ᵛ₁ 𝕊ᵛ₂},
```

with ``ℓ`` the closure's mixing length at the face. The functions are of the coordinate
``v = Riᵗ / \\sqrt{1 + Riᵗ²} ∈ (-1, 1)`` of the turbulent Richardson number of the closure's
[`TurbulentRichardsonStabilityFunction`](@ref), formed at the face as for ``Kᶜ``. As each function of
that type, ``𝕊ᵛ₁`` and ``𝕊ᵛ₂`` are the exponentials of the quadratics in ``v`` that take the values
``ℂᵛ⁻, ℂᵛ⁰, ℂᵛ⁺`` (strongly unstable, neutral, strongly stable) at ``v = -1, 0, 1``:

```math
\\ln 𝕊ᵛ = \\ln ℂᵛ⁰ + \\tfrac{1}{2} \\ln(ℂᵛ⁺ / ℂᵛ⁻) \\, v + \\left[\\tfrac{1}{2} \\ln(ℂᵛ⁺ ℂᵛ⁻) - \\ln ℂᵛ⁰\\right] v².
```

The correlation of θˡⁱ and qᵗ is ``r = 𝕏ᵛ \\operatorname{sign}(∂_z θˡⁱ ∂_z qᵗ)``, with ``𝕏ᵛ`` the
quadratic Bernstein polynomial of ``t = (1 + v) / 2 ∈ (0, 1)`` with the control values
``ℂᵛ₃⁻, ℂᵛ₃ᵐ, ℂᵛ₃⁺``:

```math
𝕏ᵛ = (1 - t)² ℂᵛ₃⁻ + 2 t (1 - t) ℂᵛ₃ᵐ + t² ℂᵛ₃⁺,
```

a convex combination of the three, so ``|r| ≤ 1`` wherever ``|ℂᵛ₃^{-,m,+}| ≤ 1``. ``𝕏ᵛ = ℂᵛ₃⁻`` and
``ℂᵛ₃⁺`` in the strongly unstable and stable limits, and ``(ℂᵛ₃⁻ + 2 ℂᵛ₃ᵐ + ℂᵛ₃⁺) / 4`` at neutral.
``ℂᵛ₃⁻ = ℂᵛ₃ᵐ = ℂᵛ₃⁺ = 1`` is the perfect correlation of the down-gradient diagnosis.

The fluxes ``-Kᶜ ∂_z a`` are unchanged; the moment time scale ``τ`` is still that of the third
moment. A [`ScalarVarianceTransport`](@ref) takes these moments as its local diagnosis, and the
flux-correlation bound ``ℂʳ`` of the [`DoubleGaussian`](@ref) applies to them after it.

The nine coefficients are scalars of [`closure_parameters`](@ref) and [`with_parameters`](@ref).
[`VarianceStabilityFunctions`](@ref)`(sf, time_scale, second_moments)` gives those that reproduce
the down-gradient diagnosis.

Fields
======

$(TYPEDFIELDS)
"""
struct VarianceStabilityFunctions{FT}
    "variance of θˡⁱ, strongly unstable"
    ℂᵛ₁⁻ :: FT
    "variance of θˡⁱ, neutral"
    ℂᵛ₁⁰ :: FT
    "variance of θˡⁱ, strongly stable"
    ℂᵛ₁⁺ :: FT
    "variance of qᵗ, strongly unstable"
    ℂᵛ₂⁻ :: FT
    "variance of qᵗ, neutral"
    ℂᵛ₂⁰ :: FT
    "variance of qᵗ, strongly stable"
    ℂᵛ₂⁺ :: FT
    "correlation of θˡⁱ and qᵗ, strongly unstable (the first control value of ``𝕏ᵛ``)"
    ℂᵛ₃⁻ :: FT
    "correlation of θˡⁱ and qᵗ, middle control value of ``𝕏ᵛ``"
    ℂᵛ₃ᵐ :: FT
    "correlation of θˡⁱ and qᵗ, strongly stable (the last control value of ``𝕏ᵛ``)"
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

```jldoctest
using Breeze

VarianceStabilityFunctions(; ℂᵛ₁⁺ = 1.2, ℂᵛ₂⁺ = 1.2)

# output
VarianceStabilityFunctions{Float64}
├── 𝕊ᵛ₁ (ℂᵛ₁⁻, ℂᵛ₁⁰, ℂᵛ₁⁺): 1.54801, 0.683608, 1.2
├── 𝕊ᵛ₂ (ℂᵛ₂⁻, ℂᵛ₂⁰, ℂᵛ₂⁺): 1.54801, 0.683608, 1.2
└── 𝕏ᵛ  (ℂᵛ₃⁻, ℂᵛ₃ᵐ, ℂᵛ₃⁺): 1.0, 1.0, 1.0
```
"""
function VarianceStabilityFunctions(FT::DataType = Oceananigans.defaults.FloatType;
                                    ℂᵛ₁⁻ = 1.5480121696035578, ℂᵛ₁⁰ = 0.6836078066514023, ℂᵛ₁⁺ = 0.3096024339207115,
                                    ℂᵛ₂⁻ = 1.5480121696035578, ℂᵛ₂⁰ = 0.6836078066514023, ℂᵛ₂⁺ = 0.3096024339207115,
                                    ℂᵛ₃⁻ = 1, ℂᵛ₃ᵐ = 1, ℂᵛ₃⁺ = 1)
    coefficients = (ℂᵛ₁⁻, ℂᵛ₁⁰, ℂᵛ₁⁺, ℂᵛ₂⁻, ℂᵛ₂⁰, ℂᵛ₂⁺, ℂᵛ₃⁻, ℂᵛ₃ᵐ, ℂᵛ₃⁺)
    validate_variance_stability_coefficients(coefficients)
    return VarianceStabilityFunctions{FT}(map(C -> convert(FT, C), coefficients)...)
end

# The ratio 𝕊ᶜ / 𝕊ᴰ of the time scale, with its endpoint values (1 for EddyTurnoverTimeScale, which has no 𝕊ᴰ)
variance_time_scale_dissipation(::DissipationTimeScale, sf) = (sf.ℂᴰ⁻, sf.ℂᴰ⁰, sf.ℂᴰ⁺)
variance_time_scale_dissipation(::EddyTurnoverTimeScale, sf) = (1, 1, 1)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The [`VarianceStabilityFunctions`](@ref) that reproduce the down-gradient diagnosis
``ℂᵛ τ Kᶜ ∂_z a ∂_z b`` of a [`DoubleGaussian`](@ref) with the time scale `time_scale`
([`DissipationTimeScale`](@ref) or [`EddyTurnoverTimeScale`](@ref)) and the second moments
`second_moments` ([`DowngradientSecondMoments`](@ref)) in a closure with the stability functions `sf`:

```math
ℂᵛ_i^{-,0,+} = \\frac{2 ℂᵛ_i}{ℂᵠ} \\frac{ℂᶜ^{-,0,+}}{ℂᴰ^{-,0,+}} \\quad (i = 1, 2), \\qquad
ℂᵛ₃^{-,m,+} = \\frac{ℂᵛ₃}{\\sqrt{ℂᵛ₁ ℂᵛ₂}}
```

(``ℂᴰ = 1`` for [`EddyTurnoverTimeScale`](@ref)). The logarithm of a function of the family is linear in
the logarithms of its three values, so ``𝕊ᵛ_i = (2 ℂᵛ_i / ℂᵠ) 𝕊ᶜ / 𝕊ᴰ`` exactly, and the moments agree to
rounding wherever ``Kᶜ`` is below `maximum_tracer_diffusivity`. It requires ``ℂᵛ₃² ≤ ℂᵛ₁ ℂᵛ₂``, the
realizability of the down-gradient covariance.
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

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The functions ``(𝕊ᵛ₁, 𝕊ᵛ₂, 𝕏ᵛ)`` of `vs` at the coordinate ``v = Riᵗ / \\sqrt{1 + Riᵗ²}``. The logarithms
are taken here, from the coefficients, so that the functions follow coefficients replaced or traced.
"""
@inline function variance_stability_functions(vs::VarianceStabilityFunctions, v)
    𝕊ᵛ₁ = evaluate_log_quadratic(log_quadratic_coefficients(vs.ℂᵛ₁⁻, vs.ℂᵛ₁⁰, vs.ℂᵛ₁⁺), v)
    𝕊ᵛ₂ = evaluate_log_quadratic(log_quadratic_coefficients(vs.ℂᵛ₂⁻, vs.ℂᵛ₂⁰, vs.ℂᵛ₂⁺), v)
    t = (1 + v) / 2
    s = 1 - t
    𝕏ᵛ = s * s * vs.ℂᵛ₃⁻ + 2 * t * s * vs.ℂᵛ₃ᵐ + t * t * vs.ℂᵛ₃⁺
    return 𝕊ᵛ₁, 𝕊ᵛ₂, 𝕏ᵛ
end

# The local scalar second moments at the interior face k from the differences Δθ, Δq across it and its
# spacing Δz. Without variance stability functions: the down-gradient diagnosis, unchanged.
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
