#####
##### Stability functions from neutral surface-layer similarity
#####

"""
    SurfaceLayerSimilarity()

Selects the constructors of [`RationalStabilityFunction`](@ref) and [`ConstantStabilityFunctions`](@ref)
whose neutral values follow from neutral surface-layer similarity.
"""
struct SurfaceLayerSimilarity end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Neutral stability functions of [`TKEBasedTurbulenceClosure`](@ref) from neutral surface-layer
similarity. In a neutral constant-stress layer the closure's mixing length is `ℓ = ℂˢ z`, with `ℂˢ`
the wall coefficient of its mixing length. Similarity requires `e = c u★²`, `Kᵘ = κ u★ z`,
`Kᶜ = κ u★ z / Pr⁰` and `ε = u★³ / (κ z)`. With `Kᵘ = 𝕊ᵘ ℓ √e`, `Kᶜ = 𝕊ᶜ ℓ √e` and
`ε = 𝕊ᴰ e^{3/2} / ℓ`, this gives

    𝕊ᵘ⁰ = κ / (ℂˢ √c),    𝕊ᶜ⁰ = κ / (ℂˢ Pr⁰ √c),    𝕊ᴰ⁰ = ℂˢ / (κ c^{3/2}),    𝕊ᵉ⁰ = 𝕊ᵘ⁰ / σᵉ.

So `𝕊ᵘ⁰ 𝕊ᴰ⁰ = 1 / c²`, and the closure's neutral TKE balance, production `Kᵘ (∂z u)² = ε`,
returns `e / u★² = 1 / √(𝕊ᵘ⁰ 𝕊ᴰ⁰) = c`. Also `𝕊ᶜ⁰ 𝕊ᴰ⁰ = 1 / (Pr⁰ c²)` for any `ℂˢ`.

Keywords: `e_u★² = c` (3.75: 3.3–5 is the observed neutral range, Panofsky and Dutton 1984;
Stull 1988), `Pr⁰` (0.85; Businger et al. 1971, Högström 1988 give 0.74–1), `κ` (0.4), `ℂˢ`
(1.316, the default of [`GradientLimitedMixingLength`](@ref); pass the closure's own) and `σᵉ`,
the TKE Schmidt number (1).
"""
function ConstantStabilityFunctions(::SurfaceLayerSimilarity; e_u★² = 3.75, Pr⁰ = 0.85, κ = 0.4, ℂˢ = 1.316, σᵉ = 1)
    c = e_u★²
    ℂᵘ = κ / (ℂˢ * sqrt(c))
    return ConstantStabilityFunctions(ℂᵘ, ℂᵘ / Pr⁰, ℂᵘ / σᵉ, ℂˢ / (κ * c * sqrt(c)))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

[`RationalStabilityFunction`](@ref) with neutral values from neutral surface-layer similarity (the
neutral values of `ConstantStabilityFunctions(SurfaceLayerSimilarity(); e_u★², Pr⁰, κ, ℂˢ, σᵉ)`) and
limits from the behaviour of second-moment closures and observations:

* Dissipation: `𝕊ᴰ = 𝕊ᴰ⁰` at every `Ri`. Observed dissipation scales with `e^{3/2} / ℓ` through the
  stability range when `ℓ` carries the stratification, as it does here through `ℓ = √e / N`
  (Mellor and Yamada 1982; Zilitinkevich et al. 2013).
* Unstable branch: `𝕊ᵘ = 𝕊ᵘ⁰`, and `𝕊ᶜ` rises to `𝕊ᵘ⁰ / Pr⁻`, with `Pr⁻ = 0.7` in free convection
  (Businger et al. 1971; Kays 1994), over the scale `Ri⁻ = 1`.
* Stable branch: the turbulent Prandtl number rises from `Pr⁰` to `Pr⁺ = 4`, roughly linearly in
  `Ri` up to `Ri ~ 1` (Anderson 2009; Zilitinkevich et al. 2013), over the scale `Ri⁺ = Ri_critical`.
  `𝕊ᵘ⁺` is set so that the steady state of the TKE equation in stratified shear, with
  `ℓ = √e / N`, sits at `Ri = Ri_critical` (0.25; Miles 1961, Mellor and Yamada 1982):
  `𝕊ᵘ(Ri_critical) / (𝕊ᶜ(Ri_critical) + 𝕊ᴰ(Ri_critical)) = Ri_critical`; above it `e` decays.
  `𝕊ᶜ⁺ = 𝕊ᵘ⁺ / Pr⁺`, so both decrease with `Ri`.
  A positive `𝕊ᵘ⁺` needs the neutral `𝕊ᵘ⁰ / (𝕊ᶜ⁰ + 𝕊ᴰ⁰)`
  not to exceed `Ri_critical` by too much: since `𝕊ᵘ⁰ / 𝕊ᴰ⁰ = κ² c / ℂˢ²`, a small wall coefficient
  (`ℂˢ ≲ 1.1` with the defaults) puts the neutral state above the critical one, and the constructor
  throws an `ArgumentError`.

`𝕊ᵉ = 𝕊ᵘ / σᵉ` at every limit.

With the defaults, `𝕊ᶜ 𝕊ᴰ` is largest in the unstable limit, `𝕊ᴰ⁰ 𝕊ᵘ⁰ / Pr⁻ = 0.101`, so the squared
correlation `ρ² = 0.75 ℂᵠ 𝕊ᶜ 𝕊ᴰ / ℂʷ₁` of [`DissipationTimeScale`](@ref) stays below `1 - ℂᵍ₁ = 0.6`
for `ℂᵠ < 7.9`: 0.152 at most for `ℂᵠ = 2`, 0.126 at neutral. In general the constraint is

    max over Ri of 3 ℂᵠ 𝕊ᶜ 𝕊ᴰ / (4 ℂʷ₁) < 1 - ℂᵍ₁,

and it is checked here for `ℂᵠ = 2`, `ℂʷ₁ = 1`, `ℂᵍ₁ = 0.4`.
"""
function RationalStabilityFunction(::SurfaceLayerSimilarity; e_u★² = 3.75, Pr⁰ = 0.85, κ = 0.4, ℂˢ = 1.316, σᵉ = 1,
                                   Pr⁻ = 0.7, Pr⁺ = 4, Ri_critical = 0.25, Ri⁻ = 1, Ri⁺ = Ri_critical)
    n = ConstantStabilityFunctions(SurfaceLayerSimilarity(); e_u★², Pr⁰, κ, ℂˢ, σᵉ)
    ℂᵘ⁰, ℂᶜ⁰, ℂᴰ⁰ = n.ℂᵘ, n.ℂᶜ, n.ℂᴰ

    # Steady state at Ri_critical: Ri (𝕊ᶜ + 𝕊ᴰ) = 𝕊ᵘ, linear in 𝕊ᵘ⁺ (𝕊ᶜ⁺ = 𝕊ᵘ⁺ / Pr⁺, 𝕊ᴰ = 𝕊ᴰ⁰)
    x = Ri_critical / (Ri_critical + Ri⁺)
    ℂᵘ⁺ = (Ri_critical * (ℂᶜ⁰ * (1 - x) + ℂᴰ⁰) - ℂᵘ⁰ * (1 - x)) / (x * (1 - Ri_critical / Pr⁺))
    ℂᵘ⁺ > 0 || throw(ArgumentError("No positive stable limit of 𝕊ᵘ gives the steady state at Ri_critical = $Ri_critical"))

    ℂᶜ⁻ = ℂᵘ⁰ / Pr⁻
    ℂᶜ⁺ = ℂᵘ⁺ / Pr⁺
    sf = RationalStabilityFunction(; ℂᵘ⁻ = ℂᵘ⁰, ℂᵘ⁰, ℂᵘ⁺, ℂᶜ⁻, ℂᶜ⁰, ℂᶜ⁺,
                                     ℂᵉ⁻ = ℂᵘ⁰ / σᵉ, ℂᵉ⁰ = ℂᵘ⁰ / σᵉ, ℂᵉ⁺ = ℂᵘ⁺ / σᵉ,
                                     ℂᴰ⁻ = ℂᴰ⁰, ℂᴰ⁰, ℂᴰ⁺ = ℂᴰ⁰, Ri⁻, Ri⁺)

    # The realizability of the double-Gaussian PDF with DissipationTimeScale(ℂᵠ = 2), ℂʷ₁ = 1, ℂᵍ₁ = 0.4
    ρ²max = 3 * 2 * max(ℂᶜ⁻ * ℂᴰ⁰, ℂᶜ⁰ * ℂᴰ⁰, ℂᶜ⁺ * ℂᴰ⁰) / 4
    ρ²max < 0.6 || @warn "RationalStabilityFunction(SurfaceLayerSimilarity()): max ρ² = $ρ²max ≥ 0.6 with ℂᵠ = 2"

    return sf
end
