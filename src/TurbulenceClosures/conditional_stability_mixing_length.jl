#####
##### `ConditionalStabilityMixingLength`: a mixing length that lets unresolved saturation variability
##### weaken the stratification that bounds it.
#####
##### A grid cell that is subsaturated in the mean may still contain saturated air. An eddy leaving
##### such a cell condenses over part of its excursion and feels less than the grid-mean stability.
##### This wrapper estimates the fraction of a trial excursion that is saturated, weakens N² towards
##### the saturated response by that fraction, and runs the wrapped model's envelope a second time:
#####
#####   ℓ₀ = 𝓔[ℓᵇ(e, N₀²)]                            the wrapped model, unchanged
#####   𝒟  = qʷ − qˢ                                   signed saturation excess, mass fraction
#####   σ𝒟² = C𝒟ᵍ ℓ₀² (∂z 𝒟)² + (C𝒟⁰)²                 unresolved width of 𝒟
#####   h  = Cʰ ℓ₀,  𝒟(s) ≃ 𝒟 + A s                    trial excursion and its linearized approach
#####   f  = ∫₀¹ Φ((𝒟 + A h ξ) / σ𝒟) dξ                saturated fraction of the excursion
#####   N₁² = N₀² − Cᶜᵒⁿᵈ f max(N₀² − Nₘ², 0)          weakening only
#####   ℓ₁ = 𝓔[ℓᵇ(e, N₁²)]                            the same envelope, a second time
#####
##### Four sweeps and a fixed three-point quadrature per level: O(Nz) per column, as the wrapped
##### model alone is. Cᶜᵒⁿᵈ = 0 returns the wrapped model's own column, bitwise, and skips the work.
#####
##### This is a length correction, not a subgrid cloud scheme. It parameterizes weakened resistance
##### to mixing. It does not resolve inhibition barriers, entrainment, parcel energy gained along a
##### path, or asymmetric updraft and downdraft statistics, and N₁² is applied to the mixing-length
##### bound alone — the Richardson number, the buoyancy flux and the stored N² are untouched.
#####

"""
$(TYPEDEF)

A mixing length that wraps another and lets **subgrid saturation variability** weaken the
stratification bounding it, for conditionally unstable air that the grid mean reports as stable.

The wrapped model — [`GradientLimitedMixingLength`](@ref), the only one supported — is evaluated
normally to give ``ℓ₀``. That length sets both a trial excursion and the width of the unresolved
variability; from them a saturated fraction of the excursion is estimated, the stability is weakened
towards its saturated value by that fraction, and **the wrapped model's own envelope is run a second
time** over the corrected stability. The wrapped model owns the surface and interior length
coefficients: this wrapper adds none and inherits whatever the model it wraps carries.

```julia
mixing_length = ConditionalStabilityMixingLength(GradientLimitedMixingLength(); Cᶜᵒⁿᵈ = 0.5)
```

# The correction

With ``𝒟 = qʷ - qˢ`` the signed saturation excess in mass-fraction units — positive in cloud, where
it is the condensate, negative in subsaturated air — its unresolved distribution is taken to be
Gaussian with width

```math
σ_𝒟² = C_𝒟^g \\, ℓ₀² \\, (∂_z 𝒟)² + (C_𝒟^0)².
```

The first term is a local gradient model: variability comes from stirring a mean gradient over the
mixing length. The second is a background width that permits variability in a well-mixed layer. It
is a phenomenological, calibratable mass-fraction width, **not an established universal constant**.
This does not reconstruct the joint temperature–moisture distribution.

A trial upward excursion of height ``h = C^h ℓ₀`` changes the saturation excess linearly,
``𝒟(s) ≃ 𝒟 + A s``, with ``A`` from local thermodynamics rather than a parcel search
([`saturation_excess_lapse_rateᶜᶜᶠ`](@ref)). The fraction of the excursion expected to be saturated
is

```math
f = ∫_0^1 Φ\\left( \\frac{𝒟 + A h ξ}{σ_𝒟} \\right) dξ,
```

with ``Φ`` the standard normal CDF, evaluated by three-point Gauss–Legendre quadrature — fixed work
per level. At ``h = 0`` this reduces to the local saturation probability ``Φ(𝒟 / σ_𝒟)``; at
``σ_𝒟 = 0`` it is the fraction of the excursion with ``𝒟(s) ≥ 0``, the saturation convention of
[`MoistStaticStability`](@ref).

The stability is then weakened, never strengthened,

```math
N₁² = N₀² - C^{\\mathrm{cond}} f \\, \\max(N₀² - N_m², 0), \\qquad 0 ≤ C^{\\mathrm{cond}} ≤ 1,
```

where ``N_m²`` is the hypothetical saturated response of [`saturated_static_stabilityᶜᶜᶠ`](@ref).
Because ``f ∈ [0, 1]`` and the correction only reduces ``N²``, ``ℓ₁ ≥ ℓ₀`` in exact arithmetic.

# What the coefficients are, and are not

Four new coefficients, all opt-in with **no calibrated nonzero defaults**: `C𝒟ᵍ ≥ 0`, `C𝒟⁰ ≥ 0` (a
mass fraction), `Cʰ ≥ 0`, and `Cᶜᵒⁿᵈ ∈ [0, 1]`. The default `Cᶜᵒⁿᵈ = 0` is the zero-strength
regression mode: it reproduces the wrapped model exactly and bypasses the correction entirely. The
trial values in the constructor's docstring are labelled as trial values, not calibrated ones.

Using ``ℓ₀`` once deliberately omits the feedback of the increased length onto the diagnosed
saturation probability; there is no fixed-point iteration.

Exact recovery of the wrapped model holds for ``C^{\\mathrm{cond}} = 0``, and for ``σ_𝒟 = 0`` wherever
the excursion stays subsaturated. It does **not** hold merely because the air is dry: a Gaussian has
infinite support, so for ``σ_𝒟 > 0`` the saturated fraction is never exactly zero and the correction
is negligible rather than absent. In air a few mass-fraction widths below saturation the weakening is
of order ``Φ(-8) ∼ 10^{-16}`` and moves ``ℓ`` by a last unit in the last place.

Fields
======

$(TYPEDFIELDS)
"""
struct ConditionalStabilityMixingLength{ML, FT} <: AbstractMixingLength
    "the mixing length this one wraps, which owns the surface and interior length coefficients"
    mixing_length :: ML
    "coefficient of the gradient contribution to the width of ``𝒟``, ``σ_𝒟² ⊃ C_𝒟^g ℓ₀² (∂_z 𝒟)²``"
    C𝒟ᵍ :: FT
    "background width of ``𝒟``, a mass fraction, which permits variability in a well-mixed layer"
    C𝒟⁰ :: FT
    "height of the trial excursion in units of the wrapped length, ``h = C^h ℓ₀``"
    Cʰ :: FT
    "strength of the weakening, in ``[0, 1]``; zero reproduces the wrapped model exactly"
    Cᶜᵒⁿᵈ :: FT
end

"""
$(TYPEDSIGNATURES)

Wrap `mixing_length` — [`GradientLimitedMixingLength`](@ref) — in the conditional-stability
correction of [`ConditionalStabilityMixingLength`](@ref).

The default `Cᶜᵒⁿᵈ = 0` leaves the wrapped model's behaviour exactly unchanged, so the feature is
off unless it is asked for. **No nonzero defaults are calibrated.** As trial values,
`C𝒟ᵍ = 1`, `C𝒟⁰ = 1e-4` (0.1 g kg⁻¹), `Cʰ = 1` and `Cᶜᵒⁿᵈ = 0.5` are of the right order — a width
comparable to the condensate a mixing length of stirring produces, an excursion of one mixing
length, and half the available weakening — but they are **guesses, not a calibration**.

The four coefficients must satisfy `C𝒟ᵍ ≥ 0`, `C𝒟⁰ ≥ 0`, `Cʰ ≥ 0` and `0 ≤ Cᶜᵒⁿᵈ ≤ 1`.

```jldoctest
using Breeze

ConditionalStabilityMixingLength(GradientLimitedMixingLength(Cˢ = 1.2), Cᶜᵒⁿᵈ = 0.5)

# output
ConditionalStabilityMixingLength{Float64}
├── mixing_length: GradientLimitedMixingLength{Float64} (Cˢ = 1.2)
├── C𝒟ᵍ: 0.0
├── C𝒟⁰: 0.0
├── Cʰ: 0.0
└── Cᶜᵒⁿᵈ: 0.5
```
"""
function ConditionalStabilityMixingLength(mixing_length::GradientLimitedMixingLength = GradientLimitedMixingLength();
                                          C𝒟ᵍ = 0.0, C𝒟⁰ = 0.0, Cʰ = 0.0, Cᶜᵒⁿᵈ = 0.0)

    C𝒟ᵍ ≥ 0 || throw(ArgumentError("C𝒟ᵍ of ConditionalStabilityMixingLength must be nonnegative, got $C𝒟ᵍ"))
    C𝒟⁰ ≥ 0 || throw(ArgumentError("C𝒟⁰ of ConditionalStabilityMixingLength must be nonnegative, got $C𝒟⁰"))
    Cʰ ≥ 0 || throw(ArgumentError("Cʰ of ConditionalStabilityMixingLength must be nonnegative, got $Cʰ"))
    0 ≤ Cᶜᵒⁿᵈ ≤ 1 ||
        throw(ArgumentError("Cᶜᵒⁿᵈ of ConditionalStabilityMixingLength must lie between 0 and 1, got $Cᶜᵒⁿᵈ"))

    return ConditionalStabilityMixingLength(mixing_length, promote(C𝒟ᵍ, C𝒟⁰, Cʰ, Cᶜᵒⁿᵈ)...)
end

@inline convert_eltype(::Type{FT}, ml::ConditionalStabilityMixingLength) where FT =
    ConditionalStabilityMixingLength(convert_eltype(FT, ml.mixing_length),
                                     convert(FT, ml.C𝒟ᵍ), convert(FT, ml.C𝒟⁰),
                                     convert(FT, ml.Cʰ), convert(FT, ml.Cᶜᵒⁿᵈ))

Base.summary(ml::ConditionalStabilityMixingLength{<:Any, FT}) where FT =
    "ConditionalStabilityMixingLength{$FT}"

# The wrapper owns no length coefficients, so its one-line summary names the model that does
mixing_length_summary(ml::ConditionalStabilityMixingLength) =
    string(summary(ml), " wrapping ", mixing_length_summary(ml.mixing_length))

function Base.show(io::IO, ml::ConditionalStabilityMixingLength)
    print(io, summary(ml), '\n',
              "├── mixing_length: ", mixing_length_summary(ml.mixing_length), '\n',
              "├── C𝒟ᵍ: ", prettysummary(ml.C𝒟ᵍ), '\n',
              "├── C𝒟⁰: ", prettysummary(ml.C𝒟⁰), '\n',
              "├── Cʰ: ", prettysummary(ml.Cʰ), '\n',
              "└── Cᶜᵒⁿᵈ: ", prettysummary(ml.Cᶜᵒⁿᵈ))
end

#####
##### The saturation excess and its approach along an excursion
#####

"""
$(TYPEDSIGNATURES)

The signed saturation excess ``𝒟 = qʷ - qˢ`` at a cell center, in mass-fraction units: the
nonprecipitating water — vapor, cloud liquid and cloud ice, rain excluded, as in
[`MoistStaticStability`](@ref) — less the saturation specific humidity at the cell's temperature and
density.

``𝒟`` is positive in cloud, where it is the condensate mass fraction, and negative in subsaturated
air, where it is the deficit of vapor below saturation. The vapor excess ``qᵛ - qˢ`` alone would not
serve: it vanishes throughout a saturated cloud in equilibrium and so carries no information about
how cloudy the cell is.
"""
@inline saturation_excessᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ) =
    nonprecipitating_waterᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ) -
    saturation_specific_humidityᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)

"""
$(TYPEDSIGNATURES)

The rate ``A = d𝒟/ds`` at which the saturation excess grows along an **unsaturated adiabatic**
displacement, at a face, in mass fraction per metre.

Along such a displacement the nonprecipitating water is conserved, so ``A = -dqˢ/ds``, and with
``qˢ = pᵛ⁺(T) / (ρ Rᵛ T)``, Clausius–Clapeyron ``d\\ln pᵛ⁺/dT = ℒ / (Rᵛ T²)``, a parcel on the dry
adiabat ``dT/ds = -g/cᵖᵈ``, and a hydrostatic environment ``d\\ln p/ds = -g/(Rᵈ T)``,

```math
A = qˢ \\frac{g}{T} \\left( \\frac{ℒ}{cᵖᵈ Rᵛ T} - \\frac{1}{Rᵈ} \\right).
```

``A > 0`` under ordinary tropospheric conditions: rising unsaturated air approaches saturation.

Two approximations are deliberate. The heat capacity and gas constant are the dry-air ones, matching
the Durran–Klemp expression already used for the saturated stability rather than mixing conventions
within one closure; and the parcel's composition is held fixed, which is exact for the unsaturated
displacement this linearization describes.
"""
@inline function saturation_excess_lapse_rateᶜᶜᶠ(i, j, k, grid, buoyancy, T, qᵛ)
    constants = buoyancy.thermodynamic_constants
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity

    Tᶠ = ℑzᵃᵃᶠ(i, j, k, grid, T)
    qˢ = ℑzᵃᵃᶠ(i, j, k, grid, saturation_specific_humidityᶜᶜᶜ, buoyancy, T, qᵛ)
    equilibrium = microphysics_phase_equilibrium(buoyancy.microphysics)
    ℒ = latent_heat(Tᶠ, constants, equilibrated_surface(equilibrium, Tᶠ))

    return qˢ * g / Tᶠ * (ℒ / (cᵖᵈ * Rᵛ * Tᶠ) - 1 / Rᵈ)
end

#####
##### The saturated fraction of a trial excursion
#####

# The standard normal CDF. `erf` is used in Breeze kernels already (aerosol activation).
@inline normal_cdf(x) = (1 + erf(x / sqrt(oftype(x, 2)))) / 2

# Three-point Gauss–Legendre on [0, 1]: exact through fifth order, fixed work per level.
@inline gauss_legendre_nodes(::Type{FT}) where FT =
    ((1 - sqrt(FT(3) / 5)) / 2, FT(1) / 2, (1 + sqrt(FT(3) / 5)) / 2)
@inline gauss_legendre_weights(::Type{FT}) where FT = (FT(5) / 18, FT(4) / 9, FT(5) / 18)

"""
$(TYPEDSIGNATURES)

The fraction of a trial excursion of height `h` expected to be saturated, given the mean saturation
excess `𝒟`, its unresolved width `σ𝒟` and its rate of change `A` along the excursion:

```math
f = ∫_0^1 Φ\\left( \\frac{𝒟 + A h ξ}{σ_𝒟} \\right) dξ,
```

by three-point Gauss–Legendre quadrature.

`σ𝒟 = 0` is taken as the limit of a sharp distribution: the fraction of the excursion whose
saturation excess is nonnegative, `𝒟(s) ≥ 0`, which is the saturation convention of
[`MoistStaticStability`](@ref) and avoids the `0/0` that dividing would produce. `h = 0` needs no
special case — every node then reports the local saturation probability `Φ(𝒟 / σ𝒟)`.
"""
@inline function saturated_fraction(𝒟, σ𝒟, A, h)
    FT = typeof(𝒟)
    ξs = gauss_legendre_nodes(FT)
    ws = gauss_legendre_weights(FT)
    sharp = σ𝒟 == 0
    f = zero(FT)
    for n in 1:3
        𝒟ₙ = 𝒟 + A * h * ξs[n]
        # The sharp limit is the indicator 𝒟ₙ ≥ 0; the division is guarded so that σ𝒟 = 0 never
        # forms 𝒟ₙ / 0, which would be NaN at 𝒟ₙ = 0 exactly.
        Φ = ifelse(sharp, FT(𝒟ₙ ≥ 0), normal_cdf(𝒟ₙ / ifelse(sharp, one(σ𝒟), σ𝒟)))
        f += ws[n] * Φ
    end
    return f
end

"""
$(TYPEDSIGNATURES)

The conditionally corrected static stability ``N₁²`` at a face, from the stored grid-mean ``N₀²``
and the wrapped model's length ``ℓ₀`` there.

The width of the unresolved saturation excess, the trial excursion and the saturated fraction are
formed as in [`ConditionalStabilityMixingLength`](@ref), and the stability is weakened towards the
hypothetical saturated response ``Nₘ²`` of [`saturated_static_stabilityᶜᶜᶠ`](@ref) — never
strengthened, so ``N₁² ≤ N₀²`` always.

``Nₘ²`` is evaluated at every face, saturated or not. Where the air is subsaturated this is a
**hypothetical** response, not the air's actual stratification: the saturation mixing ratio and the
latent heating are those the parcel would have were it saturated at the local temperature and
pressure, while the water gradient in the expression remains the air's own. It is the stability an
eddy would feel if its excursion carried it into cloud, which is precisely what the correction
weights by ``f``. It is not claimed to be an exact physical stratification of unsaturated air.
"""
@inline function conditional_static_stabilityᶜᶜᶠ(i, j, k, grid, ml, ℓ₀, N₀², buoyancy, tracers)
    T = tracers.T
    qᵛ = tracers.qᵛ

    𝒟 = ℑzᵃᵃᶠ(i, j, k, grid, saturation_excessᶜᶜᶜ, buoyancy, T, qᵛ)
    ∂z𝒟 = ∂zᶜᶜᶠ(i, j, k, grid, saturation_excessᶜᶜᶜ, buoyancy, T, qᵛ)
    A = saturation_excess_lapse_rateᶜᶜᶠ(i, j, k, grid, buoyancy, T, qᵛ)

    σ𝒟 = sqrt(ml.C𝒟ᵍ * (ℓ₀ * ∂z𝒟)^2 + ml.C𝒟⁰^2)
    h = ml.Cʰ * ℓ₀
    f = saturated_fraction(𝒟, σ𝒟, A, h)

    Nₘ² = saturated_static_stabilityᶜᶜᶠ(i, j, k, grid, buoyancy, T, qᵛ)
    weakening = ml.Cᶜᵒⁿᵈ * f * max(N₀² - Nₘ², 0)
    return N₀² - weakening
end

#####
##### The second envelope
#####

# The local bound of the second pass. `ℓ` holds the wrapped model's length: at level k it is read
# here, before `gradient_limited_sweeps!` overwrites it, so the correction needs no second field.
@inline function conditional_local_mixing_lengthᶜᶜᶠ(i, j, k, grid, ml, closure, e, N², ℓ, tracers, buoyancy)
    ℓ₀ = @inbounds ℓ[i, j, k]
    N₀² = @inbounds N²[i, j, k]
    N₁² = conditional_static_stabilityᶜᶜᶠ(i, j, k, grid, ml, ℓ₀, N₀², buoyancy, tracers)

    wrapped = ml.mixing_length
    d = wrapped.Cˢ * height_above_bottomᶜᶜᶠ(i, j, k, grid)
    vₜ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
    return local_mixing_length(d, buoyancy_penetration_depth(eltype(grid), vₜ, N₁²))
end

# Two passes of the wrapped model's envelope: the wrapped length, then the same sweeps over the
# corrected stability. `Cᶜᵒⁿᵈ = 0` returns after the first, so the zero-strength mode costs nothing
# beyond the wrapped model and reproduces it bitwise. The branch is on a closure parameter that is
# uniform over the whole grid, so it does not diverge within a warp.
@inline function fill_mixing_length!(ℓ, i, j, grid, ml::ConditionalStabilityMixingLength, closure, e, N², tracers, buoyancy)
    wrapped = ml.mixing_length
    fill_mixing_length!(ℓ, i, j, grid, wrapped, closure, e, N², tracers, buoyancy)

    if ml.Cᶜᵒⁿᵈ != 0
        gradient_limited_sweeps!(ℓ, i, j, grid, wrapped.Cˢ, conditional_local_mixing_lengthᶜᶜᶠ,
                                 ml, closure, e, N², ℓ, tracers, buoyancy)
    end

    return nothing
end

# At the cell centers the wrapper defers to the model it wraps, which owns the length coefficients;
# the stored `ℓ` it reads is already the corrected one.
@inline mixing_lengthᶜᶜᶜ(i, j, k, grid, ml::ConditionalStabilityMixingLength, closure, e, closure_fields) =
    mixing_lengthᶜᶜᶜ(i, j, k, grid, ml.mixing_length, closure, e, closure_fields)
