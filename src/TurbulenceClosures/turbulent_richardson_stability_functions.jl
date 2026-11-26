#####
##### Stability functions of the turbulent Richardson number
#####
##### Turbulence of velocity √e and size ℓ has the time scale τ = ℓ / √e, and it feels the stratification
##### and the shear through the two numbers
#####
#####   α_N = τ² N²,   α_S = τ² (S² + S²ᵐⁱⁿ).
#####
##### The turbulent Richardson number blends the numbers of the unstable and the stable side,
#####
#####   Riᵗ = Riᵗ⁻ + (Riᵗ⁺ - Riᵗ⁻) (1 + s) / 2,   Riᵗ± = α_N / (ℂᴿⁱ± α_S + ℂᵅ),   s = x / √(1 + x²),
#####
##### with x = α_N / (δ min(ℂᴿⁱ⁻ α_S + ℂᵅ, ℂᴿⁱ⁺ α_S + ℂᵅ)) and δ = 1/4. It is Ri / ℂᴿⁱ⁻ or Ri / ℂᴿⁱ⁺
##### where the shear is strong, exactly α_N / ℂᵅ where it vanishes, and is smooth through N² = 0.
##### Each function is the exponential of a quadratic in v = Riᵗ / √(1 + Riᵗ²) ∈ (-1, 1) that takes
##### the values ℂ⁻, ℂ⁰, ℂ⁺ at v = -1, 0, 1: positive, smooth, without branches.
#####

"""
The default interval ``[lower, upper]`` of each of the fifteen coefficients of
[`TurbulentRichardsonStabilityFunction`](@ref) (see [`stability_coefficient_bounds`](@ref)).
"""
const DEFAULT_STABILITY_COEFFICIENT_BOUNDS = (0.05, 20)

# The width δ of the blend of the turbulent Richardson numbers of the unstable and stable sides, in units
# of the larger of the two: the blend is 1 - δ²/(4 Riᵗ²) + O(δ⁴) complete at |Riᵗ| (= 0.984 at |Riᵗ| = 1)
const RICHARDSON_BLEND_WIDTH = 1 // 4


# The largest logarithmic slope |∂ ln 𝕊 / ∂ Riᵗ| over Riᵗ of a function whose three values are within
# [lower, upper] is κ ln(upper / lower): max over v of |A + 2 B v| (1 - v²)^{3/2}, reached with
# ln ℂ⁻ = ln lower, ln ℂ⁰ = ln ℂ⁺ = ln upper (or its mirror), at v = ∓(√73 - 3) / 16.
const MAXIMUM_SLOPE_FACTOR = let v = (sqrt(73) - 3) / 16
    (1 / 2 + v) * (1 - v^2)^(3 / 2)
end

# The largest slope ∂Riᵗ / ∂α_N of the blend, in units of 1 / min(ℂᴿⁱ± α_S + ℂᵅ): with r ≤ 1 the ratio of
# the two denominators it is r + (1 - r) g(±x), g(x) = (1 + s) / 2 + x s' / 2, s' = (1 + x²)^{-3/2},
# whose maximum is at x = √2: g = 1/2 + 2√6 / 9 ≈ 1.044.
const MAXIMUM_BLEND_SLOPE = 1 / 2 + 2 * sqrt(6) / 9

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The least ``ℂᵅ`` of [`TurbulentRichardsonStabilityFunction`](@ref) whose functions satisfy the width
condition for any values within `[lower, upper]`, at any shear:
``|∂ \\ln 𝕊 / ∂ α_N| ≤ κ κ_b \\ln(upper / lower) / ℂᵅ``, with ``κ ≈ 0.699`` the largest slope of the
quadratic in ``v`` per unit ``\\ln(upper / lower)`` and per unit ``Riᵗ``, ``κ_b = 1/2 + 2\\sqrt{6}/9 ≈ 1.044``
the largest slope of ``Riᵗ`` in ``α_N`` per unit ``1 / \\min(ℂᴿⁱ^± α_S + ℂᵅ) ≤ 1 / ℂᵅ`` (the blend of the
two sides; ``κ_b = 1`` where the shear vanishes), and the width condition asks for a logarithmic slope
below ``(1 + p_N) / (α_L |q_a|) ≈ 1`` (the design value ``α_L |q_a| = 1``). 4.37 for the default bounds
``[0.05, 20]``, 6.72 for ``[0.01, 100]``.
"""
minimum_stratification_scale(lower = DEFAULT_STABILITY_COEFFICIENT_BOUNDS[1],
                             upper = DEFAULT_STABILITY_COEFFICIENT_BOUNDS[2]) =
    MAXIMUM_SLOPE_FACTOR * MAXIMUM_BLEND_SLOPE * log(upper / lower)

"""
$(TYPEDEF)

Stability functions for [`TKEBasedTurbulenceClosure`](@ref) of the turbulent Richardson number

```math
Riᵗ = Riᵗ⁻ + (Riᵗ⁺ - Riᵗ⁻) \\frac{1 + s}{2}, \\qquad
Riᵗ^± = \\frac{α_N}{ℂᴿⁱ^± α_S + ℂᵅ}, \\qquad s = \\frac{x}{\\sqrt{1 + x²}}, \\qquad
x = \\frac{α_N}{δ \\min(ℂᴿⁱ⁻ α_S + ℂᵅ, ℂᴿⁱ⁺ α_S + ℂᵅ)},
```

with ``α_N = τ² N²``, ``α_S = τ² (S² + S²ᵐⁱⁿ)``, ``τ = ℓ / \\sqrt{e}`` and ``δ = 1/4``, where ``ℓ`` is
the stored mixing length, ``\\sqrt{e}`` is floored and reconstructed as in the diffusivities, and
``S²`` and ``S²ᵐⁱⁿ`` are those of the closure's `shear`. ``Riᵗ`` is a convex combination of the
numbers of the unstable side, ``Riᵗ⁻``, and of the stable side, ``Riᵗ⁺``, weighted by the smooth sign
``s`` of ``α_N``, which reaches ``±1`` once ``|x| ≫ 1``, i.e. once the larger of ``|Riᵗ^±|`` exceeds
``δ``. So ``Riᵗ`` is the gradient Richardson number over ``ℂᴿⁱ⁻`` (unstable) or ``ℂᴿⁱ⁺`` (stable) where
the shear is strong, exactly ``α_N / ℂᵅ`` where it vanishes (``Riᵗ⁻ = Riᵗ⁺``), and it is smooth through
``N² = 0``, where its slope ``∂Riᵗ / ∂α_N`` is the mean of the slopes of the two sides,
``½ [1 / (ℂᴿⁱ⁻ α_S + ℂᵅ) + 1 / (ℂᴿⁱ⁺ α_S + ℂᵅ)]``, from either side. It increases with ``α_N``, with a
slope at most ``1.2 / \\min(ℂᴿⁱ^± α_S + ℂᵅ) ≤ 1.2 / ℂᵅ``. Under strong shear the stable functions are
halfway through their transition, ``v = 1/2``, at ``Ri ≈ ℂᴿⁱ⁺ / \\sqrt{3}``. Each of the four functions ``𝕊ᵘ, 𝕊ᶜ, 𝕊ᵉ, 𝕊ᴰ`` (momentum, tracers, TKE,
dissipation) is the exponential of a quadratic in ``v = Riᵗ / \\sqrt{1 + Riᵗ²} ∈ (-1, 1)``,

```math
\\ln 𝕊 = \\ln ℂ⁰ + \\tfrac{1}{2} \\ln(ℂ⁺ / ℂ⁻) \\, v + \\left[\\tfrac{1}{2} \\ln(ℂ⁺ ℂ⁻) - \\ln ℂ⁰\\right] v²,
```

so that ``𝕊 = ℂ⁻, ℂ⁰, ℂ⁺`` at ``v = -1, 0, 1``: the strongly unstable, neutral and strongly stable
values. The functions are positive for any positive coefficients, smooth, and have no branches.

The fifteen coefficients are the fields [`closure_parameters`](@ref) lists and
[`with_parameters`](@ref) replaces. The logarithms are cached at construction, so a face costs
``Riᵗ``, two square roots and four exponentials. [`stability_coefficient_bounds`](@ref) gives an
interval for each, independent of the others, for a calibration; [`bounded_coefficient`](@ref) maps
an unconstrained real number into one.

The logarithmic slope of a function in ``α_N`` is at most ``κ κ_b \\ln(ℂ^{max} / ℂ^{min}) / ℂᵅ``,
``κ ≈ 0.699``, ``κ_b ≈ 1.044`` (``κ_b = 1`` where the shear vanishes): the lower bound of ``ℂᵅ``,
[`minimum_stratification_scale`](@ref), keeps it below the width condition of the stability
functions at any shear (4.37 for the default bounds). The ratio ``ℂᴿⁱ⁻ / ℂᴿⁱ⁺`` does not enter this
bound, nor the slopes of ``\\ln 𝕊`` in ``\\ln |Ri|`` under strong shear, which do not depend on the
scales ``ℂᴿⁱ^±``. Since ``α_N ≤ 1`` where ``ℓ ≤ \\sqrt{e} / N``, without
shear ``Riᵗ ≤ 1 / ℂᵅ`` on the stable side: the strongly stable values are reached through the
Richardson number.

Fields
======

$(TYPEDFIELDS)
"""
struct TurbulentRichardsonStabilityFunction{FT}
    "momentum, strongly unstable"
    ℂᵘ⁻ :: FT
    "momentum, neutral"
    ℂᵘ⁰ :: FT
    "momentum, strongly stable"
    ℂᵘ⁺ :: FT
    "tracers, strongly unstable"
    ℂᶜ⁻ :: FT
    "tracers, neutral"
    ℂᶜ⁰ :: FT
    "tracers, strongly stable"
    ℂᶜ⁺ :: FT
    "TKE, strongly unstable"
    ℂᵉ⁻ :: FT
    "TKE, neutral"
    ℂᵉ⁰ :: FT
    "TKE, strongly stable"
    ℂᵉ⁺ :: FT
    "dissipation, strongly unstable"
    ℂᴰ⁻ :: FT
    "dissipation, neutral"
    ℂᴰ⁰ :: FT
    "dissipation, strongly stable"
    ℂᴰ⁺ :: FT
    "Richardson number at which ``Riᵗ = -1`` where the shear is strong (unstable side)"
    ℂᴿⁱ⁻ :: FT
    "Richardson number at which ``Riᵗ ≈ 1`` where the shear is strong (stable side)"
    ℂᴿⁱ⁺ :: FT
    "stratification number ``α_N`` at which ``Riᵗ = 1`` where the shear vanishes"
    ℂᵅ :: FT
    "``(\\ln ℂ⁰, \\tfrac{1}{2} \\ln(ℂ⁺ / ℂ⁻), \\tfrac{1}{2} \\ln(ℂ⁺ ℂ⁻) - \\ln ℂ⁰)`` of the momentum, tracer, TKE and dissipation functions (cached)"
    quadratic_coefficients :: NTuple{4, NTuple{3, FT}}
end

const TRStabilityFunction = TurbulentRichardsonStabilityFunction

"""The names of the fifteen coefficients of [`TurbulentRichardsonStabilityFunction`](@ref)."""
const TURBULENT_RICHARDSON_COEFFICIENTS = (:ℂᵘ⁻, :ℂᵘ⁰, :ℂᵘ⁺, :ℂᶜ⁻, :ℂᶜ⁰, :ℂᶜ⁺, :ℂᵉ⁻, :ℂᵉ⁰, :ℂᵉ⁺,
                                           :ℂᴰ⁻, :ℂᴰ⁰, :ℂᴰ⁺, :ℂᴿⁱ⁻, :ℂᴿⁱ⁺, :ℂᵅ)

@inline function log_quadratic_coefficients(C⁻, C⁰, C⁺)
    l⁻, l⁰, l⁺ = log(C⁻), log(C⁰), log(C⁺)
    return (l⁰, (l⁺ - l⁻) / 2, (l⁺ + l⁻) / 2 - l⁰)
end

# From the fifteen coefficients, in the order of TURBULENT_RICHARDSON_COEFFICIENTS, with the cache
@inline function TurbulentRichardsonStabilityFunction(ℂᵘ⁻::FT, ℂᵘ⁰::FT, ℂᵘ⁺::FT, ℂᶜ⁻::FT, ℂᶜ⁰::FT, ℂᶜ⁺::FT,
                                                      ℂᵉ⁻::FT, ℂᵉ⁰::FT, ℂᵉ⁺::FT, ℂᴰ⁻::FT, ℂᴰ⁰::FT, ℂᴰ⁺::FT,
                                                      ℂᴿⁱ⁻::FT, ℂᴿⁱ⁺::FT, ℂᵅ::FT) where FT
    cache = (log_quadratic_coefficients(ℂᵘ⁻, ℂᵘ⁰, ℂᵘ⁺),
             log_quadratic_coefficients(ℂᶜ⁻, ℂᶜ⁰, ℂᶜ⁺),
             log_quadratic_coefficients(ℂᵉ⁻, ℂᵉ⁰, ℂᵉ⁺),
             log_quadratic_coefficients(ℂᴰ⁻, ℂᴰ⁰, ℂᴰ⁺))
    return TurbulentRichardsonStabilityFunction{FT}(ℂᵘ⁻, ℂᵘ⁰, ℂᵘ⁺, ℂᶜ⁻, ℂᶜ⁰, ℂᶜ⁺, ℂᵉ⁻, ℂᵉ⁰, ℂᵉ⁺,
                                                    ℂᴰ⁻, ℂᴰ⁰, ℂᴰ⁺, ℂᴿⁱ⁻, ℂᴿⁱ⁺, ℂᵅ, cache)
end

@inline stability_coefficients(sf::TRStabilityFunction) = map(name -> getfield(sf, name), TURBULENT_RICHARDSON_COEFFICIENTS)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`TurbulentRichardsonStabilityFunction`](@ref) from its fifteen positive coefficients.

The defaults are hand-tuned starting values for calibration. The neutral momentum coefficient
``ℂᵘ⁰`` and tracer coefficient ``ℂᶜ⁰`` (turbulent Prandtl number 0.80) are those of the provisional
calibration of the TKE closure to the Shen et al. (2022) LES library in the calibration harness; the
neutral dissipation coefficient ``ℂᴰ⁰ = 1 / (3.75² ℂᵘ⁰) = 0.447`` gives ``e / u★² = 3.75`` in a neutral
constant-stress layer, and the default wall coefficient ``ℂˢ = 0.4 (ℂᴰ⁰ / ℂᵘ⁰³)^{1/4} = 1.30`` gives
von Kármán's constant 0.4 there. The dissipation coefficient is the same in every regime
(``ℂᴰ⁻ = ℂᴰ⁺ = ℂᴰ⁰``), and the TKE coefficients equal the momentum coefficients (TKE Schmidt number 1).
On the unstable side ``ℂᵘ⁻ = 0.361`` is that calibration's and ``ℂᶜ⁻ = 0.45`` is above it (Prandtl
number 0.8), so that the correlation of ``w`` with the scalars of the double-Gaussian distribution,
``(3 ℂᵠ 𝕊ᶜ 𝕊ᴰ / 4)^{1/2}``, is about 0.4 in convection. On the stable side mixing decreases with
stability: ``ℂᵘ⁺ = 0.135`` and ``ℂᶜ⁺ = 0.09`` (Prandtl number 1.5) make the steady gradient Richardson
number of uniformly stratified turbulence ``ℂᵘ⁺ / (ℂᶜ⁺ + ℂᴰ⁺) = 0.25``. ``ℂᴿⁱ⁻ = 16`` fits ``Riᵗ`` to
the unstable transition of the calibrated functions. ``ℂᴿⁱ⁺ = 0.33``
puts the stable transition where turbulence observations show marginal stability: under strong shear
``v = 1/2``, half of the way from neutral to strongly stable, at ``Ri = 0.20`` (``v = 0.1`` at
``Ri = 0.045`` and ``v = 0.9`` at ``Ri = 0.69``). ``ℂᵅ = 4.4`` is just above the least value the width
condition allows within the default bounds, 4.37 ([`minimum_stratification_scale`](@ref)).

```jldoctest
using Breeze

TurbulentRichardsonStabilityFunction()

# output
TurbulentRichardsonStabilityFunction{Float64}
├── Riᵗ → -∞ (ℂᵘ⁻, ℂᶜ⁻, ℂᵉ⁻, ℂᴰ⁻): 0.36113, 0.45, 0.36113, 0.447224
├── Riᵗ = 0  (ℂᵘ⁰, ℂᶜ⁰, ℂᵉ⁰, ℂᴰ⁰): 0.159006, 0.198722, 0.159006, 0.447224
├── Riᵗ → ∞  (ℂᵘ⁺, ℂᶜ⁺, ℂᵉ⁺, ℂᴰ⁺): 0.135, 0.09, 0.135, 0.447224
└── Riᵗ± = α_N / (ℂᴿⁱ± α_S + ℂᵅ): ℂᴿⁱ⁻ = 16.0, ℂᴿⁱ⁺ = 0.33, ℂᵅ = 4.4
```
"""
function TurbulentRichardsonStabilityFunction(FT::DataType = Oceananigans.defaults.FloatType;
                                              ℂᵘ⁻ = 0.36113018152281917, ℂᵘ⁰ = 0.15900569445804447, ℂᵘ⁺ = 0.135,
                                              ℂᶜ⁻ = 0.45, ℂᶜ⁰ = 0.19872163735761372, ℂᶜ⁺ = 0.09,
                                              ℂᵉ⁻ = 0.36113018152281917, ℂᵉ⁰ = 0.15900569445804447, ℂᵉ⁺ = 0.135,
                                              ℂᴰ⁻ = 0.4472236755638624, ℂᴰ⁰ = 0.4472236755638624, ℂᴰ⁺ = 0.4472236755638624,
                                              ℂᴿⁱ⁻ = 16, ℂᴿⁱ⁺ = 0.33, ℂᵅ = 4.4)

    coefficients = (ℂᵘ⁻, ℂᵘ⁰, ℂᵘ⁺, ℂᶜ⁻, ℂᶜ⁰, ℂᶜ⁺, ℂᵉ⁻, ℂᵉ⁰, ℂᵉ⁺, ℂᴰ⁻, ℂᴰ⁰, ℂᴰ⁺, ℂᴿⁱ⁻, ℂᴿⁱ⁺, ℂᵅ)
    all(C -> C > 0, coefficients) ||
        throw(ArgumentError("The coefficients $(TURBULENT_RICHARDSON_COEFFICIENTS) of " *
                            "TurbulentRichardsonStabilityFunction must be positive, but they are $coefficients"))

    return TurbulentRichardsonStabilityFunction(map(C -> convert(FT, C), coefficients)...)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The interval of each coefficient of a [`TurbulentRichardsonStabilityFunction`](@ref) for a
calibration, as a `NamedTuple` of `(lower, upper)`: `[lower, upper]` for each, except ``ℂᵅ``, whose
lower bound is [`minimum_stratification_scale`](@ref)`(lower, upper)`, the least value with which the
functions satisfy the width condition for any values within the bounds. The intervals are
independent of each other; the functions are well defined for any positive coefficients.
"""
function stability_coefficient_bounds(lower = DEFAULT_STABILITY_COEFFICIENT_BOUNDS[1],
                                      upper = DEFAULT_STABILITY_COEFFICIENT_BOUNDS[2])
    0 < lower < upper || throw(ArgumentError("The bounds must satisfy 0 < lower < upper, but they are ($lower, $upper)"))
    ℂᵅᵐⁱⁿ = minimum_stratification_scale(lower, upper)
    ℂᵅᵐⁱⁿ < upper || throw(ArgumentError("The least ℂᵅ of the width condition, $ℂᵅᵐⁱⁿ, is above upper = $upper"))
    intervals = map(name -> name === :ℂᵅ ? (ℂᵅᵐⁱⁿ, upper) : (lower, upper), TURBULENT_RICHARDSON_COEFFICIENTS)
    return NamedTuple{TURBULENT_RICHARDSON_COEFFICIENTS}(intervals)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Whether every coefficient of `sf` is within its interval of `bounds`
([`stability_coefficient_bounds`](@ref) by default).
"""
within_bounds(sf::TRStabilityFunction, bounds = stability_coefficient_bounds()) =
    all(name -> bounds[name][1] ≤ getfield(sf, name) ≤ bounds[name][2], TURBULENT_RICHARDSON_COEFFICIENTS)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The value within `(lower, upper)` of the unconstrained real number `θ`,
``\\exp[\\ln lower + \\ln(upper / lower) σ(θ)]`` with ``σ(θ) = 1 / (1 + e^{-θ})``: for calibration code
that needs unconstrained parameters. [`unbounded_coefficient`](@ref) is its inverse.
"""
@inline bounded_coefficient(θ, lower, upper) = exp(log(lower) + log(upper / lower) / (1 + exp(-θ)))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The unconstrained real number whose [`bounded_coefficient`](@ref) in `(lower, upper)` is `C`.
"""
function unbounded_coefficient(C, lower, upper)
    σ = log(C / lower) / log(upper / lower)
    return log(σ / (1 - σ))
end

@inline convert_eltype(::Type{FT}, sf::TRStabilityFunction) where FT =
    TurbulentRichardsonStabilityFunction(map(C -> convert(FT, C), stability_coefficients(sf))...)

Adapt.@adapt_structure TurbulentRichardsonStabilityFunction

Base.summary(sf::TRStabilityFunction{FT}) where FT = "TurbulentRichardsonStabilityFunction{$FT}"

function show_stability_function_lines(io::IO, sf::TRStabilityFunction, prefix)
    print(io, prefix, "├── Riᵗ → -∞ (ℂᵘ⁻, ℂᶜ⁻, ℂᵉ⁻, ℂᴰ⁻): ", stability_endpoint_summary(sf.ℂᵘ⁻, sf.ℂᶜ⁻, sf.ℂᵉ⁻, sf.ℂᴰ⁻), '\n',
              prefix, "├── Riᵗ = 0  (ℂᵘ⁰, ℂᶜ⁰, ℂᵉ⁰, ℂᴰ⁰): ", stability_endpoint_summary(sf.ℂᵘ⁰, sf.ℂᶜ⁰, sf.ℂᵉ⁰, sf.ℂᴰ⁰), '\n',
              prefix, "├── Riᵗ → ∞  (ℂᵘ⁺, ℂᶜ⁺, ℂᵉ⁺, ℂᴰ⁺): ", stability_endpoint_summary(sf.ℂᵘ⁺, sf.ℂᶜ⁺, sf.ℂᵉ⁺, sf.ℂᴰ⁺), '\n',
              prefix, "└── Riᵗ± = α_N / (ℂᴿⁱ± α_S + ℂᵅ): ℂᴿⁱ⁻ = ", prettysummary(sf.ℂᴿⁱ⁻),
              ", ℂᴿⁱ⁺ = ", prettysummary(sf.ℂᴿⁱ⁺), ", ℂᵅ = ", prettysummary(sf.ℂᵅ))
    return nothing
end

function Base.show(io::IO, sf::TRStabilityFunction)
    print(io, summary(sf), '\n')
    show_stability_function_lines(io, sf, "")
end

#####
##### The functions of the stratification and shear numbers
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The turbulent Richardson number ``Riᵗ`` of `sf` at the stratification number `αᴺ` and the shear
number `αˢ ≥ 0`: the blend ``Riᵗ⁻ + (Riᵗ⁺ - Riᵗ⁻)(1 + s)/2`` of the numbers
``Riᵗ^± = α_N / (ℂᴿⁱ^± α_S + ℂᵅ)`` of the two sides by the smooth sign ``s = x / \\sqrt{1 + x²}``,
``x = α_N / [δ \\min(ℂᴿⁱ^± α_S + ℂᵅ)]``, ``δ =`` `RICHARDSON_BLEND_WIDTH`.
"""
@inline function turbulent_richardson_number(sf::TRStabilityFunction, αᴺ, αˢ)
    D⁻ = sf.ℂᴿⁱ⁻ * αˢ + sf.ℂᵅ
    D⁺ = sf.ℂᴿⁱ⁺ * αˢ + sf.ℂᵅ
    Riᵗ⁻ = αᴺ / D⁻
    Riᵗ⁺ = αᴺ / D⁺
    # s = x / √(1 + x²) with x = αᴺ / (δ min(D⁻, D⁺)), formed without x² so that it is finite for any αᴺ
    δD = RICHARDSON_BLEND_WIDTH * min(D⁻, D⁺)
    s = αᴺ / hypot(δD, αᴺ)
    return Riᵗ⁻ + (Riᵗ⁺ - Riᵗ⁻) * (1 + s) / 2
end

# v = Riᵗ / √(1 + Riᵗ²) ∈ [-1, 1], formed without Riᵗ² so that it is finite for any Riᵗ
@inline function bounded_richardson_coordinate(sf::TRStabilityFunction, αᴺ, αˢ)
    Riᵗ = turbulent_richardson_number(sf, αᴺ, αˢ)
    return Riᵗ / hypot(one(Riᵗ), Riᵗ)
end

@inline evaluate_log_quadratic((l⁰, A, B), v) = exp(l⁰ + v * (A + B * v))

@inline momentum_stability_function(sf::TRStabilityFunction, αᴺ, αˢ) =
    evaluate_log_quadratic(sf.quadratic_coefficients[1], bounded_richardson_coordinate(sf, αᴺ, αˢ))

@inline tracer_stability_function(sf::TRStabilityFunction, αᴺ, αˢ) =
    evaluate_log_quadratic(sf.quadratic_coefficients[2], bounded_richardson_coordinate(sf, αᴺ, αˢ))

@inline tke_stability_function(sf::TRStabilityFunction, αᴺ, αˢ) =
    evaluate_log_quadratic(sf.quadratic_coefficients[3], bounded_richardson_coordinate(sf, αᴺ, αˢ))

@inline dissipation_stability_function(sf::TRStabilityFunction, αᴺ, αˢ) =
    evaluate_log_quadratic(sf.quadratic_coefficients[4], bounded_richardson_coordinate(sf, αᴺ, αˢ))

#####
##### The functions of the local gradients, for the closures that form them at a point
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The stability functions ``(𝕊ᵘ, 𝕊ᶜ, 𝕊ᵉ, 𝕊ᴰ)`` of `sf` from the squared buoyancy frequency `N²`, the
squared shear `S²` and its floor `S²ᵐⁱⁿ`, and the squared time scale `τ²` ``= ℓ² / e`` of the
turbulence (which the functions of the Richardson number do not use).
"""
@inline local_stability_functions(sf::ConstantStabilityFunctions, N², S², S²ᵐⁱⁿ, τ²) = (sf.ℂᵘ, sf.ℂᶜ, sf.ℂᵉ, sf.ℂᴰ)

@inline function local_stability_functions(sf::RationalStabilityFunction, N², S², S²ᵐⁱⁿ, τ²)
    Ri = richardson_number(N², S², S²ᵐⁱⁿ)
    return (momentum_stability_function(sf, Ri), tracer_stability_function(sf, Ri),
            tke_stability_function(sf, Ri), dissipation_stability_function(sf, Ri))
end

@inline function local_stability_functions(sf::TRStabilityFunction, N², S², S²ᵐⁱⁿ, τ²)
    v = bounded_richardson_coordinate(sf, τ² * N², τ² * (S² + S²ᵐⁱⁿ))
    return map(q -> evaluate_log_quadratic(q, v), sf.quadratic_coefficients)
end

#####
##### Evaluation on the grid
#####

const TurbulentRichardsonClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:TRStabilityFunction}

# The stratification and shear numbers at the face, with τ² = ℓ² / e from the stored mixing length and
# √e floored and reconstructed as in the diffusivity ℓ √e. Both vanish where ℓ = 0 (the surface).
@inline function stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
    τ² = (@inbounds ℓ[i, j, k] / w★)^2
    S² = effective_shear_squaredᶜᶜᶠ(i, j, k, grid, closure, velocities, e, ℓ) + closure.shear.minimum_shear_squared
    return τ² * @inbounds(N²[i, j, k]), τ² * S²
end

# At the center, where the dissipation lives: N² and S² reconstructed from the two faces, as for the
# Richardson number, and the time scale of the reconstructed mixing length bounded by the penetration
# time ℂˡᵇ / N of the center with the blending of the mixing length, as the length of the faces is by
# ℂˡᵇ √e / N: α_N ≤ (ℂˡᵇ)² there as at the faces.
@inline function stratification_and_shear_numbersᶜᶜᶜ(i, j, k, grid, closure, velocities, N², e, ℓ)
    S² = ℑbzᵃᵃᶜ(i, j, k, grid, effective_shear_squaredᶜᶜᶠ, closure, velocities, e, ℓ) + closure.shear.minimum_shear_squared
    N²ᵢ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
    τʷ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, ℓ) / turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    stratified = N²ᵢ > 0
    τᵇ = buoyancy_length_coefficient(closure.mixing_length) / sqrt(ifelse(stratified, N²ᵢ, one(N²ᵢ)))
    τ = blend(length_blending(closure.mixing_length), τʷ, ifelse(stratified, τᵇ, convert(typeof(τᵇ), Inf)))
    τ² = τ * τ
    return τ² * N²ᵢ, τ² * S²
end

@inline function momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure::TurbulentRichardsonClosure, velocities, N², e, ℓ)
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    return momentum_stability_function(closure.stability_functions, αᴺ, αˢ)
end

@inline function tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure::TurbulentRichardsonClosure, velocities, N², e, ℓ)
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    return tracer_stability_function(closure.stability_functions, αᴺ, αˢ)
end

@inline function tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure::TurbulentRichardsonClosure, velocities, N², e, ℓ)
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    return tke_stability_function(closure.stability_functions, αᴺ, αˢ)
end

@inline function dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure::TurbulentRichardsonClosure, velocities, N², e, ℓ)
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    return dissipation_stability_function(closure.stability_functions, αᴺ, αˢ)
end

@inline function dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure::TurbulentRichardsonClosure, velocities, N², e, ℓ)
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶜ(i, j, k, grid, closure, velocities, N², e, ℓ)
    return dissipation_stability_function(closure.stability_functions, αᴺ, αˢ)
end
