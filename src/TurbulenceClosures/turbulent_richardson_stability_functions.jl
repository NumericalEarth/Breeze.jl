#####
##### Stability functions of the turbulent Richardson number
#####
##### Turbulence of velocity √e and size ℓ has the time scale τ = ℓ / √e, and it feels the stratification
##### and the shear through the two numbers
#####
#####   α_N = τ² N²,   α_S = τ² (S² + S²ᵐⁱⁿ).
#####
##### The turbulent Richardson number
#####
#####   Riᵗ = α_N / (ℂᴿⁱ α_S + ℂᵅ) = N² / (ℂᴿⁱ (S² + S²ᵐⁱⁿ) + ℂᵅ e / ℓ²)
#####
##### is Ri / ℂᴿⁱ where the shear is strong and α_N / ℂᵅ where it vanishes, and is smooth through N² = 0.
##### Each function is the exponential of a quadratic in v = Riᵗ / √(1 + Riᵗ²) ∈ (-1, 1) that takes
##### the values ℂ⁻, ℂ⁰, ℂ⁺ at v = -1, 0, 1: positive, smooth, without branches.
#####

"""
The default interval ``[lower, upper]`` of each of the fourteen coefficients of
[`TurbulentRichardsonStabilityFunction`](@ref) (see [`stability_coefficient_bounds`](@ref)).
"""
const DEFAULT_STABILITY_COEFFICIENT_BOUNDS = (0.05, 20)

# The largest logarithmic slope |∂ ln 𝕊 / ∂ Riᵗ| over Riᵗ of a function whose three values are within
# [lower, upper] is κ ln(upper / lower): max over v of |A + 2 B v| (1 - v²)^{3/2}, reached with
# ln ℂ⁻ = ln lower, ln ℂ⁰ = ln ℂ⁺ = ln upper (or its mirror), at v = ∓(√73 - 3) / 16.
const MAXIMUM_SLOPE_FACTOR = let v = (sqrt(73) - 3) / 16
    (1 / 2 + v) * (1 - v^2)^(3 / 2)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The least ``ℂᵅ`` of [`TurbulentRichardsonStabilityFunction`](@ref) whose functions satisfy the width
condition for any values within `[lower, upper]`: where the shear vanishes
``|∂ \\ln 𝕊 / ∂ α_N| ≤ κ \\ln(upper / lower) / ℂᵅ``, with ``κ ≈ 0.699`` the largest slope of the
quadratic in ``v`` per unit ``\\ln(upper / lower)`` and per unit ``Riᵗ``, and the width condition asks
for a logarithmic slope below ``(1 + p_N) / (α_L |q_a|) ≈ 1`` (the design value ``α_L |q_a| = 1``).
4.19 for the default bounds ``[0.05, 20]``, 6.44 for ``[0.01, 100]``.
"""
minimum_stratification_scale(lower = DEFAULT_STABILITY_COEFFICIENT_BOUNDS[1],
                             upper = DEFAULT_STABILITY_COEFFICIENT_BOUNDS[2]) =
    MAXIMUM_SLOPE_FACTOR * log(upper / lower)

"""
$(TYPEDEF)

Stability functions for [`TKEBasedTurbulenceClosure`](@ref) of the turbulent Richardson number

```math
Riᵗ = \\frac{α_N}{ℂᴿⁱ α_S + ℂᵅ} = \\frac{N²}{ℂᴿⁱ (S² + S²ᵐⁱⁿ) + ℂᵅ e / ℓ²},
\\qquad α_N = τ² N², \\quad α_S = τ² (S² + S²ᵐⁱⁿ), \\quad τ = ℓ / \\sqrt{e},
```

where ``ℓ`` is the stored mixing length, ``\\sqrt{e}`` is floored and reconstructed as in the
diffusivities, and ``S²`` and ``S²ᵐⁱⁿ`` are those of the closure's `shear`. ``Riᵗ`` is the gradient
Richardson number over ``ℂᴿⁱ`` where the shear is strong, ``α_N / ℂᵅ`` where it vanishes, and is
smooth through ``N² = 0``. Each of the four functions ``𝕊ᵘ, 𝕊ᶜ, 𝕊ᵉ, 𝕊ᴰ`` (momentum, tracers, TKE,
dissipation) is the exponential of a quadratic in ``v = Riᵗ / \\sqrt{1 + Riᵗ²} ∈ (-1, 1)``,

```math
\\ln 𝕊 = \\ln ℂ⁰ + \\tfrac{1}{2} \\ln(ℂ⁺ / ℂ⁻) \\, v + \\left[\\tfrac{1}{2} \\ln(ℂ⁺ ℂ⁻) - \\ln ℂ⁰\\right] v²,
```

so that ``𝕊 = ℂ⁻, ℂ⁰, ℂ⁺`` at ``v = -1, 0, 1``: the strongly unstable, neutral and strongly stable
values. The functions are positive for any positive coefficients, smooth, and have no branches.

The fourteen coefficients are the fields [`closure_parameters`](@ref) lists and
[`with_parameters`](@ref) replaces. The logarithms are cached at construction, so a face costs
``Riᵗ``, one square root and four exponentials. [`stability_coefficient_bounds`](@ref) gives an
interval for each, independent of the others, for a calibration; [`bounded_coefficient`](@ref) maps
an unconstrained real number into one.

Where the shear vanishes the logarithmic slope of a function in ``α_N`` is at most
``κ \\ln(ℂ^{max} / ℂ^{min}) / ℂᵅ``, ``κ ≈ 0.699``: the lower bound of ``ℂᵅ``,
[`minimum_stratification_scale`](@ref), keeps it below the width condition of the stability
functions (4.19 for the default bounds). Since ``α_N ≤ 1`` where ``ℓ ≤ \\sqrt{e} / N``, without
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
    "Richardson number at which ``Riᵗ = 1`` where the shear is strong"
    ℂᴿⁱ :: FT
    "stratification number ``α_N`` at which ``Riᵗ = 1`` where the shear vanishes"
    ℂᵅ :: FT
    "``(\\ln ℂ⁰, \\tfrac{1}{2} \\ln(ℂ⁺ / ℂ⁻), \\tfrac{1}{2} \\ln(ℂ⁺ ℂ⁻) - \\ln ℂ⁰)`` of the momentum, tracer, TKE and dissipation functions (cached)"
    quadratic_coefficients :: NTuple{4, NTuple{3, FT}}
end

const TRStabilityFunction = TurbulentRichardsonStabilityFunction

"""The names of the fourteen coefficients of [`TurbulentRichardsonStabilityFunction`](@ref)."""
const TURBULENT_RICHARDSON_COEFFICIENTS = (:ℂᵘ⁻, :ℂᵘ⁰, :ℂᵘ⁺, :ℂᶜ⁻, :ℂᶜ⁰, :ℂᶜ⁺, :ℂᵉ⁻, :ℂᵉ⁰, :ℂᵉ⁺,
                                           :ℂᴰ⁻, :ℂᴰ⁰, :ℂᴰ⁺, :ℂᴿⁱ, :ℂᵅ)

@inline function log_quadratic_coefficients(C⁻, C⁰, C⁺)
    l⁻, l⁰, l⁺ = log(C⁻), log(C⁰), log(C⁺)
    return (l⁰, (l⁺ - l⁻) / 2, (l⁺ + l⁻) / 2 - l⁰)
end

# From the fourteen coefficients, in the order of TURBULENT_RICHARDSON_COEFFICIENTS, with the cache
@inline function TurbulentRichardsonStabilityFunction(ℂᵘ⁻::FT, ℂᵘ⁰::FT, ℂᵘ⁺::FT, ℂᶜ⁻::FT, ℂᶜ⁰::FT, ℂᶜ⁺::FT,
                                                      ℂᵉ⁻::FT, ℂᵉ⁰::FT, ℂᵉ⁺::FT, ℂᴰ⁻::FT, ℂᴰ⁰::FT, ℂᴰ⁺::FT,
                                                      ℂᴿⁱ::FT, ℂᵅ::FT) where FT
    cache = (log_quadratic_coefficients(ℂᵘ⁻, ℂᵘ⁰, ℂᵘ⁺),
             log_quadratic_coefficients(ℂᶜ⁻, ℂᶜ⁰, ℂᶜ⁺),
             log_quadratic_coefficients(ℂᵉ⁻, ℂᵉ⁰, ℂᵉ⁺),
             log_quadratic_coefficients(ℂᴰ⁻, ℂᴰ⁰, ℂᴰ⁺))
    return TurbulentRichardsonStabilityFunction{FT}(ℂᵘ⁻, ℂᵘ⁰, ℂᵘ⁺, ℂᶜ⁻, ℂᶜ⁰, ℂᶜ⁺, ℂᵉ⁻, ℂᵉ⁰, ℂᵉ⁺,
                                                    ℂᴰ⁻, ℂᴰ⁰, ℂᴰ⁺, ℂᴿⁱ, ℂᵅ, cache)
end

@inline stability_coefficients(sf::TRStabilityFunction) = map(name -> getfield(sf, name), TURBULENT_RICHARDSON_COEFFICIENTS)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`TurbulentRichardsonStabilityFunction`](@ref) from its fourteen positive coefficients.

The defaults are the effective values of the stratification-number functions that preceded these (the
provisional calibration of the TKE closure to the Shen et al. (2022) LES library in the calibration
harness), clipped into the default bounds ``[0.05, 20]``: ``ℂᵘ⁺`` (from 0.0110), ``ℂᵉ⁻`` (0.0442),
``ℂᵉ⁰`` (0.00731), ``ℂᵉ⁺`` (0.0156) and ``ℂᴰ⁻`` (0.00541) are clipped to 0.05 and ``ℂᴰ⁰`` (0.0292)
to 0.05. ``ℂᴿⁱ = 8.7`` fits ``Riᵗ`` to the transitions of those functions (at ``|Ri| ∼ 12.7`` unstable
and ``0.11`` stable, which no single ``ℂᴿⁱ`` reproduces) and ``ℂᵅ = 4.2`` is the least value the width
condition allows within the default bounds ([`minimum_stratification_scale`](@ref)).

```jldoctest
using Breeze

TurbulentRichardsonStabilityFunction()

# output
TurbulentRichardsonStabilityFunction{Float64}
├── Riᵗ → -∞ (ℂᵘ⁻, ℂᶜ⁻, ℂᵉ⁻, ℂᴰ⁻): 0.36113, 0.0817085, 0.05, 0.05
├── Riᵗ = 0  (ℂᵘ⁰, ℂᶜ⁰, ℂᵉ⁰, ℂᴰ⁰): 0.159006, 0.198722, 0.05, 0.05
├── Riᵗ → ∞  (ℂᵘ⁺, ℂᶜ⁺, ℂᵉ⁺, ℂᴰ⁺): 0.05, 0.423567, 0.05, 0.062201
└── Riᵗ = α_N / (ℂᴿⁱ α_S + ℂᵅ): ℂᴿⁱ = 8.7, ℂᵅ = 4.2
```
"""
function TurbulentRichardsonStabilityFunction(FT::DataType = Oceananigans.defaults.FloatType;
                                              ℂᵘ⁻ = 0.36113018152281917, ℂᵘ⁰ = 0.15900569445804447, ℂᵘ⁺ = 0.05,
                                              ℂᶜ⁻ = 0.0817085, ℂᶜ⁰ = 0.19872163735761372, ℂᶜ⁺ = 0.423567,
                                              ℂᵉ⁻ = 0.05, ℂᵉ⁰ = 0.05, ℂᵉ⁺ = 0.05,
                                              ℂᴰ⁻ = 0.05, ℂᴰ⁰ = 0.05, ℂᴰ⁺ = 0.062201,
                                              ℂᴿⁱ = 8.7, ℂᵅ = 4.2)

    coefficients = (ℂᵘ⁻, ℂᵘ⁰, ℂᵘ⁺, ℂᶜ⁻, ℂᶜ⁰, ℂᶜ⁺, ℂᵉ⁻, ℂᵉ⁰, ℂᵉ⁺, ℂᴰ⁻, ℂᴰ⁰, ℂᴰ⁺, ℂᴿⁱ, ℂᵅ)
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
              prefix, "└── Riᵗ = α_N / (ℂᴿⁱ α_S + ℂᵅ): ℂᴿⁱ = ", prettysummary(sf.ℂᴿⁱ), ", ℂᵅ = ", prettysummary(sf.ℂᵅ))
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

The turbulent Richardson number ``Riᵗ = α_N / (ℂᴿⁱ α_S + ℂᵅ)`` of `sf` at the stratification number
`αᴺ` and the shear number `αˢ ≥ 0`.
"""
@inline turbulent_richardson_number(sf::TRStabilityFunction, αᴺ, αˢ) = αᴺ / (sf.ℂᴿⁱ * αˢ + sf.ℂᵅ)

# v = Riᵗ / √(1 + Riᵗ²) ∈ [-1, 1]. |Riᵗ| is capped at 10¹⁵, where v = ±1 to rounding, so that Riᵗ² stays finite in Float32
@inline function bounded_richardson_coordinate(sf::TRStabilityFunction, αᴺ, αˢ)
    Riᵗ = turbulent_richardson_number(sf, αᴺ, αˢ)
    R = convert(typeof(Riᵗ), 1e15)
    Riᵗ = clamp(Riᵗ, -R, R)
    return Riᵗ / sqrt(1 + Riᵗ^2)
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
# time 1 / N of the center, so that α_N ≤ 1 there as at the faces.
@inline function stratification_and_shear_numbersᶜᶜᶜ(i, j, k, grid, closure, velocities, N², e, ℓ)
    S² = ℑbzᵃᵃᶜ(i, j, k, grid, effective_shear_squaredᶜᶜᶠ, closure, velocities, e, ℓ) + closure.shear.minimum_shear_squared
    N²ᵢ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
    τ²ʷ = (ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, ℓ) / turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e))^2
    stratified = N²ᵢ > 0
    τ² = ifelse(stratified, min(τ²ʷ, 1 / ifelse(stratified, N²ᵢ, one(N²ᵢ))), τ²ʷ)
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
