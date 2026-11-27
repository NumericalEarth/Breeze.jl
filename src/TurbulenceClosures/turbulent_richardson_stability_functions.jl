const DEFAULT_STABILITY_COEFFICIENT_BOUNDS = (0.05, 20)

const RICHARDSON_BLEND_WIDTH = 1 // 4

const MAXIMUM_SLOPE_FACTOR = let v = (sqrt(73) - 3) / 16
    (1 / 2 + v) * (1 - v^2)^(3 / 2)
end

const MAXIMUM_BLEND_SLOPE = 1 / 2 + 2 * sqrt(6) / 9

minimum_stratification_scale(lower = DEFAULT_STABILITY_COEFFICIENT_BOUNDS[1],
                             upper = DEFAULT_STABILITY_COEFFICIENT_BOUNDS[2]) =
    MAXIMUM_SLOPE_FACTOR * MAXIMUM_BLEND_SLOPE * log(upper / lower)
"""
$(TYPEDEF)

Stability functions for [`TKEBasedTurbulenceClosure`](@ref) of the turbulent Richardson number
"""
struct TurbulentRichardsonStabilityFunction{FT}
    ℂᵘ⁻ :: FT
    ℂᵘ⁰ :: FT
    ℂᵘ⁺ :: FT
    ℂᶜ⁻ :: FT
    ℂᶜ⁰ :: FT
    ℂᶜ⁺ :: FT
    ℂᵉ⁻ :: FT
    ℂᵉ⁰ :: FT
    ℂᵉ⁺ :: FT
    ℂᴰ⁻ :: FT
    ℂᴰ⁰ :: FT
    ℂᴰ⁺ :: FT
    ℂᴿⁱ⁻ :: FT
    ℂᴿⁱ⁺ :: FT
    ℂᵅ :: FT
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

function stability_coefficient_bounds(lower = DEFAULT_STABILITY_COEFFICIENT_BOUNDS[1],
                                      upper = DEFAULT_STABILITY_COEFFICIENT_BOUNDS[2])
    0 < lower < upper || throw(ArgumentError("The bounds must satisfy 0 < lower < upper, but they are ($lower, $upper)"))
    ℂᵅᵐⁱⁿ = minimum_stratification_scale(lower, upper)
    ℂᵅᵐⁱⁿ < upper || throw(ArgumentError("The least ℂᵅ of the width condition, $ℂᵅᵐⁱⁿ, is above upper = $upper"))
    intervals = map(name -> name === :ℂᵅ ? (ℂᵅᵐⁱⁿ, upper) : (lower, upper), TURBULENT_RICHARDSON_COEFFICIENTS)
    return NamedTuple{TURBULENT_RICHARDSON_COEFFICIENTS}(intervals)
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

@inline function turbulent_richardson_number(sf::TRStabilityFunction, αᴺ, αˢ)
    D⁻ = sf.ℂᴿⁱ⁻ * αˢ + sf.ℂᵅ
    D⁺ = sf.ℂᴿⁱ⁺ * αˢ + sf.ℂᵅ
    Riᵗ⁻ = αᴺ / D⁻
    Riᵗ⁺ = αᴺ / D⁺
    δD = RICHARDSON_BLEND_WIDTH * min(D⁻, D⁺)
    s = αᴺ / hypotenuse(δD, αᴺ)
    return Riᵗ⁻ + (Riᵗ⁺ - Riᵗ⁻) * (1 + s) / 2
end

@inline function bounded_richardson_coordinate(sf::TRStabilityFunction, αᴺ, αˢ)
    Riᵗ = turbulent_richardson_number(sf, αᴺ, αˢ)
    return Riᵗ / hypotenuse(one(Riᵗ), Riᵗ)
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

@inline function stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
    τ² = (@inbounds ℓ[i, j, k] / w★)^2
    S² = effective_shear_squaredᶜᶜᶠ(i, j, k, grid, closure, velocities, e, ℓ) + closure.shear.minimum_shear_squared
    return τ² * @inbounds(N²[i, j, k]), τ² * S²
end

@inline function stratification_and_shear_numbersᶜᶜᶜ(i, j, k, grid, closure, velocities, N², e, ℓ)
    S² = ℑbzᵃᵃᶜ(i, j, k, grid, effective_shear_squaredᶜᶜᶠ, closure, velocities, e, ℓ) + closure.shear.minimum_shear_squared
    N²ᵢ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
    τʷ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, ℓ) / turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    ml = closure.mixing_length
    τᵇ⁻¹ = sqrt(max(N²ᵢ, zero(N²ᵢ))) / buoyancy_length_coefficient(ml)
    τ = inv(max(inv(τʷ), τᵇ⁻¹))
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
