#####
##### PolynomialCoefficient: Wind and stability-dependent transfer coefficients
#####

# Default neutral polynomials (a₀, a₁, a₂) from Large & Yeager (2009),
# "The global climatology of an interannually varying air–sea flux data set",
# Climate Dynamics 33(2), 341–364.
const default_neutral_drag_polynomial            = (0.142, 0.076, 2.7)
const default_neutral_sensible_heat_polynomial   = (0.128, 0.068, 2.43)
const default_neutral_latent_heat_polynomial     = (0.120, 0.070, 2.55)

#####
##### StabilityFunctionParameters: Ψ function constants
#####

"""
    StabilityFunctionParameters(FT = Oceananigans.defaults.FloatType;
        γᴰ = 19.3, γᵀ = 11.6, a = 1, b = 2/3, c = 5, d = 0.35)

Parameters for the integrated Monin-Obukhov stability functions ``Ψ^D(ζ)``
and ``Ψ^T(ζ)``.

Note: we use superscript D (drag/momentum) and T (temperature/scalar) to match the
transfer coefficient notation ``Cᴰ``, ``Cᵀ`` established in `notation.md`.
In the literature these are commonly written ``Ψ_m`` and ``Ψ_h``.

For unstable conditions (``ζ < 0``), uses Hogström (1996):
- ``φ^D = (1 - γ^D ζ)^{-1/4}``
- ``φ^T = 0.95(1 - γ^T ζ)^{-1/2}``

For stable conditions (``ζ ≥ 0``), uses Beljaars & Holtslag (1991):
- ``Ψ^D = -(a ζ + b (ζ - c/d) e^{-dζ} + bc/d)``
- ``Ψ^T = -((1 + 2aζ/3)^{3/2} + b (ζ - c/d) e^{-dζ} + bc/d - 1)``

# References

* Hogström, U. (1996). Review of some basic characteristics of the atmospheric
  surface layer. Boundary-Layer Meteorology, 78, 215-246.
* Beljaars, A. C. M., & Holtslag, A. A. M. (1991). Flux parameterization over
  land surfaces for atmospheric models. Journal of Applied Meteorology, 30, 327-341.
"""
struct StabilityFunctionParameters{FT}
    γᴰ :: FT
    γᵀ :: FT
    a :: FT
    b :: FT
    c :: FT
    d :: FT
end

function StabilityFunctionParameters(FT = Oceananigans.defaults.FloatType;
                                     γᴰ = 19.3,
                                     γᵀ = 11.6,
                                     a = 1,
                                     b = 2/3,
                                     c = 5,
                                     d = 0.35)
    return StabilityFunctionParameters{FT}(FT(γᴰ), FT(γᵀ), FT(a), FT(b), FT(c), FT(d))
end

#####
##### RichardsonNumberMapping: Li et al. (2010) regression coefficients
#####

"""
    RichardsonNumberMapping(FT = Oceananigans.defaults.FloatType; kwargs...)

Regression coefficients for the non-iterative mapping from bulk Richardson number
``Riᴮ`` to the Monin-Obukhov stability parameter ``ζ = z/L``, following
Li et al. (2010).

The superscripts u, w, s denote unstable, weakly stable, and strongly stable
regimes respectively. Subscript indices follow the original paper.

Three regimes:
- **Unstable** (``Riᴮ <`` `stable_unstable_transition`): Eq. 12
- **Weakly stable** (`stable_unstable_transition` ``≤ Riᴮ ≤`` `strongly_stable_transition`): Eq. 14
- **Strongly stable** (``Riᴮ >`` `strongly_stable_transition`): Eq. 16

# References

* Li, Y., Gao, Z., Lenschow, D. H., & Chen, F. (2010). An improved approach for
  parameterizing surface-layer turbulent transfer coefficients in numerical models.
  Boundary-Layer Meteorology, 137, 153-165.
"""
struct RichardsonNumberMapping{FT}
    # Regime thresholds
    stable_unstable_transition :: FT
    strongly_stable_transition :: FT

    # Unstable regime (Eq. 12)
    aᵘ₁₁ :: FT
    bᵘ₁₁ :: FT
    bᵘ₁₂ :: FT
    aᵘ₂₁ :: FT
    aᵘ₂₂ :: FT
    bᵘ₃₁ :: FT
    bᵘ₃₂ :: FT
    bᵘ₃₃ :: FT

    # Weakly stable regime (Eq. 14)
    aʷ₁₁ :: FT
    aʷ₁₂ :: FT
    aʷ₂₁ :: FT
    aʷ₂₂ :: FT
    bʷ₁₁ :: FT
    bʷ₁₂ :: FT
    bʷ₂₁ :: FT
    bʷ₂₂ :: FT

    # Strongly stable regime (Eq. 16)
    aˢ₁₁ :: FT
    aˢ₂₁ :: FT
    bˢ₁₁ :: FT
    bˢ₂₁ :: FT
    bˢ₂₂ :: FT
end

function RichardsonNumberMapping(FT = Oceananigans.defaults.FloatType;
                                 stable_unstable_transition = 0,
                                 strongly_stable_transition = 0.2,
                                 aᵘ₁₁ =  0.0450, bᵘ₁₁ =  0.0030, bᵘ₁₂ =  0.0059,
                                 aᵘ₂₁ = -0.0828, aᵘ₂₂ =  0.8845,
                                 bᵘ₃₁ =  0.1739, bᵘ₃₂ = -0.9213, bᵘ₃₃ = -0.1057,
                                 aʷ₁₁ =  0.5738, aʷ₁₂ = -0.4399,
                                 aʷ₂₁ = -4.901,  aʷ₂₂ = 52.50,
                                 bʷ₁₁ = -0.0539, bʷ₁₂ =  1.540,
                                 bʷ₂₁ = -0.6690, bʷ₂₂ = -3.282,
                                 aˢ₁₁ =  0.7529, aˢ₂₁ = 14.94,
                                 bˢ₁₁ =  0.1569, bˢ₂₁ = -0.3091, bˢ₂₂ = -1.303)
    return RichardsonNumberMapping{FT}(
        FT(stable_unstable_transition), FT(strongly_stable_transition),
        FT(aᵘ₁₁), FT(bᵘ₁₁), FT(bᵘ₁₂), FT(aᵘ₂₁), FT(aᵘ₂₂), FT(bᵘ₃₁), FT(bᵘ₃₂), FT(bᵘ₃₃),
        FT(aʷ₁₁), FT(aʷ₁₂), FT(aʷ₂₁), FT(aʷ₂₂), FT(bʷ₁₁), FT(bʷ₁₂), FT(bʷ₂₁), FT(bʷ₂₂),
        FT(aˢ₁₁), FT(aˢ₂₁), FT(bˢ₁₁), FT(bˢ₂₁), FT(bˢ₂₂))
end

#####
##### FittedStabilityFunction
#####

"""
    FittedStabilityFunction(scalar_roughness_length;
        richardson_number_mapping = RichardsonNumberMapping(...),
        stability_function_parameters = StabilityFunctionParameters(...))

Stability correction based on Monin-Obukhov similarity theory using the
Li et al. (2010) analytical mapping from bulk Richardson number to the
stability parameter ``ζ = z/L``.

Uses Hogström (1996) integrated stability functions for unstable conditions and
Beljaars & Holtslag (1991) for stable conditions.

Applies structurally correct (and different) corrections for momentum vs scalar transfer:
- Momentum: ``Cᴰ = Cᴰ_N [α / (α - Ψᴰ)]²``
- Scalar:   ``Cᵀ = Cᵀ_N [α / (α - Ψᴰ)] [β_h / (β_h - Ψᵀ)]``

where ``α = \\ln(z/ℓ)``, ``β_h = \\ln(z/ℓ_h)``.

`FittedStabilityFunction` is callable: `sf(Riᴮ, α, β)` returns the momentum
stability correction factor, and `sf(Riᴮ, α, β, Val(:scalar))` returns the
scalar correction factor.

# Arguments
- `scalar_roughness_length`: Roughness length for heat/moisture ``ℓ_h`` (m).

# Keyword Arguments
- `richardson_number_mapping`: [`RichardsonNumberMapping`](@ref) coefficients (default: Li et al. 2010).
- `stability_function_parameters`: [`StabilityFunctionParameters`](@ref) (default: Hogström 1996 / Beljaars & Holtslag 1991).

# References

* Li, Y., Gao, Z., Lenschow, D. H., & Chen, F. (2010). An improved approach for
  parameterizing surface-layer turbulent transfer coefficients in numerical models.
  Boundary-Layer Meteorology, 137, 153-165.
* Hogström, U. (1996). Review of some basic characteristics of the atmospheric surface layer.
  Boundary-Layer Meteorology, 78, 215-246.
* Beljaars, A. C. M., & Holtslag, A. A. M. (1991). Flux parameterization over land surfaces
  for atmospheric models. Journal of Applied Meteorology, 30, 327-341.
"""
struct FittedStabilityFunction{FT, RM, SP}
    scalar_roughness_length :: FT
    richardson_number_mapping :: RM
    stability_function_parameters :: SP
end

function FittedStabilityFunction(scalar_roughness_length;
    richardson_number_mapping = RichardsonNumberMapping(typeof(scalar_roughness_length)),
    stability_function_parameters = StabilityFunctionParameters(typeof(scalar_roughness_length)))
    return FittedStabilityFunction(scalar_roughness_length,
                                   richardson_number_mapping,
                                   stability_function_parameters)
end

Base.summary(::FittedStabilityFunction) = "FittedStabilityFunction (Li et al. 2010)"

function Base.show(io::IO, sf::FittedStabilityFunction)
    println(io, "FittedStabilityFunction (Li et al. 2010)")
    println(io, "├── scalar_roughness_length: ", sf.scalar_roughness_length, " m")
    println(io, "├── Riᴮ → ζ mapping: Li et al. (2010)")
    println(io, "├── Unstable Ψᴰ, Ψᵀ: Hogström (1996)")
    print(io,   "└── Stable Ψᴰ, Ψᵀ: Beljaars & Holtslag (1991)")
end

# Callable interface: compute stability correction factor
@inline function (sf::FittedStabilityFunction)(Riᴮ, α, β, transfer_type=Val(:momentum))
    ζ = bulk_to_flux_richardson_number(Riᴮ, α, β, sf.richardson_number_mapping)
    Ψᴰ = integrated_stability_momentum(ζ, sf.stability_function_parameters)
    Ψᵀ = integrated_stability_scalar(ζ, sf.stability_function_parameters)
    return stability_correction_factor(α, β, Ψᴰ, Ψᵀ, transfer_type)
end

#####
##### Li et al. (2010) bulk Richardson number to flux Richardson number mapping
#####

"""
$(TYPEDSIGNATURES)

Map bulk Richardson number ``Riᴮ`` to the Monin-Obukhov stability parameter
``ζ = z/L`` using the regression equations of Li et al. (2010).

# Arguments
- `Riᴮ`: Bulk Richardson number
- `α`: ``\\ln(z / ℓ)``
- `β`: ``\\ln(ℓ / ℓ_h)``
- `mapping`: [`RichardsonNumberMapping`](@ref) with regression coefficients
"""
@inline function bulk_to_flux_richardson_number(Riᴮ, α, β, mapping)
    ζ⁻ = unstable_bulk_to_flux_richardson_number(Riᴮ, α, β, mapping)
    ζʷ = weakly_stable_bulk_to_flux_richardson_number(Riᴮ, α, β, mapping)
    ζˢ = strongly_stable_bulk_to_flux_richardson_number(Riᴮ, α, β, mapping)
    return ifelse(Riᴮ < mapping.stable_unstable_transition,
                  ζ⁻,
                  ifelse(Riᴮ ≤ mapping.strongly_stable_transition, ζʷ, ζˢ))
end

# Unstable regime (Riᴮ < 0), Li et al. (2010) Eq. 12
@inline function unstable_bulk_to_flux_richardson_number(Riᴮ, α, β, mapping)
    A = mapping.aᵘ₁₁ * α
    B = (mapping.bᵘ₁₁ * β + mapping.bᵘ₁₂) * α^2 +
        (mapping.aᵘ₂₁ * β + mapping.aᵘ₂₂) * α +
        (mapping.bᵘ₃₁ * β^2 + mapping.bᵘ₃₂ * β + mapping.bᵘ₃₃)
    return A * Riᴮ^2 + B * Riᴮ
end

# Weakly stable regime (0 ≤ Riᴮ ≤ Ri*), Li et al. (2010) Eq. 14
@inline function weakly_stable_bulk_to_flux_richardson_number(Riᴮ, α, β, mapping)
    A = (mapping.aʷ₁₁ * β + mapping.aʷ₁₂) * α + (mapping.aʷ₂₁ * β + mapping.aʷ₂₂)
    B = (mapping.bʷ₁₁ * β + mapping.bʷ₁₂) * α + (mapping.bʷ₂₁ * β + mapping.bʷ₂₂)
    return A * Riᴮ^2 + B * Riᴮ
end

# Strongly stable regime (Riᴮ > Ri*), Li et al. (2010) Eq. 16
@inline function strongly_stable_bulk_to_flux_richardson_number(Riᴮ, α, β, mapping)
    return (mapping.aˢ₁₁ * α + mapping.aˢ₂₁) * Riᴮ +
            mapping.bˢ₁₁ * α + mapping.bˢ₂₁ * β + mapping.bˢ₂₂
end

#####
##### Integrated MOST stability functions
#####

"""
$(TYPEDSIGNATURES)

Integrated stability function for momentum ``Ψᴰ(ζ)``.

Note: ``Ψᴰ`` corresponds to ``Ψ_m`` in the literature.
"""
@inline function integrated_stability_momentum(ζ, params)
    FT = typeof(ζ)
    Ψ⁻ = unstable_Ψᴰ(ζ, params)
    Ψ⁺ = stable_Ψᴰ(ζ, params)
    return ifelse(ζ < zero(FT), Ψ⁻, Ψ⁺)
end

"""
$(TYPEDSIGNATURES)

Integrated stability function for scalars (heat, moisture) ``Ψᵀ(ζ)``.

Note: ``Ψᵀ`` corresponds to ``Ψ_h`` in the literature.
"""
@inline function integrated_stability_scalar(ζ, params)
    FT = typeof(ζ)
    Ψ⁻ = unstable_Ψᵀ(ζ, params)
    Ψ⁺ = stable_Ψᵀ(ζ, params)
    return ifelse(ζ < zero(FT), Ψ⁻, Ψ⁺)
end

# Hogström (1996) unstable Ψᴰ: φᴰ = (1 - γᴰ ζ)^{-1/4}
@inline function unstable_Ψᴰ(ζ, params)
    FT = typeof(ζ)
    x = sqrt(sqrt(max(1 - params.γᴰ * ζ, zero(FT))))
    return 2 * log((1 + x) / 2) + log((1 + x^2) / 2) - 2 * atan(x) + FT(π) / 2
end

# Hogström (1996) unstable Ψᵀ: φᵀ = 0.95(1 - γᵀ ζ)^{-1/2}
@inline function unstable_Ψᵀ(ζ, params)
    FT = typeof(ζ)
    y = sqrt(max(1 - params.γᵀ * ζ, zero(FT)))
    return 2 * log((1 + y) / 2)
end

# Beljaars & Holtslag (1991) stable Ψᴰ
@inline function stable_Ψᴰ(ζ, params)
    (; a, b, c, d) = params
    return -(a * ζ + b * (ζ - c / d) * exp(-d * ζ) + b * c / d)
end

# Beljaars & Holtslag (1991) stable Ψᵀ
@inline function stable_Ψᵀ(ζ, params)
    FT = typeof(ζ)
    (; a, b, c, d) = params
    x = max(1 + 2 * a / 3 * ζ, zero(FT))
    return -(x * sqrt(x) + b * (ζ - c / d) * exp(-d * ζ) + b * c / d - 1)
end

#####
##### Stability correction factors for momentum and scalar transfer
#####

@inline function stability_correction_factor(α, β, Ψᴰ, Ψᵀ, ::Val{:momentum})
    denominator_D = max(α - Ψᴰ, α / 10)
    return (α / denominator_D)^2
end

@inline function stability_correction_factor(α, β, Ψᴰ, Ψᵀ, ::Val{:scalar})
    βh = α + β
    denominator_D = max(α - Ψᴰ, α / 10)
    denominator_T = max(βh - Ψᵀ, βh / 10)
    return (α / denominator_D) * (βh / denominator_T)
end

# Default to momentum when transfer_type is not set
@inline stability_correction_factor(α, β, Ψᴰ, Ψᵀ, ::Nothing) =
    stability_correction_factor(α, β, Ψᴰ, Ψᵀ, Val(:momentum))

#####
##### PolynomialCoefficient struct
#####

"""
    PolynomialCoefficient(;
        polynomial = nothing,
        roughness_length = 1.5e-4,
        stability_function = FittedStabilityFunction(roughness_length / 7.3),
        surface = PlanarLiquidSurface()
    )

A bulk transfer coefficient that depends on wind speed and atmospheric stability,
following [Large and Yeager (2009)](@cite LargeYeager2009).

The neutral transfer coefficient at 10 m follows the Large & Yeager (2009) form:
```math
C^N_{10}(U_h) = (a_0 + a_1 U_h + a_2 / U_h) × 10^{-3}
```
where ``U_h`` is the wind speed at measurement height ``h``.

The coefficient is adjusted for measurement height using logarithmic profile theory,
and stability correction is applied based on the bulk Richardson number.

When `polynomial` is `nothing`, the appropriate Large & Yeager (2009) polynomial
will be automatically selected based on the boundary condition type:
- `BulkDrag`: `default_neutral_drag_polynomial` = `(0.142, 0.076, 2.7)` for momentum
- `BulkSensibleHeatFlux`: `default_neutral_sensible_heat_polynomial` = `(0.128, 0.068, 2.43)` for sensible heat
- `BulkVaporFlux`: `default_neutral_latent_heat_polynomial` = `(0.120, 0.070, 2.55)` for latent heat

# Keyword Arguments
- `polynomial`: Tuple `(a₀, a₁, a₂)` for the polynomial. If `nothing`, the polynomial
  is automatically selected by the boundary condition constructor.
- `roughness_length`: Surface roughness `ℓ` in meters (default: 1.5e-4, typical for ocean)
- `minimum_wind_speed`: Minimum wind speed to avoid singularity in a₂/U term (default: 0.1 m/s)
- `stability_function`: Stability correction strategy.
  Default is [`FittedStabilityFunction`](@ref) using Li et al. (2010) ``Riᴮ → ζ`` mapping
  with Hogström (1996) / Beljaars & Holtslag (1991) MOST stability functions.
  The scalar roughness length defaults to `roughness_length / 7.3` (typical ocean value).
  Use `nothing` to disable stability correction.
- `surface`: Surface type for computing saturation specific humidity in the stability correction.
  Default is `PlanarLiquidSurface()`. Use `PlanarIceSurface()` for ice surfaces.

The measurement height is automatically determined from the grid as the height of the first
cell center above the surface.

# Examples

```jldoctest
using Breeze.BoundaryConditions: PolynomialCoefficient

# Polynomial coefficient with default settings
coef = PolynomialCoefficient()

# output
PolynomialCoefficient{Float64}
├── polynomial: nothing
├── roughness_length: 0.00015 m
├── minimum_wind_speed: 0.1 m/s
├── surface: PlanarLiquidSurface
└── stability_function: FittedStabilityFunction (Li et al. 2010)
```

```jldoctest
using Breeze.BoundaryConditions: PolynomialCoefficient

# With explicit polynomial
coef = PolynomialCoefficient(polynomial = (0.142, 0.076, 2.7))

# output
PolynomialCoefficient{Float64}
├── polynomial: (0.142, 0.076, 2.7)
├── roughness_length: 0.00015 m
├── minimum_wind_speed: 0.1 m/s
├── surface: PlanarLiquidSurface
└── stability_function: FittedStabilityFunction (Li et al. 2010)
```

```jldoctest
using Breeze.BoundaryConditions: PolynomialCoefficient

# No stability correction
coef = PolynomialCoefficient(stability_function = nothing)

# output
PolynomialCoefficient{Float64}
├── polynomial: nothing
├── roughness_length: 0.00015 m
├── minimum_wind_speed: 0.1 m/s
├── surface: PlanarLiquidSurface
└── stability_function: Nothing
```

# References

* Large, W., & Yeager, S. G. (2009). The global climatology of an interannually varying air–sea flux data set. Climate dynamics, 33(2), 341-364.
* Li, Y., Gao, Z., Lenschow, D. H., & Chen, F. (2010). An improved approach for parameterizing surface-layer turbulent transfer coefficients in numerical models. Boundary-Layer Meteorology, 137, 153-165.
"""
struct PolynomialCoefficient{FT, C, SF, S, θᵛ, P, TC, TT}
    polynomial :: C
    roughness_length :: FT
    minimum_wind_speed :: FT
    stability_function :: SF
    surface :: S
    virtual_potential_temperature :: θᵛ
    surface_pressure :: P
    thermodynamic_constants :: TC
    transfer_type :: TT
end

# Constructor with sensible defaults
function PolynomialCoefficient(FT = Oceananigans.defaults.FloatType;
                               polynomial = nothing,
                               roughness_length = 1.5e-4,
                               minimum_wind_speed = 0.1,
                               stability_function = FittedStabilityFunction(FT(roughness_length / 7.3)),
                               surface = PlanarLiquidSurface(),
                               transfer_type = nothing)

    return PolynomialCoefficient(polynomial,
                                 FT(roughness_length),
                                 FT(minimum_wind_speed),
                                 stability_function,
                                 surface,
                                 nothing, nothing, nothing,
                                 transfer_type)
end

Adapt.adapt_structure(to, coef::PolynomialCoefficient) =
    PolynomialCoefficient(Adapt.adapt(to, coef.polynomial),
                          Adapt.adapt(to, coef.roughness_length),
                          Adapt.adapt(to, coef.minimum_wind_speed),
                          coef.stability_function,
                          coef.surface,
                          Adapt.adapt(to, coef.virtual_potential_temperature),
                          Adapt.adapt(to, coef.surface_pressure),
                          Adapt.adapt(to, coef.thermodynamic_constants),
                          coef.transfer_type)

function Base.show(io::IO, coef::PolynomialCoefficient{FT}) where FT
    println(io, "PolynomialCoefficient{$FT}")
    println(io, "├── polynomial: ", coef.polynomial)
    println(io, "├── roughness_length: ", coef.roughness_length, " m")
    println(io, "├── minimum_wind_speed: ", coef.minimum_wind_speed, " m/s")
    println(io, "├── surface: ", summary(coef.surface))
    print(io,   "└── stability_function: ", summary(coef.stability_function))
end

Base.summary(coef::PolynomialCoefficient) =
    string("PolynomialCoefficient(", coef.polynomial, ")")
Base.summary(::Nothing) = "Nothing"

#####
##### Neutral coefficient computation (Large & Yeager 2009 form)
#####

"""
$(TYPEDSIGNATURES)

Compute neutral transfer coefficient at 10 m using the Large & Yeager (2009) form:
C¹⁰_N(U) = (a₀ + a₁ U + a₂ / U) × 10⁻³

Wind speed is clamped to `U_min` to avoid singularity in the a₂/U term.
"""
@inline function neutral_coefficient_10m(polynomial, U₁₀, U_min)
    a₀, a₁, a₂ = polynomial
    FT = typeof(U₁₀)
    # Avoid division by zero
    U_safe = max(U₁₀, U_min)
    return (a₀ + a₁ * U_safe + a₂ / U_safe) * FT(1e-3)
end

#####
##### Bulk Richardson number
#####

"""
$(TYPEDSIGNATURES)

Compute bulk Richardson number:
Riᴮ = (g/θ̄ᵥ) × h × (θᵥ - θᵥ₀) / U²

Wind speed is clamped to `U_min` to avoid singularity.

# Arguments
- `h`: Measurement height (m)
- `θᵥ`: Virtual potential temperature at measurement height (K)
- `θᵥ₀`: Virtual potential temperature at surface (K)
- `U`: Wind speed (m/s)
- `U_min`: Minimum wind speed (m/s)
- `g`: Gravitational acceleration (m/s², default: 9.81)
"""
@inline function bulk_richardson_number(h, θᵥ, θᵥ₀, U, U_min, g = 9.81)
    # Avoid division by zero
    U_safe = max(U, U_min)
    θᵥ_mean = (θᵥ + θᵥ₀) / 2
    return (g / θᵥ_mean) * h * (θᵥ - θᵥ₀) / U_safe^2
end

#####
##### Helper functions for surface thermodynamic quantities
#####

"""
$(TYPEDSIGNATURES)

Compute virtual potential temperature over a planar `surface`
with surface temperature `T₀` and surface pressure `p₀`,

```math
θᵥ₀ = T₀ (1 + δᵛᵈ qᵛ⁺)
```

where ``qᵛ⁺`` is the saturation specific humidity at the surface
and ``δᵛᵈ = Rᵛ/Rᵈ - 1`` (≈ 0.608 for water vapor in Earth's atmosphere;
the actual value depends on the gas constants in `constants`).
"""
@inline function surface_virtual_potential_temperature(T₀, p₀, constants, surface)
    qᵛ⁺ = saturation_total_specific_moisture(T₀, p₀, constants, surface)

    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    δᵛᵈ = Rᵛ / Rᵈ - 1

    return T₀ * (1 + δᵛᵈ * qᵛ⁺)
end

#####
##### Main callable interface
#####

"""
$(TYPEDSIGNATURES)

Evaluate the bulk transfer coefficient for given conditions.

For a materialized `PolynomialCoefficient` (with `virtual_potential_temperature`,
`surface_pressure`, and `thermodynamic_constants` filled in during model construction),
the stability correction is computed internally from the stored fields.

# Arguments
- `i`, `j`: Grid indices
- `grid`: The grid
- `U`: Wind speed (m/s)
- `T₀`: Surface temperature (K) at location `(i, j)`

Returns the transfer coefficient (dimensionless).
"""
@inline function (coef::PolynomialCoefficient)(i, j, grid, U, T₀)
    # Compute neutral coefficient at 10m
    C¹⁰ = neutral_coefficient_10m(coef.polynomial, U, coef.minimum_wind_speed)

    # Adjust for measurement height using logarithmic profile:
    # C(h) = C₁₀ × [ln(10/ℓ) / ln(h/ℓ)]²
    h = znode(i, j, 1, grid, Center(), Center(), Center())
    ℓ = coef.roughness_length
    α = log(h / ℓ)
    Cʰ = C¹⁰ * (log(10 / ℓ) / α)^2

    # Apply stability correction
    return stability_corrected_coefficient(i, j, grid, coef, Cʰ, h, α, U, T₀)
end

# No stability correction (stability_function = nothing)
@inline stability_corrected_coefficient(i, j, grid,
    ::PolynomialCoefficient{<:Any, <:Any, Nothing}, Cʰ, h, α, U, T₀) = Cʰ

# FittedStabilityFunction correction (Li et al. 2010 mapping + MOST Ψ functions)
@inline function stability_corrected_coefficient(i, j, grid,
    coef::PolynomialCoefficient{<:Any, <:Any, <:FittedStabilityFunction}, Cʰ, h, α, U, T₀)

    sf = coef.stability_function
    ℓ = coef.roughness_length
    ℓh = sf.scalar_roughness_length
    β = log(ℓ / ℓh)

    θᵥ = @inbounds coef.virtual_potential_temperature[i, j, 1]
    θᵥ₀ = surface_virtual_potential_temperature(T₀, coef.surface_pressure, coef.thermodynamic_constants, coef.surface)
    Riᴮ = bulk_richardson_number(h, θᵥ, θᵥ₀, U, coef.minimum_wind_speed)

    return Cʰ * sf(Riᴮ, α, β, coef.transfer_type)
end

#####
##### Bulk coefficient evaluation
#####
#####
##### Unified interface for evaluating bulk transfer coefficients. Dispatches
##### on the coefficient type: constant Number returns directly, callable
##### PolynomialCoefficient computes wind speed and evaluates with stability correction.
#####

@inline bulk_coefficient(i, j, grid, C::Number, fields, T₀) = C

@inline function bulk_coefficient(i, j, grid, C::PolynomialCoefficient, fields, T₀)
    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    U = sqrt(U²)
    return C(i, j, grid, U, T₀)
end

#####
##### Default polynomial filling
#####

# Helper: fill in a default polynomial and transfer type for a PolynomialCoefficient
fill_polynomial(coef::PolynomialCoefficient, polynomial, transfer_type) =
    PolynomialCoefficient(polynomial,
                          coef.roughness_length,
                          coef.minimum_wind_speed,
                          coef.stability_function,
                          coef.surface,
                          nothing, nothing, nothing,
                          transfer_type)

# Type alias for PolynomialCoefficient with no polynomial set
const NothingPolynomialCoefficient = PolynomialCoefficient{<:Any, Nothing}
