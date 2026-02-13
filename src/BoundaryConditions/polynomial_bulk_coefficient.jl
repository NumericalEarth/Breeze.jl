#####
##### PolynomialCoefficient: Wind and stability-dependent transfer coefficients
#####

# Default neutral polynomials (a₀, a₁, a₂) from Large & Yeager (2009),
# "The global climatology of an interannually varying air–sea flux data set",
# Climate Dynamics 33(2), 341–364.
const default_neutral_drag_polynomial            = (0.142, 0.076, 2.7)
const default_neutral_sensible_heat_polynomial   = (0.128, 0.068, 2.43)
const default_neutral_latent_heat_polynomial     = (0.120, 0.070, 2.55)

"""
    DefaultStabilityFunction()

Stability correction factor based on the bulk Richardson number ``Ri_b``.

For unstable conditions (``Ri_b < 0``):
```math
ψ = √(1 - 16 \\, Ri_b)
```

For stable conditions (``Ri_b ≥ 0``):
```math
ψ = 1 / (1 + 10 \\, Ri_b)
```

Multiplies the neutral transfer coefficient so that unstable conditions
enhance transfer (``ψ > 1``) and stable conditions reduce it (``ψ < 1``).
"""
struct DefaultStabilityFunction end

@inline function (::DefaultStabilityFunction)(Riᵦ)
    Ψ⁻ = sqrt(max(1 - 16 * Riᵦ, 0))  # unstable branch
    Ψ⁺ = 1 / (1 + 10 * max(Riᵦ, 0))  # stable branch
    return ifelse(Riᵦ < 0, Ψ⁻, Ψ⁺)
end

Base.summary(::DefaultStabilityFunction) = "ψ(Ri) = √(1-16Ri) unstable, 1/(1+10Ri) stable"

function Base.show(io::IO, ::DefaultStabilityFunction)
    println(io, "DefaultStabilityFunction")
    println(io, "├── Riᵦ < 0 (unstable): ψ = √(1 - 16 Riᵦ)")
    print(io,   "└── Riᵦ ≥ 0 (stable):   ψ = 1 / (1 + 10 Riᵦ)")
end

const default_stability_function = DefaultStabilityFunction()

"""
    PolynomialCoefficient(;
        polynomial = nothing,
        roughness_length = 1.5e-4,
        stability_function = DefaultStabilityFunction()
    )

A bulk transfer coefficient that depends on wind speed and atmospheric stability,
following [Large and Yeager (2009)](@cite LargeYeager2009).

The neutral drag coefficient follows the Large & Yeager (2009) form:
```math
C_N(U_{10}) = (a_0 + a_1 U_{10} + a_2 / U_{10}) × 10^{-3}
```

For other measurement heights, the coefficient is adjusted using logarithmic profile theory.

When a `stability_function` is provided, the coefficient is modified using bulk Richardson number:
```math
Ri_b = \\frac{g}{θ_v} \\frac{h (θ_v - θ_{v0})}{U^2}
```

When `polynomial` is `nothing`, the appropriate Large & Yeager (2009) polynomial
will be automatically selected based on the boundary condition type:
- `BulkDrag`: `default_neutral_drag_polynomial` = `(0.142, 0.076, 2.7)` for momentum
- `BulkSensibleHeatFlux`: `default_neutral_sensible_heat_polynomial` = `(0.128, 0.068, 2.43)` for sensible heat
- `BulkVaporFlux`: `default_neutral_latent_heat_polynomial` = `(0.120, 0.070, 2.55)` for latent heat

# Keyword Arguments
- `polynomial`: Tuple `(a₀, a₁, a₂)` for the polynomial. If `nothing`, the polynomial
  is automatically selected by the boundary condition constructor.
- `roughness_length`: Surface roughness ℓ in meters (default: 1.5e-4, typical for ocean)
- `minimum_wind_speed`: Minimum wind speed to avoid singularity in a₂/U term (default: 0.1 m/s)
- `stability_function`: Callable `ψ(Riᵦ)` that computes stability correction factor from bulk Richardson number.
  Set to `nothing` to disable stability correction. Default is `DefaultStabilityFunction()`.

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
└── stability_function: ψ(Ri) = √(1-16Ri) unstable, 1/(1+10Ri) stable
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
└── stability_function: ψ(Ri) = √(1-16Ri) unstable, 1/(1+10Ri) stable
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
└── stability_function: Nothing
```

# References

* Large, W., & Yeager, S. G. (2009). The global climatology of an interannually varying air–sea flux data set. Climate dynamics, 33(2), 341-364.
"""
struct PolynomialCoefficient{FT, C, SF, θᵛ, P, TC}
    polynomial :: C
    roughness_length :: FT
    minimum_wind_speed :: FT
    stability_function :: SF
    virtual_potential_temperature :: θᵛ
    surface_pressure :: P
    thermodynamic_constants :: TC
end

# Constructor with sensible defaults
function PolynomialCoefficient(FT = Float64;
                               polynomial = nothing,
                               roughness_length = 1.5e-4,
                               minimum_wind_speed = 0.1,
                               stability_function = DefaultStabilityFunction())

    return PolynomialCoefficient(polynomial,
                                 FT(roughness_length),
                                 FT(minimum_wind_speed),
                                 stability_function,
                                 nothing, nothing, nothing)
end

Adapt.adapt_structure(to, coef::PolynomialCoefficient) =
    PolynomialCoefficient(Adapt.adapt(to, coef.polynomial),
                          Adapt.adapt(to, coef.roughness_length),
                          Adapt.adapt(to, coef.minimum_wind_speed),
                          coef.stability_function,
                          Adapt.adapt(to, coef.virtual_potential_temperature),
                          Adapt.adapt(to, coef.surface_pressure),
                          Adapt.adapt(to, coef.thermodynamic_constants))

function Base.show(io::IO, coef::PolynomialCoefficient{FT}) where FT
    println(io, "PolynomialCoefficient{$FT}")
    println(io, "├── polynomial: ", coef.polynomial)
    println(io, "├── roughness_length: ", coef.roughness_length, " m")
    println(io, "├── minimum_wind_speed: ", coef.minimum_wind_speed, " m/s")
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

Compute neutral transfer coefficient at 10m height using Large & Yeager (2009) form:
C_N(U₁₀) = (a₀ + a₁*U₁₀ + a₂/U₁₀) × 10⁻³

Wind speed is clamped to `U_min` to avoid singularity in the a₂/U₁₀ term.
"""
@inline function neutral_coefficient_10m(polynomial, U₁₀, U_min)
    a₀, a₁, a₂ = polynomial
    FT = typeof(U₁₀)
    # Avoid division by zero
    U_safe = max(U₁₀, U_min)
    return (a₀ + a₁ * U_safe + a₂ / U_safe) * FT(1e-3)
end

#####
##### Bulk Richardson number and stability correction
#####

"""
$(TYPEDSIGNATURES)

Compute bulk Richardson number:
Ri_b = (g/θᵥ) × h × (θᵥ - θᵥ₀) / U²

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
    Cʰ = C¹⁰ * (log(10 / ℓ) / log(h / ℓ))^2

    # Apply stability correction
    return stability_corrected_coefficient(i, j, grid, coef, Cʰ, U, T₀)
end

# No stability correction (stability_function = nothing)
@inline stability_corrected_coefficient(i, j, grid, ::PolynomialCoefficient{<:Any, <:Any, Nothing}, Cʰ, U, T₀) = Cʰ

# Stability correction with a function — uses stored VPT and surface pressure
@inline function stability_corrected_coefficient(i, j, grid, coef::PolynomialCoefficient, Cʰ, U, T₀)
    h = znode(i, j, 1, grid, Center(), Center(), Center())
    θᵥ = @inbounds coef.virtual_potential_temperature[i, j, 1]
    surface = PlanarLiquidSurface()
    θᵥ₀ = surface_virtual_potential_temperature(T₀, coef.surface_pressure, coef.thermodynamic_constants, surface)
    Riᵦ = bulk_richardson_number(h, θᵥ, θᵥ₀, U, coef.minimum_wind_speed)
    return Cʰ * coef.stability_function(Riᵦ)
end

#####
##### Bulk coefficient evaluation
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

# Helper: fill in a default polynomial for a PolynomialCoefficient that has `nothing`
fill_polynomial(coef::PolynomialCoefficient, polynomial) =
    PolynomialCoefficient(polynomial,
                          coef.roughness_length,
                          coef.minimum_wind_speed,
                          coef.stability_function,
                          nothing, nothing, nothing)

# Type alias for PolynomialCoefficient with no polynomial set
const NothingPolynomialCoefficient = PolynomialCoefficient{<:Any, Nothing}
