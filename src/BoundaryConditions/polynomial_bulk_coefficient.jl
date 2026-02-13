#####
##### PolynomialCoefficient: Wind and stability-dependent transfer coefficients
#####

"""
$(TYPEDSIGNATURES)

Default stability function based on bulk Richardson number.

For unstable conditions (Riᵦ < 0):
    ψ = √(1 - 16 Riᵦ)

For stable conditions (Riᵦ ≥ 0):
    ψ = 1 / (1 + 10 Riᵦ)

Returns the factor by which to multiply the neutral coefficient.
Unstable conditions enhance transfer (ψ > 1), stable conditions reduce it (ψ < 1).
"""
@inline function default_stability_function(Riᵦ)
    # Unstable: sqrt(1 - 16*Ri_b), Stable: 1/(1 + 10*Ri_b)
    Ψ⁻ = sqrt(max(1 - 16 * Riᵦ, 0))  # unstable branch. max to avoid negative sqrt
    Ψ⁺ = 1 / (1 + 10 * max(Riᵦ, 0))  # stable branch. max to handle Ri_b < 0
    return ifelse(Riᵦ < 0, Ψ⁻, Ψ⁺)
end

"""
    PolynomialCoefficient(;
        neutral_coefficients = nothing,
        roughness_length = 1.5e-4,
        stability_function = default_stability_function
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
Ri_b = \\frac{g}{θ_v} \\frac{z (θ_v - θ_{v0})}{U^2}
```

When `neutral_coefficients` is `nothing`, the appropriate Large & Yeager (2009) coefficients
will be automatically selected based on the boundary condition type:
- `BulkDrag`: `(0.142, 0.076, 2.7)` for momentum
- `BulkSensibleHeatFlux`: `(0.128, 0.068, 2.43)` for sensible heat
- `BulkVaporFlux`: `(0.120, 0.070, 2.55)` for latent heat

# Keyword Arguments
- `neutral_coefficients`: Tuple `(a₀, a₁, a₂)` for the polynomial. If `nothing`, coefficients
  are automatically selected by the boundary condition constructor.
- `roughness_length`: Surface roughness ℓ in meters (default: 1.5e-4, typical for ocean)
- `minimum_wind_speed`: Minimum wind speed to avoid singularity in a₂/U term (default: 0.1 m/s)
- `stability_function`: Function `ψ(Riᵦ)` that computes stability correction factor from bulk Richardson number.
  Set to `nothing` to disable stability correction. Default is `default_stability_function`.

The measurement height is automatically determined from the grid as the height of the first
cell center above the surface.

# Examples

```jldoctest
using Breeze.BoundaryConditions: PolynomialCoefficient

# Polynomial coefficient with default settings
coef = PolynomialCoefficient()

# output
PolynomialCoefficient{Float64}
├── neutral_coefficients: nothing
└── roughness_length: 0.00015 m
```

```jldoctest
using Breeze.BoundaryConditions: PolynomialCoefficient

# With explicit coefficients
coef = PolynomialCoefficient(neutral_coefficients = (0.142, 0.076, 2.7))

# output
PolynomialCoefficient{Float64}
├── neutral_coefficients: (0.142, 0.076, 2.7)
└── roughness_length: 0.00015 m
```

```jldoctest
using Breeze.BoundaryConditions: PolynomialCoefficient

# No stability correction
coef = PolynomialCoefficient(stability_function = nothing)

# output
PolynomialCoefficient{Float64}
├── neutral_coefficients: nothing
└── roughness_length: 0.00015 m
```
"""
struct PolynomialCoefficient{FT, C, SF, θᵛ, P, TC}
    neutral_coefficients :: C
    roughness_length :: FT
    minimum_wind_speed :: FT
    stability_function :: SF
    virtual_potential_temperature :: θᵛ
    surface_pressure :: P
    thermodynamic_constants :: TC
end

# Constructor with sensible defaults
function PolynomialCoefficient(FT = Float64;
                               neutral_coefficients = nothing,
                               roughness_length = 1.5e-4,
                               minimum_wind_speed = 0.1,
                               stability_function = default_stability_function)

    return PolynomialCoefficient(neutral_coefficients,
                                 FT(roughness_length),
                                 FT(minimum_wind_speed),
                                 stability_function,
                                 nothing, nothing, nothing)
end

Adapt.adapt_structure(to, coef::PolynomialCoefficient) =
    PolynomialCoefficient(Adapt.adapt(to, coef.neutral_coefficients),
                          Adapt.adapt(to, coef.roughness_length),
                          Adapt.adapt(to, coef.minimum_wind_speed),
                          coef.stability_function,
                          Adapt.adapt(to, coef.virtual_potential_temperature),
                          Adapt.adapt(to, coef.surface_pressure),
                          Adapt.adapt(to, coef.thermodynamic_constants))

function Base.show(io::IO, coef::PolynomialCoefficient{FT}) where FT
    println(io, "PolynomialCoefficient{$FT}")
    println(io, "├── neutral_coefficients: ", coef.neutral_coefficients)
    print(io,   "└── roughness_length: ", coef.roughness_length, " m")
end

Base.summary(coef::PolynomialCoefficient) =
    string("PolynomialCoefficient(", coef.neutral_coefficients, ")")

#####
##### Neutral coefficient computation (Large & Yeager 2009 form)
#####

"""
$(TYPEDSIGNATURES)

Compute neutral transfer coefficient at 10m height using Large & Yeager (2009) form:
C_N(U₁₀) = (a₀ + a₁*U₁₀ + a₂/U₁₀) × 10⁻³

Wind speed is clamped to `U_min` to avoid singularity in the a₂/U₁₀ term.
"""
@inline function neutral_coefficient_10m(coefficients, U₁₀, U_min)
    a₀, a₁, a₂ = coefficients
    FT = typeof(U₁₀)
    # Avoid division by zero
    U_safe = max(U₁₀, U_min)
    return (a₀ + a₁ * U_safe + a₂ / U_safe) * FT(1e-3)
end

#####
##### Height adjustment using logarithmic profile
#####

"""
$(TYPEDSIGNATURES)

Adjust transfer coefficient from 10m reference height to measurement height `z`
using logarithmic profile theory:

C(z) = C₁₀ × [ln(10/z₀) / ln(z/z₀)]²

# Arguments
- `C₁₀`: Transfer coefficient at 10m
- `z`: Measurement height (m)
- `z₀`: Roughness length (m)
"""
@inline function adjust_coefficient_for_height(C₁₀, z, z₀)
    log_ref = log(10 / z₀)
    log_z = log(z / z₀)
    return C₁₀ * (log_ref / log_z)^2
end

#####
##### Bulk Richardson number and stability correction
#####

"""
$(TYPEDSIGNATURES)

Compute bulk Richardson number:
Ri_b = (g/θᵥ) × z × (θᵥ - θᵥ₀) / U²

Wind speed is clamped to `U_min` to avoid singularity.

# Arguments
- `z`: Measurement height (m)
- `θᵥ`: Virtual potential temperature at measurement height (K)
- `θᵥ₀`: Virtual potential temperature at surface (K)
- `U`: Wind speed (m/s)
- `U_min`: Minimum wind speed (m/s)
- `g`: Gravitational acceleration (m/s², default: 9.81)
"""
@inline function bulk_richardson_number(z, θᵥ, θᵥ₀, U, U_min, g = 9.81)
    # Avoid division by zero
    U_safe = max(U, U_min)
    θᵥ_mean = (θᵥ + θᵥ₀) / 2
    return (g / θᵥ_mean) * z * (θᵥ - θᵥ₀) / U_safe^2
end

#####
##### Helper functions for surface thermodynamic quantities
#####

"""
$(TYPEDSIGNATURES)

compute virtual potential temperature over a planar `surface`
with surface temperature `T₀` and surface pressure `p₀`,

```math
θᵥ₀ = T₀ (1 + δᵛᵈ qᵛ⁺)
```

where qᵛ⁺ is the saturation total specific moisture,
and ``δᵛᵈ = Rᵛ/Rᵈ - 1 ≈ 0.608``.
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
    C₁₀ = neutral_coefficient_10m(coef.neutral_coefficients, U, coef.minimum_wind_speed)

    # Adjust for measurement height
    z = znode(i, j, 1, grid, Center(), Center(), Center())
    Cz = adjust_coefficient_for_height(C₁₀, z, coef.roughness_length)

    # Apply stability correction
    return apply_stability_correction(coef, Cz, i, j, grid, U, T₀)
end

# Stability correction with a function — uses stored VPT and surface pressure
@inline function apply_stability_correction(coef::PolynomialCoefficient, Cz, i, j, grid, U, T₀)
    z = znode(i, j, 1, grid, Center(), Center(), Center())
    θᵥ = @inbounds coef.virtual_potential_temperature[i, j, 1]
    surface = PlanarLiquidSurface()
    θᵥ₀ = surface_virtual_potential_temperature(T₀, coef.surface_pressure, coef.thermodynamic_constants, surface)
    Riᵦ = bulk_richardson_number(z, θᵥ, θᵥ₀, U, coef.minimum_wind_speed)
    return Cz * coef.stability_function(Riᵦ)
end

# No stability correction (stability_function = nothing)
@inline apply_stability_correction(::PolynomialCoefficient{<:Any, <:Any, Nothing}, Cz, args...) = Cz

#####
##### Special constructors for boundary conditions
#####

# These constructors automatically set the appropriate Large & Yeager (2009) coefficients
# based on the flux type when neutral_coefficients is nothing

"""
$(TYPEDSIGNATURES)

Create a `BulkDrag` boundary condition with a `PolynomialCoefficient`.
If `coef.neutral_coefficients` is `nothing`, automatically uses Large & Yeager (2009)
momentum coefficients `(0.142, 0.076, 2.7)`.
"""
function BulkDrag(coef::PolynomialCoefficient; direction=nothing, gustiness=0, surface_temperature)
    # If neutral_coefficients is nothing, create a new coefficient with momentum coefficients
    if isnothing(coef.neutral_coefficients)
        coef = PolynomialCoefficient(
            neutral_coefficients = (0.142, 0.076, 2.7),  # Large & Yeager (2009) momentum
            roughness_length = coef.roughness_length,
            minimum_wind_speed = coef.minimum_wind_speed,
            stability_function = coef.stability_function
        )
    end
    df = BulkDragFunction(direction, coef, gustiness, surface_temperature)
    return BoundaryCondition(Flux(), df)
end

"""
$(TYPEDSIGNATURES)

Create a `BulkSensibleHeatFlux` boundary condition with a `PolynomialCoefficient`.
If `coef.neutral_coefficients` is `nothing`, automatically uses Large & Yeager (2009)
sensible heat coefficients `(0.128, 0.068, 2.43)`.
"""
function BulkSensibleHeatFlux(coef::PolynomialCoefficient; gustiness=0, surface_temperature)
    # If neutral_coefficients is nothing, create a new coefficient with sensible heat coefficients
    if isnothing(coef.neutral_coefficients)
        coef = PolynomialCoefficient(
            neutral_coefficients = (0.128, 0.068, 2.43),  # Large & Yeager (2009) sensible heat
            roughness_length = coef.roughness_length,
            minimum_wind_speed = coef.minimum_wind_speed,
            stability_function = coef.stability_function
        )
    end
    bf = BulkSensibleHeatFluxFunction(coef, gustiness, surface_temperature, nothing, nothing, nothing)
    return BoundaryCondition(Flux(), bf)
end

"""
$(TYPEDSIGNATURES)

Create a `BulkVaporFlux` boundary condition with a `PolynomialCoefficient`.
If `coef.neutral_coefficients` is `nothing`, automatically uses Large & Yeager (2009)
latent heat coefficients `(0.120, 0.070, 2.55)`.
"""
function BulkVaporFlux(coef::PolynomialCoefficient; gustiness=0, surface_temperature)
    # If neutral_coefficients is nothing, create a new coefficient with latent heat coefficients
    if isnothing(coef.neutral_coefficients)
        coef = PolynomialCoefficient(
            neutral_coefficients = (0.120, 0.070, 2.55),  # Large & Yeager (2009) latent heat
            roughness_length = coef.roughness_length,
            minimum_wind_speed = coef.minimum_wind_speed,
            stability_function = coef.stability_function
        )
    end
    bf = BulkVaporFluxFunction(coef, gustiness, surface_temperature, nothing, nothing, nothing)
    return BoundaryCondition(Flux(), bf)
end
