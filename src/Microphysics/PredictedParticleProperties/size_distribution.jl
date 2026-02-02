#####
##### Ice Size Distribution
#####
##### The P3 scheme uses a generalized gamma distribution for ice particles.
#####

"""
    IceSizeDistributionState

State container for ice size distribution integration.
See [`IceSizeDistributionState`](@ref) constructor for details.
"""
struct IceSizeDistributionState{FT}
    intercept :: FT
    shape :: FT
    slope :: FT
    rime_fraction :: FT
    liquid_fraction :: FT
    rime_density :: FT
    # Mass-diameter power law parameters (α, β) from m = α D^β
    mass_coefficient :: FT
    mass_exponent :: FT
    ice_density :: FT
    # Reference air density for fall speed correction
    reference_air_density :: FT
    air_density :: FT
end

"""
$(TYPEDSIGNATURES)

Construct an `IceSizeDistributionState` for quadrature evaluation.

The ice particle size distribution follows a generalized gamma form:

```math
N'(D) = N_0 D^μ e^{-λD}
```

The gamma distribution is parameterized by three quantities:

- **N₀** (intercept): Sets the total number of particles
- **μ** (shape): Controls the relative abundance of small vs. large particles
- **λ** (slope): Sets the characteristic inverse diameter

For P3, these are determined from prognostic moments using the
[`distribution_parameters`](@ref) function.

**Rime and liquid properties** affect the mass-diameter relationship:

- `rime_fraction`: Fraction of mass that is rime (0 = pristine, 1 = graupel)
- `rime_density`: Density of the accreted rime layer
- `liquid_fraction`: Liquid water coating from partial melting

# Required Keyword Arguments

- `intercept`: N₀ [m^{-(4+μ)}]
- `shape`: μ [-]
- `slope`: λ [1/m]

# Optional Keyword Arguments

- `rime_fraction`: Fᶠ [-], default 0 (unrimed)
- `liquid_fraction`: Fˡ [-], default 0 (no meltwater)
- `rime_density`: ρᶠ [kg/m³], default 400
- `mass_coefficient`: α in m = α D^β [kg/m^β], default 0.0121
- `mass_exponent`: β in m = α D^β [-], default 1.9
- `ice_density`: Pure ice density [kg/m³], default 917
- `reference_air_density`: ρ₀ for fall speed correction [kg/m³], default 1.225
- `air_density`: Local air density [kg/m³], default 1.225

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2b.
"""
function IceSizeDistributionState(FT::Type{<:AbstractFloat} = Float64;
                                   intercept,
                                   shape,
                                   slope,
                                   rime_fraction = zero(FT),
                                   liquid_fraction = zero(FT),
                                   rime_density = FT(400),
                                   mass_coefficient = FT(0.0121),
                                   mass_exponent = FT(1.9),
                                   ice_density = FT(917),
                                   reference_air_density = FT(1.225),
                                   air_density = FT(1.225))
    return IceSizeDistributionState(
        FT(intercept),
        FT(shape),
        FT(slope),
        FT(rime_fraction),
        FT(liquid_fraction),
        FT(rime_density),
        FT(mass_coefficient),
        FT(mass_exponent),
        FT(ice_density),
        FT(reference_air_density),
        FT(air_density)
    )
end

"""
    size_distribution(D, state::IceSizeDistributionState)

Evaluate the ice size distribution ``N'(D)`` at diameter D.

Returns the number density of particles per unit diameter interval:

```math
N'(D) = N_0 D^μ e^{-λD}
```

The total number concentration is ``N = ∫_0^∞ N'(D) dD``.
"""
@inline function size_distribution(D, state::IceSizeDistributionState)
    N₀ = state.intercept
    μ = state.shape
    λ = state.slope
    return N₀ * D^μ * exp(-λ * D)
end

#####
##### P3 particle property regimes
#####

"""
    critical_diameter_small_ice(rime_fraction)

Threshold diameter below which ice particles are treated as small spheres.

This function returns the D_th threshold computed from the mass-diameter
relationship: D_th = (6α / (π ρᵢ))^(1/(3-β))

Using default P3 parameters (α = 0.0121, β = 1.9, ρᵢ = 917):
D_th ≈ 15 μm

See [`ice_regime_thresholds`](@ref) for the complete implementation with
explicit mass power law parameters.
"""
@inline function critical_diameter_small_ice(rime_fraction)
    FT = typeof(rime_fraction)
    # D_th = (6α / (π ρᵢ))^(1/(3-β)) with default P3 parameters
    α = FT(0.0121)
    β = FT(1.9)
    ρᵢ = FT(917)
    return (6 * α / (FT(π) * ρᵢ))^(1 / (3 - β))
end

"""
    critical_diameter_unrimed(rime_fraction, rime_density)

Threshold diameter separating unrimed aggregates from partially rimed particles.

For unrimed ice (Fᶠ = 0), this threshold is infinite (no partially rimed regime).
For rimed ice, this is the D_cr threshold from Morrison & Milbrandt (2015a).
"""
@inline function critical_diameter_unrimed(rime_fraction, rime_density)
    FT = typeof(rime_fraction)
    Fᶠ = rime_fraction
    ρᶠ = rime_density

    # For unrimed ice, return large value (no partial rime regime)
    is_unrimed = Fᶠ < FT(1e-10)

    # Default P3 parameters
    α = FT(0.0121)
    β = FT(1.9)

    # Safe rime fraction
    Fᶠ_safe = max(Fᶠ, FT(1e-10))

    # Deposited ice density (Eq. 16 from MM15a)
    k = (1 - Fᶠ_safe)^(-1 / (3 - β))
    num = ρᶠ * Fᶠ_safe
    den = (β - 2) * (k - 1) / ((1 - Fᶠ_safe) * k - 1) - (1 - Fᶠ_safe)
    ρ_dep = num / max(den, FT(1e-10))

    # Graupel density
    ρ_g = Fᶠ_safe * ρᶠ + (1 - Fᶠ_safe) * ρ_dep

    # Partial rime threshold: D_cr
    D_cr = (6 * α / (FT(π) * ρ_g * (1 - Fᶠ_safe)))^(1 / (3 - β))

    return ifelse(is_unrimed, FT(Inf), D_cr)
end

"""
    critical_diameter_graupel(rime_fraction, rime_density)

Threshold diameter separating partially rimed ice from dense graupel.

For unrimed ice (Fᶠ = 0), this threshold is infinite (no graupel regime).
For rimed ice, this is the D_gr threshold from Morrison & Milbrandt (2015a).
"""
@inline function critical_diameter_graupel(rime_fraction, rime_density)
    FT = typeof(rime_fraction)
    Fᶠ = rime_fraction
    ρᶠ = rime_density

    # For unrimed ice, return large value (no graupel regime)
    is_unrimed = Fᶠ < FT(1e-10)

    # Default P3 parameters
    α = FT(0.0121)
    β = FT(1.9)

    # Safe rime fraction
    Fᶠ_safe = max(Fᶠ, FT(1e-10))

    # Deposited ice density (Eq. 16 from MM15a)
    k = (1 - Fᶠ_safe)^(-1 / (3 - β))
    num = ρᶠ * Fᶠ_safe
    den = (β - 2) * (k - 1) / ((1 - Fᶠ_safe) * k - 1) - (1 - Fᶠ_safe)
    ρ_dep = num / max(den, FT(1e-10))

    # Graupel density
    ρ_g = Fᶠ_safe * ρᶠ + (1 - Fᶠ_safe) * ρ_dep

    # Graupel threshold: D_gr
    D_gr = (6 * α / (FT(π) * ρ_g))^(1 / (3 - β))

    return ifelse(is_unrimed, FT(Inf), D_gr)
end
