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

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2b.
"""
function IceSizeDistributionState(FT::Type{<:AbstractFloat} = Float64;
                                   intercept,
                                   shape,
                                   slope,
                                   rime_fraction = zero(FT),
                                   liquid_fraction = zero(FT),
                                   rime_density = FT(400))
    return IceSizeDistributionState(
        FT(intercept),
        FT(shape),
        FT(slope),
        FT(rime_fraction),
        FT(liquid_fraction),
        FT(rime_density)
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

!!! note
    This is a simplified placeholder. The full P3 formulation computes
    this threshold dynamically from the mass-diameter relationship.
    See [`ice_regime_thresholds`](@ref) for the complete implementation.
"""
@inline function critical_diameter_small_ice(rime_fraction)
    return 15e-6  # 15 μm (placeholder)
end

"""
    critical_diameter_unrimed(rime_fraction, rime_density)

Threshold diameter separating unrimed aggregates from partially rimed particles.

!!! note
    This is a simplified placeholder. See [`ice_regime_thresholds`](@ref).
"""
@inline function critical_diameter_unrimed(rime_fraction, rime_density)
    return 100e-6  # 100 μm (placeholder)
end

"""
    critical_diameter_graupel(rime_fraction, rime_density)

Threshold diameter separating partially rimed ice from dense graupel.

!!! note
    This is a simplified placeholder. See [`ice_regime_thresholds`](@ref).
"""
@inline function critical_diameter_graupel(rime_fraction, rime_density)
    return 500e-6  # 500 μm (placeholder)
end

