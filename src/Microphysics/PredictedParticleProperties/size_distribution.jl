#####
##### Ice Size Distribution
#####
##### The P3 scheme uses a generalized gamma distribution for ice particles.
#####

"""
    IceSizeDistributionState{FT}

State variables for evaluating integrals over the ice size distribution.

The ice particle size distribution follows a generalized gamma form:

```math
N'(D) = N_0 D^\\mu \\exp(-\\lambda D)
```

where:
- `N_0` is the intercept parameter [m^{-(4+μ)}]
- `μ` is the shape parameter (dimensionless)
- `λ` is the slope parameter [1/m]
- `D` is the particle diameter [m]

# Fields
- `intercept`: N_0, intercept parameter [m^{-(4+μ)}]
- `shape`: μ, shape parameter [-]
- `slope`: λ, slope parameter [1/m]
- `rime_fraction`: F_r, mass fraction that is rime [-]
- `liquid_fraction`: F_l, mass fraction that is liquid water on ice [-]
- `rime_density`: ρ_rim, density of rime [kg/m³]

# Derived quantities (computed from prognostic variables)
- `total_mass`: Total ice mass mixing ratio q_i [kg/kg]
- `number_concentration`: Ice number concentration N_i [1/kg]
"""
struct IceSizeDistributionState{FT}
    intercept :: FT          # N_0
    shape :: FT              # μ
    slope :: FT              # λ
    rime_fraction :: FT      # F_r
    liquid_fraction :: FT    # F_l
    rime_density :: FT       # ρ_rim
end

"""
    IceSizeDistributionState(; intercept, shape, slope, 
                               rime_fraction=0, liquid_fraction=0, rime_density=400)

Construct an `IceSizeDistributionState` with given parameters.
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

Evaluate the ice size distribution N'(D) at diameter D.

```math
N'(D) = N_0 D^\\mu \\exp(-\\lambda D)
```
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

Critical diameter D_crit separating small spherical ice from larger ice.
For unrimed ice (F_r = 0), this is approximately 15-20 μm.
"""
@inline function critical_diameter_small_ice(rime_fraction)
    # Simplified form - actual P3 uses more complex formulation
    return 15e-6  # 15 μm
end

"""
    critical_diameter_unrimed(rime_fraction, rime_density)

Critical diameter D_crit_s separating unrimed aggregates from partially rimed particles.
"""
@inline function critical_diameter_unrimed(rime_fraction, rime_density)
    # Simplified form
    return 100e-6  # 100 μm
end

"""
    critical_diameter_graupel(rime_fraction, rime_density)

Critical diameter D_crit_r separating partially rimed from fully rimed (graupel).
"""
@inline function critical_diameter_graupel(rime_fraction, rime_density)
    # Simplified form  
    return 500e-6  # 500 μm
end

