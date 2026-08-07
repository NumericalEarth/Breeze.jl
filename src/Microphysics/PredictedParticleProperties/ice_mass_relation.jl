#####
##### Mass-diameter relationship parameters
#####

"""
    IceMassPowerLaw

Power law for ice particle mass. See [`IceMassPowerLaw()`](@ref) constructor.
"""
struct IceMassPowerLaw{FT}
    coefficient :: FT
    exponent :: FT
    ice_density :: FT
end

"""
$(TYPEDSIGNATURES)

Construct power law parameters for ice particle mass: ``m(D) = α D^β``.

For vapor-grown aggregates (regime 2 in P3), the mass-diameter relationship
follows a power law with empirically-determined coefficients. This captures
the fractal nature of ice crystal aggregates, which have effective densities
much lower than pure ice.

# Physical Interpretation

The exponent ``β ≈ 1.9`` (less than 3) means density decreases with size:
- Small particles: closer to solid ice density
- Large aggregates: fluffy, low effective density

This is the key to P3's smooth transitions—as particles grow and aggregate,
their properties evolve continuously without discrete category jumps.

# Keyword Arguments

- `coefficient`: α in m(D) = α D^β [kg/m^β], default 0.0121
- `exponent`: β in m(D) = α D^β [-], default 1.9
- `ice_density`: Pure ice density [kg/m³], default 900

# References

Default parameters from [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization)
supplementary material, based on aircraft observations.
"""
function IceMassPowerLaw(FT = Oceananigans.defaults.FloatType;
                         coefficient = 0.0121,
                         exponent = 1.9,
                         ice_density = 900)
    return IceMassPowerLaw(FT(coefficient), FT(exponent), FT(ice_density))
end

#####
##### Diameter thresholds between particle regimes
#####

"""
$(TYPEDSIGNATURES)

Diameter threshold from mass power law: D = (6α / πρ)^(1/(3-β))

Used to determine boundaries between spherical ice, aggregates, and graupel.
"""
@inline function regime_threshold(α, β, ρ)
    FT = typeof(α)
    return (6 * α / (FT(π) * ρ))^(1 / (3 - β))
end

"""
    deposited_ice_density(mass, rime_fraction, rime_density)

Density of the vapor-deposited (unrimed) portion of ice particles.
Equation 16 in [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).
"""
@inline function deposited_ice_density(mass::IceMassPowerLaw, rime_fraction, rime_density)
    β = mass.exponent
    Fᶠ = rime_fraction
    ρᶠ = rime_density
    FT = typeof(β)

    # Compute rimed density (clamp Fᶠ away from both 0 and 1 to avoid
    # division by zero at Fᶠ=0 and 0*Inf=NaN at Fᶠ=1 in IEEE arithmetic)
    Fᶠ_safe = clamp(Fᶠ, eps(FT), 1 - eps(FT))
    k = (1 - Fᶠ_safe)^(-1 / (3 - β))
    num = ρᶠ * Fᶠ_safe
    den = (β - 2) * (k - 1) / ((1 - Fᶠ_safe) * k - 1) - (1 - Fᶠ_safe)
    ρ_dep_rimed = num / max(den, eps(FT))

    # Return ice_density for unrimed case, computed density otherwise
    return ifelse(Fᶠ <= eps(FT), mass.ice_density, ρ_dep_rimed)
end

"""
$(TYPEDSIGNATURES)

Bulk density of graupel particles (rime + deposited ice).
"""
@inline function graupel_density(rime_fraction, rime_density, deposited_density)
    return rime_fraction * rime_density + (1 - rime_fraction) * deposited_density
end

"""
$(TYPEDSIGNATURES)

Return (a, b) for ice mass at diameter D: m(D) = a D^b.

The relationship is piecewise across four regimes:
1. D < D_spherical: small spheres, m = (π/6)ρᵢ D³
2. D_spherical ≤ D < D_graupel: aggregates, m = α D^β
3. D_graupel ≤ D < D_partial: graupel, m = (π/6)ρ_g D³
4. D ≥ D_partial: partially rimed, m = α/(1-Fᶠ) D^β
"""
function ice_mass_coefficients(mass::IceMassPowerLaw, rime_fraction, rime_density, D)
    FT = typeof(D)
    α = mass.coefficient
    β = mass.exponent
    ρᵢ = mass.ice_density
    Fᶠ = rime_fraction

    thresholds = ice_regime_thresholds(mass, rime_fraction, rime_density)

    # Regime 1: small spheres
    a₁ = ρᵢ * FT(π) / 6
    b₁ = FT(3)

    # Regime 2: aggregates (also used for unrimed large particles)
    a₂ = FT(α)
    b₂ = FT(β)

    # Regime 3: graupel
    a₃ = thresholds.ρ_graupel * FT(π) / 6
    b₃ = FT(3)

    # Regime 4: partially rimed (avoid division by zero)
    Fᶠ_safe = min(Fᶠ, 1 - eps(FT))
    a₄ = FT(α) / (1 - Fᶠ_safe)
    b₄ = FT(β)

    # Determine which regime applies (work backwards from regime 4)
    # Note: same logic and ordering as particle_mass_ice_only in quadrature.jl
    is_regime_4 = D ≥ thresholds.partial_rime
    is_regime_3 = D ≥ thresholds.graupel
    is_regime_2 = D ≥ thresholds.spherical

    # Select coefficients: start with regime 4, override with 3, 2, 1 as conditions apply
    a = ifelse(is_regime_4, a₄, a₃)
    b = ifelse(is_regime_4, b₄, b₃)

    a = ifelse(is_regime_3, a, a₂)
    b = ifelse(is_regime_3, b, b₂)

    a = ifelse(is_regime_2, a, a₁)
    b = ifelse(is_regime_2, b, b₁)

    return (a, b)
end

"""
$(TYPEDSIGNATURES)

Compute ice particle mass at diameter D.
"""
function ice_mass(mass::IceMassPowerLaw, rime_fraction, rime_density, D)
    (a, b) = ice_mass_coefficients(mass, rime_fraction, rime_density, D)
    return a * D^b
end
