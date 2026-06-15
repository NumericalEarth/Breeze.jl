#####
##### μ-λ relationship
#####

#####
##### Two-moment closure: μ-λ relationship
#####

"""
    FixedShapeParameter

Fixed shape parameter closure: always returns a constant μ regardless of λ.
Used for tabulation with exponential PSD (μ=0) to match Fortran Table 1.
See [`FixedShapeParameter()`](@ref) constructor.
"""
struct FixedShapeParameter{FT}
    μ :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a fixed shape parameter closure.

This bypasses the empirical μ-λ relationship and uses a constant μ for all λ values.
The primary use case is tabulation: the Fortran P3 Table 1 is generated with μ=0
(exponential PSD), so using `FixedShapeParameter(0)` produces tables that match
the Fortran reference values.

# Keyword Arguments

- `μ`: Fixed shape parameter value, default 0 (exponential distribution)
"""
function FixedShapeParameter(FT::Type{<:AbstractFloat} = Float64; μ = 0)
    return FixedShapeParameter(FT(μ))
end

@inline function shape_parameter(closure::FixedShapeParameter, logλ, args...)
    return closure.μ
end

"""
    TwoMomentClosure

Updated μ-λ closure for P3, including the large-particle diagnostic.
See [`TwoMomentClosure()`](@ref) constructor.
"""
struct TwoMomentClosure{FT}
    # Constants for small particle regime (Heymsfield 2003)
    a :: FT
    b :: FT
    c :: FT
    μmax_small :: FT
    # Constants for large particle regime
    μmax_large :: FT
    D_threshold :: FT # Threshold diameter [m] (0.2 mm)
end

"""
$(TYPEDSIGNATURES)

Construct the P3 μ-λ closure which includes a diagnostic for large rimed particles.

This closure matches the logic in the official P3 Fortran code (lookup table generation).
It uses the [Heymsfield (2003)](@cite Heymsfield2003) μ–λ fit for small particles,
but switches to a diagnostic based on mean volume diameter (D_mvd) for large
particles to account for riming effects.

# Logic

1. Compute mean volume diameter ``D_{mvd} = ((L/N) / (\\frac{\\pi}{6} \\rho_g))^{1/3}``
2. If ``D_{mvd} \\le 0.2`` mm:
   Use Heymsfield (2003) relation: ``\\mu = 0.076 (0.01 \\lambda)^{0.8} - 2`` (clamped [0, 6])
3. If ``D_{mvd} > 0.2`` mm:
   ``\\mu = 0.25 (D_{mvd} - 0.2) f_\\rho F^f`` (clamped [0, 20])
   where ``f_\\rho = \\max(1, 1 + 0.00842(\\rho_g - 400))``

# Keyword Arguments

- `a`, `b`, `c`: Constants for the small-particle Heymsfield (2003) branch
- `μmax_small`: Max μ for small regime (default 6)
- `μmax_large`: Max μ for large regime (default 20)
- `D_threshold`: Threshold D_mvd [m] (default 2e-4)
"""
function TwoMomentClosure(FT = Oceananigans.defaults.FloatType;
                   a = 0.076 * 0.01^0.8,
                   b = 0.8,
                   c = 2,
                   μmax_small = 6,
                   μmax_large = 20,
                   D_threshold = 2e-4)
    return TwoMomentClosure(FT(a), FT(b), FT(c), FT(μmax_small), FT(μmax_large), FT(D_threshold))
end

"""
    shape_parameter(closure, logλ, L_ice, N_ice, rime_fraction, rime_density, liquid_fraction, mass_params)

Compute shape parameter μ.
"""
@inline function shape_parameter(closure::TwoMomentClosure, logλ, L_ice, N_ice, rime_fraction, rime_density, liquid_fraction, mass::IceMassPowerLaw)
    FT = typeof(closure.a)
    λ = exp(logλ)

    # 1. Compute graupel density (rho_g)
    # Fortran convention: for Fr=0, cgp = crp = ρ_rime × π/6, so the effective
    # density used in D_mvd is ρ_rime (not ρⁱ).  This matches the Fortran's
    # diagnostic_mui / diagnostic_mui_Fl shape parameter diagnostic.
    ρ_dep = deposited_ice_density(mass, rime_fraction, rime_density)
    ρ_g_rimed = graupel_density(rime_fraction, rime_density, ρ_dep)
    ρ_g_dry = ifelse(iszero(rime_fraction), rime_density, ρ_g_rimed)
    # blend liquid water density into bulk density (Fortran diagnostic_mui_Fl)
    ρᴸ = FT(1000)
    ρ_g = (1 - liquid_fraction) * ρ_g_dry + liquid_fraction * ρᴸ

    # 2. Compute D_mvd (Mean Volume Diameter)
    # Fortran diagnostic_mui uses mean mass per particle q = qi_tot/ni_tot,
    # then D_mvd = (q / (π/6 × ρ_g))^(1/3). L_ice alone is total mass, not per-particle.
    mean_mass = L_ice / max(N_ice, eps(FT))
    D_mvd_cubed = mean_mass / (FT(π)/6 * ρ_g)
    D_mvd = ifelse(D_mvd_cubed <= 0, FT(0), D_mvd_cubed^(1/3))

    # 3. Compute both regimes, select based on D_mvd
    # Small regime: Heymsfield 2003
    μ_small = closure.a * λ^closure.b - closure.c
    μ_small = clamp(μ_small, 0, closure.μmax_small)

    # Large regime
    D_mvd_mm = D_mvd * 1000
    D_thres_mm = closure.D_threshold * 1000
    f_ρ = max(1, 1 + FT(0.00842) * (ρ_g - 400))
    μ_large = FT(0.25) * (D_mvd_mm - D_thres_mm) * f_ρ * rime_fraction
    μ_large = clamp(μ_large, 0, closure.μmax_large)

    return ifelse(D_mvd <= closure.D_threshold, μ_small, μ_large)
end

#####
##### Three-moment closure: Z/N constraint
#####

"""
    ThreeMomentClosure

Fortran-parity three-moment closure using the upstream P3 `solve_mui` approximation.
See [`ThreeMomentClosure()`](@ref) constructor.
"""
struct ThreeMomentClosure{FT}
    μmin :: FT
    μmax :: FT
end

"""
$(TYPEDSIGNATURES)

Construct the Fortran-parity three-moment closure for gamma size distribution.

This closure follows the current upstream P3 implementation: it iterates on bulk
ice density, approximates the third diameter moment as spherical, and applies the
piecewise-polynomial `G(μ)` inversion used by `solve_mui`.

Use this closure when Fortran parity is the priority.

# Keyword Arguments

- `μmin`: Minimum shape parameter, default 0 (exponential distribution)
- `μmax`: Maximum shape parameter, default 20

# References

[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021),
[Morrison et al. (2025)](@cite Morrison2025complete3moment).
"""
function ThreeMomentClosure(FT = Oceananigans.defaults.FloatType;
                            μmin = 0,
                            μmax = 20)
    return ThreeMomentClosure(FT(μmin), FT(μmax))
end

"""
    IceRegimeThresholds

Diameter thresholds between P3 ice regimes. See [`ice_regime_thresholds`](@ref).
"""
struct IceRegimeThresholds{FT}
    spherical :: FT
    graupel :: FT
    partial_rime :: FT
    ρ_graupel :: FT
end

"""
$(TYPEDSIGNATURES)

Compute diameter thresholds separating the four P3 ice particle regimes.

P3's key innovation is a piecewise mass-diameter relationship that
transitions smoothly between ice particle types:

1. **Small spherical** (D < D_th): Dense, nearly solid ice crystals
2. **Vapor-grown aggregates** (D_th ≤ D < D_gr): Fractal aggregates, m ∝ D^β
3. **Graupel** (D_gr ≤ D < D_cr): Compact, heavily rimed particles
4. **Partially rimed** (D ≥ D_cr): Large aggregates with rimed cores

The thresholds depend on rime fraction and rime density, so they evolve
as particles rime—no ad-hoc category conversions needed.

# Arguments

- `mass`: Power law parameters for vapor-grown aggregates
- `rime_fraction`: Fraction of particle mass that is rime (0 to 1)
- `rime_density`: Density of rime layer [kg/m³]

# Returns

[`IceRegimeThresholds`](@ref) with fields:
- `spherical`: D_th threshold [m]
- `graupel`: D_gr threshold [m]
- `partial_rime`: D_cr threshold [m]
- `ρ_graupel`: Bulk density of graupel [kg/m³]

# References

See [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Equations 12-14.
"""
@inline function ice_regime_thresholds(mass::IceMassPowerLaw, rime_fraction, rime_density)
    # NOTE (M3): This duplicates regime_thresholds_from_state() in quadrature.jl.
    # Both must produce identical thresholds. If one is changed, update the other.
    α = mass.coefficient
    β = mass.exponent
    ρᵢ = mass.ice_density
    Fᶠ = rime_fraction
    ρᶠ = rime_density
    FT = typeof(α)

    D_spherical = regime_threshold(α, β, ρᵢ)

    # Compute rimed case thresholds (will be ignored if unrimed)
    # Clamp Fᶠ away from 0 and 1 to avoid 0*Inf=NaN in IEEE arithmetic
    Fᶠ_safe = clamp(Fᶠ, eps(FT), 1 - eps(FT))
    ρ_dep = deposited_ice_density(mass, Fᶠ_safe, ρᶠ)
    ρ_g = graupel_density(Fᶠ_safe, ρᶠ, ρ_dep)

    D_graupel = regime_threshold(α, β, ρ_g)
    D_partial = regime_threshold(α, β, ρ_g * (1 - Fᶠ_safe))

    # For unrimed ice (Fᶠ = 0), use Inf thresholds; otherwise use computed values
    is_unrimed = iszero(Fᶠ)
    D_graupel_out = ifelse(is_unrimed, FT(Inf), D_graupel)
    D_partial_out = ifelse(is_unrimed, FT(Inf), D_partial)
    ρ_g_out = ifelse(is_unrimed, ρᵢ, ρ_g)

    return IceRegimeThresholds(D_spherical, D_graupel_out, D_partial_out, ρ_g_out)
end
