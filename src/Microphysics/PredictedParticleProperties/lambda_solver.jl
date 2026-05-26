#####
##### Lambda Solver for P3 Ice Size Distribution
#####
##### Given prognostic moments and ice properties (rime fraction, rime density),
##### solve for the gamma distribution parameters (N₀, λ, μ).
#####
##### The solver handles the piecewise mass-diameter relationship with four regimes
##### from Morrison & Milbrandt (2015a) Equations 1-5.
#####
##### Two closures are available:
##### 1. Two-moment: Uses the μ-λ relationship (Heymsfield 2003)
##### 2. Three-moment: Uses sixth moment Z to determine μ independently
#####

"""
    solve_lambda(L_ice, N_ice, rime_fraction, rime_density;
                 liquid_fraction = zero(typeof(L_ice)),
                 mass = IceMassPowerLaw(),
                 closure = TwoMomentClosure(),
                 logλ_bounds = (log(10), log(P3_LAMBDA_MAX)),
                 max_iterations = 50,
                 tolerance = 1e-10)

Solve for slope parameter λ given ice mass and number concentrations.

Uses the secant method to find logλ such that the computed L/N ratio
matches the observed ratio. This is the two-moment solver using the
μ-λ closure relationship.

# Arguments
- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of rime [kg/m³]

# Keyword Arguments
- `liquid_fraction`: Liquid water fraction [-] (default 0)
- `mass`: Power law parameters (default: `IceMassPowerLaw()`)
- `closure`: Two-moment closure (default: `TwoMomentClosure()`)

# Returns
- `logλ`: Log of slope parameter
"""
function solve_lambda(L_ice, N_ice, rime_fraction, rime_density;
                      liquid_fraction = zero(typeof(L_ice)),
                      mass = IceMassPowerLaw(),
                      closure = TwoMomentClosure(),
                      logλ_bounds = (log(10), log(P3_LAMBDA_MAX)),
                      max_iterations = 50,
                      tolerance = 1e-10)

    FT = typeof(L_ice)
    if L_ice <= 0 || N_ice <= 0
        # No ice mass or number: return upper bound to avoid unphysical λ = 0.
        return FT(logλ_bounds[2])
    end

    target = log(L_ice) - log(N_ice)
    # Pass L_ice, N_ice to log_mass_number_ratio for P3 closure D_mvd diagnostic
    f(logλ) = log_mass_number_ratio(mass, closure, rime_fraction, rime_density, liquid_fraction, logλ, L_ice, N_ice) - target

    # Secant method
    x₀, x₁ = FT.(logλ_bounds)
    f₀, f₁ = f(x₀), f(x₁)

    for _ in 1:max_iterations
        denom = f₁ - f₀
        abs(denom) < eps(FT) && return x₁

        Δx = f₁ * (x₁ - x₀) / denom
        x₂ = clamp(x₁ - Δx, FT(logλ_bounds[1]), FT(logλ_bounds[2]))

        abs(Δx) < tolerance * abs(x₁) && return x₂

        x₀, f₀ = x₁, f₁
        x₁, f₁ = x₂, f(x₂)
    end

    return x₁
end

"""
    solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ;
                 mass = IceMassPowerLaw(),
                 logλ_bounds = (log(10), log(P3_LAMBDA_MAX)),
                 max_iterations = 50,
                 tolerance = 1e-10)

Solve for slope parameter λ given a fixed shape parameter μ (three-moment).

For three-moment ice, μ is determined from the Z/N constraint, so this
function finds λ that satisfies the L/N constraint at that μ.

# Arguments
- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `Z_ice`: Ice sixth moment [m⁶/m³] (used for initial guess)
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of rime [kg/m³]
- `μ`: Shape parameter (determined from three-moment solver)

# Returns
- `logλ`: Log of slope parameter
"""
function solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ;
                      mass = IceMassPowerLaw(),
                      logλ_bounds = (log(10), log(P3_LAMBDA_MAX)),
                      max_iterations = 50,
                      tolerance = 1e-10)

    FT = typeof(L_ice)
    if L_ice <= 0 || N_ice <= 0
        # No ice mass or number: return upper bound to avoid unphysical λ = 0.
        return FT(logλ_bounds[2])
    end

    target = log(L_ice) - log(N_ice)

    function f(logλ)
        log_L_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ)
        log_N_over_N₀ = log_gamma_moment(μ, logλ)
        return (log_L_over_N₀ - log_N_over_N₀) - target
    end

    # Use Z/N constraint for initial guess if Z is available
    if !iszero(Z_ice)
        logλ_guess = log_lambda_from_reflectivity(μ, log(Z_ice) - log(N_ice))
        logλ_guess = clamp(logλ_guess, FT(logλ_bounds[1]), FT(logλ_bounds[2]))
    else
        logλ_guess = (FT(logλ_bounds[1]) + FT(logλ_bounds[2])) / 2
    end

    # Secant method starting from Z/N guess
    x₀ = FT(logλ_bounds[1])
    x₁ = logλ_guess
    f₀, f₁ = f(x₀), f(x₁)

    for _ in 1:max_iterations
        denom = f₁ - f₀
        abs(denom) < eps(FT) && return x₁

        Δx = f₁ * (x₁ - x₀) / denom
        x₂ = clamp(x₁ - Δx, FT(logλ_bounds[1]), FT(logλ_bounds[2]))

        abs(Δx) < tolerance * abs(x₁) && return x₂

        x₀, f₀ = x₁, f₁
        x₁, f₁ = x₂, f(x₂)
    end

    return x₁
end

"""
$(TYPEDSIGNATURES)

Compute N₀ from the normalization: N = N₀ × ∫ D^μ exp(-λD) dD.
"""
function intercept_parameter(N_ice, μ, logλ)
    log_N_over_N₀ = log_gamma_moment(μ, logλ)
    return N_ice / exp(log_N_over_N₀)
end

"""
    DiameterBounds

Physical bounds on ice particle diameters for the lambda solver.
See [`DiameterBounds()`](@ref) constructor.
"""
struct DiameterBounds{FT}
    D_min :: FT
    D_max :: FT
end

# Fortran P3 lambda limiter constants. The analytic limit
# `D_max(Fr) = Dm_max1 + Dm_max2 × Fr²` and `D_min = Dm_min` is shared by all
# three Fortran lookup-table generators that build the runtime `f1pr09 =
# inv_Qmin` / `f1pr10 = inv_Qmax` entries used by `microphy_p3.f90:2934-2935`:
#   - create_p3_lookupTable_1.f90:153-155, 516-519
#   - create_p3_lookupTable_2.f90:238-240, 866-868, 997-1000
#   - create_p3_lookupTable_3.f90:77-79, 313-315
const P3_DM_MAX_BASE = 5e-3    # 5 mm  (Fortran Dm_max1 = 5000e-6)
const P3_DM_MAX_RIME = 20e-3   # 20 mm (Fortran Dm_max2 = 20000e-6)
const P3_DM_MIN      = 2e-6    # 2 μm  (Fortran Dm_min  = 2e-6)
const P3_DM_MAX_CEIL = P3_DM_MAX_BASE + P3_DM_MAX_RIME  # Fr=1 ceiling = 25 mm
const P3_LAMBDA_MAX  = 1.6e7   # Fortran brute-force search upper bound

"""
$(TYPEDSIGNATURES)

Construct diameter bounds for the lambda solver.

The P3 scheme constrains the size distribution such that the mean diameter
remains within physical limits. This prevents unphysical distributions with
extremely small or large particles.

For a gamma distribution N'(D) = N₀ D^μ exp(-λD), the mean diameter is:
  D_mean = (μ + 1) / λ

To enforce D_min ≤ D_mean ≤ D_max:
  (μ + 1) / D_max ≤ λ ≤ (μ + 1) / D_min

# Keyword Arguments

- `D_min`: Minimum mean diameter [m], default 2 μm (`Dm_min`)
- `D_max`: Maximum mean diameter [m], default 25 mm (Fortran Fr=1 ceiling
  `Dm_max1 + Dm_max2`). Prefer the `DiameterBounds(FT, rime_fraction)` form
  to recover the rime-dependent Fortran value.

# Example

```julia
bounds = DiameterBounds(; D_min=5e-6, D_max=20e-3)  # 5 μm to 20 mm
```
"""
function DiameterBounds(FT = Float64; D_min = FT(P3_DM_MIN), D_max = FT(P3_DM_MAX_CEIL))
    return DiameterBounds(FT(D_min), FT(D_max))
end

"""
$(TYPEDSIGNATURES)

Construct Fr-dependent diameter bounds matching the Fortran P3 lambda limiter.

The maximum mean diameter depends on rime fraction Fr:
  D_max = 5 mm + 20 mm × Fr²

This ranges from 5 mm (unrimed, Fr=0) to 25 mm (fully rimed, Fr=1), matching
the analytic limit baked into all three Fortran lookup-table generators
(`create_p3_lookupTable_{1,2,3}.f90`). At runtime, Fortran enforces the same
constraint via tabulated `f1pr09 = inv_Qmin` / `f1pr10 = inv_Qmax` bounds on
`N/q` (`microphy_p3.f90:2934-2935`); Julia enforces it directly on λ here and
recomputes `N₀` from the mass moment in [`distribution_parameters`](@ref).

# Arguments

- `FT`: Float type
- `rime_fraction`: Rime mass fraction Fr ∈ [0, 1]
"""
@inline function DiameterBounds(FT, rime_fraction)
    D_min = FT(P3_DM_MIN)
    D_max = FT(P3_DM_MAX_BASE) + FT(P3_DM_MAX_RIME) * rime_fraction^2
    return DiameterBounds(D_min, D_max)
end

"""
$(TYPEDSIGNATURES)

Compute λ bounds from diameter bounds for a given shape parameter μ.

For D_mean = (μ + 1) / λ:
- λ_min = (μ + 1) / D_max
- λ_max = (μ + 1) / D_min

Returns (λ_min, λ_max).
"""
@inline function lambda_bounds_from_diameter(μ, bounds::DiameterBounds)
    FT = typeof(μ)
    λ_min = (μ + 1) / bounds.D_max
    # Match Fortran: λ_max = (μ+1)/D_min (create_p3_lookupTable_1.f90 line 1071)
    λ_max = (μ + 1) / bounds.D_min
    return (λ_min, λ_max)
end

"""
$(TYPEDSIGNATURES)

Clamp λ to ensure the mean diameter stays within physical bounds.

Returns the clamped λ value.
"""
@inline function enforce_diameter_bounds(λ, μ, bounds::DiameterBounds)
    (λ_min, λ_max) = lambda_bounds_from_diameter(μ, bounds)
    return clamp(λ, λ_min, λ_max)
end

"""
    IceDistributionParameters

Result of [`distribution_parameters`](@ref). Fields: `N₀`, `λ`, `μ`.
"""
struct IceDistributionParameters{FT}
    N₀ :: FT
    λ :: FT
    μ :: FT
end

"""
$(TYPEDSIGNATURES)

Solve for gamma size distribution parameters from two prognostic moments (L, N).

This is the two-moment closure for P3: given the prognostic ice mass ``L`` and
number ``N`` concentrations, plus the predicted rime properties, compute
the complete gamma distribution:

```math
N'(D) = N₀ D^μ e^{-λD}
```

The solution proceeds in three steps:

1. **Solve for λ**: Secant method finds the slope parameter satisfying
   the L/N ratio constraint with piecewise m(D)
2. **Compute μ**: Shape parameter from μ-λ relationship
3. **Compute N₀**: Intercept from number normalization

# Arguments

- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `rime_fraction`: Mass fraction of rime [-] (0 = unrimed, 1 = fully rimed)
- `rime_density`: Density of the rime layer [kg/m³]

# Keyword Arguments

- `mass`: Power law parameters (default: `IceMassPowerLaw()`)
- `closure`: Two-moment closure (default: `TwoMomentClosure()`)

# Returns

[`IceDistributionParameters`](@ref) with fields `N₀`, `λ`, `μ`.

# Example

```julia
using Breeze.Microphysics.PredictedParticleProperties

# Typical ice cloud conditions
L_ice = 1e-4  # 0.1 g/m³
N_ice = 1e5   # 100,000 particles/m³

params = distribution_parameters(L_ice, N_ice, 0.0, 400.0)
# IceDistributionParameters(N₀=..., λ=..., μ=...)
```

# References

See [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2b.
"""
function distribution_parameters(L_ice, N_ice, rime_fraction, rime_density;
                                  liquid_fraction = zero(typeof(L_ice)),
                                  mass = IceMassPowerLaw(),
                                  closure = TwoMomentClosure(),
                                  diameter_bounds = nothing)
    FT = typeof(L_ice)

    logλ = solve_lambda(L_ice, N_ice, rime_fraction, rime_density; liquid_fraction, mass, closure)
    λ = exp(logλ)
    μ = shape_parameter(closure, logλ, L_ice, N_ice, rime_fraction, rime_density, liquid_fraction, mass)

    # Fortran always applies Fr-dependent diameter bounds (D_max = 5mm + 20mm×Fr²).
    # Default to DiameterBounds(FT, rime_fraction) when not explicitly specified.
    bounds = isnothing(diameter_bounds) ? DiameterBounds(FT, rime_fraction) : diameter_bounds
    λ = enforce_diameter_bounds(λ, μ, bounds)

    # Compute N₀ from the mass constraint: L = N₀ × ∫ m(D) D^μ exp(-λD) dD.
    # This matches Fortran (create_p3_lookupTable_1.f90 line 1054):
    #   n0 = q / ((1-Fl)*(cs1*intgrR1 + ...) + Fl*cs5*intgrR5)
    # When λ is clamped at the upper bound, the number-normalized N₀
    # (= λ^(μ+1)/Γ(μ+1)) violates the mass constraint.  The mass-constrained
    # N₀ ensures the PSD always integrates to the correct total mass.
    logλ = log(λ)
    log_M_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ;
                                     liquid_fraction)
    N₀ = L_ice / exp(log_M_over_N₀)

    return IceDistributionParameters(N₀, λ, μ)
end

"""
$(TYPEDSIGNATURES)

Solve for gamma size distribution parameters from three prognostic moments (L, N, Z).

This is the three-moment solver for P3: given the prognostic ice mass ``L``,
number ``N``, and sixth moment ``Z`` concentrations, compute the complete
gamma distribution without needing an empirical μ-λ closure:

```math
N'(D) = N₀ D^μ e^{-λD}
```

The solution uses:
1. **Z/N constraint**: Determines λ as a function of μ
2. **L/N constraint**: Used to solve for the correct μ
3. **Normalization**: N₀ from the number integral

# Advantages of Three-Moment

- Shape parameter μ evolves physically based on actual size distribution
- Better representation of size sorting during sedimentation
- Improved simulation of hail and large, heavily rimed particles
- No need for empirical μ-λ parameterization

# Arguments

- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `Z_ice`: Ice sixth moment / reflectivity [m⁶/m³]
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of the rime layer [kg/m³]

# Keyword Arguments

- `mass`: Power law parameters (default: `IceMassPowerLaw()`)
- `closure`: Three-moment closure (default: `ThreeMomentClosure()`)

# Returns

[`IceDistributionParameters`](@ref) with fields `N₀`, `λ`, `μ`.

# Example

```julia
using Breeze.Microphysics.PredictedParticleProperties

# Ice with reflectivity constraint
L_ice = 1e-4   # 0.1 g/m³
N_ice = 1e5    # 100,000 particles/m³
Z_ice = 1e-12  # Sixth moment [m⁶/m³]

params = distribution_parameters(L_ice, N_ice, Z_ice, 0.0, 400.0)
# IceDistributionParameters(N₀=..., λ=..., μ=...)
```

# References

[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) introduced three-moment ice,
[Milbrandt et al. (2024)](@cite MilbrandtEtAl2024) refined the approach.
"""
function distribution_parameters(L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                                  liquid_fraction = zero(typeof(L_ice)),
                                  mass = IceMassPowerLaw(),
                                  closure = ThreeMomentClosure(),
                                  diameter_bounds = nothing)

    FT = typeof(L_ice)

    # Handle edge cases
    if iszero(N_ice) || iszero(L_ice)
        return IceDistributionParameters(zero(FT), zero(FT), zero(FT))
    end

    # Fortran always applies Fr-dependent diameter bounds.
    bounds = isnothing(diameter_bounds) ? DiameterBounds(FT, rime_fraction) : diameter_bounds

    # If Z is zero or negative, fall back to two-moment with μ at lower bound
    if Z_ice ≤ 0
        μ = closure.μmin
        logλ = solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ; mass)
        λ = exp(logλ)
        λ = enforce_diameter_bounds(λ, μ, bounds)

        # Use mass-constrained N₀ (matching two-moment path and Fortran).
        logλ_final = log(λ)
        log_M_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ_final;
                                         liquid_fraction)
        N₀ = L_ice / exp(log_M_over_N₀)
        return IceDistributionParameters(N₀, λ, μ)
    end

    # Compute μ from three-moment constraint with density iteration.
    # Fortran solve_mui iterates up to 5 times: at each step, the bulk density
    # ρ_bulk is updated from the lookup table (entry 12), which changes M₃ and
    # hence μ. Here we compute ρ_bulk analytically from the solved (μ, λ) pair
    # via ρ_bulk = 6L / (π M₃), where M₃ = N Γ(μ+4) / (Γ(μ+1) λ³).
    ρ_bulk = FT(mass.ice_density)  # initial guess: pure ice density (900 kg/m³)
    μ = FT(0)
    logλ = FT(0)
    for _ in 1:5
        M₃ = FT(6) * L_ice / (ρ_bulk * FT(π))
        μ_new = shape_parameter_from_moments(N_ice, M₃, Z_ice, closure.μmax)
        μ_new = clamp(μ_new, closure.μmin, closure.μmax)

        # Solve for λ using actual piecewise m-D relation
        logλ = solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ_new; mass)
        λ_iter = exp(logλ)

        # Update bulk density: ρ = 6L Γ(μ+1) λ³ / (π N Γ(μ+4))
        # This is the effective spherical-equivalent density of the PSD.
        log_ratio = loggamma(μ_new + 1) - loggamma(μ_new + 4)
        ρ_bulk_new = FT(6) * L_ice * exp(log_ratio) * λ_iter^3 / (FT(π) * N_ice)
        ρ_bulk_new = clamp(ρ_bulk_new, FT(50), FT(mass.ice_density))

        # Convergence check (Fortran tolerance: |μ_old - μ_new| < 0.25)
        converged = abs(μ_new - μ) < FT(0.25)
        μ = μ_new
        ρ_bulk = ρ_bulk_new
        converged && break
    end

    # Final solve with converged μ
    logλ = solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ; mass)
    λ = exp(logλ)
    λ = enforce_diameter_bounds(λ, μ, bounds)

    # Use mass-constrained N₀ (matching two-moment path and Fortran).
    # After λ clamping, number-normalized N₀ = N × λ^(μ+1)/Γ(μ+1) violates
    # the mass constraint. Mass-constrained N₀ ensures L = N₀ × ∫m(D)D^μ e^{-λD}dD.
    logλ_final = log(λ)
    log_M_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ_final;
                                     liquid_fraction)
    N₀ = L_ice / exp(log_M_over_N₀)

    return IceDistributionParameters(N₀, λ, μ)
end
