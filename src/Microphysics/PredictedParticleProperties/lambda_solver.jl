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
                      mass = IceMassPowerLaw(typeof(L_ice)),
                      closure = TwoMomentClosure(typeof(L_ice)),
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
                 liquid_fraction = zero(typeof(L_ice)),
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
- `Z_ice`: Ice sixth moment [m⁶/m³] (retained for API symmetry; λ at fixed μ
  is determined only by `L_ice / N_ice`)
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of rime [kg/m³]
- `μ`: Shape parameter (determined from three-moment solver)

# Keyword Arguments
- `liquid_fraction`: Liquid water fraction [-] (default 0). `L_ice` is the total
  (ice + liquid coating) mass, so the λ solve must integrate the same wet m(D)
  that normalizes `N₀` in [`distribution_parameters`](@ref); the Fortran
  three-moment table generator targets the wet mass identically
  (`create_p3_lookupTable_3.f90:335`, `:366`).

# Returns
- `logλ`: Log of slope parameter
"""
function solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ;
                      liquid_fraction = zero(typeof(L_ice)),
                      mass = IceMassPowerLaw(typeof(L_ice)),
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
        log_L_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ;
                                         liquid_fraction)
        log_N_over_N₀ = log_gamma_moment(μ, logλ)
        return (log_L_over_N₀ - log_N_over_N₀) - target
    end

    # The mass-to-number ratio decreases monotonically with λ at fixed μ.
    # Bracket the solution so an inadmissible Z cannot alter or derail the mass solve.
    x_lower, x_upper = FT.(logλ_bounds)
    f_lower, f_upper = f(x_lower), f(x_upper)

    # The requested mean mass lies outside the numerical λ range.
    f_lower <= 0 && return x_lower
    f_upper >= 0 && return x_upper

    for _ in 1:max_iterations
        x = (x_lower + x_upper) / 2
        f_x = f(x)
        abs(f_x) < tolerance && return x
        x_upper - x_lower < tolerance * max(abs(x), one(FT)) && return x

        if f_x > 0
            x_lower = x
        else
            x_upper = x
        end
    end

    return (x_lower + x_upper) / 2
end

"""
$(TYPEDSIGNATURES)

Solve for the shape parameter μ from the sixth-moment constraint ``M₆ = Z``.

At fixed μ, [`solve_lambda`](@ref) first finds the λ implied by the prognostic
mass-to-number ratio. The physical mean-diameter bounds are then applied and ``N₀`` is
normalized from mass. The bounded distribution therefore always preserves ``L``; when a
diameter bound binds, its represented number concentration can differ from ``N``, matching
P3's policy of adjusting number to keep mean particle size physical.

The represented sixth moment is

```math
M₆(μ) = N₀(μ) \\frac{Γ(μ + 7)}{λ(μ)^{μ + 7}}.
```

For the P3 mass-diameter law this bounded, mass-normalized ``M₆`` decreases with μ, so
the three-moment solution is the single root of ``\\log M₆(μ) - \\log Z``, bracketed
by `closure.μmin` and `closure.μmax` and found here by bisection.

When ``Z`` lies outside the range the bounds can represent, μ is pinned to the nearest
bound, and the returned [`IceDistributionParameters`](@ref) reports the adjusted sixth
moment represented by that boundary distribution.

When no diameter limiter binds, this is the direct counterpart of the constraint used to
generate Fortran Table 3 (`create_p3_lookupTable_3.f90:288-386`). It is not equivalent to
the legacy runtime `solve_mui`: that routine estimates the geometric third moment from a
mass-weighted density, an approximation for P3's variable-density particles. Breeze's
runtime model path reads μ from Table 3 rather than calling either iterative solver.
"""
function solve_shape_parameter(L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                               liquid_fraction = zero(typeof(L_ice)),
                               mass = IceMassPowerLaw(typeof(L_ice)),
                               closure = ThreeMomentClosure(typeof(L_ice)),
                               diameter_bounds = nothing,
                               max_iterations = 60,
                               tolerance = 1e-6)

    FT = typeof(L_ice)

    # Without positive mass and number there is no distribution to solve for.
    if L_ice <= 0 || N_ice <= 0
        return FT(closure.μmin)
    end

    # A nonpositive sixth moment is below the representable range. The narrowest
    # allowable distribution supplies the nearest physical boundary value.
    Z_ice <= 0 && return FT(closure.μmax)

    log_Z = log(Z_ice)
    bounds = isnothing(diameter_bounds) ? DiameterBounds(FT, rime_fraction) : diameter_bounds

    function residual(μ)
        moments = distribution_moments_at_shape(L_ice, N_ice, Z_ice,
                                                rime_fraction, rime_density, μ,
                                                bounds; liquid_fraction, mass)
        return moments.log_sixth_moment - log_Z
    end

    μ_min = FT(closure.μmin)
    μ_max = FT(closure.μmax)

    # Z above what the widest distribution supplies, or below what the narrowest does.
    residual(μ_min) <= 0 && return μ_min
    residual(μ_max) >= 0 && return μ_max

    for _ in 1:max_iterations
        μ_max - μ_min < tolerance && break
        μ = (μ_min + μ_max) / 2
        if residual(μ) > 0
            μ_min = μ
        else
            μ_max = μ
        end
    end

    return (μ_min + μ_max) / 2
end

"""
$(TYPEDSIGNATURES)

Return the bounded slope, mass-normalized intercept, and represented zeroth and sixth
moments at fixed shape parameter μ.
"""
function distribution_moments_at_shape(L_ice, N_ice, Z_ice,
                                       rime_fraction, rime_density, μ, bounds;
                                       liquid_fraction = zero(typeof(L_ice)),
                                       mass = IceMassPowerLaw(typeof(L_ice)))
    logλ = solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ;
                        liquid_fraction, mass)
    λ = enforce_diameter_bounds(exp(logλ), μ, bounds)
    logλ = log(λ)

    log_mass = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ;
                               liquid_fraction)
    log_intercept = log(L_ice) - log_mass
    log_number = log_intercept + log_gamma_moment(μ, logλ)
    log_sixth_moment = log_intercept + log_gamma_moment(μ, logλ; k = 6)

    return (; λ, log_intercept, log_number, log_sixth_moment)
end

"""
$(TYPEDSIGNATURES)

Compute N₀ from the normalization: N = N₀ × ∫ D^μ exp(-λD) dD.

This is the number-normalized intercept. [`distribution_parameters`](@ref) returns the
mass-normalized one instead, `N₀ = L / ∫ m(D) D^μ exp(-λD) dD`; the two agree except where
the mean-diameter limiter clamps λ, in which case only the mass-normalized intercept
reproduces `L`.
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

Result of [`distribution_parameters`](@ref).

The fields `number_concentration` and `sixth_moment` are the moments represented by the
returned PSD. They equal the supplied prognostic moments when those moments admit a PSD
within the configured μ and mean-diameter bounds. When a limiter binds, they expose the
adjusted moments rather than implying that the original, inadmissible moments were retained.

`log_intercept` is ``\\log N₀``, and it is the field to build on in reduced precision.
Because ``N₀`` carries units of m^-(4+μ), its magnitude grows without physical meaning as
the distribution narrows: a 10 μm, μ = 8 distribution has ``N₀ ≈ 10⁵⁴``, which exceeds
the Float32 range even though every moment of that distribution is representable. `N₀` is
reported as `exp(log_intercept)` and therefore overflows to `Inf` in such cases, while
`log_intercept` and both represented moments stay exact. Evaluate the distribution as

```math
N'(D) = \\exp(\\log N₀ + μ \\log D - λ D)
```

rather than multiplying by ``N₀``. For the degenerate no-ice result every field is zero
except `log_intercept`, which is ``-∞`` so that ``\\exp(\\log N₀) = N₀ = 0`` still holds.
"""
struct IceDistributionParameters{FT}
    N₀ :: FT
    λ :: FT
    μ :: FT
    number_concentration :: FT
    sixth_moment :: FT
    log_intercept :: FT
end

function IceDistributionParameters(N₀::FT, λ::FT, μ::FT) where FT
    if N₀ <= 0 || λ <= 0
        return IceDistributionParameters{FT}(N₀, λ, μ, zero(FT), zero(FT), FT(-Inf))
    end

    logλ = log(λ)
    log_intercept = log(N₀)
    number_concentration = exp(log_intercept + log_gamma_moment(μ, logλ))
    sixth_moment = exp(log_intercept + log_gamma_moment(μ, logλ; k = 6))
    return IceDistributionParameters{FT}(N₀, λ, μ, number_concentration, sixth_moment,
                                         log_intercept)
end

function IceDistributionParameters(N₀, λ, μ)
    parameters = promote(N₀, λ, μ)
    return IceDistributionParameters(parameters...)
end

function IceDistributionParameters(N₀, λ, μ, number_concentration, sixth_moment, log_intercept)
    parameters = promote(N₀, λ, μ, number_concentration, sixth_moment, log_intercept)
    return IceDistributionParameters(parameters...)
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
3. **Compute N₀**: Intercept from mass normalization, retaining ``L`` if the
   mean-diameter limiter adjusts the represented number concentration

# Arguments

- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `rime_fraction`: Mass fraction of rime [-] (0 = unrimed, 1 = fully rimed)
- `rime_density`: Density of the rime layer [kg/m³]

# Keyword Arguments

- `mass`: Power law parameters (default: `IceMassPowerLaw()`)
- `closure`: Two-moment closure (default: `TwoMomentClosure()`)

# Returns

[`IceDistributionParameters`](@ref) with the PSD parameters and represented moments.

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
                                  mass = IceMassPowerLaw(typeof(L_ice)),
                                  closure = TwoMomentClosure(typeof(L_ice)),
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
    log_intercept = log(L_ice) - log_M_over_N₀
    N₀ = exp(log_intercept)
    number_concentration = exp(log_intercept + log_gamma_moment(μ, logλ))
    sixth_moment = exp(log_intercept + log_gamma_moment(μ, logλ; k = 6))

    return IceDistributionParameters(N₀, λ, μ, number_concentration, sixth_moment,
                                     log_intercept)
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
1. **L/N constraint**: Diagnoses an unconstrained λ at fixed μ ([`solve_lambda`](@ref))
2. **Diameter bounds and mass normalization**: Clamp λ and compute N₀ from ``L``
3. **Sixth-moment constraint**: Select the μ whose bounded, mass-normalized PSD
   matches ``Z`` ([`solve_shape_parameter`](@ref))

If a diameter limiter binds, the returned PSD preserves ``L`` and, where the μ bounds
permit, ``Z`` while adjusting the represented number concentration. If the requested
``Z`` is also outside the representable range, μ is pinned and the represented ``Z`` is
adjusted. Both adjusted moments are reported in [`IceDistributionParameters`](@ref).

# Advantages of Three-Moment

- Shape parameter μ evolves physically based on actual size distribution
- Better representation of size sorting during sedimentation
- Improved simulation of hail and large, heavily rimed particles
- No need for empirical μ-λ parameterization

# Arguments

- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `Z_ice`: Ice sixth moment [m⁶/m³]
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of the rime layer [kg/m³]

# Keyword Arguments

- `mass`: Power law parameters (default: `IceMassPowerLaw()`)
- `closure`: Three-moment closure (default: `ThreeMomentClosure()`)
- `diameter_bounds`: Bounds on mean particle diameter (default: P3's
  rime-fraction-dependent bounds)

# Returns

[`IceDistributionParameters`](@ref) with the PSD parameters and represented moments.

# Example

```julia
using Breeze.Microphysics.PredictedParticleProperties

# Ice with a sixth-moment constraint
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
                                  mass = IceMassPowerLaw(typeof(L_ice)),
                                  closure = ThreeMomentClosure(typeof(L_ice)),
                                  diameter_bounds = nothing)

    FT = typeof(L_ice)

    # Handle edge cases
    if N_ice <= 0 || L_ice <= 0
        return IceDistributionParameters(zero(FT), zero(FT), zero(FT))
    end

    # Fortran always applies Fr-dependent diameter bounds.
    bounds = isnothing(diameter_bounds) ? DiameterBounds(FT, rime_fraction) : diameter_bounds

    # Diagnose μ from the same physically bounded PSD that will be returned.
    μ = solve_shape_parameter(L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                              liquid_fraction, mass, closure, diameter_bounds = bounds)

    moments = distribution_moments_at_shape(L_ice, N_ice, Z_ice,
                                            rime_fraction, rime_density, μ, bounds;
                                            liquid_fraction, mass)
    N₀ = exp(moments.log_intercept)
    λ = moments.λ
    number_concentration = exp(moments.log_number)
    sixth_moment = exp(moments.log_sixth_moment)

    return IceDistributionParameters(N₀, λ, μ, number_concentration, sixth_moment,
                                     moments.log_intercept)
end
