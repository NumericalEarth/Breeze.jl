#####
##### Tabulation of P3 Integrals using TabulatedFunction pattern
#####
##### This file provides P3 integral tabulation using the same idioms as
##### Oceananigans.Utils.TabulatedFunction, but extended to 3D for the
##### (mean_particle_mass, rime_fraction, liquid_fraction) parameter space.
#####

export tabulate, TabulationParameters, P3IntegralEvaluator, TabulatedFunction1D,
       tabulated_function_1d

using Adapt: Adapt
using Oceananigans.Architectures: CPU, on_architecture

#####
##### P3IntegralEvaluator - callable struct for integral computation
#####

"""
    P3IntegralEvaluator{I, FT}

A callable struct that evaluates a P3 integral at any point in parameter space
using quadrature. This is the "function" that can be tabulated.

The evaluator is callable as `evaluator(log_mean_mass, rime_fraction, liquid_fraction)`
and returns the integral value.

# Fields
$(TYPEDFIELDS)

# Example

```julia
using Breeze.Microphysics.PredictedParticleProperties

# Create an evaluator for mass-weighted fall speed
evaluator = P3IntegralEvaluator(MassWeightedFallSpeed())

# Evaluate at a specific point (log₁₀ of mean mass, rime fraction, liquid fraction)
value = evaluator(-12.0, 0.5, 0.0)
```
"""
struct P3IntegralEvaluator{I<:AbstractP3Integral, N, W, FT}
    "The integral type being evaluated"
    integral :: I
    "Pre-computed quadrature nodes on [-1, 1]"
    nodes :: N
    "Pre-computed quadrature weights"
    weights :: W
    "Pure ice density [kg/m³]"
    pure_ice_density :: FT
    "Unrimed aggregate effective density factor"
    unrimed_density_factor :: FT
    "Fixed shape parameter override (NaN = use P3Closure, 0 = exponential PSD)"
    shape_parameter_override :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a `P3IntegralEvaluator` for the given integral type.

The evaluator pre-computes quadrature nodes and weights for efficient
repeated evaluation during tabulation.

# Keyword Arguments
- `number_of_quadrature_points`: Number of quadrature points (default 64)
- `pure_ice_density`: Pure ice density [kg/m³] (default 917)
- `unrimed_density_factor`: Effective density factor for unrimed aggregates (default 0.1)
- `shape_parameter_override`: Fixed μ for PSD (default NaN = use P3Closure; 0 = exponential)
"""
function P3IntegralEvaluator(integral::AbstractP3Integral,
                              FT::Type{<:AbstractFloat} = Float64;
                              number_of_quadrature_points::Int = 64,
                              pure_ice_density = FT(917),
                              unrimed_density_factor = FT(0.1),
                              shape_parameter_override = FT(NaN))

    nodes, weights = chebyshev_gauss_nodes_weights(FT, number_of_quadrature_points)

    return P3IntegralEvaluator(
        integral,
        nodes,
        weights,
        FT(pure_ice_density),
        FT(unrimed_density_factor),
        FT(shape_parameter_override)
    )
end

"""
    (evaluator::P3IntegralEvaluator)(log_mean_mass, rime_fraction, liquid_fraction)

Evaluate the P3 integral at the given parameter point.

# Arguments
- `log_mean_mass`: log₁₀ of mean particle mass [kg]
- `rime_fraction`: Rime mass fraction [0, 1]
- `liquid_fraction`: Liquid water fraction [0, 1]

# Returns
The evaluated integral value.
"""
@inline function (e::P3IntegralEvaluator)(log_mean_mass, rime_fraction, liquid_fraction;
                                           rime_density = typeof(log_mean_mass)(400))
    # NOTE (H2): rime_density is fixed at 400 kg/m³ for all tabulated evaluations.
    # Fortran P3 has a 4th table dimension over 5 rime densities (50, 250, 450,
    # 650, 900 kg/m³). This single-value approximation introduces 10-30% error
    # for dense graupel (ρ_r ≈ 900 kg/m³). A TabulatedFunction4D with the full
    # rime density axis is planned for a future PR.
    FT = typeof(log_mean_mass)
    mean_particle_mass = FT(10)^log_mean_mass

    # Build the ice size distribution state from physical quantities
    state = state_from_mean_particle_mass(e, mean_particle_mass, rime_fraction, liquid_fraction;
                                          rime_density)

    # Evaluate integral using pre-computed quadrature
    raw = evaluate_quadrature(e.integral, state, e.nodes, e.weights)

    # Normalize for tabulation (fall speeds become ratios, etc.)
    result = normalize_integral(e.integral, raw, mean_particle_mass, state, e.nodes, e.weights)

    # Safety: replace NaN/Inf with zero (can happen at extreme mass bounds)
    return ifelse(isfinite(result), result, zero(FT))
end

"""
    state_from_mean_particle_mass(evaluator, mean_particle_mass, rime_fraction, liquid_fraction; kwargs...)

Create an `IceSizeDistributionState` from physical quantities.

Given mean particle mass = qⁱ/Nⁱ (mass per particle), this function determines
the size distribution parameters (N₀, μ, λ) using the two-moment lambda solver.
The convention is N_ice = 1, so L_ice = mean_particle_mass. This ensures that
the raw integral ∫ f(D) N'(D) dD gives per-particle values.
"""
@inline function state_from_mean_particle_mass(e::P3IntegralEvaluator,
                                                mean_particle_mass,
                                                rime_fraction,
                                                liquid_fraction;
                                                rime_density = typeof(mean_particle_mass)(400),
                                                air_density = typeof(mean_particle_mass)(P3_REF_RHO))
    FT = typeof(mean_particle_mass)

    mass = IceMassPowerLaw(FT)

    # Use fixed shape parameter if override is set (non-NaN), else P3Closure.
    # This runs during table construction (CPU-side), not in GPU kernels.
    # The union type from the if/else is intentional — the performance cost is
    # negligible at construction time and avoids adding a type parameter to the
    # evaluator struct (which would complicate the tabulation API).
    μ_override = e.shape_parameter_override
    if isnan(μ_override)
        closure = P3Closure(FT)
    else
        closure = FixedShapeParameter(μ_override)
    end

    # Convention: N_ice = 1 particle. L_ice = mean_particle_mass.
    N_ice = one(FT)
    L_ice = max(mean_particle_mass, FT(1e-20))

    # Solve for (N₀, λ, μ) using the selected closure
    params = distribution_parameters(L_ice, N_ice, rime_fraction, rime_density;
                                      mass, closure)

    reference_air_density = FT(60000 / (287.15 * 253.15))

    return IceSizeDistributionState(
        params.N₀,
        params.μ,
        params.λ,
        rime_fraction,
        liquid_fraction,
        rime_density,
        mass.coefficient,
        mass.exponent,
        mass.ice_density,
        reference_air_density,
        air_density
    )
end

"""
    evaluate_quadrature(integral, state, nodes, weights)

Evaluate a P3 integral using pre-computed quadrature nodes and weights.
This is the core numerical integration routine.
"""
@inline function evaluate_quadrature(integral::AbstractP3Integral,
                                          state::IceSizeDistributionState,
                                          nodes, weights)
    FT = typeof(state.slope)
    λ = state.slope
    result = zero(FT)
    n = length(nodes)

    # Precompute regime thresholds once (independent of D)
    thresholds = regime_thresholds_from_state(FT, state)

    for i in 1:n
        x = @inbounds nodes[i]
        w = @inbounds weights[i]

        D = transform_to_diameter(x, λ)
        J = jacobian_diameter_transform(x, λ)
        f = integrand(integral, D, state, thresholds)

        result += w * f * J
    end

    return result
end

"""
    evaluate_quadrature(::AggregationNumber, state, nodes, weights)

Specialized double-integral evaluator for aggregation following the Fortran P3 convention:

```math
N_{agg} = \\frac{1}{2} \\int\\int (\\sqrt{A_1} + \\sqrt{A_2})^2 |V_1 - V_2| N'(D_1) N'(D_2) \\, dD_1 \\, dD_2
```

The factor of 1/2 matches the Fortran P3 table convention, which sums over the
upper triangle (j > k) to count each particle pair once. This replaces the
Wisner (1972) single-integral approximation with the full collision kernel.
"""
function evaluate_quadrature(::AggregationNumber,
                             state::IceSizeDistributionState,
                             nodes, weights)
    FT = typeof(state.slope)
    λ = state.slope
    result = zero(FT)
    n = length(nodes)

    # Precompute regime thresholds once (independent of D)
    thresholds = regime_thresholds_from_state(FT, state)

    for i in 1:n
        x₁ = @inbounds nodes[i]
        w₁ = @inbounds weights[i]
        D₁ = transform_to_diameter(x₁, λ)
        J₁ = jacobian_diameter_transform(x₁, λ)
        V₁ = terminal_velocity(D₁, state, thresholds)
        A₁ = particle_area(D₁, state, thresholds)
        N₁ = size_distribution(D₁, state)

        for j in 1:n
            x₂ = @inbounds nodes[j]
            w₂ = @inbounds weights[j]
            D₂ = transform_to_diameter(x₂, λ)
            J₂ = jacobian_diameter_transform(x₂, λ)
            V₂ = terminal_velocity(D₂, state, thresholds)
            A₂ = particle_area(D₂, state, thresholds)
            N₂ = size_distribution(D₂, state)

            # Fortran kernel: (√A₁ + √A₂)² × |V₁ - V₂|
            kernel = (sqrt(A₁) + sqrt(A₂))^2 * abs(V₁ - V₂)
            result += w₁ * w₂ * kernel * N₁ * N₂ * J₁ * J₂
        end
    end

    # Factor of 1/2 for self-collection: each pair (D₁, D₂) counted once.
    # Matches Fortran upper-triangle summation convention.
    return result * FT(0.5)
end

#####
##### Integral normalization for tabulation
#####
##### Tables store physically meaningful quantities:
##### - Fall speeds: actual velocities [m/s] (divide by appropriate moment)
##### - Effective radius: 3 × mass_integral / (4 × area_integral × ρ_ice)
##### - Mean diameter/density: mass-weighted (divide by mass integral)
##### - Per-particle integrals: ventilation, collection (N_ice = 1 → raw is per-particle)
#####

"""
    normalize_integral(integral, raw, mean_particle_mass, state, nodes, weights)

Normalize a raw quadrature integral for tabulation.

With the N_ice = 1 convention, different integral types require different
normalization to produce physically meaningful table values:

- **Number-weighted fall speed**: ∫ V N' dD / N = raw (since N = 1)
- **Mass-weighted fall speed**: ∫ V m N' dD / ∫ m N' dD (divide by mass integral)
- **Reflectivity-weighted fall speed**: ∫ V D⁶ N' dD / ∫ D⁶ N' dD
- **Effective radius**: 3 × ∫ m N' dD / (4 × ρ_ice × ∫ A N' dD)
- **Mean diameter**: ∫ D m N' dD / ∫ m N' dD
- **Mean density**: ∫ ρ m N' dD / ∫ m N' dD
- **All other integrals**: raw (per-particle since N = 1)
"""
normalize_integral(::AbstractP3Integral, raw, mean_particle_mass, state, nodes, weights) = raw

function normalize_integral(::MassWeightedFallSpeed, raw, mean_particle_mass, state, nodes, weights)
    mass_integral = evaluate_quadrature(MassMomentLambdaLimit(), state, nodes, weights)
    return raw / max(mass_integral, eps(typeof(raw)))
end

function normalize_integral(::ReflectivityWeightedFallSpeed, raw, mean_particle_mass, state, nodes, weights)
    # Reflectivity-weighted fall speed uses D⁶ weighting (not mass-squared Rayleigh)
    sixth_moment = evaluate_quadrature(SixthMomentRime(), state, nodes, weights)
    return raw / max(sixth_moment, eps(typeof(raw)))
end

function normalize_integral(::EffectiveRadius, raw_mass, mean_particle_mass, state, nodes, weights)
    FT = typeof(raw_mass)
    # raw_mass = ∫ m(D) N'(D) dD (from integrand)
    # Compute area integral: ∫ A(D) N'(D) dD
    # Use RainCollectionNumber integrand structure (V×A×N') minus V,
    # or compute inline since we don't have a dedicated AreaMoment type
    λ = state.slope
    n = length(nodes)
    area_integral = zero(FT)

    # Precompute regime thresholds once (independent of D)
    thresholds = regime_thresholds_from_state(FT, state)

    for i in 1:n
        x = @inbounds nodes[i]
        w = @inbounds weights[i]
        D = transform_to_diameter(x, λ)
        J = jacobian_diameter_transform(x, λ)
        A = particle_area(D, state, thresholds)
        Np = size_distribution(D, state)
        area_integral += w * A * Np * J
    end
    ρ_ice = FT(917)  # Pure ice density [kg/m³]
    return 3 * raw_mass / (4 * ρ_ice * max(area_integral, eps(FT)))
end

function normalize_integral(::MeanDiameter, raw, mean_particle_mass, state, nodes, weights)
    mass_integral = evaluate_quadrature(MassMomentLambdaLimit(), state, nodes, weights)
    return raw / max(mass_integral, eps(typeof(raw)))
end

function normalize_integral(::MeanDensity, raw, mean_particle_mass, state, nodes, weights)
    mass_integral = evaluate_quadrature(MassMomentLambdaLimit(), state, nodes, weights)
    return raw / max(mass_integral, eps(typeof(raw)))
end

#####
##### TabulatedFunction3D - 3D extension of TabulatedFunction pattern
#####

"""
    TabulatedFunction3D{F, T, FT}

A wrapper around a ternary callable `func(x, y, z)` that precomputes values in a
3D lookup table for fast trilinear interpolation. This extends the
`Oceananigans.Utils.TabulatedFunction` pattern to three dimensions.

The P3 scheme uses this for efficient integral evaluation during simulation,
avoiding expensive quadrature computations in GPU kernels.

# Fields
$(TYPEDFIELDS)
"""
struct TabulatedFunction3D{F, T, FT}
    "The original callable being tabulated (for reference/fallback)"
    func :: F
    "Precomputed values (3D array)"
    table :: T
    "Minimum x value (log mean particle mass)"
    x_min :: FT
    "Maximum x value"
    x_max :: FT
    "Inverse spacing in x"
    inverse_Δx :: FT
    "Minimum y value (rime fraction)"
    y_min :: FT
    "Maximum y value"
    y_max :: FT
    "Inverse spacing in y"
    inverse_Δy :: FT
    "Minimum z value (liquid fraction)"
    z_min :: FT
    "Maximum z value"
    z_max :: FT
    "Inverse spacing in z"
    inverse_Δz :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a `TabulatedFunction3D` by precomputing values of `func` over a 3D grid.

# Arguments
- `func`: Callable `func(x, y, z)` to tabulate
- `arch`: Architecture (`CPU()` or `GPU()`)
- `FT`: Float type

# Keyword Arguments
- `x_range`: Tuple `(x_min, x_max)` for first dimension (log mean mass)
- `y_range`: Tuple `(y_min, y_max)` for second dimension (rime fraction)
- `z_range`: Tuple `(z_min, z_max)` for third dimension (liquid fraction)
- `x_points`: Number of grid points in x (default 50)
- `y_points`: Number of grid points in y (default 4)
- `z_points`: Number of grid points in z (default 4)
"""
function TabulatedFunction3D(func, arch=CPU(), FT=Float64;
                              x_range,
                              y_range = (FT(0), FT(1)),
                              z_range = (FT(0), FT(1)),
                              x_points = 50,
                              y_points = 4,
                              z_points = 4)

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    Δx = (x_max - x_min) / (x_points - 1)
    Δy = (y_max - y_min) / max(y_points - 1, 1)
    Δz = (z_max - z_min) / max(z_points - 1, 1)

    inverse_Δx = 1 / Δx
    inverse_Δy = ifelse(y_points > 1, 1 / Δy, zero(FT))
    inverse_Δz = ifelse(z_points > 1, 1 / Δz, zero(FT))

    # Precompute table values on CPU first
    table = zeros(FT, x_points, y_points, z_points)

    for k in 1:z_points
        z = z_min + (k - 1) * Δz
        for j in 1:y_points
            y = y_min + (j - 1) * Δy
            for i in 1:x_points
                x = x_min + (i - 1) * Δx
                table[i, j, k] = func(x, y, z)
            end
        end
    end

    # Transfer to target architecture
    table = on_architecture(arch, table)

    return TabulatedFunction3D(
        func,
        table,
        convert(FT, x_min), convert(FT, x_max), convert(FT, inverse_Δx),
        convert(FT, y_min), convert(FT, y_max), convert(FT, inverse_Δy),
        convert(FT, z_min), convert(FT, z_max), convert(FT, inverse_Δz)
    )
end

#####
##### Trilinear interpolation for TabulatedFunction3D
#####

@inline function _clamp_and_index(val, v_min, v_max, inverse_Δv, n)
    v_clamped = clamp(val, v_min, v_max)
    fractional_idx = (v_clamped - v_min) * inverse_Δv

    # 0-based indices (guard against -ε from FP rounding when val == v_min)
    i⁻ = max(Base.unsafe_trunc(Int, fractional_idx), 0)
    i⁺ = min(i⁻ + 1, n - 1)
    ξ = fractional_idx - i⁻

    # Convert to 1-based
    return i⁻ + 1, i⁺ + 1, ξ
end

"""
    (f::TabulatedFunction3D)(x, y, z)

Evaluate the tabulated function using trilinear interpolation.
"""
@inline function (f::TabulatedFunction3D)(x, y, z)
    nx, ny, nz = size(f.table)

    i⁻, i⁺, ξx = _clamp_and_index(x, f.x_min, f.x_max, f.inverse_Δx, nx)
    j⁻, j⁺, ξy = _clamp_and_index(y, f.y_min, f.y_max, f.inverse_Δy, ny)
    k⁻, k⁺, ξz = _clamp_and_index(z, f.z_min, f.z_max, f.inverse_Δz, nz)

    # Trilinear interpolation
    @inbounds begin
        c000 = f.table[i⁻, j⁻, k⁻]
        c100 = f.table[i⁺, j⁻, k⁻]
        c010 = f.table[i⁻, j⁺, k⁻]
        c110 = f.table[i⁺, j⁺, k⁻]
        c001 = f.table[i⁻, j⁻, k⁺]
        c101 = f.table[i⁺, j⁻, k⁺]
        c011 = f.table[i⁻, j⁺, k⁺]
        c111 = f.table[i⁺, j⁺, k⁺]
    end

    # Interpolate in x
    c00 = (1 - ξx) * c000 + ξx * c100
    c10 = (1 - ξx) * c010 + ξx * c110
    c01 = (1 - ξx) * c001 + ξx * c101
    c11 = (1 - ξx) * c011 + ξx * c111

    # Interpolate in y
    c0 = (1 - ξy) * c00 + ξy * c10
    c1 = (1 - ξy) * c01 + ξy * c11

    # Interpolate in z
    return (1 - ξz) * c0 + ξz * c1
end

#####
##### GPU/architecture support for TabulatedFunction3D
#####

Oceananigans.Architectures.on_architecture(arch, f::TabulatedFunction3D) =
    TabulatedFunction3D(f.func,
                        on_architecture(arch, f.table),
                        f.x_min, f.x_max, f.inverse_Δx,
                        f.y_min, f.y_max, f.inverse_Δy,
                        f.z_min, f.z_max, f.inverse_Δz)

Adapt.adapt_structure(to, f::TabulatedFunction3D) =
    TabulatedFunction3D(nothing,
                        Adapt.adapt(to, f.table),
                        f.x_min, f.x_max, f.inverse_Δx,
                        f.y_min, f.y_max, f.inverse_Δy,
                        f.z_min, f.z_max, f.inverse_Δz)

#####
##### Pretty printing
#####

function Base.summary(f::TabulatedFunction3D)
    nx, ny, nz = size(f.table)
    return "TabulatedFunction3D with $(nx)×$(ny)×$(nz) points"
end

function Base.show(io::IO, f::TabulatedFunction3D)
    print(io, summary(f))
    print(io, " over x∈[$(f.x_min), $(f.x_max)], y∈[$(f.y_min), $(f.y_max)], z∈[$(f.z_min), $(f.z_max)]")
    if f.func !== nothing
        print(io, " of ", typeof(f.func).name.name)
    end
end

#####
##### TabulatedFunction1D - 1D lookup table for rain integrals
#####

"""
    TabulatedFunction1D{T, FT}

A 1D lookup table for fast linear interpolation of a scalar function of one
variable. Used for rain PSD integrals tabulated over `log10(λ_r)`.

Queries outside `[x_min, x_max]` are clamped to the boundary (no extrapolation
errors).

# Fields
$(TYPEDFIELDS)
"""
struct TabulatedFunction1D{T, FT}
    "Precomputed values (1D array)"
    table :: T
    "Minimum x value"
    x_min :: FT
    "Maximum x value"
    x_max :: FT
    "Inverse grid spacing"
    inverse_Δx :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a `TabulatedFunction1D` by pre-evaluating `func` on a uniform grid.

# Arguments
- `func`: Callable `func(x)` to tabulate
- `arch`: Architecture (`CPU()` or `GPU()`)
- `FT`: Floating-point type

# Keyword Arguments
- `x_range`: `(x_min, x_max)` interval
- `x_points`: Number of grid points (default 200)
"""
function TabulatedFunction1D(func, arch=CPU(), FT=Float64;
                              x_range,
                              x_points::Int = 200)
    x_points >= 2 || throw(ArgumentError("x_points must be >= 2, got $x_points"))
    x_min, x_max = FT(x_range[1]), FT(x_range[2])
    Δx = (x_max - x_min) / (x_points - 1)
    inverse_Δx = 1 / Δx

    table = zeros(FT, x_points)
    for i in 1:x_points
        x = x_min + (i - 1) * Δx
        table[i] = func(x)
    end

    table = on_architecture(arch, table)

    return TabulatedFunction1D(table, x_min, x_max, inverse_Δx)
end

"""
    tabulated_function_1d(values, x_min, x_max, inverse_Δx)

Construct a `TabulatedFunction1D` directly from a pre-filled values array.
This low-level constructor is used in tests.
"""
function tabulated_function_1d(values::AbstractVector, x_min, x_max, inverse_Δx)
    FT = eltype(values)
    return TabulatedFunction1D(values, FT(x_min), FT(x_max), FT(inverse_Δx))
end

"""
    (f::TabulatedFunction1D)(x)

Evaluate the tabulated function using linear interpolation. Values outside
`[x_min, x_max]` are clamped to the boundary value.
"""
@inline function (f::TabulatedFunction1D)(x)
    n = length(f.table)
    x_clamped = clamp(x, f.x_min, f.x_max)
    fractional_idx = (x_clamped - f.x_min) * f.inverse_Δx

    # 0-based integer index (guard against -ε from FP rounding)
    i⁻ = max(Base.unsafe_trunc(Int, fractional_idx), 0)
    i⁺ = min(i⁻ + 1, n - 1)
    ξ = fractional_idx - i⁻

    # 1-based lookup
    @inbounds c⁻ = f.table[i⁻ + 1]
    @inbounds c⁺ = f.table[i⁺ + 1]

    return (1 - ξ) * c⁻ + ξ * c⁺
end

#####
##### GPU/architecture support for TabulatedFunction1D
#####

Oceananigans.Architectures.on_architecture(arch, f::TabulatedFunction1D) =
    TabulatedFunction1D(on_architecture(arch, f.table),
                        f.x_min, f.x_max, f.inverse_Δx)

Adapt.adapt_structure(to, f::TabulatedFunction1D) =
    TabulatedFunction1D(Adapt.adapt(to, f.table),
                        f.x_min, f.x_max, f.inverse_Δx)

function Base.summary(f::TabulatedFunction1D)
    n = length(f.table)
    return "TabulatedFunction1D with $n points over [$(f.x_min), $(f.x_max)]"
end

Base.show(io::IO, f::TabulatedFunction1D) = print(io, summary(f))

#####
##### TabulationParameters
#####

"""
    TabulationParameters{FT}

Configuration for P3 integral tabulation. See constructor for details.
"""
struct TabulationParameters{FT}
    number_of_mass_points :: Int
    number_of_rime_fraction_points :: Int
    number_of_liquid_fraction_points :: Int
    minimum_log_mean_particle_mass :: FT
    maximum_log_mean_particle_mass :: FT
    number_of_quadrature_points :: Int
    shape_parameter_override :: FT
end

"""
$(TYPEDSIGNATURES)

Configure the lookup table grid for P3 integrals.

The P3 Fortran code pre-computes bulk integrals on a 3D grid indexed by:

1. **Log mean particle mass** `log₁₀(qⁱ/Nⁱ)` [log kg]: Mass per particle (linearly spaced in log)
2. **Rime fraction** `∈ [0, 1]`: Mass fraction that is rime (frozen accretion)
3. **Liquid fraction** `∈ [0, 1]`: Mass fraction that is liquid water on ice

During simulation, integral values are interpolated from this table rather
than computed via quadrature, which is much faster.

# Keyword Arguments

- `number_of_mass_points`: Grid points in mean particle mass (default 50)
- `number_of_rime_fraction_points`: Grid points in rime fraction (default 4)
- `number_of_liquid_fraction_points`: Grid points in liquid fraction (default 4)
- `minimum_log_mean_particle_mass`: Minimum log₁₀(mass) [log kg], default -15
- `maximum_log_mean_particle_mass`: Maximum log₁₀(mass) [log kg], default -5
- `number_of_quadrature_points`: Quadrature points for filling table (default 64)
- `shape_parameter_override`: Fixed μ for PSD (default NaN = use P3Closure; 0 = exponential)

# References

Table structure follows `create_p3_lookupTable_1.f90` in P3-microphysics.
"""
function TabulationParameters(FT::Type{<:AbstractFloat} = Float64;
                               number_of_mass_points::Int = 150,
                               number_of_rime_fraction_points::Int = 8,
                               number_of_liquid_fraction_points::Int = 4,
                               minimum_log_mean_particle_mass = FT(-17.3),
                               maximum_log_mean_particle_mass = FT(-5.3),
                               number_of_quadrature_points::Int = 64,
                               shape_parameter_override = FT(NaN))
    return TabulationParameters(
        number_of_mass_points,
        number_of_rime_fraction_points,
        number_of_liquid_fraction_points,
        FT(minimum_log_mean_particle_mass),
        FT(maximum_log_mean_particle_mass),
        number_of_quadrature_points,
        FT(shape_parameter_override)
    )
end

#####
##### Main tabulate interface
#####

"""
$(TYPEDSIGNATURES)

Tabulate a P3 integral using the `TabulatedFunction3D` pattern.

Creates a callable evaluator function and tabulates it over the 3D parameter space.

# Arguments
- `integral`: Integral type to tabulate (e.g., `MassWeightedFallSpeed()`)
- `arch`: `CPU()` or `GPU()` - determines where table is stored
- `params`: [`TabulationParameters`](@ref) defining the grid

# Returns
A [`TabulatedFunction3D`](@ref) that can be called like the original evaluator.

# Example

```julia
using Oceananigans
using Breeze.Microphysics.PredictedParticleProperties

# Create and tabulate a fall speed integral
params = TabulationParameters()
tabulated = tabulate(MassWeightedFallSpeed(), CPU(), params)

# Evaluate via interpolation (fast)
value = tabulated(-12.0, 0.5, 0.0)
```
"""
function tabulate(integral::AbstractP3Integral, arch=CPU(),
                  params::TabulationParameters = TabulationParameters())

    FT = typeof(params.minimum_log_mean_particle_mass)

    # Create the evaluator function
    evaluator = P3IntegralEvaluator(integral, FT;
                                     number_of_quadrature_points = params.number_of_quadrature_points,
                                     shape_parameter_override = params.shape_parameter_override)

    # Tabulate using TabulatedFunction3D
    return TabulatedFunction3D(evaluator, arch, FT;
                                x_range = (params.minimum_log_mean_particle_mass,
                                          params.maximum_log_mean_particle_mass),
                                y_range = (zero(FT), one(FT)),
                                z_range = (zero(FT), one(FT)),
                                x_points = params.number_of_mass_points,
                                y_points = params.number_of_rime_fraction_points,
                                z_points = params.number_of_liquid_fraction_points)
end

"""
$(TYPEDSIGNATURES)

Tabulate all integrals in an `IceFallSpeed` container.

Returns a new `IceFallSpeed` with `TabulatedFunction3D` fields.
"""
function tabulate(fall_speed::IceFallSpeed, arch=CPU(),
                  params::TabulationParameters = TabulationParameters())

    return IceFallSpeed(
        fall_speed.reference_air_density,
        tabulate(fall_speed.number_weighted, arch, params),
        tabulate(fall_speed.mass_weighted, arch, params),
        tabulate(fall_speed.reflectivity_weighted, arch, params)
    )
end

"""
$(TYPEDSIGNATURES)

Tabulate all integrals in an `IceDeposition` container.
"""
function tabulate(deposition::IceDeposition, arch=CPU(),
                  params::TabulationParameters = TabulationParameters())

    return IceDeposition(
        deposition.thermal_conductivity,
        deposition.vapor_diffusivity,
        tabulate(deposition.ventilation, arch, params),
        tabulate(deposition.ventilation_enhanced, arch, params),
        tabulate(deposition.small_ice_ventilation_constant, arch, params),
        tabulate(deposition.small_ice_ventilation_reynolds, arch, params),
        tabulate(deposition.large_ice_ventilation_constant, arch, params),
        tabulate(deposition.large_ice_ventilation_reynolds, arch, params)
    )
end

"""
$(TYPEDSIGNATURES)

Tabulate all integrals in an `IceBulkProperties` container.
"""
function tabulate(bulk::IceBulkProperties, arch=CPU(),
                  params::TabulationParameters = TabulationParameters())

    return IceBulkProperties(
        bulk.maximum_mean_diameter,
        bulk.minimum_mean_diameter,
        tabulate(bulk.effective_radius, arch, params),
        tabulate(bulk.mean_diameter, arch, params),
        tabulate(bulk.mean_density, arch, params),
        tabulate(bulk.reflectivity, arch, params),
        bulk.slope,  # diagnostic, not an integral
        bulk.shape,  # diagnostic, not an integral
        tabulate(bulk.shedding, arch, params)
    )
end

"""
$(TYPEDSIGNATURES)

Tabulate all integrals in an `IceCollection` container.
"""
function tabulate(collection::IceCollection, arch=CPU(),
                  params::TabulationParameters = TabulationParameters())

    return IceCollection(
        collection.ice_cloud_collection_efficiency,
        collection.ice_rain_collection_efficiency,
        tabulate(collection.aggregation, arch, params),
        tabulate(collection.rain_collection, arch, params)
    )
end

"""
$(TYPEDSIGNATURES)

Tabulate all integrals in an `IceSixthMoment` container.
"""
function tabulate(sixth::IceSixthMoment, arch=CPU(),
                  params::TabulationParameters = TabulationParameters())

    return IceSixthMoment(
        tabulate(sixth.rime, arch, params),
        tabulate(sixth.deposition, arch, params),
        tabulate(sixth.deposition1, arch, params),
        tabulate(sixth.melt1, arch, params),
        tabulate(sixth.melt2, arch, params),
        tabulate(sixth.shedding, arch, params),
        tabulate(sixth.aggregation, arch, params),
        tabulate(sixth.sublimation, arch, params),
        tabulate(sixth.sublimation1, arch, params)
    )
end

"""
$(TYPEDSIGNATURES)

Tabulate all integrals in an `IceLambdaLimiter` container.
"""
function tabulate(limiter::IceLambdaLimiter, arch=CPU(),
                  params::TabulationParameters = TabulationParameters())

    return IceLambdaLimiter(
        tabulate(limiter.small_q, arch, params),
        tabulate(limiter.large_q, arch, params)
    )
end

"""
$(TYPEDSIGNATURES)

Tabulate all integrals in an `IceRainCollection` container.
"""
function tabulate(ice_rain::IceRainCollection, arch=CPU(),
                  params::TabulationParameters = TabulationParameters())

    return IceRainCollection(
        tabulate(ice_rain.mass, arch, params),
        tabulate(ice_rain.number, arch, params),
        tabulate(ice_rain.sixth_moment, arch, params)
    )
end

"""
$(TYPEDSIGNATURES)

Tabulate all integrals in a `RainProperties` container.

Returns a new `RainProperties` with `TabulatedFunction1D` fields for
`velocity_mass`, `velocity_number`, and `evaporation`.

# Keyword Arguments
- `lambda_points`: Grid points in log10(λ_r) (default 200)
- `log_lambda_range`: `(log10_min, log10_max)` for λ_r (default `(2.5, 5.5)`)
- `quadrature_points`: Quadrature points used to fill the table (default 128)
"""
function tabulate(rain::RainProperties, arch=CPU(), FT=Float64;
                  lambda_points::Int = 200,
                  log_lambda_range = (FT(2.5), FT(5.5)),
                  quadrature_points::Int = 128)

    vel_mass_eval  = RainMassWeightedVelocityEvaluator(FT; n_points=quadrature_points)
    vel_num_eval   = RainNumberWeightedVelocityEvaluator(FT; n_points=quadrature_points)
    evap_eval      = RainEvaporationVentilationEvaluator(FT; n_points=quadrature_points)

    tab_vel_mass   = TabulatedFunction1D(vel_mass_eval,  arch, FT;
                                         x_range=log_lambda_range, x_points=lambda_points)
    tab_vel_num    = TabulatedFunction1D(vel_num_eval,   arch, FT;
                                         x_range=log_lambda_range, x_points=lambda_points)
    tab_evap       = TabulatedFunction1D(evap_eval,      arch, FT;
                                         x_range=log_lambda_range, x_points=lambda_points)

    return RainProperties(
        rain.maximum_mean_diameter,
        rain.fall_speed_coefficient,
        rain.fall_speed_exponent,
        rain.shape_parameter,
        tab_vel_num,
        tab_vel_mass,
        tab_evap
    )
end

"""
$(TYPEDSIGNATURES)

Tabulate specific integrals within a P3 microphysics scheme.

Returns a new `PredictedParticlePropertiesMicrophysics` with the specified
integrals replaced by `TabulatedFunction3D` lookup tables.

# Arguments
- `p3`: [`PredictedParticlePropertiesMicrophysics`](@ref)
- `property`: Which integrals to tabulate
  - `:ice_fall_speed`: All fall speed integrals
  - `:ice_deposition`: All deposition/ventilation integrals
- `arch`: `CPU()` or `GPU()`

# Keyword Arguments
Passed to [`TabulationParameters`](@ref): `number_of_mass_points`,
`number_of_rime_fraction_points`, etc.

# Example

```julia
using Oceananigans
using Breeze.Microphysics.PredictedParticleProperties

p3 = PredictedParticlePropertiesMicrophysics()
p3_fast = tabulate(p3, :ice_fall_speed, CPU(); number_of_mass_points=100)
```
"""
function tabulate(p3::PredictedParticlePropertiesMicrophysics{FT},
                  property::Symbol,
                  arch=CPU();
                  # Rain-specific kwargs (only used when property == :rain)
                  lambda_points::Int = 200,
                  log_lambda_range = (FT(2.5), FT(5.5)),
                  # Ice-specific kwargs (only used for ice properties)
                  number_of_mass_points::Int = 150,
                  number_of_rime_fraction_points::Int = 8,
                  number_of_liquid_fraction_points::Int = 4,
                  minimum_log_mean_particle_mass = FT(-17.3),
                  maximum_log_mean_particle_mass = FT(-5.3),
                  number_of_quadrature_points::Int = 64,
                  shape_parameter_override = FT(NaN),
                  # Shared kwargs
                  quadrature_points::Int = number_of_quadrature_points) where FT

    if property == :rain
        new_rain = tabulate(p3.rain, arch, FT;
                            lambda_points,
                            log_lambda_range,
                            quadrature_points)
        return PredictedParticlePropertiesMicrophysics(
            p3.water_density,
            p3.minimum_mass_mixing_ratio,
            p3.minimum_number_mixing_ratio,
            p3.ice,
            new_rain,
            p3.cloud,
            p3.process_rates,
            p3.precipitation_boundary_condition
        )
    end

    params = TabulationParameters(FT;
                 number_of_mass_points,
                 number_of_rime_fraction_points,
                 number_of_liquid_fraction_points,
                 minimum_log_mean_particle_mass,
                 maximum_log_mean_particle_mass,
                 number_of_quadrature_points,
                 shape_parameter_override)

    if property == :ice_fall_speed
        new_fall_speed = tabulate(p3.ice.fall_speed, arch, params)
        new_ice = IceProperties(
            p3.ice.minimum_rime_density,
            p3.ice.maximum_rime_density,
            p3.ice.maximum_shape_parameter,
            p3.ice.minimum_reflectivity,
            new_fall_speed,
            p3.ice.deposition,
            p3.ice.bulk_properties,
            p3.ice.collection,
            p3.ice.sixth_moment,
            p3.ice.lambda_limiter,
            p3.ice.ice_rain
        )
        return PredictedParticlePropertiesMicrophysics(
            p3.water_density,
            p3.minimum_mass_mixing_ratio,
            p3.minimum_number_mixing_ratio,
            new_ice,
            p3.rain,
            p3.cloud,
            p3.process_rates,
            p3.precipitation_boundary_condition
        )

    elseif property == :ice_deposition
        new_deposition = tabulate(p3.ice.deposition, arch, params)
        new_ice = IceProperties(
            p3.ice.minimum_rime_density,
            p3.ice.maximum_rime_density,
            p3.ice.maximum_shape_parameter,
            p3.ice.minimum_reflectivity,
            p3.ice.fall_speed,
            new_deposition,
            p3.ice.bulk_properties,
            p3.ice.collection,
            p3.ice.sixth_moment,
            p3.ice.lambda_limiter,
            p3.ice.ice_rain
        )
        return PredictedParticlePropertiesMicrophysics(
            p3.water_density,
            p3.minimum_mass_mixing_ratio,
            p3.minimum_number_mixing_ratio,
            new_ice,
            p3.rain,
            p3.cloud,
            p3.process_rates,
            p3.precipitation_boundary_condition
        )

    elseif property == :ice_bulk_properties
        new_bulk = tabulate(p3.ice.bulk_properties, arch, params)
        new_ice = IceProperties(
            p3.ice.minimum_rime_density,
            p3.ice.maximum_rime_density,
            p3.ice.maximum_shape_parameter,
            p3.ice.minimum_reflectivity,
            p3.ice.fall_speed,
            p3.ice.deposition,
            new_bulk,
            p3.ice.collection,
            p3.ice.sixth_moment,
            p3.ice.lambda_limiter,
            p3.ice.ice_rain
        )
        return PredictedParticlePropertiesMicrophysics(
            p3.water_density,
            p3.minimum_mass_mixing_ratio,
            p3.minimum_number_mixing_ratio,
            new_ice,
            p3.rain,
            p3.cloud,
            p3.process_rates,
            p3.precipitation_boundary_condition
        )

    elseif property == :ice_collection
        new_collection = tabulate(p3.ice.collection, arch, params)
        new_ice = IceProperties(
            p3.ice.minimum_rime_density,
            p3.ice.maximum_rime_density,
            p3.ice.maximum_shape_parameter,
            p3.ice.minimum_reflectivity,
            p3.ice.fall_speed,
            p3.ice.deposition,
            p3.ice.bulk_properties,
            new_collection,
            p3.ice.sixth_moment,
            p3.ice.lambda_limiter,
            p3.ice.ice_rain
        )
        return PredictedParticlePropertiesMicrophysics(
            p3.water_density,
            p3.minimum_mass_mixing_ratio,
            p3.minimum_number_mixing_ratio,
            new_ice,
            p3.rain,
            p3.cloud,
            p3.process_rates,
            p3.precipitation_boundary_condition
        )

    elseif property == :ice_sixth_moment
        new_sixth = tabulate(p3.ice.sixth_moment, arch, params)
        new_ice = IceProperties(
            p3.ice.minimum_rime_density,
            p3.ice.maximum_rime_density,
            p3.ice.maximum_shape_parameter,
            p3.ice.minimum_reflectivity,
            p3.ice.fall_speed,
            p3.ice.deposition,
            p3.ice.bulk_properties,
            p3.ice.collection,
            new_sixth,
            p3.ice.lambda_limiter,
            p3.ice.ice_rain
        )
        return PredictedParticlePropertiesMicrophysics(
            p3.water_density,
            p3.minimum_mass_mixing_ratio,
            p3.minimum_number_mixing_ratio,
            new_ice,
            p3.rain,
            p3.cloud,
            p3.process_rates,
            p3.precipitation_boundary_condition
        )

    else
        throw(ArgumentError("Unknown property to tabulate: $property. " *
                           "Supported: :ice_fall_speed, :ice_deposition, " *
                           ":ice_bulk_properties, :ice_collection, :ice_sixth_moment, :rain"))
    end
end

"""
$(TYPEDSIGNATURES)

Tabulate all ice integral properties for fast lookup during simulation.

This is a convenience function that tabulates fall speed, deposition,
and other integral properties in one call.

# Arguments
- `p3`: P3 microphysics scheme
- `arch`: Architecture (`CPU()` or `GPU()`)

# Keyword Arguments
Passed to [`TabulationParameters`](@ref).

# Returns
A new `PredictedParticlePropertiesMicrophysics` with all ice integrals tabulated.

# Example

```julia
using Oceananigans
using Breeze.Microphysics.PredictedParticleProperties

p3 = PredictedParticlePropertiesMicrophysics()
p3_tabulated = tabulate(p3, CPU())
```
"""
function tabulate(p3::PredictedParticlePropertiesMicrophysics{FT}, arch=CPU();
                  kwargs...) where FT

    params = TabulationParameters(FT; kwargs...)

    # Tabulate all ice integral categories
    tabulated_fall_speed    = tabulate(p3.ice.fall_speed, arch, params)
    tabulated_deposition    = tabulate(p3.ice.deposition, arch, params)
    tabulated_bulk          = tabulate(p3.ice.bulk_properties, arch, params)
    tabulated_collection    = tabulate(p3.ice.collection, arch, params)
    tabulated_sixth_moment  = tabulate(p3.ice.sixth_moment, arch, params)
    tabulated_lambda        = tabulate(p3.ice.lambda_limiter, arch, params)
    tabulated_ice_rain      = tabulate(p3.ice.ice_rain, arch, params)

    new_ice = IceProperties(
        p3.ice.minimum_rime_density,
        p3.ice.maximum_rime_density,
        p3.ice.maximum_shape_parameter,
        p3.ice.minimum_reflectivity,
        tabulated_fall_speed,
        tabulated_deposition,
        tabulated_bulk,
        tabulated_collection,
        tabulated_sixth_moment,
        tabulated_lambda,
        tabulated_ice_rain
    )

    # Tabulate rain integrals (use ice quadrature point count for consistency)
    tabulated_rain = tabulate(p3.rain, arch, FT;
                              lambda_points = 200,
                              quadrature_points = params.number_of_quadrature_points)

    return PredictedParticlePropertiesMicrophysics(
        p3.water_density,
        p3.minimum_mass_mixing_ratio,
        p3.minimum_number_mixing_ratio,
        new_ice,
        tabulated_rain,
        p3.cloud,
        p3.process_rates,
        p3.precipitation_boundary_condition
    )
end

#####
##### GPU/architecture support for P3 container structs
#####
##### When ice/rain integrals are tabulated (TabulatedFunction3D / TabulatedFunction1D),
##### the lookup table arrays must be transferred to the GPU. Scalar fields and
##### singleton integral types pass through unchanged.
#####

# --- IceFallSpeed ---

Adapt.adapt_structure(to, x::IceFallSpeed) =
    IceFallSpeed(x.reference_air_density,
                 Adapt.adapt(to, x.number_weighted),
                 Adapt.adapt(to, x.mass_weighted),
                 Adapt.adapt(to, x.reflectivity_weighted))

Oceananigans.Architectures.on_architecture(arch, x::IceFallSpeed) =
    IceFallSpeed(x.reference_air_density,
                 on_architecture(arch, x.number_weighted),
                 on_architecture(arch, x.mass_weighted),
                 on_architecture(arch, x.reflectivity_weighted))

# --- IceDeposition ---

Adapt.adapt_structure(to, x::IceDeposition) =
    IceDeposition(x.thermal_conductivity,
                  x.vapor_diffusivity,
                  Adapt.adapt(to, x.ventilation),
                  Adapt.adapt(to, x.ventilation_enhanced),
                  Adapt.adapt(to, x.small_ice_ventilation_constant),
                  Adapt.adapt(to, x.small_ice_ventilation_reynolds),
                  Adapt.adapt(to, x.large_ice_ventilation_constant),
                  Adapt.adapt(to, x.large_ice_ventilation_reynolds))

Oceananigans.Architectures.on_architecture(arch, x::IceDeposition) =
    IceDeposition(x.thermal_conductivity,
                  x.vapor_diffusivity,
                  on_architecture(arch, x.ventilation),
                  on_architecture(arch, x.ventilation_enhanced),
                  on_architecture(arch, x.small_ice_ventilation_constant),
                  on_architecture(arch, x.small_ice_ventilation_reynolds),
                  on_architecture(arch, x.large_ice_ventilation_constant),
                  on_architecture(arch, x.large_ice_ventilation_reynolds))

# --- IceBulkProperties ---

Adapt.adapt_structure(to, x::IceBulkProperties) =
    IceBulkProperties(x.maximum_mean_diameter,
                      x.minimum_mean_diameter,
                      Adapt.adapt(to, x.effective_radius),
                      Adapt.adapt(to, x.mean_diameter),
                      Adapt.adapt(to, x.mean_density),
                      Adapt.adapt(to, x.reflectivity),
                      Adapt.adapt(to, x.slope),
                      Adapt.adapt(to, x.shape),
                      Adapt.adapt(to, x.shedding))

Oceananigans.Architectures.on_architecture(arch, x::IceBulkProperties) =
    IceBulkProperties(x.maximum_mean_diameter,
                      x.minimum_mean_diameter,
                      on_architecture(arch, x.effective_radius),
                      on_architecture(arch, x.mean_diameter),
                      on_architecture(arch, x.mean_density),
                      on_architecture(arch, x.reflectivity),
                      on_architecture(arch, x.slope),
                      on_architecture(arch, x.shape),
                      on_architecture(arch, x.shedding))

# --- IceCollection ---

Adapt.adapt_structure(to, x::IceCollection) =
    IceCollection(x.ice_cloud_collection_efficiency,
                  x.ice_rain_collection_efficiency,
                  Adapt.adapt(to, x.aggregation),
                  Adapt.adapt(to, x.rain_collection))

Oceananigans.Architectures.on_architecture(arch, x::IceCollection) =
    IceCollection(x.ice_cloud_collection_efficiency,
                  x.ice_rain_collection_efficiency,
                  on_architecture(arch, x.aggregation),
                  on_architecture(arch, x.rain_collection))

# --- IceSixthMoment ---

Adapt.adapt_structure(to, x::IceSixthMoment) =
    IceSixthMoment(Adapt.adapt(to, x.rime),
                   Adapt.adapt(to, x.deposition),
                   Adapt.adapt(to, x.deposition1),
                   Adapt.adapt(to, x.melt1),
                   Adapt.adapt(to, x.melt2),
                   Adapt.adapt(to, x.shedding),
                   Adapt.adapt(to, x.aggregation),
                   Adapt.adapt(to, x.sublimation),
                   Adapt.adapt(to, x.sublimation1))

Oceananigans.Architectures.on_architecture(arch, x::IceSixthMoment) =
    IceSixthMoment(on_architecture(arch, x.rime),
                   on_architecture(arch, x.deposition),
                   on_architecture(arch, x.deposition1),
                   on_architecture(arch, x.melt1),
                   on_architecture(arch, x.melt2),
                   on_architecture(arch, x.shedding),
                   on_architecture(arch, x.aggregation),
                   on_architecture(arch, x.sublimation),
                   on_architecture(arch, x.sublimation1))

# --- IceLambdaLimiter ---

Adapt.adapt_structure(to, x::IceLambdaLimiter) =
    IceLambdaLimiter(Adapt.adapt(to, x.small_q),
                     Adapt.adapt(to, x.large_q))

Oceananigans.Architectures.on_architecture(arch, x::IceLambdaLimiter) =
    IceLambdaLimiter(on_architecture(arch, x.small_q),
                     on_architecture(arch, x.large_q))

# --- IceRainCollection ---

Adapt.adapt_structure(to, x::IceRainCollection) =
    IceRainCollection(Adapt.adapt(to, x.mass),
                      Adapt.adapt(to, x.number),
                      Adapt.adapt(to, x.sixth_moment))

Oceananigans.Architectures.on_architecture(arch, x::IceRainCollection) =
    IceRainCollection(on_architecture(arch, x.mass),
                      on_architecture(arch, x.number),
                      on_architecture(arch, x.sixth_moment))

# --- IceProperties ---

Adapt.adapt_structure(to, x::IceProperties) =
    IceProperties(x.minimum_rime_density,
                  x.maximum_rime_density,
                  x.maximum_shape_parameter,
                  x.minimum_reflectivity,
                  Adapt.adapt(to, x.fall_speed),
                  Adapt.adapt(to, x.deposition),
                  Adapt.adapt(to, x.bulk_properties),
                  Adapt.adapt(to, x.collection),
                  Adapt.adapt(to, x.sixth_moment),
                  Adapt.adapt(to, x.lambda_limiter),
                  Adapt.adapt(to, x.ice_rain))

Oceananigans.Architectures.on_architecture(arch, x::IceProperties) =
    IceProperties(x.minimum_rime_density,
                  x.maximum_rime_density,
                  x.maximum_shape_parameter,
                  x.minimum_reflectivity,
                  on_architecture(arch, x.fall_speed),
                  on_architecture(arch, x.deposition),
                  on_architecture(arch, x.bulk_properties),
                  on_architecture(arch, x.collection),
                  on_architecture(arch, x.sixth_moment),
                  on_architecture(arch, x.lambda_limiter),
                  on_architecture(arch, x.ice_rain))

# --- RainProperties ---

Adapt.adapt_structure(to, x::RainProperties) =
    RainProperties(x.maximum_mean_diameter,
                   x.fall_speed_coefficient,
                   x.fall_speed_exponent,
                   Adapt.adapt(to, x.shape_parameter),
                   Adapt.adapt(to, x.velocity_number),
                   Adapt.adapt(to, x.velocity_mass),
                   Adapt.adapt(to, x.evaporation))

Oceananigans.Architectures.on_architecture(arch, x::RainProperties) =
    RainProperties(x.maximum_mean_diameter,
                   x.fall_speed_coefficient,
                   x.fall_speed_exponent,
                   on_architecture(arch, x.shape_parameter),
                   on_architecture(arch, x.velocity_number),
                   on_architecture(arch, x.velocity_mass),
                   on_architecture(arch, x.evaporation))

# --- PredictedParticlePropertiesMicrophysics ---

Adapt.adapt_structure(to, x::PredictedParticlePropertiesMicrophysics) =
    PredictedParticlePropertiesMicrophysics(
        x.water_density,
        x.minimum_mass_mixing_ratio,
        x.minimum_number_mixing_ratio,
        Adapt.adapt(to, x.ice),
        Adapt.adapt(to, x.rain),
        x.cloud,
        x.process_rates,
        Adapt.adapt(to, x.precipitation_boundary_condition))

Oceananigans.Architectures.on_architecture(arch, x::PredictedParticlePropertiesMicrophysics) =
    PredictedParticlePropertiesMicrophysics(
        x.water_density,
        x.minimum_mass_mixing_ratio,
        x.minimum_number_mixing_ratio,
        on_architecture(arch, x.ice),
        on_architecture(arch, x.rain),
        x.cloud,
        x.process_rates,
        on_architecture(arch, x.precipitation_boundary_condition))
