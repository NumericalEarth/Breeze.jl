#####
##### Tabulation of P3 Integrals using TabulatedFunction pattern
#####
##### This file provides P3 integral tabulation using the same idioms as
##### Oceananigans.Utils.TabulatedFunction, but extended to 3D for the
##### (mean_particle_mass, rime_fraction, liquid_fraction) parameter space.
#####

export tabulate, TabulationParameters, P3IntegralEvaluator

using Adapt
using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: CPU, device, on_architecture

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
"""
function P3IntegralEvaluator(integral::AbstractP3Integral,
                              FT::Type{<:AbstractFloat} = Float64;
                              number_of_quadrature_points::Int = 64,
                              pure_ice_density = FT(917),
                              unrimed_density_factor = FT(0.1))

    nodes, weights = chebyshev_gauss_nodes_weights(FT, number_of_quadrature_points)

    return P3IntegralEvaluator(
        integral,
        nodes,
        weights,
        FT(pure_ice_density),
        FT(unrimed_density_factor)
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
                                           rime_density = typeof(log_mean_mass)(400),
                                           shape_parameter = zero(typeof(log_mean_mass)))
    FT = typeof(log_mean_mass)
    mean_particle_mass = FT(10)^log_mean_mass

    # Build the ice size distribution state from physical quantities
    state = state_from_mean_particle_mass(e, mean_particle_mass, rime_fraction, liquid_fraction;
                                          rime_density, shape_parameter)

    # Evaluate integral using pre-computed quadrature
    return evaluate_quadrature(e.integral, state, e.nodes, e.weights)
end

"""
    state_from_mean_particle_mass(evaluator, mean_particle_mass, rime_fraction, liquid_fraction; kwargs...)

Create an `IceSizeDistributionState` from physical quantities.

Given mean particle mass = qⁱ/Nⁱ (mass per particle), this function determines
the size distribution parameters (N₀, μ, λ).
"""
@inline function state_from_mean_particle_mass(e::P3IntegralEvaluator,
                                                mean_particle_mass,
                                                rime_fraction,
                                                liquid_fraction;
                                                rime_density = typeof(mean_particle_mass)(400),
                                                shape_parameter = zero(typeof(mean_particle_mass)))
    FT = typeof(mean_particle_mass)

    # Effective density: interpolate between aggregate and rime
    effective_density = (1 - rime_fraction) * e.pure_ice_density * e.unrimed_density_factor +
                        rime_fraction * rime_density

    # Characteristic diameter from mean_particle_mass = (π/6) ρ_eff D³
    characteristic_diameter = cbrt(6 * mean_particle_mass / (FT(π) * effective_density))

    # λ ~ 4 / D for exponential distribution (μ = 0)
    slope_parameter = FT(4) / max(characteristic_diameter, FT(1e-8))

    # N₀ from normalization (placeholder value for reasonable number concentration)
    intercept_parameter = FT(1e6)

    return IceSizeDistributionState(
        intercept_parameter,
        shape_parameter,
        slope_parameter,
        rime_fraction,
        liquid_fraction,
        rime_density
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

    for i in 1:n
        x = @inbounds nodes[i]
        w = @inbounds weights[i]

        D = transform_to_diameter(x, λ)
        J = jacobian_diameter_transform(x, λ)
        f = integrand(integral, D, state)

        result += w * f * J
    end

    return result
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

    # 0-based indices
    i⁻ = Base.unsafe_trunc(Int, fractional_idx)
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

on_architecture(arch, f::TabulatedFunction3D) =
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
- `minimum_log_mean_particle_mass`: Minimum log₁₀(mass) [log kg], default -18
- `maximum_log_mean_particle_mass`: Maximum log₁₀(mass) [log kg], default -5
- `number_of_quadrature_points`: Quadrature points for filling table (default 64)

# References

Table structure follows `create_p3_lookupTable_1.f90` in P3-microphysics.
"""
function TabulationParameters(FT::Type{<:AbstractFloat} = Float64;
                               number_of_mass_points::Int = 50,
                               number_of_rime_fraction_points::Int = 4,
                               number_of_liquid_fraction_points::Int = 4,
                               minimum_log_mean_particle_mass = FT(-18),
                               maximum_log_mean_particle_mass = FT(-5),
                               number_of_quadrature_points::Int = 64)
    return TabulationParameters(
        number_of_mass_points,
        number_of_rime_fraction_points,
        number_of_liquid_fraction_points,
        FT(minimum_log_mean_particle_mass),
        FT(maximum_log_mean_particle_mass),
        number_of_quadrature_points
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
                                     number_of_quadrature_points = params.number_of_quadrature_points)

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
        fall_speed.fall_speed_coefficient,
        fall_speed.fall_speed_exponent,
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
                  kwargs...) where FT

    params = TabulationParameters(FT; kwargs...)

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

    else
        throw(ArgumentError("Unknown property to tabulate: $property. " *
                           "Supported: :ice_fall_speed, :ice_deposition"))
    end
end
