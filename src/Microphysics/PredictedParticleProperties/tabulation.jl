#####
##### Tabulation of P3 Integrals
#####
##### Generate lookup tables for efficient evaluation during simulation.
##### Tables are indexed by mean particle mass, rime fraction, and liquid fraction.
#####

export tabulate, TabulationParameters

using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: device, on_architecture

"""
    TabulationParameters

Lookup table grid configuration. See [`TabulationParameters`](@ref) constructor.
"""
struct TabulationParameters{FT}
    number_of_mass_points :: Int
    number_of_rime_fraction_points :: Int
    number_of_liquid_fraction_points :: Int
    minimum_mean_particle_mass :: FT
    maximum_mean_particle_mass :: FT
    number_of_quadrature_points :: Int
end

"""
$(TYPEDSIGNATURES)

Configure the lookup table grid for P3 integrals.

The P3 Fortran code pre-computes bulk integrals on a 3D grid indexed by:

1. **Mean particle mass** `qⁱ/Nⁱ` [kg]: Mass per particle (log-spaced)
2. **Rime fraction** `∈ [0, 1]`: Mass fraction that is rime (frozen accretion)
3. **Liquid fraction** `∈ [0, 1]`: Mass fraction that is liquid water on ice

During simulation, integral values are interpolated from this table rather
than computed via quadrature, which is much faster.

# Keyword Arguments

- `number_of_mass_points`: Grid points in mean particle mass (log-spaced), default 50
- `number_of_rime_fraction_points`: Grid points in rime fraction (linear), default 4
- `number_of_liquid_fraction_points`: Grid points in liquid fraction (linear), default 4
- `minimum_mean_particle_mass`: Minimum mean particle mass [kg], default 10⁻¹⁸
- `maximum_mean_particle_mass`: Maximum mean particle mass [kg], default 10⁻⁵
- `number_of_quadrature_points`: Quadrature points for filling table, default 64

# References

Table structure follows `create_p3_lookupTable_1.f90` in P3-microphysics.
"""
function TabulationParameters(FT::Type{<:AbstractFloat} = Float64;
                               number_of_mass_points::Int = 50,
                               number_of_rime_fraction_points::Int = 4,
                               number_of_liquid_fraction_points::Int = 4,
                               minimum_mean_particle_mass = FT(1e-18),
                               maximum_mean_particle_mass = FT(1e-5),
                               number_of_quadrature_points::Int = 64)
    return TabulationParameters(
        number_of_mass_points,
        number_of_rime_fraction_points,
        number_of_liquid_fraction_points,
        FT(minimum_mean_particle_mass),
        FT(maximum_mean_particle_mass),
        number_of_quadrature_points
    )
end

"""
    mean_particle_mass_grid(params::TabulationParameters)

Generate the mean particle mass grid points (logarithmically spaced).
"""
function mean_particle_mass_grid(params::TabulationParameters{FT}) where FT
    n = params.number_of_mass_points
    log_min = log10(params.minimum_mean_particle_mass)
    log_max = log10(params.maximum_mean_particle_mass)

    return [FT(10^(log_min + (i-1) * (log_max - log_min) / (n - 1))) for i in 1:n]
end

"""
    rime_fraction_grid(params::TabulationParameters)

Generate the rime fraction grid points (linearly spaced from 0 to 1).
"""
function rime_fraction_grid(params::TabulationParameters{FT}) where FT
    n = params.number_of_rime_fraction_points
    return [FT((i-1) / (n - 1)) for i in 1:n]
end

"""
    liquid_fraction_grid(params::TabulationParameters)

Generate the liquid fraction grid points (linearly spaced from 0 to 1).
"""
function liquid_fraction_grid(params::TabulationParameters{FT}) where FT
    n = params.number_of_liquid_fraction_points
    return [FT((i-1) / (n - 1)) for i in 1:n]
end

"""
    state_from_mean_particle_mass(FT, mean_particle_mass, rime_fraction, liquid_fraction; rime_density=400)

Create an IceSizeDistributionState from physical quantities.

Given mean particle mass = qⁱ/Nⁱ (mass per particle), we need to determine
the size distribution parameters (N₀, μ, λ).

Using the gamma distribution moments:
- M₀ = N = N₀ Γ(μ+1) / λ^{μ+1}
- M₃ = q/ρ = N₀ Γ(μ+4) / λ^{μ+4}

The ratio gives mean_particle_mass ∝ Γ(μ+4) / (Γ(μ+1) λ³)
"""
function state_from_mean_particle_mass(FT, mean_particle_mass, rime_fraction, liquid_fraction;
                                        rime_density = FT(400),
                                        shape_parameter = FT(0))
    # For μ=0: mean_particle_mass ≈ 6 / λ³ * (some density factor)
    # Invert to get λ from mean_particle_mass

    # Simplified: assume particle mass m ~ ρ_eff D³
    # mean_particle_mass ~ D³ means λ ~ 1/D ~ mean_particle_mass^{-1/3}

    pure_ice_density = FT(917)
    unrimed_effective_density_factor = FT(0.1)  # Aggregates have ~10% bulk density of pure ice
    effective_density = (1 - rime_fraction) * pure_ice_density * unrimed_effective_density_factor + 
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

@kernel function _fill_integral_table!(table, integral,
                                       mean_particle_mass_values,
                                       rime_fraction_values,
                                       liquid_fraction_values,
                                       quadrature_nodes,
                                       quadrature_weights)
    i_mass, i_rime, i_liquid = @index(Global, NTuple)

    mean_particle_mass = @inbounds mean_particle_mass_values[i_mass]
    rime_fraction = @inbounds rime_fraction_values[i_rime]
    liquid_fraction = @inbounds liquid_fraction_values[i_liquid]

    # Create size distribution state for this grid point
    FT = eltype(table)
    state = state_from_mean_particle_mass(FT, mean_particle_mass, rime_fraction, liquid_fraction)

    # Evaluate integral using pre-computed quadrature nodes/weights
    @inbounds table[i_mass, i_rime, i_liquid] = evaluate_with_quadrature(
        integral, state, quadrature_nodes, quadrature_weights
    )
end

"""
    evaluate_with_quadrature(integral, state, nodes, weights)

Evaluate a P3 integral using pre-computed quadrature nodes and weights.
This avoids allocation inside kernels.
"""
@inline function evaluate_with_quadrature(integral::AbstractP3Integral,
                                          state::IceSizeDistributionState,
                                          nodes, weights)
    FT = typeof(state.slope)
    slope_parameter = state.slope
    result = zero(FT)
    number_of_quadrature_points = length(nodes)

    for i in 1:number_of_quadrature_points
        x = @inbounds nodes[i]
        w = @inbounds weights[i]

        diameter = transform_to_diameter(x, slope_parameter)
        jacobian = jacobian_diameter_transform(x, slope_parameter)
        integrand_value = integrand(integral, diameter, state)

        result += w * integrand_value * jacobian
    end

    return result
end

"""
    tabulate(integral, arch, params)

Generate a lookup table for a single P3 integral.

This pre-computes integral values on a 3D grid of (mean_particle_mass, rime_fraction, 
liquid_fraction) so that during simulation, values can be interpolated rather than computed.

# Arguments

- `integral`: Integral type to tabulate (e.g., `MassWeightedFallSpeed()`)
- `arch`: `CPU()` or `GPU()` - determines where table is stored and computed
- `params`: [`TabulationParameters`](@ref) defining the grid

# Returns

[`TabulatedIntegral`](@ref) wrapping the lookup table array.
"""
function tabulate(integral::AbstractP3Integral, arch,
                  params::TabulationParameters{FT} = TabulationParameters(FT)) where FT

    mean_particle_mass_values = mean_particle_mass_grid(params)
    rime_fraction_values = rime_fraction_grid(params)
    liquid_fraction_values = liquid_fraction_grid(params)

    n_mass = params.number_of_mass_points
    n_rime = params.number_of_rime_fraction_points
    n_liquid = params.number_of_liquid_fraction_points
    n_quadrature = params.number_of_quadrature_points

    # Pre-compute quadrature nodes and weights
    nodes, weights = chebyshev_gauss_nodes_weights(FT, n_quadrature)

    # Allocate table and transfer grid arrays to target architecture
    table = on_architecture(arch, zeros(FT, n_mass, n_rime, n_liquid))
    mass_values_on_arch = on_architecture(arch, mean_particle_mass_values)
    rime_values_on_arch = on_architecture(arch, rime_fraction_values)
    liquid_values_on_arch = on_architecture(arch, liquid_fraction_values)
    nodes_on_arch = on_architecture(arch, nodes)
    weights_on_arch = on_architecture(arch, weights)

    # Launch kernel to fill table on the target architecture
    kernel! = _fill_integral_table!(device(arch), min(256, n_mass * n_rime * n_liquid))
    kernel!(table, integral,
            mass_values_on_arch, rime_values_on_arch, liquid_values_on_arch,
            nodes_on_arch, weights_on_arch;
            ndrange = (n_mass, n_rime, n_liquid))

    return TabulatedIntegral(table)
end

"""
    tabulate(ice_fall_speed::IceFallSpeed, arch, params::TabulationParameters)

Tabulate all integrals in an IceFallSpeed container.

Returns a new IceFallSpeed with TabulatedIntegral fields.
"""
function tabulate(fall_speed::IceFallSpeed{FT}, arch,
                  params::TabulationParameters{FT} = TabulationParameters(FT)) where FT

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
    tabulate(ice_deposition::IceDeposition, arch, params::TabulationParameters)

Tabulate all integrals in an IceDeposition container.
"""
function tabulate(deposition::IceDeposition{FT}, arch,
                  params::TabulationParameters{FT} = TabulationParameters(FT)) where FT

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
    tabulate(microphysics, property, arch; kwargs...)

Tabulate specific integrals within a P3 microphysics scheme.

This provides an interface to selectively tabulate subsets of integrals,
returning a new microphysics struct with the specified integrals replaced
by lookup tables.

# Arguments

- `microphysics`: [`PredictedParticlePropertiesMicrophysics`](@ref)
- `property`: Which integrals to tabulate
  - `:ice_fall_speed`: All fall speed integrals
  - `:ice_deposition`: All deposition/ventilation integrals
- `arch`: `CPU()` or `GPU()`

# Keyword Arguments

Passed to [`TabulationParameters`](@ref): `number_of_mass_points`, 
`number_of_rime_fraction_points`, etc.

# Returns

New `PredictedParticlePropertiesMicrophysics` with tabulated integrals.

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
                  arch;
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
            p3.precipitation_boundary_condition
        )

    else
        throw(ArgumentError("Unknown property to tabulate: $property. " *
                           "Supported: :ice_fall_speed, :ice_deposition"))
    end
end
