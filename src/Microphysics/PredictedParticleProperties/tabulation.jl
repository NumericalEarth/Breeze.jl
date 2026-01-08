#####
##### Tabulation of P3 Integrals
#####
##### Generate lookup tables for efficient evaluation during simulation.
##### Tables are indexed by normalized ice mass (Q_norm), rime fraction (F_r),
##### and liquid fraction (F_l).
#####

export tabulate, TabulationParameters

"""
    TabulationParameters

Lookup table grid configuration. See [`TabulationParameters`](@ref) constructor.
"""
struct TabulationParameters{FT}
    n_Qnorm :: Int
    n_Fr :: Int
    n_Fl :: Int
    Qnorm_min :: FT
    Qnorm_max :: FT
    n_quadrature :: Int
end

"""
$(TYPEDSIGNATURES)

Configure the lookup table grid for P3 integrals.

The P3 Fortran code pre-computes bulk integrals on a 3D grid indexed by:

1. **Normalized mass** `Qnorm = q/N` [kg]: Mean mass per particle
2. **Rime fraction** `Fr ∈ [0, 1]`: Mass fraction that is rime
3. **Liquid fraction** `Fl ∈ [0, 1]`: Mass fraction that is liquid water on ice

During simulation, integral values are interpolated from this table rather
than computed via quadrature, which is much faster.

# Keyword Arguments

- `n_Qnorm`: Grid points in Qnorm (log-spaced), default 50
- `n_Fr`: Grid points in rime fraction (linear), default 4
- `n_Fl`: Grid points in liquid fraction (linear), default 4
- `Qnorm_min`: Minimum Qnorm [kg], default 10⁻¹⁸
- `Qnorm_max`: Maximum Qnorm [kg], default 10⁻⁵
- `n_quadrature`: Quadrature points for filling table, default 64

# References

Table structure follows `create_p3_lookupTable_1.f90` in P3-microphysics.
"""
function TabulationParameters(FT::Type{<:AbstractFloat} = Float64;
                               n_Qnorm::Int = 50,
                               n_Fr::Int = 4,
                               n_Fl::Int = 4,
                               Qnorm_min = FT(1e-18),
                               Qnorm_max = FT(1e-5),
                               n_quadrature::Int = 64)
    return TabulationParameters(
        n_Qnorm, n_Fr, n_Fl,
        FT(Qnorm_min), FT(Qnorm_max),
        n_quadrature
    )
end

"""
    Qnorm_grid(params::TabulationParameters)

Generate the normalized mass grid points (logarithmically spaced).
"""
function Qnorm_grid(params::TabulationParameters{FT}) where FT
    n = params.n_Qnorm
    log_min = log10(params.Qnorm_min)
    log_max = log10(params.Qnorm_max)
    
    return [FT(10^(log_min + (i-1) * (log_max - log_min) / (n - 1))) for i in 1:n]
end

"""
    Fr_grid(params::TabulationParameters)

Generate the rime fraction grid points (linearly spaced).
"""
function Fr_grid(params::TabulationParameters{FT}) where FT
    n = params.n_Fr
    return [FT((i-1) / (n - 1)) for i in 1:n]
end

"""
    Fl_grid(params::TabulationParameters)

Generate the liquid fraction grid points (linearly spaced).
"""
function Fl_grid(params::TabulationParameters{FT}) where FT
    n = params.n_Fl
    return [FT((i-1) / (n - 1)) for i in 1:n]
end

"""
    state_from_Qnorm(Qnorm, Fr, Fl; ρ_rim=400)

Create an IceSizeDistributionState from normalized quantities.

Given Q_norm = q_i/N_i (mass per particle), we need to determine
the size distribution parameters (N₀, μ, λ).

Using the gamma distribution moments:
- M₀ = N = N₀ Γ(μ+1) / λ^{μ+1}
- M₃ = q/ρ = N₀ Γ(μ+4) / λ^{μ+4}

The ratio gives Q_norm ∝ Γ(μ+4) / (Γ(μ+1) λ³)
"""
function state_from_Qnorm(FT, Qnorm, Fr, Fl; ρ_rim=FT(400), μ=FT(0))
    # For μ=0: Q_norm ≈ 6 / λ³ * (some density factor)
    # Invert to get λ from Q_norm
    
    # Simplified: assume particle mass m ~ ρ_eff D³
    # Q_norm ~ D³ means λ ~ 1/D ~ Q_norm^{-1/3}
    
    ρ_ice = FT(917)
    ρ_eff = (1 - Fr) * ρ_ice * FT(0.1) + Fr * ρ_rim
    
    # Characteristic diameter from Q_norm = (π/6) ρ_eff D³
    D_char = (6 * Qnorm / (π * ρ_eff))^(1/3)
    
    # λ ~ 4 / D for exponential distribution
    λ = FT(4) / max(D_char, FT(1e-8))
    
    # N₀ from normalization (set to give reasonable number concentration)
    N₀ = FT(1e6)  # Placeholder
    
    return IceSizeDistributionState(
        N₀, μ, λ, Fr, Fl, ρ_rim
    )
end

"""
    tabulate(integral, arch, params)

Generate a lookup table for a single P3 integral.

This pre-computes integral values on a 3D grid of (Qnorm, Fr, Fl) so that
during simulation, values can be interpolated rather than computed.

# Arguments

- `integral`: Integral type to tabulate (e.g., `MassWeightedFallSpeed()`)
- `arch`: `CPU()` or `GPU()` - determines where table is stored
- `params`: [`TabulationParameters`](@ref) defining the grid

# Returns

[`TabulatedIntegral`](@ref) wrapping the lookup table array.
"""
function tabulate(integral::AbstractP3Integral, arch, 
                  params::TabulationParameters{FT} = TabulationParameters(FT)) where FT
    
    Qnorm_vals = Qnorm_grid(params)
    Fr_vals = Fr_grid(params)
    Fl_vals = Fl_grid(params)
    
    n_Q = params.n_Qnorm
    n_Fr = params.n_Fr
    n_Fl = params.n_Fl
    n_quad = params.n_quadrature
    
    # Allocate table
    table = zeros(FT, n_Q, n_Fr, n_Fl)
    
    # Fill table
    for k in 1:n_Fl
        Fl = Fl_vals[k]
        for j in 1:n_Fr
            Fr = Fr_vals[j]
            for i in 1:n_Q
                Qnorm = Qnorm_vals[i]
                
                # Create state for this grid point
                state = state_from_Qnorm(FT, Qnorm, Fr, Fl)
                
                # Evaluate integral
                table[i, j, k] = evaluate(integral, state; n_quadrature=n_quad)
            end
        end
    end
    
    # Move to architecture if needed
    # For now, just return CPU array
    return TabulatedIntegral(table)
end

"""
    tabulate(ice_fall_speed::IceFallSpeed, arch, params::TabulationParameters)

Tabulate all integrals in an IceFallSpeed container.

Returns a new IceFallSpeed with TabulatedIntegral fields.
"""
function tabulate(fs::IceFallSpeed{FT}, arch, 
                  params::TabulationParameters{FT} = TabulationParameters(FT)) where FT
    
    return IceFallSpeed(
        fs.reference_air_density,
        fs.fall_speed_coefficient,
        fs.fall_speed_exponent,
        tabulate(fs.number_weighted, arch, params),
        tabulate(fs.mass_weighted, arch, params),
        tabulate(fs.reflectivity_weighted, arch, params)
    )
end

"""
    tabulate(ice_deposition::IceDeposition, arch, params::TabulationParameters)

Tabulate all integrals in an IceDeposition container.
"""
function tabulate(dep::IceDeposition{FT}, arch,
                  params::TabulationParameters{FT} = TabulationParameters(FT)) where FT
    
    return IceDeposition(
        dep.thermal_conductivity,
        dep.vapor_diffusivity,
        tabulate(dep.ventilation, arch, params),
        tabulate(dep.ventilation_enhanced, arch, params),
        tabulate(dep.small_ice_ventilation_constant, arch, params),
        tabulate(dep.small_ice_ventilation_reynolds, arch, params),
        tabulate(dep.large_ice_ventilation_constant, arch, params),
        tabulate(dep.large_ice_ventilation_reynolds, arch, params)
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

Passed to [`TabulationParameters`](@ref): `n_Qnorm`, `n_Fr`, etc.

# Returns

New `PredictedParticlePropertiesMicrophysics` with tabulated integrals.

# Example

```julia
using Oceananigans
using Breeze.Microphysics.PredictedParticleProperties

p3 = PredictedParticlePropertiesMicrophysics()
p3_fast = tabulate(p3, :ice_fall_speed, CPU(); n_Qnorm=100)
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

