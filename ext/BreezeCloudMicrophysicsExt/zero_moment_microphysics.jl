#####
##### Zero-moment bulk microphysics (CloudMicrophysics 0M)
#####

"""
    ZeroMomentBulkMicrophysics

Type alias for `BulkMicrophysics` with CloudMicrophysics 0M precipitation scheme.

The 0M scheme instantly removes precipitable condensate above a threshold.
Interface is identical to non-precipitating microphysics except that
`maybe_adjust_thermodynamic_state` calls CloudMicrophysics `remove_precipitation` first.
"""
const ZeroMomentCloudMicrophysics = BulkMicrophysics{<:Any, <:Parameters0M, <:Any}
const ZMCM = ZeroMomentCloudMicrophysics

AtmosphereModels.prognostic_field_names(::ZMCM) = tuple()
AtmosphereModels.materialize_microphysical_fields(bμp::ZMCM, grid, bcs) = materialize_microphysical_fields(bμp.cloud_formation, grid, bcs)
@inline AtmosphereModels.update_microphysical_fields!(μ, i, j, k, grid, bμp::ZMCM, ρ, 𝒰, constants) = update_microphysical_fields!(μ, i, j, k, grid, bμp.cloud_formation, ρ, 𝒰, constants)
@inline AtmosphereModels.grid_moisture_fractions(i, j, k, grid, bμp::ZMCM, ρ, qᵉ, μ) = grid_moisture_fractions(i, j, k, grid, bμp.cloud_formation, ρ, qᵉ, μ)
@inline AtmosphereModels.grid_microphysical_tendency(i, j, k, grid, bμp::ZMCM, name, ρ, μ, 𝒰, constants, velocities) = zero(grid)
@inline AtmosphereModels.microphysical_velocities(bμp::ZMCM, μ, name) = nothing

@inline function AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰₀, bμp::ZMCM, qᵉ, constants)
    # Initialize moisture state from equilibrium moisture qᵉ (not from stale microphysical fields)
    q₀ = MoistureMassFractions(qᵉ)
    𝒰₁ = with_moisture(𝒰₀, q₀)
    return adjust_thermodynamic_state(𝒰₁, bμp.cloud_formation, constants)
end

@inline function AtmosphereModels.grid_microphysical_tendency(i, j, k, grid, bμp::ZMCM, ::Val{:ρqᵉ}, ρ, μ, 𝒰, constants, velocities)
    # Get cloud liquid water from microphysical fields
    q = 𝒰.moisture_mass_fractions
    qˡ = q.liquid
    qⁱ = q.ice

    # remove_precipitation returns -dqᵉ/dt (rate of moisture removal)
    # Multiply by density to get the tendency for ρqᵉ
    # TODO: pass density into microphysical_tendency
    ρ = density(𝒰, constants)
    parameters_0M = bμp.categories

    return ρ * remove_precipitation(parameters_0M, qˡ, qⁱ)
end

"""
    ZeroMomentCloudMicrophysics(FT = Oceananigans.defaults.FloatType;
                                cloud_formation = SaturationAdjustment(FT),
                                τ_precip = 1000,
                                qc_0 = 5e-4,
                                S_0 = 0)

Return a `ZeroMomentCloudMicrophysics` microphysics scheme for warm-rain precipitation.

The zero-moment scheme removes cloud liquid water above a threshold at a specified rate:
- `τ_precip`: precipitation timescale in seconds (default: 1000 s)

and _either_

- `S_0`: supersaturation threshold (default: 0)
- `qc_0`: cloud liquid water threshold for precipitation (default: 5×10⁻⁴ kg/kg)

For more information see the [CloudMicrophysics.jl documentation](https://clima.github.io/CloudMicrophysics.jl/stable/Microphysics0M/).
"""
function ZeroMomentCloudMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                                     cloud_formation = SaturationAdjustment(FT),
                                     τ_precip = 1000,
                                     qc_0 = 5e-4,
                                     S_0 = 0)

    categories = Parameters0M{FT}(; τ_precip = FT(τ_precip),
                                    qc_0 = FT(qc_0),
                                    S_0 = FT(S_0))

    # Zero-moment schemes don't have explicit sedimentation, so precipitation_bottom = nothing
    return BulkMicrophysics(cloud_formation, categories, nothing)
end

#####
##### Precipitation rate diagnostic for zero-moment microphysics
#####

struct ZeroMomentPrecipitationRateKernel{C, Q}
    categories :: C
    cloud_liquid :: Q
end

Adapt.adapt_structure(to, k::ZeroMomentPrecipitationRateKernel) =
    ZeroMomentPrecipitationRateKernel(adapt(to, k.categories),
                                       adapt(to, k.cloud_liquid))

@inline function (k::ZeroMomentPrecipitationRateKernel)(i, j, k_idx, grid)
    @inbounds qˡ = k.cloud_liquid[i, j, k_idx]
    # Warm-phase only: no ice
    qⁱ = zero(qˡ)
    # remove_precipitation returns dqᵉ/dt (negative = moisture removal = precipitation)
    # We return positive precipitation rate (kg/kg/s)
    return -remove_precipitation(k.categories, qˡ, qⁱ)
end

"""
$(TYPEDSIGNATURES)

Return a `Field` representing the liquid precipitation rate (rain rate) in kg/kg/s.

For zero-moment microphysics, this is the rate at which cloud liquid water
is removed by precipitation: `-dqᵉ/dt` from the `remove_precipitation` function.
"""
function AtmosphereModels.precipitation_rate(model, microphysics::ZMCM, ::Val{:liquid})
    grid = model.grid
    qˡ = model.microphysical_fields.qˡ
    kernel = ZeroMomentPrecipitationRateKernel(microphysics.categories, qˡ)
    op = KernelFunctionOperation{Center, Center, Center}(kernel, grid)
    return Field(op)
end

# Ice precipitation not supported for zero-moment warm-phase scheme
AtmosphereModels.precipitation_rate(model, ::ZMCM, ::Val{:ice}) = nothing
