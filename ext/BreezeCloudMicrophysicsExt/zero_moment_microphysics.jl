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
const ZeroMomentCloudMicrophysics = BulkMicrophysics{<:Any, <:Parameters0M}
const ZMCM = ZeroMomentCloudMicrophysics

AtmosphereModels.prognostic_field_names(::ZMCM) = tuple()
AtmosphereModels.materialize_microphysical_fields(bμp::ZMCM, grid, bcs) = materialize_microphysical_fields(bμp.cloud_formation, grid, bcs)
@inline AtmosphereModels.update_microphysical_fields!(μ, i, j, k, grid, bμp::ZMCM, ρ, 𝒰, constants) = update_microphysical_fields!(μ, i, j, k, grid, bμp.cloud_formation, ρ, 𝒰, constants)
@inline AtmosphereModels.grid_moisture_fractions(i, j, k, grid, bμp::ZMCM, ρ, qᵉ, μ) = grid_moisture_fractions(i, j, k, grid, bμp.cloud_formation, ρ, qᵉ, μ)
@inline AtmosphereModels.microphysical_velocities(bμp::ZMCM, μ, name) = nothing

@inline function AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰₀, bμp::ZMCM, qᵉ, constants)
    # Initialize moisture state from equilibrium moisture qᵉ (not from stale microphysical fields)
    q₀ = MoistureMassFractions(qᵉ)
    𝒰₁ = with_moisture(𝒰₀, q₀)
    return adjust_thermodynamic_state(𝒰₁, bμp.cloud_formation, constants)
end

#####
##### Precipitation rates and tendencies
#####
#
# Precipitation removes condensate from the column while leaving the in-situ
# temperature unchanged (the falling drop carries only its negligible sensible
# enthalpy, with no phase change). The conserved liquid-ice variable subtracts
# the condensate latent-heat contribution by definition, so removing condensate
# at fixed temperature requires sourcing it upward (issue #772):
#
#   ∂θˡⁱ/∂t |_precip = (ℒˡᵣ Ṙˡ + ℒⁱᵣ Ṙⁱ) / (cᵖᵐ Π)
#   ∂eˡⁱ/∂t |_precip =  ℒˡᵣ Ṙˡ + ℒⁱᵣ Ṙⁱ
#
# where Ṙˡ, Ṙⁱ ≥ 0 are the per-phase condensate removal rates. As in the 1M
# bundle (cf. wpne1m_tendencies), every tendency below derives from the same
# rates, so water sink and warming source are consistent by construction.

@inline function zero_moment_precipitation_rates(bμp::ZMCM, 𝒰)
    q = 𝒰.moisture_mass_fractions
    qˡ = q.liquid
    qⁱ = q.ice

    # remove_precipitation returns the (≤ 0) total moisture removal rate dqᵉ/dt.
    Ṡ = remove_precipitation(bμp.categories, qˡ, qⁱ)

    # Partition the removal between phases proportionally to their condensate.
    # When qᶜ = 0, remove_precipitation returns 0, so the guarded fractions only
    # protect against 0/0. Advection undershoots can make one phase negative while
    # qᶜ > 0; the fractions then fall outside [0, 1] but still satisfy Ṙˡ + Ṙⁱ = -Ṡ,
    # preserving mass consistency (do not clamp).
    qᶜ = qˡ + qⁱ
    fˡ = ifelse(qᶜ > 0, qˡ / qᶜ, zero(qᶜ))
    fⁱ = ifelse(qᶜ > 0, qⁱ / qᶜ, zero(qᶜ))
    Ṙˡ = -Ṡ * fˡ
    Ṙⁱ = -Ṡ * fⁱ

    return (; Ṡ, Ṙˡ, Ṙⁱ)
end

@inline function zero_moment_latent_heating(bμp::ZMCM, 𝒰, constants)
    rates = zero_moment_precipitation_rates(bμp, 𝒰)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    return ℒˡᵣ * rates.Ṙˡ + ℒⁱᵣ * rates.Ṙⁱ
end

@inline function AtmosphereModels.microphysical_tendency(bμp::ZMCM, ::Val{:ρqᵉ}, ρ, ℳ, 𝒰, constants)
    rates = zero_moment_precipitation_rates(bμp, 𝒰)
    return ρ * rates.Ṡ
end

@inline function AtmosphereModels.microphysical_tendency(bμp::ZMCM, ::Val{:ρθ}, ρ, ℳ, 𝒰, constants)
    latent_heating = zero_moment_latent_heating(bμp, 𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Π = exner_function(𝒰, constants)
    return ρ * latent_heating / (cᵖᵐ * Π)
end

@inline function AtmosphereModels.microphysical_tendency(bμp::ZMCM, ::Val{:ρe}, ρ, ℳ, 𝒰, constants)
    latent_heating = zero_moment_latent_heating(bμp, 𝒰, constants)
    return ρ * latent_heating
end

# Assumes every formulation defines thermodynamic_density_name (ρθ or ρe today);
# a future formulation (e.g. prognostic ρη) must extend the warming tendencies too.
AtmosphereModels.microphysical_thermodynamic_names(bμp::ZMCM, formulation) =
    (AtmosphereModels.thermodynamic_density_name(formulation),)

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

The latent heat of the removed condensate is retained: the removal sinks `ρqᵉ` and
sources the thermodynamic prognostic (`ρθ` or `ρe`) from the same rate, so rain-out
does not spuriously cool the column.

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
    return BulkMicrophysics(cloud_formation, categories, nothing, nothing)
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
