using ..Thermodynamics:
    MoistureMassFractions,
    LiquidIceDensityState,
    temperature,
    with_moisture,
    with_temperature,
    WarmPhaseEquilibrium

using ..AtmosphereModels:
    AtmosphereModels,
    dynamics_density,
    standard_pressure

using Oceananigans: Oceananigans, CenterField
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: @index, @kernel
using DocStringExtensions: TYPEDSIGNATURES

"""
    struct DCMIP2016LargeScaleCondensation{S}

Reed–Jablonowski large-scale condensation microphysics: an instantaneous,
*irreversible* condensation with immediate rain-out and no re-evaporation
(no cloud or rain stage). Excess water vapor above saturation condenses, releases
its latent heat to the air, and is removed as precipitation in the same step.

The saturation/latent-heat solve is delegated to a `SaturationAdjustment`
instance (`saturation_adjustment`), so the equilibrium thermodynamics are shared
and validated. The distinction is irreversibility: the condensate is purged from
the prognostic vapor every step and cannot re-evaporate.

The prognostic moisture is the vapor density `ρqᵛ` (no condensate is retained).
"""
struct DCMIP2016LargeScaleCondensation{S}
    saturation_adjustment :: S
end

"""
$(TYPEDSIGNATURES)

Construct a `DCMIP2016LargeScaleCondensation` scheme. `equilibrium` selects the phase
equilibrium used by the underlying saturation solve (default warm-phase).
"""
function DCMIP2016LargeScaleCondensation(FT::DataType=Oceananigans.defaults.FloatType;
                                         equilibrium = WarmPhaseEquilibrium(),
                                         tolerance = 1e-3,
                                         maxiter = Inf)
    sa = SaturationAdjustment(FT; tolerance, maxiter, equilibrium)
    return DCMIP2016LargeScaleCondensation(sa)
end

const LSC = DCMIP2016LargeScaleCondensation

# Prognostic vapor density; no retained condensate fields.
AtmosphereModels.moisture_prognostic_name(::LSC) = :ρqᵛ
AtmosphereModels.prognostic_field_names(::LSC) = tuple()

AtmosphereModels.liquid_mass_fraction(::LSC, model) = nothing
AtmosphereModels.ice_mass_fraction(::LSC, model) = nothing
@inline AtmosphereModels.microphysical_velocities(::LSC, μ, name) = nothing
@inline AtmosphereModels.microphysical_tendency(::LSC, name, ρ, ℳ, 𝒰, constants) = zero(ρ)

# Diagnostic vapor + a precipitation-rate field (kg m⁻³ s⁻¹ of condensed/removed water).
AtmosphereModels.materialize_microphysical_fields(::LSC, grid, bcs) =
    (; qᵛ = CenterField(grid), precipitation_rate = CenterField(grid))

# Prognostic vapor only; no stored condensate (all moisture is vapor between steps).
@inline function AtmosphereModels.grid_moisture_fractions(i, j, k, grid, ::LSC, ρ, qᵛ, μ)
    return MoistureMassFractions(qᵛ)
end

@inline function AtmosphereModels.update_microphysical_fields!(μ, i, j, k, grid, ::LSC, ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    return nothing
end

# Like DCMIP2016KM, the diagnostic state is returned UNCHANGED: all condensation
# (and its latent heating) happens once per step in microphysics_model_update!,
# not in the per-stage equilibrium adjustment. Delegating to SA here would make
# update_state! re-condense after the kernel and double-count the latent heat.
@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰, ::LSC, qᵛ, constants) = 𝒰

AtmosphereModels.precipitation_rate(model, ::LSC, ::Val{:liquid}) = model.microphysical_fields.precipitation_rate
AtmosphereModels.precipitation_rate(model, ::LSC, ::Val{:ice}) = nothing

#####
##### Once-per-step irreversible condensation + rain-out
#####

"""
$(TYPEDSIGNATURES)

Condense supersaturation, retain the released latent heat, and remove the
condensate as precipitation — applied directly to the prognostic vapor `ρqᵛ`
and liquid-ice potential temperature density `ρθˡⁱ`.
"""
function AtmosphereModels.microphysics_model_update!(microphysics::LSC, model)
    grid = model.grid
    arch = architecture(grid)
    Δt = model.clock.last_Δt

    # Skip before the first step (Δt invalid during construction).
    (isnan(Δt) || isinf(Δt) || Δt ≤ 0) && return nothing

    pˢᵗ = standard_pressure(model.dynamics)
    constants = model.thermodynamic_constants
    ρθˡⁱ = model.formulation.potential_temperature_density
    ρqᵛ = model.moisture_density
    μ = model.microphysical_fields

    saturation_adjustment = microphysics.saturation_adjustment

    launch!(arch, grid, :xyz, _large_scale_condensation_update!,
            model.dynamics, saturation_adjustment, constants, pˢᵗ, Δt, ρθˡⁱ, ρqᵛ, μ)

    return nothing
end

@kernel function _large_scale_condensation_update!(dynamics, saturation_adjustment,
                                                   constants, pˢᵗ, Δt, ρθˡⁱ, ρqᵛ, μ)
    i, j, k = @index(Global, NTuple)

    ρ_field = dynamics_density(dynamics)

    @inbounds begin
        ρ  = ρ_field[i, j, k]
        ρqᵛ₀ = ρqᵛ[i, j, k]
        qᵗ = ρqᵛ₀ / ρ              # total water = prognostic vapor (no retained condensate)
        θ₀ = ρθˡⁱ[i, j, k] / ρ      # θˡⁱ — conserved by the reversible condensation

        # Condensation — constant-density saturation adjustment with θˡⁱ held fixed. Delegated to
        # the shared #765 secant, which saturates against the density-based qsat at the cell's own
        # ρ (true pressure p = ρRᵐT). Subsaturated cells short-circuit untouched, reproducing LSC's
        # original behavior. See adjust_thermodynamic_state(::LiquidIceDensityState, ::SA).
        𝒰₀  = LiquidIceDensityState(θ₀, MoistureMassFractions(qᵗ), pˢᵗ, ρ)
        𝒰₁  = adjust_thermodynamic_state(𝒰₀, saturation_adjustment, constants)
        q   = 𝒰₁.moisture_mass_fractions
        T   = temperature(𝒰₁, constants)
        qᵛ⁺ = q.vapor

        # Rain-out — remove all condensate but RETAIN the latent warming: the vapor-only parcel
        # keeps T, so its θˡⁱ is the condensate-free (L = 0) inversion. with_temperature on the
        # vapor-only state is exactly the old closed form θᶠ = T^(1/γ)(ρRᵐ/pˢᵗ)^((1-γ)/γ).
        𝒰ᵥ = with_moisture(𝒰₁, MoistureMassFractions(qᵛ⁺))
        θᶠ = with_temperature(𝒰ᵥ, T, constants).potential_temperature
        ρqᵛ⁺ = ρ * qᵛ⁺
        condensed_water_density = max(0, ρqᵛ₀ - ρqᵛ⁺)
        precipitation_rate = condensed_water_density / Δt

        ρqᵛ[i, j, k]  = ρqᵛ⁺
        ρθˡⁱ[i, j, k] = ρ * θᶠ
        μ.qᵛ[i, j, k] = qᵛ⁺
        μ.precipitation_rate[i, j, k] = ifelse(condensed_water_density > 0,
                                               precipitation_rate,
                                               μ.precipitation_rate[i, j, k])
    end
end
