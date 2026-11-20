#####
##### Subgrid information from the turbulence closure in the warm-phase P3 rates
#####
#
# `compute_p3_process_rates` hands the grid-mean Phase 1 rates and the closure's subgrid statistics
# 𝒮 of the cell (`AtmosphereModels.subgrid_statisticsᶜᶜᶜ`) to `p3_subgrid_phase1_rates`.
# `DiracStatistics` returns the grid-mean rates unchanged.

@inline p3_subgrid_phase1_rates(::DiracStatistics, phase1, p3, ρ, ℳ, constants, state) = phase1

#####
##### Cloud owned by the turbulence closure (`ClosureCloud`)
#####
#
# The closure diagnoses the cloud fraction C and sets the cloud liquid qᶜˡ once per time step, so
# P3 forms and evaporates no cloud liquid in any cell: condensation, droplet activation and
# condensation onto rain vanish. P3 owns the rain:
#
# * autoconversion acts on the in-cloud liquid qᶜˡ / C over the cloud fraction, enhanced by the
#   factor E for the variability of the liquid within the cloud: C E R(qᶜˡ / C);
# * accretion acts on the cloud fraction with the rain filling the cell: C R(qᶜˡ / C, qʳ);
# * rain evaporates toward the mean subsaturation (qᵛ⁺ - qᵛ) / Γ of the cell, which is all in its
#   clear fraction, at the rain's vapor-relaxation rate Λʳ, with the exact increment over the sink
#   limiting timescale so that a stiff rate cannot overshoot.
#
# Cells below freezing keep the grid-mean form of every rate but the three that vanish.

# Prescribed droplet numbers are in-cloud values; predicted grid-mean numbers are concentrated
# into the cloudy fraction.
@inline in_cloud_droplet_number(::Nothing, Nᶜˡ, C) = Nᶜˡ
@inline in_cloud_droplet_number(aerosol, Nᶜˡ, C) = Nᶜˡ / C

@inline function p3_subgrid_phase1_rates(𝒮::ClosureCloudStatistics, phase1, p3, ρ, ℳ, constants, state)
    FT = typeof(ρ)
    correlation = p3.subgrid
    parameters = p3.process_rates
    T = state.T
    qᶜˡ = max(0, ℳ.qᶜˡ)
    qʳ = ℳ.qʳ

    C = max(𝒮.cloud_fraction, correlation.minimum_cloud_fraction)
    Nᶜˡ = in_cloud_droplet_number(p3.aerosol, state.Nᶜˡ, C)

    autoconversion = C * correlation.autoconversion_enhancement *
                     rain_autoconversion_rate(p3, qᶜˡ / C, Nᶜˡ, ρ, qʳ)
    accretion = C * rain_accretion_rate(p3, qᶜˡ / C, qʳ, ρ)

    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    Γ = liquid_psychrometric_correction(constants, ℒˡ, state.qᵛ⁺ˡ, Rᵛ, T)
    clear_deficit = max(0, state.qᵛ⁺ˡ - state.qᵛ) / Γ

    transport = (; Dᵛ = state.Dᵛ, Kᵃ = state.Kᵃ, ν = state.ν)
    Λʳ = rain_vapor_relaxation_coefficient(p3, qʳ, state.nʳ, ρ, transport)
    τ = parameters.sink_limiting_timescale
    rain_evaporation = min(-clear_deficit * expm1(-Λʳ * τ), max(0, qʳ)) / τ

    warm = T ≥ parameters.freezing_temperature
    warm_rate(subgrid, grid_mean) = ifelse(warm, subgrid, grid_mean)

    return P3Phase1Rates{FT}(zero(FT), zero(FT), zero(FT),
                             warm_rate(autoconversion, phase1.autoconversion),
                             warm_rate(accretion, phase1.accretion),
                             phase1.cloud_self_collection,
                             warm_rate(rain_evaporation, phase1.rain_evaporation),
                             zero(FT),
                             phase1.rain_self_collection, phase1.rain_breakup,
                             phase1.deposition, phase1.coating_condensation,
                             phase1.coating_evaporation, phase1.partial_melting,
                             phase1.complete_melting, phase1.melting_number)
end
