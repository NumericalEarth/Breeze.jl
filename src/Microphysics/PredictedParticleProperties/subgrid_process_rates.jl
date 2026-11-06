#####
##### Subgrid (PDF) averaging of the warm-phase P3 rates
#####
#
# With a subgrid distribution of the saturation deficit s = a [qᵗ - qᵛ⁺(Tˡ)] (see
# `Breeze.Microphysics.subgrid_cloud`), the warm-phase rates are replaced by their averages over
# that distribution (design steps 7 and 8):
#
# * condensation relaxes cloud liquid toward the partial-cloud equilibrium q⁺ = σₛ G(Q),
#       dqᶜˡ/dt = (q⁺ - qᶜˡ) / τₑ,    τₑ = τ [C + (1 - C) Γ],
#   with τₑ the exact-increment timescale when the formation model is exponential, so the
#   update never overshoots q⁺. This replaces the grid-mean all-or-nothing threshold of
#   `coupled_saturation_adjustment_rates` and the liquid caps of `limit_vapor_rates`;
# * autoconversion acts on the in-cloud liquid qᶜˡ / C over the cloud fraction, enhanced by the
#   in-cloud gamma-distribution factor E (Morrison and Gettelman 2008): C E R(qᶜˡ / C);
# * accretion acts on the cloud fraction with the rain filling the cell: C R(qᶜˡ / C, qʳ);
# * rain evaporates in the clear fraction, toward the clear-air deficit
#   (1 - C) s̄ᶜˡʳ = s̄ - q⁺ < 0, at the rain's vapor-relaxation rate;
# * activation sees the supersaturated tail, with in-cloud vapor excess Γ (q⁺ - qᶜˡ) / C,
#   weighted by C.
#
# Ice processes keep their grid-mean form. `DiracStatistics` returns the grid-mean rates unchanged.

@inline p3_subgrid_phase1_rates(::DiracStatistics, phase1, p3, ρ, ℳ, 𝒰, constants, state) =
    (; phase1, cloud = nothing)

# In-cloud droplet number: prescribed numbers are already in-cloud values; predicted grid-mean
# numbers are concentrated into the cloudy fraction.
@inline in_cloud_droplet_number(::Nothing, Nᶜˡ, C) = Nᶜˡ
@inline in_cloud_droplet_number(aerosol, Nᶜˡ, C) = Nᶜˡ / C

@inline function p3_subgrid_phase1_rates(𝒮, phase1, p3, ρ, ℳ, 𝒰, constants, state)
    FT = typeof(ρ)
    correlation = p3.subgrid
    parameters = p3.process_rates
    T = state.T
    qᵛ = state.qᵛ
    qᵛ⁺ˡ = state.qᵛ⁺ˡ
    qᶜˡ = max(0, ℳ.qᶜˡ)
    qʳ = ℳ.qʳ

    # Mellor linearisation, s' = a qᵗ' - b θˡⁱ', with Γ the psychrometric correction that the
    # grid-mean caps use, so σₛ → 0 recovers their increment (qᵛ - qᵛ⁺ˡ) / Γ.
    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    Γ = liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)
    Π = exner_function(𝒰, constants)
    a = 1 / Γ
    b = a * Π * qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    s̄ = (qᵛ - qᵛ⁺ˡ) / Γ + qᶜˡ

    cloud = subgrid_cloud(𝒮, s̄, a, b, correlation.minimum_saturation_deficit_spread)
    C = cloud.C
    q⁺ = min(cloud.q⁺, max(0, qᵛ) + qᶜˡ)
    Cᶠ = max(C, correlation.minimum_cloud_fraction)

    # Condensation (step 7)
    τ = parameters.sink_limiting_timescale
    τₑ = scaled_relaxation_timescale(correlation.condensation, τ, C + (1 - C) * Γ)
    condensation = (q⁺ - qᶜˡ) / τₑ

    # Activation over the supersaturated tail
    qᵛ_tail = qᵛ⁺ˡ + Γ * max(0, q⁺ - qᶜˡ) / Cᶠ
    activation = compute_cloud_droplet_activation(p3.aerosol, p3, qᶜˡ / Cᶠ, ℳ.nᶜˡ / Cᶠ, ℳ.nᵃ,
                                                  qᵛ_tail, qᵛ⁺ˡ, T, ρ, constants)

    # Autoconversion and accretion over the cloudy fraction
    Nᶜˡ = in_cloud_droplet_number(p3.aerosol, state.Nᶜˡ, Cᶠ)
    autoconversion = Cᶠ * correlation.autoconversion_enhancement *
                     rain_autoconversion_rate(p3, qᶜˡ / Cᶠ, Nᶜˡ, ρ, qʳ)
    accretion = Cᶠ * rain_accretion_rate(p3, qᶜˡ / Cᶠ, qʳ, ρ)

    # Rain evaporation in the clear fraction
    transport = (; Dᵛ = state.Dᵛ, Kᵃ = state.Kᵃ, ν = state.ν)
    Λʳ = rain_vapor_relaxation_coefficient(p3, qʳ, state.nʳ, ρ, transport)
    window = relaxation_window(correlation.condensation, τ)
    clear_deficit = max(0, q⁺ - s̄)
    rain_evaporation = min(clear_deficit * exponential_rate_factor(Λʳ, window),
                           max(0, qʳ) / window)

    phase1_subgrid = P3Phase1Rates{FT}(condensation, activation.mass * C, activation.number * C,
                                       autoconversion, accretion, phase1.cloud_self_collection,
                                       rain_evaporation, zero(FT),
                                       phase1.rain_self_collection, phase1.rain_breakup,
                                       phase1.deposition, phase1.coating_condensation,
                                       phase1.coating_evaporation, phase1.partial_melting,
                                       phase1.complete_melting, phase1.melting_number)

    return (; phase1 = phase1_subgrid, cloud = (; q⁺, qᶜˡ, window))
end

#####
##### Vapor limiting
#####

# Grid-mean: the saturation-adjustment caps of `limit_vapor_rates`.
@inline limit_vapor_rates(::Nothing, args...) = limit_vapor_rates(args...)

# Subgrid: condensation is already bounded by its exponential approach to q⁺, so the grid-mean
# liquid threshold is dropped. The other liquid sources share the room left below q⁺; the
# ice-phase caps are unchanged.
@inline function limit_vapor_rates(cloud::NamedTuple, cond, ccn_activation_mass, ccn_activation_number,
                                   rain_cond, rain_evap, dep, coat_cond, coat_evap,
                                   nuc_q, nuc_n, qᵛ, qᵛ⁺ˡ, T, P, qᵗ, constants,
                                   dt_safety, freezing_temperature,
                                   temperature_tendency, pressure_tendency)
    room = max(0, (cloud.q⁺ - cloud.qᶜˡ) / cloud.window - max(0, cond))
    f_source = sink_limiting_factor(ccn_activation_mass + rain_cond + coat_cond, room * dt_safety, dt_safety)
    ccn_activation_mass = ccn_activation_mass * f_source
    ccn_activation_number = ccn_activation_number * f_source
    rain_cond = rain_cond * f_source
    coat_cond = coat_cond * f_source

    return limit_ice_vapor_rates(cond, ccn_activation_mass, ccn_activation_number,
                                 rain_cond, rain_evap, dep, coat_cond, coat_evap,
                                 nuc_q, nuc_n, qᵛ, T, P, qᵗ, constants,
                                 dt_safety, freezing_temperature,
                                 temperature_tendency, pressure_tendency)
end
