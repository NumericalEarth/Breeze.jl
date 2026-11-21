#####
##### Implicit relaxation of cloud condensation, rain vapor exchange and droplet activation
#####
#
# On the grid, the vapor exchange of liquid is advanced by backward Euler toward a non-negative
# target over the coming stage, from what the stage start and every other tendency leave of the
# cloud (`c̃ = c⁰ + Δt Gᵒ`, see `AtmosphereModels.stage_remainder`), instead of being capped by
# the vapor excess over `sink_limiting_timescale`:
#
#   * cloud liquid relaxes toward its equilibrium condensate (`liquid_equilibrium_condensate`)
#     at the droplets' vapor relaxation rate `εˡ`. With `ξˡ` the psychrometric correction, the
#     saturation excess of total water `s̄ = qᵛ - qᵛ⁺ˡ + ξˡ qᶜˡ` is unchanged by condensation,
#     the vapor excess of a cloud `c` is `s̄ - ξˡ c`, and the cloud grows at
#     `εˡ (s̄ - ξˡ c) / ξˡ = εˡ (s̄ / ξˡ - c)`: a relaxation toward `s̄ / ξˡ` at rate `εˡ`;
#   * rain condenses and evaporates on the vapor excess `s̄ - ξˡ c̃` at its rate `εʳ`.
#
# Cloud and rain draw on one vapor excess, so they are relaxed together
# (`AtmosphereModels.shared_excess_relaxation_rates`): the excess decays as
# `x⁺ = x / (1 + Δt (εˡ + εʳ))`, split between them by `εᵢ / Σε`. Each rate is a fraction
# `εᵢ Δt / (1 + Δt Σε) < 1` of its own excess, so neither overshoots its target at any `ε Δt`.
# The grid kernel does not weight these transfers by the Patankar fraction of their donor, which
# would limit them twice: it draws them first, and advances the explicit sinks of the donor in
# Patankar form against what they leave (`p3_sink_fractions`). Cloud evaporation is bounded by the
# cloud the stage leaves by construction. Rain evaporation relaxes the vapor deficit, not the
# rain, so it is limited there to the rain the stage leaves. Neither is budgeted against the
# cloud and rain of the state the rates see over `sink_limiting_timescale`, as the explicit
# rates are (`compute_p3_process_rates`): that budget would cap them again.
#
# Activation relaxes the droplet number toward the activated number over the activation time
# scale in the same way, with the seed mass of the activated droplets. Its target is capped by
# the aerosol available, so it asks for no more than the reservoir holds.
#
# Ice deposition, the liquid coating of ice and ice nucleation are left to the explicit rates and
# their caps in `limit_ice_vapor_rates`. Where ice shares the vapor, the liquid and ice exchange
# should be one 2×2 backward-Euler solve, for the cloud excess toward its equilibrium condensate
# and the ice excess toward ice saturation, which has a closed form; it would replace the call
# to `shared_excess_relaxation_rates` in `implicit_liquid_vapor_rates` below and the call to
# `limit_ice_vapor_rates` that follows it in `compute_p3_process_rates`.

"""
$(TYPEDEF)

What the coming stage starts from for the implicit liquid relaxation: the stage interval `Δt`,
and the cloud liquid mass `qᶜˡ` and number `nᶜˡ` per unit mass that the stage start and every
other tendency leave. `active` is false before the first time step, when the stage interval is
not known; the explicit rates are used then.
"""
struct LiquidRelaxationStage{FT, B}
    Δt :: FT
    qᶜˡ :: FT
    nᶜˡ :: FT
    active :: B
end

# A stage that is already evaluated (single-cell tests). The grid kernel passes one that reads
# the tendencies where the relaxation needs them; see `GridLiquidRelaxationStage`.
@inline liquid_relaxation_stage(stage::LiquidRelaxationStage, aerosol, ρ) = stage

# Whether the liquid vapor exchange is relaxed implicitly over the coming stage: not for the
# gridless rates, and not before the first time step.
@inline liquid_relaxation_is_active(::Nothing) = false
@inline liquid_relaxation_is_active(stage::LiquidRelaxationStage) = stage.active

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The cloud liquid mass per unit mass that cloud liquid relaxes toward, given the
`saturation_excess` `s̄ = qᵛ - qᵛ⁺ˡ + ξˡ qᶜˡ` of total water, the psychrometric correction `ξˡ`,
and the subgrid condensation correlation `Γ = ⟨s⁺⟩ - s̄` of the `condensation` option
(see `cloud_condensation_correlation`).

On the grid mean (`Γ = nothing`) a cell holds condensate where it is saturated on the mean,
`max(0, s̄) / ξˡ`. With subgrid condensation it holds the equilibrium condensate of its
distribution of saturation excess, `⟨s⁺⟩ / ξˡ = max(0, s̄ + Γ) / ξˡ`, which is non-negative too,
so the relaxation keeps its bounds.
"""
@inline liquid_equilibrium_condensate(::Nothing, saturation_excess, ξˡ) = max(0, saturation_excess) / ξˡ
@inline liquid_equilibrium_condensate(Γ, saturation_excess, ξˡ) = max(0, saturation_excess + Γ) / ξˡ

# Gridless rates (parcels, tests): the explicit caps of `limit_liquid_vapor_rates`.
@inline implicit_liquid_vapor_rates(::Nothing, liquid_rates, args...) = liquid_rates

# `liquid_rates` are the explicit liquid-phase rates with their caps, which the implicit ones
# replace where the stage interval is known; the liquid coating of ice keeps the explicit rates.
# `cloud_relaxation` and `rain_relaxation` are the relaxation coefficients that the coupled
# explicit rates were built with (`coupled_saturation_adjustment_rates`), at the same state.
@inline function implicit_liquid_vapor_rates(liquid_stage, liquid_rates, p3, ρ, ℳ, state, constants,
                                             cloud_water_adjustment, cloud_relaxation, rain_relaxation, Γ,
                                             distribution = nothing)
    FT = typeof(ρ)
    parameters = p3.process_rates
    τ = parameters.sink_limiting_timescale
    T = state.T
    qᵛ = state.qᵛ
    qᵛ⁺ˡ = state.qᵛ⁺ˡ

    # The psychrometric correction of the coupled explicit rates, evaluated at the same state.
    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    ξˡ = liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)

    # The saturation excess of total water and the cloud's equilibrium condensate. No model of
    # condensation is attached to P3 here, so the hook is called with `nothing`.
    vapor_excess = qᵛ - qᵛ⁺ˡ
    saturation_excess = vapor_excess + ξˡ * max(0, ℳ.qᶜˡ)
    equilibrium_condensate = liquid_equilibrium_condensate(Γ, saturation_excess, ξˡ)

    # Tiny cloud in subsaturated air evaporates no slower than over `sink_limiting_timescale`
    subsaturated = vapor_excess / max(qᵛ⁺ˡ, FT(parameters.floors.divisor)) <
                   -parameters.subsaturation_evaporation_threshold
    tiny_cloud = subsaturated & (ℳ.qᶜˡ < parameters.tiny_mass_evaporation_threshold) & (ℳ.qᶜˡ > 0)
    cloud_relaxation = ifelse(tiny_cloud, max(cloud_relaxation, 1 / τ), cloud_relaxation)

    # Read the stage here, where it is first needed, rather than carrying it through the rates.
    # With a prognostic vapor excess the rates see a state whose cloud is shifted by
    # `cloud_water_adjustment` to align the vapor excess with the prognostic one (`ℳ` and `state`
    # are that state), and the alignment is applied as its own rate. The cloud the stage leaves
    # is shifted likewise, so the relaxation is driven by the aligned vapor excess, as the
    # explicit rates are, and does not also carry out the alignment.
    stage = liquid_relaxation_stage(liquid_stage, p3.aerosol, ρ)
    Δt = stage.Δt
    c̃ = stage.qᶜˡ + cloud_water_adjustment
    cloud_excess = equilibrium_condensate - c̃
    rain_excess = saturation_excess / ξˡ - c̃
    cloud_rate, rain_rate = AM.shared_excess_relaxation_rates((cloud_excess, rain_excess),
                                                              (cloud_relaxation, rain_relaxation), Δt)

    # Tiny rain in subsaturated air: the backstop of the explicit rates (no target)
    tiny_rain = subsaturated & (ℳ.qʳ < parameters.tiny_mass_evaporation_threshold)
    rain_cond = ifelse(tiny_rain, zero(FT), max(0, rain_rate))
    rain_evap = ifelse(tiny_rain, max(0, ℳ.qʳ) / τ, max(0, -rain_rate))

    # With a subgrid distribution, activation acts in the part of the cell above its threshold,
    # at the mean vapor excess there (`activation_vapor`)
    activating = activation_vapor(distribution, qᵛ, qᵛ⁺ˡ, activation_relative_threshold(p3.aerosol, p3))
    activation = implicit_cloud_droplet_activation(p3.aerosol, p3, Δt, c̃, stage.nᶜˡ, ℳ,
                                                   activating.qᵛ, qᵛ⁺ˡ, ξˡ, T, ρ)
    activation = (; mass = activation.mass * activating.fraction,
                    number = activation.number * activating.fraction)

    a = stage.active
    return (; cond = ifelse(a, cloud_rate, liquid_rates.cond),
            ccn_activation_mass = ifelse(a, activation.mass, liquid_rates.ccn_activation_mass),
            ccn_activation_number = ifelse(a, activation.number, liquid_rates.ccn_activation_number),
            rain_cond = ifelse(a, rain_cond, liquid_rates.rain_cond),
            rain_evap = ifelse(a, rain_evap, liquid_rates.rain_evap),
            coat_cond = liquid_rates.coat_cond,
            coat_evap = liquid_rates.coat_evap)
end

#####
##### Implicit activation
#####

# Prescribed droplet number: the seed mass of the prescribed droplets, no more than the vapor
# excess can supply, relaxed toward over `sink_limiting_timescale` (the activation time scale
# of this path).
@inline function implicit_cloud_droplet_activation(::Nothing, p3, Δt, c̃, ñ, ℳ, qᵛ, qᵛ⁺ˡ, ξˡ, T, ρ)
    FT = typeof(ρ)
    parameters = p3.process_rates
    τ = parameters.sink_limiting_timescale
    seed_target = p3.cloud.number_concentration / ρ * activated_droplet_mass(parameters, FT)
    c̃ = max(0, c̃)
    target = min(seed_target, c̃ + max(0, (qᵛ - qᵛ⁺ˡ) / ξˡ))
    mass = max(0, AM.implicit_relaxation_rate(c̃, target, 1 / τ, Δt))
    S = (qᵛ - qᵛ⁺ˡ) / max(qᵛ⁺ˡ, parameters.floors.saturation_mass_fraction)
    supersaturated = S > parameters.activation_supersaturation_threshold
    return (; mass = ifelse(supersaturated, mass, zero(FT)), number = zero(FT))
end

@inline activation_pool(aerosol::AerosolActivation{<:Any, false}, nᵃ) = sum_aerosol_number(aerosol)
@inline activation_pool(aerosol::AerosolActivation{<:Any, true}, nᵃ) = nᵃ

# Aerosol activation: the droplet number relaxes toward the activated number over
# `activation_timescale`, capped by the aerosol available, with the seed mass of each droplet.
@inline function implicit_cloud_droplet_activation(aerosol::AerosolActivation, p3, Δt, c̃, ñ, ℳ,
                                                   qᵛ, qᵛ⁺ˡ, ξˡ, T, ρ)
    FT = typeof(ρ)
    S = (qᵛ - qᵛ⁺ˡ) / max(qᵛ⁺ˡ, aerosol.minimum_saturation_mass_fraction)
    ñ = max(0, ñ)
    target = min(total_activated_number(aerosol, T, S), ñ + max(0, activation_pool(aerosol, ℳ.nᵃ)))
    number = max(0, AM.implicit_relaxation_rate(ñ, target, 1 / aerosol.activation_timescale, Δt))
    seed_mass = 4 * FT(π) / 3 * aerosol.liquid_water_density * aerosol.activated_droplet_radius^3
    supersaturated = S > aerosol.activation_supersaturation_threshold
    number = ifelse(supersaturated, number, zero(FT))
    return (; mass = number * seed_mass, number)
end
