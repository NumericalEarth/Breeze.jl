#####
##### Ice nucleation (deposition and immersion freezing)
#####

"""
$(TYPEDSIGNATURES)

Compute ice nucleation rate from deposition/condensation freezing.

New ice crystals nucleate when temperature is below a threshold and the air
is supersaturated with respect to ice. Uses [Cooper (1986)](@cite Cooper1986).
The process is gated on cloud-side ice supersaturation. Breeze carries no subgrid
cloud fraction, which is the `SCF = 1` limit: the grid cell is treated as uniformly
cloudy, so the vapor passed in is both the grid-mean and the cloud-side value. If a
subgrid cloud-fraction path is added, pass the cloud-side vapor state here rather
than a grid-mean vapor state.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `T`: Temperature [K]
- `qᵛ`: Cloud-side vapor mass fraction [kg/kg] (grid-mean when `SCF = 1`)
- `qᵛ⁺ⁱ`: Saturation vapor mass fraction over ice [kg/kg]
- `nⁱ`: Current ice number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Tuple `(nucleated_mass_rate, nucleated_number_rate)`: mass rate [kg/kg/s]
  and number rate [1/kg/s]
"""
@inline function deposition_nucleation_rate(p3, T, qᵛ, qᵛ⁺ⁱ, nⁱ, ρ)
    FT = typeof(T)
    parameters = p3.process_rates

    nucleation_temperature_threshold = parameters.ice_nucleation_temperature_threshold
    supersaturation_threshold = parameters.ice_nucleation_supersaturation_threshold
    maximum_concentration = parameters.maximum_ice_nucleation_concentration
    nucleation_timescale = parameters.ice_nucleation_timescale
    freezing_temperature = parameters.freezing_temperature
    nucleated_ice_mass = parameters.nucleated_ice_mass
    nucleation_coefficient = parameters.ice_nucleation_coefficient
    temperature_coefficient = parameters.ice_nucleation_temperature_coefficient
    floors = parameters.floors

    # Ice supersaturation, evaluated cloud-side; in the `SCF = 1` limit `qᵛ` is
    # both the grid-mean and the cloud-side value.
    Sⁱ = (qᵛ - qᵛ⁺ⁱ) / max(qᵛ⁺ⁱ, floors.saturation_mass_fraction)

    # Conditions for nucleation
    # m6: the supersaturation threshold is inclusive
    nucleation_active = (T < nucleation_temperature_threshold) &
                        (Sⁱ >= supersaturation_threshold)

    # Cooper (1986): N_ice = c_nuc × exp(b_nuc × (T₀ - T)) [1/m³]
    # Default c_nuc = 5.0 /m³ = 0.005 /L from Cooper (1986), divided by ρ for [1/kg]
    supercooling = freezing_temperature - T
    cooper_number = nucleation_coefficient *
                    exp(temperature_coefficient * supercooling) / ρ

    # Limit to maximum and subtract existing ice
    equilibrium_number = min(cooper_number, maximum_concentration / ρ)

    # Nucleation rate: relaxation toward equilibrium
    nucleated_number_rate = clamp_positive(equilibrium_number - nⁱ) / nucleation_timescale

    # Mass nucleation rate
    nucleated_mass_rate = nucleated_number_rate * nucleated_ice_mass

    # Use one threshold on the number rate for both moments.
    active = nucleation_active & (nucleated_number_rate >= floors.rate_scale)
    nucleated_number_rate = ifelse(active, nucleated_number_rate, zero(FT))
    nucleated_mass_rate = ifelse(active, nucleated_mass_rate, zero(FT))

    return nucleated_mass_rate, nucleated_number_rate
end

@inline function immersion_freezing_rate_coefficient(nucleation_rate_coefficient,
                                                      droplet_volume,
                                                      temperature_exponent_coefficient,
                                                      supercooling,
                                                      maximum_multiplier,
                                                      divisor_floor)
    FT = typeof(nucleation_rate_coefficient + droplet_volume +
                temperature_exponent_coefficient + supercooling + maximum_multiplier)
    log_rate = log(max(nucleation_rate_coefficient * droplet_volume, FT(divisor_floor))) +
               temperature_exponent_coefficient * supercooling
    # Apply only a common numerical overflow guard. The species-level budget
    # later scales both raw moments together; capping here at 1/τ would prevent
    # newly condensed liquid from participating in the combined budget.
    maximum_rate = sqrt(floatmax(FT)) / max(maximum_multiplier, one(FT))
    return exp(min(log_rate, log(maximum_rate)))
end

"""
$(TYPEDSIGNATURES)

Compute immersion freezing rate of cloud droplets using the
[Barklie and Gokhale (1959)](@cite BarklieGokhale1959) stochastic volume-dependent
freezing parameterization.

The probability per droplet per second of freezing is ``J₀ V_{\\text{drop}} \\exp(a ΔT)``,
where ``J₀ ≈ 2`` m⁻³s⁻¹ is the nucleation rate coefficient (``a = 0.65``) and
``V_{\\text{drop}}`` is the individual droplet volume. For monodisperse cloud droplets
this gives a mass freezing rate proportional to ``(q^{cl})^2 / N^{cl}``, making freezing
negligible for small droplets.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜˡ`: Cloud droplet number concentration [1/m³]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]

# Returns
- Tuple `(frozen_mass_rate, frozen_number_rate)`: mass rate [kg/kg/s] and
  number rate [1/kg/s]
"""
@inline function immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜˡ, T, ρ)
    FT = typeof(qᶜˡ)
    parameters = p3.process_rates

    maximum_temperature = parameters.maximum_immersion_freezing_temperature
    temperature_exponent_coefficient = parameters.immersion_freezing_coefficient
    freezing_temperature = parameters.freezing_temperature
    ρᴸ = FT(parameters.liquid_water_density)
    nucleation_rate_coefficient = parameters.immersion_freezing_nucleation_coefficient
    floors = parameters.floors

    # Compute μᶜˡ dynamically from local Nᶜˡ (already [1/m³]) via Liu-Daum (2000),
    # then derive the PSD correction C(μᶜˡ) = Γ(μᶜˡ+7)Γ(μᶜˡ+1)/Γ(μᶜˡ+4)².
    # This replaces the precomputed construction-time value, allowing the correction
    # to vary spatially with the local droplet population.
    μᶜˡ = liu_daum_shape_parameter(Nᶜˡ)
    psd_correction = psd_correction_spherical_volume(μᶜˡ)

    qᶜˡ_eff = clamp_positive(qᶜˡ)

    # Conditions for freezing
    freezing_active = (T <= maximum_temperature) & (qᶜˡ_eff >= p3.minimum_mass_mixing_ratio)

    # Barklie-Gokhale (1959) stochastic immersion freezing.
    # Per-drop freezing probability is J₀ times droplet volume times exp(a ΔT).
    # For a gamma PSD, the PSD-integrated mass rate is boosted by C(μᶜˡ),
    # but the number rate has C_N = 1 (no PSD correction).
    supercooling = max(freezing_temperature - T, zero(FT))

    # Individual droplet mass and volume (monodisperse assumption)
    # Nᶜˡ is [1/m³]; convert to per-kg: nᶜˡ = Nᶜˡ/ρ [1/kg]
    nᶜˡ = max(Nᶜˡ / ρ, p3.minimum_number_mixing_ratio)
    droplet_mass = qᶜˡ_eff / nᶜˡ  # [kg]
    droplet_volume = droplet_mass / ρᴸ   # [m³]

    # The per-drop freezing coefficient (NO psd_correction) is a linear per-second
    # rate. The log form avoids overflow at very low temperatures. Any numerical cap
    # is applied equally before the two moment products so their ratio is unchanged.
    # The PSD correction applies only to the mass (6th moment) rate, not to the
    # number (3rd moment) rate.
    maximum_multiplier = max(qᶜˡ_eff * psd_correction, nᶜˡ)
    freezing_rate = immersion_freezing_rate_coefficient(
        nucleation_rate_coefficient, droplet_volume, temperature_exponent_coefficient,
        supercooling, maximum_multiplier, floors.divisor)

    # Form the raw rates. The combined cloud budget in compute_p3_process_rates
    # applies one factor to qcheti and ncheti together, after condensation and
    # competing sinks have been included.
    frozen_mass_rate = qᶜˡ_eff * psd_correction * freezing_rate
    frozen_number_rate = nᶜˡ * freezing_rate

    frozen_mass_rate = ifelse(freezing_active, frozen_mass_rate, zero(FT))
    frozen_number_rate = ifelse(freezing_active, frozen_number_rate, zero(FT))

    return frozen_mass_rate, frozen_number_rate
end

"""
$(TYPEDSIGNATURES)

Compute immersion freezing rate of rain drops.

Rain drops freeze when temperature is below a threshold. Uses
[Barklie and Gokhale (1959)](@cite BarklieGokhale1959) stochastic freezing
parameterization.

The PSD correction ``C(\\mu_r) = \\Gamma(\\mu_r+7)\\Gamma(\\mu_r+1)/\\Gamma(\\mu_r+4)^2``
is computed from the actual rain shape parameter ``\\mu_r``, not from a fixed value.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `T`: Temperature [K]
- `μʳ`: Rain PSD shape parameter [-] (0 for exponential)

# Returns
- Tuple `(frozen_mass_rate, frozen_number_rate)`: mass rate [kg/kg/s] and
  number rate [1/kg/s]
"""
@inline function immersion_freezing_rain_rate(p3, qʳ, nʳ, T, μʳ)
    FT = typeof(qʳ)
    parameters = p3.process_rates

    maximum_temperature = parameters.maximum_immersion_freezing_temperature
    temperature_exponent_coefficient = parameters.immersion_freezing_coefficient
    freezing_temperature = parameters.freezing_temperature
    ρᴸ = FT(parameters.liquid_water_density)
    nucleation_rate_coefficient = parameters.immersion_freezing_nucleation_coefficient
    floors = parameters.floors

    # Compute the PSD correction from the diagnosed rain shape parameter.
    psd_correction = psd_correction_spherical_volume(μʳ)

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # Conditions for freezing
    freezing_active = (T <= maximum_temperature) & (qʳ_eff >= p3.minimum_mass_mixing_ratio)

    # Barklie-Gokhale (1959) stochastic volume-dependent freezing.
    supercooling = max(freezing_temperature - T, zero(FT))

    # Individual rain drop mass and volume (monodisperse assumption)
    safe_rain_number = max(nʳ_eff, p3.minimum_number_mixing_ratio)
    droplet_mass = qʳ_eff / safe_rain_number  # [kg]
    droplet_volume = droplet_mass / ρᴸ       # [m³]

    # The per-drop freezing coefficient (NO psd_correction) is a linear per-second
    # rate. The log form avoids overflow at very low temperatures. Any numerical cap
    # is applied equally before the two moment products so their ratio is unchanged.
    # The PSD correction applies only to the mass (6th moment) rate, not to the
    # number (3rd moment) rate.
    maximum_multiplier = max(qʳ_eff * psd_correction, nʳ_eff)
    freezing_rate = immersion_freezing_rate_coefficient(
        nucleation_rate_coefficient, droplet_volume, temperature_exponent_coefficient,
        supercooling, maximum_multiplier, floors.divisor)

    # The combined rain budget later applies the same limiter to qrheti and
    # nrheti, preserving the mean mass of preferentially frozen drops.
    frozen_mass_rate = qʳ_eff * psd_correction * freezing_rate
    frozen_number_rate = nʳ_eff * freezing_rate

    frozen_mass_rate = ifelse(freezing_active, frozen_mass_rate, zero(FT))
    frozen_number_rate = ifelse(freezing_active, frozen_number_rate, zero(FT))

    return frozen_mass_rate, frozen_number_rate
end

#####
##### Homogeneous freezing
#####

"""
$(TYPEDSIGNATURES)

Compute homogeneous freezing rate of cloud droplets.

Below −40°C (233.15 K) all supercooled cloud liquid freezes instantaneously.
The frozen mass deposits as dense rime at ``ρ_{\\text{rim}} = 900`` kg/m³
(solid ice sphere), following
[Morrison and Milbrandt (2015)](@cite Morrison2015parameterization).

All cloud droplets are transferred to ice; the number rate is
``N_{\\text{hom}} = N^{cl} / (ρ τ_{\\text{hom}})``.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nᶜˡ`: Cloud droplet number concentration [1/m³]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]

# Returns
- Tuple `(frozen_mass_rate, frozen_number_rate)` containing the cloud-to-ice
  mass rate [kg/kg/s] and number rate [1/kg/s]

# Example

```jldoctest
using Logging
using Breeze.Microphysics.PredictedParticleProperties:
    homogeneous_freezing_cloud_rate
p3 = with_logger(NullLogger()) do
    PredictedParticlePropertiesMicrophysics()
end
Q, N = homogeneous_freezing_cloud_rate(p3, 1e-3, 100e6, 230.0, 1.2)
round.((Q, N), sigdigits=4)

# output
(0.0001, 8.333e6)
```
"""
@inline function homogeneous_freezing_cloud_rate(p3, qᶜˡ, Nᶜˡ, T, ρ)
    FT = typeof(qᶜˡ)
    parameters = p3.process_rates

    freezing_temperature = FT(parameters.homogeneous_freezing_temperature)
    freezing_timescale = FT(parameters.homogeneous_freezing_timescale)

    qᶜˡ_eff = clamp_positive(qᶜˡ)

    # Guard: temperature below threshold AND sufficient cloud liquid present
    freezing_active = (T < freezing_temperature) &
                      (qᶜˡ_eff >= p3.minimum_mass_mixing_ratio)

    # Instantaneous conversion: rate = mixing ratio / timescale
    frozen_mass_rate = qᶜˡ_eff / freezing_timescale

    # Number rate: Nᶜˡ is [1/m³] → divide by ρ for [1/kg]
    frozen_number_rate = Nᶜˡ / ρ / freezing_timescale

    # No mass-number consistency cap: the whole cloud population is transferred to
    # ice below the homogeneous freezing threshold.
    frozen_mass_rate = ifelse(freezing_active, frozen_mass_rate, zero(FT))
    frozen_number_rate = ifelse(freezing_active, frozen_number_rate, zero(FT))

    return frozen_mass_rate, frozen_number_rate
end

"""
$(TYPEDSIGNATURES)

Compute homogeneous freezing rate of rain drops.

Below −40°C (233.15 K) all supercooled rain freezes instantaneously.
The frozen mass deposits as dense rime at ``ρ_{\\text{rim}} = 900`` kg/m³,
following
[Morrison and Milbrandt (2015)](@cite Morrison2015parameterization).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `T`: Temperature [K]

# Returns
- Tuple `(frozen_mass_rate, frozen_number_rate)` containing the rain-to-ice
  mass rate [kg/kg/s] and number rate [1/kg/s]

# Example

```jldoctest
using Logging
using Breeze.Microphysics.PredictedParticleProperties:
    homogeneous_freezing_rain_rate
p3 = with_logger(NullLogger()) do
    PredictedParticlePropertiesMicrophysics()
end
Q, N = homogeneous_freezing_rain_rate(p3, 1e-3, 1e4, 220.0)
round.((Q, N), sigdigits=4)

# output
(0.0001, 1000.0)
```
"""
@inline function homogeneous_freezing_rain_rate(p3, qʳ, nʳ, T)
    FT = typeof(qʳ)
    parameters = p3.process_rates

    freezing_temperature = FT(parameters.homogeneous_freezing_temperature)
    freezing_timescale = FT(parameters.homogeneous_freezing_timescale)

    qʳ_eff = clamp_positive(qʳ)

    # Guard: temperature below threshold AND sufficient rain present
    freezing_active = (T < freezing_temperature) &
                      (qʳ_eff >= p3.minimum_mass_mixing_ratio)

    # Instantaneous conversion: rate = mixing ratio / timescale
    frozen_mass_rate = qʳ_eff / freezing_timescale

    # Number rate: nʳ already in [1/kg]
    frozen_number_rate = clamp_positive(nʳ) / freezing_timescale

    frozen_mass_rate = ifelse(freezing_active, frozen_mass_rate, zero(FT))
    frozen_number_rate = ifelse(freezing_active, frozen_number_rate, zero(FT))

    return frozen_mass_rate, frozen_number_rate
end

#####
##### Rime splintering (Hallett-Mossop secondary ice production)
#####

"""
$(TYPEDSIGNATURES)

Compute secondary ice production from rime splintering (Hallett-Mossop effect).

When rimed ice particles accrete supercooled drops, ice splinters are
ejected. This occurs only in a narrow temperature range around -5°C.
See [Hallett and Mossop (1974)](@cite HallettMossop1974).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `cloud_riming`: Cloud droplet riming rate [kg/kg/s]
- `rain_riming`: Rain riming rate [kg/kg/s]
- `T`: Temperature [K]
- `D_ice`: Mean ice diameter [m]
- `Fˡ`: Liquid fraction on ice [-]
- `surface_T`: Surface-temperature proxy for the warm-season shutoff [K]
- `qᶠ`: Existing rimed-ice mass [kg/kg]

# Returns
- Tuple (qᶜˡ_splintering_rate, qʳ_splintering_rate, n_splintering_rate): the
  cloud- and rain-branch ice mass rates [kg/kg/s] and the total number rate [1/kg/s]
"""
@inline function rime_splintering_rates(p3, cloud_riming, rain_riming, T, D_ice, Fˡ, surface_T, qᶠ)
    FT = typeof(T)
    parameters = p3.process_rates

    minimum_temperature = parameters.minimum_splintering_temperature
    maximum_temperature = parameters.maximum_splintering_temperature
    T_peak = parameters.splintering_temperature_peak
    c_splinter = parameters.splintering_rate
    # Use the Hallett-Mossop splinter crystal mass (D = 10 μm), NOT the nucleated
    # ice mass (D = 2 μm). Splinters are 125× heavier.
    mᵢ₀ = parameters.splintering_crystal_mass

    warm_branch = clamp((T - minimum_temperature) / (T_peak - minimum_temperature), zero(FT), one(FT))
    cold_branch = clamp((maximum_temperature - T) / (maximum_temperature - T_peak), zero(FT), one(FT))
    efficiency = ifelse(T <= T_peak, warm_branch, cold_branch)

    # Cloud-riming splintering applies only with a single ice category; with more
    # than one, splintering_cloud_riming_scale = 0 disables it.
    cloud_riming_eff = clamp_positive(cloud_riming) * FT(parameters.splintering_cloud_riming_scale)
    rain_riming_eff = clamp_positive(rain_riming)
    has_rime = qᶠ >= p3.minimum_mass_mixing_ratio
    active = (D_ice ≥ parameters.splintering_diameter_threshold) &
             has_rime &
             (Fˡ < parameters.maximum_splintering_liquid_fraction) &
             (surface_T < parameters.maximum_splintering_surface_temperature)

    nᶜˡ_splintering_rate = ifelse(active, efficiency * c_splinter * cloud_riming_eff, zero(FT))
    nʳ_splintering_rate = ifelse(active, efficiency * c_splinter * rain_riming_eff, zero(FT))
    n_splintering_rate = nᶜˡ_splintering_rate + nʳ_splintering_rate

    qᶜˡ_splintering_rate = nᶜˡ_splintering_rate * mᵢ₀
    qʳ_splintering_rate = nʳ_splintering_rate * mᵢ₀

    return qᶜˡ_splintering_rate, qʳ_splintering_rate, n_splintering_rate
end

@inline function rime_splintering_rate(p3, cloud_riming, rain_riming, T, D_ice, Fˡ, surface_T, qᶠ)
    qᶜˡ_splintering_rate, qʳ_splintering_rate, n_splintering_rate =
        rime_splintering_rates(p3, cloud_riming, rain_riming, T, D_ice, Fˡ, surface_T, qᶠ)
    return qᶜˡ_splintering_rate + qʳ_splintering_rate, n_splintering_rate
end
