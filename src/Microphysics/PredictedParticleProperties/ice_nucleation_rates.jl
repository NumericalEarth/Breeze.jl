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
    nucleated_number_rate = max(0, equilibrium_number - nⁱ) / nucleation_timescale

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
    # Evaluated in the log domain, and not as `per_drop * exp(a ΔT)`: at large
    # supercooling `exp(a ΔT)` overflows on its own while the product with the tiny
    # per-drop coefficient stays well in range, so forming the product directly
    # would saturate to `Inf` and then clamp to `maximum_rate`. In `Float32` with a
    # per-drop coefficient of 1e-25 and `a ΔT = 97.5` that is an 82x error.
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

[Barklie and Gokhale (1959)](@cite BarklieGokhale1959) stochastic immersion
freezing, shared by the cloud and rain paths.

The per-drop freezing probability is `J₀` times the drop volume times
`exp(a ΔT)`, evaluated on a monodisperse drop of mass `q / n_for_mass`. For a
gamma PSD the mass (6th moment) rate carries the correction
``C(μ) = Γ(μ+7)Γ(μ+1)/Γ(μ+4)²``; the number (3rd moment) rate does not.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `q`: Condensate mass fraction [kg/kg]
- `n_for_mass`: Number per unit mass used for the monodisperse drop mass [1/kg];
  floored, so the drop mass stays finite where the population vanishes
- `n_for_rate`: Number per unit mass the number rate scales with [1/kg]. Cloud
  passes the floored number here as well; rain passes the unfloored one, so a
  vanishing rain population produces no number rate
- `μ`: PSD shape parameter [-]
- `T`: Temperature [K]

# Returns
- Tuple `(frozen_mass_rate, frozen_number_rate)`: mass rate [kg/kg/s] and
  number rate [1/kg/s]
"""
@inline function stochastic_immersion_freezing(p3, q, n_for_mass, n_for_rate, μ, T)
    FT = typeof(q)
    parameters = p3.process_rates

    q_eff = max(0, q)
    psd_correction = psd_correction_spherical_volume(μ)

    freezing_active = (T <= parameters.maximum_immersion_freezing_temperature) &
                      (q_eff >= p3.minimum_mass_mixing_ratio)

    supercooling = max(parameters.freezing_temperature - T, zero(FT))

    # Individual drop mass and volume (monodisperse assumption)
    droplet_volume = q_eff / n_for_mass / FT(parameters.liquid_water_density)

    # The per-drop freezing coefficient (NO psd_correction) is a linear per-second
    # rate. Any numerical cap is applied equally before the two moment products so
    # their ratio is unchanged. The species-level budget in
    # `compute_p3_process_rates` applies one limiting factor to the mass and number
    # rates together, after condensation and competing sinks are included.
    maximum_multiplier = max(q_eff * psd_correction, n_for_rate)
    freezing_rate = immersion_freezing_rate_coefficient(
        parameters.immersion_freezing_nucleation_coefficient, droplet_volume,
        parameters.immersion_freezing_coefficient, supercooling,
        maximum_multiplier, parameters.floors.divisor)

    return (ifelse(freezing_active, q_eff * psd_correction * freezing_rate, zero(FT)),
            ifelse(freezing_active, n_for_rate * freezing_rate, zero(FT)))
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
    # μᶜˡ comes from the local Nᶜˡ (already [1/m³]) via Liu-Daum (2000) rather than
    # from a construction-time value, so the PSD correction varies with the local
    # droplet population. The relation itself is read from `p3.cloud.shape_parameters`,
    # the same container the construction-time and prognostic paths use.
    # Nᶜˡ is volumetric; nᶜˡ = Nᶜˡ/ρ is per-mass.
    nᶜˡ = max(Nᶜˡ / ρ, p3.minimum_number_mixing_ratio)
    μᶜˡ = liu_daum_shape_parameter(Nᶜˡ, p3.cloud.shape_parameters)
    return stochastic_immersion_freezing(p3, qᶜˡ, nᶜˡ, nᶜˡ, μᶜˡ, T)
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
    # nʳ is already per-mass. The drop mass needs a floored number to stay finite,
    # but the number rate scales with the unfloored one.
    nʳ_eff = max(0, nʳ)
    return stochastic_immersion_freezing(p3, qʳ, max(nʳ_eff, p3.minimum_number_mixing_ratio),
                                         nʳ_eff, μʳ, T)
end

#####
##### Homogeneous freezing
#####

"""
$(TYPEDSIGNATURES)

Instantaneous homogeneous freezing, shared by the cloud and rain paths: below the
homogeneous freezing threshold the whole population is transferred to ice over
`homogeneous_freezing_timescale`, with no mass-number consistency cap. `n` is a
number per unit mass [1/kg].
"""
@inline function homogeneous_freezing_rate(p3, q, n, T)
    FT = typeof(q)
    parameters = p3.process_rates

    freezing_temperature = FT(parameters.homogeneous_freezing_temperature)
    freezing_timescale = FT(parameters.homogeneous_freezing_timescale)

    q_eff = max(0, q)
    freezing_active = (T < freezing_temperature) &
                      (q_eff >= p3.minimum_mass_mixing_ratio)

    return (ifelse(freezing_active, q_eff / freezing_timescale, zero(FT)),
            ifelse(freezing_active, n / freezing_timescale, zero(FT)))
end

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
using Breeze.Microphysics.PredictedParticleProperties:
    homogeneous_freezing_cloud_rate
p3 = PredictedParticlePropertiesMicrophysics()
Q, N = homogeneous_freezing_cloud_rate(p3, 1e-3, 100e6, 230.0, 1.2)
round.((Q, N), sigdigits=4)

# output
(0.0001, 8.333e6)
```
"""
@inline homogeneous_freezing_cloud_rate(p3, qᶜˡ, Nᶜˡ, T, ρ) =
    homogeneous_freezing_rate(p3, qᶜˡ, Nᶜˡ / ρ, T)  # Nᶜˡ is [1/m³]; nᶜˡ = Nᶜˡ/ρ

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
using Breeze.Microphysics.PredictedParticleProperties:
    homogeneous_freezing_rain_rate
p3 = PredictedParticlePropertiesMicrophysics()
Q, N = homogeneous_freezing_rain_rate(p3, 1e-3, 1e4, 220.0)
round.((Q, N), sigdigits=4)

# output
(0.0001, 1000.0)
```
"""
@inline homogeneous_freezing_rain_rate(p3, qʳ, nʳ, T) =
    homogeneous_freezing_rate(p3, qʳ, max(0, nʳ), T)  # nʳ is already [1/kg]

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
    cloud_riming_eff = max(0, cloud_riming) * FT(parameters.splintering_cloud_riming_scale)
    rain_riming_eff = max(0, rain_riming)
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
