"""
$(TYPEDSIGNATURES)

Compute the density of newly accreted cloud rime using the Fortran P3 Ri fit.

This follows the `p3_main` cloud-riming branch: diagnose the cloud gamma PSD
from `qᶜˡ` and prescribed `Nᶜ`, compute the droplet impact speed relative to
falling ice, form the rime-impact parameter `Ri`, and apply the same piecewise
fit for `ρ_rime`. When cloud riming is inactive or the air is above freezing,
the Fortran fallback value `400 kg m⁻³` is used.

# Arguments
- `p3`: P3 microphysics scheme
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `cloud_rim`: Cloud-riming mass tendency [kg/kg/s]
- `T`: Temperature [K]
- `vᵢ`: Ice particle fall speed [m/s]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants
- `transport`: Air transport properties at `(T, P)`

# Returns
- Rime density [kg/m³]
"""
function rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport,
                      μ_c, λ_c)
    FT = typeof(T)
    prp = p3.process_rates
    qsmall = p3.minimum_mass_mixing_ratio

    ρ_rim_min = prp.minimum_rime_density
    ρ_rim_max = prp.maximum_rime_density
    T₀ = prp.freezing_temperature
    ρᴸ = prp.liquid_water_density

    # Dynamic viscosity of air. Written `η`, not `μ`, which is the PSD shape
    # parameter everywhere else in this module.
    η = transport.ν * ρ
    g = p3_gravitational_acceleration(constants, FT)

    # Fortran get_cloud_dsd2 / p3_main: the droplet impact speed is the mass-weighted
    # Stokes velocity of the cloud DSD, shared with `cloud_terminal_velocities`.
    a_cn = cloud_stokes_prefactor(g, ρᴸ, η)
    Vt_qc = cloud_mass_weighted_stokes_velocity(a_cn, μ_c, λ_c)
    D_c = (μ_c + 4) / λ_c

    # Riming impact parameter Ri = c Dᶜ |vᵢ - Vt_qc| / (T₀ - T): large drops striking
    # fast at weak supercooling pack dense rime. The supercooling floor keeps Ri
    # finite as T → T₀, and the clamp holds Ri inside the range the fit below covers.
    inverse_supercooling = inv(min(-prp.minimum_riming_supercooling, T - T₀))
    Ri = clamp(-(prp.rime_impact_coefficient * D_c) * abs(vᵢ - Vt_qc) * inverse_supercooling,
               prp.minimum_rime_impact, prp.maximum_rime_impact)

    # Cober-List rime-density fit as coded in Fortran P3 v5.5.0 `p3_main`: a
    # quadratic in g/cm³ (hence the ×10³) below Ri = 8, a linear extension above it.
    #   Ri ≤ 8:  ρᶠ = (0.051 + 0.114 Ri - 0.0055 Ri²) × 10³
    #   Ri > 8:  ρᶠ = 611 + 72.25 (Ri - 8)
    # These are the coefficients of a published fit, not independently tunable
    # parameters, so they stay inline with the formula they belong to. What *is*
    # settable is the range of Ri over which the fit is trusted, clamped above.
    ρ_rime_Ri = ifelse(
        Ri <= FT(8),
        (FT(0.051) + FT(0.114) * Ri - FT(0.0055) * Ri^2) * FT(1000),
        FT(611) + FT(72.25) * (Ri - FT(8))
    )

    active_cloud_riming = (cloud_rim >= qsmall) & (qᶜˡ >= qsmall) & (T < T₀)
    ρᶠ = ifelse(active_cloud_riming, ρ_rime_Ri, prp.unrimed_rime_density)

    return clamp(ρᶠ, ρ_rim_min, ρ_rim_max)
end

# Backward-compatible 8-arg method: uses prescribed cloud DSD (μ_c, Nᶜ from p3.cloud).
# The full 10-arg form takes locally diagnosed (μ_c, λ_c) per Fortran p3_main parity.
function rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport)
    FT = typeof(T)
    μ_c = p3.cloud.shape_parameter
    Nᶜ = p3.cloud.number_concentration
    ρᴸ = p3.process_rates.liquid_water_density
    mass_scale = FT(p3.process_rates.floors.mass_scale)
    qᶜˡ_abs = max(clamp_positive(qᶜˡ) * ρ, mass_scale)
    λ_c = cloud_lambda(Nᶜ, μ_c, qᶜˡ_abs, ρᴸ)
    return rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport, μ_c, λ_c)
end

#####
##### Phase 2: Shedding and Refreezing (liquid fraction dynamics)
#####

"""
$(TYPEDSIGNATURES)

Compute liquid shedding rate from ice particles following
[Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

PSD-integrated shedding of liquid from mixed-phase ice particles with D ≥ 9 mm
(Rasmussen et al. 2011). Matches Fortran P3 v5.5.0:

```math
q_{lshd} = F_r \\times f_{1pr28} \\times N_i \\times F_l
```

where `f1pr28 = ∫_{D≥9mm} m(D) N'(D) dD` (lookup table, Fl-blended mass),
`Fr = qirim / (qitot - qiliq)` is the rime fraction of ice-only mass, and
`Fl = qiliq / qitot` is the liquid fraction.

# Arguments
- `p3`: P3 microphysics scheme (provides shedding table)
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg] (dry ice, excluding qʷⁱ)
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime fraction (= qᶠ/qⁱ) [-]
- `Fˡ`: Liquid fraction (= qʷⁱ/(qⁱ+qʷⁱ)) [-]
- `ρᶠ`: Rime density [kg/m³]
- `m_mean`: Mean ice particle mass [kg]

# Returns
- Rate of liquid → rain shedding [kg/kg/s]
"""
function shedding_rate(p3, qʷⁱ, qⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ, m_mean, μ)
    FT = typeof(qʷⁱ)

    qʷⁱ_eff = clamp_positive(qʷⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # Lookup ∫_{D≥9mm} m(D) N'(D) dD (normalized per particle)
    f1pr28 = shedding_integral(p3.ice.bulk_properties.shedding, m_mean, Fᶠ, Fˡ, ρᶠ, μ,
                               p3.process_rates.floors.mass_scale)

    # Fortran: qlshd = Fr × f1pr28 × ni × Fl
    # Fr = rime fraction of ice-only mass (= Fᶠ in Julia convention since qⁱ excludes qʷⁱ)
    rate = Fᶠ * f1pr28 * nⁱ_eff * Fˡ

    # Bound by available liquid: qlshd ≤ qwi / dt_safety
    rate = clamp_positive(rate)
    τ_safety = p3.process_rates.sink_limiting_timescale
    rate = min(rate, qʷⁱ_eff / τ_safety)

    return rate
end

"""
$(TYPEDSIGNATURES)

Lookup the PSD-integrated shedding mass for D ≥ 9 mm particles
from tabulated `TabulatedFunction5D`.
"""
@inline function shedding_integral(table::P3Table5D, m_mean, Fᶠ, Fˡ, ρᶠ, μ, mass_scale_floor)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, FT(mass_scale_floor)))
    return table(log_m, Fᶠ, Fˡ, ρᶠ, μ)
end

"""
$(TYPEDSIGNATURES)

Compute rain number source from shedding.

Shed liquid forms rain drops of approximately 1 mm diameter.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `shed_rate`: Liquid shedding mass rate [kg/kg/s]

# Returns
- Rate of rain number increase [1/kg/s]
"""
@inline function shedding_number_rate(p3, shed_rate)
    # Liquid-fraction shedding uses 1.928e6 drops/kg (Fortran nlshd, line 3350),
    # slightly different from cloud/wet-growth shedding (1.923e6).
    m_shed = p3.process_rates.shed_drop_mass_liqfrac

    return shed_rate / m_shed
end

"""
$(TYPEDSIGNATURES)

Compute the wet growth freezing capacity following
[Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction).

The wet growth capacity is the maximum rate at which collected
hydrometeors can be frozen, determined by the ventilated heat balance:

```math
q_{wgrth} = C f_v \\left[Kᵃ(T_0-T) + \\frac{2π}{ℒᶠᵘˢ} ℒⁱ Dᵛ(ρ_{vs}-ρ_v)\\right] × N_i
```

When the collection rate (cloud + rain riming) exceeds this capacity,
the excess collected water stays liquid and is redirected into qʷⁱ.

# Arguments
- `p3`: P3 microphysics scheme
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `P`: Pressure [Pa]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants (or `nothing`)
- `transport`: Pre-computed air transport properties `(; Dᵛ, Kᵃ, ν)`

# Returns
- Wet growth capacity [kg/kg/s] (positive; zero when T ≥ T₀)
"""
function wet_growth_capacity(p3, qⁱ, qʷⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport, μ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    T₀ = prp.freezing_temperature
    below_freezing = T < T₀

    ℒᶠᵘˢ = fusion_latent_heat(constants, T)
    ℒⁱ = sublimation_latent_heat(constants, T)

    Kᵃ = transport.Kᵃ
    Dᵛ = transport.Dᵛ
    ν  = transport.ν

    q_sat0 = freezing_point_saturation_mass_fraction(constants, T₀, ρ)

    # Mean ice particle mass
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)

    # Ventilation integral (same as deposition/refreezing)
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                    p3.ice.deposition.ventilation_enhanced,
                                    m_mean, Fᶠ, Fˡ, ρᶠ, prp, ν, Dᵛ, ρ_correction, p3, μ)

    # Heat balance: sensible + latent
    Q_sensible = Kᵃ * (T₀ - T)
    Q_latent = ℒⁱ * Dᵛ * ρ * (q_sat0 - qᵛ)

    # Fortran applies 2π/Lf only to the latent term; the sensible-conduction
    # term uses the capm convention directly.
    qwgrth = C_fv * (Q_sensible + 2 * FT(π) * Q_latent / ℒᶠᵘˢ) * nⁱ_eff

    return ifelse(below_freezing, clamp_positive(qwgrth), zero(FT))
end

"""
$(TYPEDSIGNATURES)

Compute refreezing rate of liquid on ice using the heat-balance formula.

Below freezing, liquid coating on ice particles refreezes. The rate is
determined by the heat flux at the particle surface:

```math
\\frac{dm}{dt} = C f_v \\left[Kᵃ(T_0-T) + \\frac{2π}{ℒᶠᵘˢ} ρ ℒⁱ Dᵛ (q_{sat0} - q_v)\\right]
```

This mirrors the melting formula with reversed temperature gradient.
See [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization)
appendix C, section i (and Mason 1971 for the underlying heat-balance form).

# Arguments
- `p3`: P3 microphysics scheme
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `P`: Pressure [Pa]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants (or `nothing` for Fortran-matched hardcoded values)
- `transport`: Pre-computed air transport properties `(; Dᵛ, Kᵃ, ν)`

# Returns
- Rate of liquid → ice refreezing [kg/kg/s]
"""
function refreezing_rate(p3, qʷⁱ, qⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport, μ)
    FT = typeof(qʷⁱ)
    prp = p3.process_rates

    qʷⁱ_eff = clamp_positive(qʷⁱ)
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    nⁱ_eff  = clamp_positive(nⁱ)

    T₀ = prp.freezing_temperature
    below_freezing = T < T₀
    ΔT = T₀ - T  # positive when below freezing

    ℒᶠᵘˢ = fusion_latent_heat(constants, T)
    ℒⁱ = sublimation_latent_heat(constants, T)

    Kᵃ = transport.Kᵃ
    Dᵛ = transport.Dᵛ
    ν  = transport.ν

    q_sat0 = freezing_point_saturation_mass_fraction(constants, T₀, ρ)

    # Mean ice particle mass
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)

    # Ventilation integral (ice-particle capacitance; same path as deposition)
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                    p3.ice.deposition.ventilation_enhanced,
                                    m_mean, Fᶠ, Fˡ, ρᶠ, prp, ν, Dᵛ, ρ_correction, p3, μ)

    # Heat balance for refreezing:
    # Conductive: Kᵃ × (T₀ - T) removes heat from liquid → promotes freezing
    Q_sensible = Kᵃ * ΔT

    # Vapor: ℒⁱ × Dᵛ × ρ × (q_sat0 - qᵛ)
    # Subsaturated (q_sat0 > qᵛ): evaporation cools particle → promotes freezing
    # Supersaturated (q_sat0 < qᵛ): condensation warms particle → opposes freezing
    Q_latent = ℒⁱ * Dᵛ * ρ * (q_sat0 - qᵛ)

    # Only refreeze when net heat balance favors it. As in the Fortran wet-growth
    # and refreezing paths, 2π/Lf multiplies only the latent-diffusion term.
    dm_dt_refrz = clamp_positive(C_fv * (Q_sensible + 2 * FT(π) * Q_latent / ℒᶠᵘˢ))

    refrz_rate = nⁱ_eff * dm_dt_refrz

    # Limit to available liquid on ice
    τ_safety = p3.process_rates.sink_limiting_timescale
    max_refrz = qʷⁱ_eff / τ_safety
    refrz_rate = min(refrz_rate, max_refrz)

    return ifelse(below_freezing, refrz_rate, zero(FT))
end
