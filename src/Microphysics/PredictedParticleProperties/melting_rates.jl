#####
##### Melting
#####

"""
$(TYPEDSIGNATURES)

Compute ice melting rate using the heat balance equation from
Morrison & Milbrandt (2015a) Eq. 44.

The melting rate is determined by the heat flux to the particle:

```math
\\frac{dm}{dt} = -\\frac{2π \\, \\text{capm}}{ℒᶠᵘˢ} × [Kᵃ(T-T_0) + ρ ℒˡ Dᵛ(q^v - q^{v+}_0)] × f^{ve}
```

where capm = cap × D is the P3 Fortran capacitance convention (2× physical C).

where:
- C is the capacitance
- ℒᶠᵘˢ is the latent heat of fusion
- Kᵃ is thermal conductivity of air
- T_0 is the freezing temperature
- ℒˡ is latent heat of vaporization
- Dᵛ is diffusivity of water vapor
- qᵛ, qᵛ⁺(T₀) are the vapor and saturation specific humidities (total-air mass
  fractions) at T₀; qᵛ⁺(T₀) = ρᵛ⁺(T₀)/ρ so that ρ (qᵛ - qᵛ⁺(T₀)) = ρᵛ - ρᵛ⁺(T₀)
- fᵛᵉ is the ventilation factor

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `P`: Pressure [Pa]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺`: Saturation vapor mass fraction over liquid [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants
- `transport`: Pre-computed air transport properties `(; Dᵛ, Kᵃ, ν)`

# Returns
- Rate of ice → rain conversion [kg/kg/s]
"""
@inline ice_melting_rate(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ,
                         constants, transport, μⁱ) =
    ice_melting_rate_and_ventilation(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ,
                                     constants, transport, μⁱ)[1]

# Returns `(melt_rate, small, large)`. The small/large split of the ventilation
# integral sets the rain-vs-coating partition in `ice_melting_rates`, and comes
# free with the lookup the melt rate already needs — recomputing it there would
# repeat one `prepare_5d` and four `evaluate_at` at the same coordinate.
@inline function ice_melting_rate_and_ventilation(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ,
                                                  constants, transport, μⁱ)
    FT = typeof(qⁱ)
    parameters = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    T₀ = parameters.freezing_temperature

    # Only melt above freezing
    ΔT = T - T₀
    is_melting = ΔT > 0

    # Thermodynamic constants: ℒᶠᵘˢ and ℒˡ are T-dependent.
    ℒᶠᵘˢ = fusion_latent_heat(constants, T)
    ℒˡ = vaporization_latent_heat(constants, T)
    # T,P-dependent transport properties (pre-computed or computed on demand)
    Kᵃ = transport.Kᵃ       # Thermal conductivity of air [W/m/K]
    Dᵛ = transport.Dᵛ       # Diffusivity of water vapor [m²/s]
    ν  = transport.ν        # Kinematic viscosity [m²/s]

    q_sat0 = freezing_point_saturation_mass_fraction(constants, T₀, ρ)

    # Liquid fraction for Fl-blended ventilation.
    # Fl = qʷⁱ / (qⁱ + qʷⁱ): fraction of ice-particle mass that is liquid.
    qⁱ_total = max(qⁱ_eff + clamp_positive(qʷⁱ), FT(parameters.floors.mass_scale))
    Fl = clamp_positive(qʷⁱ) / qⁱ_total

    # Table lookup uses total mass per particle (Fortran qitot/nitot),
    # not dry-only mass, because tables are indexed by total mass.
    # With no particles the rate below is nⁱ_eff * dm/dt = 0 regardless, so the
    # fallback mean mass only has to sit inside the table's mass axis.
    m_mean = safe_divide(qⁱ_total, nⁱ_eff, FT(parameters.floors.mean_particle_mass_fallback))
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)

    # Use dry-ice PSD ventilation tables (small + large, Fortran f1pr24-f1pr27)
    # for melting. The total Ventilation/VentilationEnhanced tables use wet-ice PSD
    # and are not appropriate for melting (they are not flagged as melting integrals
    # during table generation, so they don't use the dry-ice PSD from the M5 fix).
    # All 4 tables share Table-1 axes so the interpolation indices are computed once.
    dep = p3.ice.deposition
    # m_mean = qⁱ/nⁱ is a per-particle mass [kg]; floor it only with a tiny log-guard,
    # NOT the bulk mass-mixing-ratio threshold `minimum_mass_mixing_ratio` (kg/kg).
    # The table clamps the coordinate to its mass axis (min ≈ 1.56e-15 kg), matching
    # Fortran's clamp of the lookup index to 1 (find_lookupTable_indices_1a).
    log_m = log10(max(m_mean, FT(parameters.floors.mass_scale)))
    sc_corr = ventilation_sc_correction(ν, Dᵛ, ρ_correction)
    prep = prepare_5d(dep.small_ice_ventilation_constant, log_m, Fᶠ, Fl, ρᶠ, μⁱ)
    small = evaluate_at(dep.small_ice_ventilation_constant, prep) +
            sc_corr * evaluate_at(dep.small_ice_ventilation_reynolds, prep)
    large = evaluate_at(dep.large_ice_ventilation_constant, prep) +
            sc_corr * evaluate_at(dep.large_ice_ventilation_reynolds, prep)
    C_fv = small + large

    # Heat flux terms (Eq. 44 from MM15a)
    # Sensible heat: Kᵃ × (T - T₀)
    Q_sensible = Kᵃ * ΔT

    # Latent heat: ℒˡ × Dᵛ × ρ × (qᵛ - q_sat0)
    # When subsaturated, this is negative and opposes melting
    Q_latent = ℒˡ * Dᵛ * ρ * (qᵛ - q_sat0)

    # Total heat flux
    Q_total = Q_sensible + Q_latent

    # Melting rate per particle (negative dm/dt → positive melt rate)
    # Uses 2π (not 4π) because ventilation integral stores capm = cap × D
    # (P3 Fortran convention), which is 2× the physical capacitance.
    dm_dt_melt = 2 * FT(π) * C_fv * Q_total / ℒᶠᵘˢ

    # Clamp to positive (only melting, not refreezing here)
    dm_dt_melt = clamp_positive(dm_dt_melt)

    # Total rate
    melt_rate = nⁱ_eff * dm_dt_melt

    # Limit melting rate: physical heat-transfer rate is the true limiter.
    # Guard against numerical overflow with a safety timescale (dt_safety).
    # The driver or time integrator must additionally limit melting to
    # available ice per dt.
    max_melt = qⁱ_eff / p3.process_rates.sink_limiting_timescale
    melt_rate = min(melt_rate, max_melt)

    return (ifelse(is_melting, melt_rate, zero(FT)), small, large)
end

"""
$(TYPEDSIGNATURES)

Compute partitioned ice melting rates using PSD-resolved partitioning (H9).

Above freezing, ice particles melt. The meltwater is partitioned using
tabulated small/large ice ventilation integrals (Fortran f1pr24-f1pr27):
- **Complete melting** (small particles, D ≤ D_crit): Meltwater sheds to rain
- **Partial melting** (large particles, D > D_crit): Meltwater stays as liquid coating (qʷⁱ)

Requires tabulated small/large ice ventilation integrals.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `T`: Temperature [K]
- `P`: Pressure [Pa]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺`: Saturation vapor mass fraction over liquid [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `constants`: Thermodynamic constants
- `transport`: Pre-computed air transport properties `(; Dᵛ, Kᵃ, ν)`

# Returns
- NamedTuple with `partial_melting` and `complete_melting` rates [kg/kg/s]
"""
@inline function ice_melting_rates(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ,
                                   constants, transport, μⁱ)
    FT = typeof(qⁱ)
    parameters = p3.process_rates

    # Total melting rate, plus the small/large ventilation split it already had to
    # look up (Fortran f1pr24-f1pr27). Passing qʷⁱ gives the Fl-blended ventilation.
    total_melt, small, large =
        ice_melting_rate_and_ventilation(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ,
                                         constants, transport, μⁱ)

    rain_fraction = psd_melting_rain_fraction(small, large)

    complete = total_melt * rain_fraction
    partial  = total_melt * (1 - rain_fraction)

    return (partial_melting = partial, complete_melting = complete)
end

# Fraction of melting that goes to rain (small particles, D ≤ D_crit), from the
# PSD-integrated small/large ice ventilation integrals computed by
# `ice_melting_rate_and_ventilation`.
# Fortran: qrmlt uses f1pr24/f1pr25, qiliqcol uses f1pr26/f1pr27.
@inline function psd_melting_rain_fraction(small, large)
    FT = typeof(small)
    total = small + large

    # With no tabulated ventilation on either side of the split there is nothing to
    # weight with, so the meltwater is divided evenly between rain and coating.
    degenerate_rain_fraction = FT(1) / 2
    return ifelse(total > eps(FT), clamp(small / total, FT(0), FT(1)), degenerate_rain_fraction)
end

"""
$(TYPEDSIGNATURES)

Compute ice number loss from melting.

Number of melted particles equals number of rain drops produced.

# Arguments
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `qⁱ_melt_rate`: Ice mass melting rate [kg/kg/s]

# Returns
- Rate of ice number loss [1/kg/s] (positive magnitude; sign applied in tendency assembly)
"""
@inline function ice_melting_number_rate(qⁱ, nⁱ, qⁱ_melt_rate)
    FT = typeof(qⁱ)

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # |∂nⁱ/∂t| = (nⁱ/qⁱ) × ∂qⁱ_melt/∂t (positive magnitude)
    # Sign convention (M7): returns positive; caller subtracts in tendency assembly.
    ratio = safe_divide(nⁱ_eff, qⁱ_eff, zero(FT))

    return ratio * qⁱ_melt_rate
end
