#####
##### Melting
#####

"""
    ice_melting_rate(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ, constants, transport)

Compute ice melting rate using the heat balance equation from
Morrison & Milbrandt (2015a) Eq. 44.

The melting rate is determined by the heat flux to the particle:

```math
\\frac{dm}{dt} = -\\frac{2π \\, \\text{capm}}{L_f} × [K_a(T-T_0) + ρ L_v D_v(q_v - q_{sat0})] × f_v
```

where capm = cap × D is the P3 Fortran capacitance convention (2× physical C).

where:
- C is the capacitance
- L_f is the latent heat of fusion
- K_a is thermal conductivity of air
- T_0 is the freezing temperature
- L_v is latent heat of vaporization
- D_v is diffusivity of water vapor
- q_v, q_sat0 are vapor mixing ratio and saturation mixing ratio at T₀
- f_v is the ventilation factor

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
- `constants`: Thermodynamic constants (or `nothing` for Fortran-matched hardcoded values)
- `transport`: Pre-computed air transport properties `(; D_v, K_a, nu)`

# Returns
- Rate of ice → rain conversion [kg/kg/s]
"""
@inline function ice_melting_rate(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ,
                                   constants, transport)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    T₀ = prp.freezing_temperature

    # Only melt above freezing
    ΔT = T - T₀
    is_melting = ΔT > 0

    # Thermodynamic constants: L_f and L_v are T-dependent when constants
    # are provided (H1), and Rᵛ follows the same runtime thermodynamic source.
    L_f = fusion_latent_heat(constants, T)
    L_v = vaporization_latent_heat(constants, T)
    thermodynamic_constants = isnothing(constants) ? ThermodynamicConstants(FT) : constants
    Rᵛ = FT(vapor_gas_constant(thermodynamic_constants))
    # T,P-dependent transport properties (pre-computed or computed on demand)
    K_a = transport.K_a       # Thermal conductivity of air [W/m/K]
    D_v = transport.D_v       # Diffusivity of water vapor [m²/s]
    nu  = transport.nu        # Kinematic viscosity [m²/s]

    # M10: use mixing ratio convention (Fortran: rho*Lv*Dv*(Qv-qsat0))
    Rᵈ = FT(dry_air_gas_constant(thermodynamic_constants))
    ε = Rᵈ / Rᵛ
    e_s0 = saturation_vapor_pressure_at_freezing(constants, T₀)
    q_sat0 = ε * e_s0 / max(P - e_s0, FT(1))

    # Mean particle properties
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # H10: Liquid fraction for Fl-blended ventilation.
    # Fl = qʷⁱ / (qⁱ + qʷⁱ): fraction of ice-particle mass that is liquid.
    qⁱ_total = max(qⁱ_eff + clamp_positive(qʷⁱ), FT(1e-20))
    Fl = clamp_positive(qʷⁱ) / qⁱ_total
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)

    # H10: Ventilation integral C(D) × f_v(D) with Fl-blended ice/rain coefficients.
    # Dispatches to PSD-integrated table or mean-mass path.
    C_fv = melting_ventilation(p3.ice.deposition.ventilation,
                                p3.ice.deposition.ventilation_enhanced,
                                m_mean, Fl, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3)

    # Heat flux terms (Eq. 44 from MM15a)
    # Sensible heat: K_a × (T - T₀)
    Q_sensible = K_a * ΔT

    # Latent heat: L_v × D_v × ρ × (qᵛ - q_sat0)
    # When subsaturated, this is negative and opposes melting
    Q_latent = L_v * D_v * ρ * (qᵛ - q_sat0)

    # Total heat flux
    Q_total = Q_sensible + Q_latent

    # Melting rate per particle (negative dm/dt → positive melt rate)
    # Uses 2π (not 4π) because ventilation integral stores capm = cap × D
    # (P3 Fortran convention), which is 2× the physical capacitance.
    dm_dt_melt = FT(2π) * C_fv * Q_total / L_f

    # Clamp to positive (only melting, not refreezing here)
    dm_dt_melt = clamp_positive(dm_dt_melt)

    # Total rate
    melt_rate = nⁱ_eff * dm_dt_melt

    # Limit melting rate: physical heat-transfer rate is the true limiter.
    # Guard against numerical overflow with a 1-second safety timescale,
    # meaning at most all ice can melt per second. The driver or time
    # integrator must additionally limit melting to available ice per dt.
    τ_safety = FT(1)  # [s] — CFL-like constraint, not a physical timescale
    max_melt = qⁱ_eff / τ_safety
    melt_rate = min(melt_rate, max_melt)

    return ifelse(is_melting, melt_rate, zero(FT))
end

"""
    ice_melting_rates(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ, constants, transport)

Compute partitioned ice melting rates using PSD-resolved partitioning (H9).

Above freezing, ice particles melt. The meltwater is partitioned using
tabulated small/large ice ventilation integrals (Fortran f1pr24-f1pr27):
- **Complete melting** (small particles, D ≤ D_crit): Meltwater sheds to rain
- **Partial melting** (large particles, D > D_crit): Meltwater stays as liquid coating (qʷⁱ)

When tables are not available, falls back to a bulk liquid-fraction heuristic.

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
- `constants`: Thermodynamic constants (or `nothing` for Fortran-matched hardcoded values)
- `transport`: Pre-computed air transport properties `(; D_v, K_a, nu)`

# Returns
- NamedTuple with `partial_melting` and `complete_melting` rates [kg/kg/s]
"""
@inline function ice_melting_rates(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ,
                                    constants, transport)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    # Get total melting rate (H10: pass qʷⁱ for Fl-blended ventilation)
    total_melt = ice_melting_rate(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺, Fᶠ, ρᶠ, ρ, constants, transport)

    # H9: PSD-resolved melting partitioning using tabulated small/large ice
    # ventilation integrals (Fortran f1pr24-f1pr27).
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))
    qⁱ_total = max(qⁱ_eff + clamp_positive(qʷⁱ), FT(1e-20))
    Fl = clamp_positive(qʷⁱ) / qⁱ_total
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)
    nu = transport.nu
    D_v = transport.D_v

    rain_fraction = psd_melting_rain_fraction(
        p3.ice.deposition.small_ice_ventilation_constant,
        p3.ice.deposition.small_ice_ventilation_reynolds,
        p3.ice.deposition.large_ice_ventilation_constant,
        p3.ice.deposition.large_ice_ventilation_reynolds,
        m_mean, Fl, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3)

    complete = total_melt * rain_fraction
    partial  = total_melt * (1 - rain_fraction)

    return (partial_melting = partial, complete_melting = complete)
end

# H9: Tabulated path — use PSD-integrated small/large ice ventilation integrals
# to compute the fraction of melting that goes to rain (small particles, D ≤ D_crit).
# Fortran: qrmlt uses f1pr24/f1pr25, qiliqcol uses f1pr26/f1pr27.
@inline function psd_melting_rain_fraction(sc::TabulatedFunction4D, sr::TabulatedFunction4D,
                                            lc::TabulatedFunction4D, lr::TabulatedFunction4D,
                                            m_mean, Fl, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, p3.minimum_mass_mixing_ratio))
    sc_corr = ventilation_sc_correction(nu, D_v, ρ_correction)

    small = sc(log_m, Fᶠ, Fl, ρᶠ) + sc_corr * sr(log_m, Fᶠ, Fl, ρᶠ)
    large = lc(log_m, Fᶠ, Fl, ρᶠ) + sc_corr * lr(log_m, Fᶠ, Fl, ρᶠ)
    total = small + large

    return ifelse(total > eps(FT), clamp(small / total, FT(0), FT(1)), FT(0.5))
end

# H9: Analytical fallback — when tables are not available, use the bulk
# liquid-fraction heuristic with a minimum rain floor.
@inline function psd_melting_rain_fraction(::AbstractDepositionIntegral, ::AbstractDepositionIntegral,
                                            ::AbstractDepositionIntegral, ::AbstractDepositionIntegral,
                                            m_mean, Fl, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3)
    FT = typeof(m_mean)
    max_liquid_fraction = prp.maximum_liquid_fraction
    fraction_to_coating = clamp_positive(max_liquid_fraction - Fl) / max_liquid_fraction
    fraction_to_coating = clamp(fraction_to_coating, FT(0), FT(1))
    min_to_rain = prp.minimum_complete_melting_fraction
    fraction_to_coating = clamp(fraction_to_coating, FT(0), FT(1) - min_to_rain)
    return 1 - fraction_to_coating
end

"""
    ice_melting_number_rate(qⁱ, nⁱ, qⁱ_melt_rate)

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
