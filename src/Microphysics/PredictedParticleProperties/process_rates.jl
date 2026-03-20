#####
##### P3 Process Rates
#####
##### Microphysical process rate calculations for the P3 scheme.
##### All rate functions take the P3 scheme as first positional argument
##### to access parameters. No keyword arguments (GPU compatibility).
#####
##### Notation follows docs/src/appendix/notation.md
#####

using Oceananigans: Oceananigans

using Breeze.Thermodynamics: temperature,
                             saturation_specific_humidity,
                             PlanarLiquidSurface,
                             PlanarIceSurface,
                             liquid_latent_heat,
                             ice_latent_heat,
                             mixture_heat_capacity,
                             vapor_gas_constant,
                             MoistureMassFractions

#####
##### Utility functions
#####

"""
    clamp_positive(x)

Return max(0, x) for numerical stability.
"""
@inline clamp_positive(x) = max(0, x)

"""
    sink_limiting_factor(total_sink, available_mass, dt_safety)

Compute proportional rescaling factor for sink rates so that
`total_sink × dt_safety` does not exceed `available_mass`.

Returns 1 when sinks are within budget, or `available_mass / (total_sink × dt_safety)`
when they exceed it. All arguments must be positive or zero.
GPU-compatible: uses `ifelse` instead of branching.
"""
@inline function sink_limiting_factor(total_sink, available_mass, dt_safety)
    projected = total_sink * dt_safety
    return ifelse(projected > available_mass,
                  available_mass / max(projected, eps(typeof(available_mass))),
                  one(typeof(available_mass)))
end

"""
    safe_divide(a, b, default)

Safe division returning `default` when b ≈ 0.
All arguments must be positional (GPU kernel compatibility).
"""
@inline function safe_divide(a, b, default)
    return ifelse(iszero(b), default, a / b)
end

# Convenience overload for common case
@inline safe_divide(a, b) = safe_divide(a, b, zero(a))

#####
##### Thermodynamic latent heat helpers (H1)
#####
##### When thermodynamic constants are available, use T-dependent latent heats
##### for energy-budget consistency with the condensation path. When `nothing`
##### is passed (backward-compatible path), fall back to the Fortran P3 v5.5.0
##### hardcoded constants.
#####

@inline _sublimation_latent_heat(::Nothing, T) = typeof(T)(2.835e6)
@inline _sublimation_latent_heat(constants, T) = ice_latent_heat(T, constants)

@inline _vaporization_latent_heat(::Nothing, T) = typeof(T)(2.5e6)
@inline _vaporization_latent_heat(constants, T) = liquid_latent_heat(T, constants)

@inline _fusion_latent_heat(constants, T) = _sublimation_latent_heat(constants, T) - _vaporization_latent_heat(constants, T)

#####
##### Ventilation Sc correction (H4)
#####
##### The ventilation-enhanced table stores 0.44 × ∫ C(D)√(V×D) N'(D) dD
##### with dimensions [m² s^(-1/2)]. At runtime, multiplying by Sc^(1/3)/√ν
##### restores the correct dimensions [m]. This helper centralizes the
##### correction so that all call sites (deposition, Z-tendency) stay in sync.
#####

"""
    ventilation_sc_correction(nu, D_v)

Schmidt number correction factor for ventilation-enhanced table values.

The P3 lookup table stores the ventilation-enhanced integral without the
`Sc^{1/3}/√ν` factor (matching the Fortran convention). This function
computes the correction that must be applied at runtime:

```math
f_{Sc} = \\frac{Sc^{1/3}}{\\sqrt{\\nu}} = \\frac{(\\nu/D_v)^{1/3}}{\\sqrt{\\nu}}
```

See `quadrature.jl` for the table storage convention.
"""
@inline function ventilation_sc_correction(nu, D_v)
    FT = typeof(nu)
    Sc = nu / max(D_v, FT(1e-30))
    return cbrt(Sc) / sqrt(nu)
end

#####
##### Table-dispatched helpers for PSD-integrated process rates
#####
##### These functions dispatch on whether a table field is a TabulatedFunction3D
##### (use PSD-integrated lookup) or Any (use mean-mass analytical fallback).
##### This pattern matches the existing _tabulated_mass_weighted_fall_speed dispatch.
#####

"""
    _deposition_ventilation(vent, vent_e, m_mean, Fᶠ, ρᶠ, prp)

Compute per-particle ventilation integral C(D) × f_v(D) for deposition.
Dispatches on table type for PSD-integrated or mean-mass path.
"""
@inline function _deposition_ventilation(vent::TabulatedFunction4D,
                                          vent_e::TabulatedFunction4D,
                                          m_mean, Fᶠ, ρᶠ, prp, nu, D_v)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, FT(1e-20)))
    Fˡ = zero(FT)
    # vent stores the constant ventilation term (0.65 × ∫ C(D) N'(D) dD)
    # vent_e stores the enhanced term (0.44 × ∫ C(D)√(V×D) N'(D) dD)  [m² s^(-1/2)]
    # Runtime correction via ventilation_sc_correction: Sc^(1/3)/√ν [s^(1/2) m^(-1)]
    # Dimensional check: table [m² s^(-1/2)] × correction [s^(1/2)/m] = [m]
    return vent(log_m, Fᶠ, Fˡ, ρᶠ) + ventilation_sc_correction(nu, D_v) * vent_e(log_m, Fᶠ, Fˡ, ρᶠ)
end

@inline function _deposition_ventilation(::AbstractDepositionIntegral, ::AbstractDepositionIntegral,
                                          m_mean, Fᶠ, ρᶠ, prp, nu, D_v)
    FT = typeof(m_mean)
    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    ρ_eff = (1 - Fᶠ) * ρ_eff_unrimed + Fᶠ * ρᶠ
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_eff))
    D_threshold = prp.ice_diameter_threshold
    # P3 Fortran convention: capm = cap × D where cap=1 for sphere, 0.48 for aggregate
    C = ifelse(D_mean < D_threshold, D_mean, FT(0.48) * D_mean)
    # H7: Blend fall speed coefficients with rime fraction (matching _collection_kernel_per_particle)
    a_V = (1 - Fᶠ) * prp.ice_fall_speed_coefficient_unrimed + Fᶠ * prp.ice_fall_speed_coefficient_rimed
    b_V = (1 - Fᶠ) * prp.ice_fall_speed_exponent_unrimed + Fᶠ * prp.ice_fall_speed_exponent_rimed
    V = a_V * D_mean^b_V
    # Schmidt number correction (Hall & Pruppacher 1976): Sc = nu / D_v
    Sc = nu / max(D_v, FT(1e-30))
    Re_term = sqrt(V * D_mean / nu)
    f_v = FT(0.65) + FT(0.44) * cbrt(Sc) * Re_term
    return C * f_v
end

"""
    _collection_kernel_per_particle(coll, m_mean, Fᶠ, ρᶠ, prp)

Compute per-particle collection kernel ⟨A × V⟩ for riming.
Table path: returns PSD-integrated ∫ V(D) A(D) N'(D) dD (per particle).
Analytical path: returns A_mean × V_mean × psd_correction.
"""
@inline function _collection_kernel_per_particle(coll::TabulatedFunction4D,
                                                  m_mean, Fᶠ, ρᶠ, prp)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, FT(1e-20)))
    return coll(log_m, Fᶠ, zero(FT), ρᶠ)
end

@inline function _collection_kernel_per_particle(::AbstractCollectionIntegral, m_mean, Fᶠ, ρᶠ, prp)
    FT = typeof(m_mean)
    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    ρ_eff = (1 - Fᶠ) * ρ_eff_unrimed + Fᶠ * ρᶠ
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_eff))
    D_mean = clamp(D_mean, prp.ice_diameter_min, prp.ice_diameter_max)
    # M9: Fortran P3 only includes ice particles with D >= 100 μm in the
    # collection integral (nrwat threshold). Zero the kernel for sub-threshold
    # particles to prevent tiny freshly-nucleated ice from riming.
    below_threshold = D_mean < FT(100e-6)
    a_V = (1 - Fᶠ) * prp.ice_fall_speed_coefficient_unrimed + Fᶠ * prp.ice_fall_speed_coefficient_rimed
    b_V = (1 - Fᶠ) * prp.ice_fall_speed_exponent_unrimed + Fᶠ * prp.ice_fall_speed_exponent_rimed
    V_mean = a_V * D_mean^b_V
    A_agg = prp.ice_projected_area_coefficient * D_mean^prp.ice_projected_area_exponent
    A_sphere = FT(π) / 4 * D_mean^2
    A_mean = (1 - Fᶠ) * A_agg + Fᶠ * A_sphere
    psd_correction = prp.riming_psd_correction
    return ifelse(below_threshold, zero(FT), A_mean * V_mean * psd_correction)
end

"""
    _aggregation_kernel(coll, m_mean, Fᶠ, ρᶠ, prp)

Compute aggregation kernel for self-collection.
Table path: uses PSD-integrated kernel from table.
Analytical path: A_mean × ΔV at mean diameter.
"""
@inline function _aggregation_kernel(coll::TabulatedFunction4D,
                                      m_mean, Fᶠ, ρᶠ, prp)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, FT(1e-20)))
    # Table stores the half-integral (Fortran convention):
    # (1/2) ∫∫ (√A₁+√A₂)² |V₁-V₂| N₁ N₂ dD₁ dD₂
    # No E_agg — collection efficiency is applied by the caller.
    return coll(log_m, Fᶠ, zero(FT), ρᶠ)
end

@inline function _aggregation_kernel(::AbstractCollectionIntegral, m_mean, Fᶠ, ρᶠ, prp)
    FT = typeof(m_mean)
    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    ρ_eff = max(FT(50), (1 - Fᶠ) * ρ_eff_unrimed + Fᶠ * ρᶠ)
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_eff))
    a_V = (1 - Fᶠ) * prp.ice_fall_speed_coefficient_unrimed + Fᶠ * prp.ice_fall_speed_coefficient_rimed
    b_V = (1 - Fᶠ) * prp.ice_fall_speed_exponent_unrimed + Fᶠ * prp.ice_fall_speed_exponent_rimed
    V_mean = a_V * D_mean^b_V
    A_agg = prp.ice_projected_area_coefficient * D_mean^prp.ice_projected_area_exponent
    A_sphere = FT(π) / 4 * D_mean^2
    A_mean = (1 - Fᶠ) * A_agg + Fᶠ * A_sphere
    ΔV = FT(0.5) * V_mean
    # Factor of 0.5 for self-collection (half-integral convention, matching table)
    return FT(0.5) * A_mean * ΔV
end

#####
##### Cloud condensation/evaporation
#####

"""
    cloud_condensation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, constants)

Compute cloud liquid condensation/evaporation rate using relaxation-to-saturation.

When the air is supersaturated (qᵛ > qᵛ⁺ˡ), excess vapor condenses onto cloud
droplets. When subsaturated, cloud liquid evaporates back to vapor. The rate
follows a relaxation timescale with a thermodynamic (psychrometric) correction
factor that accounts for latent heating during phase change.

# Arguments
- `p3`: P3 microphysics scheme (provides condensation timescale)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ˡ`: Saturation vapor mass fraction over liquid [kg/kg]
- `T`: Temperature [K]
- `q`: Moisture mass fractions (vapor, liquid, ice)
- `constants`: Thermodynamic constants

# Returns
- Rate of vapor → cloud liquid conversion [kg/kg/s]
  (positive = condensation, negative = evaporation)
"""
@inline function cloud_condensation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, constants)
    FT = typeof(qᶜˡ)
    τᶜˡ = p3.cloud.condensation_timescale

    # Thermodynamic adjustment factor (psychrometric correction)
    ℒˡ = liquid_latent_heat(T, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    Γˡ = 1 + (ℒˡ / cᵖᵐ) * dqᵛ⁺_dT

    # Relaxation toward saturation
    Sᶜᵒⁿᵈ = (qᵛ - qᵛ⁺ˡ) / (Γˡ * τᶜˡ)

    # Limit evaporation to available cloud liquid (include Γˡ for consistency)
    Sᶜᵒⁿᵈ_min = -max(0, qᶜˡ) / (Γˡ * τᶜˡ)
    return max(Sᶜᵒⁿᵈ, Sᶜᵒⁿᵈ_min)
end

#####
##### Ice deposition and sublimation
#####

"""
    ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, constants, transport)

Compute ventilation-enhanced ice deposition/sublimation rate.

Following Morrison & Milbrandt (2015a) Eq. 30, the deposition rate is:

```math
\\frac{dm}{dt} = \\frac{4πC f_v (S_i - 1)}{\\frac{L_s}{K_a T}(\\frac{L_s}{R_v T} - 1) + \\frac{R_v T}{e_{si} D_v}}
```

where f_v is the ventilation factor and C is the capacitance.

The bulk rate integrates over the size distribution:

```math
\\frac{dq^i}{dt} = ∫ \\frac{dm}{dt}(D) N'(D) dD
```

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ⁱ`: Saturation vapor mass fraction over ice [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `T`: Temperature [K]
- `P`: Pressure [Pa]
- `constants`: Thermodynamic constants (or `nothing` for Fortran-matched hardcoded values)
- `transport`: Pre-computed air transport properties `(; D_v, K_a, nu)`

# Returns
- Rate of vapor → ice conversion [kg/kg/s] (positive = deposition)
"""
@inline function ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P,
                                                  constants, transport)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # Thermodynamic constants: R_v and R_d hardcoded to match Fortran P3 v5.5.0 (L7).
    # Latent heat L_s is T-dependent when constants are provided (H1).
    R_v = FT(461.5)
    R_d = FT(287.0)
    L_s = _sublimation_latent_heat(constants, T)
    # T,P-dependent transport properties (pre-computed or computed on demand)
    K_a = transport.K_a       # Thermal conductivity of air [W/m/K]
    D_v = transport.D_v       # Diffusivity of water vapor [m²/s]
    nu  = transport.nu        # Kinematic viscosity [m²/s]

    # Saturation vapor pressure over ice
    # Derived from qᵛ⁺ⁱ: qᵛ⁺ⁱ = ε × e_si / (P - (1-ε) × e_si)
    # Rearranging: e_si = P × qᵛ⁺ⁱ / (ε + qᵛ⁺ⁱ × (1 - ε))
    ε = R_d / R_v
    qᵛ⁺ⁱ_safe = max(qᵛ⁺ⁱ, FT(1e-30))
    e_si = P * qᵛ⁺ⁱ_safe / (ε + qᵛ⁺ⁱ_safe * (1 - ε))

    # Supersaturation ratio with respect to ice
    S_i = qᵛ / max(qᵛ⁺ⁱ, FT(1e-10))

    # Mean particle mass
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # Ventilation integral C(D) × f_v(D): dispatches to PSD-integrated
    # table or mean-mass analytical path depending on p3.ice.deposition type.
    C_fv = _deposition_ventilation(p3.ice.deposition.ventilation,
                                    p3.ice.deposition.ventilation_enhanced,
                                    m_mean, Fᶠ, ρᶠ, prp, nu, D_v)

    # Denominator: thermodynamic resistance terms (Mason 1971)
    # A = L_s/(K_a × T) × (L_s/(R_v × T) - 1)
    # B = R_v × T / (e_si × D_v)
    A = L_s / (K_a * T) * (L_s / (R_v * T) - 1)
    B = R_v * T / (e_si * D_v)
    thermodynamic_factor = A + B

    # Deposition rate per particle (Eq. 30 from MM15a)
    # Uses 2π (not 4π) because the ventilation integral stores capm = cap × D
    # (P3 Fortran convention), which is 2× the physical capacitance C = D/2.
    # The product 2π × capm = 2π × 2C = 4πC is physically correct.
    dm_dt = FT(2π) * C_fv * (S_i - 1) / thermodynamic_factor

    # Scale by number concentration
    dep_rate = nⁱ_eff * dm_dt

    # Limit sublimation to available ice
    τ_dep = prp.ice_deposition_timescale
    is_sublimation = S_i < 1
    max_sublim = -qⁱ_eff / τ_dep

    return ifelse(is_sublimation, max(dep_rate, max_sublim), dep_rate)
end

# Backward-compatible: explicit transport, hardcoded latent heats
@inline function ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P,
                                                  transport::NamedTuple)
    return ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, nothing, transport)
end

# Backward-compatible: default transport, hardcoded latent heats
@inline function ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P)
    return ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, nothing,
                                            air_transport_properties(T, P))
end

#####
##### Combined P3 tendency calculation
#####

"""
    P3ProcessRates

Container for computed P3 process rates.
Includes Phase 1 (rain, deposition, melting), Phase 2 (aggregation, riming, shedding, nucleation).

Following Milbrandt et al. (2025), melting is partitioned:
- `partial_melting`: Meltwater stays on ice as liquid coating (large particles)
- `complete_melting`: Meltwater sheds to rain (small particles)
"""
struct P3ProcessRates{FT}
    # Phase 1: Cloud condensation/evaporation
    condensation :: FT             # Vapor → cloud liquid [kg/kg/s] (positive = condensation, negative = evaporation)

    # Phase 1: Rain tendencies
    autoconversion :: FT           # Cloud → rain mass [kg/kg/s]
    accretion :: FT                # Cloud → rain mass (via rain sweep-out) [kg/kg/s]
    rain_evaporation :: FT         # Rain → vapor mass [kg/kg/s]
    rain_self_collection :: FT     # Rain number reduction [1/kg/s]
    rain_breakup :: FT             # Rain number increase from breakup [1/kg/s]

    # Phase 1: Ice tendencies
    deposition :: FT               # Vapor → ice mass [kg/kg/s]
    partial_melting :: FT          # Ice → liquid coating (stays on ice) [kg/kg/s]
    complete_melting :: FT         # Ice → rain mass (sheds) [kg/kg/s]
    melting_number :: FT           # Ice number reduction from melting [1/kg/s]

    # Phase 2: Ice aggregation
    aggregation :: FT              # Ice number reduction from self-collection [1/kg/s]

    # Phase 2: Riming
    cloud_riming :: FT             # Cloud → ice via riming [kg/kg/s]
    cloud_riming_number :: FT      # Cloud number reduction [1/kg/s]
    rain_riming :: FT              # Rain → ice via riming [kg/kg/s]
    rain_riming_number :: FT       # Rain number reduction [1/kg/s]
    rime_density_new :: FT         # Density of new rime [kg/m³]

    # Phase 2: Shedding and refreezing
    shedding :: FT                 # Liquid on ice → rain [kg/kg/s]
    shedding_number :: FT          # Rain number from shedding [1/kg/s]
    refreezing :: FT               # Liquid on ice → rime [kg/kg/s]

    # Ice nucleation (deposition + immersion freezing)
    nucleation_mass :: FT          # New ice mass from deposition nucleation [kg/kg/s]
    nucleation_number :: FT        # New ice number from deposition nucleation [1/kg/s]
    cloud_freezing_mass :: FT      # Cloud → ice mass from immersion freezing [kg/kg/s]
    cloud_freezing_number :: FT    # Cloud number to ice number [1/kg/s]
    rain_freezing_mass :: FT       # Rain → ice mass from immersion freezing [kg/kg/s]
    rain_freezing_number :: FT     # Rain number to ice number [1/kg/s]

    # Rime splintering (Hallett-Mossop)
    splintering_mass :: FT         # New ice mass from splintering [kg/kg/s]
    splintering_number :: FT       # New ice number from splintering [1/kg/s]

    # Homogeneous freezing (T < -40°C, instantaneous)
    cloud_homogeneous_mass :: FT   # Cloud → ice from homogeneous freezing [kg/kg/s]
    cloud_homogeneous_number :: FT # Cloud number → ice [1/kg/s]
    rain_homogeneous_mass :: FT    # Rain → ice from homogeneous freezing [kg/kg/s]
    rain_homogeneous_number :: FT  # Rain number → ice [1/kg/s]

    # Above-freezing cloud collection (T > T₀, Fortran qcshd pathway)
    cloud_warm_collection :: FT        # Cloud → rain via warm ice collection [kg/kg/s]
    cloud_warm_collection_number :: FT # Rain number from shed 1mm drops [1/kg/s]
end

"""
    compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)

Compute all P3 process rates (Phase 1 and Phase 2) from a microphysical state.

This is the gridless version that accepts a `P3MicrophysicalState` directly,
suitable for use in GPU kernels where grid indexing is handled externally.

# Arguments
- `p3`: P3 microphysics scheme
- `ρ`: Air density [kg/m³]
- `ℳ`: P3MicrophysicalState containing all mixing ratios
- `𝒰`: Thermodynamic state
- `constants`: Thermodynamic constants

# Returns
- `P3ProcessRates` containing all computed rates
"""
@inline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
    FT = typeof(ρ)
    prp = p3.process_rates
    T₀ = prp.freezing_temperature

    # Extract from microphysical state (already specific, not density-weighted)
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nʳ = ℳ.nʳ
    qⁱ = ℳ.qⁱ
    nⁱ = ℳ.nⁱ
    qᶠ = ℳ.qᶠ
    bᶠ = ℳ.bᶠ
    qʷⁱ = ℳ.qʷⁱ

    # Rime properties
    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))
    ρᶠ = safe_divide(qᶠ, bᶠ, FT(400))

    # Thermodynamic state
    T = temperature(𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor

    # Saturation vapor mixing ratios using Breeze thermodynamics
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())

    # Moisture mass fractions for thermodynamic calculations
    q = 𝒰.moisture_mass_fractions

    # Cloud droplet number concentration
    Nᶜ = p3.cloud.number_concentration

    # =========================================================================
    # Phase 1: Cloud condensation/evaporation
    # =========================================================================
    cond = cloud_condensation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, constants)

    # =========================================================================
    # Phase 1: Rain processes
    # =========================================================================
    P = 𝒰.reference_pressure

    # Compute T,P-dependent transport properties once (Fortran P3 v5.5.0 formulas)
    # — shared by deposition, melting, and rain evaporation (eliminates triple computation)
    transport = air_transport_properties(T, P)

    autoconv = rain_autoconversion_rate(p3, qᶜˡ, Nᶜ)
    accr = rain_accretion_rate(p3, qᶜˡ, qʳ)
    rain_evap = rain_evaporation_rate(p3, qʳ, nʳ, qᵛ, qᵛ⁺ˡ, T, ρ, P, transport)
    rain_self = rain_self_collection_rate(p3, qʳ, nʳ, ρ)
    rain_br = rain_breakup_rate(p3, qʳ, nʳ, rain_self)

    # =========================================================================
    # Phase 1: Ice deposition/sublimation and melting
    # =========================================================================
    dep = ventilation_enhanced_deposition(p3, qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, constants, transport)
    dep = ifelse(qⁱ > FT(1e-20), dep, zero(FT))

    # Partitioned melting: partial stays on ice, complete goes to rain
    melt_rates = ice_melting_rates(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺ˡ, Fᶠ, ρᶠ, ρ, constants, transport)
    partial_melt = melt_rates.partial_melting
    complete_melt = melt_rates.complete_melting
    # Only complete melting removes ice particles; partial melting keeps particles as ice
    melt_n = ice_melting_number_rate(qⁱ, nⁱ, complete_melt)

    # =========================================================================
    # Phase 2: Ice aggregation
    # =========================================================================
    agg = ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)

    # =========================================================================
    # Phase 2: Riming
    # =========================================================================
    cloud_rim = cloud_riming_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    cloud_rim_n = cloud_riming_number_rate(qᶜˡ, Nᶜ, cloud_rim)

    rain_rim = rain_riming_rate(p3, qʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)
    rain_rim_n = rain_riming_number_rate(qʳ, nʳ, rain_rim)

    # Rime density for new rime (use actual ice fall speed, not placeholder)
    vᵢ = ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)
    ρᶠ_new = rime_density(p3, T, vᵢ)

    # =========================================================================
    # Phase 2: Shedding and refreezing
    # =========================================================================
    shed = shedding_rate(p3, qʷⁱ, qⁱ, T)
    shed_n = shedding_number_rate(p3, shed)
    refrz = refreezing_rate(p3, qʷⁱ, T)

    # =========================================================================
    # Ice nucleation (deposition nucleation and immersion freezing)
    # =========================================================================
    nuc_q, nuc_n = deposition_nucleation_rate(p3, T, qᵛ, qᵛ⁺ⁱ, nⁱ, ρ)
    cloud_frz_q, cloud_frz_n = immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)
    rain_frz_q, rain_frz_n = immersion_freezing_rain_rate(p3, qʳ, nʳ, T)

    # =========================================================================
    # Rime splintering (Hallett-Mossop secondary ice production)
    # =========================================================================
    spl_q, spl_n = rime_splintering_rate(p3, cloud_rim, rain_rim, T)

    # =========================================================================
    # Homogeneous freezing (T < -40°C, instantaneous conversion)
    # =========================================================================
    cloud_hom_q, cloud_hom_n = homogeneous_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)
    rain_hom_q, rain_hom_n = homogeneous_freezing_rain_rate(p3, qʳ, nʳ, T)

    # =========================================================================
    # Above-freezing cloud collection (Fortran qcshd/ncshdc pathway)
    # =========================================================================
    cloud_warm_q, cloud_warm_n = cloud_warm_collection_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ)

    # =========================================================================
    # Sink limiting: rescale sink rates so total sinks × dt_safety ≤ available
    # mass for each species. Prevents negative mixing ratios with explicit
    # time integration (Fortran P3 convention).
    # =========================================================================
    dt_safety = prp.sink_limiting_timescale

    # --- Cloud liquid sinks ---
    # Cloud evaporation (negative condensation) is already self-limited in
    # cloud_condensation_rate, so only count the positive-definite sinks.
    cloud_sink_total = autoconv + accr + cloud_rim + cloud_frz_q +
                       cloud_hom_q + cloud_warm_q
    f_cloud = sink_limiting_factor(cloud_sink_total, max(0, qᶜˡ), dt_safety)
    autoconv      = autoconv * f_cloud
    accr          = accr * f_cloud
    cloud_rim     = cloud_rim * f_cloud
    cloud_rim_n   = cloud_rim_n * f_cloud
    cloud_frz_q   = cloud_frz_q * f_cloud
    cloud_frz_n   = cloud_frz_n * f_cloud
    cloud_hom_q   = cloud_hom_q * f_cloud
    cloud_hom_n   = cloud_hom_n * f_cloud
    cloud_warm_q  = cloud_warm_q * f_cloud
    cloud_warm_n  = cloud_warm_n * f_cloud

    # --- Rain sinks ---
    # Rain evaporation is already self-limited in rain_evaporation_rate.
    rain_sink_total = rain_rim + rain_frz_q + rain_hom_q
    f_rain = sink_limiting_factor(rain_sink_total, max(0, qʳ), dt_safety)
    rain_rim      = rain_rim * f_rain
    rain_rim_n    = rain_rim_n * f_rain
    rain_frz_q    = rain_frz_q * f_rain
    rain_frz_n    = rain_frz_n * f_rain
    rain_hom_q    = rain_hom_q * f_rain
    rain_hom_n    = rain_hom_n * f_rain

    # --- Ice sinks ---
    # Sublimation (negative deposition) is already self-limited in
    # ventilation_enhanced_deposition. Only count melting as sinks here.
    ice_sink_total = partial_melt + complete_melt
    f_ice = sink_limiting_factor(ice_sink_total, max(0, qⁱ), dt_safety)
    partial_melt  = partial_melt * f_ice
    complete_melt = complete_melt * f_ice
    melt_n        = melt_n * f_ice

    # --- Vapor sinks ---
    # Only count positive condensation and positive deposition as vapor sinks.
    # Nucleation mass is always a vapor sink.
    vapor_sink_total = max(0, cond) + max(0, dep) + nuc_q
    f_vapor = sink_limiting_factor(vapor_sink_total, max(0, qᵛ), dt_safety)
    cond  = ifelse(cond > 0, cond * f_vapor, cond)
    dep   = ifelse(dep > 0, dep * f_vapor, dep)
    nuc_q = nuc_q * f_vapor
    nuc_n = nuc_n * f_vapor

    # --- Liquid on ice (qʷⁱ) sinks (M11) ---
    # Shedding and refreezing are both sinks of qʷⁱ. Without this limiting,
    # explicit time integration can drive qʷⁱ negative.
    qwi_sink_total = shed + refrz
    f_qwi = sink_limiting_factor(qwi_sink_total, max(0, qʷⁱ), dt_safety)
    shed   = shed * f_qwi
    shed_n = shed_n * f_qwi
    refrz  = refrz * f_qwi

    # Recompute splintering from sink-limited riming rates
    spl_q, spl_n = rime_splintering_rate(p3, cloud_rim, rain_rim, T)

    return P3ProcessRates(
        # Phase 1: Condensation
        cond,
        # Phase 1: Rain
        autoconv, accr, rain_evap, rain_self, rain_br,
        # Phase 1: Ice
        dep, partial_melt, complete_melt, melt_n,
        # Phase 2: Aggregation
        agg,
        # Phase 2: Riming
        cloud_rim, cloud_rim_n, rain_rim, rain_rim_n, ρᶠ_new,
        # Phase 2: Shedding and refreezing
        shed, shed_n, refrz,
        # Ice nucleation
        nuc_q, nuc_n, cloud_frz_q, cloud_frz_n, rain_frz_q, rain_frz_n,
        # Rime splintering
        spl_q, spl_n,
        # Homogeneous freezing
        cloud_hom_q, cloud_hom_n, rain_hom_q, rain_hom_n,
        # Above-freezing cloud collection
        cloud_warm_q, cloud_warm_n
    )
end

#####
##### Individual field tendencies
#####
##### These functions combine process rates into tendencies for each prognostic field.
##### Phase 1 processes: autoconversion, accretion, evaporation, deposition, melting
##### Phase 2 processes: aggregation, riming, shedding, refreezing
#####

"""
    tendency_ρqᶜˡ(rates)

Compute cloud liquid mass tendency from P3 process rates.

Cloud liquid gains from:
- Condensation (Phase 1)

Cloud liquid is consumed by:
- Autoconversion (Phase 1)
- Accretion by rain (Phase 1)
- Riming by ice (Phase 2)
- Immersion freezing (Phase 2)
- Homogeneous freezing (Phase 2, T < -40°C)
"""
@inline function tendency_ρqᶜˡ(rates::P3ProcessRates, ρ)
    # Phase 1: condensation (positive = cloud forms)
    gain = rates.condensation
    # Phase 1: autoconversion and accretion
    # Phase 2: cloud riming by ice, immersion freezing, homogeneous freezing
    # Above-freezing: cloud collected by melting ice and shed as rain
    loss = rates.autoconversion + rates.accretion + rates.cloud_riming +
           rates.cloud_freezing_mass + rates.cloud_homogeneous_mass +
           rates.cloud_warm_collection
    return ρ * (gain - loss)
end

"""
    tendency_ρqʳ(rates)

Compute rain mass tendency from P3 process rates.

Rain gains from:
- Autoconversion (Phase 1)
- Accretion (Phase 1)
- Complete melting (Phase 1) - meltwater that sheds from ice
- Shedding (Phase 2) - liquid coating shed from ice
- Warm cloud collection (above freezing) - cloud swept by melting ice → rain

Rain loses from:
- Evaporation (Phase 1)
- Riming (Phase 2)
- Immersion freezing (Phase 2)
- Homogeneous freezing (Phase 2, T < -40°C)
"""
@inline function tendency_ρqʳ(rates::P3ProcessRates, ρ)
    # Phase 1: gains from autoconv, accr, complete_melt; loses from evap
    # Phase 2: gains from shedding; loses from riming, freezing, and homogeneous freezing
    # Above-freezing: cloud collected by melting ice shed as rain (Fortran qcshd)
    gain = rates.autoconversion + rates.accretion + rates.complete_melting +
           rates.shedding + rates.cloud_warm_collection
    loss = -rates.rain_evaporation + rates.rain_riming + rates.rain_freezing_mass +
           rates.rain_homogeneous_mass  # evap is negative
    return ρ * (gain - loss)
end

"""
    tendency_ρnʳ(rates, ρ, nⁱ, qⁱ, nʳ, qʳ, prp)

Compute rain number tendency from P3 process rates.

Rain number gains from:
- Autoconversion (Phase 1)
- Complete melting (Phase 1) - new rain drops from melted ice
- Breakup (Phase 1) - large drops fragment into smaller ones
- Shedding (Phase 2)
- Warm cloud collection (above freezing) - shed 1mm drops

Rain number loses from:
- Self-collection (Phase 1)
- Evaporation (Phase 1) - proportional number removal
- Riming (Phase 2)
- Immersion freezing (Phase 2)
- Homogeneous freezing (Phase 2, T < -40°C)
"""
@inline function tendency_ρnʳ(rates::P3ProcessRates, ρ, nⁱ, qⁱ, nʳ, qʳ, prp::ProcessRateParameters)
    FT = typeof(ρ)

    # Phase 1: New drops from autoconversion
    n_from_autoconv = rates.autoconversion / prp.initial_rain_drop_mass

    # Phase 1: New drops from complete melting (conserve number)
    # Only complete_melting produces new rain drops; partial_melting stays on ice
    n_from_melt = safe_divide(nⁱ * rates.complete_melting, qⁱ, zero(FT))

    # Phase 1: Evaporation removes rain number proportionally (Fortran P3 v5.5.0)
    # nr_evap = nr * (evap_rate / qr);  evap_rate is negative, so n_evap is negative
    n_from_evap = safe_divide(nʳ * rates.rain_evaporation, qʳ, zero(FT))

    return ρ * (n_from_autoconv + n_from_melt +
                n_from_evap +
                rates.rain_self_collection +
                rates.rain_breakup +
                rates.shedding_number +
                rates.cloud_warm_collection_number +
                rates.rain_riming_number -
                rates.rain_freezing_number -
                rates.rain_homogeneous_number)
end

# Backward-compatible overload without nʳ/qʳ (no evaporation number contribution)
@inline function tendency_ρnʳ(rates::P3ProcessRates, ρ, nⁱ, qⁱ, prp::ProcessRateParameters)
    FT = typeof(ρ)
    return tendency_ρnʳ(rates, ρ, nⁱ, qⁱ, zero(FT), one(FT), prp)
end

"""
    tendency_ρqⁱ(rates)

Compute ice mass tendency from P3 process rates.

Ice gains from:
- Deposition (Phase 1)
- Cloud riming (Phase 2)
- Rain riming (Phase 2)
- Refreezing (Phase 2)
- Deposition nucleation (Phase 2)
- Immersion freezing of cloud/rain (Phase 2)
- Rime splintering (Phase 2)
- Homogeneous freezing of cloud/rain (Phase 2, T < -40°C)

Ice loses from:
- Partial melting (Phase 1) - becomes liquid coating
- Complete melting (Phase 1) - sheds to rain
"""
@inline function tendency_ρqⁱ(rates::P3ProcessRates, ρ)
    # Phase 1: deposition, melting (both partial and complete reduce ice mass)
    # Phase 2: riming (cloud + rain), refreezing, nucleation, freezing, splintering
    # Splintering mass is already part of the riming mass (splinters fragment existing rime),
    # so it is NOT added here. Instead, it is subtracted from rime mass in tendency_ρqᶠ.
    gain = rates.deposition + rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.nucleation_mass + rates.cloud_freezing_mass + rates.rain_freezing_mass +
           rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass
    # Total melting reduces ice mass (partial stays as liquid coating, complete sheds)
    loss = rates.partial_melting + rates.complete_melting
    return ρ * (gain - loss)
end

"""
    tendency_ρnⁱ(rates)

Compute ice number tendency from P3 process rates.

Ice number gains from:
- Deposition nucleation (Phase 2)
- Immersion freezing of cloud/rain (Phase 2)
- Rime splintering (Phase 2)
- Homogeneous freezing of cloud/rain (Phase 2, T < -40°C)

Ice number loses from:
- Melting (Phase 1)
- Aggregation (Phase 2)
"""
@inline function tendency_ρnⁱ(rates::P3ProcessRates, ρ)
    # Gains from nucleation, freezing, splintering, homogeneous freezing
    gain = rates.nucleation_number + rates.cloud_freezing_number +
           rates.rain_freezing_number + rates.splintering_number +
           rates.cloud_homogeneous_number + rates.rain_homogeneous_number
    # melting_number and aggregation are already negative (represent losses)
    loss_rates = rates.melting_number + rates.aggregation
    return ρ * (gain + loss_rates)
end

"""
    tendency_ρqᶠ(rates)

Compute rime mass tendency from P3 process rates.

Rime mass gains from:
- Cloud riming (Phase 2)
- Rain riming (Phase 2)
- Refreezing (Phase 2)
- Immersion freezing (frozen cloud/rain becomes rimed ice) (Phase 2)
- Homogeneous freezing (frozen cloud/rain deposits as dense rime) (Phase 2, T < -40°C)

Rime mass loses from:
- Melting (proportional to rime fraction) (Phase 1)
"""
@inline function tendency_ρqᶠ(rates::P3ProcessRates, ρ, Fᶠ)
    # Phase 2: gains from riming, refreezing, freezing, and homogeneous freezing
    # Frozen cloud/rain becomes fully rimed ice (100% rime fraction for new frozen particles)
    gain = rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.cloud_freezing_mass + rates.rain_freezing_mass +
           rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass
    # Phase 1: melts proportionally with ice mass
    # Splintering mass is subtracted from rime (splinters fragment existing rime)
    loss = Fᶠ * (rates.partial_melting + rates.complete_melting) + rates.splintering_mass
    return ρ * (gain - loss)
end

"""
    tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, qⁱ, prp)

Compute rime volume tendency from P3 process rates.

Rime volume changes with rime mass: ∂bᶠ/∂t = ∂qᶠ/∂t / ρ_rime.
Includes melt-densification (Fortran P3 v5.5.0): during melting, low-density
rime portions melt preferentially, driving the remaining rime toward 917 kg/m³.
"""
@inline function tendency_ρbᶠ(rates::P3ProcessRates, ρ, Fᶠ, ρᶠ, qⁱ, prp)
    FT = typeof(ρ)

    ρᶠ_safe = max(ρᶠ, FT(100))
    ρ_rim_new_safe = max(rates.rime_density_new, FT(100))

    # Fortran P3 v5.5.0: rho_rimeMax = 900 for rain rime and freezing
    ρ_rimemax = prp.maximum_rime_density
    ρ_rim_hom = prp.pure_ice_density          # homogeneous freezing: solid ice sphere (917 kg/m³)

    # Phase 2: Volume gain from new rime
    # Cloud riming uses Cober-List computed density; rain riming uses rho_rimeMax = 900
    # Immersion freezing uses rho_rimeMax = 900 (Fortran convention, not water density)
    volume_gain = rates.cloud_riming / ρ_rim_new_safe +
                   rates.rain_riming / ρ_rimemax +
                   rates.refreezing / ρᶠ_safe +
                   (rates.cloud_freezing_mass + rates.rain_freezing_mass) / ρ_rimemax +
                   (rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass) / ρ_rim_hom

    # Phase 1: Volume loss from melting (proportional to rime fraction)
    total_melting = rates.partial_melting + rates.complete_melting
    volume_loss = Fᶠ * total_melting / ρᶠ_safe

    # M3: Melt-densification (Fortran P3 v5.5.0 lines 3841-3844)
    # Low-density rime portions melt first → remaining ice approaches 917 kg/m³.
    # In tendency form: additional volume reduction = bᶠ × (917 - ρᶠ) × |melt| / (ρᶠ × qⁱ)
    qⁱ_safe = max(qⁱ, FT(1e-12))
    bᶠ = Fᶠ * qⁱ_safe / ρᶠ_safe
    densification = bᶠ * (ρ_rim_hom - ρᶠ_safe) * total_melting / (ρᶠ_safe * qⁱ_safe)
    # Only apply when ρᶠ < 917 and there is melting
    densification = ifelse(ρᶠ_safe < ρ_rim_hom, densification, zero(FT))

    return ρ * (volume_gain - volume_loss - densification)
end

# Backward-compatible overloads
# qⁱ cancels in the densification term (bᶠ × ... / qⁱ), so any nonzero value is correct
@inline function tendency_ρbᶠ(rates::P3ProcessRates, ρ, Fᶠ, ρᶠ, prp::ProcessRateParameters)
    return tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, one(typeof(ρ)), prp)
end

# qⁱ cancels in the densification term (bᶠ × ... / qⁱ), so any nonzero value is correct
@inline function tendency_ρbᶠ(rates::P3ProcessRates, ρ, Fᶠ, ρᶠ)
    FT = typeof(ρ)
    prp = (pure_ice_density = FT(917), maximum_rime_density = FT(900))
    return tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, one(FT), prp)
end

"""
    tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ)

Compute ice sixth moment tendency from P3 process rates.

The sixth moment (reflectivity) changes with:
- Deposition (growth) (Phase 1)
- Melting (loss) (Phase 1)
- Riming (growth) (Phase 2)
- Nucleation (growth) (Phase 2)
- Aggregation (redistribution) (Phase 2)

This simplified version uses proportional scaling (Z/q ratio).
For more accurate 3-moment treatment, use the version that accepts
the p3 scheme to access tabulated sixth moment integrals.
"""
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ, prp::ProcessRateParameters)
    FT = typeof(ρ)

    # Simplified: Z changes proportionally to mass changes
    # More accurate version would use full integral formulation
    ratio = safe_divide(zⁱ, qⁱ, zero(FT))

    # Net mass change for ice
    # Total melting (partial + complete) reduces ice mass
    total_melting = rates.partial_melting + rates.complete_melting
    mass_change = rates.deposition - total_melting +
                  rates.cloud_riming + rates.rain_riming + rates.refreezing
    z_nuc = _nucleation_sixth_moment_tendency(rates.nucleation_number, prp)

    return ρ * (ratio * mass_change + z_nuc)
end

@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ)
    FT = typeof(ρ)
    prp = ProcessRateParameters(FT)
    return tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, prp)
end

"""
    tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, p3, nu, D_v)

Compute ice sixth moment tendency using tabulated integrals when available.

Following Milbrandt et al. (2021, 2024), the sixth moment tendency is
computed by integrating the contribution of each process over the
size distribution, properly accounting for how different processes
affect particles of different sizes.

When tabulated integrals are available via `tabulate(p3, arch)`, uses
pre-computed lookup tables. Otherwise, falls back to proportional scaling.

# Arguments
- `rates`: P3ProcessRates containing mass tendencies
- `ρ`: Air density [kg/m³]
- `qⁱ`: Ice mass mixing ratio [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `zⁱ`: Ice sixth moment [m⁶/kg]
- `Fᶠ`: Rime fraction [-]
- `Fˡ`: Liquid fraction [-]
- `p3`: P3 microphysics scheme (for accessing tabulated integrals)
- `nu`: Kinematic viscosity [m²/s]
- `D_v`: Water vapor diffusivity [m²/s]

# Returns
- Tendency of density-weighted sixth moment [kg/m³ × m⁶/kg / s]
"""
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3, nu, D_v)
    FT = typeof(ρ)

    # Mean ice particle mass for table lookup
    m̄ = safe_divide(qⁱ, nⁱ, FT(1e-20))
    log_mean_mass = log10(max(m̄, FT(1e-20)))

    # Schmidt number correction for enhanced ventilation integrals
    # Table stores 0.44 × ∫ C(D)√(V×D) N'(D) dD; runtime applies Sc^(1/3)/√ν
    # (see ventilation_sc_correction and _deposition_ventilation for dimensional derivation)
    sc_correction = ventilation_sc_correction(nu, D_v)

    z_tendency = _tabulated_z_tendency(
        p3.ice.sixth_moment, log_mean_mass, Fᶠ, Fˡ, ρᶠ, rates, ρ, qⁱ, nⁱ, zⁱ,
        p3.process_rates, sc_correction
    )

    return z_tendency
end

# Backward-compatible overload without transport properties (uses reference Sc correction)
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3)
    FT = typeof(ρ)
    # Reference conditions: nu ≈ 1.5e-5, D_v ≈ 2.2e-5 → Sc ≈ 0.68, Sc^(1/3)/√ν ≈ 227
    nu_ref = FT(1.5e-5)
    D_v_ref = FT(2.2e-5)
    return tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3, nu_ref, D_v_ref)
end

# Tabulated version: use TabulatedFunction4D lookups for each process
@inline function _tabulated_z_tendency(sixth::IceSixthMoment{<:TabulatedFunction4D},
                                        log_m, Fᶠ, Fˡ, ρᶠ, rates, ρ, qⁱ, nⁱ, zⁱ,
                                        prp::ProcessRateParameters, sc_correction)
    FT = typeof(ρ)

    # Look up normalized Z contribution for each process
    # deposition/sublimation have constant + enhanced ventilation integrals;
    # enhanced terms require Sc^(1/3)/√ν correction (same as H1 for mass deposition)
    z_dep = sixth.deposition(log_m, Fᶠ, Fˡ, ρᶠ) + sc_correction * sixth.deposition1(log_m, Fᶠ, Fˡ, ρᶠ)
    z_melt = sixth.melt1(log_m, Fᶠ, Fˡ, ρᶠ) + sixth.melt2(log_m, Fᶠ, Fˡ, ρᶠ)
    z_rime = sixth.rime(log_m, Fᶠ, Fˡ, ρᶠ)
    z_agg = sixth.aggregation(log_m, Fᶠ, Fˡ, ρᶠ)
    z_shed = sixth.shedding(log_m, Fᶠ, Fˡ, ρᶠ)
    z_sub = sixth.sublimation(log_m, Fᶠ, Fˡ, ρᶠ) + sc_correction * sixth.sublimation1(log_m, Fᶠ, Fˡ, ρᶠ)

    # Total melting
    total_melting = rates.partial_melting + rates.complete_melting

    # Compute Z tendency from tabulated integrals
    # Each integral gives the normalized Z rate per unit mass rate
    z_rate = z_dep * rates.deposition +
             z_rime * (rates.cloud_riming + rates.rain_riming) +
             z_agg * rates.aggregation * safe_divide(qⁱ, nⁱ, FT(1e-12)) +  # agg is number rate
             z_shed * rates.shedding -
             z_melt * total_melting

    # Sublimation correction (when deposition is negative):
    # z_dep and z_sub are DIFFERENT table integrals (SixthMomentDeposition vs
    # SixthMomentSublimation). This is not double-counting — deposition and
    # sublimation have separate normalized Z-change rates, as in Fortran P3.
    is_sublimating = rates.deposition < 0
    z_rate = z_rate + ifelse(is_sublimating, z_sub * abs(rates.deposition), zero(FT))
    z_rate = z_rate + _nucleation_sixth_moment_tendency(rates.nucleation_number, prp)

    return ρ * z_rate
end

# Fallback: use proportional scaling when integrals are not tabulated
@inline function _tabulated_z_tendency(::IceSixthMoment, log_m, Fᶠ, Fˡ, ρᶠ, rates, ρ, qⁱ, nⁱ, zⁱ,
                                       prp::ProcessRateParameters, sc_correction)
    # Fall back to the simple proportional scaling
    return tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, prp)
end

@inline function _nucleation_sixth_moment_tendency(nucleation_number, prp::ProcessRateParameters)
    FT = typeof(nucleation_number)
    D_nuc_cubed = 6 * prp.nucleated_ice_mass / (FT(π) * prp.pure_ice_density)
    return nucleation_number * D_nuc_cubed^2
end

"""
    tendency_ρqʷⁱ(rates)

Compute liquid on ice tendency from P3 process rates.

Liquid on ice:
- Gains from partial melting above freezing (meltwater stays on ice)
- Loses from shedding (Phase 2) - liquid sheds to rain
- Loses from refreezing (Phase 2) - liquid refreezes to ice

Following Milbrandt et al. (2025), partial melting adds to the liquid coating
while complete melting sheds directly to rain.
"""
@inline function tendency_ρqʷⁱ(rates::P3ProcessRates, ρ)
    # Gains from partial melting (meltwater stays on ice as liquid coating)
    # Loses from shedding (liquid sheds to rain) and refreezing (liquid refreezes)
    gain = rates.partial_melting
    loss = rates.shedding + rates.refreezing
    return ρ * (gain - loss)
end

"""
    tendency_ρqᵛ(rates)

Compute vapor mass tendency from P3 process rates.

Vapor is consumed by:
- Condensation (vapor → cloud liquid)
- Deposition (vapor → ice)
- Deposition nucleation (vapor → ice)

Vapor is produced by:
- Cloud evaporation (negative condensation)
- Rain evaporation
- Sublimation (negative deposition)
"""
@inline function tendency_ρqᵛ(rates::P3ProcessRates, ρ)
    # Condensation: positive = vapor loss, negative = vapor gain (evap)
    # Deposition: positive = vapor loss (dep), negative = vapor gain (sublimation)
    # Rain evaporation: negative = rain loss = vapor gain
    # Nucleation: always vapor loss
    return ρ * (-rates.condensation - rates.deposition - rates.nucleation_mass - rates.rain_evaporation)
end

#####
##### Fallback methods for Nothing rates
#####
##### These are safety fallbacks that return zero tendency when rates
##### have not been computed (e.g., during incremental development).
#####

@inline tendency_ρqᶜˡ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqʳ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnʳ(::Nothing, ρ, nⁱ, qⁱ, args...) = zero(ρ)
@inline tendency_ρqⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqᶠ(::Nothing, ρ, Fᶠ) = zero(ρ)
@inline tendency_ρbᶠ(::Nothing, ρ, Fᶠ, ρᶠ, prp...) = zero(ρ)
@inline tendency_ρzⁱ(::Nothing, ρ, qⁱ, nⁱ, zⁱ) = zero(ρ)
@inline tendency_ρqʷⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqᵛ(::Nothing, ρ) = zero(ρ)
