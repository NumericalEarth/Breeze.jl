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
                             adjustment_saturation_specific_humidity,
                             saturation_specific_humidity,
                             saturation_vapor_pressure,
                             PlanarLiquidSurface,
                             PlanarIceSurface,
                             density,
                             liquid_latent_heat,
                             ice_latent_heat,
                             vapor_gas_constant,
                             MoistureMassFractions,
                             ThermodynamicConstants
using DocStringExtensions: TYPEDSIGNATURES

#####
##### CCN activation
#####

"""
$(TYPEDSIGNATURES)

Compute CCN activation rate for the 1-moment (prescribed Nᶜ) case.

Following Fortran P3 v5.5.0 (lines 3953-3963): when the air is supersaturated
and the cloud mass is below the minimum threshold for the prescribed droplet
concentration, a seed mass is created. The target cloud mass is
``N_c / ρ × m_{\\text{drop}}`` where ``m_{\\text{drop}} = (4π/3) ρ_w r^3``
for ``r = 1`` μm. The rate is limited by the available supersaturation.

# Returns
- Rate of vapor → cloud liquid conversion from CCN activation [kg/kg/s]
"""
@inline function ccn_activation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants, cᵖᵐ)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    # Mass of a newly formed cloud droplet (Fortran cons7: radius 1 μm)
    cons7 = FT(4 * FT(π) / 3 * 1000 * (1e-6)^3)

    # Target cloud mass for prescribed droplet concentration
    target_qc = Nᶜ / ρ * cons7

    # Deficit: how much mass is needed to reach the minimum
    deficit = clamp_positive(target_qc - clamp_positive(qᶜˡ))

    # Psychrometric correction (liquid saturation)
    ℒˡ = liquid_latent_heat(T, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    Γˡ = 1 + (ℒˡ / cᵖᵐ) * dqᵛ⁺_dT

    # Limit by available supersaturation (Fortran: min(tmp1, (Qv_cld-dumqvs)/ab))
    max_from_ss = clamp_positive((qᵛ - qᵛ⁺ˡ) / Γˡ)
    rate = min(deficit, max_from_ss) / prp.sink_limiting_timescale

    # Only activate when supersaturated (Fortran threshold: sup_cld > 1e-6)
    is_supersaturated = (qᵛ - qᵛ⁺ˡ) / max(qᵛ⁺ˡ, FT(1e-10)) > FT(1e-6)
    return ifelse(is_supersaturated, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Dispatch CCN activation: prescribed (Nothing) or prognostic (AerosolActivation).
Returns `(; mass, number)` named tuple.
"""
@inline function compute_ccn_activation(::Nothing, p3, qᶜˡ, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants, cᵖᵐ)
    FT = typeof(qᶜˡ)
    # Prescribed-Nᶜ path (Fortran `log_predictNc = .false.`, `nc = nccnst_2`):
    # the activation target is the scheme parameter, not the DSD-diagnosed `Nᶜ`.
    # When `qᶜˡ` is below the mass threshold, `diagnose_cloud_dsd` clamps the
    # returned `Nᶜ` toward zero — using that value would collapse `target_qc`
    # and block any seed mass from forming in a warm-bubble parcel.
    target_Nᶜ = p3.cloud.number_concentration
    mass = ccn_activation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ, target_Nᶜ, constants, cᵖᵐ)
    return (; mass, number = zero(FT))
end

@inline function compute_ccn_activation(aerosol::AerosolActivation, p3, qᶜˡ, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants, cᵖᵐ)
    result = prognostic_ccn_activation_rate(aerosol, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T)
    return (; mass = result.qcnuc, number = result.ncnuc)
end

#####
##### Ice deposition and sublimation
#####

"""
$(TYPEDSIGNATURES)

Compute ventilation-enhanced ice deposition/sublimation rate with latent-heat
psychrometric correction.

Following Fortran P3 (`qidep = epsi·(qᵛ − qᵛ⁺ⁱ)/abi`), the single-particle growth
rate uses a vapor-diffusion-only resistance, with the latent-heat effect carried
once by the psychrometric factor ``Γⁱ`` (Fortran `abi`):

```math
\\frac{dm}{dt} = \\frac{4πC f_v (S_i - 1)}{Γⁱ \\, \\dfrac{R_v T}{e_{si} D_v}}
```

where ``Γⁱ = 1 + ℒⁱ^2 q^{v+i} / (R_v T^2 c_p^m)`` is the latent-heat psychrometric
correction (Fortran P3's `abi` factor). It accounts for the reduction in the
effective supersaturation drive caused by latent heat released during deposition
and is consistent with Breeze's `SaturationAdjustment` Jacobian linearisation.
The Mason (1971) thermal-conduction resistance ``\\frac{ℒⁱ}{K_a T}(\\frac{ℒⁱ}{R_v T} - 1)``
is deliberately *not* added alongside ``Γⁱ``: the two are equivalent latent-heat
formulations, so including both double-counts the resistance and under-predicts
dep/sub. Fortran applies only `abi`, once (microphy_p3.f90:3302-3305, 3712-3714).

The bulk rate integrates over the size distribution:

```math
\\frac{dq^i}{dt} = \\int \\frac{dm}{dt}(D)\\, N'(D)\\, dD
```

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Dry ice mass fraction [kg/kg]
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ⁱ`: Saturation vapor mass fraction over ice [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `T`: Temperature [K]
- `P`: Pressure [Pa]
- `constants`: Thermodynamic constants (or `nothing` for Fortran-matched hardcoded values)
- `transport`: Pre-computed air transport properties `(; D_v, K_a, nu)`
- `q`: Moisture mass fractions used to compute mixture heat capacity for ``Γⁱ``

# Returns
- Rate of vapor → ice conversion [kg/kg/s] (positive = deposition)
"""
function ventilation_enhanced_deposition(p3, qⁱ, qʷⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P,
                                                  constants, transport, q, μ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)

    # When runtime thermodynamic constants are provided, use their gas constants
    # consistently with the latent heat and saturation calculations.
    Rᵛ = FT(vapor_gas_constant(constants))
    Rᵈ = FT(dry_air_gas_constant(constants))
    ℒⁱ = sublimation_latent_heat(constants, T)
    # T,P-dependent transport properties (pre-computed or computed on demand)
    D_v = transport.D_v       # Diffusivity of water vapor [m²/s]
    nu  = transport.nu        # Kinematic viscosity [m²/s]

    # Saturation vapor pressure over ice
    # Derived from qᵛ⁺ⁱ: qᵛ⁺ⁱ = ε × e_si / (P - (1-ε) × e_si)
    # Rearranging: e_si = P × qᵛ⁺ⁱ / (ε + qᵛ⁺ⁱ × (1 - ε))
    ε = Rᵈ / Rᵛ
    qᵛ⁺ⁱ_safe = max(qᵛ⁺ⁱ, FT(1e-30))
    e_si = P * qᵛ⁺ⁱ_safe / (ε + qᵛ⁺ⁱ_safe * (1 - ε))

    # Supersaturation ratio with respect to ice
    S_i = qᵛ / max(qᵛ⁺ⁱ, FT(1e-10))

    # Mean particle mass
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    ρ_air = density(T, P, q, constants)
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ_air)

    # PSD-integrated ventilation integral C(D) × f_v(D) from lookup table.
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                    p3.ice.deposition.ventilation_enhanced,
                                    m_mean, Fᶠ, Fˡ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)

    # Vapor-diffusion resistance (Mason 1971 "B" term). The latent-heat
    # resistance is carried once by the psychrometric factor Γⁱ (= Fortran abi)
    # below; the Mason thermal-conduction term A = ℒⁱ/(K_a T)·(ℒⁱ/(Rᵛ T) − 1) is
    # deliberately omitted, because including both A and Γⁱ double-counts the
    # latent-heat resistance (they are equivalent formulations). Fortran P3 uses a
    # D_v-based ε and divides by abi once (microphy_p3.f90:3302-3305, 3712-3714).
    B = Rᵛ * T / (e_si * D_v)

    # Latent-heat psychrometric correction Γⁱ (Fortran P3 "abi"):
    # Reduces the effective supersaturation drive to account for the
    # warming produced by the latent heat of deposition.
    # Γⁱ = 1 + Lₛ² qᵛ⁺ⁱ / (Rᵛ T² cᵖᵈ)  ≡  1 + (Lₛ/cᵖᵈ) dqᵛ⁺ⁱ/dT
    Γⁱ = ice_psychrometric_correction(constants, ℒⁱ, qᵛ⁺ⁱ_safe, Rᵛ, T)

    # Deposition rate per particle. Equivalent to Fortran qidep = epsi·(qᵛ − qᵛ⁺ⁱ)/abi
    # with epsi = 2π ρ D_v n C_fv: in the (S_i − 1) form the vapor-diffusion
    # resistance is B, so the denominator is Γⁱ·B.
    # Uses 2π (not 4π) because the ventilation integral stores capm = cap × D
    # (P3 Fortran convention), which is 2× the physical capacitance C = D/2.
    # The product 2π × capm = 2π × 2C = 4πC is physically correct.
    dm_dt = 2 * FT(π) * C_fv * (S_i - 1) / (Γⁱ * B)

    # Scale by number concentration
    dep_rate = nⁱ_eff * dm_dt

    # Apply calibration factors (Fortran P3 v5.5.0 clbfact_dep, clbfact_sub).
    # These ad hoc multipliers account for uncertainty in ice capacitance.
    is_sublimation = S_i < 1
    cal = ifelse(is_sublimation, prp.calibration_factor_sublimation,
                                 prp.calibration_factor_deposition)
    dep_rate = dep_rate * cal

    # Limit sublimation to available ice
    τ_dep = prp.ice_deposition_timescale
    max_sublim = -qⁱ_eff / τ_dep

    return ifelse(is_sublimation, max(dep_rate, max_sublim), dep_rate)
end
