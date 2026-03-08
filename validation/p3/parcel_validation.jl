#####
##### P3 Parcel Model Validation
#####
##### Runs a microphysics-only integration of a single column of air
##### by calling compute_p3_process_rates at each timestep directly.
#####
##### Tests two scenarios:
#####   1. Ice-only: deposition growth and aggregation at T = -13 °C
#####   2. Warm rain: autoconversion and accretion at T = +10 °C
#####

using Breeze
using Breeze.Thermodynamics:
    ThermodynamicConstants,
    MoistureMassFractions,
    saturation_specific_humidity,
    PlanarLiquidSurface,
    PlanarIceSurface,
    temperature,
    with_temperature,
    liquid_latent_heat,
    vapor_gas_constant

using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState

using Breeze.Microphysics.PredictedParticleProperties:
    PredictedParticlePropertiesMicrophysics,
    P3MicrophysicalState,
    compute_p3_process_rates,
    tendency_ρqᶜˡ, tendency_ρqʳ, tendency_ρnʳ,
    tendency_ρqⁱ, tendency_ρnⁱ, tendency_ρqᶠ,
    tendency_ρbᶠ, tendency_ρzⁱ, tendency_ρqʷⁱ,
    tendency_ρqᵛ

using Printf

#####
##### Setup
#####

FT = Float64
constants = ThermodynamicConstants(FT)
p3 = PredictedParticlePropertiesMicrophysics(FT)

Rd = FT(287.05)

"""
    make_state(T, p, qv, qcl, qr, qi; constants)

Build a `LiquidIcePotentialTemperatureState` at temperature T [K], pressure p [Pa],
with given mixing ratios. Uses `with_temperature` so θ_l is consistent with T.
Returns (𝒰, θ_l) where θ_l is conserved under condensation/deposition.
"""
function make_state(T, p, qv, qcl, qr, qi, constants, FT)
    q     = MoistureMassFractions(qv, qcl + qr, qi)
    𝒰seed = LiquidIcePotentialTemperatureState(FT(300), q, FT(1e5), p)
    𝒰     = with_temperature(𝒰seed, T, constants)
    return 𝒰, 𝒰.potential_temperature
end

"""
    saturation_adjustment(qv, qcl, θ_l, qr, qi, p0, constants, Rd, FT)

Analytically adjust qv and qcl so the parcel is at liquid saturation,
accounting for latent heat feedback (one Newton step with the moist denominator).
This replaces the stiff explicit Euler condensation step.
"""
function saturation_adjustment(qv, qcl, θ_l, qr, qi, p0, constants, Rd, FT)
    q    = MoistureMassFractions(qv, qcl + qr, qi)
    𝒰    = LiquidIcePotentialTemperatureState(θ_l, q, FT(1e5), p0)
    T    = temperature(𝒰, constants)
    ρ    = p0 / (Rd * T)
    qvsl = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())

    Lv   = FT(liquid_latent_heat(T, constants))
    Rv   = FT(vapor_gas_constant(constants))
    cp   = FT(constants.dry_air.heat_capacity)

    ## One Newton step: excess = (qv - qvsl) / (1 + Lv²·qvsl / (cp·Rv·T²))
    denom = 1 + Lv^2 * qvsl / (cp * Rv * T^2)

    if qv > qvsl
        δ = (qv - qvsl) / denom
        δ = min(δ, qv)
        return qv - δ, qcl + δ
    elseif qv < qvsl && qcl > 0
        δ = (qvsl - qv) / denom
        δ = min(δ, qcl)
        return qv + δ, qcl - δ
    else
        return qv, qcl
    end
end

"""
    euler_step(qcl, qr, nr, qi, ni, qf, bf, zi, qwi, qv, rates, Δt, ρ, qi_for_nr, FT)

Apply one explicit Euler step for all slow microphysical processes.
Condensation is NOT applied here — use `saturation_adjustment` after this step.
Each pairwise mass transfer is limited so no field drops below zero.
Deposition is gated on qi > 0 (no ice crystals = no diffusional growth).
"""
function euler_step(qcl, qr, nr, qi, ni, qf, bf, zi, qwi, qv,
                    rates, Δt, ρ, qi_for_nr, FT)

    # --- Individual process increments (positive = mass transferred) ---

    # Rain processes
    δauto = min(rates.autoconversion * Δt, qcl)            # cloud → rain
    δaccr = min(rates.accretion      * Δt, max(0, qcl - δauto))  # cloud → rain
    δrevap = -min(-rates.rain_evaporation * Δt, qr)        # rain → vapor (negative)
    δself  = rates.rain_self_collection * Δt               # number change (negative)

    # Ice deposition: only if ice already exists (no ice crystals → no diffusional growth)
    dep = qi > 0 ? rates.deposition : zero(FT)
    if dep >= 0
        δdep = min(dep * Δt, qv)     # vapor → ice
    else
        δdep = -min(-dep * Δt, qi)   # sublimation, limited by qi
    end

    # Melting (ice → rain): positive only
    total_melt = (rates.partial_melting + rates.complete_melting) * Δt
    δmelt = min(total_melt, qi)

    # Aggregation: number change only, negative
    δagg_n = rates.aggregation * Δt

    # Cloud riming: cloud → rime (ice surface)
    δcrime = min(rates.cloud_riming * Δt, max(0, qcl - δauto - δaccr))

    # Rain riming: rain → ice
    δrrime = min(rates.rain_riming * Δt, qr)

    # Cloud freezing: cloud → ice (immersion)
    qcl_remaining = max(0, qcl - δauto - δaccr - δcrime)
    δcfrz  = min(rates.cloud_freezing_mass * Δt, qcl_remaining)

    # Rain freezing: rain → ice
    qr_remaining = max(0, qr + δauto + δaccr - δrrime - δrevap)
    δrfrz  = min(rates.rain_freezing_mass * Δt, qr_remaining)

    # Shedding / refreezing (liquid-on-ice processes)
    δshed   = min(rates.shedding   * Δt, qwi)
    δrefrz  = min(rates.refreezing * Δt, qwi - δshed)
    δpmelt  = min(rates.partial_melting * Δt, qi)

    # Nucleation
    δnuc   = rates.nucleation_mass * Δt
    δspl   = rates.splintering_mass * Δt

    # Rime density for new rime
    ρf_new = rates.rime_density_new > 0 ? rates.rime_density_new : FT(400)

    # --- Balanced field updates (condensation handled separately via saturation_adjustment) ---
    qcl_new = max(zero(FT), qcl         - δauto - δaccr - δcrime - δcfrz)
    qr_new  = max(zero(FT), qr  + δauto + δaccr + rates.complete_melting*Δt + δshed
                                 + δrevap - δrrime - δrfrz)
    qi_new  = max(zero(FT), qi  + δdep + δcrime + δrrime + δcfrz + δrfrz
                                + δnuc + δspl + δrefrz - δpmelt - δmelt)
    qf_new  = max(zero(FT), qf  + δcrime + δrrime)
    bf_new  = max(zero(FT), bf  + (δcrime + δrrime) / ρf_new)
    qwi_new = max(zero(FT), qwi + δpmelt - δshed - δrefrz)
    qv_new  = max(zero(FT), qv         - δdep - δnuc + (-δrevap))
    zi_new  = max(zero(FT), zi  + tendency_ρzⁱ(rates, ρ, qi, ni, zi) / ρ * Δt)

    # Number updates (not flux-limited for simplicity)
    ni_new = max(zero(FT), ni + δagg_n
                              + rates.nucleation_number * Δt
                              + rates.cloud_freezing_number * Δt
                              + rates.rain_freezing_number * Δt
                              + rates.splintering_number * Δt
                              - (qi > 0 ? ni / qi * δmelt : 0.0))
    m0_rain = FT(5e-10)
    nr_new  = max(zero(FT), nr + δauto / m0_rain
                               + (qi_for_nr > 0 ? ni / qi_for_nr : 0.0) * rates.complete_melting * Δt
                               + rates.shedding_number * Δt
                               + δself
                               + rates.rain_riming_number * Δt
                               + rates.rain_freezing_number * Δt)

    return qcl_new, qr_new, nr_new, qi_new, ni_new, qf_new, bf_new, zi_new, qwi_new, qv_new
end

#####
##### Integration loop
#####

function run_scenario(p3, constants, Rd, name, T0, p0, Δt, N, output_every,
                      qcl0, qr0, nr0, qi0, ni0, qf0, bf0, zi0, qwi0, qv0, FT)

    println("="^70)
    @printf("Scenario: %s\n", name)
    println("="^70)

    ρ0 = p0 / (Rd * T0)
    qvsi0 = saturation_specific_humidity(T0, ρ0, constants, PlanarIceSurface())
    qvsl0 = saturation_specific_humidity(T0, ρ0, constants, PlanarLiquidSurface())

    @printf("Initial: T=%.1f K (%.1f°C), p=%.0f Pa, ρ=%.3f kg/m³\n",
            T0, T0-273.15, p0, ρ0)
    @printf("         qv=%.3f g/kg (%.1f%% Si, %.1f%% Sl)  qcl=%.3f g/kg  qr=%.3f g/kg\n",
            qv0*1000, qv0/qvsi0*100, qv0/qvsl0*100, qcl0*1000, qr0*1000)
    @printf("         qi=%.3f g/kg  ni=%.2e/kg  Ff=%.3f\n",
            qi0*1000, ni0, qi0 > 0 ? qf0/qi0 : 0.0)
    println()

    𝒰, θ_l = make_state(T0, p0, qv0, qcl0, qr0, qi0, constants, FT)

    # Report initial rates
    ℳ0 = P3MicrophysicalState(qcl0, qr0, nr0, qi0, ni0, qf0, bf0, zi0, qwi0)
    r0 = compute_p3_process_rates(p3, ρ0, ℳ0, 𝒰, constants)
    @printf("  condensation=%.2e  autoconv=%.2e  deposition=%.2e  aggregation=%.2e\n",
            r0.condensation, r0.autoconversion, r0.deposition, r0.aggregation)
    @printf("  cloud_riming=%.2e  cloud_freezing=%.2e  splintering_n=%.2e\n",
            r0.cloud_riming, r0.cloud_freezing_mass, r0.splintering_number)
    println()

    @printf("%-8s %-10s %-10s %-10s %-10s %-8s %-6s %-8s\n",
            "t [s]", "qcl [g/kg]", "qr [g/kg]", "qi [g/kg]", "ni [/kg]",
            "T [K]", "Ff", "Si [%]")
    println("-"^74)

    qcl, qr, nr, qi, ni, qf, bf, zi, qwi, qv = qcl0, qr0, nr0, qi0, ni0, qf0, bf0, zi0, qwi0, qv0

    for n in 1:N
        t = n * Δt
        q = MoistureMassFractions(qv, qcl + qr, qi)
        𝒰 = LiquidIcePotentialTemperatureState(θ_l, q, FT(1e5), p0)
        T = temperature(𝒰, constants)
        ρ = p0 / (Rd * T)

        ℳ = P3MicrophysicalState(qcl, qr, nr, qi, ni, qf, bf, zi, qwi)
        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)

        qi_for_nr = qi
        qcl, qr, nr, qi, ni, qf, bf, zi, qwi, qv =
            euler_step(qcl, qr, nr, qi, ni, qf, bf, zi, qwi, qv,
                       rates, Δt, ρ, qi_for_nr, FT)

        ## Saturation adjustment: condense/evaporate to maintain liquid saturation
        qv, qcl = saturation_adjustment(qv, qcl, θ_l, qr, qi, p0, constants, Rd, FT)

        if mod(n, output_every) == 0
            q_out = MoistureMassFractions(qv, qcl + qr, qi)
            𝒰_out = LiquidIcePotentialTemperatureState(θ_l, q_out, FT(1e5), p0)
            T_out = temperature(𝒰_out, constants)
            ρ_out = p0 / (Rd * T_out)
            qvsi  = saturation_specific_humidity(T_out, ρ_out, constants, PlanarIceSurface())
            Si_pct = qvsi > 0 ? qv / qvsi * 100 : 0.0
            Ff     = qi > 1e-12 ? qf / qi : zero(FT)
            @printf("%-8.0f %-10.4f %-10.4f %-10.4f %-10.2e %-8.3f %-6.3f %-8.1f\n",
                    t, qcl*1000, qr*1000, qi*1000, ni, T_out, Ff, Si_pct)
        end
    end

    q_fin  = MoistureMassFractions(qv, qcl + qr, qi)
    𝒰_fin  = LiquidIcePotentialTemperatureState(θ_l, q_fin, FT(1e5), p0)
    T_fin  = temperature(𝒰_fin, constants)
    ρ_fin  = p0 / (Rd * T_fin)
    qvsi   = saturation_specific_humidity(T_fin, ρ_fin, constants, PlanarIceSurface())
    qvsl   = saturation_specific_humidity(T_fin, ρ_fin, constants, PlanarLiquidSurface())
    println()
    @printf("Final:   T=%.2f K (%.2f°C)\n", T_fin, T_fin-273.15)
    @printf("         qcl=%.4f g/kg  qr=%.4f g/kg  qi=%.4f g/kg\n",
            qcl*1000, qr*1000, qi*1000)
    @printf("         ni=%.2e/kg  Ff=%.4f  Si=%.1f%%\n",
            ni, qi > 1e-12 ? qf/qi : 0.0, qvsi > 0 ? qv/qvsi*100 : 0.0)
    println()

    return (; T=T_fin, qcl, qr, nr, qi, ni, qf, bf, zi, qwi, qv)
end

#####
##### Scenario 1: Ice deposition and aggregation at T = -13 °C
#####
# Initial state: existing ice, vapor at 102% ice saturation.
# No cloud liquid → no immersion freezing, no riming.
# Expected: ice grows via Bergeron-Findeisen deposition; number decreases via aggregation.

let
    T0   = FT(260)
    p0   = FT(60000)
    ρ0   = p0 / (Rd * T0)
    qvsi = saturation_specific_humidity(T0, ρ0, constants, PlanarIceSurface())

    qcl0 = FT(0)
    qr0  = FT(0)
    nr0  = FT(0)
    qi0  = FT(5e-4)   # 0.5 g/kg
    ni0  = FT(1e5)    # 1e5 /kg
    qf0  = FT(0)
    bf0  = FT(0)
    zi0  = FT(1e-11)
    qwi0 = FT(0)
    qv0  = FT(1.02) * qvsi   # 2% supersaturation over ice

    run_scenario(p3, constants, Rd,
                 "Ice deposition + aggregation  (T = -13°C, no cloud liquid)",
                 T0, p0, FT(10), 600, 60,
                 qcl0, qr0, nr0, qi0, ni0, qf0, bf0, zi0, qwi0, qv0, FT)
end

#####
##### Scenario 2: Mixed-phase cloud at T = -5 °C (Hallett-Mossop zone)
#####
# Initial state: cloud liquid and ice coexist.
# Expected: riming (cloud → rime on ice), splintering (secondary ice), deposition.
# Small qcl (0.1 g/kg) to keep freezing rate manageable with Δt = 10 s.

let
    T0   = FT(268)
    p0   = FT(70000)
    ρ0   = p0 / (Rd * T0)
    qvsi = saturation_specific_humidity(T0, ρ0, constants, PlanarIceSurface())

    qcl0 = FT(1e-4)   # 0.1 g/kg
    qr0  = FT(0)
    nr0  = FT(0)
    qi0  = FT(2e-4)   # 0.2 g/kg
    ni0  = FT(1e5)
    qf0  = FT(1e-5)   # slight riming already present
    bf0  = FT(1e-5 / 400)
    zi0  = FT(1e-11)
    qwi0 = FT(0)
    qv0  = FT(1.01) * qvsi

    run_scenario(p3, constants, Rd,
                 "Mixed-phase cloud  (T = -5°C, Hallett-Mossop zone)",
                 T0, p0, FT(10), 600, 60,
                 qcl0, qr0, nr0, qi0, ni0, qf0, bf0, zi0, qwi0, qv0, FT)
end

#####
##### Scenario 3: Warm rain  (T = +10 °C)
#####
# Initial state: cloud liquid at saturation, no rain, no ice.
# Condensation is handled via saturation adjustment (not explicit Euler)
# so the stiffness of the condensation timescale is irrelevant.
# Expected: autoconversion slowly converts cloud → rain; accretion then
# rapidly depletes cloud liquid once rain is established.

let
    T0   = FT(283)
    p0   = FT(85000)
    ρ0   = p0 / (Rd * T0)
    qvsl = saturation_specific_humidity(T0, ρ0, constants, PlanarLiquidSurface())

    qcl0 = FT(5e-4)   # 0.5 g/kg cloud liquid
    qr0  = FT(0)
    nr0  = FT(0)
    qi0  = FT(0)
    ni0  = FT(0)
    qf0  = FT(0)
    bf0  = FT(0)
    zi0  = FT(0)
    qwi0 = FT(0)
    qv0  = qvsl       # at liquid saturation

    run_scenario(p3, constants, Rd,
                 "Warm rain  (T = +10°C, cloud only, 100% Sl)",
                 T0, p0, FT(10), 600, 60,
                 qcl0, qr0, nr0, qi0, ni0, qf0, bf0, zi0, qwi0, qv0, FT)
end
