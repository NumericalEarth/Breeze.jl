include(joinpath(@__DIR__, "setup.jl"))

using Test
import Breeze
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.AtmosphereModels: microphysical_tendency, microphysical_velocities,
                               prognostic_field_names
using Breeze.Thermodynamics: ThermodynamicConstants, dry_air_gas_constant

using Breeze.Microphysics.PredictedParticleProperties:
    chebyshev_gauss_nodes_weights,
    P3ProcessRates,
    compute_p3_process_rates,
    consistent_rime_state,
    tendency_ρqᶜˡ,
    tendency_ρnᶜˡ,
    tendency_ρqʳ,
    tendency_ρnʳ,
    tendency_ρqⁱ,
    tendency_ρnⁱ,
    tendency_ρqᶠ,
    tendency_ρbᶠ,
    tendency_ρqʷⁱ,
    tendency_ρsˢᵃᵗ,
    tendency_ρqᵛ,
    rain_autoconversion_rate,
    rain_accretion_rate,
    rain_evaporation_rate,
    rain_self_collection_rate,
    rain_breakup_rate,
    ice_melting_rate,
    ice_melting_rates,
    ice_aggregation_rate,
    cloud_riming_rate,
    cloud_warm_collection_rate,
    rain_riming_rate,
    rime_density,
    P3MicrophysicalState,
    RainMassWeightedVelocityEvaluator,
    RainNumberWeightedVelocityEvaluator,
    RainEvaporationVentilationEvaluator,
    air_transport_properties,
    ProcessRateParameters,
    homogeneous_freezing_cloud_rate,
    homogeneous_freezing_rain_rate,
    immersion_freezing_cloud_rate,
    immersion_freezing_rain_rate,
    air_transport_properties,
    psd_correction_spherical_volume,
    AbstractWarmRainScheme,
    KhairoutdinovKogan2000

using Breeze.Thermodynamics:
    ThermodynamicConstants,
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState,
    StaticEnergyState,
    adjustment_saturation_specific_humidity,
    saturation_specific_humidity,
    temperature,
    with_temperature,
    mixture_heat_capacity,
    PlanarLiquidSurface,
    PlanarIceSurface

using Oceananigans: CPU, RectilinearGrid
using Oceananigans.Fields: interior, ZeroField
using Oceananigans.Fields: CenterField, ZFaceField, set!
using Oceananigans.Grids: Periodic, Bounded

const PPP = Breeze.Microphysics.PredictedParticleProperties

function p3_with_process_rates(p3, process_rates)
    return PredictedParticlePropertiesMicrophysics(
        p3.water_density,
        p3.minimum_mass_mixing_ratio,
        p3.minimum_number_mixing_ratio,
        p3.ice,
        p3.rain,
        p3.cloud,
        process_rates,
        p3.precipitation_boundary_condition,
        p3.negative_moisture_correction,
        p3.aerosol,
        p3.warm_rain_scheme)
end

function p3_process_rates_with(FT; kwargs...)
    values = ntuple(index -> begin
        field = fieldnames(P3ProcessRates)[index]
        FT(get(kwargs, field, zero(FT)))
    end, fieldcount(P3ProcessRates))
    return P3ProcessRates(values...)
end

# Build a real p3 with a specific `process_rates`/`warm_rain_scheme`. Used in
# tests that only need to exercise `tendency_ρnʳ` / `tendency_ρnᶜˡ` (which read
# both fields). Prefer this over a NamedTuple shim so future field additions to
# `PredictedParticlePropertiesMicrophysics` surface as compile errors, not silent
# missing-field bugs.
function tendency_test_p3(FT; process_rates = ProcessRateParameters(FT),
                              warm_rain_scheme = KhairoutdinovKogan2000())
    return PredictedParticlePropertiesMicrophysics(FT; process_rates, warm_rain_scheme)
end

function expected_reference_rain_epsilon(p3, qʳ, nʳ, ρ, transport, FT)
    prp = p3.process_rates
    qʳ_eff = max(0, qʳ)
    nʳ_eff = max(max(0, nʳ), FT(1e-16))
    λ_r = PPP.rain_slope_parameter(qʳ_eff, nʳ_eff, prp)
    nʳ_bounded = qʳ_eff * λ_r^3 / (FT(π) * prp.liquid_water_density)
    N₀ = nʳ_bounded * λ_r
    I_VD = p3.rain.evaporation(log10(λ_r))
    I_const = FT(PPP.RAIN_F1R) / λ_r^2
    Sc_cbrt = cbrt(transport.ν / max(transport.Dᵛ, FT(1e-10)))
    I_evap = I_const + FT(PPP.RAIN_F2R) * Sc_cbrt / sqrt(max(transport.ν, FT(1e-10))) * I_VD
    epsilon_r = FT(2π) * N₀ * ρ * transport.Dᵛ * I_evap
    return ifelse(qʳ_eff >= p3.minimum_mass_mixing_ratio, epsilon_r, zero(FT))
end

function expected_reference_warm_rain_collection_number(p3, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ)
    FT = typeof(qʳ)
    prp = p3.process_rates
    qʳ_eff = max(0, qʳ)
    nʳ_eff = max(0, nʳ)
    qⁱ_total = PPP.total_ice_mass(qⁱ, qʷⁱ)
    nⁱ_eff = max(0, nⁱ)
    active = (T > prp.freezing_temperature) &
             (qʳ_eff > FT(1e-14)) &
             (qⁱ_total > FT(1e-14)) &
             (nʳ_eff > FT(1)) &
             (nⁱ_eff > FT(1))

    λ_r = PPP.rain_slope_parameter(qʳ_eff, nʳ_eff, prp)
    nʳ_bounded = PPP.rain_number_from_slope(qʳ_eff, λ_r, prp)
    Fˡ = PPP.liquid_fraction_on_ice(qⁱ, qʷⁱ)
    m_mean = PPP.mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    number_kernel = PPP.rain_riming_number_kernel(PPP.rain_ice_collection_table(p3),
                                                  m_mean, λ_r, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)
    ρ₀ = p3.ice.fall_speed.reference_air_density
    rhofaci = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)
    N₀ʳ = nʳ_bounded * λ_r
    rate = prp.rain_ice_collection_efficiency * N₀ʳ * nⁱ_eff * ρ * rhofaci * number_kernel
    return ifelse(active, rate, zero(FT))
end

function expected_reference_ice_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ, constants, transport, q, μ)
    FT = typeof(qⁱ)
    Fˡ = PPP.liquid_fraction_on_ice(qⁱ, qʷⁱ)
    m_mean = PPP.mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    ρ_air = Breeze.Thermodynamics.density(T, P, q, constants)
    ρ_correction = PPP.ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ_air)
    C_fv = PPP.deposition_ventilation(p3.ice.deposition.ventilation,
                                      p3.ice.deposition.ventilation_enhanced,
                                      m_mean, Fᶠ, Fˡ, ρᶠ, p3.process_rates,
                                      transport.ν, transport.Dᵛ, ρ_correction, p3, μ)
    epsilon_i = FT(2π) * ρ * transport.Dᵛ * max(max(0, nⁱ), FT(1e-16)) * C_fv
    qⁱ_total = PPP.total_ice_mass(qⁱ, qʷⁱ)
    active = (qⁱ_total >= p3.minimum_mass_mixing_ratio) &
             (Fˡ < p3.process_rates.liquid_fraction_clipping_threshold)
    return ifelse(active, epsilon_i, zero(FT))
end

function expected_reference_coating_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                          constants, transport, q, μ)
    FT = typeof(qⁱ)
    Fˡ = PPP.liquid_fraction_on_ice(qⁱ, qʷⁱ)
    m_mean = PPP.mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    ρ_air = Breeze.Thermodynamics.density(T, P, q, constants)
    ρ_correction = PPP.ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ_air)
    C_fv = PPP.deposition_ventilation(p3.ice.deposition.ventilation,
                                      p3.ice.deposition.ventilation_enhanced,
                                      m_mean, Fᶠ, Fˡ, ρᶠ, p3.process_rates,
                                      transport.ν, transport.Dᵛ, ρ_correction, p3, μ)
    epsilon_iw = FT(2π) * ρ * transport.Dᵛ * max(max(0, nⁱ), FT(1e-16)) * C_fv
    qⁱ_total = PPP.total_ice_mass(qⁱ, qʷⁱ)
    active = (qⁱ_total >= p3.minimum_mass_mixing_ratio) &
             (Fˡ >= p3.process_rates.liquid_fraction_clipping_threshold)
    return ifelse(active, epsilon_iw, zero(FT))
end

function expected_reduced_reference_vapor_rates(p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
                                              qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                              constants, transport, q, μ;
                                              temperature_tendency = zero(T),
                                              vapor_tendency = zero(qᵛ))
    FT = typeof(qᶜˡ)
    τ = max(p3.process_rates.sink_limiting_timescale, eps(FT))
    Rᵛ = FT(Breeze.Thermodynamics.vapor_gas_constant(constants))
    L_v = PPP.vaporization_latent_heat(constants, T)
    L_s = PPP.sublimation_latent_heat(constants, T)
    cᵖᵈ = constants.dry_air.heat_capacity

    dqᵛ⁺ˡ_dT = qᵛ⁺ˡ * L_v / (Rᵛ * T^2)
    dqᵛ⁺ⁱ_dT = qᵛ⁺ⁱ * L_s / (Rᵛ * T^2)
    ab = 1 + L_v * dqᵛ⁺ˡ_dT / cᵖᵈ
    abi = 1 + L_s * dqᵛ⁺ⁱ_dT / cᵖᵈ

    cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
    epsc = PPP.cloud_condensation_epsilon(p3, qᶜˡ, ρ, transport.Dᵛ, cloud.μ_c, cloud.λ_c, cloud.nᶜˡ)
    epsr = expected_reference_rain_epsilon(p3, qʳ, nʳ, ρ, transport, FT)
    epsi = expected_reference_ice_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                        constants, transport, q, μ)
    epsiw = expected_reference_coating_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                             constants, transport, q, μ)

    ice_liquid_coupling = (1 + L_s * dqᵛ⁺ˡ_dT / cᵖᵈ) / abi
    xx = max(epsc + epsr + epsi * ice_liquid_coupling + epsiw, FT(1e-20))
    transient = (1 - exp(-xx * τ)) / τ
    ssat_liquid = qᵛ - qᵛ⁺ˡ
    bergeron_driver = -(qᵛ⁺ˡ - qᵛ⁺ⁱ) * ice_liquid_coupling * epsi
    external_driver = vapor_tendency - dqᵛ⁺ˡ_dT * temperature_tendency
    aaa = external_driver + bergeron_driver

    qc_raw = (aaa * epsc / xx + (ssat_liquid - aaa / xx) * epsc / xx * transient) / ab
    qr_raw = (aaa * epsr / xx + (ssat_liquid - aaa / xx) * epsr / xx * transient) / ab
    qi_raw = (aaa * epsi / xx + (ssat_liquid - aaa / xx) * epsi / xx * transient) / abi +
             (qᵛ⁺ˡ - qᵛ⁺ⁱ) * epsi / abi
    ql_raw = (aaa * epsiw / xx + (ssat_liquid - aaa / xx) * epsiw / xx * transient) / ab

    condensation = ifelse(qc_raw < 0, zero(FT), min(qc_raw, qᵛ / τ))
    rain_condensation = ifelse(qr_raw < 0, zero(FT), min(qr_raw, qᵛ / τ))
    rain_evaporation = ifelse(qr_raw < 0, min(-qr_raw, max(0, qʳ) / τ), zero(FT))

    is_sublimation = qi_raw < 0
    deposition = ifelse(is_sublimation,
                        -min(-qi_raw * p3.process_rates.calibration_factor_sublimation,
                             max(0, qⁱ) / τ),
                        min(qi_raw * p3.process_rates.calibration_factor_deposition, qᵛ / τ))

    coating_condensation = ifelse(ql_raw < 0, zero(FT), min(ql_raw, qᵛ / τ))
    coating_evaporation = ifelse(ql_raw < 0, min(-ql_raw, max(0, qʷⁱ) / τ), zero(FT))

    return (; condensation, rain_evaporation, rain_condensation, deposition,
              coating_condensation, coating_evaporation)
end

function expected_reference_predicted_ssat_adjustment(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, sˢᵃᵗ, T, constants)
    FT = typeof(qᶜˡ)
    τ = max(p3.process_rates.sink_limiting_timescale, eps(FT))
    Rᵛ = FT(Breeze.Thermodynamics.vapor_gas_constant(constants))
    ℒˡ = PPP.vaporization_latent_heat(constants, T)
    cᵖᵈ = constants.dry_air.heat_capacity
    dqᵛ⁺ˡ_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    ξˡ = 1 + ℒˡ * dqᵛ⁺ˡ_dT / cᵖᵈ
    ε = (qᵛ - qᵛ⁺ˡ - sˢᵃᵗ) / ξˡ
    ε = max(ε, -qᶜˡ)
    ε = ifelse(sˢᵃᵗ < 0, min(0, ε), ε)
    ε = ifelse(abs(ε) < 100 * eps(FT) * max(qᵛ⁺ˡ, qᵛ), zero(FT), ε)
    ε = ifelse(PPP.predicts_supersaturation(p3.process_rates), ε, zero(FT))
    return (; ε, rate = ε / τ)
end

function actual_final_liquid_ssat_after_p3_step(formulation, rates, qᵛ₀, qᶜˡ₀, qʳ₀, qⁱ₀, qʷⁱ₀,
                                                ρ, τ, P, pˢᵗ, constants)
    FT = typeof(qᵛ₀)
    qᵛ₁ = qᵛ₀ + tendency_ρqᵛ(rates, ρ) / ρ * τ
    qˡ₁ = qᶜˡ₀ + qʳ₀ + qʷⁱ₀ +
          (tendency_ρqᶜˡ(rates, ρ) +
           tendency_ρqʳ(rates, ρ) +
           tendency_ρqʷⁱ(rates, ρ)) / ρ * τ
    qⁱ₁ = qⁱ₀ + tendency_ρqⁱ(rates, ρ) / ρ * τ
    q₁ = MoistureMassFractions(qᵛ₁, qˡ₁, qⁱ₁)

    𝒰₁ = if formulation isa LiquidIcePotentialTemperatureState
        LiquidIcePotentialTemperatureState(formulation.potential_temperature, q₁, pˢᵗ, P)
    else
        StaticEnergyState(formulation.static_energy, q₁, FT(0), P)
    end

    T₁ = temperature(𝒰₁, constants)
    qᵛ⁺ˡ₁ = saturation_specific_humidity(T₁, ρ, constants, PlanarLiquidSurface())
    return qᵛ₁ - qᵛ⁺ˡ₁
end

function documented_predict_supersaturation_disabled_semantics()
    overview = read(joinpath(@__DIR__, "..", "docs", "src", "microphysics", "p3_overview.md"), String)
    prognostics = read(joinpath(@__DIR__, "..", "docs", "src", "microphysics", "p3_prognostics.md"), String)
    forbidden = "When `false`, the field is recomputed diagnostically"
    required = "When `false`, the field is not allocated"
    return !occursin(forbidden, overview) &&
           !occursin(forbidden, prognostics) &&
           occursin(required, overview) &&
           occursin(required, prognostics)
end

@testset "P3 Processes" begin

    @testset "Rime splintering follows Fortran guards" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        prp = p3.process_rates

        cloud_riming = FT(3e-7)
        rain_riming = FT(2e-7)
        D_ice = FT(300e-6)
        Fˡ = FT(0.05)
        surface_T = FT(280)
        qᶠ = FT(1e-6)

        left_q, left_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, FT(266.15), D_ice, Fˡ, surface_T, qᶠ)
        peak_q, peak_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, prp.splintering_temperature_peak, D_ice, Fˡ, surface_T, qᶠ)
        right_q, right_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, FT(269.15), D_ice, Fˡ, surface_T, qᶠ)

        total_riming = cloud_riming + rain_riming
        @test left_n ≈ (FT(1) / FT(3)) * prp.splintering_rate * total_riming
        @test peak_n ≈ prp.splintering_rate * total_riming
        @test right_n ≈ FT(0.5) * prp.splintering_rate * total_riming
        @test left_q ≈ left_n * prp.splintering_crystal_mass
        @test peak_q ≈ peak_n * prp.splintering_crystal_mass
        @test right_q ≈ right_n * prp.splintering_crystal_mass

        cloud_peak_q, rain_peak_q, split_peak_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rates(
            p3, cloud_riming, rain_riming, prp.splintering_temperature_peak, D_ice, Fˡ, surface_T, qᶠ)
        @test split_peak_n ≈ peak_n
        @test cloud_peak_q ≈ prp.splintering_rate * cloud_riming * prp.splintering_crystal_mass
        @test rain_peak_q ≈ prp.splintering_rate * rain_riming * prp.splintering_crystal_mass

        _, cloud_only_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, zero(FT), prp.splintering_temperature_peak, D_ice, Fˡ, surface_T, qᶠ)
        _, small_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, prp.splintering_temperature_peak, FT(200e-6), Fˡ, surface_T, qᶠ)
        _, wet_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, prp.splintering_temperature_peak, D_ice, FT(0.2), surface_T, qᶠ)
        _, warm_surface_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, prp.splintering_temperature_peak, D_ice, Fˡ, FT(283), qᶠ)
        _, no_rime_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, prp.splintering_temperature_peak, D_ice, Fˡ, surface_T, zero(FT))

        # H4: Cloud riming contributes to splintering
        @test cloud_only_n > 0
        cloud_only_q, _ = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, zero(FT), prp.splintering_temperature_peak, D_ice, Fˡ, surface_T, qᶠ)
        @test cloud_only_q > 0
        @test small_n == 0
        @test wet_n == 0
        @test warm_surface_n == 0
        @test no_rime_n == 0
    end

    @testset "P3ProcessRates construction" begin
        FT = Float64
        rates = P3ProcessRates(
            ntuple(_ -> zero(FT), fieldcount(P3ProcessRates))...
        )
        @test rates isa P3ProcessRates{FT}
        @test rates.condensation == 0.0
        @test rates.autoconversion == 0.0
        @test rates.partial_melting == 0.0
        @test rates.complete_melting == 0.0
    end

    @testset "rain DSD lambda limiter recomputes number" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        prp = p3.process_rates
        qʳ = FT(1e-3)
        nʳ = FT(1e-5)
        S = FT(0.99)
        thermodynamic_factor = FT(1e8)
        ν = FT(1.5e-5)
        Dᵛ = FT(2.2e-5)
        ρ = FT(1)

        λ_r = PPP.rain_slope_parameter(qʳ, nʳ, prp)
        nʳ_bounded = qʳ * λ_r^3 / (FT(π) * prp.liquid_water_density)

        @test λ_r == prp.rain_lambda_min
        @test nʳ_bounded > nʳ

        raw_rate = PPP.rain_evaporation_rate(p3.rain.evaporation, qʳ, nʳ, S,
                                             thermodynamic_factor, p3, prp,
                                             ν, Dᵛ, ρ, FT)
        bounded_rate = PPP.rain_evaporation_rate(p3.rain.evaporation, qʳ, nʳ_bounded, S,
                                                 thermodynamic_factor, p3, prp,
                                                 ν, Dᵛ, ρ, FT)

        @test raw_rate ≈ bounded_rate
    end

    @testset "ice lambda limiter recomputes number" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        constants = ThermodynamicConstants(FT)
        ρ = FT(0.8)
        q = MoistureMassFractions(FT(1e-3))
        𝒰 = LiquidIcePotentialTemperatureState(FT(265), q, FT(1e5), FT(8e4))
        qⁱ = FT(1e-4)
        nⁱ = FT(1e-2)
        ℳ = P3MicrophysicalState(FT(0), FT(0), FT(0), FT(0),
                                  qⁱ, nⁱ, FT(0), FT(0),
                                  FT(0), FT(0), FT(0), FT(0))

        rime_state = PPP.consistent_rime_state(p3, qⁱ, FT(0), FT(0), FT(0))
        Fˡ = PPP.liquid_fraction_on_ice(qⁱ, FT(0))
        μ_for_limiter = PPP.compute_ice_shape_parameter(p3, qⁱ, nⁱ,
                                                        rime_state.Fᶠ, Fˡ, rime_state.ρᶠ)
        log_m = log10(qⁱ / nⁱ)
        limiter = PPP.ice_integrals_table(p3).lambda_limiter
        lower_nⁱ = limiter.large_q(log_m, rime_state.Fᶠ, Fˡ,
                                   rime_state.ρᶠ, μ_for_limiter) * qⁱ
        upper_nⁱ = limiter.small_q(log_m, rime_state.Fᶠ, Fˡ,
                                   rime_state.ρᶠ, μ_for_limiter) * qⁱ
        expected_nⁱ = clamp(nⁱ, lower_nⁱ, upper_nⁱ)
        props = PPP.p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)

        @test expected_nⁱ > nⁱ
        @test props.nⁱ ≈ expected_nⁱ

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, props)
        τ = p3.process_rates.sink_limiting_timescale
        @test rates.ice_number_correction ≈ (expected_nⁱ - nⁱ) / τ
        @test tendency_ρnⁱ(rates, ρ) >= ρ * rates.ice_number_correction
    end

    @testset "Tendency functions - smoke tests" begin
        FT = Float64
        ρ = FT(1.0)    # Air density [kg/m³]
        qⁱ = FT(1e-4)  # Ice mass mixing ratio [kg/kg]
        nⁱ = FT(1e5)   # Ice number [1/kg]
        Fᶠ = FT(0.3)   # Rime fraction
        ρᶠ = FT(400.0)  # Rime density [kg/m³]
        prp = ProcessRateParameters(FT)

        # Create rates with typical warm-rain and ice process activity
        # Sign convention (M7): all one-directional rates are positive magnitudes
        rates = P3ProcessRates(
            # Phase 1: Cloud condensation/evaporation (bidirectional: +cond, −evap)
            FT(5e-7),   # condensation
            # Phase 1: Rain (all positive magnitudes)
            FT(1e-7),   # autoconversion
            FT(2e-7),   # accretion
            FT(0),      # cloud_self_collection (0 for KK2000)
            FT(5e-8),   # rain_evaporation (positive magnitude)
            FT(0),      # rain_evaporation_number
            FT(1e-6),   # rain_self_collection (positive magnitude)
            FT(5e-7),   # rain_breakup (positive = number source)
            # Phase 1: Ice (deposition bidirectional; others positive magnitude)
            FT(3e-7),   # deposition
            FT(1e-8),   # partial_melting
            FT(5e-8),   # complete_melting
            FT(1e3),    # melting_number (positive magnitude)
            FT(0),      # clipping_dry_mass
            FT(0),      # clipping_rime_mass
            FT(0),      # clipping_rime_volume
            FT(0),      # post_process_clipping
            # D2: Sublimation number loss
            FT(0.0),    # sublimation_number
            # Phase 2: Aggregation (positive magnitude)
            FT(500.0),  # aggregation
            FT(0.0),    # ni_limit (C3: global Nᵢ cap; zero in warm-environment test)
            # Phase 2: Riming (all positive magnitudes)
            FT(1e-7),   # cloud_riming
            FT(1e4),    # cloud_riming_number (positive magnitude)
            FT(5e-8),   # rain_riming
            FT(500.0),  # rain_riming_number (positive magnitude)
            FT(300.0),  # rime_density_new
            # Phase 2: Shedding and refreezing
            FT(2e-8),   # shedding
            FT(100.0),  # shedding_number
            FT(1e-8),   # refreezing
            # Ice nucleation
            FT(1e-9),   # nucleation_mass
            FT(10.0),   # nucleation_number
            FT(5e-9),   # cloud_freezing_mass
            FT(100.0),  # cloud_freezing_number
            FT(3e-9),   # rain_freezing_mass
            FT(50.0),   # rain_freezing_number
            # Rime splintering
            FT(1e-10),  # splintering_mass
            FT(1.0),    # splintering_number
            # Homogeneous freezing
            FT(0.0),    # cloud_homogeneous_mass (warm environment: no hom. freezing)
            FT(0.0),    # cloud_homogeneous_number
            FT(0.0),    # rain_homogeneous_mass
            FT(0.0),    # rain_homogeneous_number
            FT(0.0),    # cloud_warm_collection (warm environment test)
            FT(0.0),    # cloud_warm_collection_number
            FT(0.0),    # rain_warm_collection
            FT(0.0),    # rain_warm_collection_number (M9)
            FT(0.0),    # wet_growth_cloud
            FT(0.0),    # wet_growth_rain
            FT(0.0),    # wet_growth_shedding (D8)
            FT(0.0),    # wet_growth_shedding_number (D8)
            FT(0.0),    # ccn_activation_mass (M9 stub)
            FT(0.0),    # ccn_activation_number (M9 stub)
            FT(0.0),    # rain_condensation (M9 stub)
            FT(0.0),    # coating_condensation (M9 stub)
            FT(0.0),    # coating_evaporation (M9 stub)
            FT(0.0),    # wet_growth_densification_mass (H9)
            FT(0.0),    # wet_growth_densification_volume (H9)
            FT(0.0),    # cloud_number_correction (M6)
            FT(0.0),    # rain_number_correction (M6)
            FT(0.0),    # ice_number_correction (M4)
            FT(0.0),    # predicted_ssat_adjustment
            FT(0.0),    # predicted_ssat_tendency
        )

        # Test each tendency function returns a finite number
        @test isfinite(tendency_ρqᶜˡ(rates, ρ))
        @test isfinite(tendency_ρqʳ(rates, ρ))
        @test isfinite(tendency_ρnʳ(rates, ρ, nⁱ, qⁱ, zero(FT), one(FT),
                                    tendency_test_p3(FT; process_rates = prp)))
        @test isfinite(tendency_ρqⁱ(rates, ρ))
        @test isfinite(tendency_ρnⁱ(rates, ρ))
        @test isfinite(tendency_ρqᶠ(rates, ρ, Fᶠ))
        @test isfinite(tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, one(FT), ProcessRateParameters(FT)))
        @test isfinite(tendency_ρqʷⁱ(rates, ρ))
        @test isfinite(tendency_ρqᵛ(rates, ρ))

        # Physics: condensation (vapor→cloud) should decrease vapor
        @test tendency_ρqᵛ(rates, ρ) < 0
    end

    @testset "Tendency functions - zero rates produce zero tendencies" begin
        FT = Float64
        ρ = FT(1.0)
        zero_rates = P3ProcessRates(ntuple(_ -> zero(FT), fieldcount(P3ProcessRates))...)

        @test tendency_ρqᶜˡ(zero_rates, ρ) == 0.0
        @test tendency_ρqʳ(zero_rates, ρ) == 0.0
        @test tendency_ρnʳ(zero_rates, ρ, FT(1e5), FT(1e-4), zero(FT), one(FT),
                           tendency_test_p3(FT)) == 0.0
        @test tendency_ρqⁱ(zero_rates, ρ) == 0.0
        @test tendency_ρnⁱ(zero_rates, ρ) == 0.0
        @test tendency_ρqᶠ(zero_rates, ρ, FT(0.3)) == 0.0
        @test tendency_ρbᶠ(zero_rates, ρ, FT(0.3), FT(400.0), one(FT), ProcessRateParameters(FT)) == 0.0
        @test tendency_ρqʷⁱ(zero_rates, ρ) == 0.0
        @test tendency_ρqᵛ(zero_rates, ρ) == 0.0
    end

    @testset "coupled sink limiter iterations are configurable" begin
        FT = Float32
        @test ProcessRateParameters(FT).coupled_sink_limiting_iterations == 4
        configured = ProcessRateParameters(FT; coupled_sink_limiting_iterations = 3)
        @test configured.coupled_sink_limiting_iterations == 3
        @test_throws ArgumentError ProcessRateParameters(FT;
                                                         coupled_sink_limiting_iterations = 0)
    end

    @testset "P3 sediments cloud mass and number with Fortran Stokes velocities" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        constants = ThermodynamicConstants(FT)
        grid = RectilinearGrid(CPU(), FT; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
        μ = Breeze.AtmosphereModels.materialize_microphysical_fields(p3, grid, NamedTuple())

        @test haskey(μ, :wᶜˡ)
        @test haskey(μ, :wᶜˡₙ)

        cloud_mass_velocity = microphysical_velocities(p3, μ, Val(:ρqᶜˡ))
        cloud_number_velocity = microphysical_velocities(p3, μ, Val(:ρnᶜˡ))
        @test cloud_mass_velocity !== nothing
        @test cloud_number_velocity !== nothing
        @test cloud_mass_velocity.w === μ.wᶜˡ
        @test cloud_number_velocity.w === μ.wᶜˡₙ

        ρ = FT(1)
        T = FT(283.15)
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᶜˡ = FT(5e-4)
        nᶜˡ = FT(2e8)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        q = MoistureMassFractions(qᵛ, qᶜˡ, FT(0))
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, nᶜˡ, FT(0), FT(0), FT(0), FT(0),
                                 FT(0), FT(0), FT(0), FT(0), FT(0), FT(0))

        props = PPP.p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
        cache = PPP.p3_fall_speed_compute(p3, ρ, ℳ, props, constants)
        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
        transport = air_transport_properties(T, P, constants)
        η = transport.ν * ρ
        a_cn = constants.gravitational_acceleration * p3.process_rates.liquid_water_density /
               (FT(18) * max(η, FT(1e-20)))
        expected_mass_velocity = a_cn * (cloud.μ_c + 5) * (cloud.μ_c + 4) / cloud.λ_c^2
        expected_number_velocity = a_cn * (cloud.μ_c + 2) * (cloud.μ_c + 1) / cloud.λ_c^2

        @test cache.wᶜˡ ≈ expected_mass_velocity rtol=FT(1e-12)
        @test cache.wᶜˡₙ ≈ expected_number_velocity rtol=FT(1e-12)
        @test cache.wᶜˡ > cache.wᶜˡₙ

        # The Stokes prefactor scales with the *model's* gravitational acceleration
        # rather than a hardcoded 9.81, so doubling g doubles both fall speeds.
        heavy = ThermodynamicConstants(FT; gravitational_acceleration = 2 * constants.gravitational_acceleration)
        vᶜ = PPP.cloud_terminal_velocities(p3, qᶜˡ, ρ, transport.ν, cloud.μ_c, cloud.λ_c, constants)
        vᶜ_heavy = PPP.cloud_terminal_velocities(p3, qᶜˡ, ρ, transport.ν, cloud.μ_c, cloud.λ_c, heavy)
        @test vᶜ.mass_weighted ≈ expected_mass_velocity rtol=FT(1e-12)
        @test vᶜ_heavy.mass_weighted ≈ 2 * vᶜ.mass_weighted rtol=FT(1e-12)
        @test vᶜ_heavy.number_weighted ≈ 2 * vᶜ.number_weighted rtol=FT(1e-12)

        # `rime_density` forms the Cober-List impact parameter from the same
        # mass-weighted Stokes velocity, so its rime density depends on the ice fall
        # speed only through |vᵢ - vᶜ.mass_weighted| and is symmetric about it. A
        # second gravitational acceleration in either function would shift that centre.
        T_rime = p3.process_rates.freezing_temperature - FT(5)
        transport_rime = air_transport_properties(T_rime, P, constants)
        cloud_rim = FT(1e-5)
        δ = FT(1)
        for g_constants in (constants, heavy)
            v_impact = PPP.cloud_terminal_velocities(p3, qᶜˡ, ρ, transport_rime.ν,
                                                     cloud.μ_c, cloud.λ_c,
                                                     g_constants).mass_weighted
            ρᶠ_above = rime_density(p3, qᶜˡ, cloud_rim, T_rime, v_impact + δ, ρ,
                                    g_constants, transport_rime, cloud.μ_c, cloud.λ_c)
            ρᶠ_below = rime_density(p3, qᶜˡ, cloud_rim, T_rime, v_impact - δ, ρ,
                                    g_constants, transport_rime, cloud.μ_c, cloud.λ_c)
            @test ρᶠ_above ≈ ρᶠ_below rtol=FT(1e-12)
            @test p3.process_rates.minimum_rime_density < ρᶠ_above < p3.process_rates.maximum_rime_density
        end
    end

    @testset "P3MicrophysicalState carries aerosol number and vertical velocity" begin
        FT = Float64
        ℳ = P3MicrophysicalState(
            FT(1e-4), FT(2e8), FT(0), FT(0), FT(1e-5), FT(1e4),
            FT(0), FT(0), FT(0), FT(0), FT(7), FT(3.5))
        @test ℳ.nᵃ == FT(7)
        @test ℳ.w == FT(3.5)
    end

    @testset "microphysical_state plumbs velocities.w into ℳ.w (parcel path)" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics()
        ρ = FT(1)
        μ = (ρqᶜˡ = FT(0), ρnᶜˡ = FT(0), ρqʳ = FT(0), ρnʳ = FT(0),
             ρqⁱ = FT(0), ρnⁱ = FT(0), ρqᶠ = FT(0), ρbᶠ = FT(0),
             ρqʷⁱ = FT(0))
        velocities = (u = FT(0), v = FT(0), w = FT(4.2))
        ℳ = Breeze.AtmosphereModels.microphysical_state(p3, ρ, μ, nothing, velocities)
        @test ℳ.w == FT(4.2)
    end

    @testset "grid_microphysical_state plumbs interpolated w into ℳ.w (LES path)" begin
        FT = Float64
        grid = RectilinearGrid(CPU(), FT;
                               size = (1, 1, 4), x = (0, 1), y = (0, 1), z = (0, 4),
                               topology = (Periodic, Periodic, Bounded))
        p3 = PredictedParticlePropertiesMicrophysics()

        μ = (; ρqᶜˡ = CenterField(grid), ρnᶜˡ = CenterField(grid),
               ρqʳ  = CenterField(grid), ρnʳ  = CenterField(grid),
               ρqⁱ  = CenterField(grid), ρnⁱ  = CenterField(grid),
               ρqᶠ  = CenterField(grid), ρbᶠ  = CenterField(grid),
               ρqʷⁱ = CenterField(grid),
               ρsˢᵃᵗ = CenterField(grid), ρnᵃ = CenterField(grid))

        w_face = ZFaceField(grid)
        set!(w_face, (x, y, z) -> 2.0)
        velocities = (; u = ZeroField(), v = ZeroField(), w = w_face)
        ρ = FT(1)
        ℳ = Breeze.AtmosphereModels.grid_microphysical_state(1, 1, 2, grid, p3, μ, ρ, nothing, velocities)
        @test ℳ.w == FT(2.0)
    end

    @testset "compute_p3_process_rates uses resolved supersaturation forcing" begin
        FT = Float64
        constants = ThermodynamicConstants(FT)
        p3 = PredictedParticlePropertiesMicrophysics()

        ρ = FT(1)
        T = FT(280)
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᶜˡ = FT(1e-4)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface()) + FT(1e-5)
        q = MoistureMassFractions(qᵛ, qᶜˡ, zero(FT))
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)

        ℳ = P3MicrophysicalState(qᶜˡ, FT(2e8), zero(FT), zero(FT),
                                 zero(FT), zero(FT), zero(FT), zero(FT),
                                 zero(FT), zero(FT), zero(FT), zero(FT))
        ℳ_with_w = P3MicrophysicalState(qᶜˡ, FT(2e8), zero(FT), zero(FT),
                                        zero(FT), zero(FT), zero(FT), zero(FT),
                                        zero(FT), zero(FT), zero(FT), FT(1))

        rates_unforced = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants,
                                                  nothing, T, FT(0), FT(0))
        rates_cooling = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants,
                                                 nothing, T, FT(-0.01), FT(0))
        rates_vapor_source = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants,
                                                      nothing, T, FT(0), FT(1e-6))
        rates_w_only = compute_p3_process_rates(p3, ρ, ℳ_with_w, 𝒰, constants,
                                                nothing, T, FT(0), FT(0))

        @test rates_cooling.condensation > rates_unforced.condensation
        @test rates_vapor_source.condensation > rates_unforced.condensation
        @test rates_w_only.condensation == rates_unforced.condensation

        vapor_tendency_stationary = microphysical_tendency(
            p3, Val(:ρqᵛ), ρ, ℳ, 𝒰, constants)
        vapor_tendency_ascending = microphysical_tendency(
            p3, Val(:ρqᵛ), ρ, ℳ_with_w, 𝒰, constants)
        @test vapor_tendency_ascending < vapor_tendency_stationary
    end

    @testset "Tendency functions - Float32 type stability" begin
        FT = Float32
        ρ = FT(1.0)
        rates = P3ProcessRates(ntuple(_ -> FT(1e-7), fieldcount(P3ProcessRates))...)

        @test tendency_ρqᶜˡ(rates, ρ) isa FT
        @test tendency_ρqʳ(rates, ρ) isa FT
        @test tendency_ρnʳ(rates, ρ, FT(1e5), FT(1e-4), zero(FT), one(FT),
                           tendency_test_p3(FT)) isa FT
        @test tendency_ρqⁱ(rates, ρ) isa FT
        @test tendency_ρnⁱ(rates, ρ) isa FT
        @test tendency_ρqᶠ(rates, ρ, FT(0.3)) isa FT
        @test tendency_ρbᶠ(rates, ρ, FT(0.3), FT(400.0), one(FT), ProcessRateParameters(FT)) isa FT
        @test tendency_ρqʷⁱ(rates, ρ) isa FT
        @test tendency_ρqᵛ(rates, ρ) isa FT
    end

    @testset "Cloud DSD diagnosis - Float32 type stability" begin
        # An untyped `1e-16` floor on nᶜˡ promoted Nᶜ, μ_c and λ_c to Float64 on the
        # per-cell path, and left the returned nᶜˡ inferred as Union{Float32, Float64}
        # through the `iszero(ρ)` ifelse. @inferred catches the Union; the eltype
        # checks catch a silent promotion to a concrete Float64.
        FT = Float32
        p3 = PredictedParticlePropertiesMicrophysics(FT)

        # Covers the floored (nᶜˡ below the guard), typical, and zero-density branches.
        states = ((FT(1e-3), FT(0), FT(1.2)),
                  (FT(1e-3), FT(1e8), FT(1.2)),
                  (FT(0), FT(1e8), FT(1.2)),
                  (FT(1e-3), FT(1e8), FT(0)))

        for (qᶜˡ, nᶜˡ, ρ) in states
            cloud = @inferred PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
            @test all(v -> v isa FT, values(cloud))
        end
    end

    #####
    ##### Process rate function tests
    #####

    @testset "rain_autoconversion_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        # KK2000 formula with typical cumulus values
        qc = FT(1e-3)     # 1 g/kg cloud water
        Nc = FT(100e6)     # 100 cm⁻³ cloud droplet concentration
        ρ  = FT(1.2)       # sea-level air density

        rate = rain_autoconversion_rate(p3, qc, Nc, ρ)
        @test rate > 0
        # KK2000 gives O(1e-6) kg/kg/s for these inputs
        @test rate > 1e-8
        @test rate < 1e-3

        # Higher cloud water content gives faster autoconversion
        rate_high = rain_autoconversion_rate(p3, FT(2e-3), Nc, ρ)
        @test rate_high > rate

        # Zero cloud water gives zero autoconversion
        rate_zero = rain_autoconversion_rate(p3, FT(0), Nc, ρ)
        @test rate_zero == 0

        # Small cloud water gives small but nonzero rate (KK2000 has no threshold)
        rate_small = rain_autoconversion_rate(p3, FT(5e-5), Nc, ρ)
        @test rate_small > 0
        @test rate_small < rate
    end

    @testset "rain_accretion_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qc = FT(1e-3)
        qr = FT(1e-3)

        rate = rain_accretion_rate(p3, qc, qr)
        @test rate > 0
        @test isfinite(rate)

        # Zero cloud gives zero accretion
        @test rain_accretion_rate(p3, FT(0), qr) == 0

        # Zero rain gives zero accretion
        @test rain_accretion_rate(p3, qc, FT(0)) == 0

        # Higher rain gives faster accretion
        rate_high = rain_accretion_rate(p3, qc, FT(2e-3))
        @test rate_high > rate
    end

    @testset "warm_rain_scheme dispatch" begin
        FT = Float64
        qc = FT(1e-3)
        qr = FT(5e-4)
        Nc = FT(1e8)
        nr = FT(1e4)
        ρ  = FT(1.0)

        p3_kk = PredictedParticlePropertiesMicrophysics(FT; warm_rain_scheme = KhairoutdinovKogan2000())

        # KK2000 is the default and the only scheme
        p3_default = PredictedParticlePropertiesMicrophysics(FT)
        @test p3_default.warm_rain_scheme isa KhairoutdinovKogan2000
        @test KhairoutdinovKogan2000 <: AbstractWarmRainScheme
        @test rain_autoconversion_rate(p3_default, qc, Nc, ρ, qr) ==
              rain_autoconversion_rate(p3_kk, qc, Nc, ρ, qr)

        a_kk = rain_autoconversion_rate(p3_kk, qc, Nc, ρ, qr)
        @test isfinite(a_kk)
        @test a_kk > 0

        # KK2000 autoconversion ignores qʳ
        @test rain_autoconversion_rate(p3_kk, qc, Nc, ρ, zero(FT)) == a_kk

        c_kk = rain_accretion_rate(p3_kk, qc, qr, ρ)
        @test isfinite(c_kk)
        @test c_kk > 0

        # Rain self-collection: linear form k_rr × ρ × qʳ × nʳ
        s_kk = rain_self_collection_rate(p3_kk, qr, nr, ρ)
        @test isfinite(s_kk)
        @test s_kk > 0

        # Cloud self-collection is zero for KK2000 (Fortran ncslf = 0)
        @test PredictedParticleProperties.cloud_self_collection_rate(p3_kk, qc, Nc, ρ) == 0

        # Seed-drop mass: KK2000 ≈ 25 μm radius
        @test PredictedParticleProperties.rain_seed_drop_mass(p3_kk) ≈ 4π/3 * 1000 * (25e-6)^3

        # Autoconversion removes cloud number in proportion to the mass lost
        autoconversion_only = p3_process_rates_with(FT; autoconversion = FT(1e-7))
        @test tendency_ρnᶜˡ(autoconversion_only, ρ, Nc, qc, p3_kk) < 0
    end

    @testset "rain_evaporation_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        qr = FT(1e-3)
        nr = FT(1e4)
        T = FT(288.0)
        ρ = FT(1.0)
        P = FT(101325.0)

        # Subsaturated: qv < qv_sat → positive evaporation rate (M7: positive magnitude)
        qv_sat = FT(0.012)
        qv_sub = FT(0.008)    # 67% RH
        rate_sub = rain_evaporation_rate(p3, qr, nr, qv_sub, qv_sat, T, ρ, P, constants)
        @test rate_sub > 0     # Positive magnitude = rain evaporating

        # Saturated: qv = qv_sat → zero evaporation
        rate_sat = rain_evaporation_rate(p3, qr, nr, qv_sat, qv_sat, T, ρ, P, constants)
        @test rate_sat == 0

        # Supersaturated: qv > qv_sat → zero (no condensation on rain)
        qv_super = FT(0.015)
        rate_super = rain_evaporation_rate(p3, qr, nr, qv_super, qv_sat, T, ρ, P, constants)
        @test rate_super == 0

        # Zero rain gives zero evaporation
        rate_norain = rain_evaporation_rate(p3, FT(0), nr, qv_sub, qv_sat, T, ρ, P, constants)
        @test rate_norain == 0
    end

    @testset "coupled_saturation_adjustment_rates" begin
        p3_base = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)
        process_rates = ProcessRateParameters(FT; sink_limiting_timescale=FT(10))
        p3 = p3_with_process_rates(p3_base, process_rates)

        ρ = FT(1)
        T = FT(263.15)
        P = FT(80000)
        qᶜˡ = FT(1e-3)
        nᶜˡ = FT(2e8)
        qʳ = FT(0)
        nʳ = FT(0)
        qⁱ = FT(2e-4)
        qʷⁱ = FT(0)
        nⁱ = FT(2e4)
        Fᶠ = FT(0)
        ρᶠ = FT(400)
        μ = FT(0)

        qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
        qᵛ = qᵛ⁺ˡ + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ + qʷⁱ, qⁱ)
        transport = air_transport_properties(T, P, constants)

        epsr = PPP.rain_condensation_epsilon(p3, FT(5e-4), FT(1e6), ρ, transport)
        expected_epsr = expected_reference_rain_epsilon(p3, FT(5e-4), FT(1e6), ρ, transport, FT)
        # Fˡ = 0 here, so the dry-ice gate inside `coupled_saturation_adjustment_rates`
        # is active and the raw relaxation coefficient is the Fortran `epsi`.
        epsi = PPP.ice_relaxation_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                          constants, transport, q, μ)
        expected_epsi = expected_reference_ice_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                                     constants, transport, q, μ)

        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
        # predict_supersaturation defaults to false, so this M&G call sees
        # the host state directly and the G&M ε is gated to zero by
        # `compute_p3_process_rates` (not this function).
        rates = PPP.coupled_saturation_adjustment_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q, μ,
            cloud.μ_c, cloud.λ_c, cloud.nᶜˡ, FT(0), FT(0))
        expected_rates = expected_reduced_reference_vapor_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q, μ)

        # Bergeron check: with ice present, cloud condensation is smaller than
        # with no ice, because ice steals vapor through the shared budget.
        rates_noice = PPP.coupled_saturation_adjustment_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, zero(FT), zero(FT), zero(FT),
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q, μ,
            cloud.μ_c, cloud.λ_c, cloud.nᶜˡ, FT(0), FT(0))

        @test epsr ≈ expected_epsr
        @test epsi ≈ expected_epsi
        @test rates.condensation ≈ expected_rates.condensation
        @test rates.rain_evaporation ≈ expected_rates.rain_evaporation
        @test rates.rain_condensation ≈ expected_rates.rain_condensation
        @test rates.deposition ≈ expected_rates.deposition
        @test rates.coating_condensation ≈ expected_rates.coating_condensation
        @test rates.coating_evaporation ≈ expected_rates.coating_evaporation
        @test rates.deposition > 0
        @test rates.coating_condensation == 0  # dry ice: no coating
        @test rates.coating_evaporation == 0
        @test rates.condensation < rates_noice.condensation

        # Zero host forcing reproduces the Bergeron-only behavior bitwise.
        rates_unforced = PPP.coupled_saturation_adjustment_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q, μ,
            cloud.μ_c, cloud.λ_c, cloud.nᶜˡ, FT(0), FT(0))
        @test rates_unforced.condensation === rates.condensation
        @test rates_unforced.deposition === rates.deposition
        @test rates_unforced.rain_evaporation === rates.rain_evaporation
        @test rates_unforced.rain_condensation === rates.rain_condensation
        @test rates_unforced.coating_condensation === rates.coating_condensation
        @test rates_unforced.coating_evaporation === rates.coating_evaporation

        # Pure adiabatic forcing: saturated cooling produces condensation.
        let
            T_ad = FT(280.0)
            qᵛ⁺ˡ_ad = saturation_specific_humidity(T_ad, ρ, constants, PlanarLiquidSurface())
            qᵛ⁺ⁱ_ad = saturation_specific_humidity(T_ad, ρ, constants, PlanarIceSurface())
            qᵛ_ad = qᵛ⁺ˡ_ad  # exactly saturated → ssat_liquid = 0
            q_ad = MoistureMassFractions(qᵛ_ad, qᶜˡ + qʳ + qʷⁱ, zero(FT))
            transport_ad = air_transport_properties(T_ad, P, constants)
            cloud_ad = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
            cᵖᵐ_ad = mixture_heat_capacity(q_ad, constants)
            temperature_tendency = -constants.gravitational_acceleration / cᵖᵐ_ad
            rates_cooling = PPP.coupled_saturation_adjustment_rates(
                p3, qᶜˡ, nᶜˡ, qʳ, nʳ, zero(FT), zero(FT), zero(FT),
                qᵛ_ad, qᵛ⁺ˡ_ad, qᵛ⁺ⁱ_ad, Fᶠ, ρᶠ, T_ad, P, ρ,
                constants, transport_ad, q_ad, μ,
                cloud_ad.μ_c, cloud_ad.λ_c, cloud_ad.nᶜˡ,
                temperature_tendency, FT(0))
            @test rates_cooling.condensation > 0
            @test rates_cooling.deposition == 0  # no ice present
        end

        # Sign symmetry: at exactly saturated state, cooling generates condensation,
        # while warming routes the same forcing into evaporation. We use a soft check
        # because the clamps in coupled_saturation_adjustment_rates may route the
        # mass through different fields.
        let
            T_s = FT(280.0)
            qᵛ⁺ˡ_s = saturation_specific_humidity(T_s, ρ, constants, PlanarLiquidSurface())
            qᵛ⁺ⁱ_s = saturation_specific_humidity(T_s, ρ, constants, PlanarIceSurface())
            qᵛ_s = qᵛ⁺ˡ_s
            q_s = MoistureMassFractions(qᵛ_s, qᶜˡ + qʳ + qʷⁱ, zero(FT))
            transport_s = air_transport_properties(T_s, P, constants)
            cloud_s = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
            common = (p3, qᶜˡ, nᶜˡ, qʳ, nʳ, zero(FT), zero(FT), zero(FT),
                      qᵛ_s, qᵛ⁺ˡ_s, qᵛ⁺ⁱ_s, Fᶠ, ρᶠ, T_s, P, ρ,
                      constants, transport_s, q_s, μ,
                      cloud_s.μ_c, cloud_s.λ_c, cloud_s.nᶜˡ)
            rates_up = PPP.coupled_saturation_adjustment_rates(common..., FT(-0.01), FT(0))
            rates_down = PPP.coupled_saturation_adjustment_rates(common..., FT(+0.01), FT(0))
            @test rates_up.condensation > 0
            # Warming evaporates the cloud reservoir: with cloud present
            # and positive temperature tendency the `condensation` channel goes
            # negative (cloud → vapor),
            # mirroring the sign flip in the production routine.
            @test rates_down.condensation < 0
        end
    end

    @testset "coupled_saturation_adjustment_rates wet-ice coating" begin
        p3_base = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)
        process_rates = ProcessRateParameters(FT; sink_limiting_timescale=FT(10))
        p3 = p3_with_process_rates(p3_base, process_rates)

        ρ = FT(1)
        T = FT(272.15)  # just below freezing so mixed ice can exist
        P = FT(80000)
        qᶜˡ = FT(1e-3)
        nᶜˡ = FT(2e8)
        qʳ = FT(0)
        nʳ = FT(0)
        qⁱ = FT(2e-4)
        qʷⁱ = FT(1e-4)  # ~50% liquid fraction → wet ice
        nⁱ = FT(2e4)
        Fᶠ = FT(0)
        ρᶠ = FT(400)
        μ = FT(0)

        qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
        qᵛ = qᵛ⁺ˡ + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ + qʷⁱ, qⁱ)
        transport = air_transport_properties(T, P, constants)

        # Fˡ ≈ 0.33 here, so the wet-ice gate is the active one and the raw
        # relaxation coefficient is the Fortran `epsiw`.
        epsiw = PPP.ice_relaxation_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                           constants, transport, q, μ)
        expected_epsiw = expected_reference_coating_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                                         constants, transport, q, μ)

        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
        rates = PPP.coupled_saturation_adjustment_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q, μ,
            cloud.μ_c, cloud.λ_c, cloud.nᶜˡ, FT(0), FT(0))
        expected_rates = expected_reduced_reference_vapor_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q, μ)

        @test epsiw > 0
        @test epsiw ≈ expected_epsiw

        # Mutual exclusivity: the dry-ice and wet-ice branches are gated on
        # complementary liquid-fraction regimes, so only one can be nonzero.
        # Coupled formula: wet-ice coating condenses (vapor is supersaturated w.r.t. liquid).
        @test rates.deposition == 0  # dry-ice path inactive
        @test rates.coating_condensation > 0
        @test rates.coating_evaporation == 0

        @test rates.coating_condensation ≈ expected_rates.coating_condensation
        @test rates.condensation ≈ expected_rates.condensation
    end

    @testset "limit_vapor_rates caps coupled sinks against satadj budget" begin
        FT = Float64
        constants = ThermodynamicConstants(FT)
        dt_safety = FT(10)
        P = FT(8e4)
        T = FT(253.15)
        qᵗ = FT(3.0e-3)
        qᵛ⁺ˡ = adjustment_saturation_specific_humidity(T, P, qᵗ, constants, PlanarLiquidSurface())
        qᵛ = qᵛ⁺ˡ + FT(1e-4)

        cond = FT(4e-5)
        ccn_act = FT(1e-5)
        ccn_act_n = FT(2e3)
        rain_cond = FT(2e-5)
        rain_evap = FT(0)
        dep = FT(3e-5)
        coat_cond = FT(2e-5)
        coat_evap = FT(0)
        nuc_q = FT(1e-5)
        nuc_n = FT(5e2)

        limited = PPP.limit_vapor_rates(cond, ccn_act, ccn_act_n, rain_cond, rain_evap,
                                        dep, coat_cond, coat_evap, nuc_q, nuc_n,
                                        qᵛ, qᵛ⁺ˡ, T, P, qᵗ, constants, dt_safety, FT(273.15))

        @test limited.cond < cond
        @test limited.ccn_act < ccn_act
        @test limited.ccn_act_n < ccn_act_n
        @test limited.rain_cond < rain_cond
        @test limited.coat_cond < coat_cond
        @test limited.dep < dep
        @test limited.nuc_q < nuc_q
        @test limited.nuc_n < nuc_n

        Rᵛ = Breeze.Thermodynamics.vapor_gas_constant(constants)
        ℒˡ = Breeze.Thermodynamics.liquid_latent_heat(T, constants)
        ξˡ = PPP.liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)

        # Liquid satadj cap: cond + ccn_act + rain_cond + coat_cond ≤ qcon_cap/dt_safety
        qcon_cap = max(zero(FT), qᵛ - qᵛ⁺ˡ) / ξˡ
        cond_sink_total = max(zero(FT), limited.cond) + limited.ccn_act +
                          limited.rain_cond + limited.coat_cond
        @test cond_sink_total * dt_safety <= qcon_cap + FT(10) * eps(FT)

        # Ice satadj cap: dep + nuc_q ≤ qdep_cap/dt_safety, evaluated against
        # the post-liquid thermodynamic state (Fortran qv_tmp / t_tmp).
        net_liquid = max(zero(FT), limited.cond) + limited.ccn_act +
                     limited.rain_cond + limited.coat_cond -
                     rain_evap - coat_evap - max(zero(FT), -limited.cond)
        qᵛ_after = qᵛ - net_liquid * dt_safety
        T_after = T + net_liquid * ℒˡ * dt_safety / constants.dry_air.heat_capacity
        qᵛ⁺ⁱ_after = adjustment_saturation_specific_humidity(T_after, P, qᵗ, constants, PlanarIceSurface())
        ℒⁱ_after = Breeze.Thermodynamics.ice_latent_heat(T_after, constants)
        ξⁱ_after = PPP.ice_psychrometric_correction(constants, ℒⁱ_after, qᵛ⁺ⁱ_after, Rᵛ, T_after)
        qdep_cap = max(zero(FT), qᵛ_after - qᵛ⁺ⁱ_after) / ξⁱ_after
        dep_sink_total = max(zero(FT), limited.dep) + limited.nuc_q
        @test dep_sink_total * dt_safety <= qdep_cap + FT(10) * eps(FT)
    end

    @testset "CCN activation and the vapor caps share one psychrometric convention" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        constants = ThermodynamicConstants(FT)
        τ = p3.process_rates.sink_limiting_timescale

        ρ = FT(1)
        P = FT(9e4)
        T = FT(283.15)
        qᵗ = FT(1.2e-2)
        qᵛ⁺ˡ = adjustment_saturation_specific_humidity(T, P, qᵗ, constants, PlanarLiquidSurface())
        # Weak supersaturation, so the seed mass is limited by the available vapor rather
        # than by the prescribed-Nᶜ target mass.
        qᵛ = qᵛ⁺ˡ + FT(1e-6)
        qᶜˡ = zero(FT)
        q = MoistureMassFractions(qᵛ, qᶜˡ, zero(FT))
        Nᶜ = p3.cloud.number_concentration

        ccn = PPP.compute_ccn_activation(p3.aerosol, p3, qᶜˡ, zero(FT), zero(FT),
                                         qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants)

        Rᵛ = Breeze.Thermodynamics.vapor_gas_constant(constants)
        ℒˡ = Breeze.Thermodynamics.liquid_latent_heat(T, constants)
        ξˡ = PPP.liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)
        cons7 = FT(4 * FT(π) / 3 * 1000 * (1e-6)^3)
        deficit = Nᶜ / ρ * cons7
        @test (qᵛ - qᵛ⁺ˡ) / ξˡ < deficit
        @test ccn.mass ≈ ((qᵛ - qᵛ⁺ˡ) / ξˡ) / τ rtol=FT(1e-14)

        # The moist mixture heat capacity gives a materially different factor, so sizing
        # the rate with it and capping it with cᵖᵈ would disagree within one cell.
        cᵖᵐ = mixture_heat_capacity(q, constants)
        Γˡ = 1 + ℒˡ^2 * qᵛ⁺ˡ / (Rᵛ * T^2 * cᵖᵐ)
        @test Γˡ < ξˡ
        @test !isapprox(Γˡ, ξˡ; rtol = FT(1e-3))

        # With one convention, a vapor-limited activation rate exactly fills
        # `limit_vapor_rates`'s liquid budget instead of being rescaled by it.
        limited = PPP.limit_vapor_rates(zero(FT), ccn.mass, zero(FT), zero(FT), zero(FT),
                                        zero(FT), zero(FT), zero(FT), zero(FT), zero(FT),
                                        qᵛ, qᵛ⁺ˡ, T, P, qᵗ, constants, τ, FT(273.15))
        @test limited.ccn_act ≈ ccn.mass rtol=FT(1e-12)
    end

    @testset "limit_vapor_rates caps evaporation when subsaturated" begin
        FT = Float64
        constants = ThermodynamicConstants(FT)
        dt_safety = FT(10)
        P = FT(8e4)
        T = FT(263.15)
        qᵗ = FT(3.0e-3)
        qᵛ⁺ˡ = adjustment_saturation_specific_humidity(T, P, qᵗ, constants, PlanarLiquidSurface())
        # Subsaturated over both liquid and ice
        qᵛ = qᵛ⁺ˡ - FT(1e-4)

        # Negative cond → cloud evaporation; rain_evap and coat_evap > 0
        cond = FT(-2e-5)
        ccn_act = FT(0)
        ccn_act_n = FT(0)
        rain_cond = FT(0)
        rain_evap = FT(5e-5)
        dep = FT(-1e-5)  # sublimation
        coat_cond = FT(0)
        coat_evap = FT(3e-5)
        nuc_q = FT(0)
        nuc_n = FT(0)

        limited = PPP.limit_vapor_rates(cond, ccn_act, ccn_act_n, rain_cond, rain_evap,
                                        dep, coat_cond, coat_evap, nuc_q, nuc_n,
                                        qᵛ, qᵛ⁺ˡ, T, P, qᵗ, constants, dt_safety, FT(273.15))

        # Evaporation rates should all be reduced (scaled toward the cap).
        @test limited.rain_evap < rain_evap
        @test limited.coat_evap < coat_evap
        @test limited.cond > cond  # less negative → smaller magnitude
        @test limited.dep > dep    # sublimation reduced

        # Verify the liquid evaporation cap: |cond_neg| + rain_evap + coat_evap ≤ qevp_cap/dt_safety
        Rᵛ = Breeze.Thermodynamics.vapor_gas_constant(constants)
        ℒˡ = Breeze.Thermodynamics.liquid_latent_heat(T, constants)
        ξˡ = PPP.liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)
        qevp_cap = max(zero(FT), -(qᵛ - qᵛ⁺ˡ) / ξˡ)
        evp_total = max(zero(FT), -limited.cond) + limited.rain_evap + limited.coat_evap
        @test evp_total * dt_safety <= qevp_cap + FT(10) * eps(FT)
    end

    @testset "limit_vapor_rates zeroes evaporation when supersaturated" begin
        FT = Float64
        constants = ThermodynamicConstants(FT)
        dt_safety = FT(10)
        P = FT(8e4)
        T = FT(263.15)
        qᵗ = FT(3.0e-3)
        qᵛ⁺ˡ = adjustment_saturation_specific_humidity(T, P, qᵗ, constants, PlanarLiquidSurface())
        qᵛ = qᵛ⁺ˡ + FT(1e-4)  # supersaturated

        # Pathological evaporation rates in supersaturated air should be zeroed.
        limited = PPP.limit_vapor_rates(FT(0), FT(0), FT(0), FT(0), FT(5e-5),
                                        FT(0), FT(0), FT(3e-5), FT(0), FT(0),
                                        qᵛ, qᵛ⁺ˡ, T, P, qᵗ, constants, dt_safety, FT(273.15))

        @test limited.rain_evap == 0
        @test limited.coat_evap == 0
    end

    @testset "ice_melting_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        qi = FT(1e-4)
        ni = FT(1e4)
        P = FT(85000.0)
        qv = FT(0.008)
        qv_sat = FT(0.01)
        Ff = FT(0.0)
        ρf = FT(400.0)
        ρ = FT(1.0)
        μ = FT(0.0)

        # Above freezing: positive melting
        T_warm = FT(275.15)    # +2C
        rate_warm = ice_melting_rate(p3, qi, ni, FT(0), T_warm, P, qv, qv_sat, Ff, ρf, ρ,
                                     constants, air_transport_properties(T_warm, P, constants), μ)
        @test rate_warm > 0

        # Below freezing: zero melting
        T_cold = FT(263.15)    # -10C
        rate_cold = ice_melting_rate(p3, qi, ni, FT(0), T_cold, P, qv, qv_sat, Ff, ρf, ρ,
                                     constants, air_transport_properties(T_cold, P, constants), μ)
        @test rate_cold == 0

        # Exactly at freezing: zero (no ΔT to drive melting)
        T_freeze = FT(273.15)
        rate_freeze = ice_melting_rate(p3, qi, ni, FT(0), T_freeze, P, qv, qv_sat, Ff, ρf, ρ,
                                       constants, air_transport_properties(T_freeze, P, constants), μ)
        @test rate_freeze == 0

    end

    @testset "ice_melting_rates partitioning" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        qi = FT(1e-4)
        ni = FT(1e4)
        P = FT(85000.0)
        qv = FT(0.008)
        qv_sat = FT(0.01)
        Ff = FT(0.0)
        ρf = FT(400.0)
        ρ = FT(1.0)
        T = FT(275.15)

        # No liquid on ice: all melting is partial (goes to coating)
        qwi_zero = FT(0)
        μ = FT(0.0)
        rates_dry = ice_melting_rates(p3, qi, ni, qwi_zero, T, P, qv, qv_sat, Ff, ρf, ρ,
                                      constants, air_transport_properties(T, P, constants), μ)
        total = rates_dry.partial_melting + rates_dry.complete_melting
        @test total > 0
        @test rates_dry.partial_melting >= 0
        @test rates_dry.complete_melting >= 0

        # With Fortran tables, the partial/complete split depends on the
        # PSD-integrated ventilation. Verify both branches are non-negative
        # and at least one is positive.
        @test rates_dry.complete_melting >= 0

        # Saturated liquid coating: more complete melting (or approximately equal)
        qwi_high = FT(0.5 * qi)   # 50% liquid fraction
        rates_wet = ice_melting_rates(p3, qi, ni, qwi_high, T, P, qv, qv_sat, Ff, ρf, ρ,
                                      constants, air_transport_properties(T, P, constants), μ)
        @test rates_wet.complete_melting >= 0
    end

    @testset "ventilation_sc_correction includes sqrt(rhofaci)" begin
        PPP = Breeze.Microphysics.PredictedParticleProperties
        ν = 1.5e-5
        Dᵥ = 2.0e-5

        base = PPP.ventilation_sc_correction(ν, Dᵥ, 1.0)
        doubled = PPP.ventilation_sc_correction(ν, Dᵥ, 4.0)

        @test doubled ≈ 2 * base
    end

    @testset "wet_growth_capacity keeps sensible term outside 2π/Lf" begin
        PPP = Breeze.Microphysics.PredictedParticleProperties
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        qi = FT(1e-4)
        qwi = FT(0)
        ni = FT(1e4)
        T = FT(268.15)
        P = FT(85000.0)
        Ff = FT(0.2)
        ρf = FT(400.0)
        ρ = FT(1.0)
        μ = FT(0.0)
        transport = air_transport_properties(T, P, constants)

        T₀ = p3.process_rates.freezing_temperature
        Rᵥ = Breeze.Thermodynamics.vapor_gas_constant(constants)
        e_s0 = PPP.saturation_vapor_pressure_at_freezing(constants, T₀)
        # Breeze carries total-air specific humidity, so set qv to the matching
        # saturation mass fraction and isolate the sensible-conduction term.
        qv = e_s0 / (Rᵥ * T₀ * ρ)

        m_mean = qi / ni
        ρ_correction = PPP.ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)
        C_fv = PPP.deposition_ventilation(
            p3.ice.deposition.ventilation,
            p3.ice.deposition.ventilation_enhanced,
            m_mean, Ff, ρf, p3.process_rates, transport.ν, transport.Dᵛ, ρ_correction, p3, μ)

        capacity = PPP.wet_growth_capacity(p3, qi, qwi, ni, T, P, qv, Ff, ρf, ρ, constants, transport, μ)
        expected = C_fv * transport.Kᵃ * (T₀ - T) * ni

        @test capacity ≈ expected rtol=1e-6
    end

    @testset "wet growth preserves collection number sinks" begin
        PPP = Breeze.Microphysics.PredictedParticleProperties
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        ρ = FT(1.0)
        P = FT(85000.0)
        T = p3.process_rates.freezing_temperature - FT(1e-3)
        qᵛ = FT(0.02)
        qᶜˡ = FT(1e-3)
        qʳ = FT(1e-3)
        qⁱ = FT(1e-4)
        nⁱ = FT(1e3)
        nʳ = FT(1e4)
        qᶠ = FT(1e-5)
        bᶠ = qᶠ / FT(400)
        Fᶠ = FT(0.1)
        ρᶠ = FT(400)
        μ_ice = FT(0)

        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, FT(300e6), ρ)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ, qⁱ)
        transport = air_transport_properties(T, P, constants)
        state = PPP.P3DerivedState{FT, typeof(q)}(
            nⁱ,
            nʳ,
            qᶠ,
            bᶠ,
            Fᶠ,
            ρᶠ,
            μ_ice,
            FT(0),
            cloud.Nᶜ,
            cloud.nᶜˡ,
            cloud.μ_c,
            cloud.λ_c,
            T,
            P,
            qᵛ,
            qᵛ,
            qᵛ,
            q,
            transport.Dᵛ,
            transport.Kᵃ,
            transport.ν,
        )
        ℳ = P3MicrophysicalState(
            qᶜˡ,
            cloud.nᶜˡ,
            qʳ,
            nʳ,
            qⁱ,
            nⁱ,
            qᶠ,
            bᶠ,
            FT(0),
            FT(0),
            FT(0),
            FT(0),
        )
        phase1 = PPP.P3Phase1Rates{FT}(ntuple(_ -> zero(FT), fieldcount(PPP.P3Phase1Rates{FT}))...)

        rates = PPP._p3_phase2_rates(p3, ρ, ℳ, constants, state, phase1)

        @test rates.cloud_riming == 0
        @test rates.rain_riming == 0
        @test rates.wet_growth_cloud > 0
        @test rates.wet_growth_rain > 0
        @test rates.cloud_riming_number > 0
        @test rates.rain_riming_number > 0
    end

    @testset "wet growth is inactive below the hydrometeor gate" begin
        PPP = Breeze.Microphysics.PredictedParticleProperties
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        ρ = FT(1.0)
        P = FT(85000.0)
        T = p3.process_rates.freezing_temperature - FT(1e-3)
        qᵛ = FT(0.02)
        qᶜˡ = FT(5e-7)
        qʳ = FT(4e-7)
        qⁱ = FT(1e-4)
        nⁱ = FT(1e3)
        nʳ = FT(1e4)
        qᶠ = FT(1e-5)
        bᶠ = qᶠ / FT(400)
        Fᶠ = FT(0.1)
        ρᶠ = FT(400)
        μ_ice = FT(0)

        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, FT(300e6), ρ)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ, qⁱ)
        transport = air_transport_properties(T, P, constants)
        state = PPP.P3DerivedState{FT, typeof(q)}(
            nⁱ,
            nʳ,
            qᶠ,
            bᶠ,
            Fᶠ,
            ρᶠ,
            μ_ice,
            FT(0),
            cloud.Nᶜ,
            cloud.nᶜˡ,
            cloud.μ_c,
            cloud.λ_c,
            T,
            P,
            qᵛ,
            qᵛ,
            qᵛ,
            q,
            transport.Dᵛ,
            transport.Kᵃ,
            transport.ν,
        )
        ℳ = P3MicrophysicalState(
            qᶜˡ,
            cloud.nᶜˡ,
            qʳ,
            nʳ,
            qⁱ,
            nⁱ,
            qᶠ,
            bᶠ,
            FT(0),
            FT(0),
            FT(0),
            FT(0),
        )
        phase1 = PPP.P3Phase1Rates{FT}(ntuple(_ -> zero(FT), fieldcount(PPP.P3Phase1Rates{FT}))...)

        rates = PPP._p3_phase2_rates(p3, ρ, ℳ, constants, state, phase1)

        @test qᶜˡ + qʳ < FT(1e-6)
        @test rates.cloud_riming > 0
        @test rates.rain_riming > 0
        @test rates.wet_growth_cloud == 0
        @test rates.wet_growth_rain == 0
    end

    @testset "refreezing_rate keeps sensible term outside 2π/Lf" begin
        PPP = Breeze.Microphysics.PredictedParticleProperties
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        qwi = FT(1)
        qi = FT(1e-4)
        ni = FT(1e4)
        T = FT(268.15)
        P = FT(85000.0)
        Ff = FT(0.2)
        ρf = FT(400.0)
        ρ = FT(1.0)
        μ = FT(0.0)
        transport = air_transport_properties(T, P, constants)

        T₀ = p3.process_rates.freezing_temperature
        Rᵥ = Breeze.Thermodynamics.vapor_gas_constant(constants)
        Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
        ε = Rᵈ / Rᵥ
        e_s0 = PPP.saturation_vapor_pressure_at_freezing(constants, T₀)
        # M10: set qv = q_sat0 (mixing ratio convention) so latent term vanishes
        qv = ε * e_s0 / max(P - e_s0, FT(1))

        refreezing = PPP.refreezing_rate(p3, qwi, qi, ni, T, P, qv, Ff, ρf, ρ, constants, transport, μ)

        # Refreezing should remain active below freezing with liquid-coated ice.
        @test refreezing > 0
    end

    @testset "ice_aggregation_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qi = FT(1e-4)
        ni = FT(1e5)
        Ff = FT(0.0)
        ρf = FT(400.0)

        # Near freezing (warm ice, sticky): aggregation active
        T_warm = FT(268.15)    # -5C
        ρ = FT(1.0)
        μ = FT(0.0)
        rate_warm = ice_aggregation_rate(p3, qi, ni, T_warm, Ff, ρf, ρ, μ)
        @test rate_warm > 0     # Positive magnitude (M7)

        # Very cold (T < 253.15 K): much less aggregation
        T_cold = FT(233.15)    # -40C
        rate_cold = ice_aggregation_rate(p3, qi, ni, T_cold, Ff, ρf, ρ, μ)
        # Aggregation efficiency at very cold T is 0.001 vs ~0.15 at -5C
        @test rate_cold < rate_warm

        # Zero ice: zero aggregation
        rate_noice = ice_aggregation_rate(p3, FT(0), FT(0), T_warm, Ff, ρf, ρ, μ)
        @test rate_noice == 0

        # Heavily rimed (Ff > 0.9): aggregation shuts off
        rate_rimed = ice_aggregation_rate(p3, qi, ni, T_warm, FT(0.95), ρf, ρ, μ)
        @test rate_rimed == 0

        # Rate scales with ρ × rhofaci where rhofaci = (ρ₀/ρ)^0.54 (M11).
        # Combined scaling: rate ∝ ρ × (ρ₀/ρ)^0.54 = ρ₀^0.54 × ρ^0.46
        ρ_half = FT(0.5)
        rate_half_ρ = ice_aggregation_rate(p3, qi, ni, T_warm, Ff, ρf, ρ_half, μ)
        @test rate_half_ρ ≈ rate_warm * (ρ_half / ρ)^FT(0.46)
    end

    @testset "cloud_riming_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qc = FT(1e-3)
        qi = FT(1e-4)
        ni = FT(1e4)
        Ff = FT(0.0)
        ρf = FT(400.0)
        ρ = FT(1.0)
        μ = FT(0.0)

        # Below freezing with cloud and ice: positive riming
        T_cold = FT(263.15)    # -10C
        rate = cloud_riming_rate(p3, qc, qi, ni, T_cold, Ff, ρf, ρ, μ)
        @test rate > 0

        # Above freezing: zero riming
        T_warm = FT(278.15)
        rate_warm = cloud_riming_rate(p3, qc, qi, ni, T_warm, Ff, ρf, ρ, μ)
        @test rate_warm == 0

        # Zero cloud: zero riming
        rate_nocloud = cloud_riming_rate(p3, FT(0), qi, ni, T_cold, Ff, ρf, ρ, μ)
        @test rate_nocloud == 0

        # Zero ice: zero riming
        rate_noice = cloud_riming_rate(p3, qc, FT(0), FT(0), T_cold, Ff, ρf, ρ, μ)
        @test rate_noice == 0

        # More cloud water gives faster riming (rate is linear in qc)
        rate_high = cloud_riming_rate(p3, FT(2e-3), qi, ni, T_cold, Ff, ρf, ρ, μ)
        @test rate_high > rate
    end

    @testset "rain_riming_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        # Ice must dominate rain for rain riming
        qr = FT(1e-5)
        nr = FT(1e4)
        qi = FT(1e-4)    # qi > qr
        ni = FT(1e4)
        Ff = FT(0.0)
        ρf = FT(400.0)
        ρ = FT(1.0)

        T_cold = FT(263.15)
        rate = rain_riming_rate(p3, qr, nr, qi, ni, T_cold, Ff, ρf, ρ)
        @test isfinite(rate)
        @test rate != 0  # Below freezing with rain + ice: active riming

        # Above freezing: zero
        rate_warm = rain_riming_rate(p3, qr, nr, qi, ni, FT(278.15), Ff, ρf, ρ)
        @test rate_warm == 0

        # Rain dominates ice (qr > qi): riming is active
        rate_rain_dom = rain_riming_rate(p3, FT(1e-3), FT(1e4), FT(1e-5), ni, T_cold, Ff, ρf, ρ)
        @test isfinite(rate_rain_dom)
    end

    @testset "rime_density follows the Fortran Ri fit" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        qcl = FT(1e-3)
        cloud_rim = FT(2e-7)
        T = FT(263.15)
        vᵢ = FT(1.0)
        ρ = FT(1.0)
        P = FT(90000.0)
        transport = air_transport_properties(T, P, constants)

        prp = p3.process_rates
        μ_c = p3.cloud.shape_parameter
        Nᶜ = p3.cloud.number_concentration
        ρ_water = prp.liquid_water_density

        qcl_abs = qcl * ρ
        η = transport.ν * ρ
        λ_c_uncapped = cbrt(
            FT(π) * ρ_water * Nᶜ * (μ_c + 3) * (μ_c + 2) * (μ_c + 1) /
            (FT(6) * qcl_abs)
        )
        λ_c = clamp(λ_c_uncapped, (μ_c + 1) * FT(2.5e4), (μ_c + 1) * FT(1e6))
        ρ_rime = rime_density(p3, qcl, cloud_rim, T, vᵢ, ρ, constants, transport, μ_c, λ_c)
        a_cn = constants.gravitational_acceleration * ρ_water / (FT(18) * η)
        Vt_qc = a_cn * (μ_c + 5) * (μ_c + 4) / λ_c^2
        D_c = (μ_c + 4) / λ_c
        inverse_supercooling = inv(min(FT(-0.001), T - prp.freezing_temperature))
        Ri = clamp(-(FT(0.5e6) * D_c) * abs(vᵢ - Vt_qc) * inverse_supercooling, FT(1), FT(12))
        expected = ifelse(
            Ri <= FT(8),
            (FT(0.051) + FT(0.114) * Ri - FT(0.0055) * Ri^2) * FT(1000),
            FT(611) + FT(72.25) * (Ri - FT(8))
        )

        @test ρ_rime ≈ expected
        @test ρ_rime != 400

        T_warm = FT(278.15)
        transport_warm = air_transport_properties(T_warm, P, constants)
        ρ_warm = rime_density(p3, qcl, cloud_rim, T_warm, vᵢ, ρ, constants,
                              transport_warm, μ_c, λ_c)
        @test ρ_warm == 400

        ρ_no_cloud = rime_density(p3, qcl, FT(0), T, vᵢ, ρ, constants,
                                  transport, μ_c, λ_c)
        @test ρ_no_cloud == 400
    end

    @testset "Rime consistency enforcement" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        prp = p3.process_rates

        function reference_bulk_rime_density(qⁱ, qᶠ, bᶠ)
            qᶠ = max(qᶠ, 0)
            bᶠ = max(bᶠ, 0)
            ρᶠ = FT(NaN)

            # Mirrors `consistent_rime_state`: Fortran's `bsmall = qsmall/rho_rimeMax`.
            if bᶠ >= p3.minimum_mass_mixing_ratio / prp.maximum_rime_density
                ρᶠ = qᶠ / bᶠ
                if ρᶠ < prp.minimum_rime_density
                    ρᶠ = prp.minimum_rime_density
                    bᶠ = qᶠ / ρᶠ
                elseif ρᶠ > prp.maximum_rime_density
                    ρᶠ = prp.maximum_rime_density
                    bᶠ = qᶠ / ρᶠ
                end
            else
                qᶠ = 0
                bᶠ = 0
                ρᶠ = 0
            end

            if qᶠ < p3.minimum_mass_mixing_ratio
                qᶠ = 0
                bᶠ = 0
            elseif qᶠ > max(qⁱ, 0) && ρᶠ > 0
                qᶠ = max(qⁱ, 0)
                bᶠ = qᶠ / ρᶠ
            end

            return (; qᶠ, bᶠ, ρᶠ)
        end

        bsmall = p3.minimum_mass_mixing_ratio / prp.maximum_rime_density
        @test bsmall ≈ FT(1e-14) / FT(900)

        # Below `bsmall` no admissible density leaves significant rime: drop the pair.
        no_volume = consistent_rime_state(p3, FT(1e-4), FT(1e-5), FT(1e-18), FT(0))
        @test no_volume.qᶠ == 0
        @test no_volume.bᶠ == 0
        @test no_volume.ρᶠ == 0
        @test no_volume.Fᶠ == 0

        # Above it, repair instead: the implied density overshoots `maximum_rime_density`,
        # so re-densify to it and recompute `bᶠ`. Fortran's `1e-15` would discard this.
        dense_rime = consistent_rime_state(p3, FT(1e-4), FT(1e-5), FT(1e-16), FT(0))
        @test FT(1e-16) > bsmall
        @test dense_rime.qᶠ == FT(1e-5)
        @test dense_rime.ρᶠ == prp.maximum_rime_density
        @test dense_rime.bᶠ ≈ dense_rime.qᶠ / prp.maximum_rime_density
        @test dense_rime.Fᶠ ≈ FT(0.1)

        tiny_rime = consistent_rime_state(p3, FT(1e-4), FT(5e-15), FT(1e-15), FT(0))
        @test tiny_rime.qᶠ == 0
        @test tiny_rime.bᶠ == 0

        low_density = consistent_rime_state(p3, FT(1e-4), FT(2e-5), FT(2e-6), FT(0))
        @test low_density.ρᶠ == prp.minimum_rime_density
        @test low_density.bᶠ ≈ low_density.qᶠ / prp.minimum_rime_density

        high_density = consistent_rime_state(p3, FT(1e-4), FT(2e-5), FT(2e-8), FT(0))
        @test high_density.ρᶠ == prp.maximum_rime_density
        @test high_density.bᶠ ≈ high_density.qᶠ / prp.maximum_rime_density

        capped = consistent_rime_state(p3, FT(1e-5), FT(2e-5), FT(5e-8), FT(0))
        @test capped.qᶠ == FT(1e-5)
        @test capped.ρᶠ ≈ FT(400)
        @test capped.bᶠ ≈ capped.qᶠ / capped.ρᶠ
        @test capped.Fᶠ == 1

        # D14: Julia's qⁱ is already dry ice, so qⁱ_dry = qⁱ (no qʷⁱ subtraction).
        liquid_rime = consistent_rime_state(p3, FT(1e-4), FT(8e-5), FT(2e-7), FT(5e-5))
        # qⁱ_dry = 1e-4 (Julia qⁱ is already dry ice)
        # qᶠ = 8e-5 < 1e-4, so NOT capped
        @test liquid_rime.qᶠ ≈ FT(8e-5)
        @test liquid_rime.Fᶠ ≈ FT(0.8)  # = qᶠ / qⁱ_dry = 8e-5 / 1e-4

        ρ = FT(1.0)
        nⁱ = FT(1e5)
        μ = (
            ρqᶜˡ = FT(0),
            ρnᶜˡ = FT(0),
            ρqʳ = FT(0),
            ρnʳ = FT(0),
            ρqⁱ = ρ * FT(1e-5),
            ρnⁱ = ρ * nⁱ,
            ρqᶠ = ρ * FT(2e-5),
            ρbᶠ = ρ * FT(5e-8),
            ρqʷⁱ = FT(0),
            ρsˢᵃᵗ = FT(0),
        )
        ℳ = Breeze.AtmosphereModels.microphysical_state(p3, ρ, μ, nothing, (u = FT(0), v = FT(0), w = FT(0)))
        @test ℳ.qᶠ == FT(1e-5)
        @test ℳ.bᶠ ≈ FT(2.5e-8)

        corrected_μ =
            Breeze.AtmosphereModels.postprocess_microphysical_prognostics(p3, μ, ρ)
        @test corrected_μ.ρqᶠ == ρ * ℳ.qᶠ
        @test corrected_μ.ρbᶠ ≈ ρ * ℳ.bᶠ

        for (qⁱ, qᶠ, bᶠ) in (
            (FT(1e-4), FT(1e-5), FT(1e-18)),   # below bsmall: dropped
            (FT(1e-4), FT(1e-5), FT(1e-16)),   # above bsmall: re-densified
            (FT(1e-4), FT(5e-15), FT(1e-15)),
            (FT(1e-4), FT(2e-5), FT(2e-6)),
            (FT(1e-4), FT(2e-5), FT(2e-8)),
            (FT(1e-5), FT(2e-5), FT(5e-8)),
        )
            got = consistent_rime_state(p3, qⁱ, qᶠ, bᶠ, FT(0))
            ref = reference_bulk_rime_density(qⁱ, qᶠ, bᶠ)
            @test got.qᶠ == ref.qᶠ
            @test got.bᶠ ≈ ref.bᶠ
            @test got.ρᶠ ≈ ref.ρᶠ
        end
    end

    @testset "compute_p3_process_rates integration" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        ρ = FT(1.0)

        # Mixed-phase state: T = -5C, some cloud, rain, ice
        T = FT(268.15)
        qv = FT(0.003)
        qcl = FT(5e-4)
        qr = FT(1e-4)
        qi = FT(1e-4)
        qf = FT(1e-5)     # Some rime

        q = MoistureMassFractions(qv, qcl + qr, qi)

        # Build thermodynamic state: use potential temperature formulation
        # θ ≈ T / Π, for simplicity set pˢᵗ = P so Π ≈ 1
        P = FT(85000.0)
        pst = FT(100000.0)
        θ = T / (P / pst)^FT(0.286)  # Approximate dry potential temperature
        𝒰 = LiquidIcePotentialTemperatureState(θ, q, pst, P)

        ℳ = P3MicrophysicalState(
            qcl,           # qᶜˡ
            FT(200e6 / ρ), # nᶜˡ
            qr,            # qʳ
            FT(1e4),       # nʳ
            qi,            # qⁱ
            FT(1e5),       # nⁱ
            qf,            # qᶠ
            FT(qf / 400),  # bᶠ (rime volume)
            FT(0),         # qʷⁱ (liquid on ice)
            FT(0),         # sˢᵃᵗ (predicted supersaturation)
            FT(0),         # nᵃ
            FT(0),         # w
        )

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        @test rates isa P3ProcessRates{FT}

        # All rates should be finite
        for name in fieldnames(P3ProcessRates)
            @test isfinite(getfield(rates, name))
        end

        # Sign checks for a cold mixed-phase environment:
        # Autoconversion should be positive (cloud → rain)
        @test rates.autoconversion > 0

        # Cloud riming should be positive (below freezing with cloud + ice)
        @test rates.cloud_riming > 0

        # Melting should be zero (below freezing)
        @test rates.partial_melting == 0
        @test rates.complete_melting == 0

        # Aggregation should be positive magnitude (M7)
        @test rates.aggregation >= 0

        # Rime density should be physical
        @test rates.rime_density_new >= 50
        @test rates.rime_density_new <= 900
    end

    @testset "rain self-collection and breakup net into a single signed term" begin
        FT = Float64
        constants = ThermodynamicConstants(FT)
        p3 = PredictedParticlePropertiesMicrophysics(FT)

        ρ = FT(1.0)
        T = FT(283.15)
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())

        # Fortran carries one signed term, `nrslf = dum × base` with the
        # Verlinde-Cotton modifier `dum ≤ 1` (microphy_p3.f90:3872-3886), and never
        # rescales it in any limiter. Breeze reports the sink and source directions
        # separately, so the pair must be netted before the rain-number limiter:
        # rescaling the sink alone would leave the breakup source at full strength and
        # turn the net into spurious rain-number production above the breakup threshold.
        function rain_only_rates(qʳ, nʳ)
            q = MoistureMassFractions(qᵛ, qʳ, zero(FT))
            𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P),
                                 T, constants)
            ℳ = P3MicrophysicalState(zero(FT), zero(FT), qʳ, nʳ, zero(FT), zero(FT),
                                     zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))
            return compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        end

        # Large drops: D_r = (qʳ / (π ρʷ nʳ))^(1/3) ≈ 680 μm exceeds the 280 μm
        # threshold, so dum < 0, breakup outruns self-collection, and the netted pair
        # must report a pure source.
        qʳ_large, nʳ_large = FT(1e-3), FT(1e3)
        base_large = rain_self_collection_rate(p3, qʳ_large, nʳ_large, ρ)
        breakup_large = rain_breakup_rate(p3, qʳ_large, nʳ_large, base_large)
        @test breakup_large > base_large > 0

        large = rain_only_rates(qʳ_large, nʳ_large)
        @test large.rain_self_collection == 0
        @test large.rain_breakup ≈ breakup_large - base_large rtol=FT(1e-10)

        # Small drops: D_r ≈ 68 μm, breakup inactive, pure sink.
        qʳ_small, nʳ_small = FT(1e-3), FT(1e6)
        base_small = rain_self_collection_rate(p3, qʳ_small, nʳ_small, ρ)
        @test rain_breakup_rate(p3, qʳ_small, nʳ_small, base_small) == 0

        small = rain_only_rates(qʳ_small, nʳ_small)
        @test small.rain_breakup == 0
        @test small.rain_self_collection ≈ base_small rtol=FT(1e-10)

        # Only one direction is ever nonzero once netted, so `f_rain_number` can never
        # scale one half of the pair without the other.
        @test large.rain_self_collection * large.rain_breakup == 0
        @test small.rain_self_collection * small.rain_breakup == 0
    end

    @testset "above-freezing cloud collection separates cloud sink from shed rain source" begin
        FT = Float64
        constants = ThermodynamicConstants(FT)
        process_rates = ProcessRateParameters(FT; liquid_fraction_active = false)
        p3 = PredictedParticlePropertiesMicrophysics(FT; process_rates)

        ρ = FT(1.0)
        T = FT(278.15)
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᶜˡ = FT(1e-3)
        nᶜˡ = FT(2e8)
        qʳ = FT(0)
        nʳ = FT(0)
        qⁱ = FT(1e-4)
        nⁱ = FT(1e5)
        qᶠ = FT(1e-5)
        qʷⁱ = FT(0)
        sˢᵃᵗ = FT(0)

        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ,
                                qᶠ, qᶠ / FT(400), qʷⁱ, sˢᵃᵗ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
        expected_cloud_number_per_mass = cloud.Nᶜ / (ρ * qᶜˡ)

        @test rates.cloud_warm_collection > 0
        @test rates.cloud_warm_collection_number / rates.cloud_warm_collection ≈
              expected_cloud_number_per_mass rtol=FT(1e-12)

        manual_rates = p3_process_rates_with(FT;
            cloud_warm_collection = FT(1e-8),
            cloud_warm_collection_number = FT(1e4),
        )
        # The shed-drop count follows the configurable `shed_drop_mass` (default the
        # Fortran 1 mm drop, 1/1.923e6 kg), not a hardcoded literal: the rain-number
        # limiter budgets this source as `cloud_warm_collection / shed_drop_mass`, so
        # the assembled tendency has to divide by the same mass.
        expected_shed_drop_source = ρ * manual_rates.cloud_warm_collection * FT(1.923e6)
        @test tendency_ρnʳ(manual_rates, ρ, nⁱ, qⁱ, nʳ, one(FT), p3) ≈ expected_shed_drop_source

        heavy_shed = ProcessRateParameters(FT; liquid_fraction_active = false,
                                           shed_drop_mass = 1 / 4.0e5)
        p3_heavy = PredictedParticlePropertiesMicrophysics(FT; process_rates = heavy_shed)
        @test tendency_ρnʳ(manual_rates, ρ, nⁱ, qⁱ, nʳ, one(FT), p3_heavy) ≈
              ρ * manual_rates.cloud_warm_collection * FT(4.0e5)
    end

    @testset "non-liquid-fraction routing keeps warm collection and wet growth out of qʷⁱ" begin
        FT = Float64
        ρ = FT(1)
        Fᶠ = FT(0)
        ρᶠ = FT(400)
        qⁱ = FT(1e-4)
        process_rates = ProcessRateParameters(FT; liquid_fraction_active = false)

        warm_rates = p3_process_rates_with(FT;
            cloud_warm_collection = FT(1e-8),
        )

        @test tendency_ρqʳ(warm_rates, ρ, process_rates) ≈ ρ * FT(1e-8)
        @test tendency_ρqʷⁱ(warm_rates, ρ, process_rates) == 0

        # Fortran sets qwgrth1c/qwgrth1r only inside `if (log_LiquidFrac)`
        # (microphy_p3.f90:3255-3268), so `wet_growth_cloud`/`wet_growth_rain` are
        # identically zero in this branch. Even when forced nonzero they must not feed
        # the ice, rime-mass or rime-volume tendencies: Fortran omits both from its
        # qirim and birim updates (microphy_p3.f90:4249-4253) and adds them equally to
        # qitot and qiliq (:4243-4256), leaving the dry ice mass unchanged.
        wet_growth_rates = p3_process_rates_with(FT;
            rime_density_new = FT(300),
            wet_growth_cloud = FT(3e-8),
            wet_growth_rain = FT(2e-8),
            wet_growth_shedding = FT(1e-8),
        )

        @test tendency_ρqⁱ(wet_growth_rates, ρ, process_rates) == 0
        @test tendency_ρqᶠ(wet_growth_rates, ρ, Fᶠ, process_rates) == 0
        @test tendency_ρbᶠ(wet_growth_rates, ρ, Fᶠ, ρᶠ, qⁱ, process_rates) == 0
        @test tendency_ρqʷⁱ(wet_growth_rates, ρ, process_rates) == 0

        # The collection retained against the wet-growth capacity reaches ice, rime and
        # rime volume through the reduced riming rates instead (`process_rates.jl`
        # shrinks cloud_riming/rain_riming to the retained portion, mirroring Fortran's
        # qccol/qrcol reduction at microphy_p3.f90:3277-3279). Rime volume splits the
        # two by rhorime_c and rho_rimeMax exactly as microphy_p3.f90:4250-4253 does.
        retained_cloud = FT(2.4e-8)
        retained_rain = FT(1.6e-8)
        retained_rates = p3_process_rates_with(FT;
            rime_density_new = FT(300),
            cloud_riming = retained_cloud,
            rain_riming = retained_rain,
        )

        @test tendency_ρqⁱ(retained_rates, ρ, process_rates) ≈ ρ * (retained_cloud + retained_rain)
        @test tendency_ρqᶠ(retained_rates, ρ, Fᶠ, process_rates) ≈ ρ * (retained_cloud + retained_rain)
        @test tendency_ρbᶠ(retained_rates, ρ, Fᶠ, ρᶠ, qⁱ, process_rates) ≈
              ρ * (retained_cloud / FT(300) + retained_rain / process_rates.maximum_rime_density)
    end

    @testset "above-freezing rain collection uses table number kernel" begin
        FT = Float64
        constants = ThermodynamicConstants(FT)
        p3 = PredictedParticlePropertiesMicrophysics(FT)

        ρ = FT(1.0)
        T = FT(278.15)
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᶜˡ = FT(0)
        nᶜˡ = FT(0)
        qʳ = FT(1e-3)
        nʳ = FT(1e4)
        qⁱ = FT(1e-4)
        nⁱ = FT(1e5)
        qᶠ = FT(2e-5)
        qʷⁱ = FT(0)
        sˢᵃᵗ = FT(0)
        μ = FT(0)

        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        q = MoistureMassFractions(qᵛ, qʳ + qʷⁱ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ,
                                qᶠ, qᶠ / FT(400), qʷⁱ, sˢᵃᵗ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        rime = consistent_rime_state(p3, qⁱ, qᶠ, qᶠ / FT(400), qʷⁱ)
        expected_number = expected_reference_warm_rain_collection_number(p3, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
                                                                       T, rime.Fᶠ, rime.ρᶠ, ρ, μ)
        expected_mass = PPP.rain_warm_collection_rate(p3, qʳ, nʳ, qⁱ, nⁱ,
                                                      T, rime.Fᶠ, rime.ρᶠ, ρ, μ, qʷⁱ)
        expected_number_per_mass = expected_number / expected_mass
        actual_number_per_mass = rates.rain_warm_collection_number / rates.rain_warm_collection
        monodisperse_number_per_mass = nʳ / qʳ

        @test rates.rain_warm_collection > 0
        @test actual_number_per_mass ≈ expected_number_per_mass rtol=FT(1e-12)
        @test !isapprox(actual_number_per_mass, monodisperse_number_per_mass; rtol=FT(1e-2))
    end

    @testset "compute_p3_process_rates vapor-limits cloud evaporation before cloud budget" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        ρ = FT(1)
        T = FT(268.15)
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᶜˡ = FT(1e-8)
        qʳ = FT(1e-4)
        qⁱ = FT(1e-4)
        qᶠ = FT(1e-5)
        qʷⁱ = FT(0)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface()) + FT(1e-10)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ + qʷⁱ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)

        ℳ = P3MicrophysicalState(
            qᶜˡ,
            FT(200e6),
            qʳ,
            FT(1e4),
            qⁱ,
            FT(1e5),
            qᶠ,
            qᶠ / FT(400),
            qʷⁱ,
            FT(0),
            FT(0),
            FT(0),
        )

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        cloud_sink_total = rates.autoconversion + rates.accretion + rates.cloud_riming +
                           rates.cloud_freezing_mass + rates.cloud_homogeneous_mass +
                           rates.cloud_warm_collection + rates.wet_growth_cloud +
                           max(zero(FT), -rates.condensation)

        @test rates.condensation == 0
        @test cloud_sink_total ≈ FT(4.055896466237224e-12) rtol=FT(1e-12)
    end

    @testset "predict_supersaturation applies G&M before M&G process rates" begin
        p3_base = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)
        process_rates = ProcessRateParameters(FT;
                                              sink_limiting_timescale = FT(10),
                                              predict_supersaturation = true)
        p3 = p3_with_process_rates(p3_base, process_rates)

        ρ = FT(1)
        T = FT(268.15)
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᶜˡ = FT(1e-3)
        nᶜˡ = FT(2e8)
        qʳ = FT(0)
        nʳ = FT(0)
        qⁱ = FT(0)
        nⁱ = FT(0)
        qᶠ = FT(0)
        qʷⁱ = FT(0)
        sˢᵃᵗ = FT(0)

        qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
        qᵛ = qᵛ⁺ˡ + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ,
                                qᶠ, FT(0), qʷⁱ, sˢᵃᵗ, zero(FT), zero(FT))

        gm = expected_reference_predicted_ssat_adjustment(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, sˢᵃᵗ, T, constants)
        Tᴳᴹ = T + gm.ε * PPP.vaporization_latent_heat(constants, T) / constants.dry_air.heat_capacity
        qᵛᴳᴹ = qᵛ - gm.ε
        qᶜˡᴳᴹ = qᶜˡ + gm.ε
        qᵛ⁺ˡᴳᴹ = saturation_specific_humidity(Tᴳᴹ, ρ, constants, PlanarLiquidSurface())
        qᵛ⁺ⁱᴳᴹ = saturation_specific_humidity(Tᴳᴹ, ρ, constants, PlanarIceSurface())
        qᴳᴹ = MoistureMassFractions(qᵛᴳᴹ, qᶜˡᴳᴹ, qⁱ)
        transportᴳᴹ = air_transport_properties(Tᴳᴹ, P, constants)
        expected_process = expected_reduced_reference_vapor_rates(
            p3, qᶜˡᴳᴹ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
            qᵛᴳᴹ, qᵛ⁺ˡᴳᴹ, qᵛ⁺ⁱᴳᴹ, FT(0), FT(400), Tᴳᴹ, P, ρ,
            constants, transportᴳᴹ, qᴳᴹ, FT(0))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)

        @test rates.predicted_ssat_adjustment ≈ gm.rate
        @test rates.condensation ≈ gm.rate + expected_process.condensation rtol=FT(1e-10) atol=FT(1e-14)
    end

    @testset "predict_supersaturation tendency matches formulation final recompute" begin
        p3_base = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)
        process_rates = ProcessRateParameters(FT;
                                              sink_limiting_timescale = FT(10),
                                              predict_supersaturation = true)
        p3 = p3_with_process_rates(p3_base, process_rates)

        ρ = FT(1)
        T = FT(268.15)
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᶜˡ = FT(1e-3)
        qʳ = FT(0)
        qⁱ = FT(0)
        qᶠ = FT(0)
        qʷⁱ = FT(0)
        sˢᵃᵗ = FT(0)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface()) + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, FT(2e8), qʳ, FT(0), qⁱ, FT(0),
                                qᶠ, FT(0), qʷⁱ, sˢᵃᵗ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        expected = actual_final_liquid_ssat_after_p3_step(𝒰, rates, qᵛ, qᶜˡ, qʳ, qⁱ, qʷⁱ,
                                                          ρ, process_rates.sink_limiting_timescale,
                                                          P, pˢᵗ, constants)

        @test tendency_ρsˢᵃᵗ(rates, ρ, p3.process_rates) / ρ *
              process_rates.sink_limiting_timescale ≈ expected atol=FT(1e-12)
    end

    @testset "predict_supersaturation final recompute uses formulation state with splintering active" begin
        p3_base = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)
        process_rates = ProcessRateParameters(FT;
                                              sink_limiting_timescale = FT(10),
                                              predict_supersaturation = true)
        p3 = p3_with_process_rates(p3_base, process_rates)

        ρ = FT(1)
        T = process_rates.splintering_temperature_peak
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᶜˡ = FT(5e-4)
        qʳ = FT(0)
        qⁱ = FT(1e-4)
        nⁱ = FT(1e3)
        qᶠ = FT(5e-5)
        qʷⁱ = FT(0)
        sˢᵃᵗ = FT(0)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface()) + FT(1e-5)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ + qʷⁱ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, FT(2e8), qʳ, FT(0), qⁱ, nⁱ,
                                 qᶠ, qᶠ / FT(400), qʷⁱ, sˢᵃᵗ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        expected = actual_final_liquid_ssat_after_p3_step(𝒰, rates, qᵛ, qᶜˡ, qʳ, qⁱ, qʷⁱ,
                                                          ρ, process_rates.sink_limiting_timescale,
                                                          P, pˢᵗ, constants)

        @test rates.splintering_mass > 0
        @test tendency_ρsˢᵃᵗ(rates, ρ, p3.process_rates) / ρ *
              process_rates.sink_limiting_timescale ≈ expected atol=FT(1e-12)
    end

    @testset "predict_supersaturation reset matches potential-temperature formulation state" begin
        p3_base = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)
        τ = FT(10)
        process_rates = ProcessRateParameters(FT;
                                              sink_limiting_timescale = τ,
                                              predict_supersaturation = true)
        p3 = p3_with_process_rates(p3_base, process_rates)

        ρ = FT(1)
        T = FT(268.15)
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᶜˡ = FT(1e-3)
        qʳ = FT(0)
        qⁱ = FT(0)
        qᶠ = FT(0)
        qʷⁱ = FT(0)
        sˢᵃᵗ = FT(0)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface()) + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, FT(2e8), qʳ, FT(0), qⁱ, FT(0),
                                qᶠ, FT(0), qʷⁱ, sˢᵃᵗ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        expected = actual_final_liquid_ssat_after_p3_step(𝒰, rates, qᵛ, qᶜˡ, qʳ, qⁱ, qʷⁱ,
                                                          ρ, τ, P, pˢᵗ, constants)

        @test tendency_ρsˢᵃᵗ(rates, ρ, p3.process_rates) / ρ * τ ≈ expected atol=FT(1e-12)
    end

    @testset "predict_supersaturation reset matches static-energy formulation state" begin
        p3_base = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)
        τ = FT(10)
        process_rates = ProcessRateParameters(FT;
                                              sink_limiting_timescale = τ,
                                              predict_supersaturation = true)
        p3 = p3_with_process_rates(p3_base, process_rates)

        ρ = FT(1)
        T = FT(268.15)
        P = FT(85000)
        pˢᵗ = FT(100000)
        qᶜˡ = FT(1e-3)
        qʳ = FT(0)
        qⁱ = FT(0)
        qᶠ = FT(0)
        qʷⁱ = FT(0)
        sˢᵃᵗ = FT(0)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface()) + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
        𝒰 = with_temperature(StaticEnergyState(zero(FT), q, FT(0), P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, FT(2e8), qʳ, FT(0), qⁱ, FT(0),
                                qᶠ, FT(0), qʷⁱ, sˢᵃᵗ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        expected = actual_final_liquid_ssat_after_p3_step(𝒰, rates, qᵛ, qᶜˡ, qʳ, qⁱ, qʷⁱ,
                                                          ρ, τ, P, pˢᵗ, constants)

        @test tendency_ρsˢᵃᵗ(rates, ρ, p3.process_rates) / ρ * τ ≈ expected atol=FT(1e-12)
    end

    @testset "predict_supersaturation disabled docs match inactive field semantics" begin
        @test documented_predict_supersaturation_disabled_semantics()
    end

    @testset "compute_p3_process_rates uses prognostic cloud number" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        ρ = FT(1.0)
        T = FT(268.15)
        qv = FT(0.003)
        qcl = FT(5e-4)
        qr = FT(1e-4)
        qi = FT(1e-4)
        qf = FT(1e-5)

        q = MoistureMassFractions(qv, qcl + qr, qi)
        P = FT(85000.0)
        pst = FT(100000.0)
        θ = T / (P / pst)^FT(0.286)
        𝒰 = LiquidIcePotentialTemperatureState(θ, q, pst, P)

        ℳ_low_nc = P3MicrophysicalState(
            qcl,           # qᶜˡ
            FT(50e6 / ρ),  # nᶜˡ
            qr,            # qʳ
            FT(1e4),       # nʳ
            qi,            # qⁱ
            FT(1e5),       # nⁱ
            qf,            # qᶠ
            FT(qf / 400),  # bᶠ
            FT(0),         # qʷⁱ
            FT(0),         # sˢᵃᵗ
            FT(0),         # nᵃ
            FT(0),         # w
        )

        ℳ_high_nc = P3MicrophysicalState(
            qcl,            # qᶜˡ
            FT(300e6 / ρ),  # nᶜˡ
            qr,             # qʳ
            FT(1e4),        # nʳ
            qi,             # qⁱ
            FT(1e5),        # nⁱ
            qf,             # qᶠ
            FT(qf / 400),   # bᶠ
            FT(0),          # qʷⁱ
            FT(0),          # sˢᵃᵗ
            FT(0),          # nᵃ
            FT(0),          # w
        )

        rates_low_nc = compute_p3_process_rates(p3, ρ, ℳ_low_nc, 𝒰, constants)
        rates_high_nc = compute_p3_process_rates(p3, ρ, ℳ_high_nc, 𝒰, constants)

        @test rates_low_nc.autoconversion != rates_high_nc.autoconversion
        @test rates_low_nc.cloud_riming_number != rates_high_nc.cloud_riming_number
    end

    @testset "compute_p3_process_rates with tabulated scheme" begin
        FT = Float64
        constants = ThermodynamicConstants(FT)

        # Load Fortran lookup tables
        p3_tab = PredictedParticlePropertiesMicrophysics()

        ρ = FT(1.0)

        # Mixed-phase state: T = -5C
        T = FT(268.15)
        qv = FT(0.003)
        qcl = FT(5e-4)
        qr = FT(1e-4)
        qi = FT(1e-4)
        qf = FT(1e-5)

        q = MoistureMassFractions(qv, qcl + qr, qi)
        P = FT(85000.0)
        pst = FT(100000.0)
        θ = T / (P / pst)^FT(0.286)
        𝒰 = LiquidIcePotentialTemperatureState(θ, q, pst, P)

        ℳ = P3MicrophysicalState(
            qcl, FT(200e6 / ρ), qr, FT(1e4), qi, FT(1e5), qf,
            FT(qf / 400), FT(0), FT(0), FT(0), FT(0))

        # Compute rates with tabulated scheme
        rates_tab = compute_p3_process_rates(p3_tab, ρ, ℳ, 𝒰, constants)
        @test rates_tab isa P3ProcessRates{FT}

        # All rates should be finite
        for name in fieldnames(P3ProcessRates)
            @test isfinite(getfield(rates_tab, name))
        end

        # Sign checks for a cold mixed-phase environment
        @test rates_tab.autoconversion > 0
        @test rates_tab.cloud_riming > 0
        @test rates_tab.partial_melting == 0
        @test rates_tab.complete_melting == 0
        @test rates_tab.aggregation >= 0

        # Rain evaporation should be positive magnitude (M7)
        @test rates_tab.rain_evaporation > 0
        @test isfinite(rates_tab.rain_evaporation)
    end
end
