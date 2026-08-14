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
    tendency_ρsᵛ⁺ˡ,
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

function expected_reference_rain_vapor_relaxation(p3, qʳ, nʳ, ρ, transport, FT)
    parameters = p3.process_rates
    qʳ_eff = max(0, qʳ)
    nʳ_eff = max(max(0, nʳ), FT(1e-16))
    λʳ = PPP.rain_slope_parameter(qʳ_eff, nʳ_eff, parameters)
    nʳ_bounded = qʳ_eff * λʳ^3 / (FT(π) * parameters.liquid_water_density)
    Nʳ₀ = nʳ_bounded * λʳ
    velocity_diameter_integral = p3.rain.evaporation(log10(λʳ))
    constant_integral = FT(PPP.RAIN_F1R) / λʳ^2
    schmidt_correction = cbrt(transport.ν / max(transport.Dᵛ, FT(1e-10)))
    evaporation_integral = constant_integral +
                           FT(PPP.RAIN_F2R) * schmidt_correction /
                           sqrt(max(transport.ν, FT(1e-10))) * velocity_diameter_integral
    rain_relaxation = FT(2π) * Nʳ₀ * ρ * transport.Dᵛ * evaporation_integral
    return ifelse(qʳ_eff >= p3.minimum_mass_mixing_ratio, rain_relaxation, zero(FT))
end

function expected_reference_warm_rain_collection_number(p3, qʳ, nʳ, qⁱ, qʷⁱ,
                                                        nⁱ, T, Fᶠ, ρᶠ, ρ)
    FT = typeof(qʳ)
    parameters = p3.process_rates
    qʳ_eff = max(0, qʳ)
    nʳ_eff = max(0, nʳ)
    qⁱ_total = PPP.total_ice_mass(qⁱ, qʷⁱ)
    nⁱ_eff = max(0, nⁱ)
    active = (T > parameters.freezing_temperature) &
             (qʳ_eff > FT(1e-14)) &
             (qⁱ_total > FT(1e-14)) &
             (nʳ_eff > FT(1)) &
             (nⁱ_eff > FT(1))

    λʳ = PPP.rain_slope_parameter(qʳ_eff, nʳ_eff, parameters)
    nʳ_bounded = PPP.rain_number_from_slope(qʳ_eff, λʳ, parameters)
    Fˡ = PPP.liquid_fraction_on_ice(qⁱ, qʷⁱ)
    m_mean = PPP.mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    _, number_kernel = PPP.ice_rain_collection_lookup(PPP.rain_ice_collection_table(p3),
                                                      m_mean, λʳ, Fᶠ, Fˡ, ρᶠ)
    ρ₀ = p3.ice.fall_speed.reference_air_density
    density_correction = (ρ₀ / max(ρ, FT(0.01)))^FT(0.54)
    Nʳ₀ = nʳ_bounded * λʳ
    rate = parameters.rain_ice_collection_efficiency * Nʳ₀ * nⁱ_eff * ρ *
           density_correction * number_kernel
    return ifelse(active, rate, zero(FT))
end

function expected_reference_ice_vapor_relaxation(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ,
                                                 T, P, ρ, constants, transport, q)
    FT = typeof(qⁱ)
    Fˡ = PPP.liquid_fraction_on_ice(qⁱ, qʷⁱ)
    m_mean = PPP.mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    ρ_air = Breeze.Thermodynamics.density(T, P, q, constants)
    ρ_correction = PPP.ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ_air)
    C_fv = PPP.deposition_ventilation(p3.ice.deposition.ventilation,
                                      p3.ice.deposition.ventilation_enhanced,
                                      m_mean, Fᶠ, Fˡ, ρᶠ, p3.process_rates,
                                      transport.ν, transport.Dᵛ, ρ_correction, p3)
    ice_relaxation = FT(2π) * ρ * transport.Dᵛ * max(max(0, nⁱ), FT(1e-16)) * C_fv
    qⁱ_total = PPP.total_ice_mass(qⁱ, qʷⁱ)
    active = (qⁱ_total >= p3.minimum_mass_mixing_ratio) &
             (Fˡ < p3.process_rates.liquid_fraction_clipping_threshold)
    return ifelse(active, ice_relaxation, zero(FT))
end

function expected_reference_coating_vapor_relaxation(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ,
                                                     T, P, ρ, constants, transport, q)
    FT = typeof(qⁱ)
    Fˡ = PPP.liquid_fraction_on_ice(qⁱ, qʷⁱ)
    m_mean = PPP.mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    ρ_air = Breeze.Thermodynamics.density(T, P, q, constants)
    ρ_correction = PPP.ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ_air)
    C_fv = PPP.deposition_ventilation(p3.ice.deposition.ventilation,
                                      p3.ice.deposition.ventilation_enhanced,
                                      m_mean, Fᶠ, Fˡ, ρᶠ, p3.process_rates,
                                      transport.ν, transport.Dᵛ, ρ_correction, p3)
    coating_relaxation = FT(2π) * ρ * transport.Dᵛ * max(max(0, nⁱ), FT(1e-16)) * C_fv
    qⁱ_total = PPP.total_ice_mass(qⁱ, qʷⁱ)
    active = (qⁱ_total >= p3.minimum_mass_mixing_ratio) &
             (Fˡ >= p3.process_rates.liquid_fraction_clipping_threshold)
    return ifelse(active, coating_relaxation, zero(FT))
end

function expected_reduced_reference_vapor_rates(p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
                                               qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                               constants, transport, q;
                                              temperature_tendency = zero(T),
                                              vapor_tendency = zero(qᵛ))
    FT = typeof(qᶜˡ)
    τ = max(p3.process_rates.sink_limiting_timescale, eps(FT))
    Rᵛ = FT(Breeze.Thermodynamics.vapor_gas_constant(constants))
    ℒˡ = PPP.vaporization_latent_heat(constants, T)
    ℒⁱ = PPP.sublimation_latent_heat(constants, T)
    cᵖᵈ = constants.dry_air.heat_capacity

    dqᵛ⁺ˡ_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    dqᵛ⁺ⁱ_dT = qᵛ⁺ⁱ * ℒⁱ / (Rᵛ * T^2)
    liquid_psychrometric_correction = 1 + ℒˡ * dqᵛ⁺ˡ_dT / cᵖᵈ
    ice_psychrometric_correction = 1 + ℒⁱ * dqᵛ⁺ⁱ_dT / cᵖᵈ

    cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
    cloud_relaxation = PPP.cloud_vapor_relaxation_coefficient(
        p3, qᶜˡ, ρ, transport.Dᵛ, cloud.μᶜˡ, cloud.λᶜˡ, cloud.nᶜˡ)
    rain_relaxation = expected_reference_rain_vapor_relaxation(p3, qʳ, nʳ, ρ,
                                                               transport, FT)
    ice_relaxation = expected_reference_ice_vapor_relaxation(
        p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ, constants, transport, q)
    coating_relaxation = expected_reference_coating_vapor_relaxation(
        p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ, constants, transport, q)

    ice_liquid_coupling = (1 + ℒⁱ * dqᵛ⁺ˡ_dT / cᵖᵈ) / ice_psychrometric_correction
    total_relaxation = max(cloud_relaxation + rain_relaxation +
                           ice_relaxation * ice_liquid_coupling + coating_relaxation,
                           FT(1e-20))
    transient = (1 - exp(-total_relaxation * τ)) / τ
    supersaturation = qᵛ - qᵛ⁺ˡ
    bergeron_driver = -(qᵛ⁺ˡ - qᵛ⁺ⁱ) * ice_liquid_coupling * ice_relaxation
    external_driver = vapor_tendency - dqᵛ⁺ˡ_dT * temperature_tendency
    total_driver = external_driver + bergeron_driver
    supersaturation_departure = supersaturation - total_driver / total_relaxation

    raw_cloud_growth = (total_driver * cloud_relaxation / total_relaxation +
                        supersaturation_departure * cloud_relaxation / total_relaxation * transient) /
                       liquid_psychrometric_correction
    raw_rain_growth = (total_driver * rain_relaxation / total_relaxation +
                       supersaturation_departure * rain_relaxation / total_relaxation * transient) /
                      liquid_psychrometric_correction
    raw_ice_growth = (total_driver * ice_relaxation / total_relaxation +
                      supersaturation_departure * ice_relaxation / total_relaxation * transient) /
                     ice_psychrometric_correction +
                     (qᵛ⁺ˡ - qᵛ⁺ⁱ) * ice_relaxation / ice_psychrometric_correction
    raw_coating_growth = (total_driver * coating_relaxation / total_relaxation +
                          supersaturation_departure * coating_relaxation / total_relaxation * transient) /
                         liquid_psychrometric_correction

    condensation = ifelse(raw_cloud_growth < 0, zero(FT), min(raw_cloud_growth, qᵛ / τ))
    rain_condensation = ifelse(raw_rain_growth < 0, zero(FT), min(raw_rain_growth, qᵛ / τ))
    rain_evaporation = ifelse(raw_rain_growth < 0,
                              min(-raw_rain_growth, max(0, qʳ) / τ), zero(FT))

    is_sublimation = raw_ice_growth < 0
    deposition = ifelse(is_sublimation,
                        -min(-raw_ice_growth * p3.process_rates.calibration_factor_sublimation,
                             max(0, qⁱ) / τ),
                        min(raw_ice_growth * p3.process_rates.calibration_factor_deposition,
                            qᵛ / τ))

    coating_condensation = ifelse(raw_coating_growth < 0, zero(FT),
                                  min(raw_coating_growth, qᵛ / τ))
    coating_evaporation = ifelse(raw_coating_growth < 0,
                                 min(-raw_coating_growth, max(0, qʷⁱ) / τ), zero(FT))

    return (; condensation, rain_evaporation, rain_condensation, deposition,
              coating_condensation, coating_evaporation)
end

function expected_reference_supersaturation_adjustment(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, sᵛ⁺ˡ, T, constants)
    FT = typeof(qᶜˡ)
    τ = max(p3.process_rates.sink_limiting_timescale, eps(FT))
    Rᵛ = FT(Breeze.Thermodynamics.vapor_gas_constant(constants))
    ℒˡ = PPP.vaporization_latent_heat(constants, T)
    cᵖᵈ = constants.dry_air.heat_capacity
    dqᵛ⁺ˡ_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    ξˡ = 1 + ℒˡ * dqᵛ⁺ˡ_dT / cᵖᵈ
    ε = (qᵛ - qᵛ⁺ˡ - sᵛ⁺ˡ) / ξˡ
    ε = max(ε, -qᶜˡ)
    ε = ifelse(sᵛ⁺ˡ < 0, min(0, ε), ε)
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
    theory = read(joinpath(@__DIR__, "..", "docs", "src", "microphysics", "p3_theory.md"), String)
    forbidden = "When `false`, the field is recomputed diagnostically"
    required = "When `false`, the field is not allocated"
    return !occursin(forbidden, theory) &&
           count(required, theory) >= 2
end

@testset "P3 Processes" begin

    @testset "Rime splintering respects its temperature and size guards" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        parameters = p3.process_rates

        cloud_riming = FT(3e-7)
        rain_riming = FT(2e-7)
        D_ice = FT(300e-6)
        Fˡ = FT(0.05)
        surface_T = FT(280)
        qᶠ = FT(1e-6)

        left_q, left_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, FT(266.15), D_ice, Fˡ, surface_T, qᶠ)
        peak_q, peak_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, parameters.splintering_temperature_peak, D_ice, Fˡ, surface_T, qᶠ)
        right_q, right_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, FT(269.15), D_ice, Fˡ, surface_T, qᶠ)

        total_riming = cloud_riming + rain_riming
        @test left_n ≈ (FT(1) / FT(3)) * parameters.splintering_rate * total_riming
        @test peak_n ≈ parameters.splintering_rate * total_riming
        @test right_n ≈ FT(0.5) * parameters.splintering_rate * total_riming
        @test left_q ≈ left_n * parameters.splintering_crystal_mass
        @test peak_q ≈ peak_n * parameters.splintering_crystal_mass
        @test right_q ≈ right_n * parameters.splintering_crystal_mass

        cloud_peak_q, rain_peak_q, split_peak_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rates(
            p3, cloud_riming, rain_riming, parameters.splintering_temperature_peak, D_ice, Fˡ, surface_T, qᶠ)
        @test split_peak_n ≈ peak_n
        @test cloud_peak_q ≈ parameters.splintering_rate * cloud_riming * parameters.splintering_crystal_mass
        @test rain_peak_q ≈ parameters.splintering_rate * rain_riming * parameters.splintering_crystal_mass

        _, cloud_only_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, zero(FT), parameters.splintering_temperature_peak, D_ice, Fˡ, surface_T, qᶠ)
        _, small_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, parameters.splintering_temperature_peak, FT(200e-6), Fˡ, surface_T, qᶠ)
        _, wet_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, parameters.splintering_temperature_peak, D_ice, FT(0.2), surface_T, qᶠ)
        _, warm_surface_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, parameters.splintering_temperature_peak, D_ice, Fˡ, FT(283), qᶠ)
        _, no_rime_n = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, rain_riming, parameters.splintering_temperature_peak, D_ice, Fˡ, surface_T, zero(FT))

        # H4: Cloud riming contributes to splintering
        @test cloud_only_n > 0
        cloud_only_q, _ = Breeze.Microphysics.PredictedParticleProperties.rime_splintering_rate(
            p3, cloud_riming, zero(FT), parameters.splintering_temperature_peak, D_ice, Fˡ, surface_T, qᶠ)
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
        parameters = p3.process_rates
        qʳ = FT(1e-3)
        nʳ = FT(1e-5)
        S = FT(0.99)
        thermodynamic_factor = FT(1e8)
        ν = FT(1.5e-5)
        Dᵛ = FT(2.2e-5)
        ρ = FT(1)

        λ_r = PPP.rain_slope_parameter(qʳ, nʳ, parameters)
        nʳ_bounded = qʳ * λ_r^3 / (FT(π) * parameters.liquid_water_density)

        @test λ_r == parameters.minimum_rain_slope
        @test nʳ_bounded > nʳ

        raw_rate = PPP.rain_evaporation_rate(p3.rain.evaporation, qʳ, nʳ, S,
                                             thermodynamic_factor, p3, parameters,
                                             ν, Dᵛ, ρ, FT)
        bounded_rate = PPP.rain_evaporation_rate(p3.rain.evaporation, qʳ, nʳ_bounded, S,
                                                 thermodynamic_factor, p3, parameters,
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
        log_m = log10(qⁱ / nⁱ)
        limiter = PPP.ice_integrals_table(p3).lambda_limiter
        lower_nⁱ = limiter.large_q(log_m, rime_state.Fᶠ, Fˡ,
                                   rime_state.ρᶠ) * qⁱ
        upper_nⁱ = limiter.small_q(log_m, rime_state.Fᶠ, Fˡ,
                                   rime_state.ρᶠ) * qⁱ
        expected_nⁱ = clamp(nⁱ, lower_nⁱ, upper_nⁱ)
        properties = PPP.p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
        shape_parameter = PPP.compute_ice_shape_parameter(
            p3, qⁱ, nⁱ, rime_state.Fᶠ, Fˡ, rime_state.ρᶠ)

        @test expected_nⁱ > nⁱ
        @test properties.nⁱ ≈ expected_nⁱ
        @test isfinite(shape_parameter)

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties)
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
        parameters = ProcessRateParameters(FT)

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
            FT(0.0),    # ni_limit (C3: global Nⁱ cap; zero in warm-environment test)
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
            FT(0.0),    # predicted_supersaturation_adjustment
            FT(0.0),    # predicted_supersaturation_tendency
        )

        # Test each tendency function returns a finite number
        @test isfinite(tendency_ρqᶜˡ(rates, ρ))
        @test isfinite(tendency_ρqʳ(rates, ρ))
        @test isfinite(tendency_ρnʳ(rates, ρ, nⁱ, qⁱ, zero(FT), one(FT),
                                    tendency_test_p3(FT; process_rates = parameters)))
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

    @testset "P3 sediments cloud mass and number with Stokes velocities" begin
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

        properties = PPP.p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
        cache = PPP.p3_fall_speed_compute(p3, ρ, ℳ, properties, constants)
        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
        transport = air_transport_properties(T, P, constants)
        η = transport.ν * ρ
        a_cn = constants.gravitational_acceleration * p3.process_rates.liquid_water_density /
               (FT(18) * max(η, FT(1e-20)))
        expected_mass_velocity = a_cn * (cloud.μᶜˡ + 5) * (cloud.μᶜˡ + 4) / cloud.λᶜˡ^2
        expected_number_velocity = a_cn * (cloud.μᶜˡ + 2) * (cloud.μᶜˡ + 1) / cloud.λᶜˡ^2

        @test cache.wᶜˡ ≈ expected_mass_velocity rtol=FT(1e-12)
        @test cache.wᶜˡₙ ≈ expected_number_velocity rtol=FT(1e-12)
        @test cache.wᶜˡ > cache.wᶜˡₙ

        # The Stokes prefactor scales with the *model's* gravitational acceleration
        # rather than a hardcoded 9.81, so doubling g doubles both fall speeds.
        heavy = ThermodynamicConstants(FT; gravitational_acceleration = 2 * constants.gravitational_acceleration)
        vᶜ = PPP.cloud_terminal_velocities(p3, qᶜˡ, ρ, transport.ν, cloud.μᶜˡ, cloud.λᶜˡ, constants)
        vᶜ_heavy = PPP.cloud_terminal_velocities(p3, qᶜˡ, ρ, transport.ν, cloud.μᶜˡ, cloud.λᶜˡ, heavy)
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
                                                     cloud.μᶜˡ, cloud.λᶜˡ,
                                                     g_constants).mass_weighted
            ρᶠ_above = rime_density(p3, qᶜˡ, cloud_rim, T_rime, v_impact + δ, ρ,
                                    g_constants, transport_rime, cloud.μᶜˡ, cloud.λᶜˡ)
            ρᶠ_below = rime_density(p3, qᶜˡ, cloud_rim, T_rime, v_impact - δ, ρ,
                                    g_constants, transport_rime, cloud.μᶜˡ, cloud.λᶜˡ)
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
               ρsᵛ⁺ˡ = CenterField(grid), ρnᵃ = CenterField(grid))

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
        # An untyped `1e-16` floor on nᶜˡ promoted Nᶜˡ, μᶜˡ, and λᶜˡ to Float64 on the
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
        qᶜˡ = FT(1e-3)     # 1 g/kg cloud water
        Nᶜˡ = FT(100e6)    # 100 cm⁻³ cloud droplet concentration
        ρ  = FT(1.2)       # sea-level air density

        rate = rain_autoconversion_rate(p3, qᶜˡ, Nᶜˡ, ρ)
        @test rate > 0
        # KK2000 gives O(1e-6) kg/kg/s for these inputs
        @test rate > 1e-8
        @test rate < 1e-3

        # Higher cloud water content gives faster autoconversion
        rate_high = rain_autoconversion_rate(p3, FT(2e-3), Nᶜˡ, ρ)
        @test rate_high > rate

        # Zero cloud water gives zero autoconversion
        rate_zero = rain_autoconversion_rate(p3, FT(0), Nᶜˡ, ρ)
        @test rate_zero == 0

        # Small cloud water gives small but nonzero rate (KK2000 has no threshold)
        rate_small = rain_autoconversion_rate(p3, FT(5e-5), Nᶜˡ, ρ)
        @test rate_small > 0
        @test rate_small < rate
    end

    @testset "rain_accretion_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qᶜˡ = FT(1e-3)
        qʳ = FT(1e-3)

        rate = rain_accretion_rate(p3, qᶜˡ, qʳ)
        @test rate > 0
        @test isfinite(rate)

        # Zero cloud gives zero accretion
        @test rain_accretion_rate(p3, FT(0), qʳ) == 0

        # Zero rain gives zero accretion
        @test rain_accretion_rate(p3, qᶜˡ, FT(0)) == 0

        # Higher rain gives faster accretion
        rate_high = rain_accretion_rate(p3, qᶜˡ, FT(2e-3))
        @test rate_high > rate
    end

    @testset "warm_rain_scheme dispatch" begin
        FT = Float64
        qᶜˡ = FT(1e-3)
        qʳ = FT(5e-4)
        Nᶜˡ = FT(1e8)
        nʳ = FT(1e4)
        ρ  = FT(1.0)

        p3_kk = PredictedParticlePropertiesMicrophysics(FT; warm_rain_scheme = KhairoutdinovKogan2000())

        # KK2000 is the default and the only scheme
        p3_default = PredictedParticlePropertiesMicrophysics(FT)
        @test p3_default.warm_rain_scheme isa KhairoutdinovKogan2000
        @test KhairoutdinovKogan2000 <: AbstractWarmRainScheme
        @test rain_autoconversion_rate(p3_default, qᶜˡ, Nᶜˡ, ρ, qʳ) ==
              rain_autoconversion_rate(p3_kk, qᶜˡ, Nᶜˡ, ρ, qʳ)

        autoconversion = rain_autoconversion_rate(p3_kk, qᶜˡ, Nᶜˡ, ρ, qʳ)
        @test isfinite(autoconversion)
        @test autoconversion > 0

        # KK2000 autoconversion ignores qʳ
        @test rain_autoconversion_rate(p3_kk, qᶜˡ, Nᶜˡ, ρ, zero(FT)) == autoconversion

        accretion = rain_accretion_rate(p3_kk, qᶜˡ, qʳ, ρ)
        @test isfinite(accretion)
        @test accretion > 0

        # Rain self-collection: linear form k_rr × ρ × qʳ × nʳ
        self_collection = rain_self_collection_rate(p3_kk, qʳ, nʳ, ρ)
        @test isfinite(self_collection)
        @test self_collection > 0

        # Cloud self-collection is zero for KK2000
        @test PredictedParticleProperties.cloud_self_collection_rate(p3_kk, qᶜˡ, Nᶜˡ, ρ) == 0

        # Seed-drop mass: KK2000 ≈ 25 μm radius
        @test PredictedParticleProperties.rain_seed_drop_mass(p3_kk) ≈ 4π/3 * 1000 * (25e-6)^3

        # Autoconversion removes cloud number in proportion to the mass lost
        autoconversion_only = p3_process_rates_with(FT; autoconversion = FT(1e-7))
        @test tendency_ρnᶜˡ(autoconversion_only, ρ, Nᶜˡ, qᶜˡ, p3_kk) < 0
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
        qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
        qᵛ = qᵛ⁺ˡ + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ + qʷⁱ, qⁱ)
        transport = air_transport_properties(T, P, constants)

        rain_relaxation = PPP.rain_vapor_relaxation_coefficient(
            p3, FT(5e-4), FT(1e6), ρ, transport)
        expected_rain_relaxation = expected_reference_rain_vapor_relaxation(
            p3, FT(5e-4), FT(1e6), ρ, transport, FT)
        # Fˡ = 0 here, so the dry-ice gate inside `coupled_saturation_adjustment_rates`
        # is active, so the raw relaxation coefficient is the dry-ice one.
        ice_relaxation = PPP.ice_vapor_relaxation_coefficient(
            p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ, constants, transport, q)
        expected_ice_relaxation = expected_reference_ice_vapor_relaxation(
            p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ, constants, transport, q)

        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
        # predict_supersaturation defaults to false, so this M&G call sees
        # the host state directly and the G&M ε is gated to zero by
        # `compute_p3_process_rates` (not this function).
        rates = PPP.coupled_saturation_adjustment_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q,
            cloud.μᶜˡ, cloud.λᶜˡ, cloud.nᶜˡ, FT(0), FT(0))
        expected_rates = expected_reduced_reference_vapor_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q)

        # Bergeron check: with ice present, cloud condensation is smaller than
        # with no ice, because ice steals vapor through the shared budget.
        rates_noice = PPP.coupled_saturation_adjustment_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, zero(FT), zero(FT), zero(FT),
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q,
            cloud.μᶜˡ, cloud.λᶜˡ, cloud.nᶜˡ, FT(0), FT(0))

        @test rain_relaxation ≈ expected_rain_relaxation
        @test ice_relaxation ≈ expected_ice_relaxation
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
            constants, transport, q,
            cloud.μᶜˡ, cloud.λᶜˡ, cloud.nᶜˡ, FT(0), FT(0))
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
            qᵛ_ad = qᵛ⁺ˡ_ad  # exactly saturated → supersaturation = 0
            q_ad = MoistureMassFractions(qᵛ_ad, qᶜˡ + qʳ + qʷⁱ, zero(FT))
            transport_ad = air_transport_properties(T_ad, P, constants)
            cloud_ad = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
            cᵖᵐ_ad = mixture_heat_capacity(q_ad, constants)
            temperature_tendency = -constants.gravitational_acceleration / cᵖᵐ_ad
            rates_cooling = PPP.coupled_saturation_adjustment_rates(
                p3, qᶜˡ, nᶜˡ, qʳ, nʳ, zero(FT), zero(FT), zero(FT),
                qᵛ_ad, qᵛ⁺ˡ_ad, qᵛ⁺ⁱ_ad, Fᶠ, ρᶠ, T_ad, P, ρ,
                constants, transport_ad, q_ad,
                cloud_ad.μᶜˡ, cloud_ad.λᶜˡ, cloud_ad.nᶜˡ,
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
                      constants, transport_s, q_s,
                      cloud_s.μᶜˡ, cloud_s.λᶜˡ, cloud_s.nᶜˡ)
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
        qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
        qᵛ = qᵛ⁺ˡ + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ + qʷⁱ, qⁱ)
        transport = air_transport_properties(T, P, constants)

        # Fˡ ≈ 0.33 here, so the wet-ice gate is the active one and the raw
        # relaxation coefficient is the wet-ice one.
        coating_relaxation = PPP.ice_vapor_relaxation_coefficient(
            p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ, constants, transport, q)
        expected_coating_relaxation = expected_reference_coating_vapor_relaxation(
            p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ, constants, transport, q)

        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
        rates = PPP.coupled_saturation_adjustment_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q,
            cloud.μᶜˡ, cloud.λᶜˡ, cloud.nᶜˡ, FT(0), FT(0))
        expected_rates = expected_reduced_reference_vapor_rates(
            p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
            qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
            constants, transport, q)

        @test coating_relaxation > 0
        @test coating_relaxation ≈ expected_coating_relaxation

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
        # the post-liquid thermodynamic state.
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
        # than by the prescribed-Nᶜˡ target mass.
        qᵛ = qᵛ⁺ˡ + FT(1e-6)
        qᶜˡ = zero(FT)
        q = MoistureMassFractions(qᵛ, qᶜˡ, zero(FT))
        Nᶜˡ = p3.cloud.number_concentration

        ccn = PPP.compute_ccn_activation(p3.aerosol, p3, qᶜˡ, zero(FT), zero(FT),
                                         qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜˡ, constants)

        Rᵛ = Breeze.Thermodynamics.vapor_gas_constant(constants)
        ℒˡ = Breeze.Thermodynamics.liquid_latent_heat(T, constants)
        ξˡ = PPP.liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)
        cons7 = FT(4 * FT(π) / 3 * 1000 * (1e-6)^3)
        deficit = Nᶜˡ / ρ * cons7
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
        # Above freezing: positive melting
        T_warm = FT(275.15)    # +2C
        rate_warm = ice_melting_rate(p3, qi, ni, FT(0), T_warm, P, qv, qv_sat, Ff, ρf, ρ,
                                     constants, air_transport_properties(T_warm, P, constants))
        @test rate_warm > 0

        # Below freezing: zero melting
        T_cold = FT(263.15)    # -10C
        rate_cold = ice_melting_rate(p3, qi, ni, FT(0), T_cold, P, qv, qv_sat, Ff, ρf, ρ,
                                     constants, air_transport_properties(T_cold, P, constants))
        @test rate_cold == 0

        # Exactly at freezing: zero (no ΔT to drive melting)
        T_freeze = FT(273.15)
        rate_freeze = ice_melting_rate(p3, qi, ni, FT(0), T_freeze, P, qv, qv_sat, Ff, ρf, ρ,
                                       constants, air_transport_properties(T_freeze, P, constants))
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
        rates_dry = ice_melting_rates(p3, qi, ni, qwi_zero, T, P, qv, qv_sat, Ff, ρf, ρ,
                                      constants, air_transport_properties(T, P, constants))
        total = rates_dry.partial_melting + rates_dry.complete_melting
        @test total > 0
        @test rates_dry.partial_melting >= 0
        @test rates_dry.complete_melting >= 0

        # With the tabulated integrals, the partial/complete split depends on the
        # PSD-integrated ventilation. Verify both branches are non-negative
        # and at least one is positive.
        @test rates_dry.complete_melting >= 0

        # Saturated liquid coating: more complete melting (or approximately equal)
        qwi_high = FT(0.5 * qi)   # 50% liquid fraction
        rates_wet = ice_melting_rates(p3, qi, ni, qwi_high, T, P, qv, qv_sat, Ff, ρf, ρ,
                                      constants, air_transport_properties(T, P, constants))
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
            m_mean, Ff, zero(FT), ρf, p3.process_rates, transport.ν, transport.Dᵛ,
            ρ_correction, p3)

        capacity = PPP.wet_growth_capacity(p3, qi, qwi, ni, T, P, qv, Ff, ρf,
                                           ρ, constants, transport)
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
            FT(0),
            cloud.Nᶜˡ,
            cloud.nᶜˡ,
            cloud.μᶜˡ,
            cloud.λᶜˡ,
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

        rates = PPP.p3_phase2_rates(p3, ρ, ℳ, constants, state, phase1)

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
            FT(0),
            cloud.Nᶜˡ,
            cloud.nᶜˡ,
            cloud.μᶜˡ,
            cloud.λᶜˡ,
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

        rates = PPP.p3_phase2_rates(p3, ρ, ℳ, constants, state, phase1)

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
        transport = air_transport_properties(T, P, constants)

        T₀ = p3.process_rates.freezing_temperature
        Rᵥ = Breeze.Thermodynamics.vapor_gas_constant(constants)
        Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
        ε = Rᵈ / Rᵥ
        e_s0 = PPP.saturation_vapor_pressure_at_freezing(constants, T₀)
        # M10: set qv = q_sat0 (mixing ratio convention) so latent term vanishes
        qv = ε * e_s0 / max(P - e_s0, FT(1))

        refreezing = PPP.refreezing_rate(p3, qwi, qi, ni, T, P, qv, Ff, ρf,
                                         ρ, constants, transport)

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
        rate_warm = ice_aggregation_rate(p3, qi, ni, T_warm, Ff, ρf, ρ)
        @test rate_warm > 0     # Positive magnitude (M7)

        # Very cold (T < 253.15 K): much less aggregation
        T_cold = FT(233.15)    # -40C
        rate_cold = ice_aggregation_rate(p3, qi, ni, T_cold, Ff, ρf, ρ)
        # Aggregation efficiency at very cold T is 0.001 vs ~0.15 at -5C
        @test rate_cold < rate_warm

        # Zero ice: zero aggregation
        rate_noice = ice_aggregation_rate(p3, FT(0), FT(0), T_warm, Ff, ρf, ρ)
        @test rate_noice == 0

        # Heavily rimed (Ff > 0.9): aggregation shuts off
        rate_rimed = ice_aggregation_rate(p3, qi, ni, T_warm, FT(0.95), ρf, ρ)
        @test rate_rimed == 0

        # Rate scales with ρ × rhofaci where rhofaci = (ρ₀/ρ)^0.54 (M11).
        # Combined scaling: rate ∝ ρ × (ρ₀/ρ)^0.54 = ρ₀^0.54 × ρ^0.46
        ρ_half = FT(0.5)
        rate_half_ρ = ice_aggregation_rate(p3, qi, ni, T_warm, Ff, ρf, ρ_half)
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
        # Below freezing with cloud and ice: positive riming
        T_cold = FT(263.15)    # -10C
        rate = cloud_riming_rate(p3, qc, qi, ni, T_cold, Ff, ρf, ρ)
        @test rate > 0

        # Above freezing: zero riming
        T_warm = FT(278.15)
        rate_warm = cloud_riming_rate(p3, qc, qi, ni, T_warm, Ff, ρf, ρ)
        @test rate_warm == 0

        # Zero cloud: zero riming
        rate_nocloud = cloud_riming_rate(p3, FT(0), qi, ni, T_cold, Ff, ρf, ρ)
        @test rate_nocloud == 0

        # Zero ice: zero riming
        rate_noice = cloud_riming_rate(p3, qc, FT(0), FT(0), T_cold, Ff, ρf, ρ)
        @test rate_noice == 0

        # More cloud water gives faster riming (rate is linear in qc)
        rate_high = cloud_riming_rate(p3, FT(2e-3), qi, ni, T_cold, Ff, ρf, ρ)
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

    @testset "rime_density follows the Cober-List Ri fit" begin
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

        parameters = p3.process_rates
        μᶜˡ = p3.cloud.shape_parameter
        Nᶜˡ = p3.cloud.number_concentration
        ρ_water = parameters.liquid_water_density

        qcl_abs = qcl * ρ
        η = transport.ν * ρ
        unbounded_cloud_slope = cbrt(
            FT(π) * ρ_water * Nᶜˡ * (μᶜˡ + 3) * (μᶜˡ + 2) * (μᶜˡ + 1) /
            (FT(6) * qcl_abs)
        )
        λᶜˡ = clamp(unbounded_cloud_slope, (μᶜˡ + 1) * FT(2.5e4),
                    (μᶜˡ + 1) * FT(1e6))
        ρ_rime = rime_density(p3, qcl, cloud_rim, T, vᵢ, ρ, constants,
                              transport, μᶜˡ, λᶜˡ)
        a_cn = constants.gravitational_acceleration * ρ_water / (FT(18) * η)
        Vt_qc = a_cn * (μᶜˡ + 5) * (μᶜˡ + 4) / λᶜˡ^2
        Dᶜˡ = (μᶜˡ + 4) / λᶜˡ
        inverse_supercooling = inv(min(FT(-0.001), T - parameters.freezing_temperature))
        Ri = clamp(-(FT(0.5e6) * Dᶜˡ) * abs(vᵢ - Vt_qc) * inverse_supercooling, FT(1), FT(12))
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
                              transport_warm, μᶜˡ, λᶜˡ)
        @test ρ_warm == 400

        ρ_no_cloud = rime_density(p3, qcl, FT(0), T, vᵢ, ρ, constants,
                                  transport, μᶜˡ, λᶜˡ)
        @test ρ_no_cloud == 400
    end

    @testset "Rime consistency enforcement" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        parameters = p3.process_rates

        function reference_bulk_rime_density(qⁱ, qᶠ, bᶠ)
            qᶠ = max(qᶠ, 0)
            bᶠ = max(bᶠ, 0)
            ρᶠ = FT(NaN)

            # Mirrors `consistent_rime_state`: the rime-volume floor is
            # `minimum_mass_mixing_ratio / maximum_rime_density`.
            if bᶠ >= p3.minimum_mass_mixing_ratio / parameters.maximum_rime_density
                ρᶠ = qᶠ / bᶠ
                if ρᶠ < parameters.minimum_rime_density
                    ρᶠ = parameters.minimum_rime_density
                    bᶠ = qᶠ / ρᶠ
                elseif ρᶠ > parameters.maximum_rime_density
                    ρᶠ = parameters.maximum_rime_density
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

        bsmall = p3.minimum_mass_mixing_ratio / parameters.maximum_rime_density
        @test bsmall ≈ FT(1e-14) / FT(900)

        # Below `bsmall` no admissible density leaves significant rime: drop the pair.
        no_volume = consistent_rime_state(p3, FT(1e-4), FT(1e-5), FT(1e-18), FT(0))
        @test no_volume.qᶠ == 0
        @test no_volume.bᶠ == 0
        @test no_volume.ρᶠ == 0
        @test no_volume.Fᶠ == 0

        # Above it, repair instead: the implied density overshoots `maximum_rime_density`,
        # so re-densify to it and recompute `bᶠ` rather than discarding the rime.
        dense_rime = consistent_rime_state(p3, FT(1e-4), FT(1e-5), FT(1e-16), FT(0))
        @test FT(1e-16) > bsmall
        @test dense_rime.qᶠ == FT(1e-5)
        @test dense_rime.ρᶠ == parameters.maximum_rime_density
        @test dense_rime.bᶠ ≈ dense_rime.qᶠ / parameters.maximum_rime_density
        @test dense_rime.Fᶠ ≈ FT(0.1)

        tiny_rime = consistent_rime_state(p3, FT(1e-4), FT(5e-15), FT(1e-15), FT(0))
        @test tiny_rime.qᶠ == 0
        @test tiny_rime.bᶠ == 0

        low_density = consistent_rime_state(p3, FT(1e-4), FT(2e-5), FT(2e-6), FT(0))
        @test low_density.ρᶠ == parameters.minimum_rime_density
        @test low_density.bᶠ ≈ low_density.qᶠ / parameters.minimum_rime_density

        high_density = consistent_rime_state(p3, FT(1e-4), FT(2e-5), FT(2e-8), FT(0))
        @test high_density.ρᶠ == parameters.maximum_rime_density
        @test high_density.bᶠ ≈ high_density.qᶠ / parameters.maximum_rime_density

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
            ρsᵛ⁺ˡ = FT(0),
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
        FT(0),         # sᵛ⁺ˡ (liquid supersaturation)
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

        # Physically this is one signed rate: a base self-collection rate reduced by
        # the Verlinde-Cotton breakup modifier, never rescaled by a limiter.
        # Breeze reports the sink and source directions
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

        # Large drops: Dʳ = (qʳ / (π ρᴸ nʳ))^(1/3) ≈ 680 μm exceeds the 280 μm
        # threshold, so dum < 0, breakup outruns self-collection, and the netted pair
        # must report a pure source.
        qʳ_large, nʳ_large = FT(1e-3), FT(1e3)
        base_large = rain_self_collection_rate(p3, qʳ_large, nʳ_large, ρ)
        breakup_large = rain_breakup_rate(p3, qʳ_large, nʳ_large, base_large)
        @test breakup_large > base_large > 0

        large = rain_only_rates(qʳ_large, nʳ_large)
        @test large.rain_self_collection == 0
        @test large.rain_breakup ≈ breakup_large - base_large rtol=FT(1e-10)

        # Small drops: Dʳ ≈ 68 μm, breakup inactive, pure sink.
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
        sᵛ⁺ˡ = FT(0)

        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ,
                                qᶠ, qᶠ / FT(400), qʷⁱ, sᵛ⁺ˡ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
        expected_cloud_number_per_mass = cloud.Nᶜˡ / (ρ * qᶜˡ)

        @test rates.cloud_warm_collection > 0
        @test rates.cloud_warm_collection_number / rates.cloud_warm_collection ≈
              expected_cloud_number_per_mass rtol=FT(1e-12)

        manual_rates = p3_process_rates_with(FT;
            cloud_warm_collection = FT(1e-8),
            cloud_warm_collection_number = FT(1e4),
        )
        # The shed-drop count follows the configurable `shed_drop_mass` (default the
        # mass of a 1 mm drop), not a hardcoded literal: the rain-number limiter
        # budgets this source as `cloud_warm_collection / shed_drop_mass`, so the
        # assembled tendency has to divide by the same mass. Read it from the scheme
        # rather than restating it, so retuning the default cannot desync the test.
        expected_shed_drop_source = ρ * manual_rates.cloud_warm_collection /
                                    p3.process_rates.shed_drop_mass
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

        # Wet growth is only diagnosed with liquid fraction active, so
        # `wet_growth_cloud`/`wet_growth_rain` are identically zero in this branch.
        # Even when forced nonzero they must not feed the ice, rime-mass or
        # rime-volume tendencies: the water they collect stays liquid, raising the
        # total ice mass and the coating mass equally and leaving the dry ice mass
        # unchanged.
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
        # shrinks cloud_riming/rain_riming to the retained portion). Rime volume splits
        # the two between the fresh-rime density and the maximum rime density.
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
        sᵛ⁺ˡ = FT(0)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        q = MoistureMassFractions(qᵛ, qʳ + qʷⁱ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ,
                                qᶠ, qᶠ / FT(400), qʷⁱ, sᵛ⁺ˡ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        rime = consistent_rime_state(p3, qⁱ, qᶠ, qᶠ / FT(400), qʷⁱ)
        expected_number = expected_reference_warm_rain_collection_number(p3, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
                                                                       T, rime.Fᶠ, rime.ρᶠ, ρ)
        expected_mass = PPP.rain_warm_collection_rate(p3, qʳ, nʳ, qⁱ, nⁱ,
                                                      T, rime.Fᶠ, rime.ρᶠ, ρ, qʷⁱ)
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
        sᵛ⁺ˡ = FT(0)

        qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
        qᵛ = qᵛ⁺ˡ + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ,
                                qᶠ, FT(0), qʷⁱ, sᵛ⁺ˡ, zero(FT), zero(FT))

        gm = expected_reference_supersaturation_adjustment(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ,
                                                               sᵛ⁺ˡ, T, constants)
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
            constants, transportᴳᴹ, qᴳᴹ)

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)

        @test rates.predicted_supersaturation_adjustment ≈ gm.rate
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
        sᵛ⁺ˡ = FT(0)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface()) + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, FT(2e8), qʳ, FT(0), qⁱ, FT(0),
                                qᶠ, FT(0), qʷⁱ, sᵛ⁺ˡ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        expected = actual_final_liquid_ssat_after_p3_step(𝒰, rates, qᵛ, qᶜˡ, qʳ, qⁱ, qʷⁱ,
                                                          ρ, process_rates.sink_limiting_timescale,
                                                          P, pˢᵗ, constants)

        @test tendency_ρsᵛ⁺ˡ(rates, ρ, p3.process_rates) / ρ *
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
        sᵛ⁺ˡ = FT(0)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface()) + FT(1e-5)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ + qʷⁱ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, FT(2e8), qʳ, FT(0), qⁱ, nⁱ,
                                 qᶠ, qᶠ / FT(400), qʷⁱ, sᵛ⁺ˡ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        expected = actual_final_liquid_ssat_after_p3_step(𝒰, rates, qᵛ, qᶜˡ, qʳ, qⁱ, qʷⁱ,
                                                          ρ, process_rates.sink_limiting_timescale,
                                                          P, pˢᵗ, constants)

        @test rates.splintering_mass > 0
        @test tendency_ρsᵛ⁺ˡ(rates, ρ, p3.process_rates) / ρ *
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
        sᵛ⁺ˡ = FT(0)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface()) + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
        𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, pˢᵗ, P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, FT(2e8), qʳ, FT(0), qⁱ, FT(0),
                                qᶠ, FT(0), qʷⁱ, sᵛ⁺ˡ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        expected = actual_final_liquid_ssat_after_p3_step(𝒰, rates, qᵛ, qᶜˡ, qʳ, qⁱ, qʷⁱ,
                                                          ρ, τ, P, pˢᵗ, constants)

        @test tendency_ρsᵛ⁺ˡ(rates, ρ, p3.process_rates) / ρ * τ ≈ expected atol=FT(1e-12)
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
        sᵛ⁺ˡ = FT(0)
        qᵛ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface()) + FT(1e-4)
        q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
        𝒰 = with_temperature(StaticEnergyState(zero(FT), q, FT(0), P), T, constants)
        ℳ = P3MicrophysicalState(qᶜˡ, FT(2e8), qʳ, FT(0), qⁱ, FT(0),
                                qᶠ, FT(0), qʷⁱ, sᵛ⁺ˡ, zero(FT), zero(FT))

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        expected = actual_final_liquid_ssat_after_p3_step(𝒰, rates, qᵛ, qᶜˡ, qʳ, qⁱ, qʷⁱ,
                                                          ρ, τ, P, pˢᵗ, constants)

        @test tendency_ρsᵛ⁺ˡ(rates, ρ, p3.process_rates) / ρ * τ ≈ expected atol=FT(1e-12)
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
        FT(0),         # sᵛ⁺ˡ
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
            FT(0),          # sᵛ⁺ˡ
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

        # Load the P3 lookup tables
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
