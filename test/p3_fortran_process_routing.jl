using Test
import Breeze

using Breeze.Microphysics.PredictedParticleProperties:
    P3MicrophysicalState,
    P3ProcessRates,
    PredictedParticlePropertiesMicrophysics,
    ProcessRateParameters,
    compute_p3_process_rates,
    immersion_freezing_cloud_rate,
    immersion_freezing_rain_rate,
    liu_daum_shape_parameter,
    psd_correction_spherical_volume,
    tendency_ρnʳ,
    tendency_ρnⁱ,
    tendency_ρqᶜˡ,
    tendency_ρqʳ,
    tendency_ρqⁱ,
    tendency_ρqᶠ,
    tendency_ρbᶠ,
    tendency_ρqʷⁱ,
    tendency_ρqᵛ

using Breeze.Thermodynamics:
    LiquidIcePotentialTemperatureState,
    MoistureMassFractions,
    PlanarIceSurface,
    PlanarLiquidSurface,
    ThermodynamicConstants,
    mixture_heat_capacity,
    saturation_specific_humidity,
    temperature,
    with_temperature

const P3Routing = Breeze.Microphysics.PredictedParticleProperties

function routing_thermodynamic_state(air_temperature, air_density, pressure,
                                     vapor, cloud, rain, dry_ice, coating,
                                     constants)
    FT = typeof(air_temperature)
    moisture = MoistureMassFractions(vapor, cloud + rain + coating, dry_ice)
    state = LiquidIcePotentialTemperatureState(zero(FT), moisture, FT(1e5), pressure)
    return with_temperature(state, air_temperature, constants)
end

function routing_derived_state(p3, air_density, microphysical_state,
                               thermodynamic_state, constants)
    FT = typeof(air_density)
    props = P3Routing.p3_ice_properties(
        p3, air_density, microphysical_state, thermodynamic_state, constants)
    air_temperature = temperature(thermodynamic_state, constants)
    pressure = thermodynamic_state.reference_pressure
    moisture = thermodynamic_state.moisture_mass_fractions
    vapor = moisture.vapor
    vapor_saturation_liquid = saturation_specific_humidity(
        air_temperature, air_density, constants, PlanarLiquidSurface())
    vapor_saturation_ice = saturation_specific_humidity(
        air_temperature, air_density, constants, PlanarIceSurface())
    transport = P3Routing.air_transport_properties(air_temperature, pressure)
    cloud = P3Routing.diagnose_cloud_dsd(
        p3, microphysical_state.qᶜˡ, microphysical_state.nᶜˡ, air_density)

    state = P3Routing.P3DerivedState{FT, typeof(moisture)}(
        props.nⁱ,
        microphysical_state.nʳ,
        props.qᶠ,
        props.bᶠ,
        props.Fᶠ,
        props.ρᶠ,
        props.μ_ice,
        props.Fˡ,
        props.Nᶜ,
        cloud.nᶜˡ,
        cloud.μ_c,
        cloud.λ_c,
        air_temperature,
        pressure,
        vapor,
        vapor_saturation_liquid,
        vapor_saturation_ice,
        moisture,
        transport.D_v,
        transport.K_a,
        transport.nu,
        mixture_heat_capacity(moisture, constants),
    )

    return props, state
end

@testset "P3 Fortran process routing regressions" begin
    @testset "near-liquid whole-particle cleanup transfers existing number" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        constants = ThermodynamicConstants(FT)
        τ = p3.process_rates.sink_limiting_timescale
        air_density = one(FT)
        air_temperature = FT(268.15)
        pressure = FT(8e4)
        dry_ice = zero(FT)
        coating = FT(1e-4)
        ice_number = FT(2e4)
        vapor = saturation_specific_humidity(
            air_temperature, air_density, constants, PlanarLiquidSurface())
        thermodynamic_state = routing_thermodynamic_state(
            air_temperature, air_density, pressure, vapor,
            zero(FT), zero(FT), dry_ice, coating, constants)
        microphysical_state = P3MicrophysicalState(
            zero(FT), zero(FT), zero(FT), zero(FT),
            dry_ice, ice_number, zero(FT), zero(FT), zero(FT), coating, zero(FT))

        rates = compute_p3_process_rates(
            p3, air_density, microphysical_state, thermodynamic_state, constants)

        @test rates.complete_melting == 0
        @test rates.shedding ≈ coating / τ
        @test rates.melting_number ≈ ice_number / τ
        @test tendency_ρnⁱ(rates, air_density) ≈ -air_density * ice_number / τ
        @test tendency_ρnʳ(
            rates, air_density, ice_number, dry_ice, zero(FT), one(FT), p3) ≈
              air_density * ice_number / τ
        @test tendency_ρqʳ(rates, air_density, p3.process_rates) +
              tendency_ρqʷⁱ(rates, air_density, p3.process_rates) ≈ 0 atol=eps(FT)
    end

    @testset "wet-ice vapor exchange uses total particle mass" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        constants = ThermodynamicConstants(FT)
        air_density = one(FT)
        air_temperature = FT(272.15)
        pressure = FT(8e4)
        dry_ice = FT(1e-22)
        coating = FT(1e-4)
        ice_number = FT(2e4)
        vapor_saturation = saturation_specific_humidity(
            air_temperature, air_density, constants, PlanarLiquidSurface())
        vapor = vapor_saturation + FT(1e-4)
        thermodynamic_state = routing_thermodynamic_state(
            air_temperature, air_density, pressure, vapor,
            zero(FT), zero(FT), dry_ice, coating, constants)
        microphysical_state = P3MicrophysicalState(
            zero(FT), zero(FT), zero(FT), zero(FT),
            dry_ice, ice_number, zero(FT), zero(FT), zero(FT), coating, zero(FT))
        _, state = routing_derived_state(
            p3, air_density, microphysical_state, thermodynamic_state, constants)

        phase1 = P3Routing._p3_phase1_rates(
            p3, air_density, microphysical_state, constants, state,
            zero(FT), zero(FT))

        @test dry_ice < FT(1e-20)
        @test dry_ice + coating >= p3.minimum_mass_mixing_ratio
        @test phase1.deposition == 0
        @test phase1.coating_condensation > 0
        @test phase1.coating_evaporation == 0
    end

    @testset "post-process whole-particle clip removes rime companions exactly" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        constants = ThermodynamicConstants(FT)
        τ = p3.process_rates.sink_limiting_timescale
        air_density = one(FT)
        air_temperature = FT(272)
        pressure = FT(8e4)
        cloud = FT(1e-8)
        cloud_number = FT(1e8)
        dry_ice = FT(1.1e-10)
        coating = FT(9.89e-9)
        ice_number = FT(2e4)
        rime_mass = FT(0.5) * dry_ice
        rime_volume = rime_mass / FT(400)
        sixth_moment = FT(1e-10)
        vapor = saturation_specific_humidity(
            air_temperature, air_density, constants, PlanarLiquidSurface()) +
                FT(0.01)
        thermodynamic_state = routing_thermodynamic_state(
            air_temperature, air_density, pressure, vapor,
            cloud, zero(FT), dry_ice, coating, constants)
        microphysical_state = P3MicrophysicalState(
            cloud, cloud_number, zero(FT), zero(FT),
            dry_ice, ice_number, rime_mass, rime_volume,
            sixth_moment, coating, zero(FT))

        rates = compute_p3_process_rates(
            p3, air_density, microphysical_state, thermodynamic_state, constants)
        props = P3Routing.p3_ice_properties(
            p3, air_density, microphysical_state, thermodynamic_state, constants)

        @test rates.post_process_clipping == 1
        @test rates.refreezing ≈ FT(9.89e-10)
        @test rates.clipping_dry_mass ≈ FT(1e-9)
        @test rates.clipping_rime_mass ≈ FT(9.945e-10)
        @test rates.clipping_rime_volume ≈ FT(1.112638888888889e-12)

        dry_ice_after = dry_ice +
            τ * tendency_ρqⁱ(rates, air_density) / air_density
        coating_after = coating +
            τ * tendency_ρqʷⁱ(rates, air_density, p3.process_rates) / air_density
        rime_mass_after = rime_mass +
            τ * tendency_ρqᶠ(
                rates, air_density, props.Fᶠ, p3.process_rates) / air_density
        rime_volume_after = rime_volume +
            τ * tendency_ρbᶠ(
                rates, air_density, props.Fᶠ, props.ρᶠ, dry_ice,
                p3.process_rates) / air_density
        total_water_tendency =
            tendency_ρqᵛ(rates, air_density) +
            tendency_ρqᶜˡ(rates, air_density) +
            tendency_ρqʳ(rates, air_density, p3.process_rates) +
            tendency_ρqⁱ(rates, air_density) +
            tendency_ρqʷⁱ(rates, air_density, p3.process_rates)

        @test dry_ice_after ≈ 0 atol=FT(1e-20)
        @test coating_after ≈ 0 atol=FT(1e-20)
        @test rime_mass_after ≈ 0 atol=FT(1e-20)
        @test rime_volume_after ≈ 0 atol=FT(1e-22)
        @test total_water_tendency ≈ 0 atol=eps(FT)
    end

    @testset "immersion-freezing budgets preserve frozen-particle mass" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        prp = p3.process_rates
        τ = prp.sink_limiting_timescale
        air_temperature = FT(136)
        air_density = one(FT)

        cloud_mass = FT(1e-3)
        cloud_number_density = FT(1e8)
        cloud_mass_rate, cloud_number_rate = immersion_freezing_cloud_rate(
            p3, cloud_mass, cloud_number_density, air_temperature, air_density)
        cloud_number = cloud_number_density / air_density
        cloud_shape = liu_daum_shape_parameter(cloud_number_density)
        cloud_correction = psd_correction_spherical_volume(cloud_shape)
        expected_cloud_particle_mass = cloud_correction * cloud_mass / cloud_number

        @test cloud_mass_rate > cloud_mass / τ
        @test cloud_mass_rate / cloud_number_rate ≈ expected_cloud_particle_mass

        pressure = FT(8e4)
        cloud_vapor = saturation_specific_humidity(
            air_temperature, air_density, ThermodynamicConstants(FT),
            PlanarLiquidSurface())
        cloud_thermodynamic_state = routing_thermodynamic_state(
            air_temperature, air_density, pressure, cloud_vapor,
            cloud_mass, zero(FT), zero(FT), zero(FT),
            ThermodynamicConstants(FT))
        cloud_state = P3MicrophysicalState(
            cloud_mass, cloud_number, zero(FT), zero(FT),
            zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))
        cloud_rates = compute_p3_process_rates(
            p3, air_density, cloud_state, cloud_thermodynamic_state,
            ThermodynamicConstants(FT))

        @test cloud_rates.cloud_freezing_mass > 0
        @test cloud_rates.cloud_freezing_mass < cloud_mass_rate
        @test cloud_rates.cloud_freezing_mass /
              cloud_rates.cloud_freezing_number ≈ expected_cloud_particle_mass
        @test cloud_rates.cloud_freezing_mass +
              cloud_rates.cloud_homogeneous_mass <= cloud_mass / τ

        rain_mass = FT(1e-3)
        rain_number = FT(1e4)
        rain_shape = zero(FT)
        rain_mass_rate, rain_number_rate = immersion_freezing_rain_rate(
            p3, rain_mass, rain_number, air_temperature, rain_shape)
        rain_correction = psd_correction_spherical_volume(rain_shape)
        expected_rain_particle_mass = rain_correction * rain_mass / rain_number

        @test rain_mass_rate > rain_mass / τ
        @test rain_mass_rate / rain_number_rate ≈ expected_rain_particle_mass

        rain_vapor = saturation_specific_humidity(
            air_temperature, air_density, ThermodynamicConstants(FT),
            PlanarLiquidSurface())
        rain_thermodynamic_state = routing_thermodynamic_state(
            air_temperature, air_density, pressure, rain_vapor,
            zero(FT), rain_mass, zero(FT), zero(FT),
            ThermodynamicConstants(FT))
        rain_state = P3MicrophysicalState(
            zero(FT), zero(FT), rain_mass, rain_number,
            zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))
        rain_rates = compute_p3_process_rates(
            p3, air_density, rain_state, rain_thermodynamic_state,
            ThermodynamicConstants(FT))

        @test rain_rates.rain_freezing_mass > 0
        @test rain_rates.rain_freezing_mass < rain_mass_rate
        @test rain_rates.rain_freezing_mass /
              rain_rates.rain_freezing_number ≈ expected_rain_particle_mass
        @test rain_rates.rain_freezing_mass +
              rain_rates.rain_homogeneous_mass <= rain_mass / τ
    end

    @testset "Hallett–Mossop diameter is volume-equivalent" begin
        FT = Float64
        p3 = PredictedParticlePropertiesMicrophysics(FT; three_moment_ice = true)
        constants = ThermodynamicConstants(FT)
        air_density = one(FT)
        air_temperature = p3.process_rates.splintering_temperature_peak
        pressure = FT(8e4)
        dry_ice = FT(2e-4)
        coating = FT(2e-5)
        cloud = FT(1e-4)
        cloud_number = FT(2e8)
        ice_number = FT(1e-2)
        rime_mass = FT(1e-4)
        rime_volume = rime_mass / FT(400)
        sixth_moment = FT(1e-20)
        vapor = saturation_specific_humidity(
            air_temperature, air_density, constants, PlanarLiquidSurface())
        thermodynamic_state = routing_thermodynamic_state(
            air_temperature, air_density, pressure, vapor,
            cloud, zero(FT), dry_ice, coating, constants)
        microphysical_state = P3MicrophysicalState(
            cloud, cloud_number, zero(FT), zero(FT),
            dry_ice, ice_number, rime_mass, rime_volume,
            sixth_moment, coating, zero(FT))
        props, state = routing_derived_state(
            p3, air_density, microphysical_state, thermodynamic_state, constants)
        phase1 = P3Routing.P3Phase1Rates{FT}(
            ntuple(_ -> zero(FT), fieldcount(P3Routing.P3Phase1Rates{FT}))...)

        phase2 = P3Routing._p3_phase2_rates(
            p3, air_density, microphysical_state, constants, state, phase1, FT(280))
        phase2_warm_surface = P3Routing._p3_phase2_rates(
            p3, air_density, microphysical_state, constants, state, phase1, FT(285))
        expected_diameter = cbrt(
            6 * props.qⁱ_total /
            (FT(π) * props.ρ_mean * props.nⁱ_diagnostic))
        mass_law_diameter = first(P3Routing.mean_ice_particle_diameter(
            props.qⁱ_total / props.nⁱ, props.Fᶠ, props.Fˡ,
            props.ρᶠ, p3.process_rates))

        @test props.nⁱ != props.nⁱ_diagnostic
        @test phase2.D_mean ≈ expected_diameter rtol=FT(1e-12)
        @test !isapprox(phase2.D_mean, mass_law_diameter; rtol=FT(1e-3))
        @test phase2.splintering_number > 0
        @test phase2_warm_surface.splintering_number == 0
    end

    @testset "non-liquid-fraction mode uses the Fortran fast routing" begin
        FT = Float64
        process_rates = ProcessRateParameters(FT; liquid_fraction_active = false)
        p3 = PredictedParticlePropertiesMicrophysics(FT; process_rates)
        constants = ThermodynamicConstants(FT)
        τ = process_rates.sink_limiting_timescale

        # qsmall_dry defaults to 1e-12 regardless of moment mode or liquid-fraction setting.
        @test p3.process_rates.qsmall_dry == FT(1e-12)
        p3_three_moment = PredictedParticlePropertiesMicrophysics(
            FT; three_moment_ice = true,
            process_rates = ProcessRateParameters(
                FT; liquid_fraction_active = false))
        @test p3_three_moment.process_rates.qsmall_dry == FT(1e-12)
        overridden_qsmall = FT(3e-9)
        p3_overridden = PredictedParticlePropertiesMicrophysics(
            FT; three_moment_ice = true,
            process_rates = ProcessRateParameters(
                FT; liquid_fraction_active = false,
                qsmall_dry = overridden_qsmall))
        @test p3_overridden.process_rates.qsmall_dry == overridden_qsmall

        mixed_precision = PredictedParticlePropertiesMicrophysics(
            Float64; process_rates = ProcessRateParameters(
                Float32; liquid_fraction_active = false))
        @test mixed_precision.process_rates.qsmall_dry isa Float32
        @test mixed_precision.process_rates.qsmall_dry == Float32(1e-12)

        wet_growth_temperature = process_rates.freezing_temperature - FT(1e-3)
        wet_growth_vapor = FT(0.02)
        wet_growth_cloud = FT(1e-3)
        wet_growth_cloud_number = FT(3e8)
        wet_growth_rain = FT(1e-3)
        wet_growth_rain_number = FT(1e4)
        wet_growth_ice = FT(1e-4)
        wet_growth_ice_number = FT(1e3)
        wet_growth_rime = FT(1e-5)
        wet_growth_state = P3MicrophysicalState(
            wet_growth_cloud, wet_growth_cloud_number,
            wet_growth_rain, wet_growth_rain_number,
            wet_growth_ice, wet_growth_ice_number,
            wet_growth_rime, wet_growth_rime / FT(400),
            FT(1e-10), zero(FT), zero(FT))
        wet_growth_thermodynamic_state = routing_thermodynamic_state(
            wet_growth_temperature, one(FT), FT(85000), wet_growth_vapor,
            wet_growth_cloud, wet_growth_rain, wet_growth_ice, zero(FT),
            constants)
        wet_growth_rates = compute_p3_process_rates(
            p3, one(FT), wet_growth_state, wet_growth_thermodynamic_state,
            constants)
        wet_growth_total_water_tendency =
            tendency_ρqᵛ(wet_growth_rates, one(FT)) +
            tendency_ρqᶜˡ(wet_growth_rates, one(FT)) +
            tendency_ρqʳ(wet_growth_rates, one(FT), p3.process_rates) +
            tendency_ρqⁱ(wet_growth_rates, one(FT), p3.process_rates) +
            tendency_ρqʷⁱ(wet_growth_rates, one(FT), p3.process_rates)

        @test wet_growth_rates.cloud_riming > 0
        @test wet_growth_rates.rain_riming > 0
        @test wet_growth_rates.wet_growth_cloud == 0
        @test wet_growth_rates.wet_growth_rain == 0
        @test wet_growth_rates.wet_growth_shedding > 0
        @test tendency_ρqʷⁱ(
            wet_growth_rates, one(FT), p3.process_rates) == 0
        @test wet_growth_total_water_tendency ≈ 0 atol=eps(FT)

        # The fast Fortran branch has no liquid-on-ice reservoir. A nonzero
        # restart value is cleaned into rain, but must not alter the dry-ice
        # particle diagnostics or collection rates while that cleanup occurs.
        inactive_coating = FT(2e-5)
        coated_wet_growth_state = P3MicrophysicalState(
            wet_growth_cloud, wet_growth_cloud_number,
            wet_growth_rain, wet_growth_rain_number,
            wet_growth_ice, wet_growth_ice_number,
            wet_growth_rime, wet_growth_rime / FT(400),
            FT(1e-10), inactive_coating, zero(FT))
        coated_wet_growth_rates = compute_p3_process_rates(
            p3, one(FT), coated_wet_growth_state,
            wet_growth_thermodynamic_state, constants)
        wet_growth_props = P3Routing.p3_ice_properties(
            p3, one(FT), wet_growth_state, wet_growth_thermodynamic_state,
            constants)
        coated_wet_growth_props = P3Routing.p3_ice_properties(
            p3, one(FT), coated_wet_growth_state,
            wet_growth_thermodynamic_state, constants)
        wet_growth_fall_speeds = P3Routing.p3_fall_speed_compute(
            p3, one(FT), wet_growth_state, wet_growth_props)
        coated_wet_growth_fall_speeds = P3Routing.p3_fall_speed_compute(
            p3, one(FT), coated_wet_growth_state, coated_wet_growth_props)

        @test coated_wet_growth_props.qⁱ_total == wet_growth_props.qⁱ_total
        @test coated_wet_growth_props.Fˡ == wet_growth_props.Fˡ == 0
        @test coated_wet_growth_props.μ_ice == wet_growth_props.μ_ice
        @test coated_wet_growth_fall_speeds.wⁱ == wet_growth_fall_speeds.wⁱ
        @test coated_wet_growth_fall_speeds.wⁱₙ == wet_growth_fall_speeds.wⁱₙ
        @test coated_wet_growth_fall_speeds.wⁱ_z == wet_growth_fall_speeds.wⁱ_z
        @test coated_wet_growth_rates.cloud_riming ≈ wet_growth_rates.cloud_riming
        @test coated_wet_growth_rates.rain_riming ≈ wet_growth_rates.rain_riming
        @test coated_wet_growth_rates.shedding ≈ inactive_coating / τ
        @test tendency_ρqʷⁱ(
            coated_wet_growth_rates, one(FT), p3.process_rates) ≈
            -inactive_coating / τ

        air_density = one(FT)
        air_temperature = FT(278.15)
        pressure = FT(8e4)
        dry_ice = FT(1e-10)
        coating = FT(2e-5)
        ice_number = FT(2e3)
        vapor = saturation_specific_humidity(
            air_temperature, air_density, constants, PlanarLiquidSurface())
        thermodynamic_state = routing_thermodynamic_state(
            air_temperature, air_density, pressure, vapor,
            zero(FT), zero(FT), dry_ice, coating, constants)
        microphysical_state = P3MicrophysicalState(
            zero(FT), zero(FT), zero(FT), zero(FT),
            dry_ice, ice_number, zero(FT), zero(FT), zero(FT), coating, zero(FT))

        rates = compute_p3_process_rates(
            p3, air_density, microphysical_state, thermodynamic_state, constants)

        @test rates.partial_melting == 0
        @test rates.complete_melting ≈ dry_ice / τ
        @test rates.melting_number ≈ ice_number / τ
        @test rates.coating_condensation == 0
        @test rates.coating_evaporation == 0
        @test rates.shedding ≈ coating / τ
        @test rates.shedding_number ≈ coating / (τ * process_rates.shed_drop_mass)
        @test rates.refreezing == 0
        @test tendency_ρqʷⁱ(rates, air_density, process_rates) ≈ -coating / τ
        @test tendency_ρqʳ(rates, air_density, process_rates) +
              tendency_ρqⁱ(rates, air_density, process_rates) +
              tendency_ρqʷⁱ(rates, air_density, process_rates) ≈ 0
    end
end
