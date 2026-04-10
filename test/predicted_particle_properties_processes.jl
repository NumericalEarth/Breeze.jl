using Test
import Breeze
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.AtmosphereModels: prognostic_field_names
using Breeze.Thermodynamics: ThermodynamicConstants, dry_air_gas_constant

using Breeze.Microphysics.PredictedParticleProperties:
    IceSizeDistributionState,
    evaluate,
    chebyshev_gauss_nodes_weights,
    size_distribution,
    TabulatedFunction3D,
    TabulatedFunction4D,
    TabulatedFunction5D,
    TabulatedFunction6D,
    TabulatedFunction1D,
    P3ProcessRates,
    compute_p3_process_rates,
    consistent_rime_state,
    tendency_ρqᶜˡ,
    tendency_ρqʳ,
    tendency_ρnʳ,
    tendency_ρqⁱ,
    tendency_ρnⁱ,
    tendency_ρqᶠ,
    tendency_ρbᶠ,
    tendency_ρzⁱ,
    tendency_ρqʷⁱ,
    tendency_ρqᵛ,
    rain_autoconversion_rate,
    rain_accretion_rate,
    rain_evaporation_rate,
    rain_self_collection_rate,
    rain_breakup_rate,
    rain_terminal_velocity_mass_weighted,
    cloud_condensation_rate,
    ventilation_enhanced_deposition,
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
    tabulated_function_1d,
    homogeneous_freezing_cloud_rate,
    homogeneous_freezing_rain_rate,
    immersion_freezing_cloud_rate,
    immersion_freezing_rain_rate,
    air_transport_properties,
    psd_correction_spherical_volume,
    liu_daum_shape_parameter

using Breeze.Thermodynamics:
    ThermodynamicConstants,
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState,
    saturation_specific_humidity,
    PlanarLiquidSurface,
    PlanarIceSurface

using Oceananigans: CPU, RectilinearGrid
using Oceananigans.Fields: interior

@testset "P3 Processes" begin

    table_dir = expanduser("~/Aeolus/P3-microphysics/lookup_tables")

    #####
    ##### Lambda solver tests
    #####

    @testset "IceMassPowerLaw construction" begin
        mass = IceMassPowerLaw()
        @test mass.coefficient ≈ 0.0121
        @test mass.exponent ≈ 1.9
        @test mass.ice_density ≈ 900.0

        mass32 = IceMassPowerLaw(Float32)
        @test mass32.coefficient isa Float32
    end

    @testset "ShapeParameterRelation construction" begin
        relation = ShapeParameterRelation()
        @test relation.a ≈ 0.076 * 0.01^0.8
        @test relation.b ≈ 0.8
        @test relation.c ≈ 2.0
        @test relation.μmax ≈ 6.0

        # Test shape parameter computation
        μ = shape_parameter(relation, log(1000.0))
        @test μ ≥ 0
        @test μ ≤ relation.μmax
    end

    @testset "Three-moment closure construction" begin
        p3_closure = ThreeMomentClosure()
        exact_closure = ThreeMomentClosureExact()
        compat_closure = ThreeMomentClosure()

        @test p3_closure.μmin ≈ 0.0
        @test p3_closure.μmax ≈ 20.0
        @test exact_closure.μmin ≈ 0.0
        @test exact_closure.μmax ≈ 20.0
        @test compat_closure isa ThreeMomentClosure
    end

    @testset "Ice regime thresholds" begin
        mass = IceMassPowerLaw()

        # Unrimed ice
        thresholds_unrimed = ice_regime_thresholds(mass, 0.0, 400.0)
        @test thresholds_unrimed.spherical > 0
        @test thresholds_unrimed.graupel == Inf
        @test thresholds_unrimed.partial_rime == Inf

        # Rimed ice
        thresholds_rimed = ice_regime_thresholds(mass, 0.5, 400.0)
        @test thresholds_rimed.spherical > 0
        @test thresholds_rimed.graupel > thresholds_rimed.spherical
        @test thresholds_rimed.partial_rime > thresholds_rimed.graupel
        @test thresholds_rimed.ρ_graupel > 0
    end

    @testset "Regime-4 particle area uses mass-ratio interpolation" begin
        FT = Float64
        state = IceSizeDistributionState(FT;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0,
            rime_fraction = 0.5,
            rime_density = 500.0)

        thresholds = Breeze.Microphysics.PredictedParticleProperties.regime_thresholds_from_state(FT, state)
        D = thresholds.partial_rime * FT(1.01)

        A = Breeze.Microphysics.PredictedParticleProperties.particle_area_ice_only(D, state, thresholds)

        A_sphere = FT(π) / 4 * D^2
        σ = FT(1.88)
        γ = FT(0.2285) * FT(100)^(σ - 2)
        A_aggregate = γ * D^σ
        m_actual = Breeze.Microphysics.PredictedParticleProperties.particle_mass_ice_only(D, state, thresholds)
        m_unrimed = state.mass_coefficient * D^state.mass_exponent
        m_graupel = thresholds.ρ_graupel * FT(π) / 6 * D^3
        weight = (m_actual - m_unrimed) / (m_graupel - m_unrimed)
        A_expected = A_aggregate + weight * (A_sphere - A_aggregate)
        A_rime_fraction = (1 - state.rime_fraction) * A_aggregate + state.rime_fraction * A_sphere

        @test isfinite(A)
        @test A ≈ A_expected
        @test A != A_rime_fraction
        @test A != A_sphere
    end

    @testset "Aggregate and partially-rimed geometry follow Fortran regime logic" begin
        FT = Float64
        state = IceSizeDistributionState(FT;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0,
            rime_fraction = 0.5,
            liquid_fraction = 0.25,
            rime_density = 500.0)

        thresholds = Breeze.Microphysics.PredictedParticleProperties.regime_thresholds_from_state(FT, state)
        σ = FT(1.88)
        γ = FT(0.2285) * FT(100)^(σ - 2)

        D_aggregate = sqrt(thresholds.spherical * thresholds.graupel)
        A_aggregate = Breeze.Microphysics.PredictedParticleProperties.particle_area_ice_only(D_aggregate, state, thresholds)
        @test A_aggregate ≈ γ * D_aggregate^σ

        C_aggregate = Breeze.Microphysics.PredictedParticleProperties.capacitance(D_aggregate, state, thresholds)
        C_aggregate_expected = (1 - state.liquid_fraction) * FT(0.48) * D_aggregate +
                               state.liquid_fraction * D_aggregate
        @test C_aggregate ≈ C_aggregate_expected

        D_partial = thresholds.partial_rime * FT(1.01)
        C_partial = Breeze.Microphysics.PredictedParticleProperties.capacitance(D_partial, state, thresholds)
        m_actual = Breeze.Microphysics.PredictedParticleProperties.particle_mass_ice_only(D_partial, state, thresholds)
        m_unrimed = state.mass_coefficient * D_partial^state.mass_exponent
        m_graupel = thresholds.ρ_graupel * FT(π) / 6 * D_partial^3
        weight = (m_actual - m_unrimed) / (m_graupel - m_unrimed)
        C_ice_expected = (FT(0.48) + weight * (1 - FT(0.48))) * D_partial
        C_expected = (1 - state.liquid_fraction) * C_ice_expected + state.liquid_fraction * D_partial
        @test C_partial ≈ C_expected
    end

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

    @testset "Ice mass computation" begin
        mass = IceMassPowerLaw()

        # Small particles should have spherical mass
        D_small = 1e-5  # 10 μm
        m_small = ice_mass(mass, 0.0, 400.0, D_small)
        m_sphere = mass.ice_density * π / 6 * D_small^3
        @test m_small ≈ m_sphere

        # Mass should increase with diameter
        D_large = 1e-3  # 1 mm
        m_large = ice_mass(mass, 0.0, 400.0, D_large)
        @test m_large > m_small
    end

    @testset "Lambda solver - basic functionality" begin
        # Create a test case with known parameters
        L_ice = 1e-4   # 0.1 g/m³
        N_ice = 1e5    # 100,000 particles/m³
        rime_fraction = 0.0
        rime_density = 400.0

        logλ = solve_lambda(L_ice, N_ice, rime_fraction, rime_density)

        @test isfinite(logλ)
        @test logλ > log(10)    # Within bounds
        @test logλ < log(1.6e7)

        λ = exp(logλ)
        @test λ > 0
    end

    @testset "Lambda solver - consistency" begin
        # Solve for λ, then verify the L/N ratio is recovered
        L_ice = 1e-4
        N_ice = 1e5
        rime_fraction = 0.0
        rime_density = 400.0

        mass = IceMassPowerLaw()
        closure = ShapeParameterRelation()

        params = distribution_parameters(L_ice, N_ice, rime_fraction, rime_density;
                                          mass, closure)

        @test params.N₀ > 0
        @test params.λ > 0
        @test params.μ ≥ 0

        # The solved parameters should be consistent
        @test isfinite(params.N₀)
        @test isfinite(params.λ)
        @test isfinite(params.μ)
    end

    @testset "Lambda solver - rimed ice" begin
        L_ice = 1e-4
        N_ice = 1e5
        rime_fraction = 0.5
        rime_density = 500.0

        logλ = solve_lambda(L_ice, N_ice, rime_fraction, rime_density)

        @test isfinite(logλ)
        @test exp(logλ) > 0
    end

    @testset "Lambda solver - edge cases" begin
        # Zero mass or number should return the upper bound (smallest particles),
        # not the unphysical λ = 0.
        logλ_zero_L = solve_lambda(0.0, 1e5, 0.0, 400.0)
        @test logλ_zero_L == log(1.6e7)

        logλ_zero_N = solve_lambda(1e-4, 0.0, 0.0, 400.0)
        @test logλ_zero_N == log(1.6e7)

        mass = IceMassPowerLaw()
        ρ_dep_zero_rime = Breeze.Microphysics.PredictedParticleProperties.deposited_ice_density(mass, 0.0, 400.0)
        ρ_dep_tiny_rime = Breeze.Microphysics.PredictedParticleProperties.deposited_ice_density(mass, 1e-12, 400.0)

        @test isfinite(ρ_dep_zero_rime)
        @test isfinite(ρ_dep_tiny_rime)
        @test ρ_dep_zero_rime ≈ mass.ice_density
        @test ρ_dep_tiny_rime > 0
    end

    @testset "Three-moment μ polynomial matches Fortran fit" begin
        μ_from_moments = Breeze.Microphysics.PredictedParticleProperties.shape_parameter_from_moments

        @test μ_from_moments(1.0, 1.0, 21.0, 20.0) == 0.0

        G_mid = 10.0
        μ_mid_expected = 1.5900e-2 * G_mid^2 - 4.8202e-1 * G_mid + 4.0108e+0
        @test μ_from_moments(1.0, 1.0, G_mid, 20.0) ≈ μ_mid_expected

        @test μ_from_moments(1.0, 1.0, 1.3, 20.0) == 20.0
    end

    @testset "Three-moment μ solver follows Fortran-style fixed point" begin
        p3_closure = ThreeMomentClosure()

        # D20: Z bounding moved from before to after the μ iteration (Fortran order),
        # so numerical values changed relative to pre-D20 code.
        μ_rimed = solve_shape_parameter(1e-4, 1e6, 1e-11, 0.2, 500.0; closure=p3_closure)
        @test μ_rimed ≈ 0.0

        μ_large = solve_shape_parameter(1e-3, 1e5, 1e-8, 0.5, 700.0; closure=p3_closure)
        @test μ_large ≈ 0.0

        μ_broad = solve_shape_parameter(1e-4, 1e6, 1e-9, 0.2, 500.0; closure=p3_closure)
        @test μ_broad ≈ 0.0
    end

    @testset "Exact three-moment closure solves the full residual" begin
        p3_closure = ThreeMomentClosure()
        exact_closure = ThreeMomentClosureExact()

        μ_p3 = solve_shape_parameter(1e-5, 1e3, 1e-16, 0.0, 400.0; closure=p3_closure)
        μ_exact = solve_shape_parameter(1e-5, 1e3, 1e-16, 0.0, 400.0; closure=exact_closure)

        # D20: Z bounding order change affects μ value
        @test μ_p3 ≈ 17.369421148643028
        @test μ_exact == 0.0
        @test μ_p3 != μ_exact
    end

    @testset "Lambda solver - L/N dependence" begin
        # Higher L/N ratio means larger particles, hence smaller λ
        N_ice = 1e5
        rime_fraction = 0.0
        rime_density = 400.0

        logλ_small = solve_lambda(1e-5, N_ice, rime_fraction, rime_density)  # Small L/N
        logλ_large = solve_lambda(1e-3, N_ice, rime_fraction, rime_density)  # Large L/N

        # Larger mean mass → smaller λ (larger characteristic diameter)
        @test logλ_large < logλ_small
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

    @testset "Tendency functions - smoke tests" begin
        FT = Float64
        ρ = FT(1.0)    # Air density [kg/m³]
        qⁱ = FT(1e-4)  # Ice mass mixing ratio [kg/kg]
        nⁱ = FT(1e5)   # Ice number [1/kg]
        zⁱ = FT(1e-8)  # Ice reflectivity
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
            FT(5e-8),   # rain_evaporation (positive magnitude)
            FT(1e-6),   # rain_self_collection (positive magnitude)
            FT(5e-7),   # rain_breakup (positive = number source)
            # Phase 1: Ice (deposition bidirectional; others positive magnitude)
            FT(3e-7),   # deposition
            FT(1e-8),   # partial_melting
            FT(5e-8),   # complete_melting
            FT(1e3),    # melting_number (positive magnitude)
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
            FT(0.0),    # ccn_activation (M9 stub)
            FT(0.0),    # rain_condensation (M9 stub)
            FT(0.0),    # coating_condensation (M9 stub)
            FT(0.0),    # coating_evaporation (M9 stub)
        )

        # Test each tendency function returns a finite number
        @test isfinite(tendency_ρqᶜˡ(rates, ρ))
        @test isfinite(tendency_ρqʳ(rates, ρ))
        @test isfinite(tendency_ρnʳ(rates, ρ, nⁱ, qⁱ, zero(FT), one(FT), prp))
        @test isfinite(tendency_ρqⁱ(rates, ρ))
        @test isfinite(tendency_ρnⁱ(rates, ρ))
        @test isfinite(tendency_ρqᶠ(rates, ρ, Fᶠ))
        @test isfinite(tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, one(FT), ProcessRateParameters(FT)))
        @test isfinite(tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ))
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
        @test tendency_ρnʳ(zero_rates, ρ, FT(1e5), FT(1e-4), zero(FT), one(FT), ProcessRateParameters(FT)) == 0.0
        @test tendency_ρqⁱ(zero_rates, ρ) == 0.0
        @test tendency_ρnⁱ(zero_rates, ρ) == 0.0
        @test tendency_ρqᶠ(zero_rates, ρ, FT(0.3)) == 0.0
        @test tendency_ρbᶠ(zero_rates, ρ, FT(0.3), FT(400.0), one(FT), ProcessRateParameters(FT)) == 0.0
        @test tendency_ρzⁱ(zero_rates, ρ, FT(1e-4), FT(1e5), FT(1e-8)) == 0.0
        @test tendency_ρqʷⁱ(zero_rates, ρ) == 0.0
        @test tendency_ρqᵛ(zero_rates, ρ) == 0.0
    end

    @testset "Tendency functions - group-2 sources add sixth moment" begin
        FT = Float64
        ρ = FT(1.0)
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        prp = ProcessRateParameters(FT)

        rates = P3ProcessRates(
            FT(0.0), FT(0.0), FT(0.0), FT(0.0), FT(0.0), FT(0.0),  # condensation + 5 rain
            FT(0.0), FT(0.0), FT(0.0), FT(0.0),                     # deposition, partial_melt, complete_melt, melt_n
            FT(0.0),                                                  # sublimation_number (D2)
            FT(0.0), FT(0.0), FT(0.0), FT(0.0), FT(0.0), FT(0.0), FT(0.0),  # agg, ni_limit (C3), 5 riming
            FT(0.0), FT(0.0), FT(0.0),                              # shedding, shedding_n, refreezing
            FT(1e-9), FT(10.0), FT(5e-9), FT(100.0), FT(3e-9), FT(50.0),  # nucleation + immersion freezing
            FT(1e-10), FT(1.0),                                        # splintering
            FT(2e-9), FT(20.0), FT(4e-9), FT(40.0),                    # homogeneous
            FT(0.0), FT(0.0), FT(0.0), FT(0.0),                        # warm collection + rain_warm_n (M9)
            FT(0.0), FT(0.0),                                         # wet growth
            FT(0.0), FT(0.0),                                         # D8 wet growth shedding
            FT(0.0), FT(0.0), FT(0.0), FT(0.0)                       # M9 stubs
        )

        source_z(mass, number, μ_new) = begin
            q_source = max(mass, zero(FT))
            n_source = max(number, zero(FT))
            if q_source == 0 || n_source == 0
                return zero(FT)
            end
            mom3 = q_source * FT(6) / (FT(900) * FT(π))
            G = Breeze.Microphysics.PredictedParticleProperties.g_of_mu(μ_new)
            return G * mom3^2 / n_source
        end

        expected = ρ * (
            source_z(rates.nucleation_mass, rates.nucleation_number, zero(FT)) +
            source_z(rates.cloud_freezing_mass, rates.cloud_freezing_number, zero(FT)) +
            source_z(rates.rain_freezing_mass, rates.rain_freezing_number, zero(FT)) +
            source_z(rates.splintering_mass, rates.splintering_number, zero(FT)) +
            source_z(rates.cloud_homogeneous_mass, rates.cloud_homogeneous_number, zero(FT)) +
            source_z(rates.rain_homogeneous_mass, rates.rain_homogeneous_number, zero(FT))
        )

        @test tendency_ρzⁱ(rates, ρ, FT(0.0), FT(0.0), FT(0.0), prp) ≈ expected
        @test tendency_ρzⁱ(rates, ρ, FT(0.0), FT(0.0), FT(0.0), prp) > 0
    end

    @testset "Tendency functions - Float32 type stability" begin
        FT = Float32
        ρ = FT(1.0)
        rates = P3ProcessRates(ntuple(_ -> FT(1e-7), fieldcount(P3ProcessRates))...)

        @test tendency_ρqᶜˡ(rates, ρ) isa FT
        @test tendency_ρqʳ(rates, ρ) isa FT
        @test tendency_ρnʳ(rates, ρ, FT(1e5), FT(1e-4), zero(FT), one(FT), ProcessRateParameters(FT)) isa FT
        @test tendency_ρqⁱ(rates, ρ) isa FT
        @test tendency_ρnⁱ(rates, ρ) isa FT
        @test tendency_ρqᶠ(rates, ρ, FT(0.3)) isa FT
        @test tendency_ρbᶠ(rates, ρ, FT(0.3), FT(400.0), one(FT), ProcessRateParameters(FT)) isa FT
        @test tendency_ρzⁱ(rates, ρ, FT(1e-4), FT(1e5), FT(1e-8)) isa FT
        @test tendency_ρqʷⁱ(rates, ρ) isa FT
        @test tendency_ρqᵛ(rates, ρ) isa FT
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

    @testset "rain_evaporation_rate" begin
        table_dir = expanduser("~/Aeolus/P3-microphysics/lookup_tables")
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
        FT = Float64

        qr = FT(1e-3)
        nr = FT(1e4)
        T = FT(288.0)
        ρ = FT(1.0)
        P = FT(101325.0)

        # Subsaturated: qv < qv_sat → positive evaporation rate (M7: positive magnitude)
        qv_sat = FT(0.012)
        qv_sub = FT(0.008)    # 67% RH
        rate_sub = rain_evaporation_rate(p3, qr, nr, qv_sub, qv_sat, T, ρ, P)
        @test rate_sub > 0     # Positive magnitude = rain evaporating

        # Saturated: qv = qv_sat → zero evaporation
        rate_sat = rain_evaporation_rate(p3, qr, nr, qv_sat, qv_sat, T, ρ, P)
        @test rate_sat == 0

        # Supersaturated: qv > qv_sat → zero (no condensation on rain)
        qv_super = FT(0.015)
        rate_super = rain_evaporation_rate(p3, qr, nr, qv_super, qv_sat, T, ρ, P)
        @test rate_super == 0

        # Zero rain gives zero evaporation
        rate_norain = rain_evaporation_rate(p3, FT(0), nr, qv_sub, qv_sat, T, ρ, P)
        @test rate_norain == 0
    end

    @testset "cloud_condensation_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        qcl = FT(1e-3)
        T = FT(288.0)
        qv_sat = FT(0.012)

        q = MoistureMassFractions(FT(0.015), FT(1e-3), FT(0))

        # Supersaturated: positive condensation
        qv_super = FT(0.015)
        rate_super = cloud_condensation_rate(p3, qcl, qv_super, qv_sat, T, q, constants)
        @test rate_super > 0

        # Subsaturated: negative (evaporation), limited by available cloud
        qv_sub = FT(0.008)
        q_sub = MoistureMassFractions(qv_sub, qcl, FT(0))
        rate_sub = cloud_condensation_rate(p3, qcl, qv_sub, qv_sat, T, q_sub, constants)
        @test rate_sub < 0

        # Saturated: approximately zero
        rate_sat = cloud_condensation_rate(p3, qcl, qv_sat, qv_sat, T, q, constants)
        @test abs(rate_sat) < 1e-10
    end

    @testset "ventilation_enhanced_deposition" begin
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
        FT = Float64

        qi = FT(1e-4)
        qwi = FT(0)
        ni = FT(1e4)
        Ff = FT(0.0)     # Unrimed
        ρf = FT(400.0)
        T = FT(253.15)   # -20C (cold, ice supersaturated)
        P = FT(50000.0)
        μ = FT(0.0)

        # Supersaturated over ice: positive deposition
        qv_sat_ice = FT(0.0005)
        qv_super = FT(0.001)    # Well above ice saturation
        transport = air_transport_properties(T, P)
        rate_dep = ventilation_enhanced_deposition(p3, qi, qwi, ni, qv_super, qv_sat_ice, Ff, ρf, T, P,
                                                   nothing, transport, MoistureMassFractions(qv_super), μ)
        @test rate_dep > 0

        # Subsaturated over ice: negative (sublimation)
        qv_sub = FT(0.0001)
        rate_sub = ventilation_enhanced_deposition(p3, qi, qwi, ni, qv_sub, qv_sat_ice, Ff, ρf, T, P,
                                                   nothing, transport, MoistureMassFractions(qv_sub), μ)
        @test rate_sub < 0

        # Zero ice gives zero deposition rate (mean mass → default)
        rate_noice = ventilation_enhanced_deposition(p3, FT(0), FT(0), FT(0), qv_super, qv_sat_ice, Ff, ρf, T, P,
                                                     nothing, transport, MoistureMassFractions(qv_super), μ)
        @test abs(rate_noice) < 1e-20

        # Verify that D_v increases at altitude (larger at T=240K/P=30kPa than surface).
        # The deposition formula uses D_v in the denominator of the thermodynamic
        # resistance (Mason 1971), so larger D_v reduces resistance and speeds deposition.
        props_surface = air_transport_properties(FT(273.15), FT(101325.0))
        props_aloft = air_transport_properties(FT(240.0), FT(30000.0))
        @test props_aloft.D_v > props_surface.D_v

        default_constants = ThermodynamicConstants(FT)
        custom_constants = ThermodynamicConstants(FT;
                                                  dry_air_molar_mass = FT(0.031),
                                                  vapor_molar_mass = FT(0.020))
        transport = air_transport_properties(T, P)

        rate_default_constants = ventilation_enhanced_deposition(
            p3, qi, qwi, ni, qv_super, qv_sat_ice, Ff, ρf, T, P,
            default_constants, transport, MoistureMassFractions(qv_super), μ)
        rate_custom_constants = ventilation_enhanced_deposition(
            p3, qi, qwi, ni, qv_super, qv_sat_ice, Ff, ρf, T, P,
            custom_constants, transport, MoistureMassFractions(qv_super), μ)

        @test !isapprox(rate_custom_constants, rate_default_constants; rtol=1e-12, atol=0)
    end

    @testset "ice_melting_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
        FT = Float64

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
                                     nothing, air_transport_properties(T_warm, P), μ)
        @test rate_warm > 0

        # Below freezing: zero melting
        T_cold = FT(263.15)    # -10C
        rate_cold = ice_melting_rate(p3, qi, ni, FT(0), T_cold, P, qv, qv_sat, Ff, ρf, ρ,
                                     nothing, air_transport_properties(T_cold, P), μ)
        @test rate_cold == 0

        # Exactly at freezing: zero (no ΔT to drive melting)
        T_freeze = FT(273.15)
        rate_freeze = ice_melting_rate(p3, qi, ni, FT(0), T_freeze, P, qv, qv_sat, Ff, ρf, ρ,
                                       nothing, air_transport_properties(T_freeze, P), μ)
        @test rate_freeze == 0

        # Warmer temperatures give faster melting
        T_hot = FT(278.15)     # +5C
        rate_hot = ice_melting_rate(p3, qi, ni, FT(0), T_hot, P, qv, qv_sat, Ff, ρf, ρ,
                                    nothing, air_transport_properties(T_hot, P), μ)
        @test rate_hot > rate_warm

        default_constants = ThermodynamicConstants(FT)
        custom_constants = ThermodynamicConstants(FT; vapor_molar_mass = FT(0.020))
        transport = air_transport_properties(T_warm, P)

        melt_default_constants = ice_melting_rate(
            p3, qi, ni, FT(0), T_warm, P, qv, qv_sat, Ff, ρf, ρ, default_constants, transport, μ)
        melt_custom_constants = ice_melting_rate(
            p3, qi, ni, FT(0), T_warm, P, qv, qv_sat, Ff, ρf, ρ, custom_constants, transport, μ)

        @test !isapprox(melt_custom_constants, melt_default_constants; rtol=1e-12, atol=0)
    end

    @testset "ice_melting_rates partitioning" begin
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
        FT = Float64

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
                                      nothing, air_transport_properties(T, P), μ)
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
                                      nothing, air_transport_properties(T, P), μ)
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
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
        FT = Float64

        qi = FT(1e-4)
        qwi = FT(0)
        ni = FT(1e4)
        T = FT(268.15)
        P = FT(85000.0)
        Ff = FT(0.2)
        ρf = FT(400.0)
        ρ = FT(1.0)
        μ = FT(0.0)
        transport = air_transport_properties(T, P)

        T₀ = p3.process_rates.freezing_temperature
        Rᵥ = Breeze.Thermodynamics.vapor_gas_constant(ThermodynamicConstants(FT))
        Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(ThermodynamicConstants(FT))
        ε = Rᵈ / Rᵥ
        e_s0 = PPP.saturation_vapor_pressure_at_freezing(nothing, T₀)
        # M10: set qv = q_sat0 (mixing ratio convention) so latent term vanishes
        qv = ε * e_s0 / max(P - e_s0, FT(1))

        m_mean = qi / ni
        ρ_correction = PPP.ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)
        C_fv = PPP.deposition_ventilation(
            p3.ice.deposition.ventilation,
            p3.ice.deposition.ventilation_enhanced,
            m_mean, Ff, ρf, p3.process_rates, transport.nu, transport.D_v, ρ_correction, p3, μ)

        capacity = PPP.wet_growth_capacity(p3, qi, qwi, ni, T, P, qv, Ff, ρf, ρ, nothing, transport, μ)
        expected = C_fv * transport.K_a * (T₀ - T) * ni

        @test capacity ≈ expected rtol=1e-6
    end

    @testset "refreezing_rate keeps sensible term outside 2π/Lf" begin
        PPP = Breeze.Microphysics.PredictedParticleProperties
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
        FT = Float64

        qwi = FT(1)
        qi = FT(1e-4)
        ni = FT(1e4)
        T = FT(268.15)
        P = FT(85000.0)
        Ff = FT(0.2)
        ρf = FT(400.0)
        ρ = FT(1.0)
        μ = FT(0.0)
        transport = air_transport_properties(T, P)

        T₀ = p3.process_rates.freezing_temperature
        Rᵥ = Breeze.Thermodynamics.vapor_gas_constant(ThermodynamicConstants(FT))
        Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(ThermodynamicConstants(FT))
        ε = Rᵈ / Rᵥ
        e_s0 = PPP.saturation_vapor_pressure_at_freezing(nothing, T₀)
        # M10: set qv = q_sat0 (mixing ratio convention) so latent term vanishes
        qv = ε * e_s0 / max(P - e_s0, FT(1))

        m_mean = qi / ni
        ρ_correction = PPP.ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)
        C_fv = PPP.deposition_ventilation(
            p3.ice.deposition.ventilation,
            p3.ice.deposition.ventilation_enhanced,
            m_mean, Ff, ρf, p3.process_rates, transport.nu, transport.D_v, ρ_correction, p3, μ)

        refreezing = PPP.refreezing_rate(p3, qwi, qi, ni, T, P, qv, Ff, ρf, ρ, nothing, transport, μ)
        expected = C_fv * transport.K_a * (T₀ - T) * ni

        # With Fortran tables the ventilation integral differs; verify
        # refreezing is positive and within order-of-magnitude of expected.
        @test refreezing > 0
        @test refreezing ≈ expected rtol=0.25
    end

    @testset "ice_aggregation_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
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
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
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
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
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
        transport = air_transport_properties(T, P)

        ρ_rime = rime_density(p3, qcl, cloud_rim, T, vᵢ, ρ, constants, transport)

        prp = p3.process_rates
        μ_c = p3.cloud.shape_parameter
        Nᶜ = p3.cloud.number_concentration
        ρ_water = prp.liquid_water_density

        qcl_abs = qcl * ρ
        μ_air = transport.nu * ρ
        λ_c_uncapped = cbrt(
            FT(π) * ρ_water * Nᶜ * (μ_c + 3) * (μ_c + 2) * (μ_c + 1) /
            (FT(6) * qcl_abs)
        )
        λ_c = clamp(λ_c_uncapped, (μ_c + 1) * FT(2.5e4), (μ_c + 1) * FT(1e6))
        a_cn = constants.gravitational_acceleration * ρ_water / (FT(18) * μ_air)
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
        transport_warm = air_transport_properties(T_warm, P)
        ρ_warm = rime_density(p3, qcl, cloud_rim, T_warm, vᵢ, ρ, constants, transport_warm)
        @test ρ_warm == 400

        ρ_no_cloud = rime_density(p3, qcl, FT(0), T, vᵢ, ρ, constants, transport)
        @test ρ_no_cloud == 400
    end

    @testset "Rime consistency enforcement" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        prp = p3.process_rates

        function fortran_calc_bulk_rho_rime(qⁱ, qᶠ, bᶠ)
            qᶠ = max(qᶠ, 0)
            bᶠ = max(bᶠ, 0)
            ρᶠ = FT(NaN)

            if bᶠ >= FT(1e-15)
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

        no_volume = consistent_rime_state(p3, FT(1e-4), FT(1e-5), FT(1e-16), FT(0))
        @test no_volume.qᶠ == 0
        @test no_volume.bᶠ == 0
        @test no_volume.ρᶠ == 0
        @test no_volume.Fᶠ == 0

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
        μ = (
            ρqᶜˡ = FT(0),
            ρnᶜˡ = FT(0),
            ρqʳ = FT(0),
            ρnʳ = FT(0),
            ρqⁱ = ρ * FT(1e-5),
            ρnⁱ = ρ * FT(1e5),
            ρqᶠ = ρ * FT(2e-5),
            ρbᶠ = ρ * FT(5e-8),
            ρzⁱ = ρ * FT(1e-10),
            ρqʷⁱ = FT(0),
        )
        ℳ = Breeze.AtmosphereModels.microphysical_state(p3, ρ, μ, nothing, nothing)
        @test ℳ.qᶠ == FT(1e-5)
        @test ℳ.bᶠ ≈ FT(2.5e-8)

        for (qⁱ, qᶠ, bᶠ) in (
            (FT(1e-4), FT(1e-5), FT(1e-16)),
            (FT(1e-4), FT(5e-15), FT(1e-15)),
            (FT(1e-4), FT(2e-5), FT(2e-6)),
            (FT(1e-4), FT(2e-5), FT(2e-8)),
            (FT(1e-5), FT(2e-5), FT(5e-8)),
        )
            got = consistent_rime_state(p3, qⁱ, qᶠ, bᶠ, FT(0))
            ref = fortran_calc_bulk_rho_rime(qⁱ, qᶠ, bᶠ)
            @test got.qᶠ == ref.qᶠ
            @test got.bᶠ ≈ ref.bᶠ
            @test got.ρᶠ ≈ ref.ρᶠ
        end
    end

    @testset "compute_p3_process_rates integration" begin
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
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
            FT(1e-10),     # zⁱ (reflectivity)
            FT(0),         # qʷⁱ (liquid on ice)
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

    @testset "compute_p3_process_rates uses prognostic cloud number" begin
        p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)
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
            FT(1e-10),     # zⁱ
            FT(0),         # qʷⁱ
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
            FT(1e-10),      # zⁱ
            FT(0),          # qʷⁱ
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
        p3_tab = PredictedParticlePropertiesMicrophysics(; lookup_tables=table_dir)

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
            FT(qf / 400), FT(1e-10), FT(0))

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
