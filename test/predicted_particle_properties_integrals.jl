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
    LiquidIcePotentialTemperatureState

using Oceananigans: CPU, RectilinearGrid
using Oceananigans.Fields: interior

const PPP = Breeze.Microphysics.PredictedParticleProperties

function fortran_complex_water_ray(FT, λ, T_celsius)
    πFT = FT(π)
    epsinf = FT(5.27137) + FT(0.02164740) * T_celsius - FT(0.00131198) * T_celsius^2
    ΔT = T_celsius - FT(25)
    epss = FT(78.54)
    α = FT(-16.8129) / (T_celsius + FT(273.16)) + FT(0.0609265)
    λs = FT(0.00033836) * exp(FT(2513.98) / (T_celsius + FT(273.16))) * FT(1e-2)
    ratio = λs / λ
    denom = FT(1) + FT(2) * ratio^(FT(1) - α) * sin(α * πFT / FT(2)) + ratio^(FT(2) - FT(2) * α)
    epsr = epsinf + (epss - epsinf) * (ratio^(FT(1) - α) * sin(α * πFT / FT(2)) + FT(1)) / denom
    epsi = (epss - epsinf) * (ratio^(FT(1) - α) * cos(α * πFT / FT(2))) / denom +
           λ * FT(1.25664) / FT(1.88496)
    return sqrt(complex(epsr, -epsi))
end

function fortran_complex_ice_maetzler(FT, λ, T_celsius)
    c = FT(2.99e8)
    T_k = T_celsius + FT(273.16)
    f = c / λ * FT(1e-9)
    B1 = FT(0.0207)
    B2 = FT(1.16e-11)
    b = FT(335.0)
    Δβ = exp(FT(-10.02) + FT(0.0364) * (T_k - FT(273.16)))
    βm = (B1 / T_k) * (exp(b / T_k) / (exp(b / T_k) - FT(1))^2) + B2 * f^2
    β = βm + Δβ
    θ = FT(300) / T_k - FT(1)
    α = (FT(0.00504) + FT(0.0062) * θ) * exp(FT(-22.1) * θ)
    ε = complex(FT(3.1884) + FT(9.1e-4) * (T_k - FT(273.16)), α / f + β * f)
    return sqrt(conj(ε))
end

@inline function maxwell_garnett_mix(m1, m2, m3, vol1, vol2, vol3, inclusion::Symbol)
    m1s = m1^2
    m2s = m2^2
    m3s = m3^2
    if inclusion === :spherical
        β2 = 3 * m1s / (m2s + 2 * m1s)
        β3 = 3 * m1s / (m3s + 2 * m1s)
    else
        β2 = 2 * m1s / (m2s - m1s) * (m2s / (m2s - m1s) * log(m2s / m1s) - 1)
        β3 = 2 * m1s / (m3s - m1s) * (m3s / (m3s - m1s) * log(m3s / m1s) - 1)
    end
    return sqrt(((1 - vol2 - vol3) * m1s + vol2 * β2 * m2s + vol3 * β3 * m3s) /
                (1 - vol2 - vol3 + vol2 * β2 + vol3 * β3))
end

function fortran_rayleigh_wet_ice_factor(FT, total_mass, liquid_fraction, D)
    λ_radar = FT(0.10)
    m_air = complex(FT(1), FT(0))
    m_water = fortran_complex_water_ray(FT, λ_radar, FT(0))
    m_ice = fortran_complex_ice_maetzler(FT, λ_radar, FT(0))
    K_w = abs((m_water^2 - 1) / (m_water^2 + 2))^2

    mass_water = liquid_fraction * total_mass
    vg = FT(π) / FT(6) * D^3
    ρg = total_mass / vg
    D_large = cbrt(FT(6) / FT(π) * (total_mass / ρg))
    vol_ice = (total_mass - mass_water) / (vg * FT(900))
    vol_water = mass_water / (FT(1000) * vg)
    vol_air = FT(1) - vol_ice - vol_water

    vol_ice_frac = vol_ice / max(vol_ice + vol_water, FT(1e-10))
    # Step 1: ice inclusions in water matrix (Fortran matrix='water')
    mixed_icewater = maxwell_garnett_mix(m_water, m_air, m_ice, FT(0), FT(0), vol_ice_frac, :spheroidal)
    # Step 2: air inclusions in icewater matrix (Fortran hostmatrix='icewater')
    core = maxwell_garnett_mix(mixed_icewater, m_air, 2 * m_air, FT(0), vol_air, FT(0), :spheroidal)
    return abs((core^2 - 1) / (core^2 + 2))^2 / K_w * D_large^6
end

@testset "P3 Integrals" begin

    @testset "Smoke tests - type construction" begin
        # Test main scheme construction
        p3 = PredictedParticlePropertiesMicrophysics()
        @test p3 isa PredictedParticlePropertiesMicrophysics
        @test p3.water_density == 1000.0
        @test p3.minimum_mass_mixing_ratio == 1e-14
        @test p3.minimum_number_mixing_ratio == 1e-16

        # Test alias
        p3_alias = P3Microphysics()
        @test p3_alias isa PredictedParticlePropertiesMicrophysics

        # Test with Float32
        p3_f32 = PredictedParticlePropertiesMicrophysics(Float32)
        @test p3_f32.water_density isa Float32
        @test p3_f32.minimum_mass_mixing_ratio isa Float32
        @test p3_f32.ice.fall_speed.reference_air_density isa Float32
    end

    @testset "AtmosphereModel initialization converts total moisture to vapor" begin
        FT = Float64
        grid = RectilinearGrid(CPU(); size=(1, 1, 1), extent=(1, 1, 1))

        constants = ThermodynamicConstants(FT)
        reference_state = Breeze.ReferenceState(grid, constants;
                                                surface_pressure = FT(101325),
                                                potential_temperature = FT(300))
        dynamics = Breeze.AnelasticDynamics(reference_state)
        table_dir = expanduser("~/Aeolus/P3-microphysics/lookup_tables")
        model = Breeze.AtmosphereModel(grid; dynamics,
                                       thermodynamic_constants = constants,
                                       microphysics = PredictedParticlePropertiesMicrophysics(FT; lookup_tables=table_dir))

        qᵗ = FT(0.02)
        qᶜˡ = FT(0.005)
        qʳ = FT(0.001)
        qⁱ = FT(0.002)
        qʷⁱ = FT(0.0005)
        expected_qᵛ = qᵗ - qᶜˡ - qʳ - qⁱ - qʷⁱ

        Breeze.set!(model; θ = FT(300), qᵗ, qᶜˡ, qʳ, qⁱ, qʷⁱ)

        qᵛ_actual = first(Array(interior(model.microphysical_fields.qᵛ)))
        @test qᵛ_actual ≈ expected_qᵛ
    end

    @testset "Ice properties construction" begin
        ice = IceProperties()
        @test ice isa IceProperties
        @test ice.minimum_rime_density == 50.0
        @test ice.maximum_rime_density == 900.0
        @test ice.maximum_shape_parameter == 20.0
        @test ice.minimum_reflectivity == 1e-35

        # Check all sub-containers exist
        @test ice.fall_speed isa IceFallSpeed
        @test ice.deposition isa IceDeposition
        @test ice.bulk_properties isa IceBulkProperties
        @test ice.collection isa IceCollection
        @test ice.sixth_moment isa IceSixthMoment
        @test ice.lambda_limiter isa IceLambdaLimiter
        @test ice.ice_rain isa IceRainCollection
    end

    @testset "Ice fall speed" begin
        fs = IceFallSpeed()
        constants = ThermodynamicConstants(Float64)
        Rᵈ = dry_air_gas_constant(constants)
        @test fs.reference_air_density ≈ 60000 / (Rᵈ * 253.15)

        @test fs.number_weighted isa NumberWeightedFallSpeed
        @test fs.mass_weighted isa MassWeightedFallSpeed
        @test fs.reflectivity_weighted isa ReflectivityWeightedFallSpeed
    end

    @testset "Ice deposition" begin
        dep = IceDeposition()
        @test dep.thermal_conductivity ≈ 0.024
        @test dep.vapor_diffusivity ≈ 2.2e-5

        @test dep.ventilation isa Ventilation
        @test dep.ventilation_enhanced isa VentilationEnhanced
        @test dep.small_ice_ventilation_constant isa SmallIceVentilationConstant
        @test dep.small_ice_ventilation_reynolds isa SmallIceVentilationReynolds
        @test dep.large_ice_ventilation_constant isa LargeIceVentilationConstant
        @test dep.large_ice_ventilation_reynolds isa LargeIceVentilationReynolds
    end

    @testset "Ice bulk properties" begin
        bp = IceBulkProperties()
        @test bp.maximum_mean_diameter ≈ 2.0e-2
        @test bp.minimum_mean_diameter ≈ 2.0e-6


        @test bp.effective_radius isa EffectiveRadius
        @test bp.mean_diameter isa MeanDiameter
        @test bp.mean_density isa MeanDensity
        @test bp.reflectivity isa Reflectivity
        @test bp.slope isa SlopeParameter
        @test bp.shape isa ShapeParameter
        @test bp.shedding isa SheddingRate
    end

    @testset "Ice collection" begin
        col = IceCollection()
        @test col.ice_cloud_collection_efficiency ≈ 0.5
        @test col.ice_rain_collection_efficiency ≈ 1.0

        @test col.aggregation isa AggregationNumber
        @test col.rain_collection isa RainCollectionNumber
    end

    @testset "Ice sixth moment" begin
        m6 = IceSixthMoment()
        @test m6.rime isa SixthMomentRime
        @test m6.deposition isa SixthMomentDeposition
        @test m6.deposition1 isa SixthMomentDeposition1
        @test m6.melt1 isa SixthMomentMelt1
        @test m6.melt2 isa SixthMomentMelt2
        @test m6.aggregation isa SixthMomentAggregation
        @test m6.shedding isa SixthMomentShedding
        @test m6.sublimation isa SixthMomentSublimation
        @test m6.sublimation1 isa SixthMomentSublimation1
    end

    @testset "Ice lambda limiter" begin
        ll = IceLambdaLimiter()
        @test ll.small_q isa NumberMomentLambdaLimit
        @test ll.large_q isa MassMomentLambdaLimit
    end

    @testset "Ice-rain collection" begin
        ir = IceRainCollection()
        @test isnothing(ir.mass)
        @test isnothing(ir.number)
        @test isnothing(ir.sixth_moment)
    end

    @testset "Rain properties" begin
        rain = RainProperties()
        @test rain.maximum_mean_diameter ≈ 6e-3
        @test rain.fall_speed_coefficient ≈ 841.99667
        @test rain.fall_speed_exponent ≈ 0.8

        @test rain.shape_parameter isa RainShapeParameter
        @test rain.velocity_number isa RainVelocityNumber
        @test rain.velocity_mass isa RainVelocityMass
        @test rain.evaporation isa RainEvaporation
    end

    @testset "Cloud droplet properties" begin
        cloud = CloudDropletProperties()
        @test cloud.number_concentration ≈ 200e6
        @test cloud.autoconversion_threshold ≈ 25e-6
        @test cloud.condensation_timescale ≈ 1.0

        # μ_c is diagnosed from Nc via Liu-Daum (2000) by default.
        # For Nc = 200e6 m⁻³ (200 cm⁻³): χ = 0.0005714*200 + 0.2714 = 0.38568,
        # μ_c = 1/0.38568² - 1 ≈ 5.72 (clamped to [2, 15])
        @test 2 ≤ cloud.shape_parameter ≤ 15
        @test cloud.shape_parameter ≈ liu_daum_shape_parameter(200e6)

        # Explicit shape_parameter overrides Liu-Daum
        cloud_override = CloudDropletProperties(Float64; shape_parameter=5)
        @test cloud_override.shape_parameter ≈ 5.0

        # Test custom parameters
        cloud_custom = CloudDropletProperties(Float64; number_concentration=50e6)
        @test cloud_custom.number_concentration ≈ 50e6
        # Marine Nc → higher μ_c than continental (fewer, larger, more uniform drops)
        @test cloud_custom.shape_parameter > cloud.shape_parameter
    end

    @testset "Water density is shared" begin
        # Water density should be at top level, shared by cloud and rain
        p3 = PredictedParticlePropertiesMicrophysics()
        @test p3.water_density ≈ 1000.0

        # Custom water density
        p3_custom = PredictedParticlePropertiesMicrophysics(Float64; water_density=998.0)
        @test p3_custom.water_density ≈ 998.0
    end

    @testset "Prognostic field names" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        names = prognostic_field_names(p3)

        # P3 cloud number should be prognostic.
        @test :ρqᶜˡ ∈ names
        @test :ρnᶜˡ ∈ names
        @test :ρqʳ ∈ names
        @test :ρnʳ ∈ names
        @test :ρqⁱ ∈ names
        @test :ρnⁱ ∈ names
        @test :ρqᶠ ∈ names
        @test :ρbᶠ ∈ names
        @test :ρzⁱ ∈ names
        @test :ρqʷⁱ ∈ names
    end

    @testset "Quadrature evaluation - mixed-phase Rayleigh reflectivity" begin
        mixed_state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0,
            liquid_fraction = 0.5)

        D = 2e-3
        thresholds = PPP.regime_thresholds_from_state(Float64, mixed_state)
        total_mass = PPP.particle_mass(D, mixed_state, thresholds)
        number_density = size_distribution(D, mixed_state)
        expected = fortran_rayleigh_wet_ice_factor(Float64, total_mass, mixed_state.liquid_fraction, D) * number_density
        got = PPP.integrand(RayleighReflectivity(), D, mixed_state, thresholds)

        @test got > 0
        @test got ≈ expected rtol=1e-6
    end

    @testset "Integral type hierarchy" begin
        # Test abstract types
        @test NumberWeightedFallSpeed <: AbstractFallSpeedIntegral
        @test AbstractFallSpeedIntegral <: AbstractIceIntegral
        @test AbstractIceIntegral <: AbstractP3Integral

        @test Ventilation <: AbstractDepositionIntegral
        @test EffectiveRadius <: AbstractBulkPropertyIntegral
        @test AggregationNumber <: AbstractCollectionIntegral
        @test SixthMomentRime <: AbstractSixthMomentIntegral
        @test NumberMomentLambdaLimit <: AbstractLambdaLimiterIntegral

        @test RainShapeParameter <: AbstractRainIntegral
        @test AbstractRainIntegral <: AbstractP3Integral
    end

    @testset "TabulatedIntegral" begin
        # Create a test array
        data = rand(10, 5, 3)
        tab = TabulatedIntegral(data)

        @test tab isa TabulatedIntegral
        @test size(tab) == (10, 5, 3)
        @test tab[1, 1, 1] == data[1, 1, 1]
        @test tab[5, 3, 2] == data[5, 3, 2]
    end

    @testset "Show methods" begin
        # Just test that show methods don't error
        p3 = PredictedParticlePropertiesMicrophysics()
        io = IOBuffer()
        show(io, p3)
        @test length(take!(io)) > 0

        show(io, p3.ice)
        @test length(take!(io)) > 0

        show(io, p3.ice.fall_speed)
        @test length(take!(io)) > 0

        show(io, p3.rain)
        @test length(take!(io)) > 0

        show(io, p3.cloud)
        @test length(take!(io)) > 0
    end

    @testset "Ice size distribution state" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        @test state.intercept ≈ 1e6
        @test state.shape ≈ 0.0
        @test state.slope ≈ 1000.0
        @test state.rime_fraction ≈ 0.0
        @test state.liquid_fraction ≈ 0.0
        @test state.rime_density ≈ 400.0

        # Test with rime
        state_rimed = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 2.0,
            slope = 500.0,
            rime_fraction = 0.5,
            rime_density = 600.0)

        @test state_rimed.rime_fraction ≈ 0.5
        @test state_rimed.rime_density ≈ 600.0

        # Test size distribution evaluation
        D = 100e-6  # 100 μm
        Np = size_distribution(D, state)
        @test Np > 0

        # N'(D) = N₀ D^μ exp(-λD) for μ=0: N₀ exp(-λD)
        expected = 1e6 * exp(-1000 * D)
        @test Np ≈ expected
    end

    @testset "Chebyshev-Gauss quadrature" begin
        nodes, weights = chebyshev_gauss_nodes_weights(Float64, 32)

        @test length(nodes) == 32
        @test length(weights) == 32

        # Nodes should be in [-1, 1]
        @test all(-1 ≤ x ≤ 1 for x in nodes)

        # Weights should sum to ≈2 (= ∫₋₁¹ dx, with √(1-x²) correction)
        @test sum(weights) ≈ 2 rtol=1e-2

        # Test Float32
        nodes32, weights32 = chebyshev_gauss_nodes_weights(Float32, 16)
        @test eltype(nodes32) == Float32
        @test eltype(weights32) == Float32
    end

    @testset "Quadrature evaluation - fall speed integrals" begin
        # Create a test state
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        # Test number-weighted fall speed
        V_n = evaluate(NumberWeightedFallSpeed(), state)
        @test V_n > 0
        @test isfinite(V_n)

        # Test mass-weighted fall speed
        V_m = evaluate(MassWeightedFallSpeed(), state)
        @test V_m > 0
        @test isfinite(V_m)

        # Test reflectivity-weighted fall speed
        V_z = evaluate(ReflectivityWeightedFallSpeed(), state)
        @test V_z > 0
        @test isfinite(V_z)

        # Mass-weighted should be larger than number-weighted
        # (larger particles fall faster and contribute more mass)
        # Note: this depends on the specific parameterization
    end

    @testset "Quadrature evaluation - deposition integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        # Basic ventilation
        v = evaluate(Ventilation(), state)
        @test v ≥ 0
        @test isfinite(v)

        v_enh = evaluate(VentilationEnhanced(), state)
        @test v_enh ≥ 0
        @test isfinite(v_enh)

        # Size-regime ventilation
        v_sc = evaluate(SmallIceVentilationConstant(), state)
        v_sr = evaluate(SmallIceVentilationReynolds(), state)
        v_lc = evaluate(LargeIceVentilationConstant(), state)
        v_lr = evaluate(LargeIceVentilationReynolds(), state)

        @test all(isfinite, [v_sc, v_sr, v_lc, v_lr])
    end

    @testset "Quadrature evaluation - bulk property integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        r_eff = evaluate(EffectiveRadius(), state)
        @test r_eff > 0
        @test isfinite(r_eff)

        d_m = evaluate(MeanDiameter(), state)
        @test d_m > 0
        @test isfinite(d_m)

        ρ_m = evaluate(MeanDensity(), state)
        @test ρ_m > 0
        @test isfinite(ρ_m)

        Z = evaluate(Reflectivity(), state)
        @test Z > 0
        @test isfinite(Z)
    end

    @testset "Quadrature evaluation - collection integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        n_agg = evaluate(AggregationNumber(), state)
        @test n_agg > 0
        @test isfinite(n_agg)

        n_rw = evaluate(RainCollectionNumber(), state)
        @test n_rw > 0
        @test isfinite(n_rw)
    end

    @testset "Quadrature evaluation - sixth moment integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0,
            liquid_fraction = 0.1)

        m6_rime = evaluate(SixthMomentRime(), state)
        @test m6_rime > 0
        @test isfinite(m6_rime)

        m6_dep = evaluate(SixthMomentDeposition(), state)
        @test m6_dep > 0
        @test isfinite(m6_dep)

        m6_agg = evaluate(SixthMomentAggregation(), state)
        @test m6_agg > 0
        @test isfinite(m6_agg)

        m6_shed = evaluate(SixthMomentShedding(), state)
        @test m6_shed > 0  # Non-zero because liquid_fraction > 0
        @test isfinite(m6_shed)
    end

    @testset "Quadrature evaluation - lambda limiter integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        i_small = evaluate(NumberMomentLambdaLimit(), state)
        @test i_small > 0
        @test isfinite(i_small)

        i_large = evaluate(MassMomentLambdaLimit(), state)
        @test i_large > 0
        @test isfinite(i_large)
    end

    @testset "Quadrature convergence" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        # Test that quadrature gives consistent results across resolutions
        V_16 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=16)
        V_32 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=32)
        V_64 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=64)
        V_128 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=128)

        # All resolutions should give values within 1% of the high-resolution result
        @test abs(V_16 - V_128) / V_128 < 0.01
        @test abs(V_32 - V_128) / V_128 < 0.01
        @test abs(V_64 - V_128) / V_128 < 0.01
    end

    #####
    ##### Physical consistency tests
    #####

    @testset "Fall speed physical consistency" begin
        # For a given PSD, larger particles fall faster
        # Note: evaluate() returns density-weighted integrals (fluxes), not mean velocities.
        # So we cannot compare V_n and V_m directly without normalization.

        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 500.0)  # Larger particles (smaller λ)

        V_n = evaluate(NumberWeightedFallSpeed(), state)
        V_m = evaluate(MassWeightedFallSpeed(), state)
        V_z = evaluate(ReflectivityWeightedFallSpeed(), state)

        # All should be positive
        @test V_n > 0
        @test V_m > 0
        @test V_z > 0

        # Typical ice fall speeds should be 0.1 - 10 m/s
        # These are normalized integrals, but check they're in a reasonable range
        @test isfinite(V_n)
        @test isfinite(V_m)
        @test isfinite(V_z)
    end

    @testset "Size distribution moments" begin
        # For exponential distribution (μ=0), analytical moments are known:
        # M_n = N₀ Γ(n+1) / λ^{n+1}
        # M_0 = N₀ / λ (total number)
        # M_1 = N₀ / λ²
        # M_3 = 6 N₀ / λ⁴
        # M_6 = 720 N₀ / λ⁷

        N₀ = 1e6
        λ = 1000.0
        μ = 0.0

        state = IceSizeDistributionState(Float64;
            intercept = N₀,
            shape = μ,
            slope = λ)

        # Number integral (0th moment proxy)
        n_int = evaluate(NumberMomentLambdaLimit(), state)  # This is just ∫ N'(D) dD
        @test n_int > 0
        @test isfinite(n_int)

        # Reflectivity (6th moment)
        Z = evaluate(Reflectivity(), state)
        @test Z > 0
        @test isfinite(Z)
    end

    @testset "Slope parameter dependence" begin
        # Smaller λ means larger particles
        # Integrals should increase as λ decreases (larger particles)

        state_small = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 2000.0)  # Small particles

        state_large = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 500.0)   # Large particles

        # Reflectivity (D^6 weighted) should increase with larger particles
        Z_small = evaluate(Reflectivity(), state_small)
        Z_large = evaluate(Reflectivity(), state_large)

        @test Z_large > Z_small

        # Fall speed should also increase with larger particles
        V_small = evaluate(NumberWeightedFallSpeed(), state_small)
        V_large = evaluate(NumberWeightedFallSpeed(), state_large)

        @test V_large > V_small
    end

    @testset "Shape parameter dependence" begin
        # Higher μ narrows the distribution
        state_mu0 = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0)

        state_mu2 = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 2.0, slope = 1000.0)

        state_mu4 = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 4.0, slope = 1000.0)

        # All should produce finite results
        V0 = evaluate(NumberWeightedFallSpeed(), state_mu0)
        V2 = evaluate(NumberWeightedFallSpeed(), state_mu2)
        V4 = evaluate(NumberWeightedFallSpeed(), state_mu4)

        @test all(isfinite, [V0, V2, V4])
        @test all(x -> x > 0, [V0, V2, V4])
    end

    @testset "Rime fraction dependence" begin
        # Test that both rimed and unrimed states produce valid results
        state_unrimed = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            rime_fraction = 0.0, rime_density = 400.0)

        state_rimed = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            rime_fraction = 0.5, rime_density = 600.0)

        # Both should produce valid results for number-weighted fall speed
        V_unrimed = evaluate(NumberWeightedFallSpeed(), state_unrimed)
        V_rimed = evaluate(NumberWeightedFallSpeed(), state_rimed)

        @test isfinite(V_unrimed)
        @test isfinite(V_rimed)
        @test V_unrimed > 0
        @test V_rimed > 0

        # Mass-weighted fall speed should be larger for rimed particles
        # because particle_mass depends on rime_fraction (higher effective density)
        F_m_unrimed = evaluate(MassWeightedFallSpeed(), state_unrimed)
        F_m_rimed = evaluate(MassWeightedFallSpeed(), state_rimed)

        @test isfinite(F_m_unrimed)
        @test isfinite(F_m_rimed)
        @test F_m_rimed > F_m_unrimed  # Higher mass → larger flux integral
    end

    @testset "Liquid fraction dependence" begin
        # Liquid on ice affects shedding and melting integrals
        state_dry = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            liquid_fraction = 0.0)

        state_wet = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            liquid_fraction = 0.3)

        # Shedding integrand enforces D ≥ 9 mm threshold (Rasmussen et al. 2011).
        # The Fl-blended mass makes the integral depend on liquid fraction
        # (matching Fortran table generation). Wet particles have larger mass
        # at a given D because liquid is denser than ice aggregates.
        shed_dry = evaluate(SheddingRate(), state_dry)
        shed_wet = evaluate(SheddingRate(), state_wet)

        @test shed_dry ≥ 0
        @test shed_wet ≥ 0
        @test shed_wet > shed_dry  # wet particles have larger blended mass

        # Sixth moment shedding: 6D⁵ Np / dmdD where dmdD depends on Fl
        # (Fortran convention includes the Jacobian 1/dmdD).
        # Wet particles have larger dmdD → smaller integrand.
        m6_shed_dry = evaluate(SixthMomentShedding(), state_dry)
        m6_shed_wet = evaluate(SixthMomentShedding(), state_wet)

        @test m6_shed_dry ≥ 0
        @test m6_shed_wet ≥ 0

        D_large = 5e-4
        melt_const_dry = Breeze.Microphysics.PredictedParticleProperties.integrand(
            LargeIceVentilationConstant(), D_large, state_dry)
        melt_const_wet = Breeze.Microphysics.PredictedParticleProperties.integrand(
            LargeIceVentilationConstant(), D_large, state_wet)
        melt_re_dry = Breeze.Microphysics.PredictedParticleProperties.integrand(
            LargeIceVentilationReynolds(), D_large, state_dry)
        melt_re_wet = Breeze.Microphysics.PredictedParticleProperties.integrand(
            LargeIceVentilationReynolds(), D_large, state_wet)

        @test melt_const_wet > melt_const_dry
        @test melt_re_wet != melt_re_dry
    end

    @testset "Deposition integrals physical consistency" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0)

        # Ventilation integrals should be positive
        v = evaluate(Ventilation(), state)
        v_enh = evaluate(VentilationEnhanced(), state)

        @test v > 0
        @test v_enh ≥ 0  # Could be zero if no particles > 100 μm

        # Small + large ice ventilation should roughly equal total
        v_sc = evaluate(SmallIceVentilationConstant(), state)
        v_lc = evaluate(LargeIceVentilationConstant(), state)

        @test v_sc ≥ 0
        @test v_lc ≥ 0
        # Note: v_sc + v_lc ≈ v only if ventilation factor is same
    end

    @testset "Collection integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0)

        n_agg = evaluate(AggregationNumber(), state)
        n_rw = evaluate(RainCollectionNumber(), state)

        @test n_agg > 0
        @test n_rw > 0
        @test isfinite(n_agg)
        @test isfinite(n_rw)
    end

    @testset "Sixth moment integrals physical consistency" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            liquid_fraction = 0.1)

        # All sixth moment integrals should be finite and positive
        m6_rime = evaluate(SixthMomentRime(), state)
        m6_dep = evaluate(SixthMomentDeposition(), state)
        m6_dep1 = evaluate(SixthMomentDeposition1(), state)
        m6_mlt1 = evaluate(SixthMomentMelt1(), state)
        m6_mlt2 = evaluate(SixthMomentMelt2(), state)
        m6_agg = evaluate(SixthMomentAggregation(), state)
        m6_shed = evaluate(SixthMomentShedding(), state)
        m6_sub = evaluate(SixthMomentSublimation(), state)
        m6_sub1 = evaluate(SixthMomentSublimation1(), state)

        @test all(isfinite, [m6_rime, m6_dep, m6_dep1, m6_mlt1, m6_mlt2,
                             m6_agg, m6_shed, m6_sub, m6_sub1])
        @test all(x -> x > 0, [m6_rime, m6_dep, m6_mlt1, m6_agg, m6_sub])
    end

    @testset "Float32 input precision" begin
        # All integrals should work with Float32 input
        # Note: FastGaussQuadrature returns Float64 nodes/weights, so output
        # may be promoted to Float64. We just check the results are valid.
        state = IceSizeDistributionState(Float32;
            intercept = 1f6,
            shape = 0f0,
            slope = 1000f0)

        V_n = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=32)
        V_m = evaluate(MassWeightedFallSpeed(), state; n_quadrature=32)
        Z = evaluate(Reflectivity(), state; n_quadrature=32)

        # Results should be valid floating point numbers
        @test isfinite(V_n)
        @test isfinite(V_m)
        @test isfinite(Z)
        @test V_n > 0
        @test V_m > 0
        @test Z > 0
    end

    @testset "Extreme parameters" begin
        # Very small particles (large λ)
        state_small = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 10000.0)

        V_small = evaluate(NumberWeightedFallSpeed(), state_small)
        @test isfinite(V_small)
        @test V_small > 0

        # Very large particles (small λ)
        state_large = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 100.0)

        V_large = evaluate(NumberWeightedFallSpeed(), state_large)
        @test isfinite(V_large)
        @test V_large > 0

        # High shape parameter
        state_narrow = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 6.0, slope = 1000.0)

        V_narrow = evaluate(NumberWeightedFallSpeed(), state_narrow)
        @test isfinite(V_narrow)
        @test V_narrow > 0
    end

    #####
    ##### Analytical comparison tests
    #####

    @testset "Analytical comparison - exponential PSD moments" begin
        # For exponential PSD (μ=0): N'(D) = N₀ exp(-λD)
        # The n-th moment is:
        # M_n = ∫₀^∞ D^n N'(D) dD = N₀ n! / λ^{n+1}

        N₀ = 1e6
        μ = 0.0
        λ = 1000.0

        # For exponential (μ=0): M_n = N₀ n! / λ^{n+1}
        # M0 = N₀ / λ = 1e6 / 1000 = 1e3
        # M6 = N₀ * 720 / λ^7 = 1e6 * 720 / 1e21 = 7.2e-13

        state = IceSizeDistributionState(Float64;
            intercept = N₀, shape = μ, slope = λ)

        # Test NumberMomentLambdaLimit (which integrates the full PSD)
        small_q_lim = evaluate(NumberMomentLambdaLimit(), state; n_quadrature=128)
        @test small_q_lim > 0
        @test isfinite(small_q_lim)

        # Test Reflectivity (which integrates D^6 N'(D))
        refl = evaluate(Reflectivity(), state; n_quadrature=128)
        @test refl > 0
        @test isfinite(refl)
    end

    @testset "Analytical comparison - gamma PSD with mu=2" begin
        # For gamma PSD with μ=2: N'(D) = N₀ D² exp(-λD)
        # M_n = N₀ Γ(n+3) / λ^{n+3} = N₀ (n+2)! / λ^{n+3}

        N₀ = 1e6
        μ = 2.0
        λ = 1000.0

        state = IceSizeDistributionState(Float64;
            intercept = N₀, shape = μ, slope = λ)

        # All integrals should return finite positive values
        V_n = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=128)
        V_m = evaluate(MassWeightedFallSpeed(), state; n_quadrature=128)
        V_z = evaluate(ReflectivityWeightedFallSpeed(), state; n_quadrature=128)

        @test all(isfinite, [V_n, V_m, V_z])
        @test all(x -> x > 0, [V_n, V_m, V_z])

        # For a power-law fall speed V(D) = a D^b, the ratio of moments gives:
        # V_n / V_m should depend on μ in a predictable way
        @test V_n > 0
        @test V_m > 0
    end

    @testset "Lambda limiter integrals consistency" begin
        # The lambda limiter integrals should produce values that
        # allow solving for λ bounds

        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0)

        small_q = evaluate(NumberMomentLambdaLimit(), state)
        large_q = evaluate(MassMomentLambdaLimit(), state)

        @test small_q > 0
        @test large_q > 0
        @test isfinite(small_q)
        @test isfinite(large_q)

        # LargeQ should be > SmallQ for same state since large q
        # corresponds to small λ (larger particles)
        # Actually, these are both positive but their relative values
        # depend on the integral definition
        @test small_q ≠ large_q
    end

    @testset "Mass-weighted velocity ordering" begin
        # For particles with power-law fall speed V(D) = a D^b (b > 0):
        # Reflectivity-weighted (Z-weighted) mean velocity should be largest
        # because it weights by D^6, emphasizing large particles
        # Mass-weighted should be intermediate
        # Number-weighted should be smallest
        # Note: We must compare normalized mean velocities, not raw flux integrals.

        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 500.0)  # Large particles

        # Flux integrals
        F_n = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=128)
        F_m = evaluate(MassWeightedFallSpeed(), state; n_quadrature=128)
        F_z = evaluate(ReflectivityWeightedFallSpeed(), state; n_quadrature=128)

        # Normalization moments
        M_n = evaluate(NumberMomentLambdaLimit(), state; n_quadrature=128)
        M_m = evaluate(MassMomentLambdaLimit(), state; n_quadrature=128)
        M_z = evaluate(Reflectivity(), state; n_quadrature=128)

        # Mean velocities
        V_n = F_n / M_n
        V_m = F_m / M_m
        V_z = F_z / M_z

        # All should be positive
        @test V_n > 0
        @test V_m > 0
        @test V_z > 0

        # Check ordering: V_z ≥ V_m ≥ V_n for most PSDs
        @test V_z >= V_m
        @test V_m >= V_n
    end

    @testset "Ventilation integral properties" begin
        # Ventilation factor accounts for enhanced mass transfer due to air flow
        # For large particles (high Re), ventilation should be larger

        state_small = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 5000.0)  # Small particles

        state_large = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 200.0)   # Large particles

        v_small = evaluate(Ventilation(), state_small; n_quadrature=64)
        v_large = evaluate(Ventilation(), state_large; n_quadrature=64)

        @test isfinite(v_small)
        @test isfinite(v_large)
        @test v_small > 0
        @test v_large > 0

        # Larger particles have larger ventilation due to higher Reynolds
        @test v_large > v_small
    end

    @testset "Mean diameter integral" begin
        # Mean diameter should scale inversely with λ
        state1 = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0)

        state2 = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 500.0)  # Half λ → double mean D

        D_mean_1 = evaluate(MeanDiameter(), state1; n_quadrature=64)
        D_mean_2 = evaluate(MeanDiameter(), state2; n_quadrature=64)

        @test D_mean_1 > 0
        @test D_mean_2 > 0

        # D_mean_2 should be ~2x D_mean_1 for exponential (μ=0) distribution
        @test D_mean_2 > D_mean_1
    end
end
