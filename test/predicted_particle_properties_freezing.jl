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

@testset "P3 Tabulated and Freezing" begin

    @testset "Tabulated sixth-moment melting matches Fortran branch split" begin
        FT = Float64
        p3_tab = PredictedParticlePropertiesMicrophysics()

        qⁱ = FT(1e-4)
        nⁱ = FT(1e5)
        zⁱ = FT(1e-8)
        ρ = FT(1.0)
        Fᶠ = FT(0.2)
        Fˡ = FT(0.3)
        ρᶠ = FT(400.0)
        ν = FT(1.5e-5)
        D_v = FT(2.2e-5)
        μ = FT(0.0)
        λ_r = FT(1e4)

        partial_only = P3ProcessRates(ntuple(index -> begin
            index == 8 ? FT(1e-7) : zero(FT)
        end, fieldcount(P3ProcessRates))...)

        complete_only = P3ProcessRates(ntuple(index -> begin
            index == 9 ? FT(1e-7) : zero(FT)
        end, fieldcount(P3ProcessRates))...)

        coat_cond_only = P3ProcessRates(ntuple(index -> begin
            index == 45 ? FT(1e-8) : zero(FT)
        end, fieldcount(P3ProcessRates))...)

        coat_evap_only = P3ProcessRates(ntuple(index -> begin
            index == 46 ? FT(1e-8) : zero(FT)
        end, fieldcount(P3ProcessRates))...)

        rain_rime_only = P3ProcessRates(ntuple(index -> begin
            index == 16 ? FT(1e-7) : zero(FT)
        end, fieldcount(P3ProcessRates))...)

        @test tendency_ρzⁱ(partial_only, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3_tab, ν, D_v, μ, zero(FT)) ≈ 0
        @test tendency_ρzⁱ(complete_only, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3_tab, ν, D_v, μ, zero(FT)) != 0

        lt1 = Breeze.Microphysics.PredictedParticleProperties.lookup_table_1(p3_tab)
        lt2 = Breeze.Microphysics.PredictedParticleProperties.lookup_table_2(p3_tab)
        log_m = log10(qⁱ / nⁱ)
        ρ_correction = Breeze.Microphysics.PredictedParticleProperties.ice_air_density_correction(
            p3_tab.ice.fall_speed.reference_air_density, ρ)
        sc_correction = Breeze.Microphysics.PredictedParticleProperties.ventilation_sc_correction(
            ν, D_v, ρ_correction)

        mass_dep_combined = lt1.deposition.ventilation(log_m, Fᶠ, Fˡ, ρᶠ, μ) +
                            sc_correction * lt1.deposition.ventilation_enhanced(log_m, Fᶠ, Fˡ, ρᶠ, μ)
        z_dep_combined = lt1.sixth_moment.deposition(log_m, Fᶠ, Fˡ, ρᶠ, μ) +
                         sc_correction * lt1.sixth_moment.deposition1(log_m, Fᶠ, Fˡ, ρᶠ, μ)
        z_sub_combined = lt1.sixth_moment.sublimation(log_m, Fᶠ, Fˡ, ρᶠ, μ) +
                         sc_correction * lt1.sixth_moment.sublimation1(log_m, Fᶠ, Fˡ, ρᶠ, μ)
        expected_coat_cond = ρ * z_dep_combined * FT(1e-8) / (nⁱ * mass_dep_combined)
        expected_coat_evap = -ρ * z_sub_combined * FT(1e-8) / (nⁱ * mass_dep_combined)

        @test tendency_ρzⁱ(coat_cond_only, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3_tab, ν, D_v, μ, zero(FT), λ_r) ≈ expected_coat_cond
        @test tendency_ρzⁱ(coat_evap_only, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3_tab, ν, D_v, μ, zero(FT), λ_r) ≈ expected_coat_evap

        # Rain riming Z uses Table 2 sixth_moment divided by the mass kernel (Fortran convention:
        # zqrcol = N0r × m6collr × env, while qrcol = N0r × 10^f1pr08 × Ni × env).
        log_λ_r = log10(λ_r)
        rain_mass_kernel = exp10(lt2.mass(log_m, log_λ_r, Fᶠ, Fˡ, ρᶠ, μ))
        expected_rain_rime = ρ * lt2.sixth_moment(log_m, log_λ_r, Fᶠ, Fˡ, ρᶠ, μ) * FT(1e-7) / (nⁱ * rain_mass_kernel)
        fallback_cloud_rime = ρ * lt1.sixth_moment.rime(log_m, Fᶠ, Fˡ, ρᶠ, μ) * FT(1e-7) / nⁱ
        rain_rime_tendency = tendency_ρzⁱ(rain_rime_only, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3_tab, ν, D_v, μ, zero(FT), λ_r)

        @test rain_rime_tendency ≈ expected_rain_rime
        @test !isapprox(rain_rime_tendency, fallback_cloud_rime)
    end

    #####
    ##### Rain PSD lookup table tests (TabulatedFunction1D)
    #####

    @testset "TabulatedFunction1D - smoke test" begin
        # Tabulate sin(x) on [0, π]
        x_min = 0.0
        x_max = π
        n = 100
        xs = range(x_min, x_max; length=n)
        values = sin.(xs)
        Δx = (x_max - x_min) / (n - 1)
        f = tabulated_function_1d(values, x_min, x_max, 1 / Δx)

        @test f isa TabulatedFunction1D

        # Check interpolated values match sin(x) within 1e-3 at 50 interior points
        test_points = range(x_min + 0.01, x_max - 0.01; length=50)
        for x in test_points
            @test abs(f(x) - sin(x)) < 1e-3
        end
    end

    @testset "TabulatedFunction1D - boundary clamping" begin
        n = 20
        xs = range(0.0, 1.0; length=n)
        values = xs .^ 2
        Δx = 1.0 / (n - 1)
        f = tabulated_function_1d(values, 0.0, 1.0, 1 / Δx)

        # Values outside range should clamp to boundary
        @test f(-1.0) ≈ f(0.0) atol=1e-10
        @test f(2.0) ≈ f(1.0) atol=1e-10
    end

    @testset "TabulatedFunction6D - construction and interpolation" begin
        f(x, y, z, w, v, u) = x * y + z * w + v * u
        FT = Float64

        f6d = TabulatedFunction6D(f, CPU(), FT;
                                   x_range=(0.0, 1.0), y_range=(0.0, 1.0),
                                   z_range=(0.0, 1.0), w_range=(0.0, 1.0),
                                   v_range=(0.0, 1.0), u_range=(0.0, 1.0),
                                   x_points=5, y_points=5, z_points=5,
                                   w_points=5, v_points=5, u_points=5)

        @test f6d isa TabulatedFunction6D
        @test size(f6d.table) == (5, 5, 5, 5, 5, 5)

        # Interpolation at grid points should be exact
        @test f6d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) ≈ 0.0
        @test f6d(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) ≈ 3.0
        @test f6d(0.5, 0.5, 0.5, 0.5, 0.5, 0.5) ≈ 0.75 atol=0.05

        # Clamping: out-of-range inputs should clamp to boundary values
        @test f6d(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0) ≈ f6d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        @test f6d(2.0, 0.0, 0.0, 0.0, 0.0, 0.0) ≈ f6d(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end

    @testset "RainMassWeightedVelocityEvaluator - monotonicity" begin
        evaluator = RainMassWeightedVelocityEvaluator()

        # λ_r = 1000 m⁻¹ → D_mean = 1mm (large drops, fast)
        # λ_r = 10000 m⁻¹ → D_mean = 100μm (small drops, slow)
        V_large = evaluator(log10(1000.0))
        V_small = evaluator(log10(10000.0))

        @test V_large > 0
        @test V_small > 0
        @test V_large > V_small  # Larger drops (small λ_r) fall faster
    end

    @testset "RainMassWeightedVelocityEvaluator - analytical comparison" begin
        # For simple power law V(D) = ar * D^br (valid ~134μm to 1.5mm):
        # V_mass = ar * Γ(4 + br) / (Γ(4) * λ_r^br)
        # At λ_r = 5000 m⁻¹ (D_mean = 200μm, intermediate drops):
        # ar = 842, br = 0.8 (Fortran P3 rain fall speed coefficients)
        using SpecialFunctions: gamma

        ar = 841.99667
        br = 0.8
        λ_r = 5000.0
        # Analytical: V_mass = ar * Γ(4+br) / (Γ(4) * λ^br)
        V_analytical = ar * gamma(4 + br) / (gamma(4) * λ_r^br)

        evaluator = RainMassWeightedVelocityEvaluator()
        V_numerical = evaluator(log10(λ_r))

        # Should agree within 30% (power law is approximate; piecewise formula differs)
        @test abs(V_numerical - V_analytical) / V_analytical < 0.30
    end

    @testset "RainNumberWeightedVelocityEvaluator - positive and monotone" begin
        evaluator = RainNumberWeightedVelocityEvaluator()

        V_large = evaluator(log10(1000.0))
        V_small = evaluator(log10(10000.0))

        @test V_large > 0
        @test V_small > 0
        @test V_large > V_small
    end

    @testset "RainEvaporationVentilationEvaluator - large λ_r limit" begin
        # M3: Evaluator now returns Reynolds integral only: I_Re = ∫ D √Re exp(-λD) dD
        # At λ_r → ∞ (tiny drops), √Re → 0, so I_Re → 0 (but stays positive).
        # The full evaporation integral is assembled at runtime:
        #   I_evap = f1r/λ² + f2r × Sc^(1/3) × I_Re
        evaluator = RainEvaporationVentilationEvaluator()

        λ_r = 1e5   # Large (very tiny drops)
        I_Re = evaluator(log10(λ_r))

        # Reynolds integral should be positive but small relative to 1/λ²
        @test I_Re > 0
        @test I_Re < 1.0 / λ_r^2   # upper bound: √Re contribution is small for tiny drops
    end

    @testset "RainEvaporationVentilationEvaluator - positive" begin
        evaluator = RainEvaporationVentilationEvaluator()

        for log_λ in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            I = evaluator(log_λ)
            @test I > 0
            @test isfinite(I)
        end
    end

    @testset "rain_evaporation_rate sign with tabulated scheme" begin
        # With tabulated rain, evaporation in subsaturated air should be positive magnitude (M7)
        p3_tab = PredictedParticlePropertiesMicrophysics()

        FT = Float64
        qr = FT(1e-3)
        nr = FT(1e4)
        T = FT(288.0)
        ρ = FT(1.0)
        P = FT(101325.0)
        qv_sat = FT(0.012)
        qv_sub = FT(0.008)   # 67% RH — subsaturated

        rate_sub = rain_evaporation_rate(p3_tab, qr, nr, qv_sub, qv_sat, T, ρ, P)
        @test rate_sub > 0   # Positive magnitude (M7)

        # Saturated: zero evaporation
        rate_sat = rain_evaporation_rate(p3_tab, qr, nr, qv_sat, qv_sat, T, ρ, P)
        @test rate_sat == 0
    end

    @testset "tabulated rain evaporation - positive, finite, bounded" begin
        # Verify PSD-integrated rain evaporation from Fortran tables
        # is physically reasonable.
        p3_tab = PredictedParticlePropertiesMicrophysics()

        FT = Float64
        qr = FT(1e-3)
        nr = FT(1e4)
        T = FT(288.0)
        ρ = FT(1.0)
        P = FT(101325.0)
        qv_sat = FT(0.012)
        qv_sub = FT(0.008)

        rate_tab = rain_evaporation_rate(p3_tab, qr, nr, qv_sub, qv_sat, T, ρ, P)

        # Should be positive magnitude (M7) and finite
        @test rate_tab > 0
        @test isfinite(rate_tab)

        # Physically reasonable (not zero, not astronomical)
        @test rate_tab < 1.0   # Cannot evaporate more than all rain per second
    end

    @testset "tabulated rain terminal velocity - positive and monotone" begin
        p3_tab = PredictedParticlePropertiesMicrophysics()

        FT = Float64
        ρ = FT(1.0)

        # Large drops (small nr relative to qr → large mean mass)
        qr_large = FT(1e-3)
        nr_large_drops = FT(1e2)   # Few large drops

        # Small drops (many drops for same qr → small mean mass)
        nr_small_drops = FT(1e5)   # Many small drops

        V_large = rain_terminal_velocity_mass_weighted(p3_tab, qr_large, nr_large_drops, ρ)
        V_small = rain_terminal_velocity_mass_weighted(p3_tab, qr_large, nr_small_drops, ρ)

        @test V_large > 0
        @test V_small > 0
        @test V_large > V_small  # Larger drops fall faster
    end

    @testset "Homogeneous freezing" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        # --- homogeneous_freezing_cloud_rate ---

        # Above threshold (T = 240 K > 233.15 K): all rates must be zero
        Q_hom, N_hom = homogeneous_freezing_cloud_rate(p3, FT(1e-3), FT(100e6), FT(240.0), FT(1.0))
        @test Q_hom == 0
        @test N_hom == 0

        # Below threshold (T = 230 K): cloud freezing activates
        qcl = FT(1e-3)
        Nc = FT(100e6)
        ρ = FT(1.2)
        T_cold = FT(230.0)
        Q_hom, N_hom = homogeneous_freezing_cloud_rate(p3, qcl, Nc, T_cold, ρ)
        @test Q_hom > 0
        @test N_hom > 0

        # D25: Fortran has no mass-number consistency cap — all nc transfers to ice.
        # With trace qᶜˡ and large Nc, freezing still activates.
        qcl_trace = FT(1e-7)
        Nc_continental = FT(750e6)
        Q_trace, N_trace = homogeneous_freezing_cloud_rate(p3, qcl_trace, Nc_continental, T_cold, ρ)
        @test Q_trace > 0
        @test N_trace > 0

        # Below threshold with qᶜˡ below guard (1e-14): zero rates
        Q_hom_tiny, N_hom_tiny = homogeneous_freezing_cloud_rate(p3, FT(1e-15), Nc, T_cold, ρ)
        @test Q_hom_tiny == 0
        @test N_hom_tiny == 0

        # --- homogeneous_freezing_rain_rate ---

        # Above threshold (T = 240 K > 233.15 K): all rates must be zero
        Q_hom_r, N_hom_r = homogeneous_freezing_rain_rate(p3, FT(1e-3), FT(1e4), FT(240.0))
        @test Q_hom_r == 0
        @test N_hom_r == 0

        # Below threshold (T = 220 K): rain freezing activates
        qr = FT(1e-3)
        nr = FT(1e4)
        T_very_cold = FT(220.0)

        Q_hom_r, N_hom_r = homogeneous_freezing_rain_rate(p3, qr, nr, T_very_cold)
        @test Q_hom_r > 0
        @test N_hom_r > 0

        # Below threshold with qʳ below guard (1e-14): zero rates
        Q_hom_r_tiny, N_hom_r_tiny = homogeneous_freezing_rain_rate(p3, FT(1e-15), nr, T_very_cold)
        @test Q_hom_r_tiny == 0
        @test N_hom_r_tiny == 0

        # Exactly at threshold (T = 233.15 K): should be zero (guard is T < T_threshold)
        Q_at, N_at = homogeneous_freezing_cloud_rate(p3, qcl, Nc, FT(233.15), ρ)
        @test Q_at == 0

        # --- Type stability ---
        # Float32 inputs produce Float32 outputs
        Q32, N32 = homogeneous_freezing_cloud_rate(p3, Float32(1e-3), Float32(100e6), Float32(230.0), Float32(1.2))
        @test Q32 isa Float32
        @test N32 isa Float32

        Q32r, N32r = homogeneous_freezing_rain_rate(p3, Float32(1e-3), Float32(1e4), Float32(220.0))
        @test Q32r isa Float32
        @test N32r isa Float32
    end

    @testset "Immersion freezing PSD weighting (H1)" begin
        p3 = PredictedParticlePropertiesMicrophysics(Float64)

        # Cloud immersion freezing: PSD correction on mass only.
        # Large drops freeze preferentially, so mean frozen mass > mean drop mass.
        T = 260.0
        ρ = 1.0
        Nc = 100e6  # [1/m³]
        qcl = 1e-3  # [kg/kg]
        Q_frz, N_frz = immersion_freezing_cloud_rate(p3, qcl, Nc, T, ρ)
        m_mean = qcl / (Nc / ρ)  # mean drop mass [kg]
        @test Q_frz / max(N_frz, 1e-30) > m_mean
        @test Q_frz > 0
        @test N_frz > 0

        # Rain immersion freezing: same split (PSD correction on mass only).
        qr = 1e-3
        nr = 1e4   # [1/kg]
        μ_r = 0.0  # exponential rain PSD (Fortran P3 v5.5.0 mu_r_constant = 0)
        Q_frz_r, N_frz_r = immersion_freezing_rain_rate(p3, qr, nr, T, μ_r)
        m_mean_r = qr / nr
        @test Q_frz_r / max(N_frz_r, 1e-30) > m_mean_r
        @test Q_frz_r > 0
        @test N_frz_r > 0

        # Above threshold temperature: zero rates
        Q_warm, N_warm = immersion_freezing_cloud_rate(p3, qcl, Nc, 280.0, ρ)
        @test Q_warm == 0
        @test N_warm == 0
    end

    #####
    ##### Air transport properties tests (Phase A)
    #####

    @testset "Air transport properties - reference values" begin
        # T=273.15K, P=101325Pa: D_v ≈ 2.23e-5, K_a ≈ 0.024, nu ≈ 1.33e-5
        # Formula: D_v = 8.794e-5 * T^1.81 / P, K_a = 1414 * 1.496e-6 * T^1.5 / (T+120),
        #          nu  = K_a / 1414 * 287.15 * T / P
        props = air_transport_properties(273.15, 101325.0)
        @test props.D_v ≈ 2.23e-5 atol=5e-7
        @test props.K_a ≈ 0.0243 atol=5e-4
        @test props.nu ≈ 1.33e-5 atol=5e-7

        # T=250K, P=50000Pa: D_v ≈ 3.85e-5 (colder T but much lower P → higher D_v)
        props_cold_hi = air_transport_properties(250.0, 50000.0)
        @test props_cold_hi.D_v ≈ 3.85e-5 atol=5e-6
    end

    @testset "Air transport properties - monotonicity" begin
        # D_v increases with T at fixed P
        props_cold = air_transport_properties(240.0, 101325.0)
        props_warm = air_transport_properties(300.0, 101325.0)
        @test props_warm.D_v > props_cold.D_v

        # D_v decreases with P at fixed T
        props_lo_p = air_transport_properties(273.15, 50000.0)
        props_hi_p = air_transport_properties(273.15, 101325.0)
        @test props_lo_p.D_v > props_hi_p.D_v

        # K_a increases with T (mu_air increases with T)
        @test props_warm.K_a > props_cold.K_a
    end

    @testset "Air transport properties - Float32 type stability" begin
        props32 = air_transport_properties(Float32(273.15), Float32(101325.0))
        @test props32.D_v isa Float32
        @test props32.K_a isa Float32
        @test props32.nu isa Float32
    end

    #####
    ##### PSD correction for spherical volume (Phase B)
    #####

    @testset "psd_correction_spherical_volume - exact values" begin
        # mu=0: Γ(7)*Γ(1) / Γ(4)² = 720 * 1 / 36 = 20.0 (exact)
        @test psd_correction_spherical_volume(0.0) ≈ 20.0 atol=1e-10

        # mu=2: Γ(9)*Γ(3) / Γ(6)² = 40320 * 2 / 14400 = 5.6 (exact)
        @test psd_correction_spherical_volume(2.0) ≈ 5.6 atol=1e-6

        # mu=5: value is smaller (distribution narrows → less enhancement)
        val_mu5 = psd_correction_spherical_volume(5.0)
        @test val_mu5 ≈ 2.945 atol=0.01
        @test isfinite(val_mu5)
    end

    @testset "psd_correction_spherical_volume - monotonicity" begin
        # Correction decreases with increasing mu (narrower distribution → less PSD broadening)
        vals = [psd_correction_spherical_volume(Float64(mu)) for mu in 0:10]
        for i in 2:length(vals)
            @test vals[i] < vals[i-1]
        end
        # All values must be positive and finite
        @test all(isfinite, vals)
        @test all(v -> v > 0, vals)
    end

    @testset "psd_correction_spherical_volume - Float32 type stability" begin
        val32 = psd_correction_spherical_volume(Float32(0.0))
        @test val32 isa Float32
        @test val32 ≈ Float32(20.0) atol=Float32(1e-3)
    end

    @testset "psd_correction_spherical_volume - analytical identity at mu=0" begin
        # At mu=0 the formula gives exp(loggamma(7) + loggamma(1) - 2*loggamma(4))
        # = exp(log(720) + log(1) - 2*log(6)) = 720 / 36 = 20
        @test psd_correction_spherical_volume(0.0) ≈ 20.0 rtol=1e-12
    end

    #####
    ##### ProcessRateParameters default PSD correction values (Phase B + Step 4)
    #####

    @testset "ProcessRateParameters PSD correction defaults" begin
        prp = ProcessRateParameters(Float64)

        # freezing_cloud_psd_correction: psd_correction_spherical_volume(2.3) ≈ 5.08
        @test prp.freezing_cloud_psd_correction ≈ psd_correction_spherical_volume(2.3) rtol=1e-6

        # freezing_rain_psd_correction: psd_correction_spherical_volume(0.0)
        # Gamma(7)*Gamma(1)/Gamma(4)^2 = 720/36 = 20.0
        @test prp.freezing_rain_psd_correction ≈ psd_correction_spherical_volume(0.0) rtol=1e-6
        @test prp.freezing_rain_psd_correction ≈ 20.0 atol=0.01

        @test prp.reference_air_density ≈ 100000 / (dry_air_gas_constant(ThermodynamicConstants(Float64)) * 273.15) rtol=1e-12
        @test prp.nucleation_supersaturation_threshold == 0.05
        @test prp.rain_lambda_min == 500.0
        @test prp.rain_lambda_max == 100000.0

        # riming_psd_correction should remain unchanged at 2.0
        @test prp.riming_psd_correction ≈ 2.0
    end

    @testset "Vapor + cloud + rain + ice mass conservation" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        ρ = FT(1.0)

        # Create rates with typical mixed-phase values, including homogeneous freezing
        # Sign convention (M7): all one-directional rates are positive magnitudes
        rates = P3ProcessRates(
            FT(5e-7),   # condensation (bidirectional)
            FT(1e-7),   # autoconversion
            FT(2e-7),   # accretion
            FT(5e-8),   # rain_evaporation (positive magnitude)
            FT(1e-6),   # rain_self_collection (positive magnitude)
            FT(5e-7),   # rain_breakup
            FT(3e-7),   # deposition (bidirectional)
            FT(1e-8),   # partial_melting
            FT(5e-8),   # complete_melting
            FT(1e3),    # melting_number (positive magnitude)
            FT(0.0),    # sublimation_number (D2: nisub)
            FT(500.0),  # aggregation (positive magnitude)
            FT(0.0),    # ni_limit (C3: global Nᵢ cap)
            FT(1e-7),   # cloud_riming
            FT(1e4),    # cloud_riming_number (positive magnitude)
            FT(5e-8),   # rain_riming
            FT(500.0),  # rain_riming_number (positive magnitude)
            FT(300.0),  # rime_density_new
            FT(2e-8),   # shedding
            FT(100.0),  # shedding_number
            FT(1e-8),   # refreezing
            FT(1e-9),   # nucleation_mass
            FT(10.0),   # nucleation_number
            FT(5e-9),   # cloud_freezing_mass
            FT(100.0),  # cloud_freezing_number
            FT(3e-9),   # rain_freezing_mass
            FT(50.0),   # rain_freezing_number
            FT(1e-10),  # splintering_mass
            FT(1.0),    # splintering_number
            FT(2e-7),   # cloud_homogeneous_mass
            FT(1e5),    # cloud_homogeneous_number
            FT(1e-7),   # rain_homogeneous_mass
            FT(500.0),  # rain_homogeneous_number
            FT(1e-8),   # cloud_warm_collection (above-freezing cloud collection → qʷⁱ)
            FT(1e4),    # cloud_warm_collection_number
            FT(5e-9),   # rain_warm_collection (above-freezing rain collection → qʷⁱ)
            FT(1e2),    # rain_warm_collection_number (M9)
            FT(3e-8),   # wet_growth_cloud (cloud riming redirected to qʷⁱ)
            FT(2e-8),   # wet_growth_rain (rain riming redirected to qʷⁱ)
            FT(1e-8),   # wet_growth_shedding (D8: excess → rain)
            FT(1e-8 * 1.923e6),  # wet_growth_shedding_number (D8)
            FT(0.0),    # ccn_activation_mass (M9 stub)
            FT(0.0),    # ccn_activation_number (M9 stub)
            FT(0.0),    # rain_condensation (M9 stub)
            FT(0.0),    # coating_condensation (M9 stub)
            FT(0.0),    # coating_evaporation (M9 stub)
            FT(0.0),    # wet_growth_densification_mass (H9)
            FT(0.0),    # wet_growth_densification_volume (H9)
            FT(0.0),    # cloud_number_correction (M6)
            FT(0.0),    # rain_number_correction (M6)
        )

        # Compute total water tendency: vapor + cloud + rain + ice + liquid_on_ice
        # These should sum to zero (water is neither created nor destroyed)
        dqv = tendency_ρqᵛ(rates, ρ)
        dqc = tendency_ρqᶜˡ(rates, ρ)
        dqr = tendency_ρqʳ(rates, ρ)
        dqi = tendency_ρqⁱ(rates, ρ)
        dqwi = tendency_ρqʷⁱ(rates, ρ)

        total_water_tendency = dqv + dqc + dqr + dqi + dqwi
        @test abs(total_water_tendency) < 1e-15 * ρ
    end

end
