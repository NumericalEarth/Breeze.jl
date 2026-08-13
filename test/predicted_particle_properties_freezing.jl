include(joinpath(@__DIR__, "setup.jl"))

using Test
import Breeze
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.AtmosphereModels: prognostic_field_names
using Breeze.Thermodynamics: ThermodynamicConstants, dry_air_gas_constant

using Breeze.Microphysics.PredictedParticleProperties:
    chebyshev_gauss_nodes_weights,
    TabulatedFunction6D,
    make_lookup_table,
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
    tendency_ρqʷⁱ,
    tendency_ρqᵛ,
    rain_autoconversion_rate,
    rain_accretion_rate,
    rain_evaporation_rate,
    rain_self_collection_rate,
    rain_breakup_rate,
    rain_terminal_velocities,
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

    #####
    ##### 6D lookup table interpolation (the Breeze-owned multilinear blend that
    ##### the Fortran rain-ice collection tables are read into)
    #####

    @testset "TabulatedFunction6D - construction and interpolation" begin
        f(x, y, z, w, v, u) = x * y + z * w + v * u
        FT = Float64

        n = 5
        axis = range(FT(0), FT(1); length=n)
        data = FT[f(axis[i], axis[j], axis[k], axis[l], axis[m], axis[p])
                  for i in 1:n, j in 1:n, k in 1:n, l in 1:n, m in 1:n, p in 1:n]
        ranges = ntuple(_ -> (FT(0), FT(1)), 6)
        f6d = make_lookup_table(data, ranges, CPU())

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
        constants = ThermodynamicConstants(FT)
        qr = FT(1e-3)
        nr = FT(1e4)
        T = FT(288.0)
        ρ = FT(1.0)
        P = FT(101325.0)
        qv_sat = FT(0.012)
        qv_sub = FT(0.008)   # 67% RH — subsaturated

        rate_sub = rain_evaporation_rate(p3_tab, qr, nr, qv_sub, qv_sat, T, ρ, P, constants)
        @test rate_sub > 0   # Positive magnitude (M7)

        # Saturated: zero evaporation
        rate_sat = rain_evaporation_rate(p3_tab, qr, nr, qv_sat, qv_sat, T, ρ, P, constants)
        @test rate_sat == 0
    end

    @testset "tabulated rain evaporation - positive, finite, bounded" begin
        # Verify PSD-integrated rain evaporation from Fortran tables
        # is physically reasonable.
        p3_tab = PredictedParticlePropertiesMicrophysics()

        FT = Float64
        constants = ThermodynamicConstants(FT)
        qr = FT(1e-3)
        nr = FT(1e4)
        T = FT(288.0)
        ρ = FT(1.0)
        P = FT(101325.0)
        qv_sat = FT(0.012)
        qv_sub = FT(0.008)

        rate_tab = rain_evaporation_rate(p3_tab, qr, nr, qv_sub, qv_sat, T, ρ, P, constants)

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

        V_large = rain_terminal_velocities(p3_tab, qr_large, nr_large_drops, ρ).mass_weighted
        V_small = rain_terminal_velocities(p3_tab, qr_large, nr_small_drops, ρ).mass_weighted

        @test V_large > 0
        @test V_small > 0
        @test V_large > V_small  # Larger drops fall faster
    end

    @testset "ice terminal velocity resolves sub-qmin mean particle mass (issue #6)" begin
        # The lookup coordinate is the mean *particle* mass m̄ = qⁱ/nⁱ [kg], which
        # must not be floored by the bulk mass-mixing-ratio threshold
        # `minimum_mass_mixing_ratio` (1e-14 kg/kg). The Table-1 mass axis extends
        # down to ≈1.56e-15 kg, and newly nucleated ice is ≈3.77e-15 kg. Fortran
        # (find_lookupTable_indices_1a) uses the raw m̄ and clamps only the table
        # index; the Julia table clamp reproduces that, so small ice must fall
        # slower than heavier ice instead of collapsing to a single speed.
        p3_tab = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        ρ  = FT(0.8)
        Fᶠ = FT(0.0)
        ρᶠ = FT(400.0)
        nⁱ = FT(1e6)

        # Mean masses spanning the (buggy) 1e-14 floor but inside the mass axis.
        m_new  = FT(3.77e-15)   # newly nucleated ice, below the 1e-14 floor
        m_mid  = FT(6e-15)      # still below the floor
        m_qmin = FT(1e-14)      # exactly the old floor value

        v(m) = PredictedParticleProperties.ice_terminal_velocity_mass_weighted(p3_tab, m * nⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

        # Before the fix all three collapse to a single speed (the 1e-14 coordinate).
        @test v(m_new) < v(m_mid) < v(m_qmin)
        @test v(m_new) > 0
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
        qᶜˡ = FT(1e-3)
        Nᶜˡ = FT(100e6)
        ρ = FT(1.2)
        T_cold = FT(230.0)
        frozen_mass_rate, frozen_number_rate = homogeneous_freezing_cloud_rate(p3, qᶜˡ, Nᶜˡ, T_cold, ρ)
        @test frozen_mass_rate > 0
        @test frozen_number_rate > 0

        # D25: Fortran has no mass-number consistency cap — all nc transfers to ice.
        # With trace qᶜˡ and large Nᶜˡ, freezing still activates.
        qᶜˡ_trace = FT(1e-7)
        Nᶜˡ_continental = FT(750e6)
        Q_trace, N_trace = homogeneous_freezing_cloud_rate(p3, qᶜˡ_trace, Nᶜˡ_continental, T_cold, ρ)
        @test Q_trace > 0
        @test N_trace > 0

        # Below threshold with qᶜˡ below guard (1e-14): zero rates
        Q_hom_tiny, N_hom_tiny = homogeneous_freezing_cloud_rate(p3, FT(1e-15), Nᶜˡ, T_cold, ρ)
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
        Q_at, N_at = homogeneous_freezing_cloud_rate(p3, qᶜˡ, Nᶜˡ, FT(233.15), ρ)
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
        Nᶜˡ = 100e6  # [1/m³]
        qᶜˡ = 1e-3  # [kg/kg]
        Q_frz, N_frz = immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜˡ, T, ρ)
        m_mean = qᶜˡ / (Nᶜˡ / ρ)  # mean drop mass [kg]
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
        Q_warm, N_warm = immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜˡ, 280.0, ρ)
        @test Q_warm == 0
        @test N_warm == 0

        # In the non-saturated regime, Barklie-Gokhale freezing is the Fortran
        # linear per-second rate. The safety cap only applies when the projected
        # sink would consume all available droplets.
        T_moderate = 230.0
        parameters = p3.process_rates
        nᶜˡ = Nᶜˡ / ρ
        V_drop = (qᶜˡ / nᶜˡ) / parameters.liquid_water_density
        linear_rate = parameters.immersion_freezing_nucleation_coefficient *
                      V_drop *
                      exp(parameters.immersion_freezing_coefficient *
                          (parameters.freezing_temperature - T_moderate))
        _, N_moderate = immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜˡ, T_moderate, ρ)
        @test linear_rate * parameters.sink_limiting_timescale < 1
        @test N_moderate ≈ nᶜˡ * linear_rate

        # Very cold supercell states make the Barklie-Gokhale exponential
        # effectively instantaneous. The raw rate is intentionally NOT capped at
        # 1/τ here (commit 48b073e3): the all-available-over-τ limit is applied
        # later by the combined cloud/rain budget in compute_p3_process_rates
        # (Fortran parity). The log-form overflow guard only keeps the extreme-cold
        # exponential finite, so at the rate-function level we require the moment
        # rates to be finite (no Float32 overflow) and non-negative.
        FT = Float32
        p3_32 = PredictedParticlePropertiesMicrophysics(FT)
        T_very_cold = FT(136.18727)
        ρ_cold = FT(0.12194309)

        qcl_trace = FT(4.8146696e-8)
        Q_cold, N_cold = immersion_freezing_cloud_rate(p3_32, qcl_trace,
                                                       p3_32.cloud.number_concentration,
                                                       T_very_cold, ρ_cold)
        @test isfinite(Q_cold)
        @test isfinite(N_cold)
        @test Q_cold >= 0
        @test N_cold >= 0

        qr_trace = FT(8.707481e-11)
        nr_trace = FT(130.94022)
        Q_rain_cold, N_rain_cold = immersion_freezing_rain_rate(p3_32, qr_trace,
                                                                nr_trace, T_very_cold, FT(0))
        @test isfinite(Q_rain_cold)
        @test isfinite(N_rain_cold)
        @test Q_rain_cold >= 0
        @test N_rain_cold >= 0
    end

    #####
    ##### Air transport properties tests (Phase A)
    #####

    @testset "Air transport properties - reference values" begin
        constants = ThermodynamicConstants()
        # T=273.15K, P=101325Pa: Dᵛ ≈ 2.23e-5, Kᵃ ≈ 0.024, ν ≈ 1.33e-5
        # Formula: Dᵛ = 8.794e-5 * T^1.81 / P, Kᵃ = 1414 * 1.496e-6 * T^1.5 / (T+120),
        #          ν  = Kᵃ / 1414 * 287.15 * T / P
        properties = air_transport_properties(273.15, 101325.0, constants)
        @test properties.Dᵛ ≈ 2.23e-5 atol=5e-7
        @test properties.Kᵃ ≈ 0.0243 atol=5e-4
        @test properties.ν ≈ 1.33e-5 atol=5e-7

        custom_constants = ThermodynamicConstants(; dry_air_molar_mass=0.03)
        custom_props = air_transport_properties(273.15, 101325.0, custom_constants)
        @test custom_props.Dᵛ == properties.Dᵛ
        @test custom_props.Kᵃ == properties.Kᵃ
        @test custom_props.ν / properties.ν ≈
              dry_air_gas_constant(custom_constants) / dry_air_gas_constant(constants)

        # T=250K, P=50000Pa: Dᵛ ≈ 3.85e-5 (colder T but much lower P → higher Dᵛ)
        props_cold_hi = air_transport_properties(250.0, 50000.0, constants)
        @test props_cold_hi.Dᵛ ≈ 3.85e-5 atol=5e-6
    end

    @testset "Air transport properties - monotonicity" begin
        constants = ThermodynamicConstants()
        # Dᵛ increases with T at fixed P
        props_cold = air_transport_properties(240.0, 101325.0, constants)
        props_warm = air_transport_properties(300.0, 101325.0, constants)
        @test props_warm.Dᵛ > props_cold.Dᵛ

        # Dᵛ decreases with P at fixed T
        props_lo_p = air_transport_properties(273.15, 50000.0, constants)
        props_hi_p = air_transport_properties(273.15, 101325.0, constants)
        @test props_lo_p.Dᵛ > props_hi_p.Dᵛ

        # Kᵃ increases with T (mu_air increases with T)
        @test props_warm.Kᵃ > props_cold.Kᵃ
    end

    @testset "Air transport properties - Float32 type stability" begin
        constants = ThermodynamicConstants(Float32)
        props32 = air_transport_properties(Float32(273.15), Float32(101325.0), constants)
        @test props32.Dᵛ isa Float32
        @test props32.Kᵃ isa Float32
        @test props32.ν isa Float32
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
    ##### ProcessRateParameters defaults
    #####
    ##### The immersion-freezing PSD correction is not stored in
    ##### `ProcessRateParameters`. `immersion_freezing_cloud_rate` evaluates
##### `psd_correction_spherical_volume` from the locally diagnosed Liu-Daum μᶜˡ,
    ##### so its correction varies with cloud droplet number.
    ##### `immersion_freezing_rain_rate` evaluates the same function at fixed μ_r = 0,
    ##### so the rain correction is constant.
    #####

    @testset "ProcessRateParameters defaults" begin
        parameters = ProcessRateParameters(Float64)

        @test parameters.reference_air_density ≈ 100000 / (dry_air_gas_constant(ThermodynamicConstants(Float64)) * 273.15) rtol=1e-12
        @test parameters.ice_nucleation_supersaturation_threshold == 0.05
        @test parameters.minimum_rain_slope == 500.0
        @test parameters.maximum_rain_slope == 100000.0
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
            FT(0),      # cloud_self_collection (0 for KK2000)
            FT(5e-8),   # rain_evaporation (positive magnitude)
            FT(0),      # rain_evaporation_number
            FT(1e-6),   # rain_self_collection (positive magnitude)
            FT(5e-7),   # rain_breakup
            FT(3e-7),   # deposition (bidirectional)
            FT(1e-8),   # partial_melting
            FT(5e-8),   # complete_melting
            FT(1e3),    # melting_number (positive magnitude)
            FT(0),      # clipping_dry_mass
            FT(0),      # clipping_rime_mass
            FT(0),      # clipping_rime_volume
            FT(0),      # post_process_clipping
            FT(0.0),    # sublimation_number (D2: nisub)
            FT(500.0),  # aggregation (positive magnitude)
            FT(0.0),    # ni_limit (C3: global Nⁱ cap)
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
            # wet_growth_shedding is nonzero ONLY in the dry (non-liquid-fraction)
            # branch, where it is the excess cloud collection shed to rain and
            # wet_growth_cloud/rain are zero (process_rates.jl:464-473). It is
            # mutually exclusive with the wet_growth_cloud/rain set above, so under
            # this default (liquid-fraction) routing it must be 0 — otherwise the
            # struct describes an unreachable state and double-charges cloud.
            FT(0.0),    # wet_growth_shedding (dry-branch only; 0 under LF routing)
            FT(0.0),    # wet_growth_shedding_number (dry-branch only)
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
