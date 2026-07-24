using Test

import Breeze
using Breeze.Microphysics.PredictedParticleProperties

const PPP = Breeze.Microphysics.PredictedParticleProperties

@testset "P3 three-moment audit regressions" begin
    FT = Float32
    p3 = PredictedParticlePropertiesMicrophysics(FT; three_moment_ice = true)

    @testset "Table-3 coordinates preserve small dimensional moments" begin
        qⁱ = FT(1e-9)
        nⁱ = FT(1e-1)
        zⁱ = FT(1e-19)
        Fᶠ = FT(0)
        Fˡ = FT(0)
        ρᶠ = FT(0)

        table = PPP.three_moment_shape_table(p3)
        expected = table.shape(log10(zⁱ / qⁱ), ρᶠ, log10(qⁱ / nⁱ), Fᶠ, Fˡ)
        actual = @inferred PPP.compute_ice_shape_parameter(p3, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ)

        @test qⁱ < eps(FT)
        @test zⁱ < eps(FT)
        @test actual isa FT
        @test actual ≈ expected rtol=FT(5e-6)
        @test actual > 0
    end

    @testset "Sixth-moment bounds use Table-3 mean density" begin
        qⁱ = FT(1e-9)
        nⁱ = FT(1e-1)
        zⁱ = FT(1e-18)
        Fᶠ = FT(0)
        Fˡ = FT(0)
        ρᶠ = FT(0)
        μ = PPP.compute_ice_shape_parameter(p3, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ)

        table3 = PPP.three_moment_shape_table(p3)
        expected_density = table3.mean_density(log10(zⁱ / qⁱ), ρᶠ,
                                               log10(qⁱ / nⁱ), Fᶠ, Fˡ)
        table3_density = @inferred PPP.ice_mean_density(p3, qⁱ, nⁱ, zⁱ,
                                                      Fᶠ, Fˡ, ρᶠ, μ)
        table1_density = PPP.ice_mean_density_for_bounds(PPP.ice_integrals_table(p3),
                                                        qⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ, μ)

        bounded = PPP.bound_ice_sixth_moment(p3, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, μ)
        expected_table3_bound = PPP.enforce_z_bounds(zⁱ, qⁱ, nⁱ, table3_density,
                                                    FT(0), FT(20))
        stale_table1_bound = PPP.enforce_z_bounds(zⁱ, qⁱ, nⁱ, table1_density,
                                                 FT(0), FT(20))

        @test table3_density ≈ expected_density rtol=FT(5e-6)
        @test !isapprox(table3_density, table1_density; rtol=FT(1e-3))
        @test bounded ≈ expected_table3_bound rtol=FT(5e-6)
        @test !isapprox(bounded, stale_table1_bound; rtol=FT(1e-3))
    end

    @testset "Sixth-moment writeback preserves Fortran limiter ordering" begin
        ρ = FT(1)
        qⁱ = FT(2e-4)
        nⁱ = FT(1e-2)
        zⁱ = FT(1e-20)
        state = PPP.P3MicrophysicalState(
            FT(0), FT(0), FT(0), FT(0), qⁱ, nⁱ,
            FT(0), FT(0), zⁱ, FT(0), FT(0), FT(0), FT(0))

        bounds = @inferred PPP.p3_ice_moment_bounds(
            p3, ρ, qⁱ, nⁱ, zⁱ, FT(0), FT(0), FT(0))
        stale_μ = PPP.compute_ice_shape_parameter(
            p3, qⁱ, nⁱ, zⁱ, FT(0), FT(0), FT(0))
        stale_bound = PPP.bound_ice_sixth_moment(
            p3, qⁱ, nⁱ, zⁱ, FT(0), FT(0), FT(0), stale_μ)

        μ_fields = (;ρz̃ⁱ = reshape(FT[0], 1, 1, 1))
        bounded_state = @inferred PPP.clamp_ice_sixth_moment_dispatch(
            PPP.three_moment_shape_table(p3), μ_fields, 1, 1, 1, p3, ρ, state)
        reconstructed_zⁱ = (μ_fields.ρz̃ⁱ[1, 1, 1] / ρ)^2 / nⁱ

        @test bounds.nⁱ != nⁱ
        @test !isapprox(bounds.zⁱ, stale_bound; rtol=FT(1e-3))
        @test bounded_state.zⁱ ≈ bounds.zⁱ rtol=FT(5e-6)
        @test reconstructed_zⁱ ≈ bounds.zⁱ rtol=FT(5e-6)
    end

    @testset "Small sixth moments retain their physical tendencies" begin
        nⁱ = FT(1e3)
        zⁱ = FT(1e-25)
        tendency_ρz = FT(1e-26)
        tendency_ρn = FT(1)
        expected_z̃_tendency = (nⁱ * tendency_ρz + zⁱ * tendency_ρn) /
                              (2 * sqrt(zⁱ * nⁱ))

        z̃_tendency = @inferred PPP.z̃ⁱ_tendency(nⁱ, zⁱ, tendency_ρz, tendency_ρn)
        @test z̃_tendency isa FT
        @test z̃_tendency ≈ expected_z̃_tendency rtol=FT(5e-6)

        source_mass = FT(1e-10)
        source_number = FT(1e-8)
        μ_source = FT(2)
        moment3_source = source_mass * FT(6) / (FT(900) * FT(π))
        expected_source = PPP.g_of_mu(μ_source) * moment3_source^2 / source_number
        source = @inferred PPP.initiated_ice_sixth_moment_tendency(source_mass, source_number,
                                                                  μ_source)

        @test source_number < eps(FT)
        @test source isa FT
        @test source ≈ expected_source rtol=FT(5e-6)
    end

    @testset "Tabulated sublimation does not use an epsilon denominator" begin
        rate_names = fieldnames(PPP.P3ProcessRates)
        rates = PPP.P3ProcessRates(ntuple(i ->
            rate_names[i] === :deposition ? FT(-1e-12) : zero(FT),
            fieldcount(PPP.P3ProcessRates))...)

        ρ = FT(1)
        qⁱ = FT(1e-16)
        nⁱ = FT(1e-2)
        zⁱ = FT(1e-20)
        Fᶠ = FT(0)
        Fˡ = FT(0)
        ρᶠ = FT(400)
        μ = FT(10)
        nu = FT(1.5e-5)
        D_v = FT(2e-5)

        table = PPP.ice_integrals_table(p3)
        log_mean_mass = log10(qⁱ / nⁱ)
        ρ_correction = PPP.ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)
        sc_correction = PPP.ventilation_sc_correction(nu, D_v, ρ_correction)
        prep = PPP.prepare_5d(table.deposition.ventilation,
                             log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
        mass_integral = PPP.evaluate_at(table.deposition.ventilation, prep) +
                        sc_correction * PPP.evaluate_at(table.deposition.ventilation_enhanced, prep)
        z_integral = PPP.evaluate_at(table.sixth_moment.sublimation, prep) +
                     sc_correction * PPP.evaluate_at(table.sixth_moment.sublimation1, prep)
        denominator = nⁱ * mass_integral
        expected = -ρ * max(0, z_integral) * abs(rates.deposition) / denominator

        actual = PPP.tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ,
                                  p3, nu, D_v, μ, FT(0))

        @test 0 < denominator < eps(FT)
        @test actual ≈ expected rtol=FT(5e-6)
    end

    @testset "Homogeneous cloud freezing uses cloud shape" begin
        rate_names = fieldnames(PPP.P3ProcessRates)
        cloud_mass = FT(5e-10)
        cloud_number = FT(50)
        rain_mass = FT(6e-10)
        rain_number = FT(60)
        rates = PPP.P3ProcessRates(ntuple(i -> begin
            name = rate_names[i]
            name === :cloud_homogeneous_mass ? cloud_mass :
            name === :cloud_homogeneous_number ? cloud_number :
            name === :rain_homogeneous_mass ? rain_mass :
            name === :rain_homogeneous_number ? rain_number :
            zero(FT)
        end, fieldcount(PPP.P3ProcessRates))...)

        μ_rain = FT(0)
        μ_cloud = FT(10)
        expected = PPP.initiated_ice_sixth_moment_tendency(cloud_mass, cloud_number, μ_cloud) +
                   PPP.initiated_ice_sixth_moment_tendency(rain_mass, rain_number, μ_rain)
        stale_rain_shape = PPP.initiated_ice_sixth_moment_tendency(cloud_mass, cloud_number, μ_rain) +
                           PPP.initiated_ice_sixth_moment_tendency(rain_mass, rain_number, μ_rain)
        actual = @inferred PPP.group2_ice_sixth_moment_tendency(rates, p3.process_rates,
                                                               μ_rain, μ_cloud)

        @test actual isa FT
        @test actual ≈ expected rtol=FT(5e-6)
        @test !isapprox(actual, stale_rain_shape; rtol=FT(1e-3))
    end
end
