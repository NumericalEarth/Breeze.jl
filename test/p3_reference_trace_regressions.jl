include(joinpath(@__DIR__, "setup.jl"))

using Test
import Breeze

const PPP = Breeze.Microphysics.PredictedParticleProperties

@testset "P3 trace-regime behavior" begin
    @testset "Float32 rain quadrature retains the mass-weighted velocity" begin
        evaluator_32 = PPP.RainMassWeightedVelocity(Float32)
        evaluator_64 = PPP.RainMassWeightedVelocity(Float64)

        for λ_r in (500, 1_000, 10_000, 100_000)
            velocity_32 = evaluator_32(log10(Float32(λ_r)))
            velocity_64 = evaluator_64(log10(Float64(λ_r)))

            @test velocity_32 > 0
            @test isapprox(Float64(velocity_32), velocity_64; rtol=5e-4)
        end
    end

    p3 = PPP.PredictedParticlePropertiesMicrophysics(Float64)
    qsmall = p3.minimum_mass_mixing_ratio
    qtrace = qsmall / 2
    ρ = 0.8

    @testset "warm-rain qsmall gates" begin
        # These are the KK2000 gates; the default selection is what puts them under test.
        @test p3.warm_rain_scheme isa PPP.KhairoutdinovKogan2000

        @test PPP.rain_accretion_rate(p3, qtrace, qsmall, ρ) == 0
        @test PPP.rain_accretion_rate(p3, qsmall, qtrace, ρ) == 0
        @test PPP.rain_accretion_rate(p3, qsmall, qsmall, ρ) > 0

        @test PPP.rain_self_collection_rate(p3, qtrace, 1e3, ρ) == 0
        @test PPP.rain_self_collection_rate(p3, qsmall, 1e3, ρ) > 0
        @test PPP.rain_breakup_rate(p3, qtrace, 1e3, 1.0) == 0

        # KK2000 carries no explicit cloud self-collection sink.
        @test PPP.cloud_self_collection_rate(p3, qsmall, 1e8, ρ) == 0
    end

    @testset "terminal-velocity qsmall gates" begin
        rain_trace = PPP.rain_terminal_velocities(p3, qtrace, 1e2, ρ)
        rain_active = PPP.rain_terminal_velocities(p3, qsmall, 1e2, ρ)

        @test rain_trace.mass_weighted == 0
        @test rain_trace.number_weighted == 0
        @test rain_active.mass_weighted > 0
        @test rain_active.number_weighted > 0

        ice_trace = PPP.ice_terminal_velocities(p3, qtrace, 1e2, 0.0, 400.0, ρ)
        ice_active = PPP.ice_terminal_velocities(p3, qsmall, 1e2, 0.0, 400.0, ρ)

        @test ice_trace.mass_weighted == 0
        @test ice_trace.number_weighted == 0
        @test ice_active.mass_weighted > 0
        @test ice_active.number_weighted > 0
        @test PPP.ice_terminal_velocity_mass_weighted(p3, qtrace, 1e2, 0.0, 400.0, ρ) == 0
    end

    @testset "aggregation and riming use mass-only activity gates" begin
        temperature_cold = 268.0
        temperature_warm = 278.0
        rime_fraction = 0.0
        rime_density = 400.0
        active_ice_mass = 1e-8

        @test PPP.ice_aggregation_rate(p3, qtrace, 0.5, temperature_cold,
                                       rime_fraction, rime_density, ρ) == 0
        @test PPP.ice_aggregation_rate(p3, qsmall, 0.5, temperature_cold,
                                       rime_fraction, rime_density, ρ) > 0

        @test PPP.cloud_riming_rate(p3, qsmall, active_ice_mass, 0.5, temperature_cold,
                                    rime_fraction, rime_density, ρ) > 0
        @test PPP.cloud_riming_rate(p3, qtrace, active_ice_mass, 0.5, temperature_cold,
                                    rime_fraction, rime_density, ρ) == 0
        @test PPP.cloud_warm_collection_rate(p3, qsmall, active_ice_mass, 0.5, temperature_warm,
                                             rime_fraction, rime_density, ρ)[1] > 0

        @test PPP.rain_riming_rate(p3, qsmall, 0.5, qsmall, 0.5, temperature_cold,
                                   rime_fraction, rime_density, ρ) > 0
        @test PPP.rain_riming_number_rate(p3, qsmall, 0.5, qsmall, 0.5,
                                          temperature_cold, rime_fraction, rime_density,
                                          ρ) > 0
        @test PPP.rain_warm_collection_rate(p3, qsmall, 0.5, qsmall, 0.5,
                                            temperature_warm, rime_fraction, rime_density,
                                            ρ) > 0
        @test PPP.rain_riming_rate(p3, qtrace, 0.5, qsmall, 0.5, temperature_cold,
                                   rime_fraction, rime_density, ρ) == 0
    end
end
