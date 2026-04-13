using Test
using Breeze.Microphysics.PredictedParticleProperties:
    AerosolMode,
    AerosolActivation,
    activated_number,
    total_activated_number,
    sum_aerosol_number,
    prognostic_ccn_activation_rate

@testset "Aerosol Activation" begin
    FT = Float64

    @testset "AerosolMode construction" begin
        mode = AerosolMode(FT)
        # Default ammonium sulfate (Fortran P3): βact = vi * osm * epsm * mw * rhoa / (map * rhow)
        expected_beta = 3.0 * 1.0 * 0.9 * 0.018 * 1777.0 / (0.132 * 1000.0)
        @test mode.solute_activity ≈ expected_beta rtol=1e-10
        @test mode.number_mixing_ratio == 300e6
        @test mode.mean_radius == 0.05e-6
    end

    @testset "Single-mode activated number" begin
        mode = AerosolMode(FT)
        aerosol = AerosolActivation(mode)
        T = FT(280.0)
        S = FT(0.003)  # 0.3% supersaturation

        N_act = activated_number(mode, aerosol, T, S)
        # Must be positive and less than total aerosol
        @test N_act > 0
        @test N_act <= mode.number_mixing_ratio
        # At high supersaturation, nearly all aerosol activates
        N_high = activated_number(mode, aerosol, T, FT(0.05))
        @test N_high > 0.9 * mode.number_mixing_ratio
        # At zero supersaturation, none activates
        N_zero = activated_number(mode, aerosol, T, FT(0.0))
        @test N_zero ≈ 0 atol=1e-3
    end

    @testset "Multi-mode activation" begin
        mode1 = AerosolMode(FT; number_mixing_ratio=300e6, mean_radius=0.05e-6, geometric_std=2.0)
        mode2 = AerosolMode(FT; number_mixing_ratio=100e6, mean_radius=1.3e-6,  geometric_std=2.5)
        aerosol = AerosolActivation(mode1, mode2)

        T = FT(280.0)
        S = FT(0.003)

        N_total = total_activated_number(aerosol, T, S)
        @test N_total > 0
        @test N_total <= sum_aerosol_number(aerosol)
        @test sum_aerosol_number(aerosol) == 400e6
    end

    @testset "Prognostic CCN activation rate" begin
        mode = AerosolMode(FT)
        aerosol = AerosolActivation(mode)

        nᶜˡ = FT(100e6)    # current cloud number [kg⁻¹]
        qᵛ = FT(0.015)      # vapor mixing ratio [kg/kg]
        qᵛ⁺ˡ = FT(0.0145)   # saturation mixing ratio (supersaturated)
        T = FT(280.0)

        result = prognostic_ccn_activation_rate(aerosol, nᶜˡ, qᵛ, qᵛ⁺ˡ, T)

        # Supersaturated: should produce positive rates
        @test result.ncnuc > 0
        @test result.qcnuc > 0
        # Mass = number * seed mass
        seed_mass = 4π/3 * 1000 * (1e-6)^3
        @test result.qcnuc ≈ result.ncnuc * seed_mass rtol=1e-10

        # Subsaturated: should produce zero rates
        qᵛ_sub = FT(0.014)
        result_sub = prognostic_ccn_activation_rate(aerosol, nᶜˡ, qᵛ_sub, qᵛ⁺ˡ, T)
        @test result_sub.ncnuc == 0
        @test result_sub.qcnuc == 0
    end

    @testset "Float32 support" begin
        mode = AerosolMode(Float32)
        aerosol = AerosolActivation(mode)
        T = Float32(280.0)
        S = Float32(0.003)
        N_act = activated_number(mode, aerosol, T, S)
        @test N_act isa Float32
        @test N_act > 0
    end
end

@testset "Prognostic CCN integration with P3" begin
    using Breeze.Microphysics.PredictedParticleProperties:
        PredictedParticlePropertiesMicrophysics

    FT = Float64

    # Construct P3 with prognostic CCN
    p3 = PredictedParticlePropertiesMicrophysics(FT;
        aerosol = AerosolActivation(AerosolMode(FT)))

    @test !isnothing(p3.aerosol)
    @test length(p3.aerosol.modes) == 1

    # Construct P3 with prescribed CCN (default)
    p3_prescribed = PredictedParticlePropertiesMicrophysics(FT)
    @test isnothing(p3_prescribed.aerosol)
end
