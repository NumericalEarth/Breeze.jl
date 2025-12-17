using Breeze
using Breeze.Microphysics: KesslerMicrophysics, prognostic_field_names
using Test

# Import helper functions directly from the module
const BM = Breeze.Microphysics

@testset "KesslerMicrophysics construction" begin
    @testset "Default construction [$(FT)]" for FT in (Float32, Float64)
        km = KesslerMicrophysics(FT)
        
        @test km isa KesslerMicrophysics{FT}
        @test km.autoconversion_rate == FT(0.001)
        @test km.autoconversion_threshold == FT(0.001)
        @test km.accretion_rate == FT(2.2)
    end

    @testset "Custom parameters [$(FT)]" for FT in (Float32, Float64)
        km = KesslerMicrophysics(FT;
            autoconversion_rate = 0.002,
            autoconversion_threshold = 0.0005,
            accretion_rate = 3.0
        )
        
        @test km.autoconversion_rate == FT(0.002)
        @test km.autoconversion_threshold == FT(0.0005)
        @test km.accretion_rate == FT(3.0)
    end
end

@testset "KesslerMicrophysics interface" begin
    @testset "Prognostic field names" begin
        km = KesslerMicrophysics()
        names = prognostic_field_names(km)
        # Only cloud liquid and rain are prognostic; vapor is diagnosed from qᵗ
        @test names == (:ρqᶜˡ, :ρqʳ)
    end
end

@testset "Mass fraction ↔ mixing ratio conversion" begin
    @testset "mass_fraction_to_mixing_ratio" begin
        # Simple case: no moisture → division by 1
        @test BM.mass_fraction_to_mixing_ratio(0.01, 0.0) ≈ 0.01
        
        # With 2% total moisture → division by 0.98
        qᵗ = 0.02
        q = 0.01
        r = BM.mass_fraction_to_mixing_ratio(q, qᵗ)
        @test r ≈ q / (1 - qᵗ)
    end
    
    @testset "mixing_ratio_to_mass_fraction" begin
        qᵗ = 0.02
        r = 0.001  # mixing ratio (or tendency)
        q = BM.mixing_ratio_to_mass_fraction(r, qᵗ)
        @test q ≈ r * (1 - qᵗ)
    end
end

@testset "Source term calculations" begin
    @testset "Autoconversion rate" begin
        km = KesslerMicrophysics(Float64)
        
        # Below threshold: no autoconversion
        @test BM.autoconversion_rate(0.0005, km) == 0.0
        
        # At threshold: no autoconversion
        @test BM.autoconversion_rate(0.001, km) == 0.0
        
        # Above threshold: positive autoconversion
        Aₖ = BM.autoconversion_rate(0.002, km)
        @test Aₖ ≈ 0.001 * (0.002 - 0.001) atol=1e-10
    end

    @testset "Accretion rate" begin
        km = KesslerMicrophysics(Float64)
        
        # No cloud water: no accretion
        @test BM.accretion_rate(0.0, 0.001, km) == 0.0
        
        # No rain: no accretion
        @test BM.accretion_rate(0.001, 0.0, km) == 0.0
        
        # Both present: positive accretion
        qˡ = 0.001
        qʳ = 0.001
        Kₖ = BM.accretion_rate(qˡ, qʳ, km)
        @test Kₖ ≈ 2.2 * qˡ * qʳ^0.875 atol=1e-10
    end

    @testset "Condensation rate" begin
        # Supersaturated: condensation occurs
        qᵛ = 0.012
        qᵛ⁺ = 0.010
        D = 1.5
        Cₖ = BM.condensation_rate(qᵛ, qᵛ⁺, D)
        @test Cₖ ≈ (qᵛ - qᵛ⁺) / D atol=1e-10
        
        # Subsaturated: no condensation
        Cₖ = BM.condensation_rate(0.008, 0.010, D)
        @test Cₖ == 0.0
    end

    @testset "Cloud evaporation rate" begin
        D = 1.5
        
        # Supersaturated: no evaporation
        Eₖ = BM.cloud_evaporation_rate(0.012, 0.001, 0.010, D)
        @test Eₖ == 0.0
        
        # Subsaturated with cloud water: evaporation occurs
        qᵛ = 0.008
        qˡ = 0.002
        qᵛ⁺ = 0.010
        Eₖ = BM.cloud_evaporation_rate(qᵛ, qˡ, qᵛ⁺, D)
        expected = min(qˡ, (qᵛ⁺ - qᵛ) / D)
        @test Eₖ ≈ expected atol=1e-10
        
        # Subsaturated with limited cloud water
        qˡ_small = 0.0001
        Eₖ = BM.cloud_evaporation_rate(qᵛ, qˡ_small, qᵛ⁺, D)
        @test Eₖ == qˡ_small  # Limited by available cloud water
    end

    @testset "Rain evaporation rate" begin
        # Saturated: no evaporation
        Eʳ = BM.rain_evaporation_rate(1.0, 0.010, 0.001, 0.010)
        @test Eʳ == 0.0
        
        # Supersaturated: no evaporation
        Eʳ = BM.rain_evaporation_rate(1.0, 0.012, 0.001, 0.010)
        @test Eʳ == 0.0
        
        # No rain: no evaporation
        Eʳ = BM.rain_evaporation_rate(1.0, 0.008, 0.0, 0.010)
        @test Eʳ == 0.0
        
        # Subsaturated with rain: evaporation occurs
        Eʳ = BM.rain_evaporation_rate(1.0, 0.008, 0.001, 0.010)
        @test Eʳ > 0.0
    end

    @testset "Terminal velocity" begin
        # Reference density at surface (would come from ρᵣ[1,1,1] in practice)
        ρ₀ = 1.0
        
        # No rain: zero terminal velocity
        wₜ = BM.rain_terminal_velocity(1.0, 0.0, ρ₀)
        @test wₜ == 0.0
        
        # With rain: positive terminal velocity
        wₜ = BM.rain_terminal_velocity(1.0, 0.001, ρ₀)
        @test wₜ > 0.0
        
        # Higher density decreases terminal velocity
        wₜ_low_ρ = BM.rain_terminal_velocity(0.5, 0.001, ρ₀)
        wₜ_high_ρ = BM.rain_terminal_velocity(1.5, 0.001, ρ₀)
        @test wₜ_low_ρ > wₜ_high_ρ
    end
end

@testset "Mass conservation" begin
    # The sum of vapor, cloud, and rain tendencies should be zero
    # (neglecting sedimentation which is handled separately)
    km = KesslerMicrophysics(Float64)
    
    # Test parameters
    ρ = 1.0
    qᵛ = 0.012
    qˡ = 0.001
    qʳ = 0.0005
    qᵛ⁺ = 0.010
    T = 288.0
    L = 2.5e6
    cₚ = 1005.0
    
    D = BM.condensation_denominator(T, qᵛ⁺, L, cₚ)
    
    # Compute all rates
    Cₖ = BM.condensation_rate(qᵛ, qᵛ⁺, D)
    Eₖ = BM.cloud_evaporation_rate(qᵛ, qˡ, qᵛ⁺, D)
    Aₖ = BM.autoconversion_rate(qˡ, km)
    Kₖ = BM.accretion_rate(qˡ, qʳ, km)
    Eʳ = BM.rain_evaporation_rate(ρ, qᵛ, qʳ, qᵛ⁺)
    
    # Tendencies (without density factor)
    dqᵛ_dt = -Cₖ + Eₖ + Eʳ
    dqˡ_dt = Cₖ - Eₖ - Aₖ - Kₖ
    dqʳ_dt = Aₖ + Kₖ - Eʳ
    
    # Total water tendency should be zero (mass conservation)
    total_tendency = dqᵛ_dt + dqˡ_dt + dqʳ_dt
    @test abs(total_tendency) < 1e-15
end
