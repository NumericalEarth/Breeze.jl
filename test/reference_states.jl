#####
##### Tests for ReferenceState, compute_reference_state!, and related functions
#####

using Breeze
using Breeze.Thermodynamics:
    compute_reference_state!,
    compute_hydrostatic_reference!,
    dry_air_gas_constant,
    vapor_gas_constant,
    saturation_specific_humidity,
    PlanarLiquidSurface

using Oceananigans
using Oceananigans.Fields: ZeroField
using GPUArraysCore: @allowscalar
using Test

@testset "ReferenceState [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), x=(0, 100), y=(0, 100), z=(0, 10000),
                           topology=(Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(FT)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    g = constants.gravitational_acceleration

    #####
    ##### Constructor: default moisture fields are ZeroField
    #####

    @testset "Default constructor produces ZeroField moisture" begin
        ref = ReferenceState(grid, constants)
        @test ref.vapor_mass_fraction isa ZeroField
        @test ref.liquid_mass_fraction isa ZeroField
        @test ref.ice_mass_fraction isa ZeroField
        @test ref.pressure isa Field
        @test ref.density isa Field
        @test ref.temperature isa Field
    end

    @testset "Constructor with moisture = 0 allocates Field" begin
        ref = ReferenceState(grid, constants; vapor_mass_fraction=0)
        @test ref.vapor_mass_fraction isa Field
        @test ref.liquid_mass_fraction isa ZeroField
        @test ref.ice_mass_fraction isa ZeroField
    end

    @testset "Constructor with moisture function allocates Field" begin
        qᵛ(z) = 0.01 * exp(-z / 2500)
        ref = ReferenceState(grid, constants; vapor_mass_fraction=qᵛ)
        @test ref.vapor_mass_fraction isa Field
        # Check the profile was set
        qᵛ₁ = @allowscalar ref.vapor_mass_fraction[1, 1, 1]
        @test qᵛ₁ > 0
    end

    #####
    ##### surface_density(ref::ReferenceState)
    #####

    @testset "surface_density" begin
        ref = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
        ρ₀ = surface_density(ref)
        # Surface density should be close to p₀ / (Rᵈ T₀) where T₀ ≈ θ₀ Π₀
        # For θ₀=300 and p₀=101325, Π₀ ≈ 1, so T₀ ≈ 300 K
        ρ_expected = FT(101325) / (Rᵈ * FT(300))  # approximate
        @test ρ₀ isa FT
        @test isapprox(ρ₀, ρ_expected; rtol=0.01)
    end

    #####
    ##### compute_reference_state! — dry isothermal atmosphere
    #####
    #
    # For constant T and zero moisture, the hydrostatic equation gives:
    #   p(z) = p₀ exp(-g z / (Rᵈ T))
    #   ρ(z) = p(z) / (Rᵈ T)

    @testset "compute_reference_state! dry isothermal" begin
        T₀ = FT(250)
        p₀ = FT(101325)
        ref = ReferenceState(grid, constants; surface_pressure=p₀, vapor_mass_fraction=0)

        compute_reference_state!(ref, T₀, FT(0), constants)

        Nz = grid.Nz
        for k in 1:Nz
            z = @allowscalar Oceananigans.Grids.znode(1, 1, k, grid, Center(), Center(), Center())
            p_exact = p₀ * exp(-g * z / (Rᵈ * T₀))
            ρ_exact = p_exact / (Rᵈ * T₀)

            p_ref = @allowscalar ref.pressure[1, 1, k]
            ρ_ref = @allowscalar ref.density[1, 1, k]
            T_ref = @allowscalar ref.temperature[1, 1, k]

            @test T_ref ≈ T₀
            @test isapprox(p_ref, p_exact; rtol=FT(1e-4))
            @test isapprox(ρ_ref, ρ_exact; rtol=FT(1e-4))
        end
    end

    #####
    ##### compute_reference_state! — moist isothermal atmosphere
    #####
    #
    # For constant T and constant qᵛ (no condensate):
    #   Rᵐ = (1 - qᵛ) Rᵈ + qᵛ Rᵛ
    #   p(z) = p₀ exp(-g z / (Rᵐ T))
    #   ρ(z) = p(z) / (Rᵐ T)

    @testset "compute_reference_state! moist isothermal" begin
        T₀ = FT(280)
        qᵛ = FT(0.015)
        p₀ = FT(101325)
        Rᵐ = (1 - qᵛ) * Rᵈ + qᵛ * Rᵛ

        ref = ReferenceState(grid, constants; surface_pressure=p₀, vapor_mass_fraction=0)
        compute_reference_state!(ref, T₀, qᵛ, constants)

        Nz = grid.Nz
        for k in 1:Nz
            z = @allowscalar Oceananigans.Grids.znode(1, 1, k, grid, Center(), Center(), Center())
            p_exact = p₀ * exp(-g * z / (Rᵐ * T₀))
            ρ_exact = p_exact / (Rᵐ * T₀)

            p_ref = @allowscalar ref.pressure[1, 1, k]
            ρ_ref = @allowscalar ref.density[1, 1, k]

            @test isapprox(p_ref, p_exact; rtol=FT(1e-4))
            @test isapprox(ρ_ref, ρ_exact; rtol=FT(1e-4))
        end

        # Verify moisture was set
        qᵛ_ref = @allowscalar ref.vapor_mass_fraction[1, 1, 1]
        @test qᵛ_ref ≈ qᵛ
    end

    #####
    ##### compute_reference_state! — 5-argument form with individual mass fractions
    #####

    @testset "compute_reference_state! with individual mass fractions" begin
        T₀ = FT(260)
        qᵛ = FT(0.01)
        qˡ = FT(1e-4)
        qⁱ = FT(5e-5)
        p₀ = FT(101325)

        ref = ReferenceState(grid, constants;
                             surface_pressure=p₀,
                             vapor_mass_fraction=0,
                             liquid_mass_fraction=0,
                             ice_mass_fraction=0)

        compute_reference_state!(ref, T₀, qᵛ, qˡ, qⁱ, constants)

        # Verify moisture fields were set
        @test @allowscalar(ref.vapor_mass_fraction[1, 1, 1]) ≈ qᵛ
        @test @allowscalar(ref.liquid_mass_fraction[1, 1, 1]) ≈ qˡ
        @test @allowscalar(ref.ice_mass_fraction[1, 1, 1]) ≈ qⁱ

        # Verify pressure is physically reasonable
        p_top = @allowscalar ref.pressure[1, 1, grid.Nz]
        @test p_top < p₀  # pressure decreases with height
        @test p_top > 0    # still positive

        # Ideal gas consistency: ρ = p / (Rᵐ T)
        Rᵐ = (1 - qᵛ - qˡ - qⁱ) * Rᵈ + qᵛ * Rᵛ
        for k in 1:grid.Nz
            p_ref = @allowscalar ref.pressure[1, 1, k]
            ρ_ref = @allowscalar ref.density[1, 1, k]
            @test isapprox(ρ_ref, p_ref / (Rᵐ * T₀); rtol=FT(1e-5))
        end
    end

    #####
    ##### compute_reference_state! with function profiles
    #####

    @testset "compute_reference_state! with function profiles" begin
        p₀ = FT(101325)

        T_profile(z) = max(FT(210), FT(300) - FT(0.0065) * z)
        q_profile(z) = FT(0.015) * exp(-z / 3000)

        ref = ReferenceState(grid, constants; surface_pressure=p₀, vapor_mass_fraction=0)
        compute_reference_state!(ref, T_profile, q_profile, constants)

        # Temperature should follow the profile
        z₁ = @allowscalar Oceananigans.Grids.znode(1, 1, 1, grid, Center(), Center(), Center())
        T₁ = @allowscalar ref.temperature[1, 1, 1]
        @test isapprox(T₁, T_profile(z₁); rtol=FT(1e-5))

        # Moisture should follow the profile
        qᵛ₁ = @allowscalar ref.vapor_mass_fraction[1, 1, 1]
        @test isapprox(qᵛ₁, q_profile(z₁); rtol=FT(1e-5))

        # Pressure should decrease monotonically
        for k in 2:grid.Nz
            pᵏ = @allowscalar ref.pressure[1, 1, k]
            pᵏ⁻¹ = @allowscalar ref.pressure[1, 1, k-1]
            @test pᵏ < pᵏ⁻¹
        end

        # Density should decrease monotonically
        for k in 2:grid.Nz
            ρᵏ = @allowscalar ref.density[1, 1, k]
            ρᵏ⁻¹ = @allowscalar ref.density[1, 1, k-1]
            @test ρᵏ < ρᵏ⁻¹
        end
    end

    #####
    ##### compute_reference_state! overwrites previous state
    #####

    @testset "compute_reference_state! overwrites previous state" begin
        p₀ = FT(101325)
        ref = ReferenceState(grid, constants; surface_pressure=p₀, vapor_mass_fraction=0)

        # First adjustment: warm atmosphere
        compute_reference_state!(ref, FT(300), FT(0), constants)
        ρ_warm = @allowscalar ref.density[1, 1, 1]

        # Second adjustment: cold atmosphere → higher density
        compute_reference_state!(ref, FT(200), FT(0), constants)
        ρ_cold = @allowscalar ref.density[1, 1, 1]

        @test ρ_cold > ρ_warm
    end
end
