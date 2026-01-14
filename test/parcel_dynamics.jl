#####
##### Unit tests for ParcelDynamics module
#####

using Breeze
using Breeze.ParcelDynamics:
    ParcelDynamics,
    ParcelModel,
    ParcelState,
    EnvironmentalProfile,
    adiabatic_adjustment,
    environmental_velocity,
    environmental_pressure,
    environmental_density

using Breeze.Thermodynamics:
    StaticEnergyState,
    LiquidIcePotentialTemperatureState,
    MoistureMassFractions,
    temperature,
    mixture_heat_capacity

using Breeze.AtmosphereModels: NothingMicrophysicalState

using Oceananigans.TimeSteppers: time_step!

using Test

#####
##### EnvironmentalProfile tests
#####

@testset "EnvironmentalProfile construction" begin
    # Minimal profile with constants
    profile = EnvironmentalProfile(
        temperature = z -> 300.0,
        pressure = z -> 1e5,
        density = z -> 1.2,
        specific_humidity = z -> 0.01
    )

    @test environmental_velocity(profile, 0.0) == (0.0, 0.0, 0.0)
    @test environmental_pressure(profile, 500.0) == 1e5
    @test environmental_density(profile, 1000.0) == 1.2

    # Profile with 3D velocities
    profile_3d = EnvironmentalProfile(
        temperature = z -> 288.0 - 0.0065 * z,
        pressure = z -> 101325.0 * exp(-z / 8500),
        density = z -> 1.225 * exp(-z / 8500),
        specific_humidity = z -> 0.015 * exp(-z / 2500),
        u = z -> 5.0,
        v = z -> 2.0,
        w = z -> 1.0 + 0.001 * z
    )

    u, v, w = environmental_velocity(profile_3d, 1000.0)
    @test u == 5.0
    @test v == 2.0
    @test w ≈ 2.0  # 1.0 + 0.001 * 1000
end

#####
##### ParcelState tests
#####

@testset "ParcelState construction [$(FT)]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    g = constants.gravitational_acceleration

    # Create a StaticEnergyState
    T_init = FT(288.0)
    z_init = FT(0.0)
    p_init = FT(101325.0)
    qᵗ = FT(0.015)
    q = MoistureMassFractions(qᵗ)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    e_init = cᵖᵐ * T_init + g * z_init

    𝒰 = StaticEnergyState(e_init, q, z_init, p_init)
    ℳ = NothingMicrophysicalState(FT)

    parcel = ParcelState(FT(0), FT(0), z_init, FT(1.2), qᵗ, 𝒰, ℳ)

    @test parcel.x == 0
    @test parcel.y == 0
    @test parcel.z == z_init
    @test parcel.ρ == FT(1.2)
    @test parcel.qᵗ == qᵗ
    @test parcel.thermodynamic_state === 𝒰
    @test parcel.microphysical_state === ℳ
end

#####
##### ParcelDynamics tests
#####

@testset "ParcelDynamics construction" begin
    profile = EnvironmentalProfile(
        temperature = z -> 288.0,
        pressure = z -> 101325.0,
        density = z -> 1.2,
        specific_humidity = z -> 0.01
    )

    dynamics = ParcelDynamics(profile)

    @test dynamics.profile === profile
    @test dynamics.state === nothing
end

#####
##### Adiabatic adjustment tests
#####

@testset "Adiabatic adjustment [$(FT)]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    g = constants.gravitational_acceleration

    @testset "StaticEnergyState conserves energy" begin
        T_init = FT(288.0)
        z_init = FT(0.0)
        p_init = FT(101325.0)
        qᵗ = FT(0.010)
        q = MoistureMassFractions(qᵗ)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        e_init = cᵖᵐ * T_init + g * z_init

        𝒰_init = StaticEnergyState(e_init, q, z_init, p_init)

        # Adjust to new height
        z_new = FT(1000.0)
        p_new = FT(90000.0)
        𝒰_new = adiabatic_adjustment(𝒰_init, z_new, p_new, constants)

        # Static energy should be conserved
        @test 𝒰_new.static_energy ≈ e_init
        @test 𝒰_new.height == z_new
        @test 𝒰_new.reference_pressure == p_new

        # Temperature should decrease (adiabatic cooling)
        T_new = temperature(𝒰_new, constants)
        @test T_new < T_init
    end

    @testset "LiquidIcePotentialTemperatureState conserves θˡⁱ" begin
        θ_init = FT(300.0)
        p_init = FT(101325.0)
        pˢᵗ = FT(1e5)
        qᵗ = FT(0.010)
        q = MoistureMassFractions(qᵗ)

        𝒰_init = LiquidIcePotentialTemperatureState(θ_init, q, pˢᵗ, p_init)

        # Adjust to new pressure
        z_new = FT(1000.0)
        p_new = FT(90000.0)
        𝒰_new = adiabatic_adjustment(𝒰_init, z_new, p_new, constants)

        # Potential temperature should be conserved
        @test 𝒰_new.potential_temperature ≈ θ_init
        @test 𝒰_new.reference_pressure == p_new
        @test 𝒰_new.standard_pressure == pˢᵗ
    end
end
