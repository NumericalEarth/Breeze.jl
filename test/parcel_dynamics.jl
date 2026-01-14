#####
##### Unit tests for ParcelDynamics module
#####

using Breeze
using Breeze.ParcelDynamics:
    ParcelModel,
    ParcelState,
    EnvironmentalProfile,
    step_parcel!,
    adiabatic_adjustment,
    compute_moisture_fractions,
    environmental_velocity,
    environmental_pressure,
    environmental_density

using Breeze.Thermodynamics:
    StaticEnergyState,
    LiquidIcePotentialTemperatureState,
    MoistureMassFractions,
    temperature,
    mixture_heat_capacity

using Breeze.AtmosphereModels: TrivialMicrophysicalState

using CloudMicrophysics
using Test

# Get extension types
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
OneMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics
TwoMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.TwoMomentCloudMicrophysics
WarmPhaseOneMomentState = BreezeCloudMicrophysicsExt.WarmPhaseOneMomentState
WarmPhaseTwoMomentState = BreezeCloudMicrophysicsExt.WarmPhaseTwoMomentState

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
    ℳ = TrivialMicrophysicalState(FT)

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
##### ParcelModel tests
#####

@testset "ParcelModel construction" begin
    profile = EnvironmentalProfile(
        temperature = z -> 288.0,
        pressure = z -> 101325.0,
        density = z -> 1.2,
        specific_humidity = z -> 0.01
    )

    microphysics = OneMomentCloudMicrophysics()
    constants = ThermodynamicConstants()

    model = ParcelModel(profile, microphysics, constants)

    @test model.profile === profile
    @test model.microphysics === microphysics
    @test model.constants === constants
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

#####
##### Microphysical state moisture fractions
#####

@testset "compute_moisture_fractions from microphysical states" begin
    qᵗ = 0.020

    # Trivial state: all vapor
    ℳ_trivial = TrivialMicrophysicalState(Float64)
    q_trivial = compute_moisture_fractions(ℳ_trivial, qᵗ)
    @test q_trivial.vapor ≈ qᵗ
    @test q_trivial.liquid ≈ 0

    # One-moment: cloud + rain
    ℳ_1m = WarmPhaseOneMomentState(0.002, 0.001)  # qᶜˡ = 2 g/kg, qʳ = 1 g/kg
    q_1m = compute_moisture_fractions(ℳ_1m, qᵗ)
    @test q_1m.liquid ≈ 0.003  # qᶜˡ + qʳ
    @test q_1m.vapor ≈ qᵗ - 0.003

    # Two-moment: cloud + rain with number concentrations
    ℳ_2m = WarmPhaseTwoMomentState(0.002, 100e6, 0.001, 1e3)
    q_2m = compute_moisture_fractions(ℳ_2m, qᵗ)
    @test q_2m.liquid ≈ 0.003
    @test q_2m.vapor ≈ qᵗ - 0.003
end

#####
##### Full parcel stepping tests
#####

@testset "step_parcel! integration [$(FT)]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    g = constants.gravitational_acceleration

    # Environmental profile
    T_env(z) = FT(288.15) - FT(0.0065) * z
    p_env(z) = FT(101325.0) * (T_env(z) / FT(288.15))^(g / (FT(287.0) * FT(0.0065)))
    ρ_env(z) = p_env(z) / (FT(287.0) * T_env(z))

    profile = EnvironmentalProfile(
        temperature = T_env,
        pressure = p_env,
        density = ρ_env,
        specific_humidity = z -> FT(0.015) * exp(-z / FT(2500)),
        w = z -> FT(1.0)  # 1 m/s updraft
    )

    microphysics = OneMomentCloudMicrophysics()
    model = ParcelModel(profile, microphysics, constants)

    # Initialize parcel
    z₀ = FT(0.0)
    qᵗ = FT(0.015)
    q = MoistureMassFractions(qᵗ)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    e_init = cᵖᵐ * T_env(z₀) + g * z₀
    𝒰 = StaticEnergyState(e_init, q, z₀, p_env(z₀))
    ℳ = WarmPhaseOneMomentState(FT(0), FT(0))

    parcel = ParcelState(FT(0), FT(0), z₀, ρ_env(z₀), qᵗ, 𝒰, ℳ)

    @testset "Position update" begin
        Δt = FT(10.0)  # 10 second time step
        new_parcel = step_parcel!(parcel, model, Δt)

        # Parcel should have moved up by w * Δt = 10 m
        @test new_parcel.z ≈ FT(10.0)
        @test new_parcel.x ≈ FT(0.0)  # No horizontal motion
        @test new_parcel.y ≈ FT(0.0)
    end

    @testset "Conservation and microphysics" begin
        # Run for 100 steps
        current = parcel
        Δt = FT(1.0)
        for _ in 1:100
            current = step_parcel!(current, model, Δt)
        end

        # Parcel should have risen 100 m
        @test current.z ≈ FT(100.0) atol=FT(1e-6)

        # Total moisture should be conserved
        @test current.qᵗ ≈ qᵗ

        # Pressure should have decreased
        @test current.thermodynamic_state.reference_pressure < p_env(z₀)
    end
end
