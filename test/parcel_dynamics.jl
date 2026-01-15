#####
##### Unit tests for ParcelDynamics module
#####

using Oceananigans
using Breeze
using Breeze.ParcelDynamics:
    ParcelDynamics,
    ParcelModel,
    ParcelState,
    adiabatic_adjustment,
    compute_parcel_tendencies!

using Breeze.Thermodynamics:
    StaticEnergyState,
    LiquidIcePotentialTemperatureState,
    MoistureMassFractions,
    temperature,
    mixture_heat_capacity

using Breeze.AtmosphereModels: NothingMicrophysicalState, microphysical_tendency
using Breeze.Microphysics: SaturationAdjustment, DCMIP2016KesslerMicrophysics

using Test

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
    @test parcel.microphysics_prognostics === ℳ
end

#####
##### ParcelDynamics construction tests
#####

@testset "ParcelDynamics construction" begin
    dynamics = ParcelDynamics()

    @test dynamics.state === nothing
    @test dynamics.density === nothing
    @test dynamics.pressure === nothing
    @test dynamics.surface_pressure == 101325.0
    @test dynamics.standard_pressure == 1e5
end

#####
##### AtmosphereModel with ParcelDynamics tests
#####

@testset "AtmosphereModel(grid; dynamics=ParcelDynamics()) and set!" begin
    grid = RectilinearGrid(size=10, z=(0, 1000), topology=(Flat, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics=ParcelDynamics())

    @test model isa ParcelModel
    @test model.dynamics isa ParcelDynamics
    # After materialization, state is a ParcelState (mutable, so fields can be updated)
    @test model.dynamics.state isa ParcelState

    # Define environmental profiles
    T(z) = 288.0 - 0.0065 * z
    p(z) = 101325.0 * exp(-z / 8500)
    ρ(z) = p(z) / (287.0 * T(z))

    # Set profiles and initial position
    set!(model, T=T, p=p, ρ=ρ, z=0.0, w=1.0)

    @test model.dynamics.density !== nothing
    @test model.dynamics.pressure !== nothing
    @test model.dynamics.state isa ParcelState
    @test model.dynamics.state.z ≈ 0.0
end

@testset "time_step! for ParcelModel" begin
    grid = RectilinearGrid(size=10, z=(0, 1000), topology=(Flat, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics=ParcelDynamics())

    T(z) = 288.0 - 0.0065 * z
    p(z) = 101325.0 * exp(-z / 8500)
    ρ(z) = p(z) / (287.0 * T(z))

    set!(model, T=T, p=p, ρ=ρ, z=0.0, w=1.0)

    @test model.clock.time == 0.0
    @test model.clock.iteration == 0

    # Step forward
    Δt = 10.0
    time_step!(model, Δt)

    @test model.dynamics.state.z ≈ 10.0  # w=1 m/s × 10s = 10m
    @test model.clock.time ≈ Δt
    @test model.clock.iteration == 1

    # Run more steps
    for _ in 1:9
        time_step!(model, Δt)
    end

    @test model.dynamics.state.z ≈ 100.0
    @test model.clock.time ≈ 100.0
    @test model.clock.iteration == 10
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
##### ParcelModel with microphysics schemes
#####

@testset "ParcelModel with Nothing microphysics" begin
    grid = RectilinearGrid(size=10, z=(0, 1000), topology=(Flat, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics=nothing)

    T(z) = 288.0 - 0.0065 * z
    p(z) = 101325.0 * exp(-z / 8500)
    ρ(z) = p(z) / (287.0 * T(z))

    set!(model, T=T, p=p, ρ=ρ, z=0.0, w=1.0)

    # Compute tendencies (this calls microphysical_tendency)
    compute_parcel_tendencies!(model)

    # Check tendencies are computed
    tendencies = model.dynamics.tendencies
    @test tendencies.Gz ≈ 1.0  # w = 1 m/s
    @test tendencies.Ge ≈ 0.0  # No microphysics
    @test tendencies.Gqᵗ ≈ 0.0  # No microphysics

    # Time step should work
    time_step!(model, 10.0)
    @test model.dynamics.state.z ≈ 10.0
end

@testset "ParcelModel with SaturationAdjustment microphysics" begin
    grid = RectilinearGrid(size=10, z=(0, 1000), topology=(Flat, Flat, Bounded))
    microphysics = SaturationAdjustment()
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

    T(z) = 288.0 - 0.0065 * z
    p(z) = 101325.0 * exp(-z / 8500)
    ρ(z) = p(z) / (287.0 * T(z))

    set!(model, T=T, p=p, ρ=ρ, z=0.0, w=1.0)

    # Verify state-based microphysical_tendency is callable
    constants = model.thermodynamic_constants
    state = model.dynamics.state
    ρ_val = state.ρ
    𝒰 = state.𝒰
    ℳ = NothingMicrophysicalState(typeof(ρ_val))

    # This tests that the state-based interface exists for SaturationAdjustment
    tendency_e = microphysical_tendency(microphysics, Val(:ρe), ρ_val, ℳ, 𝒰, constants)
    tendency_qt = microphysical_tendency(microphysics, Val(:ρqᵗ), ρ_val, ℳ, 𝒰, constants)
    @test tendency_e == 0.0
    @test tendency_qt == 0.0

    # Compute tendencies (this calls microphysical_tendency internally)
    compute_parcel_tendencies!(model)

    tendencies = model.dynamics.tendencies
    @test tendencies.Gz ≈ 1.0  # w = 1 m/s
    @test tendencies.Ge ≈ 0.0  # SaturationAdjustment operates via state adjustment
    @test tendencies.Gqᵗ ≈ 0.0

    # Time step should work
    time_step!(model, 10.0)
    @test model.dynamics.state.z ≈ 10.0
end

@testset "ParcelModel with DCMIP2016KesslerMicrophysics" begin
    grid = RectilinearGrid(size=10, z=(0, 1000), topology=(Flat, Flat, Bounded))
    microphysics = DCMIP2016KesslerMicrophysics()
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

    T(z) = 288.0 - 0.0065 * z
    p(z) = 101325.0 * exp(-z / 8500)
    ρ(z) = p(z) / (287.0 * T(z))

    set!(model, T=T, p=p, ρ=ρ, z=0.0, w=1.0)

    # Verify state-based microphysical_tendency is callable
    constants = model.thermodynamic_constants
    state = model.dynamics.state
    ρ_val = state.ρ
    𝒰 = state.𝒰
    ℳ = NothingMicrophysicalState(typeof(ρ_val))

    # This tests that the state-based interface exists for DCMIP2016Kessler
    tendency_e = microphysical_tendency(microphysics, Val(:ρe), ρ_val, ℳ, 𝒰, constants)
    tendency_qt = microphysical_tendency(microphysics, Val(:ρqᵗ), ρ_val, ℳ, 𝒰, constants)
    @test tendency_e == 0.0
    @test tendency_qt == 0.0

    # Compute tendencies (this calls microphysical_tendency internally)
    compute_parcel_tendencies!(model)

    tendencies = model.dynamics.tendencies
    @test tendencies.Gz ≈ 1.0  # w = 1 m/s
    @test tendencies.Ge ≈ 0.0  # DCMIP2016Kessler operates via microphysics_model_update!
    @test tendencies.Gqᵗ ≈ 0.0

    # Time step should work
    time_step!(model, 10.0)
    @test model.dynamics.state.z ≈ 10.0
end
