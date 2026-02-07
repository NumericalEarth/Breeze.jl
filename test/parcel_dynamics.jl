#####
##### Unit tests for ParcelDynamics module
#####

using Oceananigans
using Oceananigans.Units: kilometers, minutes
using Breeze
using CloudMicrophysics
using Breeze.ParcelModels:
    ParcelDynamics,
    ParcelModel,
    ParcelState,
    adjust_adiabatically,
    compute_parcel_tendencies!

using Breeze.Thermodynamics:
    StaticEnergyState,
    LiquidIcePotentialTemperatureState,
    MoistureMassFractions,
    TetensFormula,
    temperature,
    mixture_heat_capacity

using Breeze.AtmosphereModels: NothingMicrophysicalState, microphysical_tendency
using Breeze.Microphysics: SaturationAdjustment, DCMIP2016KesslerMicrophysics

# Helper function to create thermodynamic constants compatible with DCMIP2016 Kessler
# The Kessler scheme uses TetensFormula for saturation vapor pressure
function kessler_thermodynamic_constants()
    tetens = TetensFormula(liquid_temperature_offset=36)
    return ThermodynamicConstants(; saturation_vapor_pressure=tetens)
end

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
    μ = NothingMicrophysicalState(FT)

    ρ = FT(1.2)
    ρqᵗ = ρ * qᵗ
    ρℰ = ρ * e_init
    parcel = ParcelState(FT(0), FT(0), z_init, ρ, qᵗ, ρqᵗ, e_init, ρℰ, 𝒰, μ)

    @test parcel.x == 0
    @test parcel.y == 0
    @test parcel.z == z_init
    @test parcel.ρ == ρ
    @test parcel.qᵗ == qᵗ
    @test parcel.ρqᵗ == ρqᵗ
    @test parcel.ℰ == e_init
    @test parcel.ρℰ == ρℰ
    @test parcel.𝒰 === 𝒰
    @test parcel.μ === μ
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
        𝒰_new = adjust_adiabatically(𝒰_init, z_new, p_new, constants)

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
        𝒰_new = adjust_adiabatically(𝒰_init, z_new, p_new, constants)

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
    tendencies = model.dynamics.timestepper.G
    @test tendencies.Gz ≈ 1.0  # w = 1 m/s
    # With specific quantity evolution, tendencies for e and qᵗ are zero
    # (no microphysical sources) giving exact conservation
    @test tendencies.Ge ≈ 0.0
    @test tendencies.Gqᵗ ≈ 0.0

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
    # Microphysical sources are zero (SaturationAdjustment operates via state adjustment)
    tendency_e = microphysical_tendency(microphysics, Val(:ρe), ρ_val, ℳ, 𝒰, constants)
    tendency_qt = microphysical_tendency(microphysics, Val(:ρqᵗ), ρ_val, ℳ, 𝒰, constants)
    @test tendency_e == 0.0
    @test tendency_qt == 0.0

    # Compute tendencies (this calls microphysical_tendency internally)
    compute_parcel_tendencies!(model)

    tendencies = model.dynamics.timestepper.G
    @test tendencies.Gz ≈ 1.0  # w = 1 m/s
    # Tendencies are zero (SaturationAdjustment operates via state adjustment, not tendencies)
    @test tendencies.Ge ≈ 0.0
    @test tendencies.Gqᵗ ≈ 0.0

    # Time step should work
    time_step!(model, 10.0)
    @test model.dynamics.state.z ≈ 10.0
end

@testset "ParcelModel with DCMIP2016KesslerMicrophysics" begin
    grid = RectilinearGrid(size=10, z=(0, 1000), topology=(Flat, Flat, Bounded))
    microphysics = DCMIP2016KesslerMicrophysics()
    constants = kessler_thermodynamic_constants()
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics, thermodynamic_constants=constants)

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
    # Microphysical sources are zero (operates via microphysics_model_update!)
    tendency_e = microphysical_tendency(microphysics, Val(:ρe), ρ_val, ℳ, 𝒰, constants)
    tendency_qt = microphysical_tendency(microphysics, Val(:ρqᵗ), ρ_val, ℳ, 𝒰, constants)
    @test tendency_e == 0.0
    @test tendency_qt == 0.0

    # Compute tendencies (this calls microphysical_tendency internally)
    compute_parcel_tendencies!(model)

    tendencies = model.dynamics.timestepper.G
    @test tendencies.Gz ≈ 1.0  # w = 1 m/s
    # Tendencies are zero (DCMIP2016Kessler operates via microphysics_model_update!, not tendencies)
    @test tendencies.Ge ≈ 0.0
    @test tendencies.Gqᵗ ≈ 0.0

    # Time step should work
    time_step!(model, 10.0)
    @test model.dynamics.state.z ≈ 10.0
end

#####
##### Adiabatic ascent in isentropic atmosphere
#####

using Oceananigans: interpolate

@testset "Adiabatic ascent: parcel temperature matches environment in isentropic atmosphere" begin
    # In an isentropic atmosphere (constant potential temperature θ), a parcel
    # ascending adiabatically should have the same temperature as the environment
    # at all heights. This tests that the parcel model correctly conserves
    # specific quantities (static energy, moisture) during ascent.

    grid = RectilinearGrid(size=100, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics=nothing)

    # Create an isentropic reference state (constant θ = 300 K)
    reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                     surface_pressure = 101325,
                                     potential_temperature = 300)

    # Set environmental profiles from the isentropic reference state
    # Use dry air (no moisture) to isolate the temperature conservation test
    set!(model;
         θ = reference_state.potential_temperature,
         p = reference_state.pressure,
         ρ = reference_state.density,
         qᵗ = 0,  # Dry air
         z = 0,
         w = 1)   # 1 m/s updraft

    # Record initial state
    constants = model.thermodynamic_constants
    T_initial = temperature(model.dynamics.state.𝒰, constants)
    z_initial = model.dynamics.state.z
    qᵗ_initial = model.dynamics.state.qᵗ
    e_initial = model.dynamics.state.ℰ

    # Run simulation for 20 minutes (parcel rises 1200 m at 1 m/s)
    simulation = Simulation(model; Δt=1.0, stop_time=20minutes)
    run!(simulation)

    z_final = model.dynamics.state.z
    qᵗ_final = model.dynamics.state.qᵗ
    e_final = model.dynamics.state.ℰ

    # Get parcel and environmental temperatures at final height
    T_parcel = temperature(model.dynamics.state.𝒰, constants)
    T_environment = interpolate(z_final, model.temperature)

    # In an isentropic atmosphere, parcel temperature should match environment
    # Allow 1 K tolerance for numerical errors
    @test abs(T_parcel - T_environment) < 1.0

    # Specific static energy should be EXACTLY conserved (specific quantity evolution)
    @test e_final == e_initial

    # Parcel should have risen to expected height
    @test z_final ≈ 1200.0 atol=1.0
end

@testset "Adiabatic ascent with moisture: specific humidity conserved" begin
    # Test that specific humidity qᵗ is conserved during adiabatic ascent
    # when there are no microphysical sources/sinks.

    grid = RectilinearGrid(size=100, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics=nothing)

    reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                     surface_pressure = 101325,
                                     potential_temperature = 300)

    # Environmental moisture profile (not used by parcel, but needed for initialization)
    qᵗ_env(z) = 0.012 * exp(-z / 2500)

    set!(model;
         θ = reference_state.potential_temperature,
         p = reference_state.pressure,
         ρ = reference_state.density,
         qᵗ = qᵗ_env,
         z = 0,
         w = 1)

    qᵗ_initial = model.dynamics.state.qᵗ
    e_initial = model.dynamics.state.ℰ

    # Run simulation for 15 minutes
    simulation = Simulation(model; Δt=1.0, stop_time=15minutes)
    run!(simulation)

    qᵗ_final = model.dynamics.state.qᵗ
    e_final = model.dynamics.state.ℰ

    # Specific quantities should be EXACTLY conserved (specific quantity evolution)
    # Static energy is exactly conserved
    @test e_final == e_initial

    # Moisture conserved to floating-point precision (minor rounding in RK3)
    @test isapprox(qᵗ_final, qᵗ_initial, rtol=1e-14)
end

#####
##### ParcelModel with OneMomentCloudMicrophysics
#####

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
OneMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics

@testset "ParcelModel with OneMomentCloudMicrophysics" begin
    grid = RectilinearGrid(size=100, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

    @test model isa ParcelModel
    @test model.dynamics isa ParcelDynamics

    reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                     surface_pressure = 101325,
                                     potential_temperature = 300)

    qᵗ₀ = 0.015
    Hq = 2500
    qᵗ(z) = qᵗ₀ * exp(-z / Hq)

    set!(model,
         θ = reference_state.potential_temperature,
         p = reference_state.pressure,
         ρ = reference_state.density,
         qᵗ = qᵗ,
         z = 0, w = 1)

    @test model.dynamics.state isa ParcelState
    @test model.dynamics.state.z ≈ 0.0
    @test model.dynamics.state.μ !== nothing

    # Check microphysical state is initialized correctly
    μ = model.dynamics.state.μ
    @test haskey(μ, :ρqᶜˡ)
    @test haskey(μ, :ρqʳ)

    # Time step should work
    simulation = Simulation(model; Δt=1.0, stop_time=5minutes)
    run!(simulation)

    @test model.dynamics.state.z ≈ 300.0 atol=1.0  # 5 min at 1 m/s = 300m
    @test model.clock.time ≈ 300.0
end

@testset "ParcelModel 1M microphysics forms cloud during ascent" begin
    grid = RectilinearGrid(size=100, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

    reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                     surface_pressure = 101325,
                                     potential_temperature = 300)

    # High initial moisture to ensure supersaturation during ascent
    qᵗ₀ = 0.015
    qᵗ(z) = qᵗ₀ * exp(-z / 2500)

    set!(model,
         θ = reference_state.potential_temperature,
         p = reference_state.pressure,
         ρ = reference_state.density,
         qᵗ = qᵗ,
         z = 0, w = 1)

    # Verify initial state has zero cloud/rain
    @test model.dynamics.state.μ.ρqᶜˡ ≈ 0.0
    @test model.dynamics.state.μ.ρqʳ ≈ 0.0

    # Run long enough for condensation to occur (above LCL)
    simulation = Simulation(model; Δt=1.0, stop_time=60minutes)
    run!(simulation)

    # After rising through LCL, cloud liquid should form
    qᶜˡ_final = model.dynamics.state.μ.ρqᶜˡ / model.dynamics.state.ρ
    @test qᶜˡ_final > 0  # Cloud should have formed

    # Check final height
    @test model.dynamics.state.z ≈ 3600.0 atol=10.0  # 60 min at 1 m/s
end

@testset "ParcelModel 1M microphysics produces rain via autoconversion" begin
    grid = RectilinearGrid(size=100, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

    reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                     surface_pressure = 101325,
                                     potential_temperature = 300)

    qᵗ₀ = 0.015
    qᵗ(z) = qᵗ₀ * exp(-z / 2500)

    set!(model,
         θ = reference_state.potential_temperature,
         p = reference_state.pressure,
         ρ = reference_state.density,
         qᵗ = qᵗ,
         z = 0, w = 1)

    # Run long enough for cloud formation and autoconversion
    simulation = Simulation(model; Δt=1.0, stop_time=120minutes)
    run!(simulation)

    # Extract final microphysical state
    ρ_final = model.dynamics.state.ρ
    qᶜˡ_final = model.dynamics.state.μ.ρqᶜˡ / ρ_final
    qʳ_final = model.dynamics.state.μ.ρqʳ / ρ_final

    # Both cloud and rain should be present after 2 hours of ascent
    @test qᶜˡ_final > 0  # Cloud liquid present
    @test qʳ_final > 0   # Rain produced via autoconversion

    # Total water should be conserved (vapor + cloud + rain = initial qᵗ at parcel altitude)
    q = model.dynamics.state.𝒰.moisture_mass_fractions
    qᵗ_total = q.vapor + qᶜˡ_final + qʳ_final
    @test qᵗ_total ≈ model.dynamics.state.qᵗ rtol=1e-10
end

#####
##### ParcelModel with TwoMomentCloudMicrophysics
#####

TwoMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.TwoMomentCloudMicrophysics

@testset "ParcelModel with TwoMomentCloudMicrophysics" begin
    grid = RectilinearGrid(size=100, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

    @test model isa ParcelModel
    @test model.dynamics isa ParcelDynamics

    reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                     surface_pressure = 101325,
                                     potential_temperature = 300)

    qᵗ₀ = 0.015
    Hq = 2500
    qᵗ(z) = qᵗ₀ * exp(-z / Hq)

    set!(model,
         θ = reference_state.potential_temperature,
         p = reference_state.pressure,
         ρ = reference_state.density,
         qᵗ = qᵗ,
         z = 0, w = 1)

    @test model.dynamics.state isa ParcelState
    @test model.dynamics.state.z ≈ 0.0
    @test model.dynamics.state.μ !== nothing

    # Check microphysical state has all four 2M fields
    μ = model.dynamics.state.μ
    @test haskey(μ, :ρqᶜˡ)
    @test haskey(μ, :ρnᶜˡ)
    @test haskey(μ, :ρqʳ)
    @test haskey(μ, :ρnʳ)

    # Time step should work
    simulation = Simulation(model; Δt=1.0, stop_time=5minutes)
    run!(simulation)

    @test model.dynamics.state.z ≈ 300.0 atol=1.0  # 5 min at 1 m/s = 300m
    @test model.clock.time ≈ 300.0
end

@testset "ParcelModel 2M microphysics forms cloud during ascent" begin
    grid = RectilinearGrid(size=100, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

    reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                     surface_pressure = 101325,
                                     potential_temperature = 300)

    # High initial moisture to ensure supersaturation during ascent
    qᵗ₀ = 0.015
    qᵗ(z) = qᵗ₀ * exp(-z / 2500)

    set!(model,
         θ = reference_state.potential_temperature,
         p = reference_state.pressure,
         ρ = reference_state.density,
         qᵗ = qᵗ,
         z = 0, w = 1)

    # Initialize with some droplet number (CCN activation)
    nᶜˡ₀ = 100e6  # 100 million droplets per kg
    model.dynamics.state.μ = (; ρqᶜˡ=0.0, ρnᶜˡ=1.2 * nᶜˡ₀, ρqʳ=0.0, ρnʳ=0.0)

    # Run long enough for condensation to occur (above LCL)
    simulation = Simulation(model; Δt=1.0, stop_time=60minutes)
    run!(simulation)

    # After rising through LCL, cloud liquid should form
    qᶜˡ_final = model.dynamics.state.μ.ρqᶜˡ / model.dynamics.state.ρ
    nᶜˡ_final = model.dynamics.state.μ.ρnᶜˡ / model.dynamics.state.ρ
    @test qᶜˡ_final > 0  # Cloud should have formed
    @test nᶜˡ_final > 0  # Droplet number should be present

    # Check final height
    @test model.dynamics.state.z ≈ 3600.0 atol=10.0  # 60 min at 1 m/s
end

@testset "ParcelModel 2M microphysics produces rain via autoconversion" begin
    grid = RectilinearGrid(size=100, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

    reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                     surface_pressure = 101325,
                                     potential_temperature = 300)

    qᵗ₀ = 0.015
    qᵗ(z) = qᵗ₀ * exp(-z / 2500)

    set!(model,
         θ = reference_state.potential_temperature,
         p = reference_state.pressure,
         ρ = reference_state.density,
         qᵗ = qᵗ,
         z = 0, w = 1)

    # Initialize with droplet number for 2M scheme
    nᶜˡ₀ = 100e6
    model.dynamics.state.μ = (; ρqᶜˡ=0.0, ρnᶜˡ=1.2 * nᶜˡ₀, ρqʳ=0.0, ρnʳ=0.0)

    # Run long enough for cloud formation and autoconversion
    simulation = Simulation(model; Δt=1.0, stop_time=120minutes)
    run!(simulation)

    # Extract final microphysical state
    ρ_final = model.dynamics.state.ρ
    qᶜˡ_final = model.dynamics.state.μ.ρqᶜˡ / ρ_final
    qʳ_final = model.dynamics.state.μ.ρqʳ / ρ_final
    nᶜˡ_final = model.dynamics.state.μ.ρnᶜˡ / ρ_final
    nʳ_final = model.dynamics.state.μ.ρnʳ / ρ_final

    # Both cloud and rain should be present after 2 hours of ascent
    @test qᶜˡ_final > 0  # Cloud liquid present
    @test qʳ_final > 0   # Rain produced via autoconversion
    @test nᶜˡ_final > 0  # Cloud droplet number present
    @test nʳ_final > 0   # Rain drop number produced

    # Total water should be conserved
    q = model.dynamics.state.𝒰.moisture_mass_fractions
    qᵗ_total = q.vapor + qᶜˡ_final + qʳ_final
    @test qᵗ_total ≈ model.dynamics.state.qᵗ rtol=1e-10
end
