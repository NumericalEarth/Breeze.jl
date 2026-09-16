include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: specific_prognostic_moisture_from_total
using Breeze.ParcelModels: ParcelDynamics
using Breeze.Microphysics: adjust_thermodynamic_state, DCMIP2016KesslerMicrophysics
using Breeze.Thermodynamics: MoistureMassFractions, LiquidIceDensityState, LiquidIcePotentialTemperatureState,
                             StaticEnergyState, TetensFormula, with_temperature, with_moisture, equilibrated_surface,
                             dry_air_gas_constant, vapor_gas_constant,
                             density, temperature, saturation_specific_humidity, total_specific_moisture,
                             potential_temperature_from_temperature, temperature_from_potential_temperature
using CloudMicrophysics
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, TwoMomentCloudMicrophysics

@testset "Moist temperature conversion [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    T = FT(283.15)
    pˢᵗ = FT(95000)
    ℒˡ = constants.liquid.reference_latent_heat
    ℒⁱ = constants.ice.reference_latent_heat
    moistures = (MoistureMassFractions(FT(0)),
                 MoistureMassFractions(FT(0.008)),
                 MoistureMassFractions(FT(0.008), FT(0.001)),
                 MoistureMassFractions(FT(0.008), FT(0.001), FT(0.0005)))

    for p in FT.((95000, 90000, 50000)), q in moistures
        θ = potential_temperature_from_temperature(T, p, pˢᵗ, constants, q)

        # Pin θˡⁱ against its definition θ = (T - (ℒˡqˡ + ℒⁱqⁱ)/cᵖᵐ) / Π rather than only
        # round-tripping, which shares Π and cᵖᵐ with the inverse and so cannot detect a
        # latent-heat term dropped from both directions.
        Rᵐ = mixture_gas_constant(q, constants)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        Π = (p / pˢᵗ)^(Rᵐ / cᵖᵐ)
        @test θ ≈ (T - (ℒˡ * q.liquid + ℒⁱ * q.ice) / cᵖᵐ) / Π

        @test temperature_from_potential_temperature(θ, p, pˢᵗ, constants, q) ≈ T
        ρ = density(T, p, q, constants)
        state = LiquidIceDensityState(θ, q, pˢᵗ, ρ)
        @test temperature(state, constants) ≈ T
    end

    q = MoistureMassFractions(FT(0.008))
    θ = potential_temperature_from_temperature(290, FT(90000), pˢᵗ, constants, q)
    @test θ ≈ potential_temperature_from_temperature(FT(290), FT(90000), pˢᵗ, constants, q.vapor)
    @test temperature_from_potential_temperature(θ, FT(90000), pˢᵗ, constants, q.vapor) ≈ FT(290)

    ρ = FT(1.1)
    equilibrium = WarmPhaseEquilibrium()
    microphysics = SaturationAdjustment(FT; equilibrium)
    qᵛ = saturation_specific_humidity(T, ρ, constants, equilibrium)
    q = MoistureMassFractions(qᵛ, FT(0.001))
    p = ρ * mixture_gas_constant(q, constants) * T
    θ = potential_temperature_from_temperature(T, p, pˢᵗ, constants, q)
    qᵉ = specific_prognostic_moisture_from_total(microphysics, total_specific_moisture(q), (;), ρ)
    state = LiquidIceDensityState(θ, MoistureMassFractions(qᵉ), pˢᵗ, ρ)
    adjusted = adjust_thermodynamic_state(state, microphysics, constants)
    @test temperature(adjusted, constants) ≈ T atol=microphysics.solver.abstol
    @test adjusted.moisture_mass_fractions.liquid ≈ q.liquid rtol=1e-3
end

@testset "Parcel initialization preserves θ and relative humidity [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=4, z=(0, 100), topology=(Flat, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(FT), microphysics=nothing)
    constants = model.thermodynamic_constants
    T, p, ℋ = FT.((300, 80000, 0.8))
    ρqᵛ = ℋ * saturation_specific_humidity(T, one(FT), constants, WarmPhaseEquilibrium())
    pᵛ = ρqᵛ * vapor_gas_constant(constants) * T
    ρ = (p - pᵛ) / (dry_air_gas_constant(constants) * T) + ρqᵛ
    qᵛ = ρqᵛ / ρ
    pˢᵗ = model.dynamics.standard_pressure
    θ = potential_temperature_from_temperature(T, p, pˢᵗ, constants, qᵛ)

    for previous_moisture in FT.((0, 0.04))
        set!(model; T, p, ρ, qᵗ=previous_moisture)
        set!(model; θ, p, ρ, ℋ, z=FT(50))
        initialized = model.dynamics.state.𝒰
        initialized_temperature = temperature(initialized, constants)
        initialized_vapor = initialized.moisture_mass_fractions.vapor
        initialized_humidity = initialized_vapor / saturation_specific_humidity(initialized_temperature, ρ, constants, WarmPhaseEquilibrium())
        @test initialized_humidity ≈ ℋ rtol=20eps(FT)
        @test initialized_vapor ≈ qᵛ rtol=100eps(FT)
        @test potential_temperature_from_temperature(initialized_temperature, p, pˢᵗ, constants, initialized_vapor) ≈ θ
    end
end

@testset "Saturation adjustment retains precipitation [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    ρ, pˢᵗ = FT(1.1), FT(1e5)
    grid = RectilinearGrid(default_arch, FT; size=(4, 4, 4), extent=(100, 100, 100))

    for equilibrium in (WarmPhaseEquilibrium(), MixedPhaseEquilibrium(FT))
        mixed = equilibrium isa MixedPhaseEquilibrium
        T = mixed ? FT(253.15) : FT(283.15)
        λ = mixed ? equilibrated_surface(equilibrium, T).liquid_fraction : one(FT)
        qʳ, qˢ = FT(0.002), mixed ? FT(0.001) : zero(FT)
        adjustment = SaturationAdjustment(FT; equilibrium)

        for cloud in (zero(FT), FT(0.001))
            qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, equilibrium)
            qᵛ = iszero(cloud) ? qᵛ⁺ / 2 : qᵛ⁺
            q = MoistureMassFractions(qᵛ, λ * cloud + qʳ, (1 - λ) * cloud + qˢ)
            p = ρ * mixture_gas_constant(q, constants) * T
            θ = potential_temperature_from_temperature(T, p, pˢᵗ, constants, q)
            energy_state = with_temperature(StaticEnergyState(zero(FT), q, FT(50), p), T, constants)
            states = (LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, p),
                      LiquidIceDensityState(θ, q, pˢᵗ, ρ), energy_state)
            for solver in (adjustment.solver, FixedIterations(6)), state in states
                solver_adjustment = SaturationAdjustment(FT; equilibrium, solver)
                adjusted = @inferred adjust_thermodynamic_state(state, solver_adjustment, constants, (qʳ, qˢ))
                @test temperature(adjusted, constants) ≈ T atol=FT(5e-4)
                @test adjusted.moisture_mass_fractions.vapor ≈ q.vapor rtol=FT(1e-3)
                @test adjusted.moisture_mass_fractions.liquid ≈ q.liquid rtol=FT(1e-3)
                @test adjusted.moisture_mass_fractions.ice ≈ q.ice atol=FT(1e-7)
                @test total_specific_moisture(adjusted) ≈ total_specific_moisture(state)
                # Adjustment changes only the partition, keeping the conserved variable
                # and the pressure/density constraint exactly as supplied.
                @test with_moisture(adjusted, state.moisture_mass_fractions) === state
            end

            microphysics = OneMomentCloudMicrophysics(FT; cloud_formation=adjustment)
            model = AtmosphereModel(grid; microphysics, thermodynamic_constants=constants, dynamics=CompressibleDynamics())
            precipitation = mixed ? (; qʳ, qˢ) : (; qʳ)
            set!(model; ρ, θ, qᵗ=total_specific_moisture(q), precipitation...)
            @test all(abs.(Array(interior(model.temperature)) .- T) .≤ FT(5e-4))
            @test all(abs.(Array(interior(model.microphysical_fields.qᶜˡ)) .- λ * cloud) .≤ FT(1e-6))
            diagnosed_water = model.microphysical_fields.qᵛ + model.microphysical_fields.qˡ
            if mixed
                diagnosed_water += model.microphysical_fields.qⁱ
            end
            @test all(Array(interior(Field(diagnosed_water))) .≈ total_specific_moisture(q))
        end
    end
end

@testset "Parcel condensate overshoots preserve water and energy [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=4, z=(0, 100), topology=(Flat, Flat, Bounded))
    for equilibrium in (WarmPhaseEquilibrium(), MixedPhaseEquilibrium(FT))
        mixed = equilibrium isa MixedPhaseEquilibrium
        adjustment = SaturationAdjustment(FT; equilibrium)
        microphysics = OneMomentCloudMicrophysics(FT; cloud_formation=adjustment)
        model = AtmosphereModel(grid; dynamics=ParcelDynamics(FT), microphysics)
        set!(model; T=FT(283.15), p=FT(1e5), ρ=one(FT), qᵗ=FT(0.01), z=FT(50))
        state = model.dynamics.state
        energy = state.ℰ
        state.μ = mixed ? (ρqʳ=FT(0.012), ρqˢ=FT(0.003)) : (ρqʳ=FT(0.012),)
        time_step!(model, zero(FT))
        q = state.𝒰.moisture_mass_fractions
        @test total_specific_moisture(q) ≈ FT(0.01)
        @test q.vapor ≥ -eps(FT)
        @test q.liquid ≥ 0
        @test q.ice ≥ 0
        @test state.ℰ == energy
        @test sum(values(state.μ)) / state.ρ ≈ FT(0.01)
        @test isfinite(temperature(state.𝒰, model.thermodynamic_constants))
    end
end

@testset "Moisture conversion counts independent species [$FT]" for FT in test_float_types()
    ρ = FT(0.8)
    qᵛ, qᶜˡ, qᶜⁱ, qʳ, qˢ = FT.((0.008, 0.001, 0.0005, 0.002, 0.0003))
    qᵗ = qᵛ + qᶜˡ + qᶜⁱ + qʳ + qˢ
    μ = (ρqᶜˡ = ρ * qᶜˡ, ρqᶜⁱ = ρ * qᶜⁱ, ρqʳ = ρ * qʳ, ρqˢ = ρ * qˢ,
         ρnᶜˡ = ρ * FT(1e8), ρnʳ = ρ * FT(1e5))

    for microphysics in (nothing, InstantaneousPrecipitation(FT), SaturationAdjustment(FT), BulkMicrophysics(FT))
        @test specific_prognostic_moisture_from_total(microphysics, qᵗ, (;), ρ) ≈ qᵗ
    end

    warm_total = qᵛ + qᶜˡ + qʳ
    for microphysics in (DCMIP2016KesslerMicrophysics(FT), OneMomentCloudMicrophysics(FT), TwoMomentCloudMicrophysics(FT))
        @test specific_prognostic_moisture_from_total(microphysics, warm_total, μ, ρ) ≈ qᵛ
    end

    warm_adjustment = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    mixed_adjustment = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    warm = OneMomentCloudMicrophysics(FT; cloud_formation=warm_adjustment)
    mixed = OneMomentCloudMicrophysics(FT; cloud_formation=mixed_adjustment)
    @test specific_prognostic_moisture_from_total(warm, warm_total, μ, ρ) ≈ qᵛ + qᶜˡ
    @test specific_prognostic_moisture_from_total(mixed, qᵗ, μ, ρ) ≈ qᵛ + qᶜˡ + qᶜⁱ

    # Conversion alone must not create water when precipitation exceeds the total.
    state = BreezeCloudMicrophysicsExt.WarmPhaseOneMomentState(FT(0.001), FT(0.012))
    qᵉ = specific_prognostic_moisture_from_total(warm, FT(0.01), state)
    @test qᵉ + state.qʳ ≈ FT(0.01)

    p3 = PredictedParticlePropertiesMicrophysics(FT)
    p3_densities = merge(μ, (ρqⁱ = ρ * qᶜⁱ, ρqʷⁱ = ρ * qˢ, ρqᶠ = ρ * qᶜⁱ / 2))
    @test specific_prognostic_moisture_from_total(p3, qᵗ, p3_densities, ρ) ≈ qᵛ
end

@testset "Total-water initialization with cloud and rain [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=(4, 4, 4), extent=(100, 100, 100))
    microphysics = DCMIP2016KesslerMicrophysics(FT)
    thermodynamic_constants = ThermodynamicConstants(FT; saturation_vapor_pressure=TetensFormula(FT))
    model = AtmosphereModel(grid; microphysics, thermodynamic_constants, dynamics=CompressibleDynamics())
    ρ, qᵛ, qᶜˡ, qʳ = FT.((0.8, 0.008, 0.001, 0.002))
    qᵗ = qᵛ + qᶜˡ + qʳ

    # Pull interiors to the CPU before reducing — `all(≈(x), ::CuArray)` would compile the
    # keyword-carrying closure into a GPU kernel and fail.
    cpu(field) = Array(interior(field))

    for density_input in ((; ρ), (; ρᵈ=ρ * (1 - qᵗ))),
        moisture_input in ((; qᵗ), (; ρqᵗ=ρ * qᵗ))
        set!(model; density_input..., moisture_input..., qᶜˡ, qʳ, θ=FT(300))
        @test all(cpu(model.moisture_density) .≈ ρ * qᵛ)
        @test all(cpu(model.dynamics.total_density) .≈ ρ)
        @test all(cpu(model.dynamics.dry_density) .≈ ρ * (1 - qᵗ))
        @test all(cpu(model.microphysical_fields.ρqᶜˡ) .≈ ρ * qᶜˡ)
        @test all(cpu(model.microphysical_fields.ρqʳ) .≈ ρ * qʳ)
    end

    # Condensates exceeding the total moisture would leave negative vapor; set! rejects them.
    @test_throws ArgumentError set!(model; ρ, qᵗ=FT(0.002), qᶜˡ, qʳ, θ=FT(300))
end

@testset "Total-water validation uses local moisture scales [$FT]" for FT in all_float_types()
    grid = RectilinearGrid(default_arch, FT; size=4, z=(0, 100), topology=(Flat, Flat, Bounded))
    microphysics = DCMIP2016KesslerMicrophysics(FT)
    thermodynamic_constants = ThermodynamicConstants(FT; saturation_vapor_pressure=TetensFormula(FT))
    model = AtmosphereModel(grid; microphysics, thermodynamic_constants, dynamics=CompressibleDynamics())
    ρ = FT(0.8)

    # Even a small condensate cannot be supplied with zero total water. In Float32,
    # an absolute tolerance of 10eps(FT) used to accept this and leave negative vapor.
    @test_throws ArgumentError set!(model; ρ, qᵗ=FT(0), qᶜˡ=FT(5e-7), qʳ=FT(0), θ=FT(300))

    # A humid cell must not set the tolerance for the dry cells in the same column.
    qᵗ(z) = ifelse(z < 50, FT(0), FT(0.02))
    qᶜˡ(z) = ifelse(z < 50, FT(1e-17), FT(0.001))
    @test_throws ArgumentError set!(model; ρ, qᵗ, qᶜˡ, qʳ=FT(0), θ=FT(300))

    # A relative deficit must be rejected at both small and large moisture scales.
    for water in FT.((1e-8, 1e-3))
        @test_throws ArgumentError set!(model; ρ, qᵗ=water, qᶜˡ=FT(1.001) * water, qʳ=FT(0), θ=FT(300))
    end

    # Cancellation at the precision of the inputs is allowed without changing the
    # supplied water budget, including a round-off-sized negative vapor residual.
    cloud = FT(0.001)
    water = prevfloat(cloud)
    set!(model; ρ, qᵗ=water, qᶜˡ=cloud, qʳ=FT(0), θ=FT(300))
    total_moisture_density = Array(interior(model.moisture_density)) .+
                            Array(interior(model.microphysical_fields.ρqᶜˡ))
    @test all(isapprox.(total_moisture_density, ρ * water; rtol=10eps(FT)))

    set!(model; ρ, qᵗ=FT(0), qᶜˡ=FT(0), qʳ=FT(0), θ=FT(300))
    @test all(iszero, Array(interior(model.moisture_density)))
end

@testset "Parcel initialization partitions carried condensate [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=4, z=(0, 100), topology=(Flat, Flat, Bounded))
    T, p, ρ, z = FT.((283.15, 1e5, 1.2, 50))
    qᶜˡ, qʳ = FT(0.001), FT(0.002)

    # Non-equilibrium: the carried cloud and rain enter the initial static energy, so the
    # parcel starts at the environmental temperature with that condensate.
    microphysics = OneMomentCloudMicrophysics(FT)
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(FT), microphysics)
    constants = model.thermodynamic_constants
    qᵗ = FT(0.01)
    set!(model; T, p, ρ, qᵗ, z)
    state = model.dynamics.state
    state.μ = merge(state.μ, (ρqᶜˡ=ρ * qᶜˡ, ρqʳ=ρ * qʳ))
    set!(model; T, p, ρ, qᵗ, z)
    q = state.𝒰.moisture_mass_fractions
    @test q.vapor ≈ qᵗ - qᶜˡ - qʳ
    @test q.liquid ≈ qᶜˡ + qʳ
    @test temperature(state.𝒰, constants) ≈ T rtol=10eps(FT)
    @test state.ℰ ≈ with_temperature(StaticEnergyState(zero(FT), q, z, p), T, constants).static_energy
    @test state.ρℰ ≈ ρ * state.ℰ

    # Saturation adjustment: rain enters the static energy at the environmental temperature, and
    # the supersaturated remainder is then equilibrated at fixed static energy, as at every substep.
    adjustment = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation=adjustment)
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(FT), microphysics)
    constants = model.thermodynamic_constants
    qᵗ = FT(0.012)
    set!(model; T, p, ρ, qᵗ, z)
    state = model.dynamics.state
    state.μ = (ρqʳ=ρ * qʳ,)
    set!(model; T, p, ρ, qᵗ, z)
    q = state.𝒰.moisture_mass_fractions
    T_parcel = temperature(state.𝒰, constants)
    qᵛ⁺ = saturation_specific_humidity(T_parcel, density(state.𝒰, constants), constants, WarmPhaseEquilibrium())
    @test total_specific_moisture(q) ≈ qᵗ
    @test q.liquid > qʳ
    @test q.vapor ≈ qᵛ⁺ rtol=FT(1e-3)
    @test T_parcel > T
    q₀ = MoistureMassFractions(qᵗ - qʳ, qʳ)
    @test state.ℰ ≈ with_temperature(StaticEnergyState(zero(FT), q₀, z, p), T, constants).static_energy
end
