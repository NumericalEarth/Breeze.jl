include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.Thermodynamics: Thermodynamics as TD
using Breeze.PotentialTemperatureFormulations: PotentialTemperatureFormulations as PTF, Dry, LiquidIce,
                                                IsochoricPressureWork, IsobaricPressureWork
using Breeze.CompressibleEquations: MomentumForceWeighting
using Breeze.Microphysics: saturation_adjustment_step
using Oceananigans
using Test

const constants = ThermodynamicConstants(Float64)
const tetens_constants = ThermodynamicConstants(Float64; saturation_vapor_pressure=TetensFormula(Float64))

const test_compositions = (TD.MoistureMassFractions(0.0, 0.0, 0.0),
                           TD.MoistureMassFractions(0.02, 0.0, 0.0),
                           TD.MoistureMassFractions(0.012, 0.002, 0.0),
                           TD.MoistureMassFractions(0.02, 0.006, 0.0),
                           TD.MoistureMassFractions(0.002, 0.0005, 0.001))

dry_kappa(c) = TD.dry_air_gas_constant(c) / c.dry_air.heat_capacity
gas_potential_temperature(T, ρ, q, c; pˢᵗ=1e5) = T * (pˢᵗ / (ρ * TD.mixture_gas_constant(q, c) * T))^dry_kappa(c)

@testset "DryDensityState" begin
    T, ρ, pˢᵗ = 285.0, 1.0, 1e5
    for q in test_compositions
        θ = gas_potential_temperature(T, ρ, q, constants)
        𝒰 = TD.DryDensityState(θ, q, pˢᵗ, ρ)
        @test TD.temperature(𝒰, constants) ≈ T rtol=1e-13
        @test TD.air_pressure(𝒰, constants) ≈ ρ * TD.mixture_gas_constant(q, constants) * T rtol=1e-13
        @test TD.exner_function(𝒰, constants) ≈ T / θ rtol=1e-13
        @test TD.with_temperature(TD.DryDensityState(0.0, q, pˢᵗ, ρ), T, constants).potential_temperature ≈ θ rtol=1e-13
    end
end

@testset "Expansion and heating coefficients against the first law" begin
    T, ρ = 285.0, 1.0
    ϵ = 1e-7
    for q in test_compositions
        Rᵐ = TD.mixture_gas_constant(q, constants)
        cᵖᵐ = TD.mixture_heat_capacity(q, constants)
        cᵛᵐ = cᵖᵐ - Rᵐ
        κᵈ = dry_kappa(constants)
        θ = gas_potential_temperature(T, ρ, q, constants)

        # Adiabatic expansion at fixed composition: cᵛᵐ dT = − Rᵐ T ∇⋅𝐮 dt, d ln ρ = − ∇⋅𝐮 dt
        Tₑ = T - Rᵐ * T * ϵ / cᵛᵐ
        ρₑ = ρ * (1 - ϵ)
        dθ = (gas_potential_temperature(Tₑ, ρₑ, q, constants) - θ) / ϵ
        S = PTF.divergence_expansion_coefficient(θ, q, constants)
        @test isapprox(dθ, -S; atol=1e-5 * θ)

        # CM1's acoustic Exner term: Dπ/Dt = − κᵈ (cᵖᵐ / cᵛᵐ) π ∇⋅𝐮
        Π = T / θ
        dΠ = (Tₑ / gas_potential_temperature(Tₑ, ρₑ, q, constants) - Π) / ϵ
        @test dΠ ≈ -κᵈ * (cᵖᵐ / cᵛᵐ) * Π rtol=1e-6

        # Heating at fixed volume
        Q = 1e-3
        dθ_Q = (gas_potential_temperature(T + Q / cᵛᵐ, ρ, q, constants) - θ) / Q
        isochoric = PTF.heating_response(Dry(), θ, T, q, constants)
        @test dθ_Q ≈ isochoric rtol=1e-5

        # The pairs agree: 1/(cᵖᵐ Π) + θ (κᵐ − κᵈ) / (cᵛᵐ T) = (1 − κᵈ) / (cᵛᵐ Π)
        isobaric = PTF.heating_response(Dry(pressure_work=IsobaricPressureWork()), θ, T, q, constants)
        @test isobaric + θ * (Rᵐ / cᵖᵐ - κᵈ) / (cᵛᵐ * T) ≈ isochoric rtol=1e-13
    end
end

@testset "Phase-change weights against conservation of internal energy" begin
    T, ρ, pˢᵗ = 285.0, 1.0, 1e5
    c = constants
    ℒˡ₀ = TD.absolute_zero_latent_heat(c, c.liquid)
    ℒⁱ₀ = TD.absolute_zero_latent_heat(c, c.ice)
    eᵛ = (c.vapor.heat_capacity - TD.vapor_gas_constant(c)) * T + ℒˡ₀
    eˡ = c.liquid.heat_capacity * T
    eⁱ = c.ice.heat_capacity * T + ℒˡ₀ - ℒⁱ₀
    ϵ = 1e-7
    for q in test_compositions[2:end]
        cᵛᵐ = TD.mixture_heat_capacity(q, c) - TD.mixture_gas_constant(q, c)
        qᵈ = 1 - q.vapor - q.liquid - q.ice
        θ = gas_potential_temperature(T, ρ, q, c)
        Wᵛ, Wˡ, Wⁱ = PTF.phase_change_weights(Dry(), T, q, ρ, qᵈ * θ, pˢᵗ, DryExponent(), c)
        for (Δqᵛ, Δqˡ, Δqⁱ, analytic) in ((-ϵ, ϵ, 0.0, Wˡ - Wᵛ), (0.0, ϵ, -ϵ, Wˡ - Wⁱ))
            T₊ = T - (eᵛ * Δqᵛ + eˡ * Δqˡ + eⁱ * Δqⁱ) / cᵛᵐ
            q₊ = TD.MoistureMassFractions(q.vapor + Δqᵛ, q.liquid + Δqˡ, q.ice + Δqⁱ)
            brute = ρ * qᵈ * (gas_potential_temperature(T₊, ρ, q₊, c) - θ) / ϵ
            @test brute ≈ analytic rtol=1e-5
        end
    end
end

@testset "Equation of state and the acoustic pressure coefficient" begin
    T, ρ, pˢᵗ = 285.0, 1.0, 1e5
    γᵈ = 1 / (1 - dry_kappa(constants))
    for q in test_compositions
        qᵈ = 1 - q.vapor - q.liquid - q.ice
        ρᵈ = qᵈ * ρ
        Rᵐ = TD.mixture_gas_constant(q, constants)
        θ = gas_potential_temperature(T, ρ, q, constants)
        Π = T / θ
        pressure(Θ) = TD.air_pressure(TD.DryDensityState(Θ / ρᵈ, q, pˢᵗ, ρ), constants)
        Θ = ρᵈ * θ
        C = (pressure(Θ * (1 + 1e-6)) - pressure(Θ * (1 - 1e-6))) / (2e-6 * Θ)
        @test C ≈ γᵈ * (ρ / ρᵈ) * Rᵐ * Π rtol=1e-8

        # θ + S = θ γᵐ / γᵈ, so that the expansion source raises the pressure change of the loop's
        # flux to the moist sound speed γᵐ Rᵐ T (per unit total mass, under the force −qᵈ∇p)
        S = PTF.divergence_expansion_coefficient(θ, q, constants)
        cᵖᵐ = TD.mixture_heat_capacity(q, constants)
        γᵐ = cᵖᵐ / (cᵖᵐ - Rᵐ)
        @test θ + S ≈ θ * γᵐ / γᵈ rtol=1e-12
        @test qᵈ * C * (θ + S) ≈ γᵐ * Rᵐ * T rtol=1e-8
    end
end

@testset "Saturation adjustment step" begin
    c = tetens_constants
    Rᵈ, Rᵛ = TD.dry_air_gas_constant(c), TD.vapor_gas_constant(c)
    ℒˡ₀ = TD.absolute_zero_latent_heat(c, c.liquid)
    cᵛᵈ = c.dry_air.heat_capacity - Rᵈ
    cᵛᵛ = c.vapor.heat_capacity - Rᵛ
    cˡ = c.liquid.heat_capacity
    energy(T, rᵛ, rˡ) = (cᵛᵈ + cᵛᵛ * rᵛ + cˡ * rˡ) * T + ℒˡ₀ * rᵛ
    tetens = c.saturation_vapor_pressure
    saturation(T, ρᵈ) = (pᵛ⁺ = tetens.reference_saturation_vapor_pressure *
                               exp(tetens.liquid_coefficient * (T - tetens.reference_temperature) / (T - tetens.liquid_temperature_offset));
                         pᵛ⁺ / (Rᵛ * T * ρᵈ))
    ρᵈ = 1.0

    # Supersaturated: condenses to saturation at fixed internal energy, and warms
    T, rᵛ, rˡ = saturation_adjustment_step(IsochoricPressureWork(), 285.0, 0.015, 0.001, ρᵈ, 1e-6, c)
    @test T > 285
    @test rᵛ + rˡ ≈ 0.016 rtol=1e-14
    @test rᵛ ≈ saturation(T, ρᵈ) rtol=1e-3
    # (the iteration freezes the heat capacity and the latent heat at the first guess, as CM1's does)
    @test energy(T, rᵛ, rˡ) ≈ energy(285.0, 0.015, 0.001) rtol=1e-3

    # Subsaturated with cloud: evaporates the cloud, and cools
    T, rᵛ, rˡ = saturation_adjustment_step(IsochoricPressureWork(), 285.0, 0.001, 0.0005, ρᵈ, 1e-6, c)
    @test T < 285
    @test rˡ == 0
    @test rᵛ ≈ 0.0015 rtol=1e-14

    # Subsaturated and clear: nothing happens
    @test saturation_adjustment_step(IsochoricPressureWork(), 285.0, 0.001, 0.0, ρᵈ, 1e-6, c) == (285.0, 0.001, 0.0)
end

@testset "Construction" begin
    @test PotentialTemperatureFormulation() isa LiquidIcePotentialTemperatureFormulation
    @test PotentialTemperatureFormulation(flavor=Dry()) isa DryPotentialTemperatureFormulation
    @test DryPotentialTemperatureFormulation().exner_exponent isa DryExponent
    @test DryPotentialTemperatureFormulation().fixed_volume_phase_change isa AM.FixedVolumePhaseChange
    @test contains(sprint(show, DryPotentialTemperatureFormulation(pressure_work=IsobaricPressureWork())), "IsobaricPressureWork")

    grid = RectilinearGrid(default_arch; size=(4, 4), halo=(5, 5), x=(0, 1e3), z=(0, 1e3), topology=(Periodic, Flat, Bounded))
    dynamics() = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300.0)
    @test AtmosphereModel(grid; dynamics=dynamics(), formulation=:DryPotentialTemperature).formulation isa DryPotentialTemperatureFormulation
    @test AtmosphereModel(grid; dynamics=dynamics(), formulation=:DryPotentialTemperature).dynamics.momentum_force_weighting isa MomentumForceWeighting

    # LiquidIce is unweighted unless asked for
    @test AtmosphereModel(grid; dynamics=dynamics()).dynamics.momentum_force_weighting === nothing

    @test_throws ArgumentError AtmosphereModel(grid; dynamics=dynamics(), formulation=DryPotentialTemperatureFormulation(),
                                               microphysics=SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()))
    @test_throws ArgumentError AtmosphereModel(grid; dynamics=dynamics(), formulation=DryPotentialTemperatureFormulation(),
                                               thermodynamic_constants=tetens_constants,
                                               microphysics=DCMIP2016KesslerMicrophysics())

    anelastic = AnelasticDynamics(ReferenceState(grid, ThermodynamicConstants(eltype(grid))))
    @test_throws ArgumentError AtmosphereModel(grid; dynamics=anelastic, formulation=DryPotentialTemperatureFormulation())

    model = AtmosphereModel(grid; dynamics=dynamics(), formulation=DryPotentialTemperatureFormulation())
    @test_throws ArgumentError set!(model; θˡⁱ=300)
end

@testset "Dry air: identical to θˡⁱ with the dry exponent [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(32, 32), halo=(5, 5), x=(-10e3, 10e3), z=(0, 10e3),
                           topology=(Periodic, Flat, Bounded))
    θᵢ(x, z) = 300 + 0.002z + 2exp(-(x^2 + (z - 2e3)^2) / 1e6)

    function evolve(formulation)
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=z -> 300 + 0.002z)
        model = AtmosphereModel(grid; dynamics, formulation, advection=WENO(order=5))
        set!(model; θ=θᵢ, ρ=Breeze.HydrostaticallyBalancedDensity())
        for n in 1:100
            time_step!(model, 1)
        end
        return model
    end

    dry = evolve(DryPotentialTemperatureFormulation())
    li = evolve(LiquidIcePotentialTemperatureFormulation(exner_exponent=DryExponent(), temperature_solver=nothing))
    tolerance = FT == Float64 ? 1e-10 : 1e-3
    Θᵈ = Array(interior(AM.thermodynamic_density(dry.formulation)))
    Θˡ = Array(interior(AM.thermodynamic_density(li.formulation)))
    @test maximum(abs, Θᵈ .- Θˡ) / maximum(abs, Θˡ) ≤ tolerance
    @test maximum(abs, Array(interior(dry.velocities.w)) .- Array(interior(li.velocities.w))) ≤ (FT == Float64 ? 1e-9 : 1e-2)
end

@testset "Moist closed box conserves mass and water [$weighted]" for weighted in (MomentumForceWeighting(), nothing)
    FT = Float64
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(32, 32), halo=(5, 5), x=(-10e3, 10e3), z=(0, 10e3),
                           topology=(Bounded, Flat, Bounded))
    formulation = weighted === nothing ? LiquidIcePotentialTemperatureFormulation() : DryPotentialTemperatureFormulation()
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=z -> 300 + 0.002z,
                                    momentum_force_weighting=weighted)
    model = AtmosphereModel(grid; dynamics, formulation, advection=WENO(order=5), thermodynamic_constants=tetens_constants,
                            microphysics=OperatorSplitSaturationAdjustment())
    q₀(x, z) = 0.016exp(-z / 3000)
    set!(model; θ=(x, z) -> 300 + 0.002z + 2exp(-(x^2 + (z - 2e3)^2) / 1e6), qᵛ=q₀, ρ=Breeze.HydrostaticallyBalancedDensity())

    water(m) = sum(interior(m.moisture_density)) + sum(interior(m.microphysical_fields.ρqᶜˡ))
    drymass(m) = sum(interior(m.dynamics.dry_density))
    w₀, m₀ = water(model), drymass(model)
    for n in 1:60
        time_step!(model, 1)
    end
    @test water(model) ≈ w₀ rtol=1e-12
    @test drymass(model) ≈ m₀ rtol=1e-13
    @test maximum(interior(model.microphysical_fields.ρqᶜˡ)) > 0
    @test isfinite(maximum(abs, interior(model.velocities.w)))
end
