include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.Thermodynamics: Thermodynamics as TD
using Breeze.PotentialTemperatureFormulations: PotentialTemperatureFormulations as PTF
using Breeze.AtmosphereModels: AtmosphereModels as AM
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Test

const exponents = (VaporEquivalentExponent(), DryExponent())

# Specific enthalpy with vapor at absolute zero as the zero of every phase's latent energy
function specific_enthalpy(T, q, constants)
    ℒˡ₀ = TD.absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = TD.absolute_zero_latent_heat(constants, constants.ice)
    return TD.mixture_heat_capacity(q, constants) * T + ℒˡ₀ * q.vapor + (ℒˡ₀ - ℒⁱ₀) * q.ice
end

pressure_state(θ, q, p, exponent) = TD.LiquidIcePotentialTemperatureState(θ, q, oftype(θ, 1e5), p, exponent)

@testset "Liquid-ice potential temperature definition" begin
    @testset "Inversion round trip [$exponent, $FT]" for exponent in exponents, FT in (Float32, Float64)
        constants = ThermodynamicConstants(FT)
        q = TD.MoistureMassFractions(FT(0.012), FT(0.001), FT(0.0005))
        for p in FT.((101500, 70000, 40000, 30000)), T in FT.((240, 270, 300))
            𝒰 = @inferred TD.with_temperature(pressure_state(zero(FT), q, p, exponent), T, constants)
            @test 𝒰.exner_exponent === exponent
            @test (@inferred TD.temperature(𝒰, constants)) ≈ T rtol=4eps(FT)

            # Density-closed state (compressible core): Newton inversion at constant density
            ρ = p / (TD.mixture_gas_constant(q, constants) * T)
            solver = NewtonSolver(FT; abstol=10eps(FT) * T, maxiter=20)
            𝒰ρ = TD.with_temperature(TD.LiquidIceDensityState(zero(FT), q, FT(1e5), ρ, solver, exponent), T, constants)
            @test 𝒰ρ.exner_exponent === exponent
            @test (@inferred TD.temperature(𝒰ρ, constants)) ≈ T rtol=10eps(FT)
            # Both closures define the same θˡⁱ at the same (T, p, q)
            @test 𝒰ρ.potential_temperature ≈ 𝒰.potential_temperature rtol=10eps(FT)
        end
    end

    @testset "Phase change at fixed θˡⁱ, p and qᵗ conserves enthalpy [$exponent]" for exponent in exponents
        constants = ThermodynamicConstants(Float64)
        q₀ = TD.MoistureMassFractions(0.01, 0.001, 0.0005)
        δ = 1e-4
        # condensation, deposition, freezing
        transfers = ((-δ, δ, 0.0), (-δ, 0.0, δ), (0.0, -δ, δ))
        for (p, T) in ((100000.0, 300.0), (70000.0, 280.0), (40000.0, 255.0), (30000.0, 240.0))
            𝒰₀ = TD.with_temperature(pressure_state(0.0, q₀, p, exponent), T, constants)
            h₀ = specific_enthalpy(T, q₀, constants)
            for (Δqᵛ, Δqˡ, Δqⁱ) in transfers
                q₁ = TD.MoistureMassFractions(q₀.vapor + Δqᵛ, q₀.liquid + Δqˡ, q₀.ice + Δqⁱ)
                T₁ = TD.temperature(TD.with_moisture(𝒰₀, q₁), constants)
                @test specific_enthalpy(T₁, q₁, constants) ≈ h₀ atol=1e-9
            end

            # The heat released per unit condensate is the temperature-dependent latent heat
            ε = 1e-8
            q₁ = TD.MoistureMassFractions(q₀.vapor - ε, q₀.liquid + ε, q₀.ice)
            T₁ = TD.temperature(TD.with_moisture(𝒰₀, q₁), constants)
            ℒᵉᶠᶠ = TD.mixture_heat_capacity(q₁, constants) * (T₁ - T) / ε
            @test ℒᵉᶠᶠ ≈ TD.liquid_latent_heat(T, constants) rtol=1e-6
        end
    end

    @testset "Unsaturated air [$exponent]" for exponent in exponents
        constants = ThermodynamicConstants(Float64)
        q = TD.MoistureMassFractions(0.015)
        for p in (100000.0, 50000.0), T in (250.0, 300.0)
            𝒰 = TD.with_temperature(pressure_state(0.0, q, p, exponent), T, constants)
            Π = TD.exner_function(𝒰, constants)
            @test 𝒰.potential_temperature ≈ T / Π rtol=4eps(Float64)
            S = PTF.expansion_work_source(𝒰.potential_temperature, T, q, exponent, constants)
            if exponent isa VaporEquivalentExponent
                # κ̂ = Rᵐ / cᵖᵐ in unsaturated air: no expansion source
                @test abs(S) < 1e-12 * T
                @test Π ≈ (p / 1e5)^(TD.mixture_gas_constant(q, constants) / TD.mixture_heat_capacity(q, constants))
            else
                # The dry exponent leaves a small clear-air source, (Rᵐ - κᵈ cᵖᵐ) T / cᵖᵐ / Π per unit ln p
                @test S < 0
                @test Π == (p / 1e5)^(TD.dry_air_gas_constant(constants) / constants.dry_air.heat_capacity)
            end
        end
    end

    @testset "Expansion source restores the first law [$exponent]" for exponent in exponents
        constants = ThermodynamicConstants(Float64)
        q = TD.MoistureMassFractions(0.01, 0.002, 0.0005)
        ε = 1e-5
        for (p, T) in ((100000.0, 295.0), (70000.0, 280.0), (40000.0, 255.0))
            θ = TD.with_temperature(pressure_state(0.0, q, p, exponent), T, constants).potential_temperature
            S = PTF.expansion_work_source(θ, T, q, exponent, constants)
            # dT along d ln p = ε at fixed composition, with θ changing at S per unit ln p
            T⁺ = TD.temperature(pressure_state(θ + S * ε, q, p * exp(ε), exponent), constants)
            T⁻ = TD.temperature(pressure_state(θ - S * ε, q, p * exp(-ε), exponent), constants)
            cᵖᵐ = TD.mixture_heat_capacity(q, constants)
            Rᵐ = TD.mixture_gas_constant(q, constants)
            @test cᵖᵐ * (T⁺ - T⁻) / 2ε ≈ Rᵐ * T rtol=1e-7
            # Without the source, fixed θˡⁱ cools cloudy air at the wrong rate
            T⁺₀ = TD.temperature(pressure_state(θ, q, p * exp(ε), exponent), constants)
            T⁻₀ = TD.temperature(pressure_state(θ, q, p * exp(-ε), exponent), constants)
            @test !isapprox(cᵖᵐ * (T⁺₀ - T⁻₀) / 2ε, Rᵐ * T; rtol=1e-3)
        end
    end

    @testset "Sedimentation content matches ∂θˡⁱ/∂q [$exponent]" for exponent in exponents
        constants = ThermodynamicConstants(Float64)
        q = TD.MoistureMassFractions(0.01, 0.002, 0.0005)
        compressible = CompressibleDynamics(ExplicitTimeStepping())
        η = 1e-6
        for dynamics in (nothing, compressible), (p, T) in ((100000.0, 295.0), (40000.0, 255.0))
            θ = TD.with_temperature(pressure_state(0.0, q, p, exponent), T, constants).potential_temperature
            content = PTF.potential_temperature_condensate_content(dynamics, exponent, constants, q, θ, T)
            for (n, phase) in enumerate((Val(:liquid), Val(:ice)))
                Δq = AM.sedimentation_composition_increment(dynamics, q, phase)
                θ_along(s) = TD.with_temperature(pressure_state(0.0, TD.MoistureMassFractions(q.vapor + s * Δq.vapor,
                                                                                               q.liquid + s * Δq.liquid,
                                                                                               q.ice + s * Δq.ice),
                                                                p, exponent), T, constants).potential_temperature
                ∂θ∂q = (θ_along(η) - θ_along(-η)) / 2η
                @test content.χ[n] ≈ ∂θ∂q rtol=1e-7
            end
            # Heating response at fixed pressure and composition on the anelastic core
            if isnothing(dynamics)
                Π = TD.exner_function(pressure_state(θ, q, p, exponent), constants)
                @test content.∂φ∂h ≈ 1 / (TD.vapor_equivalent_heat_capacity(q, constants) * Π) rtol=1e-12
            end
        end
    end
end

@testset "LiquidIcePotentialTemperatureFormulation exponent" begin
    @test LiquidIcePotentialTemperatureFormulation().exponent isa VaporEquivalentExponent
    @test LiquidIcePotentialTemperatureFormulation(exponent=DryExponent()).exponent isa DryExponent
    @test_throws ArgumentError LiquidIcePotentialTemperatureFormulation(exponent=:dry)
    @test contains(sprint(show, LiquidIcePotentialTemperatureFormulation(exponent=DryExponent())), "exponent: DryExponent()")

    @testset "Anelastic model [$exponent, $FT]" for exponent in exponents, FT in (Float32, Float64)
        grid = RectilinearGrid(default_arch, FT; size=(2, 2, 8), x=(0, 1000), y=(0, 1000), z=(0, 4000))
        formulation = LiquidIcePotentialTemperatureFormulation(; exponent)
        ρE_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(100)))
        model = AtmosphereModel(grid; formulation, boundary_conditions=(; ρE=ρE_bcs),
                                microphysics=SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()))
        @test model.formulation.exponent === exponent
        @test model.formulation.potential_temperature isa Field
        @test model.formulation.potential_temperature_density.boundary_conditions.bottom.condition.exner_exponent === exponent

        # Saturated above the lowest kilometer, rising
        set!(model, θ = (x, y, z) -> 300 + 3e-3 * z, qᵗ = 0.02, w = (x, y, z) -> sin(π * z / 4000))
        @test maximum(model.microphysical_fields.qˡ) > 0

        # The θˡⁱ diagnostic uses the formulation's definition
        θˡⁱ = Field(LiquidIcePotentialTemperature(model))
        @test Array(interior(θˡⁱ)) ≈ Array(interior(model.formulation.potential_temperature)) rtol=10eps(FT)

        time_step!(model, 1)
        @test all(isfinite, Array(interior(model.formulation.potential_temperature)))
        @test all(isfinite, Array(interior(model.temperature)))
    end

    @testset "Symbol formulation gives the default exponent" begin
        grid = RectilinearGrid(default_arch; size=(1, 1, 4), extent=(1, 1, 1000))
        model = AtmosphereModel(grid; formulation=:LiquidIcePotentialTemperature)
        @test model.formulation.exponent isa VaporEquivalentExponent
    end

    @testset "Compressible model [$exponent]" for exponent in exponents
        grid = RectilinearGrid(default_arch; size=(2, 2, 8), x=(0, 1000), y=(0, 1000), z=(0, 4000))
        dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=300)
        formulation = LiquidIcePotentialTemperatureFormulation(; exponent)
        model = AtmosphereModel(grid; dynamics, formulation,
                                microphysics=SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()))
        @test model.formulation.exponent === exponent
        set!(model; ρ=model.dynamics.reference_state.density, θ=300, qᵗ=0.015)
        time_step!(model, 0.1)
        @test all(isfinite, Array(interior(model.formulation.potential_temperature)))
    end

    # In uniform, saturated air rising at uniform w, advection cancels in the interior and the ρθ
    # tendency is the expansion source alone, ρᵈ S D ln p / Dt with D ln p / Dt = −g ρ w / p.
    # The "Expansion source restores the first law" test checks S itself.
    @testset "Compressible expansion source [$exponent]" for exponent in exponents
        grid = RectilinearGrid(default_arch; size=(1, 1, 6), x=(0, 100), y=(0, 100), z=(0, 600))
        dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=300)
        formulation = LiquidIcePotentialTemperatureFormulation(; exponent)
        model = AtmosphereModel(grid; dynamics, formulation,
                                microphysics=SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()))
        constants = model.thermodynamic_constants
        g = constants.gravitational_acceleration
        set!(model; ρ=1.1, θ=295, qᵗ=0.025, w=2)
        update_state!(model)
        AM.compute_tendencies!(model)

        qᵛ = Array(interior(AM.specific_humidity(model)))
        qˡ = Array(interior(AM.liquid_mass_fraction(model)))
        T = Array(interior(model.temperature))
        θ = Array(interior(model.formulation.potential_temperature))
        p = Array(interior(AM.dynamics_pressure(model.dynamics)))
        ρ = Array(interior(AM.total_density(model.dynamics)))
        ρᵈ = Array(interior(AM.dynamics_density(model.dynamics)))
        G = Array(interior(model.timestepper.Gⁿ.ρθ))

        for k in 2:5
            @test qˡ[1, 1, k] > 0
            q = TD.MoistureMassFractions(qᵛ[1, 1, k], qˡ[1, 1, k])
            S = PTF.expansion_work_source(θ[1, 1, k], T[1, 1, k], q, exponent, constants)
            expected = ρᵈ[1, 1, k] * S * (- g * ρ[1, 1, k] * 2 / p[1, 1, k])
            @test G[1, 1, k] < 0
            @test G[1, 1, k] ≈ expected rtol=1e-10
        end
    end

    # A volumetric heating Q reaches ρᵈ θˡⁱ as (ρᵈ / ρ) Q / (ĉᵖ Π) on the compressible core
    # (NumericalEarth/Breeze.jl#1010)
    @testset "Compressible heating density ratio" begin
        constants = ThermodynamicConstants(Float64; gravitational_acceleration=0)
        qᵛ, T, p, Q = 0.02, 280.0, 1e5, 1.0
        q = TD.MoistureMassFractions(qᵛ)
        ρ = p / (TD.mixture_gas_constant(q, constants) * T)
        grid = RectilinearGrid(default_arch; size=(1, 1, 2), x=(0, 1), y=(0, 1), z=(0, 200))
        dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=T)
        model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants,
                                forcing=(; ρE=Forcing((x, y, z, t) -> Q)))
        set!(model; ρ, T, qᵛ)
        update_state!(model)
        AM.compute_tendencies!(model)
        observed = first(Array(interior(model.timestepper.Gⁿ.ρθ)))
        expected = (1 - qᵛ) * Q / TD.mixture_heat_capacity(q, constants)
        @test observed ≈ expected rtol=1e-6
    end
end
