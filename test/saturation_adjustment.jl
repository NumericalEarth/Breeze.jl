include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "supposition_setup.jl"))

using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

using Breeze.Thermodynamics:
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState,
    StaticEnergyState,
    exner_function,
    density,
    with_moisture,
    saturation_specific_humidity,
    mixture_heat_capacity,
    total_specific_moisture,
    PlanarLiquidSurface,
    PlanarMixedPhaseSurface

using Breeze.MoistAirBuoyancies: compute_boussinesq_adjustment_temperature
using Breeze.Microphysics: compute_temperature, adjust_thermodynamic_state

using Breeze: adjustment_saturation_specific_humidity

solver_tol(::Type{Float64}) = 1e-6
solver_tol(::Type{Float32}) = 1e-3
test_tol(FT::Type{Float64}) = 10 * sqrt(solver_tol(FT))
test_tol(FT::Type{Float32}) = sqrt(solver_tol(FT))
# test_tol is in kelvin; the equivalent relative tolerance for temperatures of O(300 K)
relative_test_tol(FT) = test_tol(FT) / 300

test_thermodynamics = (:StaticEnergy, :LiquidIcePotentialTemperature)

@testset "Saturation adjustment recovers the temperature [$(FT)]" for FT in all_float_types()
    constants = ThermodynamicConstants(FT)
    microphysics = SaturationAdjustment(FT; solver=SecantSolver(FT; abstol=solver_tol(FT)), equilibrium=WarmPhaseEquilibrium())
    rtol = relative_test_tol(FT)
    g = constants.gravitational_acceleration
    z = zero(FT)

    # A state assembled from (T, qᵛ⁺(T, p, qᵗ), qᵗ − qᵛ⁺) is already in equilibrium, so the
    # adjustment must return T within the solver tolerance; subsaturated draws (qᵗ ≤ qᵛ⁺)
    # exercise the identity branch.
    @breeze_check function saturation_adjustment_recovers_temperature(T = spstn_temperatures(FT; lo=270, hi=320),
                                                                      p = spstn_pressures(FT; lo=7e4, hi=1.05e5),
                                                                      qᵗ = spstn_floats(FT; lo=1e-3, hi=5e-2))
        qᵛ⁺ = adjustment_saturation_specific_humidity(T, p, qᵗ, constants, microphysics.equilibrium)
        saturated = qᵗ > qᵛ⁺
        event!("saturated", saturated)
        q = saturated ? MoistureMassFractions(qᵛ⁺, qᵗ - qᵛ⁺) : MoistureMassFractions(qᵗ)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        s = cᵖᵐ * T + g * z - constants.liquid.reference_latent_heat * q.liquid
        T★ = compute_temperature(StaticEnergyState(s, q, z, p), microphysics, constants)
        return qᵛ⁺ isa FT && isapprox(T★, T; rtol)
    end
end

# Liquid fraction of the condensate implied by an equilibrated surface
equilibrium_liquid_fraction(surface::PlanarMixedPhaseSurface) = surface.liquid_fraction
equilibrium_liquid_fraction(::PlanarLiquidSurface) = 1

@testset "Saturation adjustment invariants [$(FT)]" for FT in all_float_types()
    constants = ThermodynamicConstants(FT)
    g = constants.gravitational_acceleration
    tol = test_tol(FT)
    equilibria = (("warm", WarmPhaseEquilibrium()),
                  ("mixed", MixedPhaseEquilibrium(FT; freezing_temperature=273.15, homogeneous_ice_nucleation_temperature=233.15)))

    @testset "$name phase" for (name, equilibrium) in equilibria
        microphysics = SaturationAdjustment(FT; solver=SecantSolver(FT; abstol=solver_tol(FT)), equilibrium)

        # Starting from an all-vapor state with total moisture qᵗ, the adjustment must conserve
        # total water and the prognostic thermodynamic variable, keep every fraction
        # non-negative, leave unsaturated states all-vapor, put saturated states on the
        # saturation curve with the equilibrium liquid/ice partition, and be idempotent.
        function adjustment_invariants_hold(𝒰₀, qᵗ, thermodynamic_variable)
            𝒰★ = adjust_thermodynamic_state(𝒰₀, microphysics, constants)
            q = 𝒰★.moisture_mass_fractions
            qᶜ = q.liquid + q.ice
            T★ = compute_temperature(𝒰★, nothing, constants)
            qᵛ⁺ = saturation_specific_humidity(𝒰★, constants, equilibrium)
            event!("condensing", qᶜ > 0)

            conserved = isapprox(total_specific_moisture(q), qᵗ; rtol = spstn_rounding_rtol(FT)) &&
                        thermodynamic_variable(𝒰★) == thermodynamic_variable(𝒰₀)
            nonnegative = q.vapor >= 0 && q.liquid >= 0 && q.ice >= 0
            equilibrated = qᶜ == 0 ? qᵗ <= qᵛ⁺ * (1 + tol) : isapprox(q.vapor, qᵛ⁺; rtol = FT(0.1) * tol)
            λ = equilibrium_liquid_fraction(Breeze.Microphysics.equilibrated_surface(equilibrium, T★))
            partitioned = abs(q.liquid - λ * qᶜ) <= tol * qᶜ
            q★★ = adjust_thermodynamic_state(𝒰★, microphysics, constants).moisture_mass_fractions
            idempotent = isapprox(q★★.vapor, q.vapor; rtol = FT(0.1) * tol)
            return conserved && nonnegative && equilibrated && partitioned && idempotent
        end

        # Total moisture is drawn relative to saturation, from dry air to threefold supersaturation.
        @breeze_check function static_energy_adjustment_invariants(T = spstn_temperatures(FT; lo=230, hi=320),
                                                                   p = spstn_pressures(FT; lo=5e4, hi=1.05e5),
                                                                   z = spstn_heights(FT; lo=0, hi=5e3),
                                                                   f = spstn_floats(FT; lo=0, hi=3))
            qᵛ⁺₀ = equilibrium_saturation_specific_humidity(T, p, zero(FT), constants, equilibrium)
            qᵗ = min(f * qᵛ⁺₀, FT(0.05))
            q₀ = MoistureMassFractions(qᵗ)
            𝒰₀ = StaticEnergyState(mixture_heat_capacity(q₀, constants) * T + g * z, q₀, z, p)
            return adjustment_invariants_hold(𝒰₀, qᵗ, 𝒰 -> 𝒰.static_energy)
        end

        @breeze_check function potential_temperature_adjustment_invariants(θ = spstn_temperatures(FT; lo=230, hi=330),
                                                                           p = spstn_pressures(FT; lo=5e4, hi=1.05e5),
                                                                           f = spstn_floats(FT; lo=0, hi=3))
            𝒰_dry = LiquidIcePotentialTemperatureState(θ, MoistureMassFractions(zero(FT)), FT(1e5), p)
            T = compute_temperature(𝒰_dry, nothing, constants)
            qᵛ⁺₀ = equilibrium_saturation_specific_humidity(T, p, zero(FT), constants, equilibrium)
            qᵗ = min(f * qᵛ⁺₀, FT(0.05))
            𝒰₀ = with_moisture(𝒰_dry, MoistureMassFractions(qᵗ))
            return adjustment_invariants_hold(𝒰₀, qᵗ, 𝒰 -> 𝒰.potential_temperature)
        end
    end
end

@testset "Warm-phase saturation adjustment [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=288)

    atol = test_tol(FT)
    microphysics = SaturationAdjustment(FT; solver=SecantSolver(FT; abstol=solver_tol(FT)), equilibrium=WarmPhaseEquilibrium())

    pᵣ = @allowscalar first(reference_state.pressure)
    g = constants.gravitational_acceleration
    z = zero(FT)

    # Test 1: absolute zero
    q₀ = MoistureMassFractions{FT} |> zero
    𝒰₀ = StaticEnergyState(zero(FT), q₀, z, pᵣ)
    @test compute_temperature(𝒰₀, microphysics, constants) == 0

    # Test 2: unsaturated conditions
    T₁ = FT(300)
    ρ₁ = density(T₁, pᵣ, q₀, constants)
    qᵛ⁺ = saturation_specific_humidity(T₁, ρ₁, constants, constants.liquid)
    qᵗ = qᵛ⁺ / 2

    q₁ = MoistureMassFractions(qᵗ)
    cᵖᵐ = mixture_heat_capacity(q₁, constants)
    s₁ = cᵖᵐ * T₁ + g * z
    𝒰₁ = StaticEnergyState(s₁, q₁, z, pᵣ)

    @test compute_temperature(𝒰₁, microphysics, constants) ≈ T₁ atol=atol
    @test compute_temperature(𝒰₁, nothing, constants) ≈ T₁ atol=atol

    @testset "AtmosphereModel with $formulation thermodynamics [$FT]" for formulation in test_thermodynamics
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation, microphysics)
        ρᵣ = @allowscalar first(reference_state.density)

        # The scalar adjustment is covered by the property test above; here the model path is
        # exercised on a small (T, qᵗ) sweep.
        for T₂ in 280:20:320, qᵗ₂ in 1e-2:2e-2:5e-2
            @testset let T₂=T₂, qᵗ₂=qᵗ₂
                T₂ = convert(FT, T₂)
                qᵗ₂ = convert(FT, qᵗ₂)
                qᵛ⁺₂ = adjustment_saturation_specific_humidity(T₂, pᵣ, qᵗ₂, constants, microphysics.equilibrium)

                if qᵗ₂ > qᵛ⁺₂ # saturated conditions
                    qˡ₂ = qᵗ₂ - qᵛ⁺₂
                    q₂ = MoistureMassFractions(qᵛ⁺₂, qˡ₂)
                    cᵖᵐ = mixture_heat_capacity(q₂, constants)
                    ℒˡᵣ = constants.liquid.reference_latent_heat
                    s₂ = cᵖᵐ * T₂ + g * z - ℒˡᵣ * qˡ₂

                    set!(model, ρs = ρᵣ * s₂, qᵗ = qᵗ₂)
                    T★ = @allowscalar first(model.temperature)
                    qᵛ = @allowscalar first(model.microphysical_fields.qᵛ)
                    qˡ = @allowscalar first(model.microphysical_fields.qˡ)

                    @test T★ ≈ T₂ atol=atol
                    @test qᵛ ≈ qᵛ⁺₂ atol=atol
                    @test qˡ ≈ qˡ₂ atol=atol
                end
            end
        end
    end
end

function test_liquid_fraction(T, Tᶠ, Tʰ)
    T′ = clamp(T, Tʰ, Tᶠ)
    return (T′ - Tʰ) / (Tᶠ - Tʰ)
end

@testset "Mixed-phase saturation adjustment (AtmosphereModel) [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))

    constants = ThermodynamicConstants(FT)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    g = constants.gravitational_acceleration
    z = zero(FT)

    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=288)
    pᵣ = @allowscalar first(reference_state.pressure)
    ρᵣ = @allowscalar first(reference_state.density)

    atol = test_tol(FT)
    Tʰ = FT(233.15)
    Tᶠ = FT(273.15)

    equilibrium = MixedPhaseEquilibrium(FT; freezing_temperature=Tᶠ, homogeneous_ice_nucleation_temperature=Tʰ)
    microphysics = SaturationAdjustment(FT; solver=SecantSolver(FT; abstol=solver_tol(FT)), equilibrium)

    # Test only one formulation to reduce test count (StaticEnergy is representative)
    formulation = :StaticEnergy
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation, microphysics)

    # Test constructor
    @test microphysics isa SaturationAdjustment
    @test microphysics.equilibrium isa MixedPhaseEquilibrium{FT}

    @testset "equilibrated_surface" begin
        surface_above_freezing = Breeze.Microphysics.equilibrated_surface(equilibrium, FT(300))
        @test surface_above_freezing isa PlanarMixedPhaseSurface{FT}
        @test surface_above_freezing.liquid_fraction == 1

        surface_below_homogeneous = Breeze.Microphysics.equilibrated_surface(equilibrium, FT(200))
        @test surface_below_homogeneous.liquid_fraction == 0

        T_mid = FT(253.15)
        surface_midway = Breeze.Microphysics.equilibrated_surface(equilibrium, T_mid)
        λ_expected = test_liquid_fraction(T_mid, Tᶠ, Tʰ)
        @test surface_midway.liquid_fraction ≈ λ_expected
    end

    @testset "Temperatures above freezing (warm phase equivalence)" begin
        T_warm = FT(300)
        qᵗ = FT(0.02)
        qᵛ⁺ = equilibrium_saturation_specific_humidity(T_warm, pᵣ, qᵗ, constants, equilibrium)

        if qᵗ > qᵛ⁺
            qˡ = qᵗ - qᵛ⁺
            q = MoistureMassFractions(qᵛ⁺, qˡ)
            cᵖᵐ = mixture_heat_capacity(q, constants)
            s = cᵖᵐ * T_warm + g * z - ℒˡᵣ * qˡ

            𝒰 = StaticEnergyState(s, q, z, pᵣ)
            T★ = compute_temperature(𝒰, microphysics, constants)
            @test T★ ≈ T_warm atol=atol

            set!(model, ρs = ρᵣ * s, qᵗ = qᵗ)
            T★ = @allowscalar first(model.temperature)
            qᵛm = @allowscalar first(model.microphysical_fields.qᵛ)
            qˡm = @allowscalar first(model.microphysical_fields.qˡ)
            qⁱm = @allowscalar first(model.microphysical_fields.qⁱ)

            @test T★ ≈ T_warm atol=atol
            @test qᵛm ≈ qᵛ⁺ atol=atol
            @test qˡm ≈ qˡ atol=atol
            @test qⁱm ≈ zero(FT) atol=atol
        end
    end

    @testset "Temperatures below homogeneous ice nucleation (all ice)" begin
        T_cold = FT(220)
        qᵗ = FT(0.01)
        qᵛ⁺ = equilibrium_saturation_specific_humidity(T_cold, pᵣ, qᵗ, constants, equilibrium)

        if qᵗ > qᵛ⁺
            qⁱ = qᵗ - qᵛ⁺
            q = MoistureMassFractions(qᵛ⁺, zero(FT), qⁱ)
            cᵖᵐ = mixture_heat_capacity(q, constants)
            s = cᵖᵐ * T_cold + g * z - ℒⁱᵣ * qⁱ

            𝒰 = StaticEnergyState(s, q, z, pᵣ)
            T★ = compute_temperature(𝒰, microphysics, constants)
            @test T★ ≈ T_cold atol=atol

            set!(model, ρs = ρᵣ * s, qᵗ = qᵗ)
            T★ = @allowscalar first(model.temperature)
            qᵛm = @allowscalar first(model.microphysical_fields.qᵛ)
            qˡm = @allowscalar first(model.microphysical_fields.qˡ)
            qⁱm = @allowscalar first(model.microphysical_fields.qⁱ)

            @test T★ ≈ T_cold atol=atol
            @test qᵛm ≈ qᵛ⁺ atol=atol
            @test qˡm ≈ zero(FT) atol=atol
            @test qⁱm ≈ qⁱ atol=atol
        end
    end

    @testset "Mixed-phase range temperatures" begin
        # Test one temperature in the mixed phase range
        T = FT(253.15)
        λ = test_liquid_fraction(T, Tᶠ, Tʰ)
        qᵗ = FT(0.015)
        qᵛ⁺ = equilibrium_saturation_specific_humidity(T, pᵣ, qᵗ, constants, equilibrium)

        if qᵗ > qᵛ⁺
            q_condensate = qᵗ - qᵛ⁺
            qˡ = λ * q_condensate
            qⁱ = (1 - λ) * q_condensate
            q = MoistureMassFractions(qᵛ⁺, qˡ, qⁱ)

            @test q.vapor + q.liquid + q.ice ≈ qᵗ

            cᵖᵐ = mixture_heat_capacity(q, constants)
            s = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

            𝒰_unadjusted = StaticEnergyState(s, MoistureMassFractions(qᵗ), z, pᵣ)
            T★ = compute_temperature(𝒰_unadjusted, microphysics, constants)
            @test T★ ≈ T atol=atol

            set!(model, ρs = ρᵣ * s, qᵗ = qᵗ)
            T★ = @allowscalar first(model.temperature)
            qᵛm = @allowscalar first(model.microphysical_fields.qᵛ)
            qˡm = @allowscalar first(model.microphysical_fields.qˡ)
            qⁱm = @allowscalar first(model.microphysical_fields.qⁱ)

            @test T★ ≈ T atol=atol
            @test qᵛm ≈ qᵛ⁺ atol=atol
            @test qˡm ≈ qˡ atol=atol
            @test qⁱm ≈ qⁱ atol=atol
        end
    end
end

@testset "Saturation adjustment NaN robustness [$(FT)]" for FT in test_float_types()
    # Regression test: adjust_thermodynamic_state must never return NaN.
    # The secant iteration can stagnate (r₂ ≈ r₁) in Float32, producing ΔTΔr = Inf,
    # then T₂ = ±Inf, then NaN on the next iteration. We test a highly supersaturated
    # state that stresses the solver (qᵗ well above saturation).
    Oceananigans.defaults.FloatType = FT
    constants = ThermodynamicConstants(FT)
    g = constants.gravitational_acceleration
    z = zero(FT)

    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=288)
    pᵣ = @allowscalar first(reference_state.pressure)

    equilibrium = WarmPhaseEquilibrium()
    microphysics = SaturationAdjustment(FT; solver=SecantSolver(FT; abstol=1e-3, maxiter=10), equilibrium)

    # Highly supersaturated: qᵗ = 0.05 at T = 300 K (saturation is ~0.022)
    T_ref = FT(300)
    qᵗ = FT(0.05)
    q = MoistureMassFractions(qᵗ)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    s = cᵖᵐ * T_ref + g * z  # energy without condensate correction (all-vapor initial)
    𝒰₀ = StaticEnergyState(s, q, z, pᵣ)

    𝒰₁ = adjust_thermodynamic_state(𝒰₀, microphysics, constants)
    T★ = compute_temperature(𝒰₁, nothing, constants)

    @test isfinite(T★)
    @test 𝒰₁.moisture_mass_fractions.liquid > 0   # condensate formed
    @test isfinite(𝒰₁.moisture_mass_fractions.liquid)
end

@testset "Saturation adjustment (MoistAirBuoyancies)" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=288)
    atol = test_tol(FT)

    pᵣ = @allowscalar reference_state.pressure[1, 1, 1]
    p₀ = reference_state.base_pressure
    z = FT(0.5)

    # Case 0: Absolute zero potential temperature
    θ₀ = zero(FT)
    q₀ = MoistureMassFractions{FT} |> zero
    𝒰₀ = LiquidIcePotentialTemperatureState(θ₀, q₀, p₀, pᵣ)
    T₀ = compute_boussinesq_adjustment_temperature(𝒰₀, constants)
    @test T₀ == 0

    # Case 1: Unsaturated, dry
    θ₁ = FT(300)
    qᵗ₁ = zero(FT)
    q₁ = MoistureMassFractions(qᵗ₁)
    𝒰₁ = LiquidIcePotentialTemperatureState(θ₁, q₁, p₀, pᵣ)
    Π₁ = exner_function(𝒰₁, constants)
    T_dry₁ = Π₁ * θ₁

    T₁ = compute_boussinesq_adjustment_temperature(𝒰₁, constants)
    @test isapprox(T₁, T_dry₁; atol=atol)

    # Case 2: Unsaturated, humid
    θ₂ = FT(300)
    q₂ = MoistureMassFractions{FT} |> zero
    𝒰₂ = LiquidIcePotentialTemperatureState(θ₂, q₂, p₀, pᵣ)
    Π₂ = exner_function(𝒰₂, constants)
    T_dry₂ = Π₂ * θ₂

    ρ₂ = density(T_dry₂, pᵣ, q₂, constants)
    qᵛ⁺₂ = saturation_specific_humidity(T_dry₂, ρ₂, constants, constants.liquid)
    qᵗ₂ = qᵛ⁺₂ / 2
    q₂ = MoistureMassFractions(qᵗ₂)
    𝒰₂ = with_moisture(𝒰₂, q₂)

    T₂ = compute_boussinesq_adjustment_temperature(𝒰₂, constants)
    Π₂ = exner_function(𝒰₂, constants)
    T_dry₂ = Π₂ * θ₂
    @test isapprox(T₂, T_dry₂; atol=atol)

    # Case 3: Saturated
    T₃ = θ̃ = FT(300)
    qᵗ = FT(0.025)
    q̃ = MoistureMassFractions(qᵗ)
    𝒰 = LiquidIcePotentialTemperatureState(θ̃, q̃, p₀, pᵣ)
    qᵛ⁺ = equilibrium_saturation_specific_humidity(T₃, pᵣ, qᵗ, constants, constants.liquid)
    @test qᵗ > qᵛ⁺

    qˡ = qᵗ - qᵛ⁺
    q₃ = MoistureMassFractions(qᵛ⁺, qˡ)
    𝒰₃ = with_moisture(𝒰, q₃)
    Π₃ = exner_function(𝒰₃, constants)
    cᵖᵐ = mixture_heat_capacity(q₃, constants)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    θ₃ = (T₃ - ℒˡᵣ / cᵖᵐ * qˡ) / Π₃
    𝒰₃ = LiquidIcePotentialTemperatureState(θ₃, q₃, p₀, pᵣ)

    T₃_solve = compute_boussinesq_adjustment_temperature(𝒰₃, constants)
    @test isapprox(T₃_solve, T₃; atol=atol)
end
