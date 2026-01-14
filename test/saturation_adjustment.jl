using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

include("test_utils.jl")

using Breeze.Thermodynamics:
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState,
    StaticEnergyState,
    exner_function,
    density,
    with_moisture,
    saturation_specific_humidity,
    mixture_heat_capacity,
    PlanarMixedPhaseSurface

using Breeze.MoistAirBuoyancies: compute_boussinesq_adjustment_temperature
using Breeze.Microphysics: compute_temperature

using Breeze: adjustment_saturation_specific_humidity

solver_tol(::Type{Float64}) = 1e-6
solver_tol(::Type{Float32}) = 1e-3
test_tol(FT::Type{Float64}) = 10 * sqrt(solver_tol(FT))
test_tol(FT::Type{Float32}) = sqrt(solver_tol(FT))

test_thermodynamics = (:StaticEnergy, :LiquidIcePotentialTemperature)

@testset "Warm-phase saturation adjustment [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=288)

    atol = test_tol(FT)
    microphysics = SaturationAdjustment(FT; tolerance=solver_tol(FT), equilibrium=WarmPhaseEquilibrium())

    # Sample a single cell
    pᵣ = @allowscalar first(reference_state.pressure)
    g = constants.gravitational_acceleration
    z = zero(FT)

    # First test: absolute zero
    q₀ = MoistureMassFractions{FT} |> zero
    𝒰₀ = StaticEnergyState(zero(FT), q₀, z, pᵣ)
    @test compute_temperature(𝒰₀, microphysics, constants) == 0

    # Second unsaturated test: choose T, pick qᵗ well below saturation
    T₁ = FT(300)
    ρ₁ = density(T₁, pᵣ, q₀, constants)
    qᵛ⁺ = saturation_specific_humidity(T₁, ρ₁, constants, constants.liquid)
    qᵗ = qᵛ⁺ / 2 # comfortably unsaturated

    q₁ = MoistureMassFractions(qᵗ)
    cᵖᵐ = mixture_heat_capacity(q₁, constants)
    e₁ = cᵖᵐ * T₁ + g * z #  + ℒ₀ * qᵗ
    𝒰₁ = StaticEnergyState(e₁, q₁, z, pᵣ)

    @test compute_temperature(𝒰₁, microphysics, constants) ≈ T₁ atol=atol
    @test compute_temperature(𝒰₁, nothing, constants) ≈ T₁ atol=atol

    @testset "AtmosphereModel with $formulation thermodynamics [$FT]" for formulation in test_thermodynamics
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation, microphysics)
        ρᵣ = @allowscalar first(reference_state.density)

        # Reduced parameter sweep for faster testing (was 14×21 = 294 per FT, now 5×7 = 35)
        for T₂ in 270:12:320, qᵗ₂ in 1e-2:7e-3:5e-2
            @testset let T₂=T₂, qᵗ₂=qᵗ₂
                T₂ = convert(FT, T₂)
                qᵗ₂ = convert(FT, qᵗ₂)
                qᵛ⁺₂ = adjustment_saturation_specific_humidity(T₂, pᵣ, qᵗ₂, constants, microphysics.equilibrium)
                @test qᵛ⁺₂ isa FT

                if qᵗ₂ > qᵛ⁺₂ # saturated conditions
                    qˡ₂ = qᵗ₂ - qᵛ⁺₂
                    q₂ = MoistureMassFractions(qᵛ⁺₂, qˡ₂)
                    cᵖᵐ = mixture_heat_capacity(q₂, constants)
                    ℒˡᵣ = constants.liquid.reference_latent_heat
                    e₂ = cᵖᵐ * T₂ + g * z - ℒˡᵣ * qˡ₂

                    𝒰₂ = StaticEnergyState(e₂, q₂, z, pᵣ)
                    T★ = compute_temperature(𝒰₂, microphysics, constants)
                    @test T★ ≈ T₂ atol=atol

                    # Parcel test for AtmosphereModel
                    set!(model, ρe = ρᵣ * e₂, qᵗ = qᵗ₂)
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

    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=288)
    pᵣ = @allowscalar first(reference_state.pressure)
    ρᵣ = @allowscalar first(reference_state.density)

    atol = test_tol(FT)
    Tʰ = FT(233.15)  # Homogeneous ice nucleation temperature
    Tᶠ = FT(273.15)  # Freezing temperature

    equilibrium = MixedPhaseEquilibrium(FT; freezing_temperature=Tᶠ, homogeneous_ice_nucleation_temperature=Tʰ)
    microphysics = SaturationAdjustment(FT; tolerance=solver_tol(FT), equilibrium)

    @testset "AtmosphereModel with $formulation thermodynamics [$FT]" for formulation in test_thermodynamics
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation, microphysics)

        # Test 1: Constructor and equilibrated_surface utility
        @test microphysics isa SaturationAdjustment
        @test microphysics.equilibrium isa MixedPhaseEquilibrium{FT}
        @test microphysics.equilibrium.freezing_temperature == Tᶠ
        @test microphysics.equilibrium.homogeneous_ice_nucleation_temperature == Tʰ

        @test model.microphysics isa SaturationAdjustment
        @test model.microphysics.equilibrium isa MixedPhaseEquilibrium{FT}
        @test model.microphysics.equilibrium.freezing_temperature == Tᶠ
        @test model.microphysics.equilibrium.homogeneous_ice_nucleation_temperature == Tʰ

        @testset "equilibrated_surface" begin
            # Test equilibrated_surface at different temperatures
            surface_above_freezing = Breeze.Microphysics.equilibrated_surface(equilibrium, FT(300))
            @test surface_above_freezing isa PlanarMixedPhaseSurface{FT}
            @test surface_above_freezing.liquid_fraction == 1  # Above freezing, all liquid

            surface_below_homogeneous_ice_nucleation = Breeze.Microphysics.equilibrated_surface(equilibrium, FT(200))
            @test surface_below_homogeneous_ice_nucleation isa PlanarMixedPhaseSurface{FT}
            @test surface_below_homogeneous_ice_nucleation.liquid_fraction == 0  # Below homogeneous nucleation, all ice

            T_mid = FT(253.15)  # Midway between Tᶠ and Tʰ
            surface_midway = Breeze.Microphysics.equilibrated_surface(equilibrium, T_mid)
            @test surface_midway isa PlanarMixedPhaseSurface{FT}
            λ_expected = test_liquid_fraction(T_mid, Tᶠ, Tʰ)
            @test surface_midway.liquid_fraction ≈ λ_expected
        end

        # Test 2: Temperatures above freezing - should match warm phase behavior
        @testset "Temperatures above freezing (warm phase equivalence)" begin
            T_warm = FT(300)
            qᵗ = FT(0.02)
            qᵛ⁺ = equilibrium_saturation_specific_humidity(T_warm, pᵣ, qᵗ, constants, equilibrium)
            atol = test_tol(FT)

            if qᵗ > qᵛ⁺  # saturated conditions
                # For warm temperatures, all condensate should be liquid
                qˡ = qᵗ - qᵛ⁺
                q = MoistureMassFractions(qᵛ⁺, qˡ)
                cᵖᵐ = mixture_heat_capacity(q, constants)
                e = cᵖᵐ * T_warm + g * z - ℒˡᵣ * qˡ

                𝒰 = StaticEnergyState(e, q, z, pᵣ)
                T★ = compute_temperature(𝒰, microphysics, constants)
                @test T★ ≈ T_warm atol=atol

                # Parcel test for AtmosphereModel
                set!(model, ρe = ρᵣ * e, qᵗ = qᵗ)
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

        # Test 3: Temperatures below homogeneous ice nucleation - all ice
        @testset "Temperatures below homogeneous ice nucleation (all ice)" begin
            T_cold = FT(220)  # Below Tʰ
            qᵗ = FT(0.01)
            qᵛ⁺ = equilibrium_saturation_specific_humidity(T_cold, pᵣ, qᵗ, constants, equilibrium)
            atol = test_tol(FT)

            if qᵗ > qᵛ⁺  # saturated conditions
                # All condensate should be ice
                qⁱ = qᵗ - qᵛ⁺
                q = MoistureMassFractions(qᵛ⁺, zero(FT), qⁱ)
                cᵖᵐ = mixture_heat_capacity(q, constants)
                e = cᵖᵐ * T_cold + g * z - ℒⁱᵣ * qⁱ

                𝒰 = StaticEnergyState(e, q, z, pᵣ)
                T★ = compute_temperature(𝒰, microphysics, constants)
                @test T★ ≈ T_cold atol=atol

                set!(model, ρe = ρᵣ * e, qᵗ = qᵗ)
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

        # Test 4: Mixed-phase range temperatures with moist static energy verification
        @testset "Mixed-phase range temperatures with moist static energy" begin
            atol = test_tol(FT)

            # Reduced from 4 to 3 temperatures
            for T in 240:15:270
                @testset let T=T
                    T = convert(FT, T)
                    λ = test_liquid_fraction(T, Tᶠ, Tʰ)
                    qᵗ = FT(0.015)
                    qᵛ⁺ = equilibrium_saturation_specific_humidity(T, pᵣ, qᵗ, constants, equilibrium)

                    if qᵗ > qᵛ⁺  # saturated conditions
                        # Partition condensate between liquid and ice based on λ
                        q_condensate = qᵗ - qᵛ⁺
                        qˡ = λ * q_condensate
                        qⁱ = (1 - λ) * q_condensate
                        q = MoistureMassFractions(qᵛ⁺, qˡ, qⁱ)

                        # Verify partitioning sums correctly
                        @test q.vapor + q.liquid + q.ice ≈ qᵗ

                        # Compute moist static energy: e = cᵖᵐ*T + g*z - ℒˡᵣ*qˡ - ℒⁱᵣ*qⁱ
                        cᵖᵐ = mixture_heat_capacity(q, constants)
                        e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

                        # Verify moist static energy can recover temperature
                        𝒰 = StaticEnergyState(e, q, z, pᵣ)
                        T_recovered = (e - g * z + ℒˡᵣ * q.liquid + ℒⁱᵣ * q.ice) / mixture_heat_capacity(q, constants)
                        @test T_recovered ≈ T

                        # Test saturation adjustment recovers temperature
                        𝒰_unadjusted = StaticEnergyState(e, MoistureMassFractions(qᵗ), z, pᵣ)
                        T★ = compute_temperature(𝒰_unadjusted, microphysics, constants)
                        @test T★ ≈ T atol=atol

                        set!(model, ρe = ρᵣ * e, qᵗ = qᵗ)
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
        end
    end

    # Test 5: Verify moist static energy formula with various moisture fractions
    @testset "Moist static energy formula verification" begin
        atol = test_tol(FT)
        T = FT(253.15)  # Midway in mixed-phase range
        λ = test_liquid_fraction(T, Tᶠ, Tʰ)

        # Reduced from 6 to 3 moisture values
        for qᵗ in FT.(5e-3:1e-2:3e-2)
            @testset let qᵗ=qᵗ
                qᵛ⁺ = equilibrium_saturation_specific_humidity(T, pᵣ, qᵗ, constants, equilibrium)

                if qᵗ > qᵛ⁺  # saturated conditions
                    qᶜ = qᵗ - qᵛ⁺
                    qˡ = λ * qᶜ
                    qⁱ = (1 - λ) * qᶜ
                    q = MoistureMassFractions(qᵛ⁺, qˡ, qⁱ)

                    # Compute moist static energy
                    cᵖᵐ = mixture_heat_capacity(q, constants)
                    e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

                    # Test with saturation adjustment
                    𝒰 = StaticEnergyState(e, MoistureMassFractions(qᵗ), z, pᵣ)
                    T★ = compute_temperature(𝒰, microphysics, constants)
                    @test T★ ≈ T atol=atol
                end
            end
        end
    end

    # Test 6: Verify partitioning matches temperature-dependent λ
    @testset "Condensate partitioning verification" begin
        atol = test_tol(FT)
        # Reduced from 4 to 3 temperatures
        for T_partition in 235:15:265
            @testset let T_partition=T_partition
                T_partition = convert(FT, T_partition)
                λ_expected = test_liquid_fraction(T_partition, Tᶠ, Tʰ)

                qᵗ = FT(0.02)
                qᵛ⁺ = equilibrium_saturation_specific_humidity(T_partition, pᵣ, qᵗ, constants, equilibrium)

                if qᵗ > qᵛ⁺  # saturated conditions
                    q_condensate = qᵗ - qᵛ⁺
                    qˡ = λ_expected * q_condensate
                    qⁱ = (1 - λ_expected) * q_condensate
                    q = MoistureMassFractions(qᵛ⁺, qˡ, qⁱ)

                    # Verify partitioning
                    if q_condensate > 0
                        λ_actual = q.liquid / q_condensate
                        @test λ_actual ≈ λ_expected
                        @test q.ice / q_condensate ≈ (1 - λ_expected)
                    end

                    # Verify moist static energy
                    cᵖᵐ = mixture_heat_capacity(q, constants)
                    e = cᵖᵐ * T_partition + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

                    𝒰 = StaticEnergyState(e, MoistureMassFractions(qᵗ), z, pᵣ)
                    T★ = compute_temperature(𝒰, microphysics, constants)
                    @test T★ ≈ T_partition atol=atol
                end
            end
        end
    end
end

@testset "Saturation adjustment (MoistAirBuoyancies)" for FT in test_float_types()
    # Minimal grid and reference state
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=288)
    atol = test_tol(FT)

    # Sample a single cell
    pᵣ = @allowscalar reference_state.pressure[1, 1, 1]
    ρᵣ = @allowscalar reference_state.density[1, 1, 1]
    p₀ = reference_state.surface_pressure
    z = FT(0.5)

    # Case 0: Absolute zero potential temperature returns zero temperature
    θ₀ = zero(FT)
    q₀ = MoistureMassFractions{FT} |> zero
    𝒰₀ = LiquidIcePotentialTemperatureState(θ₀, q₀, p₀, pᵣ)
    T₀ = compute_boussinesq_adjustment_temperature(𝒰₀, constants)
    @test T₀ == 0

    # Case 1: Unsaturated, dry (qᵗ = 0)
    θ₁ = FT(300)
    qᵗ₁ = zero(FT)
    q₁ = MoistureMassFractions(qᵗ₁)
    𝒰₁ = LiquidIcePotentialTemperatureState(θ₁, q₁, p₀, pᵣ)
    Π₁ = exner_function(𝒰₁, constants)
    T_dry₁ = Π₁ * θ₁

    T₁ = compute_boussinesq_adjustment_temperature(𝒰₁, constants)
    @test isapprox(T₁, T_dry₁; atol=atol)

    # Case 2: Unsaturated, humid but below saturation at dry temperature
    θ₂ = FT(300)
    q₂ = MoistureMassFractions{FT} |> zero
    𝒰₂ = LiquidIcePotentialTemperatureState(θ₂, q₂, p₀, pᵣ)
    Π₂ = exner_function(𝒰₂, constants)
    T_dry₂ = Π₂ * θ₂

    # Choose qᵗ well below saturation at T_dry₂
    ρ₂ = density(T_dry₂, pᵣ, q₂, constants)
    qᵛ⁺₂ = saturation_specific_humidity(T_dry₂, ρ₂, constants, constants.liquid)
    qᵗ₂ = qᵛ⁺₂ / 2
    q₂ = MoistureMassFractions(qᵗ₂)
    𝒰₂ = with_moisture(𝒰₂, q₂)

    T₂ = compute_boussinesq_adjustment_temperature(𝒰₂, constants)
    Π₂ = exner_function(𝒰₂, constants)
    T_dry₂ = Π₂ * θ₂
    @test isapprox(T₂, T_dry₂; atol=atol)

    # Case 3: Saturated, humid (qᵗ = qᵛ⁺)
    T₃ = θ̃ = FT(300)
    qᵗ = FT(0.025)
    q̃ = MoistureMassFractions(qᵗ)
    𝒰 = LiquidIcePotentialTemperatureState(θ̃, q̃, p₀, pᵣ)
    qᵛ⁺ = equilibrium_saturation_specific_humidity(T₃, pᵣ, qᵗ, constants, constants.liquid)
    @test qᵗ > qᵛ⁺ # otherwise the test is wrong

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
