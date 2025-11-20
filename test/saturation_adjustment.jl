using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

using Breeze.Thermodynamics:
    MoistureMassFractions,
    PotentialTemperatureState,
    MoistStaticEnergyState,
    exner_function,
    density,
    with_moisture,
    saturation_specific_humidity,
    mixture_heat_capacity,
    PlanarMixedPhaseSurface

using Breeze.MoistAirBuoyancies: compute_boussinesq_adjustment_temperature
using Breeze.Microphysics: compute_temperature

using Breeze.Microphysics:
    adjustment_saturation_specific_humidity

@testset "Warm-phase saturation adjustment (AtmosphereModel) [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    thermo = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, thermo; base_pressure=101325, potential_temperature=288)

    tol = FT(1e-3)
    microphysics = SaturationAdjustment(FT; tolerance=tol, equilibrium=WarmPhaseEquilibrium())

    # Sample a single cell
    pᵣ = @allowscalar first(reference_state.pressure)
    g = thermo.gravitational_acceleration
    z = zero(FT)

    # First test: absolute zero
    q₀ = MoistureMassFractions{FT} |> zero
    𝒰₀ = MoistStaticEnergyState(zero(FT), q₀, z, pᵣ)
    @test compute_temperature(𝒰₀, microphysics, thermo) == 0

    # Second unsaturated test: choose T, pick qᵗ well below saturation
    T₁ = FT(300)
    ρ₁ = density(pᵣ, T₁, q₀, thermo)
    qᵛ⁺ = saturation_specific_humidity(T₁, ρ₁, thermo, thermo.liquid)
    qᵗ = qᵛ⁺ / 2 # comfortably unsaturated

    q₁ = MoistureMassFractions(qᵗ)
    cᵖᵐ = mixture_heat_capacity(q₁, thermo)
    e₁ = cᵖᵐ * T₁ + g * z #  + ℒ₀ * qᵗ
    𝒰₁ = MoistStaticEnergyState(e₁, q₁, z, pᵣ)

    @test compute_temperature(𝒰₁, microphysics, thermo) ≈ T₁ atol=sqrt(tol)
    @test compute_temperature(𝒰₁, nothing, thermo) ≈ T₁ atol=sqrt(tol)

    formulation = AnelasticFormulation(reference_state)
    model = AtmosphereModel(grid; thermodynamics=thermo, formulation, microphysics)
    ρᵣ = @allowscalar first(reference_state.density)

    # Many more tests that touch saturated conditions
    for T₂ in 270:4:320, qᵗ₂ in 1e-2:2e-3:5e-2
        @testset let T₂=T₂, qᵗ₂=qᵗ₂
            T₂ = convert(FT, T₂)
            qᵗ₂ = convert(FT, qᵗ₂)
            qᵛ⁺₂ = adjustment_saturation_specific_humidity(T₂, pᵣ, qᵗ₂, thermo, microphysics.equilibrium)
            @test qᵛ⁺₂ isa FT

            if qᵗ₂ > qᵛ⁺₂ # saturated conditions
                qˡ₂ = qᵗ₂ - qᵛ⁺₂
                q₂ = MoistureMassFractions(qᵛ⁺₂, qˡ₂)
                cᵖᵐ = mixture_heat_capacity(q₂, thermo)
                ℒˡᵣ = thermo.liquid.reference_latent_heat
                e₂ = cᵖᵐ * T₂ + g * z - ℒˡᵣ * qˡ₂

                𝒰₂ = MoistStaticEnergyState(e₂, q₂, z, pᵣ)
                T★ = compute_temperature(𝒰₂, microphysics, thermo)
                @test T★ ≈ T₂ atol=sqrt(tol)

                # Parcel test for AtmosphereModel
                set!(model, ρe = ρᵣ * e₂, qᵗ = qᵗ₂)    
                T★ = @allowscalar first(model.temperature)
                qᵛ = @allowscalar first(model.microphysical_fields.qᵛ)
                qˡ = @allowscalar first(model.microphysical_fields.qˡ)

                @test T★ ≈ T₂ atol=sqrt(tol)
                @test qᵛ ≈ qᵛ⁺₂ 
                @test qˡ ≈ qˡ₂
            end
        end
    end
end

function test_liquid_fraction(T, Tᶠ, Tʰ) 
    T′ = clamp(T, Tʰ, Tᶠ)
    return (T′ - Tʰ) / (Tᶠ - Tʰ)
end

@testset "Mixed-phase saturation adjustment (AtmosphereModel) [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    thermo = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, thermo; base_pressure=101325, potential_temperature=288)

    tol = 1e-3
    Tᶠ = FT(273.15)  # Freezing temperature
    Tʰ = FT(233.15)  # Homogeneous ice nucleation temperature

    equilibrium = MixedPhaseEquilibrium(FT; freezing_temperature=Tᶠ, homogeneous_ice_nucleation_temperature=Tʰ)
    microphysics = SaturationAdjustment(FT; tolerance=tol, equilibrium=equilibrium)
    formulation = AnelasticFormulation(reference_state)
    model = AtmosphereModel(grid; thermodynamics=thermo, formulation, microphysics)
    ρᵣ = @allowscalar first(reference_state.density)

    # Sample a single cell
    pᵣ = @allowscalar first(reference_state.pressure)
    g = thermo.gravitational_acceleration
    z = zero(FT)
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    ℒⁱᵣ = thermo.ice.reference_latent_heat

    # Test 1: Constructor and equilibrated_surface utility
    @testset "Constructor and equilibrated_surface" begin
        @test microphysics isa SaturationAdjustment
        @test microphysics.equilibrium isa MixedPhaseEquilibrium{FT}
        @test microphysics.equilibrium.freezing_temperature == Tᶠ
        @test microphysics.equilibrium.homogeneous_ice_nucleation_temperature == Tʰ

        @test model.microphysics isa SaturationAdjustment
        @test model.microphysics.equilibrium isa MixedPhaseEquilibrium{FT}
        @test model.microphysics.equilibrium.freezing_temperature == Tᶠ
        @test model.microphysics.equilibrium.homogeneous_ice_nucleation_temperature == Tʰ

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
        qᵛ⁺ = adjustment_saturation_specific_humidity(T_warm, pᵣ, qᵗ, thermo, equilibrium)

        if qᵗ > qᵛ⁺  # saturated conditions
            # For warm temperatures, all condensate should be liquid
            qˡ = qᵗ - qᵛ⁺
            q = MoistureMassFractions(qᵛ⁺, qˡ)
            cᵖᵐ = mixture_heat_capacity(q, thermo)
            e = cᵖᵐ * T_warm + g * z - ℒˡᵣ * qˡ

            𝒰 = MoistStaticEnergyState(e, q, z, pᵣ)
            T★ = compute_temperature(𝒰, microphysics, thermo)
            @test T★ ≈ T_warm atol=sqrt(tol)

            # Parcel test for AtmosphereModel
            set!(model, ρe = ρᵣ * e, qᵗ = qᵗ)
            T★ = @allowscalar first(model.temperature)
            qᵛm = @allowscalar first(model.microphysical_fields.qᵛ)
            qˡm = @allowscalar first(model.microphysical_fields.qˡ)
            qⁱm = @allowscalar first(model.microphysical_fields.qⁱ)

            @test T★ ≈ T_warm atol=sqrt(tol)
            @test qᵛm ≈ qᵛ⁺
            @test qˡm ≈ qˡ
            @test qⁱm ≈ zero(FT) atol=sqrt(tol)
        end
    end

    # Test 3: Temperatures below homogeneous ice nucleation - all ice
    @testset "Temperatures below homogeneous ice nucleation (all ice)" begin
        T_cold = FT(220)  # Below Tʰ
        qᵗ = FT(0.01)
        qᵛ⁺ = adjustment_saturation_specific_humidity(T_cold, pᵣ, qᵗ, thermo, equilibrium)

        if qᵗ > qᵛ⁺  # saturated conditions
            # All condensate should be ice
            qⁱ = qᵗ - qᵛ⁺
            q = MoistureMassFractions(qᵛ⁺, zero(FT), qⁱ)
            cᵖᵐ = mixture_heat_capacity(q, thermo)
            e = cᵖᵐ * T_cold + g * z - ℒⁱᵣ * qⁱ

            𝒰 = MoistStaticEnergyState(e, q, z, pᵣ)
            T★ = compute_temperature(𝒰, microphysics, thermo)
            @test T★ ≈ T_cold atol=sqrt(tol)

            set!(model, ρe = ρᵣ * e, qᵗ = qᵗ)
            T★ = @allowscalar first(model.temperature)
            qᵛm = @allowscalar first(model.microphysical_fields.qᵛ)
            qˡm = @allowscalar first(model.microphysical_fields.qˡ)
            qⁱm = @allowscalar first(model.microphysical_fields.qⁱ)

            @test T★ ≈ T_cold atol=sqrt(tol)
            @test qᵛm ≈ qᵛ⁺
            @test qˡm ≈ zero(FT)
            @test qⁱm ≈ qⁱ
        end
    end

    # Test 4: Mixed-phase range temperatures with moist static energy verification
    @testset "Mixed-phase range temperatures with moist static energy" begin
        for T in [FT(240), FT(250), FT(260), FT(270)]
            @testset let T=T
                λ = test_liquid_fraction(T, Tᶠ, Tʰ)
                qᵗ = FT(0.015)
                qᵛ⁺ = adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo, equilibrium)

                if qᵗ > qᵛ⁺  # saturated conditions
                    # Partition condensate between liquid and ice based on λ
                    q_condensate = qᵗ - qᵛ⁺
                    qˡ = λ * q_condensate
                    qⁱ = (1 - λ) * q_condensate
                    q = MoistureMassFractions(qᵛ⁺, qˡ, qⁱ)

                    # Verify partitioning sums correctly
                    @test q.vapor + q.liquid + q.ice ≈ qᵗ

                    # Compute moist static energy: e = cᵖᵐ*T + g*z - ℒˡᵣ*qˡ - ℒⁱᵣ*qⁱ
                    cᵖᵐ = mixture_heat_capacity(q, thermo)
                    e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

                    # Verify moist static energy can recover temperature
                    𝒰 = MoistStaticEnergyState(e, q, z, pᵣ)
                    T_recovered = (e - g * z + ℒˡᵣ * q.liquid + ℒⁱᵣ * q.ice) / mixture_heat_capacity(q, thermo)
                    @test T_recovered ≈ T

                    # Test saturation adjustment recovers temperature
                    𝒰_unadjusted = MoistStaticEnergyState(e, MoistureMassFractions(qᵗ), z, pᵣ)
                    T★ = compute_temperature(𝒰_unadjusted, microphysics, thermo)
                    @test T★ ≈ T atol=sqrt(tol)

                    set!(model, ρe = ρᵣ * e, qᵗ = qᵗ)
                    T★ = @allowscalar first(model.temperature)
                    qᵛm = @allowscalar first(model.microphysical_fields.qᵛ)
                    qˡm = @allowscalar first(model.microphysical_fields.qˡ)
                    qⁱm = @allowscalar first(model.microphysical_fields.qⁱ)

                    @test T★ ≈ T atol=sqrt(tol)
                    @test qᵛm ≈ qᵛ⁺
                    @test qˡm ≈ qˡ
                    @test qⁱm ≈ qⁱ
                end
            end
        end
    end

    # Test 5: Verify moist static energy formula with various moisture fractions
    @testset "Moist static energy formula verification" begin
        T = FT(253.15)  # Midway in mixed-phase range
        λ = test_liquid_fraction(T, Tᶠ, Tʰ)

        for qᵗ in FT.(5e-3:5e-3:3e-2)
            @testset let qᵗ=qᵗ
                qᵛ⁺ = adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo, equilibrium)

                if qᵗ > qᵛ⁺  # saturated conditions
                    qᶜ = qᵗ - qᵛ⁺
                    qˡ = λ * qᶜ
                    qⁱ = (1 - λ) * qᶜ
                    q = MoistureMassFractions(qᵛ⁺, qˡ, qⁱ)

                    # Compute moist static energy
                    cᵖᵐ = mixture_heat_capacity(q, thermo)
                    e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

                    # Test with saturation adjustment
                    𝒰 = MoistStaticEnergyState(e, MoistureMassFractions(qᵗ), z, pᵣ)
                    T★ = compute_temperature(𝒰, microphysics, thermo)
                    @test T★ ≈ T atol=sqrt(tol)
                end
            end
        end
    end

    # Test 6: Verify partitioning matches temperature-dependent λ
    @testset "Condensate partitioning verification" begin
        for T_partition in [FT(235), FT(245), FT(255), FT(265)]
            @testset let T_partition=T_partition
                λ_expected = test_liquid_fraction(T_partition, Tᶠ, Tʰ)

                qᵗ = FT(0.02)
                qᵛ⁺ = adjustment_saturation_specific_humidity(T_partition, pᵣ, qᵗ, thermo, equilibrium)

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
                    cᵖᵐ = mixture_heat_capacity(q, thermo)
                    e = cᵖᵐ * T_partition + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

                    𝒰 = MoistStaticEnergyState(e, MoistureMassFractions(qᵗ), z, pᵣ)
                    T★ = compute_temperature(𝒰, microphysics, thermo)
                    @test T★ ≈ T_partition atol=sqrt(tol)
                end
            end
        end
    end
end

@testset "Saturation adjustment (MoistAirBuoyancies)" for FT in (Float32, Float64)
    # Minimal grid and reference state
    # grid = RectilinearGrid(FT, size=(), topology=(Flat, Flat, Flat))
    grid = RectilinearGrid(default_arch, FT; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    thermo = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, thermo; base_pressure=101325, potential_temperature=288)

    # Sample a single cell
    pᵣ = @allowscalar reference_state.pressure[1, 1, 1]
    ρᵣ = @allowscalar reference_state.density[1, 1, 1]
    p₀ = reference_state.base_pressure
    z = FT(0.5)

    # Case 0: Absolute zero potential temperature returns zero temperature
    θ₀ = zero(FT)
    q₀ = MoistureMassFractions{FT} |> zero
    𝒰₀ = PotentialTemperatureState(θ₀, q₀, p₀, pᵣ)
    T₀ = compute_boussinesq_adjustment_temperature(𝒰₀, thermo)
    @test T₀ == 0

    # Helper for tolerances
    atol_T = FT === Float64 ? 1e-6 : FT(1e-3)

    # Case 1: Unsaturated, dry (qᵗ = 0)
    θ₁ = FT(300)
    qᵗ₁ = zero(FT)
    q₁ = MoistureMassFractions(qᵗ₁)
    𝒰₁ = PotentialTemperatureState(θ₁, q₁, p₀, pᵣ)
    Π₁ = exner_function(𝒰₁, thermo)
    T_dry₁ = Π₁ * θ₁

    T₁ = compute_boussinesq_adjustment_temperature(𝒰₁, thermo)
    @test isapprox(T₁, T_dry₁; atol=atol_T)

    # Case 2: Unsaturated, humid but below saturation at dry temperature
    θ₂ = FT(300)
    q₂ = MoistureMassFractions{FT} |> zero
    𝒰₂ = PotentialTemperatureState(θ₂, q₂, p₀, pᵣ)
    Π₂ = exner_function(𝒰₂, thermo)
    T_dry₂ = Π₂ * θ₂

    # Choose qᵗ well below saturation at T_dry₂
    ρ₂ = density(pᵣ, T_dry₂, q₂, thermo)
    qᵛ⁺₂ = saturation_specific_humidity(T_dry₂, ρ₂, thermo, thermo.liquid)
    qᵗ₂ = qᵛ⁺₂ / 2
    q₂ = MoistureMassFractions(qᵗ₂)
    𝒰₂ = with_moisture(𝒰₂, q₂)

    T₂ = compute_boussinesq_adjustment_temperature(𝒰₂, thermo)
    Π₂ = exner_function(𝒰₂, thermo)
    T_dry₂ = Π₂ * θ₂
    @test isapprox(T₂, T_dry₂; atol=atol_T)

    # Case 3: Saturated, humid (qᵗ = qᵛ⁺)
    T₃ = θ̃ = FT(300)
    qᵗ = FT(0.025)
    q̃ = MoistureMassFractions(qᵗ)
    𝒰 = PotentialTemperatureState(θ̃, q̃, p₀, pᵣ)
    qᵛ⁺ = Breeze.MoistAirBuoyancies.adjustment_saturation_specific_humidity(T₃, 𝒰, thermo)
    @test qᵗ > qᵛ⁺ # otherwise the test is wrong

    qˡ = qᵗ - qᵛ⁺
    q₃ = MoistureMassFractions(qᵛ⁺, qˡ)
    𝒰₃ = with_moisture(𝒰, q₃)
    Π₃ = exner_function(𝒰₃, thermo)
    cᵖᵐ = mixture_heat_capacity(q₃, thermo)
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    θ₃ = (T₃ - ℒˡᵣ / cᵖᵐ * qˡ) / Π₃
    𝒰₃ = PotentialTemperatureState(θ₃, q₃, p₀, pᵣ)

    T₃_solve = compute_boussinesq_adjustment_temperature(𝒰₃, thermo)
    @test isapprox(T₃_solve, T₃; atol=atol_T)
end
