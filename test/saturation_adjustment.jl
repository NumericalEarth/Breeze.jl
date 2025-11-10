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
    mixture_heat_capacity

using Breeze.MoistAirBuoyancies: compute_boussinesq_adjustment_temperature
using Breeze.Microphysics:
    WarmPhaseSaturationAdjustment,
    compute_temperature,
    adjustment_saturation_specific_humidity

@testset "Saturation adjustment (AtmosphereModel) [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    thermo = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, thermo; base_pressure=101325, potential_temperature=288)

    tol = FT(1e-3)
    microphysics = WarmPhaseSaturationAdjustment(tolerance=tol)

    # Sample a single cell
    pᵣ = @allowscalar first(reference_state.pressure)
    g = thermo.gravitational_acceleration
    z = FT(1/2)

    # First test: absolute zero
    q₀ = MoistureMassFractions(zero(FT), zero(FT), zero(FT))
    𝒰₀ = MoistStaticEnergyState(zero(FT), q₀, z, pᵣ)
    @test compute_temperature(𝒰₀, microphysics, thermo) == 0

    # Second unsaturated test: choose T, pick qᵗ well below saturation
    T₁ = FT(300)
    ρ₁ = density(pᵣ, T₁, q₀, thermo)
    qᵛ⁺ = saturation_specific_humidity(T₁, ρ₁, thermo, thermo.liquid)
    qᵗ = qᵛ⁺ / 2 # comfortably unsaturated

    q₁ = MoistureMassFractions(qᵗ, zero(FT), zero(FT))
    cᵖᵐ = mixture_heat_capacity(q₁, thermo)
    e₁ = cᵖᵐ * T₁ + g * z #  + ℒ₀ * qᵗ
    𝒰₁ = MoistStaticEnergyState(e₁, q₁, z, pᵣ)

    @test compute_temperature(𝒰₁, microphysics, thermo) ≈ T₁ atol=sqrt(tol)
    @test compute_temperature(𝒰₁, nothing, thermo) ≈ T₁ atol=sqrt(tol)

    # Many more tests that touch saturated conditions
    for T₂ in 270:4:320, qᵗ₂ in 1e-2:2e-3:5e-2
        @testset let T₂=T₂, qᵗ₂=qᵗ₂
            T₂ = convert(FT, T₂)
            qᵗ₂ = convert(FT, qᵗ₂)
            qᵛ⁺₂ = adjustment_saturation_specific_humidity(T₂, pᵣ, qᵗ₂, thermo)

            if qᵗ₂ > qᵛ⁺₂ # saturated conditions
                qˡ₂ = qᵗ₂ - qᵛ⁺₂
                q₂ = MoistureMassFractions(qᵛ⁺₂, qˡ₂, zero(FT))
                cᵖᵐ = mixture_heat_capacity(q₂, thermo)
                ℒˡᵣ = thermo.liquid.reference_latent_heat
                e₂ = cᵖᵐ * T₂ + g * z - ℒˡᵣ * qˡ₂

                𝒰₂ = MoistStaticEnergyState(e₂, q₂, z, pᵣ)
                T★ = compute_temperature(𝒰₂, microphysics, thermo)
                @test T★ ≈ T₂ atol=sqrt(tol)
            end
            #=
            else # unsaturated conditions
            q₂ = MoistureMassFractions(qᵗ₂, zero(FT), zero(FT))
            cᵖᵐ = mixture_heat_capacity(q₂, thermo)
            e₂ = cᵖᵐ * T₂ + g * z
            𝒰₂ = MoistStaticEnergyState(e₂, q₂, z, pᵣ)
            @test compute_temperature(𝒰₂, microphysics, thermo) ≈ T₂ atol=sqrt(tol)
            @test compute_temperature(𝒰₂, nothing, thermo) ≈ T₂ atol=sqrt(tol)
            end
            =#
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
    q₀ = MoistureMassFractions(zero(FT), zero(FT), zero(FT))
    𝒰₀ = PotentialTemperatureState(θ₀, q₀, z, p₀, pᵣ, ρᵣ)
    T₀ = compute_boussinesq_adjustment_temperature(𝒰₀, thermo)
    @test T₀ == 0

    # Helper for tolerances
    atol_T = FT === Float64 ? 1e-6 : FT(1e-3)

    # Case 1: Unsaturated, dry (qᵗ = 0)
    θ₁ = FT(300)
    qᵗ₁ = zero(FT)
    q₁ = MoistureMassFractions(qᵗ₁, zero(FT), zero(FT))
    𝒰₁ = PotentialTemperatureState(θ₁, q₁, z, p₀, pᵣ, ρᵣ)
    Π₁ = exner_function(𝒰₁, thermo)
    T_dry₁ = Π₁ * θ₁

    T₁ = compute_boussinesq_adjustment_temperature(𝒰₁, thermo)
    @test isapprox(T₁, T_dry₁; atol=atol_T)

    # Case 2: Unsaturated, humid but below saturation at dry temperature
    θ₂ = FT(300)
    q₂ = MoistureMassFractions(zero(FT), zero(FT), zero(FT))
    𝒰₂ = PotentialTemperatureState(θ₂, q₂, z, p₀, pᵣ, ρᵣ)
    Π₂ = exner_function(𝒰₂, thermo)
    T_dry₂ = Π₂ * θ₂

    # Choose qᵗ well below saturation at T_dry₂
    ρ₂ = density(pᵣ, T_dry₂, q₂, thermo)
    qᵛ⁺₂ = saturation_specific_humidity(T_dry₂, ρ₂, thermo, thermo.liquid)
    qᵗ₂ = qᵛ⁺₂ / 2
    q₂ = MoistureMassFractions(qᵗ₂, zero(FT), zero(FT))
    𝒰₂ = with_moisture(𝒰₂, q₂)

    T₂ = compute_boussinesq_adjustment_temperature(𝒰₂, thermo)
    Π₂ = exner_function(𝒰₂, thermo)
    T_dry₂ = Π₂ * θ₂
    @test isapprox(T₂, T_dry₂; atol=atol_T)

    # Case 3: Saturated, humid (qᵗ = qᵛ⁺)
    T₃ = θ̃ = FT(300)
    qᵗ = FT(0.025)
    q̃ = MoistureMassFractions(qᵗ, zero(FT), zero(FT))
    𝒰 = PotentialTemperatureState(θ̃, q̃, z, p₀, pᵣ, ρᵣ)
    qᵛ⁺ = Breeze.MoistAirBuoyancies.adjustment_saturation_specific_humidity(T₃, 𝒰, thermo)
    @test qᵗ > qᵛ⁺ # otherwise the test is wrong

    qˡ = qᵗ - qᵛ⁺
    q₃ = MoistureMassFractions(qᵛ⁺, qˡ, zero(FT))
    𝒰₃ = with_moisture(𝒰, q₃)
    Π₃ = exner_function(𝒰₃, thermo)
    cᵖᵐ = mixture_heat_capacity(q₃, thermo)
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    θ₃ = (T₃ - ℒˡᵣ / cᵖᵐ * qˡ) / Π₃
    𝒰₃ = PotentialTemperatureState(θ₃, q₃, z, p₀, pᵣ, ρᵣ)

    T₃_solve = compute_boussinesq_adjustment_temperature(𝒰₃, thermo)
    @test isapprox(T₃_solve, T₃; atol=atol_T)
end
