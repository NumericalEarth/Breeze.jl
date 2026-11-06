include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: microphysical_tendency
using Breeze.Microphysics: thermodynamic_adjustment_factor, ice_thermodynamic_adjustment_factor
using Breeze.Solvers: NewtonSolver
using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, LiquidIceDensityState,
                             StaticEnergyState, MoistureMassFractions, PlanarLiquidSurface,
                             PlanarIceSurface, density, temperature, with_moisture, with_temperature,
                             saturation_specific_humidity
using CloudMicrophysics
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, TwoMomentCloudMicrophysics,
                                   WarmPhaseOneMomentState, MixedPhaseOneMomentState,
                                   WarmPhaseTwoMomentState, two_moment_cloud_microphysics_categories

@testset "Cloud relaxation follows the thermodynamic constraint [$FT]" for FT in all_float_types()
    constants = ThermodynamicConstants(FT)
    τ = FT(10)
    active = ConstantRateCondensateFormation(inv(τ))
    inactive = ConstantRateCondensateFormation(zero(FT))
    warm_state = WarmPhaseOneMomentState(zero(FT), zero(FT), zero(FT))
    mixed_state = MixedPhaseOneMomentState(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))
    two_moment_state = WarmPhaseTwoMomentState(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT),
                                               (; u=zero(FT), v=zero(FT), w=zero(FT)))

    # `(wˡ, wⁱ)` routes condensate formed from vapor into liquid or ice.
    for (surface, phase, (wˡ, wⁱ), T, qᵛ) in
        ((PlanarLiquidSurface(), Val(:ρqᶜˡ), (1, 0), FT(300), FT(0.025)),
         (PlanarIceSurface(), Val(:ρqᶜⁱ), (0, 1), FT(260), FT(0.003)))

        # No initial condensate or precipitation, and only one active phase-change process.
        if surface isa PlanarLiquidSurface
            cloud_formation = NonEquilibriumCloudFormation(active)
            warm = OneMomentCloudMicrophysics(FT; cloud_formation)
            mixed = OneMomentCloudMicrophysics(FT; cloud_formation=NonEquilibriumCloudFormation(active, inactive))
            categories = two_moment_cloud_microphysics_categories(FT; aerosol_activation=nothing)
            two_moment = TwoMomentCloudMicrophysics(FT; cloud_formation, categories)
            cases = (("warm 1M", warm, warm_state), ("mixed 1M liquid", mixed, mixed_state),
                     ("warm 2M", two_moment, two_moment_state))
        else
            mixed = OneMomentCloudMicrophysics(FT; cloud_formation=NonEquilibriumCloudFormation(inactive, active))
            cases = (("mixed 1M ice", mixed, mixed_state),)
        end

        q = MoistureMassFractions(qᵛ, zero(FT), zero(FT))
        formed(δq) = MoistureMassFractions(qᵛ - δq, wˡ * δq, wⁱ * δq)
        𝒰ᵖ = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(1e5)), T, constants)
        𝒰ˢ = with_temperature(StaticEnergyState(zero(FT), q, zero(FT), FT(1e5)), T, constants)
        𝒰ᵈ = with_temperature(LiquidIceDensityState(zero(FT), q, FT(1e5), density(𝒰ᵖ, constants)), T, constants)

        @testset "$name / $(nameof(typeof(𝒰)))" for (name, microphysics, ℳ) in cases, 𝒰 in (𝒰ᵖ, 𝒰ˢ, 𝒰ᵈ)
            qᵛ⁺ = saturation_specific_humidity(𝒰, constants, surface)

            # Γ = 1 + dqᵛ⁺/dqᶜ relaxes the supersaturation over τ. Forming condensate at the actual
            # state's conserved variables selects fixed pressure or fixed density, and includes
            # the latent heating and, at fixed pressure, the drop in the mixture gas constant.
            δ = FT === Float64 ? FT(1e-6) : FT(5e-4)
            𝒰ᶜ⁺ = with_moisture(𝒰, formed(δ))
            𝒰ᶜ⁻ = with_moisture(𝒰, formed(-δ))
            dqᵛ⁺_dqᶜ = (saturation_specific_humidity(𝒰ᶜ⁺, constants, surface) -
                        saturation_specific_humidity(𝒰ᶜ⁻, constants, surface)) / 2δ
            Γ = 1 + dqᵛ⁺_dqᶜ

            expected = (qᵛ - qᵛ⁺) / (Γ * τ)
            ρ = density(𝒰, constants)
            rate = @inferred microphysical_tendency(microphysics, phase, ρ, ℳ, 𝒰, constants)
            vapor = @inferred microphysical_tendency(microphysics, Val(:ρqᵛ), ρ, ℳ, 𝒰, constants)

            @test expected > 0
            @test rate / ρ ≈ expected rtol=(FT === Float64 ? 1e-7 : 2e-3)
            @test vapor ≈ -rate
        end
    end
end

@testset "Relaxation with existing condensate [$FT]" for FT in all_float_types()
    constants = ThermodynamicConstants(FT)
    δ = FT === Float64 ? FT(1e-6) : FT(1e-4)
    tolerance = FT === Float64 ? 1e-7 : 5e-4

    for (surface, adjustment_factor, (wˡ, wⁱ), T, qᵛ) in
        ((PlanarLiquidSurface(), thermodynamic_adjustment_factor, (1, 0), FT(300), FT(0.025)),
         (PlanarIceSurface(), ice_thermodynamic_adjustment_factor, (0, 1), FT(260), FT(0.003)))

        # The density state with the converged and the non-iterated temperature inversion, and the
        # reference-pressure state.
        states(q, κ̂) = (("Newton density", LiquidIceDensityState(FT(0), q, FT(1e5), FT(1.2), NewtonSolver(FT), κ̂)),
                        ("closed-form density", LiquidIceDensityState(FT(0), q, FT(1e5), FT(1.2), nothing, κ̂)),
                        ("reference pressure", LiquidIcePotentialTemperatureState(FT(0), q, FT(1e5), FT(1e5), κ̂)))

        @testset "$(typeof(surface)), $(typeof(κ̂)), $label, qˡ=$qˡ, qⁱ=$qⁱ" for
            κ̂ in (VaporEquivalentExponent(), DryExponent()),
            (qˡ, qⁱ) in ((FT(0), FT(0)), (FT(0.01), FT(0)), (FT(0), FT(0.01)), (FT(0.005), FT(0.005))),
            (label, 𝒰₀) in states(MoistureMassFractions(qᵛ, qˡ, qⁱ), κ̂)

            𝒰 = with_temperature(𝒰₀, T, constants)
            T₀ = temperature(𝒰, constants)
            qᵛ⁺ = saturation_specific_humidity(𝒰, constants, surface)

            # Form condensate at the state's conserved variables and fixed total water. This
            # differentiates the full saturation response, including the latent heating, the
            # temperature solver, and the drop in the mixture gas constant at fixed pressure.
            q⁺ = MoistureMassFractions(qᵛ - δ, qˡ + wˡ * δ, qⁱ + wⁱ * δ)
            q⁻ = MoistureMassFractions(qᵛ + δ, qˡ - wˡ * δ, qⁱ - wⁱ * δ)
            𝒰⁺ = with_moisture(𝒰, q⁺)
            𝒰⁻ = with_moisture(𝒰, q⁻)
            dqᵛ⁺_dqᶜ = (saturation_specific_humidity(𝒰⁺, constants, surface) -
                        saturation_specific_humidity(𝒰⁻, constants, surface)) / 2δ
            Γ = @inferred adjustment_factor(qᵛ⁺, T₀, 𝒰, constants)

            @test Γ ≈ 1 + dqᵛ⁺_dqᶜ rtol=tolerance
        end
    end
end
