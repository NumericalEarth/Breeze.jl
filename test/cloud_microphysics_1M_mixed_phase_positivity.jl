include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudIce, CloudLiquid
using Oceananigans
using Test

using Breeze.Thermodynamics:
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState,
    density,
    with_temperature

using Breeze.AtmosphereModels: microphysical_tendency
using Breeze.Thermodynamics: temperature

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, MixedPhaseOneMomentState,
                                   mp1m_tendencies, mpne1m_tendencies, mixed_phase_precipitation_rates,
                                   donor_limiter, τⁿᵘᵐ

function mixed_phase_state(FT, constants; T, qᵛ, qᶜˡ, qᶜⁱ, qʳ, qˢⁿ, p = 90000)
    q = MoistureMassFractions(FT(qᵛ), FT(qᶜˡ + qʳ), FT(qᶜⁱ + qˢⁿ))
    𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(p)), FT(T), constants)
    ρ = density(𝒰, constants)
    ℳ = MixedPhaseOneMomentState(FT(qᶜˡ), FT(qᶜⁱ), FT(qʳ), FT(qˢⁿ))
    return ρ, ℳ, 𝒰
end

saturation_adjustment_1M(FT) =
    OneMomentCloudMicrophysics(FT; cloud_formation = SaturationAdjustment(FT; equilibrium = MixedPhaseEquilibrium(FT)))

non_equilibrium_1M(FT) =
    OneMomentCloudMicrophysics(FT; cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT)))

@testset "Saturation-adjustment mixed-phase 1M makes snow [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    microphysics = saturation_adjustment_1M(FT)
    ρ, ℳ, 𝒰 = mixed_phase_state(FT, constants; T = 258, qᵛ = 1.5e-3, qᶜˡ = 0, qᶜⁱ = 1e-3, qʳ = 0, qˢⁿ = 1e-4)

    G = mp1m_tendencies(microphysics, ρ, ℳ, 𝒰, constants)
    @test G.ρqˢⁿ > 0
    @test microphysical_tendency(microphysics, Val(:ρqˢⁿ), ρ, ℳ, 𝒰, constants) == G.ρqˢⁿ
    @test microphysical_tendency(microphysics, Val(:ρqʳ), ρ, ℳ, 𝒰, constants) == G.ρqʳ
    @test microphysical_tendency(microphysics, Val(:ρqᵉ), ρ, ℳ, 𝒰, constants) == G.ρqᵉ

    # Above freezing, snow melts into rain
    ρ, ℳ, 𝒰 = mixed_phase_state(FT, constants; T = 280, qᵛ = 6e-3, qᶜˡ = 0, qᶜⁱ = 0, qʳ = 1e-5, qˢⁿ = 1e-3)
    G = mp1m_tendencies(microphysics, ρ, ℳ, 𝒰, constants)
    @test G.ρqˢⁿ < 0
    @test G.ρqʳ > 0
end

@testset "Mixed-phase 1M conserves water and stays non-negative over Δt = τⁿᵘᵐ [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    sa = saturation_adjustment_1M(FT)
    ne = non_equilibrium_1M(FT)
    Δt = FT(τⁿᵘᵐ)

    for T in 250:4:286, qᶜˡ in (0, 1e-3), qᶜⁱ in (0, 1e-3), qʳ in (0, 1e-6, 1e-4, 1e-3), qˢⁿ in (0, 1e-4, 8e-3), qᵛ in (2e-3, 6e-3)
        ρ, ℳ, 𝒰 = mixed_phase_state(FT, constants; T, qᵛ, qᶜˡ, qᶜⁱ, qʳ, qˢⁿ)

        G = mp1m_tendencies(sa, ρ, ℳ, 𝒰, constants)
        scale = max(abs(G.ρqᵉ), abs(G.ρqʳ), abs(G.ρqˢⁿ), floatmin(FT))
        @test abs(G.ρqᵉ + G.ρqʳ + G.ρqˢⁿ) ≤ 10eps(FT) * scale
        @test ρ * FT(qʳ)  + Δt * G.ρqʳ  ≥ -10eps(FT) * ρ * FT(qʳ + qˢⁿ + qᶜˡ + qᶜⁱ + qᵛ)
        @test ρ * FT(qˢⁿ) + Δt * G.ρqˢⁿ ≥ -10eps(FT) * ρ * FT(qʳ + qˢⁿ + qᶜˡ + qᶜⁱ + qᵛ)

        G = mpne1m_tendencies(ne, ρ, ℳ, 𝒰, constants)
        Σ = G.ρqᵛ + G.ρqᶜˡ + G.ρqᶜⁱ + G.ρqʳ + G.ρqˢⁿ
        scale = max(maximum(abs, values(G)), floatmin(FT))
        @test abs(Σ) ≤ 10eps(FT) * scale
        total = ρ * FT(qʳ + qˢⁿ + qᶜˡ + qᶜⁱ + qᵛ)
        for (name, content) in ((:ρqᶜˡ, qᶜˡ), (:ρqᶜⁱ, qᶜⁱ), (:ρqʳ, qʳ), (:ρqˢⁿ, qˢⁿ))
            @test ρ * FT(content) + Δt * G[name] ≥ -10eps(FT) * total
        end
    end
end

@testset "Donor limiter caps rain–snow collection [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    Δt = FT(τⁿᵘᵐ)

    # Abundant snow just below freezing: rain is collected on a sub-second timescale.
    ρ, ℳ, 𝒰 = mixed_phase_state(FT, constants; T = 270, qᵛ = 3.5e-3, qᶜˡ = 0, qᶜⁱ = 0, qʳ = 1e-4, qˢⁿ = 8e-3)
    for microphysics in (saturation_adjustment_1M(FT), non_equilibrium_1M(FT))
        rates = mixed_phase_precipitation_rates(microphysics, ρ, ℳ, 𝒰.moisture_mass_fractions,
                                                temperature(𝒰, constants),
                                                microphysics.categories.freezing_temperature, constants)
        @test Δt * rates.Sʳˢⁿ > ℳ.qʳ   # the unlimited rate would overdraw rain within one step
        G = microphysics isa BreezeCloudMicrophysicsExt.MP1M ? mp1m_tendencies(microphysics, ρ, ℳ, 𝒰, constants) :
                                                               mpne1m_tendencies(microphysics, ρ, ℳ, 𝒰, constants)
        @test ρ * ℳ.qʳ + Δt * G.ρqʳ ≥ -10eps(FT) * ρ * ℳ.qˢⁿ
        @test G.ρqˢⁿ > 0
    end

    @test donor_limiter(FT(1e-3), FT(1e-6)) == 1                       # demand fits: untouched
    @test donor_limiter(FT(1e-4), FT(1e-4)) ≈ FT(1e-4) / (τⁿᵘᵐ * FT(1e-4))
    @test donor_limiter(FT(-1e-6), FT(1e-4)) == 0                      # empty donor gives nothing
    @test donor_limiter(zero(FT), zero(FT)) == 1
end

@testset "Mixed-phase 1M models make snow and keep rain non-negative [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (2, 2, 4), x = (0, 1_000), y = (0, 1_000), z = (0, 1_000))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, base_pressure = 101325, potential_temperature = 255)
    dynamics = AnelasticDynamics(reference_state)

    for microphysics in (saturation_adjustment_1M(FT), non_equilibrium_1M(FT))
        model = AtmosphereModel(grid; dynamics, microphysics)
        set!(model; θ = 255, qᵗ = 0.004)
        for _ in 1:30
            time_step!(model, 10)
        end
        ρqˢⁿ = Array(interior(model.microphysical_fields.ρqˢⁿ))
        ρqʳ = Array(interior(model.microphysical_fields.ρqʳ))
        @test all(isfinite, ρqˢⁿ)
        @test maximum(ρqˢⁿ) > 0
        @test minimum(ρqʳ) ≥ -sqrt(eps(FT))
    end
end
