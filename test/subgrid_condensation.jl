include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudIce, CloudLiquid
using Enzyme
using Oceananigans
using Test

using Breeze.AtmosphereModels: DiracStatistics, GaussianStatistics, GaussianSubgridCorrelation,
                               NoSubgridCorrelation, subgrid_correlation
using Breeze.Microphysics: gaussian_cloud_fraction, gaussian_equilibrium_condensate,
                           normal_cumulative_distribution, condensation_rate,
                           thermodynamic_adjustment_factor
using Breeze.Thermodynamics: MoistureMassFractions, LiquidIcePotentialTemperatureState, density,
                             with_temperature, saturation_specific_humidity, PlanarLiquidSurface

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, MixedPhaseOneMomentState, mpne1m_tendencies

G(Q) = gaussian_equilibrium_condensate(Q, one(Q))

@testset "Gaussian saturation-deficit closed forms" begin
    @test abs(normal_cumulative_distribution(0.0) - 0.5) < 1e-7
    @test abs(normal_cumulative_distribution(1.0) - 0.8413447460685429) < 1e-7
    @test abs(normal_cumulative_distribution(-3.0) - 0.0013498980316301) < 1e-8
    @test abs(gaussian_cloud_fraction(0.0, 1e-3) - 0.5) < 1e-7

    # Limits: G(Q) → 0 as Q → -∞ and G(Q) → Q as Q → +∞
    @test G(-6.0) < 1e-9
    @test abs(G(6.0) - 6) < 1e-6

    # Dirac limit: σₛ G(s̄ / σₛ) → max(0, s̄)
    for s̄ in (-1e-3, 0.0, 2e-3)
        @test abs(gaussian_equilibrium_condensate(s̄, 1e-9) - max(0, s̄)) < 1e-9
    end

    # dG/dQ = Φ(Q): finite differences and Enzyme
    for Q in (-2.0, -0.5, 0.0, 0.7, 3.0)
        δ = 1e-5
        fd = (G(Q + δ) - G(Q - δ)) / 2δ
        @test abs(fd - normal_cumulative_distribution(Q)) < 1e-6
        ad = first(first(Enzyme.autodiff(Enzyme.Reverse, G, Enzyme.Active, Enzyme.Active(Q))))
        @test abs(ad - fd) < 1e-6
    end
end

function warm_state(FT, constants; T, relative_humidity, qᶜˡ, p = 90000)
    q₀ = MoistureMassFractions(FT(0.01), FT(qᶜˡ), zero(FT))
    𝒰₀ = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q₀, FT(1e5), FT(p)), FT(T), constants)
    qᵛ⁺ = saturation_specific_humidity(FT(T), density(𝒰₀, constants), constants, PlanarLiquidSurface())
    q = MoistureMassFractions(FT(relative_humidity) * qᵛ⁺, FT(qᶜˡ), zero(FT))
    𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(p)), FT(T), constants)
    ρ = density(𝒰, constants)
    return ρ, MixedPhaseOneMomentState(FT(qᶜˡ), zero(FT), zero(FT), zero(FT)), 𝒰
end

non_equilibrium_1M(FT; subgrid = NoSubgridCorrelation()) =
    OneMomentCloudMicrophysics(FT; cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT)), subgrid)

@testset "Gaussian condensation in MPNE1M [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    dirac = non_equilibrium_1M(FT)
    gaussian = non_equilibrium_1M(FT; subgrid = GaussianSubgridCorrelation(FT; minimum_saturation_deficit_spread = 1e-9))
    @test subgrid_correlation(dirac) isa NoSubgridCorrelation
    @test subgrid_correlation(gaussian) isa GaussianSubgridCorrelation

    # The Dirac statistics are the grid-mean rates, bitwise
    for relative_humidity in (0.9, 1.0, 1.02), qᶜˡ in (0, 1e-4)
        ρ, ℳ, 𝒰 = warm_state(FT, constants; T = 285, relative_humidity, qᶜˡ)
        @test mpne1m_tendencies(dirac, ρ, ℳ, 𝒰, constants) ===
              mpne1m_tendencies(dirac, ρ, ℳ, 𝒰, constants, DiracStatistics{FT}())
    end

    # Zero variance and a negligible floor: the Gaussian rate is the grid-mean rate where saturated
    ρ, ℳ, 𝒰 = warm_state(FT, constants; T = 285, relative_humidity = 1.01, qᶜˡ = 1e-4)
    Gᴰ = mpne1m_tendencies(dirac, ρ, ℳ, 𝒰, constants)
    Gᴳ = mpne1m_tendencies(gaussian, ρ, ℳ, 𝒰, constants, GaussianStatistics{FT}())
    @test isapprox(Gᴳ.ρqᶜˡ, Gᴰ.ρqᶜˡ; rtol = FT == Float64 ? 1e-3 : 1e-2)

    # ... and the grid-mean evaporation rate where subsaturated
    ρ, ℳ, 𝒰 = warm_state(FT, constants; T = 285, relative_humidity = 0.95, qᶜˡ = 1e-4)
    Gᴰ = mpne1m_tendencies(dirac, ρ, ℳ, 𝒰, constants)
    Gᴳ = mpne1m_tendencies(gaussian, ρ, ℳ, 𝒰, constants, GaussianStatistics{FT}())
    @test isapprox(Gᴳ.ρqᶜˡ, Gᴰ.ρqᶜˡ; rtol = FT == Float64 ? 1e-3 : 1e-2)

    # Subsaturated cell (RH 0.92) holding cloud liquid: the grid-mean rate evaporates it all,
    # a Gaussian spread σₛ ≈ 1 g/kg holds an equilibrium liquid
    ρ, ℳ, 𝒰 = warm_state(FT, constants; T = 285, relative_humidity = 0.92, qᶜˡ = 5e-5)
    σ²ᵗ = FT(1e-3)^2
    statistics = GaussianStatistics(σ²ᵗ, zero(FT), zero(FT))
    q = 𝒰.moisture_mass_fractions
    τ = 1 / gaussian.cloud_formation.liquid.rate
    T = FT(285)
    qᵛ⁺ = saturation_specific_humidity(T, density(𝒰, constants), constants, PlanarLiquidSurface())
    Γ = thermodynamic_adjustment_factor(qᵛ⁺, T, q, constants)
    Sᴰ = condensation_rate(q.vapor, qᵛ⁺, ℳ.qᶜˡ, T, ρ, q, τ, constants)
    Sᴳ = condensation_rate(q.vapor, qᵛ⁺, ℳ.qᶜˡ, T, ρ, q, τ, constants, 𝒰, statistics, gaussian.subgrid)
    @test Sᴰ ≈ -ℳ.qᶜˡ / (Γ * τ)
    s̄ = (q.vapor - qᵛ⁺) / Γ + ℳ.qᶜˡ
    qˡ⁺ = gaussian_equilibrium_condensate(s̄, FT(1e-3) / Γ)
    @test qˡ⁺ > 0
    C = gaussian_cloud_fraction(s̄, FT(1e-3) / Γ)
    @test Sᴳ ≈ (qˡ⁺ - ℳ.qᶜˡ) / (τ * (C + (1 - C) * Γ))

    # At its equilibrium liquid (same s̄: the extra liquid taken from the vapor, times Γ) the Gaussian
    # rate vanishes, while the grid-mean rate still evaporates
    relative_humidity = FT(0.92) - Γ * (qˡ⁺ - FT(5e-5)) / qᵛ⁺
    ρ, ℳ, 𝒰 = warm_state(FT, constants; T = 285, relative_humidity, qᶜˡ = qˡ⁺)
    q = 𝒰.moisture_mass_fractions
    Sᴳ = condensation_rate(q.vapor, qᵛ⁺, ℳ.qᶜˡ, T, ρ, q, τ, constants, 𝒰, statistics, gaussian.subgrid)
    Sᴰ = condensation_rate(q.vapor, qᵛ⁺, ℳ.qᶜˡ, T, ρ, q, τ, constants)
    @test abs(Sᴳ) < 1e-2 * abs(Sᴰ)

    # Conservation is untouched
    G = mpne1m_tendencies(gaussian, ρ, ℳ, 𝒰, constants, statistics)
    @test abs(G.ρqᵛ + G.ρqᶜˡ + G.ρqᶜⁱ + G.ρqʳ + G.ρqˢⁿ) ≤ 10eps(FT) * maximum(abs, values(G))
end

@testset "Single column with TKE closure and Gaussian subgrid condensation [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = 20, z = (0, 2000), topology = (Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 290)
    dynamics = AnelasticDynamics(reference_state)
    closure = TKEBasedTurbulenceClosure()

    function column(subgrid)
        microphysics = non_equilibrium_1M(FT; subgrid)
        model = AtmosphereModel(grid; dynamics, microphysics, closure)
        θ₀(z) = 290 + 5 * max(0, z - 1000) / 1000
        qᵗ₀(z) = z < 1000 ? FT(0.010) : FT(0.004)
        set!(model; θ = θ₀, qᵗ = qᵗ₀, ρe = 0.1, u = 1)
        for _ in 1:3
            time_step!(model, 10)
        end
        return model
    end

    dirac = column(NoSubgridCorrelation())
    gaussian = column(GaussianSubgridCorrelation(FT))
    @test all(isfinite, interior(gaussian.microphysical_fields.ρqᶜˡ))
    @test all(≥(0), interior(gaussian.microphysical_fields.ρqᶜˡ))
    # The background spread keeps liquid in the near-saturated mixed layer where the grid-mean scheme has none
    @test sum(interior(gaussian.microphysical_fields.ρqᶜˡ)) > sum(interior(dirac.microphysical_fields.ρqᶜˡ))
end
