include(joinpath(@__DIR__, "setup.jl"))

using Test

using Breeze
using Breeze.AtmosphereModels: StageStart, patankar_fraction, patankar_sink_fraction,
                                stage_interval_is_known, stage_remainder
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions,
                             LiquidIcePotentialTemperatureState, PlanarLiquidSurface,
                             saturation_specific_humidity, density, temperature, with_temperature
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticlePropertiesMicrophysics,
                                                       P3MicrophysicalState,
                                                       compute_p3_process_rates

const PPP = Breeze.Microphysics.PredictedParticleProperties

#####
##### The P3 process tendencies leave every prognostic non-negative over the coming stage
#####
##### The tendency kernel is exercised on one cell at a time: random process rates, evaluated at
##### one state, are applied to a *different* stage-start state, alongside random other
##### tendencies, over stages shorter and far longer than `sink_limiting_timescale`. This is
##### what a multi-stage time stepper does, and what the rates alone cannot survive.
#####

# A small deterministic uniform generator, so a failing state can be reproduced
# without adding a test dependency.
mutable struct LinearCongruentialGenerator
    state :: UInt64
end

function uniform!(generator::LinearCongruentialGenerator)
    generator.state = generator.state * 6364136223846793005 + 1442695040888963407
    return (generator.state >> 11) / 2^53
end

const p3_density_names = (:ρqᵛ, :ρqᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ)

# A random P3 state: specific quantities in the order of `p3_density_names`,
# with its thermodynamic state and density.
function random_p3_state(rng, p3, constants, FT)
    T = FT(200 + 100 * uniform!(rng))
    p = FT(200e2 + 800e2 * uniform!(rng))
    saturation_ratio = FT(0.4 + 0.7 * uniform!(rng))
    present(probability) = uniform!(rng) < probability

    qᶜˡ = present(0.5) ? FT(10.0^(-8 + 5 * uniform!(rng))) : zero(FT)
    qʳ = present(0.6) ? FT(10.0^(-9 + 6 * uniform!(rng))) : zero(FT)
    nʳ = qʳ > 0 ? FT(10.0^(5 * uniform!(rng))) : (present(0.3) ? FT(10.0^(4 * uniform!(rng))) : zero(FT))
    qⁱ = present(0.6) ? FT(10.0^(-9 + 6 * uniform!(rng))) : zero(FT)
    nⁱ = qⁱ > 0 ? FT(10.0^(1 + 6 * uniform!(rng))) : (present(0.3) ? FT(10.0^(5 * uniform!(rng))) : zero(FT))
    qʷⁱ = (qⁱ > 0) & present(0.5) ? qⁱ * FT(uniform!(rng)) : zero(FT)
    qᶠ = (qⁱ > 0) & present(0.5) ? qⁱ * FT(uniform!(rng)) : zero(FT)
    bᶠ = qᶠ / FT(100 + 800 * uniform!(rng))

    # Iterate the vapor fraction to the target saturation ratio: density depends on moisture.
    thermodynamic_state(qᵛ) = with_temperature(
        LiquidIcePotentialTemperatureState(zero(FT), MoistureMassFractions(qᵛ, qᶜˡ + qʳ + qʷⁱ, qⁱ),
                                           FT(1e5), p), T, constants)
    qᵛ = FT(1e-6)
    for _ in 1:10
        ρ = density(thermodynamic_state(qᵛ), constants)
        qᵛ = saturation_ratio * saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    end

    𝒰 = thermodynamic_state(qᵛ)
    ρ = density(𝒰, constants)
    z = zero(FT)
    ℳ = P3MicrophysicalState(qᶜˡ, FT(p3.cloud.number_concentration), qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ, z, z, z)
    specific = (qᵛ, qᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ)

    return ρ, 𝒰, ℳ, specific
end

one_cell_fields(values) = NamedTuple{p3_density_names}(map(v -> fill(v, 1, 1, 1), values))

@testset "P3 process tendencies keep every prognostic non-negative [$FT]" for FT in (Float32, Float64)
    p3 = PredictedParticlePropertiesMicrophysics(FT; negative_moisture_correction = nothing)
    constants = ThermodynamicConstants(FT)
    rng = LinearCongruentialGenerator(1234)
    roundoff = 100 * eps(FT)

    @testset "Δt = $Δt s" for Δt in (5, 11.25, 45, 200)
        Δt = FT(Δt)
        negative = zeros(Int, length(p3_density_names))
        water_imbalance = zero(FT)

        for _ in 1:2000
            ρ, 𝒰, ℳ, specific = random_p3_state(rng, p3, constants, FT)
            properties = PPP.p3_process_properties(p3, ρ, ℳ)
            rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties,
                                             temperature(𝒰, constants), zero(FT), zero(FT))

            # The stage starts from a state unrelated to the one the rates saw (empty for
            # a third of the species), and every other tendency removes part of it or adds to it.
            c⁰ = map(q -> uniform!(rng) < 0.3 ? zero(FT) : ρ * q * FT(2 * uniform!(rng)), specific)
            Gᵒ = map((c, q) -> (-FT(uniform!(rng)) * c + (uniform!(rng) < 0.5 ? FT(uniform!(rng)) * ρ * q : zero(FT))) / Δt,
                     c⁰, specific)
            c̃ = c⁰ .+ Δt .* Gᵒ

            fields = one_cell_fields(c⁰)
            G = one_cell_fields(Gᵒ)
            stage = StageStart(fields, fields, false, Δt)

            φ = PPP.p3_sink_fractions(1, 1, 1, G, stage, p3, ρ, rates)
            limited_rates = PPP.patankar_limited_rates(rates, φ)
            result = PPP.p3_tendencies(p3, ρ, ℳ, properties, limited_rates)
            relaxations = PPP.p3_number_relaxations(1, 1, 1, G, stage, p3, ρ, ℳ, properties, limited_rates)
            rime = PPP.p3_rime_limits(1, 1, 1, G, stage, result)

            Gᵖ = (result.tendency_ρqᵛ, result.tendency_ρqᶜˡ, result.tendency_ρqʳ,
                  result.tendency_ρnʳ + relaxations.rain, result.tendency_ρqⁱ,
                  result.tendency_ρnⁱ + relaxations.ice, result.tendency_ρqᶠ + rime.mass,
                  result.tendency_ρbᶠ + rime.volume, result.tendency_ρqʷⁱ)

            c⁺ = c̃ .+ Δt .* Gᵖ
            tolerance = roundoff .* (abs.(c̃) .+ ρ .* abs.(specific) .+ Δt .* abs.(Gᵖ))
            negative .+= c⁺ .< .-tolerance

            # Vapor, cloud, rain, ice and the liquid on ice exchange mass among themselves only.
            water_tendencies = (Gᵖ[1], Gᵖ[2], Gᵖ[3], Gᵖ[5], Gᵖ[9])
            scale = sum(abs, water_tendencies)
            water_imbalance = max(water_imbalance, abs(sum(water_tendencies)) / max(scale, floatmin(FT)))
        end

        @test all(negative .== 0)
        @test water_imbalance < roundoff
    end
end

@testset "Patankar sink fraction [$FT]" for FT in (Float32, Float64)
    fields = (; c = fill(FT(2), 1, 1, 1))
    G = fill(FT(-0.1), 1, 1, 1)
    Δt = FT(10)
    stage = StageStart(fields, fields, false, Δt)

    # c̃ = 2 - 10 × 0.1 = 1; a loss of 0.3 over 10 s asks for three times that.
    @test stage_interval_is_known(stage)
    @test stage_remainder(1, 1, 1, G, stage, Val(:c)) ≈ 1
    @test patankar_fraction(FT(1), Δt, FT(0.3)) ≈ FT(1 / 4)
    @test patankar_fraction(FT(-1), Δt, FT(0.3)) == 0
    @test patankar_fraction(FT(1), Δt, FT(-0.3)) == 1
    φ = patankar_sink_fraction(1, 1, 1, G, stage, Val(:c), FT(0.3))
    @test φ ≈ FT(1 / 4)
    @test 1 - Δt * φ * FT(0.3) ≈ FT(1 / 4)   # c⁺ = c̃ / (1 + Δt L / c̃)

    # No loss, no limit; nothing left, no sink.
    @test patankar_sink_fraction(1, 1, 1, G, stage, Val(:c), zero(FT)) == 1
    @test patankar_sink_fraction(1, 1, 1, fill(FT(-1), 1, 1, 1), stage, Val(:c), FT(0.3)) == 0

    # The stage interval is not known before the first time step.
    unknown = StageStart(fields, fields, false, FT(Inf))
    @test !stage_interval_is_known(unknown)
    @test patankar_sink_fraction(1, 1, 1, G, unknown, Val(:c), FT(0.3)) == 1

    # A stage that starts from the step-start state budgets against that state.
    initial = (; c = fill(FT(4), 1, 1, 1))
    from_initial = StageStart(initial, fields, true, Δt)
    @test stage_remainder(1, 1, 1, G, from_initial, Val(:c)) ≈ 3
    @test patankar_sink_fraction(1, 1, 1, G, from_initial, Val(:c), FT(0.3)) ≈ FT(1 / 2)
end
