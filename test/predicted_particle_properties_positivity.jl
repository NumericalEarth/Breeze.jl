include(joinpath(@__DIR__, "setup.jl"))

using Test

using Breeze
using Breeze.AtmosphereModels: StageStart, patankar_fraction, patankar_sink_fraction,
                                sequential_sink_fractions, stage_interval_is_known, stage_remainder
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions,
                             LiquidIcePotentialTemperatureState, PlanarLiquidSurface,
                             saturation_specific_humidity, density, temperature, with_temperature,
                             air_pressure
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

            # The stage starts from a state unrelated to the one the rates saw (empty for
            # a third of the species), and every other tendency removes part of it or adds to it.
            c⁰ = map(q -> uniform!(rng) < 0.3 ? zero(FT) : ρ * q * FT(2 * uniform!(rng)), specific)
            Gᵒ = map((c, q) -> (-FT(uniform!(rng)) * c + (uniform!(rng) < 0.5 ? FT(uniform!(rng)) * ρ * q : zero(FT))) / Δt,
                     c⁰, specific)
            c̃ = c⁰ .+ Δt .* Gᵒ

            fields = one_cell_fields(c⁰)
            G = one_cell_fields(Gᵒ)
            stage = StageStart(fields, fields, false, Δt)

            # As in the tendency kernel, the liquid vapor exchange is relaxed implicitly over the stage.
            rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties,
                                             temperature(𝒰, constants), zero(FT), zero(FT),
                                             nothing, PPP.GridLiquidRelaxationStage(1, 1, 1, G, stage))

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

@testset "Sequential sink fractions [$FT]" for FT in (Float32, Float64)
    fields = (; c = fill(FT(2), 1, 1, 1))
    G = fill(FT(-0.1), 1, 1, 1)
    Δt = FT(10)
    stage = StageStart(fields, fields, false, Δt)

    # c̃ = 1. A first draw of 0.6 fits and is taken in full; a loss of 0.1 over 10 s then asks
    # for 1 of the 0.4 that is left.
    κ, φ = sequential_sink_fractions(1, 1, 1, G, stage, Val(:c), FT(0.06), FT(0.1))
    @test κ == 1
    @test φ ≈ FT(0.4) / FT(1.4)
    @test Δt * (κ * FT(0.06) + φ * FT(0.1)) < 1

    # A first draw that asks for more than there is takes all of it and leaves the others nothing.
    κ, φ = sequential_sink_fractions(1, 1, 1, G, stage, Val(:c), FT(0.4), FT(0.1))
    @test κ ≈ FT(1 / 4)
    @test φ == 0

    # No first draw: the Patankar fraction. Nothing left: no sink. Unknown interval: no limit.
    @test sequential_sink_fractions(1, 1, 1, G, stage, Val(:c), zero(FT), FT(0.3)) ==
          (1, patankar_sink_fraction(1, 1, 1, G, stage, Val(:c), FT(0.3)))
    @test sequential_sink_fractions(1, 1, 1, fill(FT(-1), 1, 1, 1), stage, Val(:c), FT(0.4), FT(0.1)) == (0, 0)
    unknown = StageStart(fields, fields, false, FT(Inf))
    @test sequential_sink_fractions(1, 1, 1, G, unknown, Val(:c), FT(0.4), FT(0.1)) == (1, 1)
    @test sequential_sink_fractions(1, 1, 1, G, unknown, Val(:c), zero(FT), zero(FT)) == (1, 1)
end

#####
##### The implicitly relaxed transfers are not limited a second time
#####
##### Cloud evaporation and droplet activation are advanced by backward Euler over the stage.
##### Over a stage far longer than their time scale, each removes from its donor what backward
##### Euler prescribes, with no further factor from the sink fractions.
#####

const p3_aerosol_density_names = (p3_density_names..., :ρnᶜˡ, :ρnᵃ)
one_cell_aerosol_fields(values) = NamedTuple{p3_aerosol_density_names}(map(v -> fill(v, 1, 1, 1), values))

@testset "Implicit P3 transfers follow backward Euler over a long stage [$FT]" for FT in (Float32, Float64)
    constants = ThermodynamicConstants(FT)
    T, p = FT(285), FT(9e4)
    z = zero(FT)
    thermodynamic_state(qᵛ, qᶜˡ) = with_temperature(
        LiquidIcePotentialTemperatureState(z, MoistureMassFractions(qᵛ, qᶜˡ, z), FT(1e5), p), T, constants)
    saturation(qᶜˡ) = saturation_specific_humidity(T, density(thermodynamic_state(FT(1e-2), qᶜˡ), constants),
                                                   constants, PlanarLiquidSurface())

    @testset "Cloud evaporation into subsaturated air" begin
        p3 = PredictedParticlePropertiesMicrophysics(FT; negative_moisture_correction = nothing)
        qᶜˡ = FT(3e-4)
        qᵛ = saturation(qᶜˡ) / 2
        𝒰 = thermodynamic_state(qᵛ, qᶜˡ)
        ρ = density(𝒰, constants)
        nᶜˡ = FT(p3.cloud.number_concentration)
        ℳ = P3MicrophysicalState(qᶜˡ, nᶜˡ, ntuple(_ -> z, 10)...)
        properties = PPP.p3_process_properties(p3, ρ, ℳ)

        # The droplets' vapor relaxation rate
        transport = PPP.air_transport_properties(T, air_pressure(𝒰, constants), constants)
        cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
        εˡ = PPP.cloud_vapor_relaxation_coefficient(p3, qᶜˡ, ρ, transport.Dᵛ, cloud.μᶜˡ, cloud.λᶜˡ, cloud.nᶜˡ)

        # The stage starts from the state the rates see, and nothing else acts on the cloud.
        c⁰ = (ρ * qᵛ, ρ * qᶜˡ, z, z, z, z, z, z, z)
        fields = one_cell_fields(c⁰)
        G = one_cell_fields(map(zero, c⁰))

        # Stages shorter and far longer than `sink_limiting_timescale` (10 s) and than 1 / εˡ
        @test FT(0.1) < εˡ * FT(0.5) < 1
        for Δt in FT.((0.5, 2, 50, 200, 2000))
            long_stage = εˡ * Δt > 10
            @test long_stage == (Δt ≥ 50)
            stage = StageStart(fields, fields, false, Δt)
            rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties, T, z, z,
                                             nothing, PPP.GridLiquidRelaxationStage(1, 1, 1, G, stage))
            φ = PPP.p3_sink_fractions(1, 1, 1, G, stage, p3, ρ, rates)
            limited_rates = PPP.patankar_limited_rates(rates, φ)

            # Evaporation is drawn in full: the air is dry enough that the equilibrium condensate
            # is zero, and backward Euler leaves 1 / (1 + εˡ Δt) of the cloud, whether the stage
            # is shorter or longer than `sink_limiting_timescale`...
            @test rates.condensation < 0
            @test φ.cloud_evaporation == 1
            @test limited_rates.condensation == rates.condensation
            remaining = (qᶜˡ + Δt * limited_rates.condensation) / qᶜˡ
            @test remaining ≈ 1 / (1 + εˡ * Δt) rtol = 100 * eps(FT) * (1 + εˡ * Δt)
            # ...which a second, Patankar, factor would have left at more than half of it.
            long_stage && @test remaining < FT(0.1)

            # The explicit cloud sinks are limited against what evaporation leaves.
            @test rates.autoconversion > 0
            @test 0 < φ.cloud ≤ 1
            long_stage && @test φ.cloud < 1
            explicit_loss = rates.autoconversion + rates.accretion
            @test φ.cloud ≈ patankar_fraction(ρ * qᶜˡ * remaining, Δt, ρ * explicit_loss)
            result = PPP.p3_tendencies(p3, ρ, ℳ, properties, limited_rates)
            @test ρ * qᶜˡ + Δt * result.tendency_ρqᶜˡ ≥ 0
        end
    end

    @testset "Aerosol activation from a small reservoir" begin
        aerosol = PPP.AerosolActivation(PPP.AerosolMode(FT); prognostic = true)
        p3 = PredictedParticlePropertiesMicrophysics(FT; aerosol, negative_moisture_correction = nothing)
        qᵛ = saturation(z) + FT(1e-3)
        𝒰 = thermodynamic_state(qᵛ, z)
        ρ = density(𝒰, constants)
        nᶜˡ, nᵃ = FT(1e6), FT(2e7)
        ℳ = P3MicrophysicalState(z, nᶜˡ, z, z, z, z, z, z, z, z, nᵃ, z)
        properties = PPP.p3_process_properties(p3, ρ, ℳ)

        c⁰ = (ρ * qᵛ, z, z, z, z, z, z, z, z, ρ * nᶜˡ, ρ * nᵃ)
        fields = one_cell_aerosol_fields(c⁰)
        G = one_cell_aerosol_fields(map(zero, c⁰))

        for x in FT.((10, 100, 1000))
            Δt = x * FT(aerosol.activation_timescale)
            stage = StageStart(fields, fields, false, Δt)
            rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties, T, z, z,
                                             nothing, PPP.GridLiquidRelaxationStage(1, 1, 1, G, stage))
            φ = PPP.p3_sink_fractions(1, 1, 1, G, stage, p3, ρ, rates)
            limited_rates = PPP.patankar_limited_rates(rates, φ)

            # The activated number exceeds the reservoir, so the droplet number relaxes toward
            # all of it, and backward Euler leaves 1 / (1 + x) of the reservoir, not (1 + x) / (1 + 2x).
            @test rates.ccn_activation_number > 0
            @test φ.aerosol == 1
            @test φ.condensation == 1
            @test limited_rates.ccn_activation_number == rates.ccn_activation_number
            @test limited_rates.ccn_activation_mass == rates.ccn_activation_mass
            remaining = (nᵃ - Δt * limited_rates.ccn_activation_number) / nᵃ
            @test remaining ≈ 1 / (1 + x) rtol = 100 * eps(FT) * (1 + x)
        end
    end
end

# The mass that the limited rates draw from vapor, cloud liquid and rain: every loss, with no gain.
function p3_donor_losses(rates)
    vapor = max(0, rates.condensation) + max(0, rates.deposition) + rates.nucleation_mass +
            rates.ccn_activation_mass + rates.rain_condensation + rates.coating_condensation
    cloud = max(0, -rates.condensation) + rates.autoconversion + rates.accretion + rates.cloud_riming +
            rates.cloud_freezing_mass + rates.cloud_homogeneous_mass + rates.cloud_warm_collection +
            rates.wet_growth_cloud + rates.wet_growth_shedding
    rain = rates.rain_evaporation + rates.rain_riming + rates.rain_freezing_mass +
           rates.rain_homogeneous_mass + rates.rain_warm_collection + rates.wet_growth_rain
    return vapor, cloud, rain
end

@testset "Implicit and explicit P3 draws together stay within the stage remainder [$FT]" for FT in (Float32, Float64)
    aerosol = PPP.AerosolActivation(PPP.AerosolMode(FT); prognostic = true)
    p3 = PredictedParticlePropertiesMicrophysics(FT; aerosol, negative_moisture_correction = nothing)
    constants = ThermodynamicConstants(FT)
    rng = LinearCongruentialGenerator(4321)
    roundoff = 100 * eps(FT)

    @testset "Δt = $Δt s" for Δt in (5, 45, 200, 5000)
        Δt = FT(Δt)
        overdrawn = zeros(Int, 4)             # vapor, cloud, rain, aerosol
        limited_twice = zeros(Int, 4)         # an implicit draw that fits but was scaled
        evaporating, condensing, activating = 0, 0, 0

        for _ in 1:2000
            ρ, 𝒰, ℳ₀, specific = random_p3_state(rng, p3, constants, FT)
            nᶜˡ = FT(10.0^(4 + 4 * uniform!(rng)))
            nᵃ = uniform!(rng) < 0.2 ? zero(FT) : FT(10.0^(5 + 4 * uniform!(rng)))
            ℳ = P3MicrophysicalState(ℳ₀.qᶜˡ, nᶜˡ, ℳ₀.qʳ, ℳ₀.nʳ, ℳ₀.qⁱ, ℳ₀.nⁱ, ℳ₀.qᶠ, ℳ₀.bᶠ, ℳ₀.qʷⁱ,
                                     zero(FT), nᵃ, zero(FT))
            specific = (specific..., nᶜˡ, nᵃ)
            properties = PPP.p3_process_properties(p3, ρ, ℳ)

            # A stage that starts from a state unrelated to the one the rates saw, as above
            c⁰ = map(q -> uniform!(rng) < 0.3 ? zero(FT) : ρ * q * FT(2 * uniform!(rng)), specific)
            Gᵒ = map((c, q) -> (-FT(uniform!(rng)) * c + (uniform!(rng) < 0.5 ? FT(uniform!(rng)) * ρ * q : zero(FT))) / Δt,
                     c⁰, specific)
            c̃ = max.(0, c⁰ .+ Δt .* Gᵒ)

            fields = one_cell_aerosol_fields(c⁰)
            G = one_cell_aerosol_fields(Gᵒ)
            stage = StageStart(fields, fields, false, Δt)

            rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties,
                                             temperature(𝒰, constants), zero(FT), zero(FT),
                                             nothing, PPP.GridLiquidRelaxationStage(1, 1, 1, G, stage))
            φ = PPP.p3_sink_fractions(1, 1, 1, G, stage, p3, ρ, rates)
            limited_rates = PPP.patankar_limited_rates(rates, φ)

            # Everything drawn from a donor, implicit and explicit, fits in what the stage leaves of it.
            available = (c̃[1], c̃[2], c̃[3], c̃[11])
            drawn = Δt .* ρ .* (p3_donor_losses(limited_rates)..., limited_rates.ccn_activation_number)
            overdrawn .+= drawn .> available .* (1 + roundoff)

            # An implicit draw that fits in what the stage leaves of its donor is taken in full.
            condensation = max(0, rates.condensation) + rates.ccn_activation_mass + rates.rain_condensation
            cloud_evaporation = max(0, -rates.condensation)
            asked = Δt .* ρ .* (condensation, cloud_evaporation, rates.rain_evaporation, rates.ccn_activation_number)
            fits = asked .≤ available
            unscaled = (limited_rates.rain_condensation == rates.rain_condensation,
                        limited_rates.condensation == rates.condensation,
                        limited_rates.rain_evaporation == rates.rain_evaporation,
                        limited_rates.ccn_activation_number == rates.ccn_activation_number)
            fits = (fits[1], ifelse(rates.condensation < 0, fits[2], fits[1]), fits[3], fits[4] & fits[1])
            limited_twice .+= fits .& .!unscaled

            evaporating += rates.condensation < 0
            condensing += rates.condensation > 0
            activating += rates.ccn_activation_number > 0
        end

        @test all(overdrawn .== 0)
        @test all(limited_twice .== 0)
        # The sample exercises each implicit transfer.
        @test evaporating > 100
        @test condensing > 20
        @test activating > 20
    end
end

#####
##### A kinematic column without negative-density repair
#####
##### P3 repairs nothing between stages, so its prognostics stay non-negative only if transport
##### keeps them so. This is the setup of `examples/kinematic_driver.jl`, shortened. With plain
##### WENO the condensate densities go negative within minutes, grow, and drive the temperature
##### negative; with positivity-bounded WENO on every non-negative scalar none goes negative.
#####

@testset "P3 kinematic column stays non-negative with bounded advection" begin
    using Breeze: ReferenceState, PrescribedDynamics, PrognosticVaporExcess, SaturationSpecificHumidity,
                  surface_density
    using Breeze.AtmosphereModels: prognostic_field_names
    using Breeze.Microphysics.PredictedParticleProperties: AerosolActivation, AerosolMode
    using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition

    grid = RectilinearGrid(default_arch; size=40, z=(0, 12e3), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    θ₀, p₀, ℋ₀, W₀, nᵃ₀ = 300, 1e5, 0.8, 2, 3e8
    θᵢ(z) = θ₀ + 43 * (z / 12e3)^(5/4)
    reference_state = ReferenceState(grid, constants; base_pressure=p₀, potential_temperature=θᵢ)
    dynamics = PrescribedDynamics(reference_state; divergence_correction=true)
    aerosol = AerosolActivation(AerosolMode(number_mixing_ratio=nᵃ₀); prognostic=true)
    microphysics = PredictedParticlePropertiesMicrophysics(; aerosol, vapor_excess=PrognosticVaporExcess())

    ρ₀ = surface_density(reference_state)
    qᵛ⁺ˡ₀ = saturation_specific_humidity(θ₀, ρ₀, constants, PlanarLiquidSurface())
    bcs = NamedTuple(name => FieldBoundaryConditions(bottom=ValueBoundaryCondition(0))
                     for name in prognostic_field_names(microphysics))
    bcs = merge(bcs, (; ρθ = FieldBoundaryConditions(bottom=ValueBoundaryCondition(ρ₀ * θ₀)),
                        ρqᵗ = FieldBoundaryConditions(bottom=ValueBoundaryCondition(ρ₀ * ℋ₀ * qᵛ⁺ˡ₀)),
                        ρδqᵛ⁺ˡ = FieldBoundaryConditions(bottom=ValueBoundaryCondition(ρ₀ * (ℋ₀ - 1) * qᵛ⁺ˡ₀)),
                        ρnᵃ = FieldBoundaryConditions(bottom=ValueBoundaryCondition(ρ₀ * nᵃ₀)),
                        w = FieldBoundaryConditions(bottom=NormalFlowBoundaryCondition(W₀),
                                                    top=NormalFlowBoundaryCondition(W₀))))

    weno = WENO(order=5)
    nonnegative_weno = WENO(order=5, bounds=(0.0, 1e30))
    signed_scalars = (:ρθ, :ρδqᵛ⁺ˡ)
    scalar_advection = NamedTuple(name => ifelse(name in signed_scalars, weno, nonnegative_weno)
                                  for name in (:ρθ, :ρqᵛ, prognostic_field_names(microphysics)...))

    model = AtmosphereModel(grid; dynamics, microphysics, boundary_conditions=bcs,
                            momentum_advection=weno, scalar_advection, thermodynamic_constants=constants)
    set!(model; θ=θᵢ, ℋ=ℋ₀, w=W₀)
    μ = model.microphysical_fields
    set!(model; ρδqᵛ⁺ˡ = reference_state.density * (μ.qᵛ - SaturationSpecificHumidity(model)))

    names = (:ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ, :ρnᵃ)
    minima = Dict(name => Inf for name in names)
    for _ in 1:600
        time_step!(model, 2)
        for name in names
            minima[name] = min(minima[name], minimum(getproperty(μ, name)))
        end
    end

    @test maximum(μ.ρqᶜˡ) > 0
    @test all(minima[name] ≥ 0 for name in names)
    @test minimum(model.moisture_density) > 0
end
