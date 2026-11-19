include(joinpath(@__DIR__, "setup.jl"))

using Test

using Breeze
using Breeze.AtmosphereModels: AtmosphereModels, StageStart
using Breeze.Microphysics.PredictedParticleProperties: AerosolActivation, AerosolMode,
                                                       PredictedParticlePropertiesMicrophysics

using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             PlanarIceSurface, saturation_specific_humidity, density, temperature,
                             with_temperature
using Breeze.Microphysics.PredictedParticleProperties: P3MicrophysicalState, compute_p3_process_rates

using Oceananigans: Bounded, CenterField, Flat, RectilinearGrid, WENO, interior, prognostic_fields, set!,
                     time_step!
using Oceananigans.Utils: AdaptiveVerticallyImplicitDiscretization

const PPP = Breeze.Microphysics.PredictedParticleProperties

#####
##### The relaxation of a number into its size bounds
#####

@testset "Relaxation of a number into its size bounds [$FT]" for FT in (Float32, Float64)
    τ = FT(10)
    lower, upper = FT(2), FT(5)
    field = fill(zero(FT), 1, 1, 1)
    relaxation(c̃, Δt; explicit = zero(FT)) =
        PPP.bounded_number_relaxation(c̃, lower, upper, explicit, τ, StageStart(field, field, false, Δt))

    @testset "Δt = $Δt s" for Δt in FT.((1, 10, 300, 1e6))
        # Zero within the bounds, and at them
        @test all(iszero, relaxation.(FT.((2, 2.5, 3, 4.5, 5)), Δt))

        # Continuous across the bounds, with a slope no larger than one: no switch
        for bound in (lower, upper), δ in FT.((1e-1, 1e-3, 1e-5))
            roundoff = 4 * eps(bound)
            @test abs(relaxation(bound + δ, Δt) - relaxation(bound - δ, Δt)) * Δt ≤ δ + roundoff
        end

        # Outside them the number ends between where it was and the bound, for any Δt
        for c̃ in FT.((-3, 0, 1, 7, 100))
            c⁺ = c̃ + Δt * relaxation(c̃, Δt)
            bound = clamp(c̃, lower, upper)
            @test min(c̃, bound) ≤ c⁺ ≤ max(c̃, bound)
            @test c⁺ ≈ (c̃ + Δt / τ * bound) / (1 + Δt / τ)
        end

        # The explicit relaxation that the rates carry is taken back
        @test relaxation(FT(7), Δt; explicit = FT(0.3)) ≈ relaxation(FT(7), Δt) - FT(0.3)
    end

    # Before the first time step the stage interval is not known, and the explicit relaxation stands
    @test iszero(relaxation(FT(7), FT(Inf); explicit = FT(0.3)))
end

#####
##### The weight of the relaxation where the stage leaves no mass
#####

@testset "Weight of the number relaxation [$FT]" for FT in (Float32, Float64)
    qˢ = FT(1e-14)
    weight(q̃) = PPP.number_relaxation_weight(q̃, qˢ)

    # None where the species is absent, all of it from twice the threshold, linear in between
    @test all(iszero, weight.(FT.((-1e-3, -1e-20, 0, 1e-20, 1e-14))))
    @test all(isone, weight.(FT.((2e-14, 1e-12, 1e-3))))
    @test weight(FT(1.5e-14)) ≈ FT(0.5)
    @test weight(FT(1.25e-14)) ≈ FT(0.25)

    # Continuous at both ends of the ramp
    for q̃ in (qˢ, 2qˢ), δ in FT.((1e-2, 1e-4)) .* qˢ
        @test abs(weight(q̃ + δ) - weight(q̃ - δ)) ≤ 2δ / qˢ * (1 + 10 * eps(FT))
    end

    # A threshold of zero is the step at no mass
    @test PPP.number_relaxation_weight(FT(1e-20), zero(FT)) == 1
    @test PPP.number_relaxation_weight(zero(FT), zero(FT)) == 0
    @test PPP.number_relaxation_weight(FT(-1e-20), zero(FT)) == 0

    # The weighted relaxation leaves the number between where it was and the bound. Where the
    # weight is zero a number above the lower bound is left alone, and one below it, a negative
    # one among them, still rises to it.
    τ, Δt = FT(10), FT(300)
    lower, upper = FT(2), FT(5)
    field = fill(zero(FT), 1, 1, 1)
    stage = StageStart(field, field, false, Δt)
    for c̃ in FT.((-3, 0, 1, 7, 100)), w in FT.((0, 0.3, 1))
        c⁺ = c̃ + Δt * PPP.bounded_number_relaxation(c̃, lower, upper, zero(FT), τ, stage, w)
        bound = clamp(c̃, lower, upper)
        @test min(c̃, bound) ≤ c⁺ ≤ max(c̃, bound)
        w == 0 && c̃ ≥ lower && @test c⁺ == c̃
        c̃ < lower && @test c⁺ ≈ (c̃ + Δt / τ * lower) / (1 + Δt / τ)
        w == 1 && @test c⁺ == c̃ + Δt * PPP.bounded_number_relaxation(c̃, lower, upper, zero(FT), τ, stage)
    end
    for w in FT.((0, 0.3, 1)) # no mass: the bounds are zero, and a negative number is brought back
        c⁺ = -FT(3) + Δt * PPP.bounded_number_relaxation(-FT(3), zero(FT), zero(FT), zero(FT), τ, stage, w)
        @test c⁺ ≈ -FT(3) / (1 + Δt / τ)
    end

    # Where the weight is zero the explicit relaxation of the rates is still taken back
    @test PPP.bounded_number_relaxation(FT(7), zero(FT), zero(FT), FT(0.3), τ, stage, zero(FT)) == -FT(0.3)
end

#####
##### A cell that the stage leaves without mass keeps its number
#####
##### Transport that is not sign-preserving leaves cells with no rain or ice mass, or a little
##### less than none, and a positive number. The bounds of no mass are zero, and a relaxation
##### toward them would drain the number, so that mass carried back into the cell would have the
##### largest particles the scheme admits. The number is left to transport instead.
#####

@testset "A cell without mass keeps its number [$FT]" for FT in (Float32, Float64)
    p3 = PredictedParticlePropertiesMicrophysics(FT; negative_moisture_correction = nothing)
    constants = ThermodynamicConstants(FT)
    z = zero(FT)
    names = (:ρqᵛ, :ρqᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ)
    one_cell(values) = NamedTuple{names}(map(v -> fill(v, 1, 1, 1), values))

    # Air at 0.8 bar and 265 K, saturated over ice: nothing nucleates, deposits or sublimates
    T, p = FT(265), FT(8e4)
    thermodynamic_state(qᵛ) = with_temperature(
        LiquidIcePotentialTemperatureState(z, MoistureMassFractions(qᵛ, z, z), FT(1e5), p), T, constants)
    qᵛ = FT(1e-3)
    for _ in 1:10
        qᵛ = saturation_specific_humidity(T, density(thermodynamic_state(qᵛ), constants), constants, PlanarIceSurface())
    end
    𝒰 = thermodynamic_state(qᵛ)
    ρ = density(𝒰, constants)
    ρqˢ = ρ * FT(p3.minimum_mass_mixing_ratio)
    nʳ, nⁱ = FT(1e3), FT(1e4)

    # The number tendencies of rain and ice over a stage that starts with the mass densities `ρqʳ`
    # and `ρqⁱ` and the numbers above, and the relaxations among them.
    function number_tendencies(ρqʳ, ρqⁱ, Δt)
        qʳ, qⁱ = max(0, ρqʳ) / ρ, max(0, ρqⁱ) / ρ
        ℳ = P3MicrophysicalState(z, FT(p3.cloud.number_concentration), qʳ, nʳ, qⁱ, nⁱ, z, z, z, z, z, z)
        properties = PPP.p3_process_properties(p3, ρ, ℳ)
        fields = one_cell((ρ * qᵛ, z, ρqʳ, ρ * nʳ, ρqⁱ, ρ * nⁱ, z, z, z))
        G = one_cell(ntuple(_ -> z, 9))
        stage = StageStart(fields, fields, false, Δt)
        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties, temperature(𝒰, constants), z, z,
                                         nothing, PPP.GridLiquidRelaxationStage(1, 1, 1, G, stage))
        φ = PPP.p3_sink_fractions(1, 1, 1, G, stage, p3, ρ, properties, rates)
        limited_rates = PPP.patankar_limited_rates(rates, φ)
        result = PPP.p3_tendencies(p3, ρ, ℳ, properties, limited_rates)
        relaxations = PPP.p3_number_relaxations(1, 1, 1, G, stage, p3, ρ, ℳ, properties, limited_rates, result)
        return (rain = result.tendency_ρnʳ + relaxations.rain, ice = result.tendency_ρnⁱ + relaxations.ice,
                rain_relaxation = relaxations.rain, ice_relaxation = relaxations.ice,
                rain_explicit = ρ * limited_rates.rain_number_correction,
                ice_explicit = ρ * (limited_rates.ice_number_correction - limited_rates.ni_limit))
    end

    @testset "Δt = $Δt s" for Δt in FT.((1, 120, 1e4))
        # No mass, less than none, and a trace below the threshold: the numbers do not change,
        # and the explicit relaxation that the rates carry toward no number is taken back.
        for ρq in (z, -FT(1e-9), -ρqˢ, ρqˢ / 2)
            G = number_tendencies(ρq, ρq, Δt)
            @test G.rain_explicit < 0
            @test G.ice_explicit < 0
            @test G.rain_relaxation == -G.rain_explicit
            @test G.ice_relaxation == -G.ice_explicit
            @test abs(G.rain) * Δt ≤ 10 * eps(FT) * ρ * nʳ
            @test abs(G.ice) * Δt ≤ 10 * eps(FT) * ρ * nⁱ
        end
    end

    # With a mass well above the threshold the same numbers, far more than that mass can carry,
    # are relaxed toward its bounds.
    G = number_tendencies(FT(1e-9), FT(1e-12), FT(1))
    @test G.rain_relaxation + G.rain_explicit < 0
    @test G.ice_relaxation + G.ice_explicit < 0
end

#####
##### Perturbation growth in a mixed-phase column
#####
##### Cloud droplets (with a prognostic number), rain and ice form within a few time steps of a
##### saturated column that is below freezing aloft, at a time step much longer than
##### `sink_limiting_timescale`. Two runs whose initial states differ by one part in 10¹² have to
##### stay close. A relaxation of the numbers that is switched on where the state the rates are
##### evaluated at is out of its size bounds sets them apart by order one within two time steps.
#####

function mixed_phase_column(; perturbation = 0)
    grid = RectilinearGrid(default_arch; size = 40, z = (0, 10000), topology = (Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure = 101500, potential_temperature = 292)
    microphysics = PredictedParticlePropertiesMicrophysics(; thermodynamic_constants = constants,
                                                           aerosol = AerosolActivation(AerosolMode()))

    # Sedimentation is advanced implicitly where it is fast, and keeps the fields non-negative
    names = unique((AtmosphereModels.moisture_prognostic_name(microphysics),
                    AtmosphereModels.prognostic_field_names(microphysics)...))
    non_negative_advection() = WENO(order = 5, bounds = (0.0, Inf),
                                    time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl = 1/4))
    scalar_advection = NamedTuple{Tuple(names)}(map(_ -> non_negative_advection(), names))

    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state),
                            thermodynamic_constants = constants, microphysics, scalar_advection)

    # The perturbation changes sign every few cells
    wiggle(z) = 1 + perturbation * sin(z / 97)
    θ = CenterField(grid)
    qᵗ = CenterField(grid)
    set!(θ, z -> (292 + 3.5e-3 * z) * wiggle(z))
    set!(qᵗ, z -> 15e-3 * exp(-z / 2800) * wiggle(z))
    set!(model; θ, qᵗ)

    return model
end

@testset "Perturbation growth in a mixed-phase P3 column" begin
    Δt = 120
    reference = mixed_phase_column()
    perturbed = mixed_phase_column(perturbation = 1e-12)
    names = (:ρθ, :ρqᵛ, :ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ)

    column(model, name) = vec(Array(interior(prognostic_fields(model)[name])))

    # The largest difference of each field, relative to the largest value of the field
    function relative_difference(name)
        a, b = column(reference, name), column(perturbed, name)
        scale = max(maximum(abs, a), maximum(abs, b))
        return scale > 0 ? maximum(abs, a .- b) / scale : zero(scale)
    end

    growth = zeros(length(names))

    for _ in 1:15
        time_step!(reference, Δt)
        time_step!(perturbed, Δt)
        growth .= max.(growth, map(relative_difference, names))
    end

    # The column is mixed-phase, and every number is exercised
    for name in (:ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ)
        @test maximum(column(reference, name)) > 0
    end

    # 10⁻¹² grows to no more than 10⁻⁸ over 15 time steps (it reaches a few 10⁻⁹ in the rain and
    # cloud numbers, which are new within the first step, and stays at 10⁻¹² in ρθ)
    @testset "$name" for (name, g) in zip(names, growth)
        @test g < 1e-8
    end
end
