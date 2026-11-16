include(joinpath(@__DIR__, "setup.jl"))

using Test

using Breeze
using Breeze.AtmosphereModels: AtmosphereModels, StageStart
using Breeze.Microphysics.PredictedParticleProperties: AerosolActivation, AerosolMode,
                                                       PredictedParticlePropertiesMicrophysics

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
