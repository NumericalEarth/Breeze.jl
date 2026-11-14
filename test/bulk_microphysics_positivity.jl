include(joinpath(@__DIR__, "setup.jl"))

#####
##### Patankar-limited sinks of the CloudMicrophysics one- and two-moment schemes: the sink
##### budgets, and non-negativity and water conservation in an AtmosphereModel at time steps
##### much longer than the numerical timescale τⁿᵘᵐ.
#####

using Breeze
using Breeze.AtmosphereModels: StageStart, patankar_fraction, stage_interval_is_known
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudIce, CloudLiquid
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, TwoMomentCloudMicrophysics,
                                   one_moment_cloud_microphysics_categories,
                                   two_moment_cloud_microphysics_categories,
                                   TimescaleSinkBudget, StageSinkBudget, donor_limiter,
                                   sink_fraction, diagnostic_sink_fraction, sink_rate_cap,
                                   negative_value_relaxation, relaxation_donor_fraction,
                                   coupled_sink_fractions

mixed_phase_non_equilibrium_1M(FT; kw...) =
    OneMomentCloudMicrophysics(FT; cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT)), kw...)

mixed_phase_saturation_adjustment_1M(FT; kw...) =
    OneMomentCloudMicrophysics(FT; cloud_formation = SaturationAdjustment(FT; equilibrium = MixedPhaseEquilibrium(FT)), kw...)

is_density(name) = startswith(string(name), "ρq") || startswith(string(name), "ρn")
is_water(name) = startswith(string(name), "ρq")

@testset "Sink budgets [$FT]" for FT in (Float32, Float64)
    ρ = FT(1)
    τ = FT(10)
    Δt = FT(60)
    fields = (; ρq = fill(FT(2e-3), 1, 1, 1))
    G = (; ρq = fill(FT(-1e-5), 1, 1, 1))
    stage = StageStart(fields, fields, false, Δt)
    budget = StageSinkBudget(1, 1, 1, G, stage, τ)
    gridless = TimescaleSinkBudget(τ)

    @test patankar_fraction(FT(1), FT(10), FT(0.3)) ≈ FT(1 / 4)
    @test patankar_fraction(FT(-1), FT(10), FT(0.3)) == 0
    @test patankar_fraction(FT(1), FT(10), zero(FT)) == 1

    # The gridless budget lets a reservoir lose its content over τ.
    q, ΣS = FT(2e-3), FT(1e-3)
    @test @inferred(sink_fraction(gridless, Val(:ρq), ρ, q, ΣS)) == donor_limiter(q, ΣS, τ)
    @test sink_fraction(gridless, Val(:ρq), ρ, q, ΣS) ≈ FT(1 / 5)

    # On the grid c̃ = 2e-3 - 60 × 1e-5 = 1.4e-3, and the loss asks for 60 × 1e-3: the
    # Patankar fraction leaves c̃² / (c̃ + Δt L) > 0 whatever τ is.
    c̃ = FT(1.4e-3)
    φ = @inferred sink_fraction(budget, Val(:ρq), ρ, q, ΣS)
    @test φ ≈ c̃ / (c̃ + Δt * ΣS)
    @test c̃ - Δt * φ * ΣS > 0
    @test sink_fraction(StageSinkBudget(1, 1, 1, G, stage, FT(1)), Val(:ρq), ρ, q, ΣS) == φ

    # No demand, no limit; an empty reservoir at the evaluated state gives nothing.
    @test sink_fraction(budget, Val(:ρq), ρ, q, zero(FT)) == 1
    @test sink_fraction(budget, Val(:ρq), ρ, FT(-1e-6), ΣS) == 0
    @test sink_fraction(budget, Val(:ρq), ρ, q, 0) == 1  # disabled processes return integer zeros

    # A diagnostic reservoir is budgeted against its content at the evaluated state.
    @test @inferred(diagnostic_sink_fraction(budget, ρ, q, ΣS)) ≈ q / (q + Δt * ΣS)
    @test diagnostic_sink_fraction(gridless, ρ, q, ΣS) == donor_limiter(q, ΣS, τ)

    # A single process is capped only where its reservoir is empty.
    @test sink_rate_cap(gridless, q) ≈ q / τ
    @test sink_rate_cap(budget, q) == FT(Inf)
    @test sink_rate_cap(budget, FT(-1e-6)) == 0

    # Before the first time step the stage interval is unknown and the grid budget falls
    # back to the fixed timescale.
    unknown = StageSinkBudget(1, 1, 1, G, StageStart(fields, fields, false, FT(Inf)), τ)
    @test !stage_interval_is_known(unknown.stage)
    @test sink_fraction(unknown, Val(:ρq), ρ, q, ΣS) == sink_fraction(gridless, Val(:ρq), ρ, q, ΣS)
    @test diagnostic_sink_fraction(unknown, ρ, q, ΣS) == donor_limiter(q, ΣS, τ)
    @test sink_rate_cap(unknown, q) == sink_rate_cap(gridless, q)

    # A negative value relaxes to zero by backward Euler over the stage: with
    # c̃ = c⁰ + Δt (Gᵒ + Gᵖ) < 0, the relaxed value c̃ + Δt δ = c̃ τ / (τ + Δt) stays negative
    # and is closer to zero for any Δt.
    negative = (; ρq = fill(FT(-1e-4), 1, 1, 1))
    Gᵒ = (; ρq = fill(FT(1e-7), 1, 1, 1))
    Gᵖ = FT(2e-7)
    relaxing = StageSinkBudget(1, 1, 1, Gᵒ, StageStart(negative, negative, false, Δt), τ)
    c̃ = FT(-1e-4) + Δt * (FT(1e-7) + Gᵖ)
    δ = @inferred negative_value_relaxation(relaxing, Val(:ρq), ρ, FT(-1e-4), Gᵖ)
    @test δ ≈ -c̃ / (τ + Δt)
    @test c̃ < c̃ + Δt * δ < 0
    @test negative_value_relaxation(budget, Val(:ρq), ρ, q, Gᵖ) == 0

    # The gridless relaxation replaces the tendency of a negative reservoir with -ρ q / τ.
    @test Gᵖ + negative_value_relaxation(gridless, Val(:ρq), ρ, FT(-1e-4), Gᵖ) ≈ FT(1e-4) / τ
    @test negative_value_relaxation(gridless, Val(:ρq), ρ, q, Gᵖ) == 0

    # What a relaxation takes from its donor leaves the donor non-negative.
    Gᵖ = FT(-1e-5)
    c̃ = FT(2e-3) + Δt * (FT(-1e-5) + Gᵖ)
    g = @inferred relaxation_donor_fraction(budget, Val(:ρq), Gᵖ, FT(1e-3))
    @test g ≈ c̃ / (c̃ + Δt * FT(1e-3))
    @test c̃ - Δt * g * FT(1e-3) > 0
    @test relaxation_donor_fraction(budget, Val(:ρq), Gᵖ, zero(FT)) == 1
    @test relaxation_donor_fraction(gridless, Val(:ρq), Gᵖ, FT(1e-3)) == 1

    # Mass and number: on the grid each transfer takes the fraction of what it draws on.
    @test coupled_sink_fractions(budget, FT(0.5), FT(0.25)) == (FT(0.5), FT(0.25), FT(0.25))
    @test coupled_sink_fractions(budget, FT(0.25), FT(0.5)) == (FT(0.25), FT(0.25), FT(0.5))
    @test coupled_sink_fractions(gridless, FT(0.5), FT(0.25)) == (FT(0.25), FT(0.25), FT(0.25))
    @test coupled_sink_fractions(unknown, FT(0.5), FT(0.25)) == (FT(0.25), FT(0.25), FT(0.25))
end

# A single cell with no transport: a closed column one cell deep, whose precipitation
# cannot leave through the bottom. Only microphysics changes it, so its water is conserved.
function single_cell_model(FT, microphysics)
    grid = RectilinearGrid(default_arch, FT; size = 1, z = (0, 500), topology = (Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(FT); base_pressure = 9e4, potential_temperature = 280)
    return AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), microphysics)
end

# Step `model` and return the smallest value each density took and the largest relative
# change of the total water.
function minima_and_water_drift!(model, Δt, steps)
    fields = prognostic_fields(model)
    names = filter(is_density, keys(fields))
    water_names = filter(is_water, names)
    total_water() = sum(@allowscalar(first(interior(fields[name]))) for name in water_names)

    initial_water = total_water()
    minima = Dict(name => Inf for name in names)
    drift = zero(initial_water)

    for _ in 1:steps
        time_step!(model, Δt)
        for name in names
            minima[name] = min(minima[name], minimum(fields[name]))
        end
        drift = max(drift, abs(total_water() - initial_water) / initial_water)
    end

    return minima, drift
end

@testset "A cell of rain and snow stays non-negative at Δt = 60 s ≫ τⁿᵘᵐ [$FT]" for FT in test_float_types()
    Δt = 60
    steps = 30
    precipitation_boundary_condition = ImpenetrableBoundaryCondition()

    # Dry air holding cloud, rain and snow: everything evaporates, sublimates, melts or is
    # collected much faster than the time step. θ sets a cold, a near-freezing and a warm cell.
    @testset "Mixed-phase non-equilibrium 1M [θ = $θ]" for θ in (262, 272, 285)
        microphysics = mixed_phase_non_equilibrium_1M(FT; precipitation_boundary_condition)
        @test microphysics.categories.τⁿᵘᵐ == 10
        model = single_cell_model(FT, microphysics)
        set!(model; θ, qᵗ = 5e-3, qᶜˡ = 1e-3, qᶜⁱ = 1e-3, qʳ = 1e-3, qˢⁿ = 1e-3)
        minima, drift = minima_and_water_drift!(model, Δt, steps)

        @test keys(minima) == Set((:ρqᵛ, :ρqᶜˡ, :ρqᶜⁱ, :ρqʳ, :ρqˢⁿ))
        @test all(minimum ≥ 0 for minimum in values(minima))
        @test drift < 100 * eps(FT)
    end

    @testset "Mixed-phase saturation-adjustment 1M [θ = $θ]" for θ in (262, 272, 285)
        microphysics = mixed_phase_saturation_adjustment_1M(FT; precipitation_boundary_condition)
        model = single_cell_model(FT, microphysics)
        set!(model; θ, qᵗ = 5e-3, qʳ = 1e-3, qˢⁿ = 1e-3)
        minima, drift = minima_and_water_drift!(model, Δt, steps)

        @test keys(minima) == Set((:ρqᵉ, :ρqʳ, :ρqˢⁿ))
        @test all(minimum ≥ 0 for minimum in values(minima))
        @test drift < 100 * eps(FT)
    end

    @testset "Warm-phase non-equilibrium 2M [θ = $θ]" for θ in (285, 300)
        microphysics = TwoMomentCloudMicrophysics(FT; precipitation_boundary_condition)
        @test microphysics.categories.τⁿᵘᵐ == 10
        model = single_cell_model(FT, microphysics)
        set!(model; θ, qᵗ = 5e-3, qᶜˡ = 1e-3, nᶜˡ = 1e8, qʳ = 1e-3, nʳ = 1e4)
        minima, drift = minima_and_water_drift!(model, Δt, steps)

        @test keys(minima) == Set((:ρqᵛ, :ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :ρnᵃ))
        @test all(minimum ≥ 0 for minimum in values(minima))
        @test drift < 100 * eps(FT)
    end
end

# The moist bubble of the P3 positivity test (multirate_infinitesimal_step.jl): bounded
# advection of every scalar, no repair of negative densities, and a coefficient set whose
# stages are convex combinations of forward-Euler steps no longer than Δt / C.
function bounded_bubble_model(microphysics)
    FT = Float64
    grid = RectilinearGrid(default_arch, FT; size = (32, 24), halo = (5, 5),
                           x = (-10e3, 10e3), z = (0, 12e3), topology = (Bounded, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5,
                                    standard_pressure = 1e5, reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, microphysics, momentum_advection = WENO(order=5),
                            scalar_advection = WENO(order=5, bounds=(0.0, 1e30)),
                            timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients = :SSPMIS43))

    constants = model.thermodynamic_constants
    θᵢ(x, z) = 300 + 3 * cos(π * min(1, sqrt((x / 2e3)^2 + ((z - 2e3) / 2e3)^2)) / 2)^2
    ρᵢ(x, z) = adiabatic_hydrostatic_density(z, 1e5, 300, 1e5, constants)
    set!(model; ρ = ρᵢ, θ = θᵢ, qᵗ = 0.012)
    return model
end

@testset "Bulk microphysics stays non-negative under strong-stability-preserving MIS [$scheme]" for scheme in (:one_moment, :two_moment)
    # The sinks are budgeted over the stage interval Δt / C, so the result does not rely on
    # τⁿᵘᵐ exceeding it: τⁿᵘᵐ = 1 s is well below Δt / C ≈ 5.6 s.
    FT = Float64
    τⁿᵘᵐ = 1
    microphysics = scheme === :one_moment ?
        mixed_phase_non_equilibrium_1M(FT; categories = one_moment_cloud_microphysics_categories(FT; τⁿᵘᵐ)) :
        TwoMomentCloudMicrophysics(FT; categories = two_moment_cloud_microphysics_categories(FT; τⁿᵘᵐ = FT(τⁿᵘᵐ)))
    @test isnothing(microphysics.negative_moisture_correction)

    model = bounded_bubble_model(microphysics)
    fields = prognostic_fields(model)
    names = filter(is_density, keys(fields))
    condensates = filter(name -> name ∉ (:ρqᵛ, :ρnᵃ), names)
    minima = Dict(name => Inf for name in names)
    maxima = Dict(name => -Inf for name in names)

    Δt = 10
    for _ in 1:180
        time_step!(model, Δt)
        for name in names
            minima[name] = min(minima[name], minimum(fields[name]))
            maxima[name] = max(maxima[name], maximum(fields[name]))
        end
    end

    C = model.timestepper.tableau.strong_stability_coefficient
    @test Δt / C > τⁿᵘᵐ

    # Every species forms, so the test exercises all the sinks...
    @test all(maxima[name] > 0 for name in condensates)

    # ...and none of them is negative. Patankar-limited sinks approach zero without reaching
    # it, leaving neighbouring cells many orders of magnitude apart, and the bounded advection
    # of such a field is non-negative only to the roundoff of the larger neighbour: the
    # smallest value seen is of order -1e-50 kg m⁻³.
    @test all(minima[name] ≥ -eps(FT) * maxima[name] for name in names)
    @test minima[:ρqᵛ] > 0
end
