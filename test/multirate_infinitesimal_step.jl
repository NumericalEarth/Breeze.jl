include(joinpath(@__DIR__, "setup.jl"))

#####
##### MultirateInfinitesimalStep: the coefficient tables, the reduction to
##### AcousticRungeKutta3 under the Wicker–Skamarock tableau, rest-state
##### preservation, and a short thermal-bubble integration for every set.
#####

using Breeze
using Breeze.TimeSteppers: multirate_infinitesimal_step_tableau, number_of_stages
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: update_state!
using LinearAlgebra: istril
using Test

const mis_coefficient_sets = (:WickerSkamarock3, :MIS2, :MIS3C, :MIS4, :MIS4a, :TVDMISA, :TVDMISB, :MIS3opt, :SSPMIS43, :SSPMIS42, :MIS54, :MISF53, :MISR33, :MISR43)

function mis_bubble_model(; timestepper, coefficients = :MIS2, moist = false, Δθ = 2, qᵗ = 0.012,
                          scalar_advection = WENO(order=5),
                          pressure_coefficient = FrozenCompositionPressureCoefficient())
    grid = RectilinearGrid(default_arch; size = (16, 8), halo = (5, 5),
                           x = (-10e3, 10e3), z = (0, 10e3),
                           topology = (Bounded, Flat, Bounded))
    θ₀ = 300
    p₀ = 1e5
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(; pressure_coefficient);
                                    base_pressure = p₀, standard_pressure = p₀,
                                    reference_potential_temperature = θ₀)
    microphysics = moist ? SaturationAdjustment(; equilibrium = WarmPhaseEquilibrium()) : nothing
    timestepper_kwargs = timestepper === :MultirateInfinitesimalStep ? (; coefficients) : NamedTuple()
    model = AtmosphereModel(grid; dynamics, microphysics, momentum_advection = WENO(order=5), scalar_advection,
                            timestepper, timestepper_kwargs)
    constants = model.thermodynamic_constants
    θᵢ(x, z) = θ₀ + Δθ * cos(π * min(1, sqrt((x / 2e3)^2 + ((z - 2e3) / 2e3)^2)) / 2)^2
    ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, p₀, constants)
    moist ? set!(model; ρ = ρᵢ, θ = θᵢ, qᵗ) : set!(model; ρ = ρᵢ, θ = θᵢ)
    return model
end

@testset "MIS coefficient tables" begin
    for name in mis_coefficient_sets
        tableau = multirate_infinitesimal_step_tableau(name)
        s = number_of_stages(tableau)
        # Consistency: the last stage reaches tⁿ⁺¹, and every stage interval is positive.
        @test tableau.c[s] ≈ 1 atol = 1e-8
        @test all(tableau.d .> 0)
        @test istril(tableau.α, -1)
        @test istril(tableau.γ, -1)
        @test istril(tableau.β)
    end
end

@testset "MIS with the Wicker–Skamarock tableau tracks AcousticRungeKutta3 [$FT]" for FT in (Float64,)
    for moist in (false, true)
        reference = mis_bubble_model(; timestepper = :AcousticRungeKutta3, moist)
        mis = mis_bubble_model(; timestepper = :MultirateInfinitesimalStep, coefficients = :WickerSkamarock3, moist)
        @test mis.timestepper isa MultirateInfinitesimalStep
        for _ in 1:5
            time_step!(reference, 5)
            time_step!(mis, 5)
        end
        @test mis.clock.time ≈ reference.clock.time
        # MIS keeps the step-start linearization for every stage, while AcousticRungeKutta3
        # re-linearizes per stage, so the two agree to the (second-order) linearization error.
        for name in (:ρᵈ, :ρθ)
            a = Array(interior(prognostic_fields(mis)[name]))
            b = Array(interior(prognostic_fields(reference)[name]))
            @test maximum(abs, a .- b) / maximum(abs, b) < 1e-5
        end
        w_mis = Array(interior(mis.velocities.w))
        w_reference = Array(interior(reference.velocities.w))
        @test maximum(abs, w_mis .- w_reference) / maximum(abs, w_reference) < 1e-2
    end
end

@testset "MIS preserves a resting atmosphere [$coefficients]" for coefficients in mis_coefficient_sets
    model = mis_bubble_model(; timestepper = :MultirateInfinitesimalStep, coefficients, Δθ = 0)

    # The discrete reference state, exactly: a continuous-formula θ balances only to truncation error.
    ref = model.dynamics.reference_state
    Rᵈ = Breeze.dry_air_gas_constant(model.thermodynamic_constants)
    parent(model.dynamics.dry_density) .= parent(ref.density)
    ρθ = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
    parent(ρθ) .= parent(ref.pressure) ./ (Rᵈ .* parent(ref.exner_function))
    fill!(parent(model.momentum.ρu), 0)
    fill!(parent(model.momentum.ρw), 0)
    update_state!(model)

    for _ in 1:10
        time_step!(model, 5)
    end
    @test maximum(abs, interior(model.velocities.w)) < 1e-10
    @test maximum(abs, interior(model.velocities.u)) < 1e-10
end

@testset "MIS thermal bubble [$coefficients, moist = $moist]" for coefficients in mis_coefficient_sets, moist in (false, true)
    model = mis_bubble_model(; timestepper = :MultirateInfinitesimalStep, coefficients, moist)
    for _ in 1:20
        time_step!(model, 5)
    end
    w = Array(interior(model.velocities.w))
    @test all(isfinite, w)
    @test maximum(w) > 0.1
    @test model.clock.time ≈ 100
end

# MIS linearizes once per step with the substepper it shares with AcousticRungeKutta3, so the
# phase-equilibrium pressure coefficient (NumericalEarth/Breeze.jl#1086) reaches its cache too:
# different from the frozen slope where the bubble has condensed (about 10% lower in lightly loaded
# cloud; in heavily loaded cloud the ∂p/∂ρᵈ part it carries can outweigh that), identical elsewhere,
# and the step runs.
@testset "MIS carries the phase-equilibrium pressure coefficient" begin
    pressure_coefficient = PhaseEquilibriumPressureCoefficient()
    equilibrium = mis_bubble_model(; timestepper = :MultirateInfinitesimalStep, moist = true, qᵗ = 0.03, pressure_coefficient)
    frozen = mis_bubble_model(; timestepper = :MultirateInfinitesimalStep, moist = true, qᵗ = 0.03)
    @test equilibrium.timestepper.substepper.pressure_coefficient === pressure_coefficient

    for model in (equilibrium, frozen)
        time_step!(model, 5)
    end

    qˡ = Array(interior(equilibrium.microphysical_fields.qˡ))
    Cᵉ = Array(interior(equilibrium.timestepper.substepper.linearization_pressure_coefficient))
    Cᶠ = Array(interior(frozen.timestepper.substepper.linearization_pressure_coefficient))
    cloudy = qˡ .> 0
    @test any(cloudy)
    @test all(Cᵉ[.!cloudy] .== Cᶠ[.!cloudy])
    @test all(Cᵉ[cloudy] .!= Cᶠ[cloudy])
    @test all(0.8 .< Cᶠ[cloudy] ./ Cᵉ[cloudy] .< 1.25)
    @test all(1.05 .< Cᶠ[cloudy .& (qˡ .< 0.005)] ./ Cᵉ[cloudy .& (qˡ .< 0.005)] .< 1.2)

    for _ in 1:19
        time_step!(equilibrium, 5)
    end
    @test all(isfinite, Array(interior(equilibrium.velocities.w)))
end

@testset "MIS with adaptive-implicit vertical advection" begin
    using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization, adaptive_advection_timestep
    grid = RectilinearGrid(default_arch; size = (16, 8), halo = (5, 5), x = (-10e3, 10e3), z = (0, 10e3),
                           topology = (Bounded, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5, standard_pressure = 1e5,
                                    reference_potential_temperature = 300)
    advection = WENO(order = 5, time_discretization = AdaptiveVerticallyImplicitDiscretization(; cfl = 0.5))
    model = AtmosphereModel(grid; dynamics, advection, timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients = :MIS3C))
    set!(model; ρ = model.dynamics.reference_state.density,
         θ = (x, z) -> 300 + 2 * cos(π * min(1, sqrt((x / 2e3)^2 + ((z - 2e3) / 2e3)^2)) / 2)^2)
    for _ in 1:10
        time_step!(model, 20)
    end
    ts = model.timestepper
    @test all(isfinite, Array(interior(model.velocities.w)))
    # The split is sized for the stage the next tendencies feed: stage 1 of the next step, d₁ Δt.
    @test adaptive_advection_timestep(ts, model.clock) ≈ ts.tableau.d[1] * 20
end

@testset "FrozenWeightsWENO reconstruction" begin
    using Oceananigans.Advection: materialize_advection, LeftBias, RightBias,
                                  _biased_interpolate_xᶠᵃᵃ, _biased_interpolate_zᵃᵃᶠ
    using Oceananigans.BoundaryConditions: fill_halo_regions!
    using Breeze.Advection: step_start_tracer

    grid = RectilinearGrid(CPU(); size = (16, 8), halo = (5, 5), x = (0, 1), z = (0, 1),
                           topology = (Periodic, Flat, Bounded))
    weno = materialize_advection(WENO(order=9), grid)
    frozen = materialize_advection(FrozenWeightsWENO(order=9), grid)
    upwind = materialize_advection(UpwindBiased(order=9), grid)
    @test frozen isa FrozenWeightsWENO
    @test sprint(show, FrozenWeightsWENO(order=9)) ==
        "FrozenWeightsWENO(order=9)\n└── weno: WENO{5, Float64, Nothing}(order=9)"
    @test_throws ArgumentError FrozenWeightsWENO(WENO(order=5, bounds=(0, 1)))

    ψ = CenterField(grid)
    set!(ψ, (x, z) -> sin(2π * x) + 0.3 * (z > 0.5) + 0.1 * cos(6π * x) * z)
    fill_halo_regions!(ψ)
    reference = step_start_tracer(frozen)

    # With the step-start copy equal to the reconstructed field the scheme is WENO, in the interior and
    # through the lower-order buffer schemes near the bounded top and bottom.
    parent(reference) .= parent(ψ)
    @test @inferred(_biased_interpolate_xᶠᵃᵃ(5, 1, 4, grid, frozen, LeftBias, ψ)) isa Float64
    for i in 1:16, k in 1:8, bias in (LeftBias, RightBias)
        @test _biased_interpolate_xᶠᵃᵃ(i, 1, k, grid, frozen, bias, ψ) ≈ _biased_interpolate_xᶠᵃᵃ(i, 1, k, grid, weno, bias, ψ) atol=1e-14
        @test _biased_interpolate_zᵃᵃᶠ(i, 1, k, grid, frozen, bias, ψ) ≈ _biased_interpolate_zᵃᵃᶠ(i, 1, k, grid, weno, bias, ψ) atol=1e-14
    end

    # A uniform step-start copy makes every smoothness indicator vanish, so the weights are the
    # optimal ones and the reconstruction of the (non-smooth) ψ is the linear upwind-biased one.
    fill!(parent(reference), 1)
    for i in 1:16, k in 1:8, bias in (LeftBias, RightBias)
        @test _biased_interpolate_xᶠᵃᵃ(i, 1, k, grid, frozen, bias, ψ) ≈ _biased_interpolate_xᶠᵃᵃ(i, 1, k, grid, upwind, bias, ψ) atol=1e-12
    end
end

@testset "FrozenWeightsWENO in AtmosphereModel" begin
    using Breeze.Advection: step_start_tracer
    using Breeze.AtmosphereModels: moisture_prognostic_name, specific_prognostic_moisture
    using Breeze.TimeSteppers: tendencies_feed_first_stage

    qᵗᵢ(x, z) = 0.012 * exp(-z / 3e3) * (1 + 0.2 * (abs(x) < 3e3))
    grid = RectilinearGrid(default_arch; size = (16, 8), halo = (5, 5), x = (-10e3, 10e3), z = (0, 10e3),
                           topology = (Bounded, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5, standard_pressure = 1e5,
                                    reference_potential_temperature = 300)
    @test_throws ArgumentError AtmosphereModel(grid; dynamics, advection = FrozenWeightsWENO(order=5))

    # Every tendency evaluation of a Runge–Kutta stepper feeds its own stage, so the copy is refreshed
    # each time and the scheme tracks WENO.
    reference = mis_bubble_model(; timestepper = :AcousticRungeKutta3, moist = true, qᵗ = qᵗᵢ)
    frozen = mis_bubble_model(; timestepper = :AcousticRungeKutta3, moist = true, qᵗ = qᵗᵢ,
                              scalar_advection = FrozenWeightsWENO(order=5))
    moist_name = moisture_prognostic_name(frozen.microphysics)
    @test frozen.advection[moist_name] isa FrozenWeightsWENO
    @test tendencies_feed_first_stage(frozen.timestepper)
    for _ in 1:5
        time_step!(reference, 5)
        time_step!(frozen, 5)
    end
    for name in keys(prognostic_fields(frozen))
        a = Array(interior(prognostic_fields(frozen)[name]))
        b = Array(interior(prognostic_fields(reference)[name]))
        @test maximum(abs, a .- b) <= 1e-12 * maximum(abs, b)
    end

    # Under MIS the stage-1 tendencies are those of WENO, the copy holds the step-start tracer for
    # the whole step (so later stages see frozen weights and the trajectory departs from WENO), and
    # it is refreshed by the tendency evaluation that ends the step.
    reference = mis_bubble_model(; timestepper = :MultirateInfinitesimalStep, coefficients = :MIS3C, moist = true, qᵗ = qᵗᵢ)
    frozen = mis_bubble_model(; timestepper = :MultirateInfinitesimalStep, coefficients = :MIS3C, moist = true, qᵗ = qᵗᵢ,
                              scalar_advection = FrozenWeightsWENO(order=5))
    update_state!(reference; compute_tendencies = true)
    update_state!(frozen; compute_tendencies = true)
    Ga = Array(interior(frozen.timestepper.Gⁿ[moist_name]))
    Gb = Array(interior(reference.timestepper.Gⁿ[moist_name]))
    @test maximum(abs, Ga .- Gb) <= 1e-12 * maximum(abs, Gb)

    frozen_q = step_start_tracer(frozen.advection[moist_name])
    q = specific_prognostic_moisture(frozen)
    @test Array(interior(frozen_q)) == Array(interior(q))

    for _ in 1:5
        time_step!(reference, 5)
        time_step!(frozen, 5)
    end
    @test tendencies_feed_first_stage(frozen.timestepper)
    @test Array(interior(frozen_q)) == Array(interior(q))
    a = Array(interior(prognostic_fields(frozen)[moist_name]))
    b = Array(interior(prognostic_fields(reference)[moist_name]))
    @test all(isfinite, a)
    @test maximum(abs, a .- b) > 0
end

#####
##### Order of accuracy with moisture
#####
##### The acoustic loop applies the stage-entry pressure gradient and buoyancy at weight one. The
##### part its linear operator does not carry — here the dependence of pressure on vapor — is slow
##### and has to be recombined across stages (`StageEntryForceRemainder`); left at weight one it is
##### integrated at first order by every non-diagonal tableau.
#####

using Oceananigans.BoundaryConditions: fill_halo_regions!

# A small vapor anomaly carried by a sheared wind over a resting, exactly balanced neutral
# atmosphere: linear, so the reference of any consistent scheme is the same, and the only thing
# that moves the air is the pressure the vapor adds. The substep size is the same for every Δt,
# no damping and a centered vertical solve, so the acoustic loop's own error does not scale with Δt.
function vapor_anomaly_density(; timestepper, coefficients = nothing, Δt, stop_time = 8, substep = 1/1024, N = 16)
    L = 200N
    grid = RectilinearGrid(default_arch, Float64; size = (N, N), halo = (5, 5), x = (0, L), z = (0, L),
                           topology = (Periodic, Flat, Bounded))
    time_discretization = SplitExplicitTimeDiscretization(; substeps = round(Int, Δt / substep),
                                                          forward_weight = 0.5, damping = NoDivergenceDamping())
    dynamics = CompressibleDynamics(time_discretization; base_pressure = 1e5, standard_pressure = 1e5,
                                    reference_potential_temperature = 300)
    timestepper_kwargs = isnothing(coefficients) ? NamedTuple() : (; coefficients)
    model = AtmosphereModel(grid; dynamics, advection = UpwindBiased(order = 5), timestepper, timestepper_kwargs)

    # The discrete reference state, exactly, as in the resting-atmosphere test.
    reference = model.dynamics.reference_state
    Rᵈ = Breeze.dry_air_gas_constant(model.thermodynamic_constants)
    ρ = model.dynamics.dry_density
    ρθ = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
    parent(ρ) .= parent(reference.density)
    parent(ρθ) .= parent(reference.pressure) ./ (Rᵈ .* parent(reference.exner_function))

    qᵛ = CenterField(grid)
    u = XFaceField(grid)
    set!(qᵛ, (x, z) -> 1e-5 * cos(π * min(1, sqrt((x - L / 2)^2 + (z - L / 2)^2) / (3L / 16)) / 2)^2)
    set!(u, (x, z) -> 10 + 10 * (z - L / 2) / L)
    ρqᵛ = prognostic_fields(model).ρqᵛ
    interior(ρqᵛ) .= interior(ρ) .* interior(qᵛ)
    interior(model.momentum.ρu) .= interior(ρ) .* interior(u)
    fill_halo_regions!((ρqᵛ, model.momentum.ρu))
    update_state!(model)

    for _ in 1:round(Int, stop_time / Δt)
        time_step!(model, Δt)
    end

    return Array(interior(ρ))
end

@testset "MIS order of accuracy with moisture [$coefficients]" for (coefficients, Δts, order) in
        ((:TVDMISB, (1/16, 1/32, 1/64), 1.8), (:SSPMIS43, (1/8, 1/16, 1/32), 1.8), (:MIS4a, (1/2, 1/4, 1/8), 2.7))
    reference = vapor_anomaly_density(; timestepper = :AcousticRungeKutta3, Δt = 1/256)
    errors = map(Δts) do Δt
        ρ = vapor_anomaly_density(; timestepper = :MultirateInfinitesimalStep, coefficients, Δt)
        maximum(abs, ρ .- reference)
    end
    @test all(isfinite, errors)
    @test log2(errors[1] / errors[3]) / log2(Δts[1] / Δts[3]) ≥ order
end

@testset "MIS falls back where the stage-entry force remainder cannot be recombined" begin
    # A diagonal tableau has nothing to recombine and holds no step-start state.
    diagonal = mis_bubble_model(; timestepper = :MultirateInfinitesimalStep, coefficients = :WickerSkamarock3)
    @test isnothing(diagonal.timestepper.step_start)
    recombined = mis_bubble_model(; timestepper = :MultirateInfinitesimalStep, coefficients = :TVDMISB)
    @test !isnothing(recombined.timestepper.step_start)

    # Re-linearizing per stage leaves no single fast operator to define the remainder against.
    grid = RectilinearGrid(default_arch; size = (8, 8), halo = (5, 5), x = (-10e3, 10e3), z = (0, 10e3),
                           topology = (Bounded, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    timestepper_kwargs = (; coefficients = :TVDMISB, relinearize_stages = true)
    relinearized = @test_logs (:warn, r"first-order accurate") match_mode=:any AtmosphereModel(grid; dynamics,
        timestepper = :MultirateInfinitesimalStep, timestepper_kwargs)
    @test isnothing(relinearized.timestepper.step_start)
end

@testset "The default MIS coefficients" begin
    grid = RectilinearGrid(default_arch; size = (8, 8), halo = (5, 5), x = (-10e3, 10e3), z = (0, 10e3),
                           topology = (Bounded, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, timestepper = :MultirateInfinitesimalStep)
    @test model.timestepper.tableau.β == multirate_infinitesimal_step_tableau(:MIS4a, eltype(grid)).β
end
