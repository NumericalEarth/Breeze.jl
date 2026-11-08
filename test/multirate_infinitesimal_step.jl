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

function mis_bubble_model(; timestepper, coefficients = :MIS2, moist = false, Δθ = 2)
    grid = RectilinearGrid(default_arch; size = (16, 8), halo = (5, 5),
                           x = (-10e3, 10e3), z = (0, 10e3),
                           topology = (Bounded, Flat, Bounded))
    θ₀ = 300
    p₀ = 1e5
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    base_pressure = p₀, standard_pressure = p₀,
                                    reference_potential_temperature = θ₀)
    microphysics = moist ? SaturationAdjustment(; equilibrium = WarmPhaseEquilibrium()) : nothing
    timestepper_kwargs = timestepper === :MultirateInfinitesimalStep ? (; coefficients) : NamedTuple()
    model = AtmosphereModel(grid; dynamics, microphysics, advection = WENO(order=5),
                            timestepper, timestepper_kwargs)
    constants = model.thermodynamic_constants
    θᵢ(x, z) = θ₀ + Δθ * cos(π * min(1, sqrt((x / 2e3)^2 + ((z - 2e3) / 2e3)^2)) / 2)^2
    ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, p₀, constants)
    moist ? set!(model; ρ = ρᵢ, θ = θᵢ, qᵗ = 0.012) : set!(model; ρ = ρᵢ, θ = θᵢ)
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
