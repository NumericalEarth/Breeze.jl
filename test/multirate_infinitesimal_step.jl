include(joinpath(@__DIR__, "setup.jl"))

#####
##### MultirateInfinitesimalStep: the coefficient tables, the reduction to
##### AcousticRungeKutta3 under the Wicker–Skamarock tableau, rest-state
##### preservation, and a short thermal-bubble integration for every set.
#####

using Breeze
using Breeze.AtmosphereModels: stage_time_step, moisture_prognostic_name
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticlePropertiesMicrophysics
using Breeze.CompressibleEquations: freeze_linearization_state!, compute_pressure_remainder!,
                                    StageEntryForceRemainder, linearized_pressure_remainder,
                                    pressure_change, moisture_content_arguments,
                                    horizontal_substep_form, ForwardBackwardSubsteps, SymmetricSubsteps
using Breeze.TimeSteppers: MultirateInfinitesimalStepTableau, multirate_infinitesimal_step_tableau, number_of_stages,
                           store_step_start_state!, stage_storage, structured_stage_storage,
                           TwoRegisterStorage, AccumulatorStorage
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization, adaptive_advection_timestep
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!
using CloudMicrophysics: CloudMicrophysics
using LinearAlgebra: isdiag, istril, istriu
using Test

const OneMomentCloudMicrophysics = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt).OneMomentCloudMicrophysics

# Every coefficient set repeats the integrations below with a time stepper of its own. By default
# we test one set per storage layout plus the default; BREEZE_TEST_EXHAUSTIVE runs all of them.
const all_mis_coefficient_sets = (:WickerSkamarock3, :TVDMISB, :MIS4a, :SSPMIS43, :LowStorageMIS52,
                                  :ConvexLinearMIS3, :ConvexLinearMIS4, :ConvexLinearMIS5, :ConvexLinearMIS6)
const mis_coefficient_sets = exhaustive_tests() ? all_mis_coefficient_sets :
                                                  (:WickerSkamarock3, :SSPMIS43, :ConvexLinearMIS6)
const recombining_mis_coefficient_sets = filter(!=(:WickerSkamarock3), mis_coefficient_sets)

# `coefficients = nothing` leaves the time stepper its default set.
function mis_bubble_model(; timestepper = :MultirateInfinitesimalStep, coefficients = :SSPMIS43, moist = false, Δθ = 2, qᵗ = 0.012,
                          momentum_advection = WENO(order=5), scalar_advection = WENO(order=5),
                          post_loop_scalar_tendencies = nothing, timestepper_kwargs = NamedTuple())
    grid = RectilinearGrid(default_arch; size = (16, 8), halo = (5, 5),
                           x = (-10e3, 10e3), z = (0, 10e3),
                           topology = (Bounded, Flat, Bounded))
    θ₀ = 300
    p₀ = 1e5
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    base_pressure = p₀, standard_pressure = p₀,
                                    reference_potential_temperature = θ₀)
    microphysics = moist ? SaturationAdjustment(; equilibrium = WarmPhaseEquilibrium()) : nothing
    if timestepper === :MultirateInfinitesimalStep && !isnothing(coefficients)
        timestepper_kwargs = merge(timestepper_kwargs, (; coefficients))
    end
    if !isnothing(post_loop_scalar_tendencies)
        timestepper_kwargs = merge(timestepper_kwargs, (; post_loop_scalar_tendencies))
    end
    model = AtmosphereModel(grid; dynamics, microphysics, momentum_advection, scalar_advection,
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
    # The two build their scalar tendencies at different times by default, so each timing is
    # requested of both.
    for moist in (false, true), post_loop_scalar_tendencies in (false, true)
        reference = mis_bubble_model(; timestepper = :AcousticRungeKutta3, moist, post_loop_scalar_tendencies)
        mis = mis_bubble_model(; coefficients = :WickerSkamarock3, moist, post_loop_scalar_tendencies)
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
    model = mis_bubble_model(; coefficients, Δθ = 0)
    set_discrete_reference_state!(model)
    update_state!(model)

    for _ in 1:10
        time_step!(model, 5)
    end
    @test maximum(abs, interior(model.velocities.w)) < 1e-10
    @test maximum(abs, interior(model.velocities.u)) < 1e-10
end

@testset "MIS thermal bubble [$coefficients, moist = $moist]" for coefficients in mis_coefficient_sets, moist in (false, true)
    model = mis_bubble_model(; coefficients, moist)
    for _ in 1:20
        time_step!(model, 5)
    end
    w = Array(interior(model.velocities.w))
    @test all(isfinite, w)
    @test maximum(w) > 0.1
    @test model.clock.time ≈ 100
end

# The saturated column of `acoustic_substepping_stability.jl`: a seeded 2Δx mode, undamped at Δt = 10 s, under
# saturation adjustment. With the moisture partition held in the loop's pressure the mode grew to several m/s.
@testset "MIS keeps a seeded 2Δx mode in saturated air at its amplitude [$label]" for (label, qᵗ) in
        (("saturated throughout", 0.02), ("cloud layer", (x, z) -> ifelse(1500 < z < 4000, 0.02, 0.0)))

    Nx, Nz, Δx = 32, 40, 3e3
    θ̄(z) = 280 + 0.01z
    θᵢ(x, z) = θ̄(z) * (1 + 1e-5 * cos(π * round(Int, x / Δx - 0.5)))

    grid = RectilinearGrid(default_arch; size = (Nx, Nz), halo = (5, 5), x = (0, Nx * Δx), z = (0, 10e3),
                           topology = (Periodic, Flat, Bounded))
    td = SplitExplicitTimeDiscretization(; damping = NoDivergenceDamping())
    dynamics = CompressibleDynamics(td; base_pressure = 1e5, reference_potential_temperature = θ̄)
    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    model = AtmosphereModel(grid; dynamics, microphysics, advection = WENO(order = 5),
                            timestepper = :MultirateInfinitesimalStep)
    set!(model; θ = θᵢ, u = 15, ρ = Breeze.HydrostaticallyBalancedDensity())
    set!(model; qᵗ)

    for _ in 1:360
        time_step!(model, 10)
    end

    # rms over k = 2:16, which spans the base of the cloud layer, of the (−1)ⁱ projection of w
    w = Array(interior(model.velocities.w))
    checkerboard_amplitude = sqrt(sum(k -> (sum((-1)^i * w[i, 1, k] for i in 1:Nx) / Nx)^2, 2:16) / 15)
    @test checkerboard_amplitude < 1e-4

    qˡ = Array(interior(model.microphysical_fields.qˡ))
    @test minimum(qˡ[:, 1, 8:15]) > 0
end

# The force remainder MIS recombines as slow (`StageEntryForceRemainder`) must vanish to first order in a
# 10⁻⁶ change of any density, under saturation adjustment with and without precipitation and with carried cloud.
@testset "MIS force remainder vanishes to first order [$label]" for (label, cloud_formation, condensates) in
        (("warm-phase saturation adjustment", :warm, (;)),
         ("one-moment, mixed-phase saturation adjustment", :mixed, (; ρqʳ = 2e-3, ρqˢⁿ = 1e-3)),
         ("one-moment, non-equilibrium cloud", :non_equilibrium, (; ρqᶜˡ = 3e-3, ρqʳ = 2e-3)))

    FT = Float64
    temperature_solver = Breeze.NewtonSolver(FT; reltol=0, abstol=1e-11, maxiter=50)
    solver = Breeze.SecantSolver(FT; reltol=0, abstol=1e-11, maxiter=200)

    microphysics = if cloud_formation == :warm
        SaturationAdjustment(FT; equilibrium = WarmPhaseEquilibrium(), solver)
    elseif cloud_formation == :mixed
        OneMomentCloudMicrophysics(FT; cloud_formation = SaturationAdjustment(FT; equilibrium = MixedPhaseEquilibrium(FT), solver))
    else
        OneMomentCloudMicrophysics(FT)
    end

    grid = RectilinearGrid(default_arch, FT; size=6, z=(0, 600), topology=(Flat, Flat, Bounded))
    formulation = LiquidIcePotentialTemperatureFormulation(; temperature_solver)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); standard_pressure=1e5)

    # Warm cloud below 200 m, unsaturated air to 400 m, and cloud near 260 K above.
    ρᵈ(z) = z < 400 ? 1.1 : 0.8
    θ(z) = z < 200 ? 285.0 : z < 400 ? 300.0 : 295.0
    ρqᵐ(z) = z < 200 ? 0.02 : z < 400 ? 0.005 : 0.004
    moisture_name = moisture_prognostic_name(microphysics)
    names = (:ρθ, moisture_name, :ρᵈ, keys(condensates)...)

    for perturbed in (map(tuple, names)..., names)
        model = AtmosphereModel(grid; formulation, dynamics, microphysics, timestepper = :MultirateInfinitesimalStep,
                                timestepper_kwargs = (; coefficients = :SSPMIS43)) # β with a recombined remainder
        set!(model; ρᵈ, ρθ = z -> ρᵈ(z) * θ(z), NamedTuple{(moisture_name,)}((ρqᵐ,))..., condensates...)
        update_state!(model)

        timestepper = model.timestepper
        substepper = timestepper.substepper
        U = prognostic_fields(model)

        for name in keys(U)
            parent(timestepper.Uⁿ[name]) .= parent(U[name])
        end

        store_step_start_state!(timestepper.step_start, model)
        freeze_linearization_state!(substepper, model)
        pⁿ = Array(interior(model.dynamics.pressure))

        for name in perturbed
            parent(U[name]) .*= 1 + 1e-6
        end

        update_state!(model)
        compute_pressure_remainder!(timestepper.step_start, substepper, model, timestepper.Uⁿ)
        remainder = StageEntryForceRemainder(Val(:z), substepper, model, timestepper.step_start)
        δp = Array(interior(model.dynamics.pressure)) .- pⁿ

        for k in 1:6
            @test abs(@allowscalar linearized_pressure_remainder(1, 1, k, grid, remainder)) < 1e-4 * abs(δp[1, 1, k])
        end
    end
end

# In clear air the pressure change since the step start is evaluated from the changes of the densities
# (`clear_air_pressure_change`); it must equal the difference of the diagnosed pressures to their rounding.
@testset "The clear-air pressure change is the change of the diagnosed pressure [$(nameof(typeof(exner_exponent)))]" for
        exner_exponent in (VaporEquivalentExponent(), DryExponent())

    FT = Float64
    grid = RectilinearGrid(default_arch, FT; size=6, z=(0, 600), topology=(Flat, Flat, Bounded))
    formulation = LiquidIcePotentialTemperatureFormulation(; exner_exponent)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); standard_pressure=1e5)

    for perturbed in ((:ρθ,), (:ρqᵛ,), (:ρᵈ,), (:ρθ, :ρqᵛ, :ρᵈ))
        model = AtmosphereModel(grid; formulation, dynamics, timestepper = :MultirateInfinitesimalStep,
                                timestepper_kwargs = (; coefficients = :SSPMIS43)) # β with a recombined remainder
        set!(model; ρᵈ = z -> 1.1 - z / 3000, ρθ = z -> 330 - z / 20, ρqᵛ = z -> 0.012 - z / 1e5)
        update_state!(model)

        timestepper = model.timestepper
        step_start = timestepper.step_start
        U = prognostic_fields(model)

        for name in keys(U)
            parent(timestepper.Uⁿ[name]) .= parent(U[name])
        end

        store_step_start_state!(step_start, model)
        pⁿ = Array(interior(model.dynamics.pressure))

        for (n, name) in enumerate(perturbed)
            parent(U[name]) .*= 1 + n * 1e-3
        end

        update_state!(model)
        δp = Array(interior(model.dynamics.pressure)) .- pⁿ
        state, _ = moisture_content_arguments(timestepper.substepper, model, timestepper.Uⁿ)

        for k in 1:6
            δpᶜ = @allowscalar pressure_change(1, 1, k, grid, state, model.dynamics.pressure, step_start.pressure,
                                               step_start.condensate_mass_fraction, timestepper.Uⁿ.ρθ, U.ρθ,
                                               timestepper.Uⁿ.ρᵈ, timestepper.Uⁿ.ρqᵛ, U.ρqᵛ)
            @test abs(δp[1, 1, k]) > 1
            @test isapprox(δpᶜ, δp[1, 1, k]; rtol = 1e-10)
        end
    end
end

@testset "MIS with adaptive-implicit vertical advection" begin
    advection = WENO(order = 5, time_discretization = AdaptiveVerticallyImplicitDiscretization(; cfl = 0.5))
    model = mis_bubble_model(; coefficients = :TVDMISB, momentum_advection = advection, scalar_advection = advection)
    for _ in 1:10
        time_step!(model, 20)
    end
    ts = model.timestepper
    @test all(isfinite, Array(interior(model.velocities.w)))
    # The split is sized for the stage the next tendencies feed: stage 1 of the next step, d₁ Δt.
    @test adaptive_advection_timestep(ts, model.clock) ≈ ts.tableau.d[1] * 20
end

@testset "MIS strong-stability coefficients" begin
    coefficient(name) = multirate_infinitesimal_step_tableau(name).strong_stability_coefficient

    # Wicker–Skamarock RK3 is not a convex combination of forward-Euler steps.
    @test coefficient(:WickerSkamarock3) == 0
    @test coefficient(:MIS4a) == 0

    # TVDMISB reduces to a method with the coefficient of its design, and SSPMIS43 to the
    # coefficient it was optimized for.
    @test coefficient(:TVDMISB) ≈ 0.75 atol = 1e-6
    @test coefficient(:SSPMIS43) ≈ 1.79 atol = 1e-2

    # Forward Euler has coefficient one, and the three-stage Shu–Osher method too.
    forward_euler = MultirateInfinitesimalStepTableau(Float64, zeros(1, 1), ones(1, 1), zeros(1, 1))
    @test forward_euler.strong_stability_coefficient ≈ 1 atol = 1e-6

    α = [0 0 0; 0 0 0; 0 0 0]
    β = [1 0 0; 1//4 1//4 0; 1//6 1//6 2//3]
    shu_osher = MultirateInfinitesimalStepTableau(Float64, α, β, zeros(3, 3))
    @test shu_osher.strong_stability_coefficient ≈ 1 atol = 1e-6
end

@testset "P3 stays non-negative under strong-stability-preserving MIS [$coefficients]" for coefficients in (:TVDMISB, :SSPMIS43)
    # A moist bubble that makes cloud, rain, ice, rime and liquid on ice, with bounded
    # advection of every scalar and no repair of negative densities. Every stage of these
    # coefficient sets is a convex combination of forward-Euler steps no longer than Δt / C,
    # P3 budgets its sinks over that interval, and the bounded scheme is within its Courant
    # budget, so nothing may go negative.
    FT = Float64
    grid = RectilinearGrid(default_arch, FT; size = (32, 24), halo = (5, 5),
                           x = (-10e3, 10e3), z = (0, 12e3), topology = (Bounded, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5,
                                    standard_pressure = 1e5, reference_potential_temperature = 300)
    microphysics = PredictedParticlePropertiesMicrophysics(FT; negative_moisture_correction = nothing)
    model = AtmosphereModel(grid; dynamics, microphysics, momentum_advection = WENO(order=5),
                            scalar_advection = WENO(order=5, bounds=(0.0, 1e30)),
                            timestepper = :MultirateInfinitesimalStep, timestepper_kwargs = (; coefficients))

    constants = model.thermodynamic_constants
    θᵢ(x, z) = 300 + 3 * cos(π * min(1, sqrt((x / 2e3)^2 + ((z - 2e3) / 2e3)^2)) / 2)^2
    ρᵢ(x, z) = adiabatic_hydrostatic_density(z, 1e5, 300, 1e5, constants)
    set!(model; ρ = ρᵢ, θ = θᵢ, qᵗ = 0.012)

    Δt = 10
    μ = model.microphysical_fields
    names = (:ρqᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ)
    minima = Dict(name => Inf for name in names)
    maxima = Dict(name => -Inf for name in names)
    minimum_vapor = Inf

    for _ in 1:180
        time_step!(model, Δt)
        for name in names
            field = getproperty(μ, name)
            minima[name] = min(minima[name], minimum(field))
            maxima[name] = max(maxima[name], maximum(field))
        end
        minimum_vapor = min(minimum_vapor, minimum(model.moisture_density))
    end

    C = model.timestepper.tableau.strong_stability_coefficient
    @test stage_time_step(model.timestepper, model.clock) ≈ Δt / C

    # Every species forms, so the test exercises the mixed-phase sinks...
    @test all(maxima[name] > 0 for name in names)

    # ...and none of them, nor the vapor, is ever negative.
    @test all(minima[name] ≥ 0 for name in names)
    @test minimum_vapor > 0
end

@testset "MIS forward-Euler interval [$coefficients]" for coefficients in mis_coefficient_sets
    # Δt / C, and the whole step where there is no strong-stability coefficient to divide by.
    model = mis_bubble_model(; coefficients)
    Δt = 2
    time_step!(model, Δt)
    C = model.timestepper.tableau.strong_stability_coefficient
    @test stage_time_step(model.timestepper, model.clock) ≈ (C > 0 ? Δt / C : Δt)
end

#####
##### Order of accuracy with moisture
#####
##### The dependence of pressure on vapor is slow and recombined across stages (`StageEntryForceRemainder`);
##### left at weight one it is integrated at first order by every non-diagonal tableau.
#####

# A small vapor anomaly in a sheared wind over an exactly balanced neutral atmosphere, linear in the anomaly.
# The acoustic substep is the same for every Δt, undamped and centered, so the loop's own error does not
# scale with Δt; the forward–backward substeps of the reference are first order, so its substep is 16 times shorter.
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
    set_discrete_reference_state!(model)
    ρ = model.dynamics.dry_density

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

const vapor_anomaly_reference = vapor_anomaly_density(; timestepper = :AcousticRungeKutta3, Δt = 1/256, substep = 1/16384)

# The time steps and the order to reach, where they differ from (1/8, 1/16, 1/32) and 1.8
const order_of_accuracy_cases = Dict(:TVDMISB => ((1/16, 1/32, 1/64), 1.8), :MIS4a => ((1/2, 1/4, 1/8), 2.7))

@testset "MIS order of accuracy with moisture [$coefficients]" for coefficients in recombining_mis_coefficient_sets
    Δts, order = get(order_of_accuracy_cases, coefficients, ((1/8, 1/16, 1/32), 1.8))
    errors = map(Δts) do Δt
        ρ = vapor_anomaly_density(; timestepper = :MultirateInfinitesimalStep, coefficients, Δt)
        maximum(abs, ρ .- vapor_anomaly_reference)
    end
    @test all(isfinite, errors)
    @test log2(errors[1] / errors[3]) / log2(Δts[1] / Δts[3]) ≥ order
end

# Anelastic dynamics on a single column have no fast tendency: the step is the slow method of the
# coefficient set, Yᵢ = Uⁿ + Σⱼ (αᵢⱼ + γᵢⱼ) (Yⱼ - Uⁿ) + Δt Σⱼ βᵢⱼ S(Yⱼ₋₁), with no substepper. A
# tracer that decays at the rate of its specific value, ∂t(ρc) = -c / τ, follows that recursion
# to round-off at every level, and the exact decay to the order of the slow method.
@testset "MIS on a single column with anelastic dynamics [$coefficients, $FT]" for coefficients in (:SSPMIS43, :MIS4a, :WickerSkamarock3), FT in all_float_types()
    grid = RectilinearGrid(default_arch, FT; size = 8, z = (0, 1000), topology = (Flat, Flat, Bounded))
    τ = 100
    decay(z, t, c, τ) = -c / τ
    forcing = (; ρc = Forcing(decay, field_dependencies = :ρc, parameters = τ))
    model = AtmosphereModel(grid; tracers = :ρc, forcing, timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients))
    @test model.dynamics isa AnelasticDynamics
    @test isnothing(model.timestepper.substepper)

    (; α, β, γ) = multirate_infinitesimal_step_tableau(coefficients)
    s = size(β, 1)
    ρ = Float64.(Array(interior(model.dynamics.reference_state.density))[1, 1, :])
    function slow_method(Δt, λ)
        Y = [1.0]
        for i in 1:s
            push!(Y, 1 + sum((α[i, j] + γ[i, j]) * (Y[j+1] - 1) for j in 1:i-1; init = 0.0) + Δt * sum(β[i, j] * λ * Y[j] for j in 1:i))
        end
        return Y[end]
    end

    errors = map((10, 20)) do Δt
        set!(model.tracers.ρc, model.dynamics.reference_state.density)
        update_state!(model)
        t = model.clock.time
        time_step!(model, Δt)
        @test model.clock.time ≈ t + Δt
        c = Float64.(Array(interior(model.tracers.ρc))[1, 1, :]) ./ ρ
        @test c ≈ slow_method.(Δt, -1 ./ (τ .* ρ)) rtol = 100 * eps(FT)
        maximum(abs, c .- exp.(-Δt ./ (τ .* ρ)))
    end
    # at least second order; in Float32 the error at these steps is round-off
    FT == Float64 && @test errors[2] / errors[1] > 2^2.9
end

@testset "MIS with anelastic dynamics needs a single column" begin
    grid = RectilinearGrid(default_arch; size = (4, 4, 4), extent = (1, 1, 1))
    @test_throws ArgumentError AtmosphereModel(grid; timestepper = :MultirateInfinitesimalStep)
end

# Stages that start from combinations of earlier increments need substeps without a skew between momentum and
# pressure: forward–backward substeps amplified the 3Δx acoustic mode of the bubble below by 1.2 per step.
@testset "MIS substep form [$coefficients]" for coefficients in mis_coefficient_sets
    model = mis_bubble_model(; coefficients)
    form = coefficients === :WickerSkamarock3 ? ForwardBackwardSubsteps() : SymmetricSubsteps()
    @test horizontal_substep_form(model.timestepper) === form
    reference = mis_bubble_model(; timestepper = :AcousticRungeKutta3)
    @test horizontal_substep_form(reference.timestepper) === ForwardBackwardSubsteps()
end

@testset "MIS leaves acoustic modes bounded without divergence damping [$coefficients, Δt = $Δt]" for
        coefficients in recombining_mis_coefficient_sets, Δt in (5, 10)

    grid = RectilinearGrid(default_arch; size = (40, 48), halo = (5, 5), x = (-20e3, 20e3), z = (0, 12e3),
                           topology = (Periodic, Flat, Bounded))
    td = SplitExplicitTimeDiscretization(; damping = NoDivergenceDamping())
    dynamics = CompressibleDynamics(td; base_pressure = 1e5, reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 5),
                            timestepper = :MultirateInfinitesimalStep, timestepper_kwargs = (; coefficients))
    θᵢ(x, z) = 300 + 2 * cos(π * min(1, sqrt((x / 2e3)^2 + ((z - 2e3) / 2e3)^2)) / 2)^2
    set!(model; θ = θᵢ, qᵛ = 0.01, ρ = Breeze.HydrostaticallyBalancedDensity())

    for _ in 1:(600 ÷ Δt)
        time_step!(model, Δt)
    end

    # The bubble's own circulation after 10 minutes
    @test 9 < maximum(abs, interior(model.velocities.w)) < 12
    @test maximum(abs, interior(model.velocities.u)) < 8
end

@testset "MIS falls back where the stage-entry force remainder cannot be recombined" begin
    # A diagonal tableau has nothing to recombine and holds no step-start state.
    diagonal = mis_bubble_model(; coefficients = :WickerSkamarock3)
    @test isnothing(diagonal.timestepper.step_start)
    recombined = mis_bubble_model(; coefficients = :TVDMISB)
    @test !isnothing(recombined.timestepper.step_start)

    # Re-linearizing per stage leaves no single fast operator to define the remainder against.
    relinearized = @test_logs (:warn, r"first-order accurate") match_mode=:any mis_bubble_model(;
        coefficients = :TVDMISB, timestepper_kwargs = (; relinearize_stages = true))
    @test isnothing(relinearized.timestepper.step_start)
end

# A diagonal β recombines no slow tendency of an earlier stage: the stepper holds neither the step-start force
# state nor, with post-loop scalar tendencies, the mass fluxes that correct recombined advective tendencies.
# The sets for flux-limited transport (strong-stability coefficient `C`) combine the stage states convexly,
# each stage a convex combination of the step-start state and one forward-Euler step of length Δt / C.
@testset "MIS coefficient sets with a diagonal β [$coefficients]" for (coefficients, C, ϵ) in
        ((:LowStorageMIS52, nothing, 1e-12), (:ConvexLinearMIS3, 0.6056, 0), (:ConvexLinearMIS4, 1.2957, 0),
         (:ConvexLinearMIS5, 1.4282, 0), (:ConvexLinearMIS6, 1.7656, 0))

    tableau = multirate_infinitesimal_step_tableau(coefficients)
    @test isdiag(tableau.β)
    @test all(tableau.β .≥ 0)
    @test istriu(tableau.α, -1)
    @test istriu(tableau.γ, -1)

    # Convex stage states, so the slow method is strong-stability preserving
    stage_weights = tableau.α .+ tableau.γ
    @test all(stage_weights .≥ -ϵ)
    @test all(sum(stage_weights, dims = 2) .≤ 1 + ϵ)

    if isnothing(C)
        @test all(tableau.α .≥ 0)
        @test all(sum(tableau.α, dims = 2) .≤ 1 + 1e-12)
        @test tableau.strong_stability_coefficient > 1.9
    else
        s = number_of_stages(tableau)
        forward_euler_steps = [tableau.β[1, 1]; [tableau.β[i, i] / stage_weights[i, i-1] for i in 2:s]]
        @test maximum(forward_euler_steps) ≈ 1 / C atol = 1e-3
        @test tableau.strong_stability_coefficient ≈ C atol = 1e-3
    end

    model = mis_bubble_model(; coefficients, post_loop_scalar_tendencies = true)
    @test isnothing(model.timestepper.step_start)
    @test isnothing(model.timestepper.transport_mass_fluxes)
end

@testset "The default MIS coefficients" begin
    model = mis_bubble_model(; coefficients = nothing)
    @test model.timestepper.tableau.β == multirate_infinitesimal_step_tableau(:ConvexLinearMIS6, eltype(model.grid)).β
end

#####
##### Storage layouts: chosen from the structure of the coefficients, and the same step to round-off
#####

@testset "MIS storage layouts" begin
    layout(coefficients, FT = Float64) = stage_storage(multirate_infinitesimal_step_tableau(coefficients, FT))

    for FT in (Float32, Float64)
        @test layout(:WickerSkamarock3, FT) isa TwoRegisterStorage
        @test layout(:LowStorageMIS52, FT) isa TwoRegisterStorage
        for coefficients in (:ConvexLinearMIS3, :ConvexLinearMIS4, :ConvexLinearMIS5, :ConvexLinearMIS6)
            @test layout(coefficients, FT) isa TwoRegisterStorage
        end
        @test layout(:SSPMIS43, FT) isa AccumulatorStorage
        @test layout(:MIS4a, FT) isa AccumulatorStorage
        @test layout(:TVDMISB, FT) isa AccumulatorStorage
    end

    # The round-off of a zero coefficient does not change the layout, and a coefficient does
    low_storage = multirate_infinitesimal_step_tableau(:LowStorageMIS52)
    (; α, β, γ) = low_storage
    perturbed(A, i, j, δ) = (B = copy(A); B[i, j] += δ; B)
    @test stage_storage(MultirateInfinitesimalStepTableau(Float64, α, perturbed(β, 3, 1, 1e-17), γ)) isa TwoRegisterStorage
    @test stage_storage(MultirateInfinitesimalStepTableau(Float64, perturbed(α, 4, 1, 1e-17), β, γ)) isa TwoRegisterStorage
    @test stage_storage(MultirateInfinitesimalStepTableau(Float64, α, perturbed(β, 3, 1, 1e-3), γ)) isa AccumulatorStorage

    # A diagonal β with α or γ off the first subdiagonal takes accumulators
    wicker_skamarock = multirate_infinitesimal_step_tableau(:WickerSkamarock3)
    off_subdiagonal = perturbed(wicker_skamarock.α, 3, 1, 1e-3)
    @test stage_storage(MultirateInfinitesimalStepTableau(Float64, off_subdiagonal, wicker_skamarock.β, wicker_skamarock.γ)) isa AccumulatorStorage
    @test stage_storage(MultirateInfinitesimalStepTableau(Float64, perturbed(α, 4, 1, 1e-3), β, γ)) isa AccumulatorStorage

    structured_stage_storage[] = false
    @test layout(:LowStorageMIS52) isa AccumulatorStorage
    structured_stage_storage[] = true

    # Two registers: the step-start state and the tendencies, and nothing else per prognostic
    for coefficients in (:WickerSkamarock3, :LowStorageMIS52), post_loop_scalar_tendencies in (false, true)
        model = mis_bubble_model(; coefficients, moist = true, post_loop_scalar_tendencies)
        ts = model.timestepper
        @test ts.Uⁿ === ts.U⁰
        @test isempty(ts.accumulators)
        @test isempty(ts.transport_density_accumulators)
        @test isnothing(ts.step_start)
        @test isnothing(ts.transport_mass_fluxes)
    end
end

function storage_layout_state(; coefficients, structured, FT, topology, post_loop_scalar_tendencies, Nt = 3)
    Oceananigans.defaults.FloatType = FT
    structured_stage_storage[] = structured
    grid = RectilinearGrid(default_arch; size = (16, 12), halo = (5, 5), x = (-8e3, 8e3), z = (0, 8e3), topology)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5, standard_pressure = 1e5,
                                    reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 5), tracers = :ρc,
                            microphysics = SaturationAdjustment(; equilibrium = WarmPhaseEquilibrium()),
                            timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies))
    structured_stage_storage[] = true
    Oceananigans.defaults.FloatType = Float64

    constants = model.thermodynamic_constants
    bubble(x, z) = cos(π * min(1, sqrt((x / 2e3)^2 + ((z - 2e3) / 2e3)^2)) / 2)^2
    set!(model; θ = (x, z) -> 300 + 2 * bubble(x, z), qᵗ = (x, z) -> 0.004 + 0.014 * bubble(x, z),
                ρ = (x, z) -> adiabatic_hydrostatic_density(z, 1e5, 300, 1e5, constants),
                u = (x, z) -> 10 * sin(2π * z / 8e3), ρc = (x, z) -> exp(-((z - 3e3) / 1e3)^2 - (x / 2e3)^2))
    for _ in 1:Nt
        time_step!(model, 6)
    end
    return map(field -> Array(interior(field)), prognostic_fields(model))
end

# Single precision with the stage-state round-off in the substepper's perturbation fields (periodic), double
# with it in fields of their own (bounded). The layouts differ in round-off alone, which three steps of this
# bubble amplify to a few hundred ulps of the momentum, and to about 1500 on some CI platforms in single precision.
@testset "MIS storage layouts reproduce the accumulators [$coefficients, $FT, post_loop = $post_loop_scalar_tendencies, $(topology[1])]" for
        coefficients in (:LowStorageMIS52,), post_loop_scalar_tendencies in (false, true),
        (FT, topology) in ((Float32, (Periodic, Flat, Bounded)), (Float64, (Bounded, Flat, Bounded)))

    kw = (; coefficients, FT, topology, post_loop_scalar_tendencies)
    accumulated = storage_layout_state(; structured = false, kw...)
    structured = storage_layout_state(; structured = true, kw...)
    @test stage_storage(multirate_infinitesimal_step_tableau(coefficients)) isa TwoRegisterStorage

    for name in keys(accumulated)
        scale = maximum(abs, accumulated[name])
        @test maximum(abs, structured[name] .- accumulated[name]) ≤ 4000 * eps(scale)
    end
end
