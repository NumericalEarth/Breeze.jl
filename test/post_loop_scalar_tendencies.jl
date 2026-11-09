include(joinpath(@__DIR__, "setup.jl"))

#####
##### Post-loop scalar tendencies (issue #1089, cause 2): with `post_loop_scalar_tendencies = true`
##### the moisture and tracer tendencies are built after each stage's own acoustic loop, with that
##### loop's time-averaged transport velocity, so their mass fluxes are q/(1 − q) times the dry-mass
##### flux continuity applied and a uniform mass fraction stays uniform to round-off.
#####

using Breeze
using Breeze.AtmosphereModels: specific_prognostic_moisture, total_density, transport_velocities,
                               compute_scalar_tendencies!, compute_scalar_tendency!, SplitScalarTendencies,
                               tracer_density_to_specific!, tracer_specific_to_density!
using Breeze.TimeSteppers: PreLoopScalarTendencies, PostLoopScalarTendencies, scalar_prognostic_names,
                           fast_tendency_scratch
using Oceananigans
using Oceananigans: TendencyCallsite, fields
using Statistics: mean
using Test

const q₀ = 0.01
const c₀ = 1e-3
θ_bubble(x, z) = 300 + 2 * cos(π * min(1, sqrt((x / 2e3)^2 + ((z - 2e3) / 2e3)^2)) / 2)^2

# The issue's reproducer: a dry 2 K bubble, uniform vapor, no microphysics, plus a uniform tracer.
function uniform_scalar_bubble(; timestepper, coefficients = nothing, post_loop_scalar_tendencies)
    grid = RectilinearGrid(default_arch; size = (40, 48), halo = (5, 5), x = (-20e3, 20e3), z = (0, 12e3),
                           topology = (Periodic, Flat, Bounded))
    time_discretization = SplitExplicitTimeDiscretization(; damping = NoDivergenceDamping(), forward_weight = 0.65)
    dynamics = CompressibleDynamics(time_discretization; base_pressure = 1e5, reference_potential_temperature = 300)
    timestepper_kwargs = (; post_loop_scalar_tendencies)
    isnothing(coefficients) || (timestepper_kwargs = merge(timestepper_kwargs, (; coefficients)))
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 5), tracers = :ρc,
                            timestepper, timestepper_kwargs)
    set!(model; θ = θ_bubble, qᵛ = q₀, ρ = HydrostaticallyBalancedDensity())
    parent(model.tracers.ρc) .= c₀ .* parent(total_density(model.dynamics))
    return model
end

moisture_uniformity_error(model) = maximum(abs, interior(specific_prognostic_moisture(model)) .- q₀) / q₀

function tracer_uniformity_error(model)
    c = Array(interior(model.tracers.ρc)) ./ Array(interior(total_density(model.dynamics)))
    return maximum(abs, c .- c₀) / c₀
end

const steppers = ((:AcousticRungeKutta3, nothing), (:MultirateInfinitesimalStep, :MIS3C), (:MultirateInfinitesimalStep, :TVDMISB))

@testset "Post-loop scalar tendencies keep a uniform mass fraction uniform [$timestepper, $coefficients]" for (timestepper, coefficients) in steppers
    Δt = 10
    Nt = 60 # 10 minutes: the bubble reaches max|w| ≈ 10 m/s

    pre_loop = uniform_scalar_bubble(; timestepper, coefficients, post_loop_scalar_tendencies = false)
    @test pre_loop.timestepper.scalar_tendencies isa PreLoopScalarTendencies
    for _ in 1:Nt
        time_step!(pre_loop, Δt)
    end
    # Advecting the scalars with the preceding loop's mean velocity leaves a spurious gradient.
    @test moisture_uniformity_error(pre_loop) > 1e-6
    @test tracer_uniformity_error(pre_loop) > 1e-6

    post_loop = uniform_scalar_bubble(; timestepper, coefficients, post_loop_scalar_tendencies = true)
    @test post_loop.timestepper.scalar_tendencies isa PostLoopScalarTendencies
    for _ in 1:Nt
        time_step!(post_loop, Δt)
    end
    @test maximum(abs, interior(post_loop.velocities.w)) > 5
    @test moisture_uniformity_error(post_loop) < 1e-12
    @test tracer_uniformity_error(post_loop) < 1e-12
end

@testset "TendencyCallsite callbacks edit the post-loop scalar tendencies once per stage [$timestepper]" for (timestepper, coefficients) in steppers[1:2]
    # A horizontally uniform rest state with a uniform tendency added by a callback: after one step
    # ρu and ρc each equal c Δt. Momentum takes the callback's pre-loop edit; the post-loop build
    # overwrites the scalar edit, so the callback fires again and ρc takes exactly one edit.
    grid = RectilinearGrid(default_arch; size = (4, 4, 4), x = (0, 1000), y = (0, 1000), z = (0, 1000),
                           topology = (Periodic, Periodic, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    timestepper_kwargs = (; post_loop_scalar_tendencies = true)
    isnothing(coefficients) || (timestepper_kwargs = merge(timestepper_kwargs, (; coefficients)))
    model = AtmosphereModel(grid; dynamics, tracers = :ρc, timestepper, timestepper_kwargs)
    set!(model; ρ = model.dynamics.reference_state.density, θ = 300)

    c = 1e-3
    Δt = 1
    function add_tendency!(m)
        interior(m.timestepper.Gⁿ.ρu) .+= c
        interior(m.timestepper.Gⁿ.ρc) .+= c
        return nothing
    end
    callbacks = [Callback(add_tendency!, IterationInterval(1); callsite = TendencyCallsite())]
    time_step!(model, Δt; callbacks)

    @test isapprox(maximum(model.momentum.ρu), c * Δt)
    @test isapprox(minimum(model.momentum.ρu), c * Δt)
    @test isapprox(maximum(model.tracers.ρc), c * Δt)
    @test isapprox(minimum(model.tracers.ρc), c * Δt)
end

@testset "Closure scalar sources survive the post-loop build" begin
    # A neutral sheared column: TKE grows by shear production, a closure source evaluated at the
    # stage-entry state before the loop. The post-loop build must add to it, not overwrite it, so
    # the two modes agree to round-off in a column, where the scalars have no advective tendency.
    grid = RectilinearGrid(default_arch; size = 16, z = (0, 1000), topology = (Flat, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    closure = TKEBasedTurbulenceClosure()
    mean_tke = Dict{Bool, Float64}()
    for post_loop_scalar_tendencies in (false, true)
        model = AtmosphereModel(grid; dynamics, closure, timestepper = :AcousticRungeKutta3,
                                timestepper_kwargs = (; post_loop_scalar_tendencies))
        set!(model; ρ = model.dynamics.reference_state.density, θ = 300, u = z -> 0.02z, ρe = 1e-3)
        for _ in 1:20
            time_step!(model, 2)
        end
        mean_tke[post_loop_scalar_tendencies] = mean(interior(model.tracers.ρe))
    end
    @test mean_tke[false] > 2e-3 # production is active
    @test isapprox(mean_tke[true], mean_tke[false]; rtol = 1e-10)
end

@testset "The split post-loop build sums to the full scalar tendency" begin
    # A multirate infinitesimal step builds each post-loop scalar tendency in two parts from one
    # evaluation — the advective part into its own scratch, the rest added to `Gⁿ` — instead of
    # recomputing the advection to separate them. The parts must sum to the full tendency to
    # round-off, with a closure, a forcing and a non-uniform tracer so that both parts are nonzero.
    grid = RectilinearGrid(default_arch; size = (8, 8), halo = (5, 5), x = (-20e3, 20e3), z = (0, 8e3),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    ρc_forcing = Forcing((x, z, t) -> 1e-6 * exp(-(x / 5e3)^2))
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 5), tracers = :ρc,
                            closure = ScalarDiffusivity(ν = 10, κ = 10), forcing = (; ρc = ρc_forcing),
                            timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients = :MIS3C, post_loop_scalar_tendencies = true))
    set!(model; θ = θ_bubble, qᵛ = (x, z) -> q₀ * exp(-((x / 4e3)^2 + (z / 2e3)^2)),
         ρ = HydrostaticallyBalancedDensity(), u = 5)
    set!(model.tracers.ρc, (x, z) -> c₀ * exp(-((x / 3e3)^2 + ((z - 2e3) / 2e3)^2)))
    parent(model.tracers.ρc) .*= parent(total_density(model.dynamics))
    time_step!(model, 1)

    ts = model.timestepper
    names = scalar_prognostic_names(model)
    @test ts.fast_tendency isa NamedTuple
    @test all(name -> haskey(ts.fast_tendency, name), names)
    pre_loop = AtmosphereModel(grid; dynamics, timestepper = :MultirateInfinitesimalStep).timestepper
    @test pre_loop.fast_tendency isa Field

    tracer_density_to_specific!(model)
    velocities = transport_velocities(model)
    compute_scalar_tendencies!(model, fields(model), velocities, compute_scalar_tendency!)
    full = Dict(name => Array(interior(ts.Gⁿ[name])) for name in names)

    for name in names
        fill!(parent(ts.Gⁿ[name]), 0)
    end
    compute_scalar_tendencies!(model, fields(model), velocities, SplitScalarTendencies(ts.fast_tendency))
    tracer_specific_to_density!(model)

    for name in names
        slow = Array(interior(ts.Gⁿ[name]))
        fast = Array(interior(fast_tendency_scratch(ts, name)))
        scale = maximum(abs, full[name])
        @test scale > 0
        @test maximum(abs, fast) > 0
        @test maximum(abs, slow) > 0
        @test maximum(abs, slow .+ fast .- full[name]) < 1e-13 * scale
    end
end
