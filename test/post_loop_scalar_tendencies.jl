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
                               tracer_density_to_specific!, tracer_specific_to_density!, SpeciesBorrowing
using Breeze.TimeSteppers: PreLoopScalarTendencies, PostLoopScalarTendencies, scalar_prognostic_names,
                           fast_tendency_scratch, fold_stage_increments, structured_stage_storage,
                           stage_increment_source, acoustic_prognostic_names, IncrementFromState, StoredIncrement,
                           TwoRegisterStorage, AccumulatorStorage
using Oceananigans
using Oceananigans: TendencyCallsite, fields, prognostic_fields
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!
using CloudMicrophysics
using Statistics: mean
using Test

const OneMomentCloudMicrophysics = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt).OneMomentCloudMicrophysics

const q₀ = 0.01
const c₀ = 1e-3
θ_bubble(x, z) = 300 + 2 * cos(π * min(1, sqrt((x / 2e3)^2 + ((z - 2e3) / 2e3)^2)) / 2)^2

# The issue's reproducer: a dry 2 K bubble, uniform vapor, no microphysics, plus a uniform tracer,
# or with `rest = true` a horizontally uniform rest state on the same grid.
function uniform_scalar_bubble(; timestepper, coefficients = nothing, post_loop_scalar_tendencies, rest = false)
    grid = RectilinearGrid(default_arch; size = (40, 48), halo = (5, 5), x = (-20e3, 20e3), z = (0, 12e3),
                           topology = (Periodic, Flat, Bounded))
    time_discretization = SplitExplicitTimeDiscretization(; damping = NoDivergenceDamping(), forward_weight = 0.65)
    dynamics = CompressibleDynamics(time_discretization; base_pressure = 1e5, reference_potential_temperature = 300)
    timestepper_kwargs = (; post_loop_scalar_tendencies)
    isnothing(coefficients) || (timestepper_kwargs = merge(timestepper_kwargs, (; coefficients)))
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 5), tracers = :ρc,
                            timestepper, timestepper_kwargs)
    if rest
        set!(model; ρ = model.dynamics.reference_state.density, θ = 300)
    else
        set!(model; θ = θ_bubble, qᵛ = q₀, ρ = HydrostaticallyBalancedDensity())
        parent(model.tracers.ρc) .= c₀ .* parent(total_density(model.dynamics))
    end
    return model
end

moisture_uniformity_error(model) = maximum(abs, interior(specific_prognostic_moisture(model)) .- q₀) / q₀

function tracer_uniformity_error(model)
    c = Array(interior(model.tracers.ρc)) ./ Array(interior(total_density(model.dynamics)))
    return maximum(abs, c .- c₀) / c₀
end

# A warm bubble in a sheared wind, with an updraft whose vertical Courant number over a full step, 0.8,
# exceeds that of the adaptive implicit split, 0.5, from the first step. Returns the model and the
# largest vertical Courant number of the run.
function adaptive_implicit_bubble(timestepper, coefficients; Δt = 10, Nt = 20)
    FT = Float64
    grid = RectilinearGrid(default_arch, FT; size = (16, 40), halo = (5, 5), x = (-4e3, 4e3), z = (0, 4e3),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5,
                                    reference_potential_temperature = 300)
    adaptive() = AdaptiveVerticallyImplicitDiscretization(FT; cfl = 0.5)
    momentum_advection = WENO(FT; order = 5, time_discretization = adaptive())
    scalar_advection = WENO(FT; order = 5, bounds = (0, 1), time_discretization = adaptive())
    timestepper_kwargs = (; post_loop_scalar_tendencies = true)
    isnothing(coefficients) || (timestepper_kwargs = merge(timestepper_kwargs, (; coefficients)))
    model = AtmosphereModel(grid; dynamics, momentum_advection, scalar_advection, tracers = :ρc,
                            timestepper, timestepper_kwargs)

    bubble(x, z) = cos(π * min(1, sqrt(x^2 + (z - 1500)^2) / 1e3) / 2)^2
    set!(model; θ = (x, z) -> 300 + 2 * bubble(x, z), qᵛ = q₀, ρ = HydrostaticallyBalancedDensity(),
         u = (x, z) -> 10 * z / 4e3, w = (x, z) -> 8 * bubble(x, z))
    parent(model.tracers.ρc) .= c₀ .* parent(total_density(model.dynamics))

    maximum_w = maximum(abs, interior(model.velocities.w))
    for _ in 1:Nt
        time_step!(model, Δt)
        maximum_w = max(maximum_w, maximum(abs, interior(model.velocities.w)))
    end
    return model, maximum_w * Δt / (4e3 / 40)
end

const steppers = ((:AcousticRungeKutta3, nothing), (:MultirateInfinitesimalStep, :TVDMISB))

# The sweeps over coefficient sets, which repeat every integration with a time stepper of their own,
# run one representative, and all members in the exhaustive tests (BREEZE_TEST_EXHAUSTIVE).
const swept_coefficients = test_sweep((:SSPMIS43, :TVDMISB, :MIS4a))
const swept_steppers = map(coefficients -> (:MultirateInfinitesimalStep, coefficients),
                           test_sweep((:LowStorageMIS52, :SSPMIS43, :ConvexLinearMIS3, :ConvexLinearMIS4,
                                       :ConvexLinearMIS5, :ConvexLinearMIS6)))

@testset "Post-loop scalar tendencies keep a uniform mass fraction uniform [$timestepper, $coefficients]" for
        (timestepper, coefficients) in (steppers..., swept_steppers...)
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

    # Every stage applies its own loop's mass flux to a uniform mass fraction. A multirate infinitesimal step
    # recombines the stages' fluxes with its tableau, and each stage transports the scalars with the mass flux
    # that makes the recombined one the stage's own (`ScalarMassFluxCorrection`).
    @test moisture_uniformity_error(post_loop) < 1e-12
    @test tracer_uniformity_error(post_loop) < 1e-12

    # The implicit remainder of adaptive implicit vertical advection forms the mass fraction it transports
    # with the density the loop leaves in a cell it has compressed (`StageEndDensity`).
    adaptive, vertical_courant_number = adaptive_implicit_bubble(timestepper, coefficients)
    @test vertical_courant_number > 0.5 # the implicit remainder is active
    @test moisture_uniformity_error(adaptive) < 1e-12
    @test tracer_uniformity_error(adaptive) < 1e-12
end

#####
##### A passive scalar in a uniform wind
#####
##### The post-loop advective tendency has to be recombined across the stages like any other slow tendency:
##### applied at weight one it is integrated with the tableau (α, diag(d), γ), an anti-diffusive scheme
##### that amplifies a blob at every Courant number.
#####

function advected_blob_growth(; coefficients, advection, Δt, post_loop_scalar_tendencies = true, Nt = 200, U = 20)
    Nx, Nz, Δx = 64, 8, 200
    grid = RectilinearGrid(default_arch, Float64; size = (Nx, Nz), halo = (5, 5), x = (0, Nx * Δx), z = (0, Nz * Δx),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5, standard_pressure = 1e5,
                                    reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, momentum_advection = advection, scalar_advection = advection,
                            timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies))
    set_discrete_reference_state!(model) # so that the wind stays uniform
    ρ = model.dynamics.dry_density

    qᵛ = CenterField(grid)
    set!(qᵛ, (x, z) -> 1e-6 * cos(π * min(1, abs(x - Nx * Δx / 2) / (8Δx)) / 2)^2)
    ρqᵛ = prognostic_fields(model).ρqᵛ
    interior(ρqᵛ) .= interior(ρ) .* interior(qᵛ)
    interior(model.momentum.ρu) .= interior(ρ) .* U
    fill_halo_regions!((ρqᵛ, model.momentum.ρu))
    update_state!(model)

    mass_fraction() = Array(interior(ρqᵛ)) ./ Array(interior(ρ))
    initial_maximum = maximum(mass_fraction())
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    return maximum(mass_fraction()) / initial_maximum
end

@testset "Post-loop MIS advection does not amplify a passive scalar [$coefficients, Courant number $courant]" for
        coefficients in swept_coefficients, courant in (0.5, 1.0)
    for advection in (WENO(order = 5), WENO(order = 5, bounds = (0, 1)))
        growth = advected_blob_growth(; coefficients, advection, Δt = courant * 200 / 20)
        @test growth ≤ 1 + 1e-3
    end
end

#####
##### A passive tracer in an accelerating flow
#####
##### The recombined post-loop tendencies have to carry each stage's own mass flux with the tracer's own
##### reconstruction; with another reconstruction the tracer is first-order accurate where the flow accelerates.
#####

# A tracer blob offset from a warm bubble that accelerates from rest. The dynamics are dry, so
# the tracer is passive, and the substep size is the same for every Δt.
function accelerating_bubble_tracer(; coefficients, Δt, stop_time = 16, substep = 1/64, N = 32)
    L = 200N
    grid = RectilinearGrid(default_arch, Float64; size = (N, N), halo = (5, 5), x = (-L / 2, L / 2), z = (0, L),
                           topology = (Periodic, Flat, Bounded))
    time_discretization = SplitExplicitTimeDiscretization(; substeps = round(Int, Δt / substep),
                                                          forward_weight = 0.5, damping = NoDivergenceDamping())
    dynamics = CompressibleDynamics(time_discretization; base_pressure = 1e5, standard_pressure = 1e5,
                                    reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, advection = UpwindBiased(order = 5), tracers = :ρc,
                            timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies = true))

    blob(x, z, x₀) = cos(π * min(1, sqrt((x - x₀)^2 + (z - L / 3)^2) / (L / 4)) / 2)^2
    ρᵢ(x, z) = adiabatic_hydrostatic_density(z, 1e5, 300, 1e5, model.thermodynamic_constants)
    set!(model; ρ = ρᵢ, θ = (x, z) -> 300 + 4 * blob(x, z, 0))
    c = CenterField(grid)
    set!(c, (x, z) -> blob(x, z, L / 8))
    parent(model.tracers.ρc) .= parent(c) .* parent(total_density(model.dynamics))
    update_state!(model)

    for _ in 1:round(Int, stop_time / Δt)
        time_step!(model, Δt)
    end

    return Array(interior(model.tracers.ρc)) ./ Array(interior(total_density(model.dynamics)))
end

@testset "Post-loop MIS tracers are second-order accurate in an accelerating flow [$coefficients]" for
        coefficients in swept_coefficients
    c = map(Δt -> accelerating_bubble_tracer(; coefficients, Δt), (2, 1, 1/2))
    differences = (maximum(abs, c[1] .- c[2]), maximum(abs, c[2] .- c[3]))
    @test all(isfinite, differences)
    @test log2(differences[1] / differences[2]) ≥ 1.7
end

@testset "TendencyCallsite callbacks edit the post-loop scalar tendencies once per stage [$timestepper]" for (timestepper, coefficients) in steppers
    # A horizontally uniform rest state with a uniform tendency added by a callback: after one step
    # ρu and ρc each equal c Δt. Momentum takes the callback's pre-loop edit; the post-loop build
    # overwrites the scalar edit, so the callback fires again and ρc takes exactly one edit.
    model = uniform_scalar_bubble(; timestepper, coefficients, post_loop_scalar_tendencies = true, rest = true)

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

# A sheared bubble with a diffused, forced, non-uniform tracer, so that every part of the scalar tendency is
# nonzero, stepped `Nt` times under post-loop scalar tendencies.
const forced_tracer_grid = RectilinearGrid(default_arch; size = (8, 8), halo = (5, 5), x = (-20e3, 20e3), z = (0, 8e3),
                                           topology = (Periodic, Flat, Bounded))
const forced_tracer_dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)

function forced_tracer_model(; coefficients, Nt, timestepper_kwargs = NamedTuple(), boundary_conditions = NamedTuple())
    forcing = (; ρc = Forcing((x, z, t) -> 1e-6 * exp(-(x / 5e3)^2)))
    model = AtmosphereModel(forced_tracer_grid; dynamics = forced_tracer_dynamics, advection = WENO(order = 5),
                            tracers = :ρc, closure = ScalarDiffusivity(ν = 10, κ = 10), forcing, boundary_conditions,
                            timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies = true, timestepper_kwargs...))
    set!(model; θ = θ_bubble, qᵛ = (x, z) -> q₀ * exp(-((x / 4e3)^2 + (z / 2e3)^2)),
         ρ = HydrostaticallyBalancedDensity(), u = 5)
    set!(model.tracers.ρc, (x, z) -> c₀ * exp(-((x / 3e3)^2 + ((z - 2e3) / 2e3)^2)))
    parent(model.tracers.ρc) .*= parent(total_density(model.dynamics))
    for _ in 1:Nt
        time_step!(model, 1)
    end
    return model
end

@testset "The split post-loop build sums to the full scalar tendency" begin
    # A multirate infinitesimal step builds each post-loop scalar tendency in two parts from one
    # evaluation, the advective part into its own scratch and the rest added to `Gⁿ`. The parts
    # must sum to the full tendency to round-off.
    model = forced_tracer_model(; coefficients = :TVDMISB, Nt = 1)
    grid = model.grid

    ts = model.timestepper
    names = scalar_prognostic_names(model)
    # Only the moisture and condensate densities keep a field: the loop's pressure reads no other tendency.
    @test ts.fast_tendency isa NamedTuple
    @test keys(ts.fast_tendency) == (:ρqᵛ,)
    @test fast_tendency_scratch(ts, :ρc) === ts.substepper.stage_end_moisture_content
    pre_loop = AtmosphereModel(grid; dynamics = forced_tracer_dynamics, timestepper = :MultirateInfinitesimalStep,
                               timestepper_kwargs = (; coefficients = :SSPMIS43, post_loop_scalar_tendencies = false)).timestepper
    @test pre_loop.fast_tendency isa Field

    tracer_density_to_specific!(model)
    velocities = transport_velocities(model)
    compute_scalar_tendencies!(model, fields(model), velocities, compute_scalar_tendency!)
    full = Dict(name => Array(interior(ts.Gⁿ[name])) for name in names)

    for name in names
        fill!(parent(ts.Gⁿ[name]), 0)
    end
    advective_tendencies = NamedTuple{names}(map(name -> CenterField(grid), names))
    compute_scalar_tendencies!(model, fields(model), velocities, SplitScalarTendencies(advective_tendencies))
    tracer_specific_to_density!(model)

    for name in names
        slow = Array(interior(ts.Gⁿ[name]))
        fast = Array(interior(advective_tendencies[name]))
        scale = maximum(abs, full[name])
        @test scale > 0
        @test maximum(abs, fast) > 0
        @test maximum(abs, slow) > 0
        @test maximum(abs, slow .+ fast .- full[name]) < 1e-13 * scale
    end
end

@testset "Scalars without a tendency field of their own take the same step [$coefficients]" for
        coefficients in test_sweep((:TVDMISB, :SSPMIS43))
    # Only the moisture and condensate densities keep a tendency field between stages; the other scalars
    # build their advective tendency in shared scratch. A time stepper not told which densities those are
    # (`moisture_forcing_names = nothing`) keeps a field for every scalar: the step must be the same, bit
    # for bit, with a closure, a forcing, a flux boundary condition and the mass-flux correction.
    boundary_conditions = (; ρc = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1e-5)))
    shared = forced_tracer_model(; coefficients, Nt = 3, boundary_conditions)
    separate = forced_tracer_model(; coefficients, Nt = 3, boundary_conditions,
                                   timestepper_kwargs = (; moisture_forcing_names = nothing))

    @test keys(shared.timestepper.fast_tendency) == (:ρqᵛ,)
    @test keys(separate.timestepper.fast_tendency) == (:ρθ, :ρqᵛ, :ρc)
    @test maximum(abs, interior(shared.tracers.ρc)) > 0

    for (u, v) in zip(prognostic_fields(shared), prognostic_fields(separate))
        @test Array(interior(u)) == Array(interior(v))
    end
end

#####
##### Positivity in an accelerating, sheared flow
#####
##### Each stage of a set with a non-diagonal β transports the scalars with the mass flux Fⱼ + Rⱼ: its own
##### loop's plus the correction that makes the recombined fluxes carry each stage's own. Rⱼ is largest where
##### the flow accelerates and diverges, and sharp-edged scalars next to zeros show any flux the bounded
##### reconstruction does not bound. The undershoots are those of the bounded scheme at a horizontal Courant
##### number near one, an order of magnitude and more below those of the unbounded scheme.
#####

const qᵛ₀ = 0.012 # the vapor slab's mass fraction

# `courant_number` is that of the initial wind, whose maximum is 25 m/s. The transport velocity
# reaches a Courant number about 15% higher as the bubble rises.
function sheared_bubble_scalars(; coefficients, courant_number, uniform = false, stop_time = 240,
                                scalar_advection = WENO(order = 5, bounds = (0, 1)))
    Nx, Nz, Δ = 32, 24, 200
    Lx, Lz = Nx * Δ, Nz * Δ
    grid = RectilinearGrid(default_arch, Float64; size = (Nx, Nz), halo = (5, 5), x = (-Lx / 2, Lx / 2), z = (0, Lz),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5,
                                    reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, momentum_advection = WENO(order = 5), scalar_advection, tracers = :ρc,
                            timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies = true))

    bubble(x, z) = cos(π * min(1, sqrt(x^2 + (z - 1200)^2) / 1000) / 2)^2
    uᵢ(x, z) = 20 * tanh((z - 2000) / 600) - 5 * sin(2π * x / Lx) * exp(-z / 1500)
    # A slab of vapor with sharp edges, and a tracer strip two cells wide.
    qᵢ(x, z) = uniform ? qᵛ₀ : ifelse(abs(x + 800) < 1000 && z < 1800, qᵛ₀, 0.0)
    cᵢ(x, z) = uniform ? 1.0 : ifelse(abs(x - 300) < Δ && 600 < z < 2400, 1.0, 0.0)
    set!(model; θ = (x, z) -> 300 + 8 * bubble(x, z), qᵛ = qᵢ, ρ = HydrostaticallyBalancedDensity(), u = uᵢ)
    ρ = total_density(model.dynamics)
    c = CenterField(grid)
    set!(c, cᵢ)
    parent(model.tracers.ρc) .= parent(c) .* parent(ρ)
    update_state!(model)

    Δt = courant_number * Δ / maximum(abs, interior(model.velocities.u))
    mass_fractions() = (q = Array(interior(prognostic_fields(model).ρqᵛ)) ./ Array(interior(ρ)),
                        c = Array(interior(model.tracers.ρc)) ./ Array(interior(ρ)))

    lowest = (q = Inf, c = Inf)
    highest = (q = -Inf, c = -Inf)
    for _ in 1:ceil(Int, stop_time / Δt)
        time_step!(model, Δt)
        fractions = mass_fractions()
        lowest = map((l, f) -> min(l, minimum(f)), lowest, fractions)
        highest = map((h, f) -> max(h, maximum(f)), highest, fractions)
    end

    return (; lowest, highest, maximum_w = maximum(abs, interior(model.velocities.w)))
end

@testset "Post-loop MIS scalars stay positive in an accelerating, sheared flow [$coefficients]" for
        coefficients in (:SSPMIS43, :TVDMISB)
    scalars = sheared_bubble_scalars(; coefficients, courant_number = 0.75)
    @test scalars.maximum_w > 5 # the bubble has accelerated

    # Lowest mass fractions over the run, relative to the initial maximum: -1.0e-4 (`:SSPMIS43`)
    # and -2.1e-4 (`:TVDMISB`) for the vapor, -1.2e-6 and -2.6e-6 for the tracer. The unbounded
    # scheme gives -5.2e-3 and -1.6e-4 or lower, and a Courant number of 1.4 gives -2.4e-2 and -3.7e-3.
    @test scalars.lowest.q ≥ -1e-3 * qᵛ₀
    @test scalars.lowest.c ≥ -2e-5

    # A uniform mass fraction stays uniform to round-off in the same flow.
    uniform = sheared_bubble_scalars(; coefficients, courant_number = 0.75, uniform = true, stop_time = 120)
    @test max(abs(uniform.lowest.q - qᵛ₀), abs(uniform.highest.q - qᵛ₀)) < 1e-12 * qᵛ₀
    @test max(abs(uniform.lowest.c - 1), abs(uniform.highest.c - 1)) < 1e-12
end

@testset "The positivity bound of the post-loop MIS scalars needs the bounded scheme" begin
    unbounded = sheared_bubble_scalars(; coefficients = :SSPMIS43, courant_number = 0.75, scalar_advection = WENO(order = 5))
    @test unbounded.lowest.q < -1e-3 * qᵛ₀
    @test unbounded.lowest.c < -2e-5
end

#####
##### Stage increments with an active negative-moisture correction
#####
##### `update_state!` corrects negative moisture between stages. The increment ΔYᵢ = Yᵢ − Uⁿ that later
##### stages start from and are forced with must be that of the corrected state, with both storage layouts.
#####

# Sharp-edged rain in a sheared, overturning flow: unbounded WENO undershoots at its edges, so the
# correction has negative rain to repair after every stage.
function raining_bubble(; post_loop_scalar_tendencies, negative_moisture_correction = SpeciesBorrowing(),
                        coefficients = :TVDMISB)
    grid = RectilinearGrid(default_arch; size = (16, 16), halo = (5, 5), x = (-8e3, 8e3), z = (0, 8e3),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5, standard_pressure = 1e5,
                                    reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 5), tracers = :ρc,
                            microphysics = OneMomentCloudMicrophysics(; negative_moisture_correction),
                            timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies))
    constants = model.thermodynamic_constants
    set!(model; θ = θ_bubble, qᵗ = 0.018, ρ = (x, z) -> adiabatic_hydrostatic_density(z, 1e5, 300, 1e5, constants),
                u = (x, z) -> 15 * sin(2π * z / 8e3), w = (x, z) -> 4 * sin(2π * x / 16e3))
    set!(model; ρqʳ = (x, z) -> ifelse(abs(x) < 2e3 && 2e3 < z < 5e3, 2e-3, 0.0),
                ρc = (x, z) -> exp(-((z - 2e3) / 1e3)^2 - (x / 2e3)^2))
    return model
end

function raining_bubble_state(Nt; fold, structured = true, kw...)
    fold_stage_increments[] = fold
    structured_stage_storage[] = structured
    model = raining_bubble(; kw...)
    fold_stage_increments[] = true
    structured_stage_storage[] = true
    for _ in 1:Nt
        time_step!(model, 6)
    end
    return map(field -> Array(interior(field)), prognostic_fields(model)), model.timestepper.storage
end

@testset "MIS stage increments with an active negative-moisture correction [post_loop = $post_loop_scalar_tendencies]" for post_loop_scalar_tendencies in (false, true)
    # The setup gives the correction something to do: uncorrected, a step leaves negative rain.
    uncorrected, _ = raining_bubble_state(1; fold = true, post_loop_scalar_tendencies, negative_moisture_correction = nothing)
    @test minimum(uncorrected.ρqʳ) < 0

    # Only the prognostics the acoustic loop advances form their increments in the assembly.
    model = raining_bubble(; post_loop_scalar_tendencies)
    for name in keys(prognostic_fields(model))
        source = stage_increment_source(model, name)
        @test source isa (name ∈ acoustic_prognostic_names(model) ? IncrementFromState : StoredIncrement)
    end

    for Nt in (1, 3)
        folded, _ = raining_bubble_state(Nt; fold = true, post_loop_scalar_tendencies)
        stored, _ = raining_bubble_state(Nt; fold = false, post_loop_scalar_tendencies)
        @test minimum(folded.ρqʳ) ≥ 0
        for name in keys(folded)
            scale = maximum(abs, stored[name])
            # The two paths differ in round-off alone; GPUs fuse multiply-adds differently.
            @test maximum(abs, folded[name] .- stored[name]) ≤ 400 * eps(scale)
        end
    end
end

# A set with two registers reads the increments from the corrected prognostic fields, and reproduces to
# round-off the same set with accumulators stored after the correction, as in "MIS storage layouts
# reproduce the accumulators" of `multirate_infinitesimal_step.jl`.
@testset "MIS storage layouts agree with an active negative-moisture correction [$coefficients, post_loop = $post_loop_scalar_tendencies]" for
        (coefficients, layout) in ((:LowStorageMIS52, TwoRegisterStorage),),
        post_loop_scalar_tendencies in (false, true)

    kw = (; coefficients, post_loop_scalar_tendencies)
    uncorrected, _ = raining_bubble_state(1; fold = true, negative_moisture_correction = nothing, kw...)
    @test minimum(uncorrected.ρqʳ) < 0

    structured, storage = raining_bubble_state(3; fold = true, kw...)
    accumulated, accumulator_storage = raining_bubble_state(3; fold = true, structured = false, kw...)
    @test storage isa layout
    @test accumulator_storage isa AccumulatorStorage
    @test minimum(structured.ρqʳ) ≥ 0

    for name in keys(accumulated)
        scale = maximum(abs, accumulated[name])
        @test maximum(abs, structured[name] .- accumulated[name]) ≤ 1000 * eps(scale)
    end
end

#####
##### Microphysical sinks are budgeted against the advection of the same stage
#####
##### A sink is budgeted against what the stage-start state and every other tendency of the stage
##### leave of a species. The microphysical sources are added right after the post-loop build of the
##### scalars' other tendencies, and where that build leaves the advection is a matter of the storage
##### layout: `TwoRegisterStorage` (a diagonal `β` and first-subdiagonal `α` and `γ`) adds the
##### whole tendency to `Gⁿ`, and `AccumulatorStorage` keeps each scalar's advective tendency apart
##### from `Gⁿ`, which then holds no advection. A sink budgeted against the stage-start state and
##### `Gⁿ` alone can then remove what advection removes too, and the two together leave a species
##### negative where neither does alone: at the edge of a cloud that the wind carries into dry air,
##### cloud evaporation takes all the cloud that advection is also carrying away.
##### `separate_stage_tendencies` hands the microphysics what `Gⁿ` lacks, and nothing that it holds.
#####

using Breeze.AtmosphereModels: separate_stage_tendencies
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticlePropertiesMicrophysics
using Breeze.TimeSteppers: multirate_infinitesimal_step_tableau, post_loop_scalar_tendency_kernel
using CloudMicrophysics.Parameters: CloudIce, CloudLiquid

# One coefficient set and more for each storage layout, all strong-stability preserving. The last
# is a set whose structure takes `TwoRegisterStorage`, made to accumulate.
const sink_budget_layouts = ((:SSPMIS43, true, AccumulatorStorage),
                             (:LowStorageMIS52, true, TwoRegisterStorage),
                             (:ConvexLinearMIS4, true, TwoRegisterStorage),
                             (:LowStorageMIS52, false, AccumulatorStorage))

function with_stage_storage(build, structured)
    structured_stage_storage[] = structured
    try
        return build()
    finally
        structured_stage_storage[] = true
    end
end

@testset "Tendencies held apart from Gⁿ when the microphysical sources are added [$coefficients, $layout]" for
        (coefficients, structured, layout) in sink_budget_layouts

    @test multirate_infinitesimal_step_tableau(coefficients).strong_stability_coefficient > 0

    grid = RectilinearGrid(default_arch; size = (8, 8), halo = (5, 5), x = (-20e3, 20e3), z = (0, 8e3),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    ρc_forcing = Forcing((x, z, t) -> 1e-6 * exp(-(x / 5e3)^2))

    build(post_loop_scalar_tendencies) = with_stage_storage(structured) do
        AtmosphereModel(grid; dynamics, advection = WENO(order = 5), tracers = :ρc,
                        closure = ScalarDiffusivity(ν = 10, κ = 10), forcing = (; ρc = ρc_forcing),
                        timestepper = :MultirateInfinitesimalStep,
                        timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies))
    end

    # Pre-loop, `update_state!` builds the whole tendency into `Gⁿ` with every layout.
    pre_loop = build(false)
    @test pre_loop.timestepper.storage isa layout
    @test isnothing(separate_stage_tendencies(pre_loop.timestepper, pre_loop))

    model = build(true)
    ts = model.timestepper
    @test ts.storage isa layout
    set!(model; θ = θ_bubble, qᵛ = (x, z) -> q₀ * exp(-((x / 4e3)^2 + (z / 2e3)^2)),
         ρ = HydrostaticallyBalancedDensity(), u = 5)
    set!(model.tracers.ρc, (x, z) -> c₀ * exp(-((x / 3e3)^2 + ((z - 2e3) / 2e3)^2)))
    parent(model.tracers.ρc) .*= parent(total_density(model.dynamics))

    time_step!(model, 1)

    # The tendencies as the first stage builds them, with a mass flux that no layout corrects:
    # the whole of each...
    ts.stage[] = 1
    names = scalar_prognostic_names(model)
    zero_scalar_tendencies!() = foreach(name -> fill!(parent(ts.Gⁿ[name]), 0), names)
    tracer_density_to_specific!(model)
    velocities = transport_velocities(model)
    compute_scalar_tendencies!(model, fields(model), velocities, compute_scalar_tendency!)
    full = Dict(name => Array(interior(ts.Gⁿ[name])) for name in names)

    # ...its advective part...
    zero_scalar_tendencies!()
    compute_scalar_tendencies!(model, fields(model), velocities, SplitScalarTendencies(map(similar, ts.Gⁿ[names])))
    advective = Dict(name => full[name] .- Array(interior(ts.Gⁿ[name])) for name in names)

    # ...and what the stage's own build leaves in `Gⁿ` and apart from it for the microphysics.
    zero_scalar_tendencies!()
    compute_scalar_tendencies!(model, fields(model), velocities, post_loop_scalar_tendency_kernel(ts, model))
    tracer_specific_to_density!(model)
    separate = separate_stage_tendencies(ts, model)

    # With accumulators the moisture and microphysical densities hold their advection apart; a plain
    # tracer, which no microphysical budget reads, has its advection built after the microphysics.
    if layout === AccumulatorStorage
        @test haskey(separate, :ρqᵛ)
        @test !haskey(separate, :ρc)
    else
        @test isnothing(separate)
    end
    held_apart = isnothing(separate) ? names : filter(name -> haskey(separate, name), names)

    for name in held_apart
        G = Array(interior(ts.Gⁿ[name]))
        apart = isnothing(separate) ? zero(G) : Array(interior(separate[name]))
        scale = maximum(abs, full[name])
        @test maximum(abs, advective[name]) > 1e-3 * scale

        # Together they are the whole tendency, once: nothing missing, and nothing counted twice.
        @test maximum(abs, G .+ apart .- full[name]) < 1e-13 * scale

        # What is held apart is the advection, which `Gⁿ` then lacks.
        if layout === AccumulatorStorage
            @test maximum(abs, apart .- advective[name]) < 1e-13 * scale
        end
    end
end

# The moist bubble of the positivity tests (multirate_infinitesimal_step.jl,
# bulk_microphysics_positivity.jl), carried by a 20 m s⁻¹ wind: bounded advection of every
# scalar, no repair of negative densities, and strong-stability-preserving coefficients.
function advected_cloud_extrema(microphysics; coefficients = :SSPMIS43, structured = true, Δt = 10, Nt = 120)
    FT = Float64
    grid = RectilinearGrid(default_arch, FT; size = (32, 24), halo = (5, 5),
                           x = (-10e3, 10e3), z = (0, 12e3), topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5,
                                    standard_pressure = 1e5, reference_potential_temperature = 300)
    model = with_stage_storage(structured) do
        AtmosphereModel(grid; dynamics, microphysics, momentum_advection = WENO(order=5),
                        scalar_advection = WENO(order=5, bounds=(0.0, 1e30)),
                        timestepper = :MultirateInfinitesimalStep,
                        timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies = true))
    end

    constants = model.thermodynamic_constants
    θᵢ(x, z) = 300 + 3 * cos(π * min(1, sqrt((x / 2e3)^2 + ((z - 2e3) / 2e3)^2)) / 2)^2
    ρᵢ(x, z) = adiabatic_hydrostatic_density(z, 1e5, 300, 1e5, constants)
    set!(model; ρ = ρᵢ, θ = θᵢ, qᵗ = 0.012, u = 20)

    prognostics = prognostic_fields(model)
    names = filter(name -> startswith(string(name), "ρq") || startswith(string(name), "ρn") ||
                           name === :ρbᶠ, keys(prognostics))
    minima = Dict(name => Inf for name in names)
    maxima = Dict(name => -Inf for name in names)

    for _ in 1:Nt
        time_step!(model, Δt)
        for name in names
            minima[name] = min(minima[name], minimum(prognostics[name]))
            maxima[name] = max(maxima[name], maximum(prognostics[name]))
        end
    end

    return minima, maxima, model.timestepper.storage
end

@testset "Post-loop advection counts against the microphysical sink budgets [$label, $coefficients, $layout]" for
        (label, microphysics) in
            (("P3", PredictedParticlePropertiesMicrophysics(Float64; negative_moisture_correction = nothing)),
             ("1M", OneMomentCloudMicrophysics(Float64; negative_moisture_correction = nothing,
                                               cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(Float64), CloudIce(Float64))))),
        (coefficients, structured, layout) in sink_budget_layouts

    minima, maxima, storage = advected_cloud_extrema(microphysics; coefficients, structured)
    @test storage isa layout

    # Cloud, rain and ice form and are carried through dry air...
    @test all(maxima[name] > 0 for name in keys(maxima))

    # ...and nothing goes negative, beyond the roundoff of the bounded advection of fields whose
    # neighbouring cells are many orders of magnitude apart (see bulk_microphysics_positivity.jl):
    # a few rounding errors of the largest value, which the sums of a stage accumulate (P3 rime
    # volume reaches -1.25 eps times its maximum with `:ConvexLinearMIS4`). Budgeted without the
    # advective tendency, P3 leaves cloud liquid at -4e-7 kg m⁻³ and the one-moment scheme rain at
    # -2e-4 kg m⁻³, 4% of its maximum, with `:SSPMIS43`.
    @test all(minima[name] ≥ -4 * eps(Float64) * maxima[name] for name in keys(minima))
end
