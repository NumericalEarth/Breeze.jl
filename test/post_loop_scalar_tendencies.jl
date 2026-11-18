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
using Oceananigans: TendencyCallsite, fields, prognostic_fields
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

const steppers = ((:AcousticRungeKutta3, nothing), (:MultirateInfinitesimalStep, :TVDMISB))

# The sweeps over the recombining coefficient sets, which repeat every integration with a time
# stepper of its own, run the default set, and the others in the exhaustive tests
# (BREEZE_TEST_EXHAUSTIVE, set on one CI runner).
const swept_coefficients = test_sweep((:SSPMIS43, :TVDMISB, :MIS4a))

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

    # Every stage applies its own loop's mass flux, q/(1 − q) times the dry-mass flux continuity
    # applied, to a uniform mass fraction, which therefore stays uniform to round-off.
    # Wicker–Skamarock applies each stage's flux alone. A multirate infinitesimal step recombines
    # the stages' fluxes with its tableau (leaving them at weight one would be unstable: see the
    # passive-scalar test below), and each stage transports the scalars with the mass flux that
    # makes the recombined one the stage's own (`ScalarMassFluxCorrection`).
    @test moisture_uniformity_error(post_loop) < 1e-12
    @test tracer_uniformity_error(post_loop) < 1e-12
end

#####
##### A passive scalar in a uniform wind
#####
##### Advection can only reduce the maximum of a blob. The post-loop advective tendency has to be
##### recombined across the stages of a multirate infinitesimal step like any other slow tendency:
##### applied at weight one from the current stage it is integrated with the tableau (α, diag(d), γ),
##### first order with b̃·c < 1/2 for every non-diagonal set — an anti-diffusive scheme that
##### amplifies the blob at every Courant number.
#####

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans: prognostic_fields

function advected_blob_growth(; coefficients, advection, Δt, post_loop_scalar_tendencies = true, Nt = 200, U = 20)
    Nx, Nz, Δx = 64, 8, 200
    grid = RectilinearGrid(default_arch, Float64; size = (Nx, Nz), halo = (5, 5), x = (0, Nx * Δx), z = (0, Nz * Δx),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5, standard_pressure = 1e5,
                                    reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, momentum_advection = advection, scalar_advection = advection,
                            timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies))

    # The discrete reference state, exactly, so that the wind stays uniform.
    reference = model.dynamics.reference_state
    Rᵈ = Breeze.dry_air_gas_constant(model.thermodynamic_constants)
    ρ = model.dynamics.dry_density
    parent(ρ) .= parent(reference.density)
    parent(Breeze.AtmosphereModels.thermodynamic_density(model.formulation)) .= parent(reference.pressure) ./ (Rᵈ .* parent(reference.exner_function))

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
##### The recombined post-loop tendencies of a multirate infinitesimal step have to carry each
##### stage's own mass flux with the tracer's own reconstruction. With the mass flux of the
##### correction carried by another reconstruction (a first-order upwind one, in the correction's
##### own direction), the two differ by a truncation error that does not decrease with the time
##### step, and the tracer is first-order accurate in time wherever the flow accelerates: the
##### differences below then halve with the time step instead of falling fourfold.
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

@testset "TendencyCallsite callbacks edit the post-loop scalar tendencies once per stage [$timestepper]" for (timestepper, coefficients) in steppers[1:2]
    # A horizontally uniform rest state with a uniform tendency added by a callback: after one step
    # ρu and ρc each equal c Δt. Momentum takes the callback's pre-loop edit; the post-loop build
    # overwrites the scalar edit, so the callback fires again and ρc takes exactly one edit.
    # The grid, dynamics and advection are those of `uniform_scalar_bubble`, so that this model has
    # its type (which includes the grid size) and the test compiles nothing new.
    grid = RectilinearGrid(default_arch; size = (40, 48), halo = (5, 5), x = (-20e3, 20e3), z = (0, 12e3),
                           topology = (Periodic, Flat, Bounded))
    time_discretization = SplitExplicitTimeDiscretization(; damping = NoDivergenceDamping(), forward_weight = 0.65)
    dynamics = CompressibleDynamics(time_discretization; base_pressure = 1e5, reference_potential_temperature = 300)
    timestepper_kwargs = (; post_loop_scalar_tendencies = true)
    isnothing(coefficients) || (timestepper_kwargs = merge(timestepper_kwargs, (; coefficients)))
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 5), tracers = :ρc, timestepper, timestepper_kwargs)
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
                            timestepper_kwargs = (; coefficients = :TVDMISB, post_loop_scalar_tendencies = true))
    set!(model; θ = θ_bubble, qᵛ = (x, z) -> q₀ * exp(-((x / 4e3)^2 + (z / 2e3)^2)),
         ρ = HydrostaticallyBalancedDensity(), u = 5)
    set!(model.tracers.ρc, (x, z) -> c₀ * exp(-((x / 3e3)^2 + ((z - 2e3) / 2e3)^2)))
    parent(model.tracers.ρc) .*= parent(total_density(model.dynamics))
    time_step!(model, 1)

    ts = model.timestepper
    names = scalar_prognostic_names(model)
    # Only the moisture and condensate densities keep a field: the loop's pressure reads no other tendency.
    @test ts.fast_tendency isa NamedTuple
    @test keys(ts.fast_tendency) == (:ρqᵛ,)
    @test fast_tendency_scratch(ts, :ρc) === ts.substepper.stage_end_moisture_content
    pre_loop = AtmosphereModel(grid; dynamics, timestepper = :MultirateInfinitesimalStep,
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
    # The loop's pressure reads the tendencies of the moisture and condensate densities alone, so
    # only those keep a field between stages. The other scalars build their advective tendency in
    # shared scratch just before they are assembled. A time stepper that is not told which
    # densities those are (`moisture_forcing_names = nothing`) keeps a field for every scalar and
    # builds every advective tendency at once: the step must be the same, bit for bit, with a
    # closure, a forcing, a flux boundary condition and the mass-flux correction of the later stages.
    grid = RectilinearGrid(default_arch; size = (8, 8), halo = (5, 5), x = (-20e3, 20e3), z = (0, 8e3),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    ρc_forcing = Forcing((x, z, t) -> 1e-6 * exp(-(x / 5e3)^2))
    ρc_boundary_conditions = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1e-5))

    function stepped_model(timestepper_kwargs)
        model = AtmosphereModel(grid; dynamics, advection = WENO(order = 5), tracers = :ρc,
                                closure = ScalarDiffusivity(ν = 10, κ = 10), forcing = (; ρc = ρc_forcing),
                                boundary_conditions = (; ρc = ρc_boundary_conditions),
                                timestepper = :MultirateInfinitesimalStep,
                                timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies = true, timestepper_kwargs...))
        set!(model; θ = θ_bubble, qᵛ = (x, z) -> q₀ * exp(-((x / 4e3)^2 + (z / 2e3)^2)),
             ρ = HydrostaticallyBalancedDensity(), u = 5)
        set!(model.tracers.ρc, (x, z) -> c₀ * exp(-((x / 3e3)^2 + ((z - 2e3) / 2e3)^2)))
        parent(model.tracers.ρc) .*= parent(total_density(model.dynamics))
        for _ in 1:3
            time_step!(model, 1)
        end
        return model
    end

    shared = stepped_model(NamedTuple())
    separate = stepped_model((; moisture_forcing_names = nothing))

    @test keys(shared.timestepper.fast_tendency) == (:ρqᵛ,)
    @test keys(separate.timestepper.fast_tendency) == (:ρθ, :ρqᵛ, :ρc)
    @test maximum(abs, interior(shared.tracers.ρc)) > 0

    for (u, v) in zip(prognostic_fields(shared), prognostic_fields(separate))
        @test Array(interior(u)) == Array(interior(v))
    end
end

#####
##### Stage increments of a multirate infinitesimal step with an active negative-moisture correction
#####
##### `update_state!` corrects negative moisture between stages, and the next slow tendencies are
##### evaluated at the corrected state. The increment ΔYᵢ = Yᵢ − Uⁿ that later stages start from and
##### are forced with is that of the corrected state too, with every storage layout: the layout with
##### two registers reads it from the prognostic fields, and the layouts with increment registers and
##### with accumulators store it once `update_state!` has run. (They used to store it when the stage
##### completed, before the correction, so that the layouts differed wherever the correction was
##### active.) The increments of the acoustic prognostics, which `update_state!` does not change, are
##### formed in the next stage's assembly or stored with the others; either way the step is the same.
#####

using CloudMicrophysics
using Breeze.AtmosphereModels: SpeciesBorrowing
using Breeze.TimeSteppers: fold_stage_increments, structured_stage_storage, stage_increment_source,
                           acoustic_prognostic_names, IncrementFromState, StoredIncrement,
                           TwoRegisterStorage, IncrementStorage, AccumulatorStorage
using Oceananigans: prognostic_fields

const OneMomentCloudMicrophysics = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt).OneMomentCloudMicrophysics

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

# The increments are those of the corrected state with every layout: a coefficient set that keeps two
# registers reads them from the corrected prognostic fields, and reproduces to round-off the same
# set with increment registers stored after the correction and with accumulators. Three steps of
# this bubble amplify the round-off the layouts differ by, as in "MIS storage layouts reproduce the
# accumulators".
@testset "MIS storage layouts agree with an active negative-moisture correction [$coefficients, post_loop = $post_loop_scalar_tendencies]" for
        (coefficients, layout) in ((:LowStorageMIS52, TwoRegisterStorage), (:DiagonalMIS42, IncrementStorage)),
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
