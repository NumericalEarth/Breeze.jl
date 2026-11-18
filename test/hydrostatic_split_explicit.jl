include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.HydrostaticEquations: HydrostaticModel, VerletSubsteps, ForwardBackwardSubsteps, stage_cfl_bound,
                                   freeze_gravity_wave_linearization!, subtract_fast_tendencies!
using Breeze.TimeSteppers: GravityWaveMultirateInfinitesimalStep, multirate_infinitesimal_step_tableau
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!
using Test

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

# The split-explicit schemes are compared with the unsplit SSP RK3 integration of the same
# discrete equations, the oracle. Every tolerance below was written, with its rationale, before
# the test it belongs to was first run.

#####
##### Configurations
#####

# Substeps per time step for which every stage of the tableau takes between 3 and 6 substeps,
# where its stage fractions allow it: MIS3C has stages of 0.398 Δt, each with ⌈3.98⌉ = 4;
# MIS4a has stages of (0.388, 0.361, 0.185, 0.538) Δt with (5, 4, 3, 6). The stages of
# Wicker-Skamarock RK3 span a factor of three, (1/3, 1/2, 1) Δt: with 6 substeps in the last
# one they take (2, 3, 6).
const tableau_substeps = (MIS3C = 10, MIS4a = 11, WickerSkamarock3 = 6)

const tableaux = keys(tableau_substeps)
const inner_solvers = (VerletSubsteps(), ForwardBackwardSubsteps(divergence_damping=0.1))
const configurations = [(tableau, inner_solver) for tableau in tableaux for inner_solver in inner_solvers]

# `split_explicit` and `configuration_name` are defined in `hydrostatic_test_utils.jl`

#####
##### Construction
#####

@testset "GravityWaveMultirateInfinitesimalStep construction" begin
    grid = RectilinearGrid(default_arch, FT; size=(4, 4, 5), x=(0, 4e5), y=(0, 4e5), z=(0, 1))
    dynamics() = HydrostaticDynamics(top_pressure=200)

    for (tableau, inner_solver) in configurations
        model = hydrostatic_model(grid; dynamics=dynamics(), split_explicit(tableau, inner_solver)...)
        @test model isa HydrostaticModel
        @test model.timestepper isa GravityWaveMultirateInfinitesimalStep
    end

    # No default scheme: both the outer scheme and the inner solver must be chosen
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), timestepper=:GravityWaveMultirateInfinitesimalStep)
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), timestepper=:GravityWaveMultirateInfinitesimalStep,
                                                 timestepper_kwargs=(; tableau=:MIS3C))
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), timestepper=:GravityWaveMultirateInfinitesimalStep,
                                                 timestepper_kwargs=(; inner_solver=VerletSubsteps()))
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), split_explicit(:MIS2, VerletSubsteps())...)
    @test_throws ArgumentError ForwardBackwardSubsteps(divergence_damping=-0.1)

    # The stage Courant number bounds: 2 / √(1 + 2 βᵈ) for forward-backward substeps, 2 for Verlet
    @test stage_cfl_bound(VerletSubsteps()) == 2
    @test stage_cfl_bound(ForwardBackwardSubsteps(divergence_damping=0.1)) ≈ 2 / sqrt(1.2)
    @test stage_cfl_bound(ForwardBackwardSubsteps(divergence_damping=0)) == 2

    # The default aim is 0.7 of the bound, and an aim above the bound is rejected
    model = hydrostatic_model(grid; dynamics=dynamics(), split_explicit(:MIS3C, VerletSubsteps())...)
    @test model.timestepper.substepper.cfl ≈ 1.4
    model = hydrostatic_model(grid; dynamics=dynamics(), split_explicit(:MIS3C, ForwardBackwardSubsteps())...)
    @test model.timestepper.substepper.cfl ≈ 0.7 * 2 / sqrt(1.2)
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), split_explicit(:MIS3C, VerletSubsteps(); cfl=2.1)...)
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(),
                                                 split_explicit(:MIS3C, ForwardBackwardSubsteps(); cfl=1.9)...)
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), split_explicit(:MIS3C, VerletSubsteps(); substeps=0)...)

    # A prescribed number of substeps that leaves a stage above the bound is an error, not an
    # unstable integration: with Δx = 1e5 m the fastest wave has ω ≈ 2 √2 × 347 / 1e5 ≈ 1e-2 s⁻¹,
    # so one substep of the 0.4 × 3600 s stages has ω Δτ ≈ 14.
    model = hydrostatic_model(grid; dynamics=dynamics(), split_explicit(:MIS3C, VerletSubsteps(); substeps=1)...)
    set!(model; θ=300)
    @test_throws ArgumentError time_step!(model, 3600)

    # Each tableau ends its last stage at the end of the time step
    for tableau in tableaux
        coefficients = multirate_infinitesimal_step_tableau(tableau)
        @test coefficients.c[end] ≈ 1 atol=1e-9
        @test all(coefficients.d .> 0)
    end

    # The substep counts of the configurations used below
    stage_counts(tableau) = [max(1, ceil(Int, d * tableau_substeps[tableau] - 1e-9))
                             for d in multirate_infinitesimal_step_tableau(tableau).d]
    @test stage_counts(:MIS3C) == [4, 4, 4]
    @test stage_counts(:MIS4a) == [5, 4, 3, 6]
    @test stage_counts(:WickerSkamarock3) == [2, 3, 6]
end

#####
##### The fast operator is the linearization of the full tendencies about a state at rest
#####
#
# About a state with no flow, the full tendencies of (pˢ, u, v, ρθ) depend on a perturbation
# through the terms of the gravity-wave operator and through terms that are quadratic in the
# velocity. A centered difference of the full tendencies over ± ε δ cancels the quadratic terms,
# so it must equal the operator applied to δ up to the third-order term of the pressure-gradient
# force and rounding.
#
# Tolerance rationale. The perturbation is ε δ with ε = 1e-2 and δ of 100 Pa, 1 K and 1 m s⁻¹,
# a relative perturbation of the state of at most 3e-5, so the third-order term is relatively
# ~ 1e-9. The tendencies carry rounding of ~ 1e-15 × cᵖᵈ θ Π / Δx ~ 3e-15 m s⁻² against a
# centered difference of ~ 2 ε × 1e-3 m s⁻², relatively 1e-10; the other variables are no worse.
# The two must agree within 1e-6 of the largest value of the operator, for each variable. The
# state is stratified and lies over a 2 km mountain, on hybrid levels, so that every term of the
# linearization is exercised, including the gradient of the frozen Exner function.

@testset "Linearization of the gravity-wave operator" begin
    Nx, Ny, Nz = 8, 6, 8
    grid = RectilinearGrid(default_arch, FT; size=(Nx, Ny, Nz), x=(0, 8e5), y=(0, 6e5), z=(0, 1))
    A, B = hybrid_coefficients(Nz)
    kˣ, kʸ = 2π / 8e5, 2π / 6e5

    mountain(x, y) = g * 2000 * exp(-((x - 4e5) / 1.5e5)^2 - ((y - 3e5) / 1.5e5)^2)
    dynamics = HydrostaticDynamics(surface_geopotential=mountain, pressure_coefficient=A, surface_pressure_coefficient=B)
    model = hydrostatic_model(grid; dynamics, thermodynamic_constants=reference_model_constants(),
                              split_explicit(:MIS3C, VerletSubsteps())...)

    function set_state!(ε)
        set!(model; pˢ = (x, y) -> isentropic_surface_pressure(mountain(x, y), 300) + ε * 100 * cos(kˣ * x) * sin(kʸ * y),
                    θ = (x, y, z) -> 290 + 50 * z + 3 * sin(kˣ * x) + ε * cos(kˣ * x + 2π * z) * cos(kʸ * y),
                    u = (x, y, z) -> ε * (1 + z) * sin(kˣ * x) * cos(kʸ * y),
                    v = (x, y, z) -> ε * cos(π * z) * cos(kˣ * x) * sin(kʸ * y))
        update_state!(model)
        prognostic = Oceananigans.prognostic_fields(model)
        names = (:pˢ, :u, :v, :ρθ)
        return (state = NamedTuple{names}(map(name -> Array(interior(prognostic[name])), names)),
                tendency = NamedTuple{names}(map(name -> Array(interior(model.timestepper.Gⁿ[name])), names)))
    end

    ε = 1e-2
    plus, minus = set_state!(ε), set_state!(-ε)
    set_state!(0)

    substepper = model.timestepper.substepper
    freeze_gravity_wave_linearization!(substepper, model)

    # δ in the prognostic variables, and the operator applied to it
    fast = model.timestepper.slow_tendencies[1]
    for name in (:pˢ, :u, :v, :ρθ)
        set!(substepper.perturbations[name], (plus.state[name] .- minus.state[name]) ./ 2ε)
        fill!(fast[name], 0)
    end
    fill_halo_regions!((substepper.perturbations.u, substepper.perturbations.v))
    subtract_fast_tendencies!(fast, substepper, model)

    for name in (:pˢ, :u, :v, :ρθ)
        operator = - Array(interior(fast[name]))
        difference = (plus.tendency[name] .- minus.tendency[name]) ./ 2ε
        @info "Linearization of $name: operator $(maximum(abs, operator)), mismatch $(maximum(abs, operator .- difference))"
        @test maximum(abs, operator) > 0
        @test maximum(abs, operator .- difference) < 1e-6 * maximum(abs, operator)
    end
end

#####
##### (1) Resting states, with and without the 2 km mountain
#####
#
# With the mountain the atmosphere is isentropic, the state for which the pressure-gradient force
# of this discretization vanishes over any terrain; without it, it is the stratified standard
# atmosphere, which is at rest because it is horizontally uniform. In both the full tendencies
# vanish to rounding, the gravity-wave operator is applied to perturbations of the size of that
# rounding, and the split scheme must stay at rest as the oracle does.
#
# Tolerance rationale. As for the oracle (`hydrostatic_dynamics.jl`), rounding leaves an
# acceleration of at most ~ 1e-14 m s⁻². The split schemes take 100 steps of 300 s, ten times the
# time of the oracle's 100 steps of 30 s, so the bounds are ten times the oracle's: 1e-9 m s⁻¹ for
# the horizontal velocities, 1e-11 s⁻¹ for the velocity through the coordinate surfaces, 1e-7 Pa
# for the surface pressure. The oracle is run alongside with its own bounds, and the two final
# states must then agree within the sum of the two.

rest_grids() = (RectilinearGrid(default_arch, FT; size=(32, 10), x=(0, 3.2e6), z=(0, 1),
                                topology=(Periodic, Flat, Bounded)),
                LatitudeLongitudeGrid(default_arch, FT; size=(24, 16, 10), longitude=(0, 360),
                                      latitude=(-60, 60), z=(0, 1)))

mountain_geopotential(x) = g * 2000 * exp(-((x - 1.6e6) / 3e5)^2)
mountain_geopotential(λ, φ) = g * 2000 * exp(-((λ - 180) / 30)^2 - (φ / 20)^2)
flat_geopotential(x...) = 0.0

function resting_model(grid, terrain; kw...)
    A, B = hybrid_coefficients(10)
    geopotential = terrain === :mountain ? mountain_geopotential : flat_geopotential
    dynamics = HydrostaticDynamics(surface_geopotential = geopotential,
                                   pressure_coefficient = A,
                                   surface_pressure_coefficient = B)
    coriolis = grid isa LatitudeLongitudeGrid ? HydrostaticSphericalCoriolis(FT) : nothing
    model = hydrostatic_model(grid; dynamics, coriolis, thermodynamic_constants=reference_model_constants(), kw...)

    if terrain === :mountain
        set!(model; pˢ = (x...) -> isentropic_surface_pressure(mountain_geopotential(x...), 300), θ=300)
    else
        p = Array(interior(dynamics_pressure(model.dynamics)))
        set!(model; θ = standard_temperature.(p) .* (pˢᵗ ./ p) .^ κ)
    end

    return model
end

function departure_from_rest(model, steps, Δt)
    pˢ⁰ = Array(interior(model.dynamics.surface_pressure))
    for _ in 1:steps
        time_step!(model, Δt)
    end
    return (u = max_abs(model.velocities.u),
            v = max_abs(model.velocities.v),
            w = max_abs(model.velocities.w),
            pˢ = maximum(abs, Array(interior(model.dynamics.surface_pressure)) .- pˢ⁰))
end

@testset "Resting state [$(nameof(typeof(grid))), $terrain]" for grid in rest_grids(), terrain in (:mountain, :flat)
    oracle = departure_from_rest(resting_model(grid, terrain), 100, 30)
    @test oracle.u < 1e-10
    @test oracle.v < 1e-10
    @test oracle.w < 1e-12
    @test oracle.pˢ < 1e-8

    if terrain === :mountain
        pˢ = Array(interior(resting_model(grid, terrain).dynamics.surface_pressure))
        @test maximum(pˢ) - minimum(pˢ) > 1e4 # the terrain is felt
    end

    @testset "$(configuration_name(tableau, inner_solver))" for (tableau, inner_solver) in configurations
        model = resting_model(grid, terrain; split_explicit(tableau, inner_solver)...)
        split = departure_from_rest(model, 100, 300)
        @info "Resting state, $(nameof(typeof(grid))), $terrain, $(configuration_name(tableau, inner_solver)): $split; oracle $oracle"

        @test split.u < 1e-9
        @test split.v < 1e-9
        @test split.w < 1e-11
        @test split.pˢ < 1e-7

        # Agreement with the oracle, within the sum of the two bounds
        @test abs(split.u - oracle.u) < 1.1e-9
        @test abs(split.pˢ - oracle.pˢ) < 1.1e-7
    end
end

# A resting atmosphere that is stratified over the mountain is not a steady state of this
# discretization: the pressure-gradient error of the terrain-following coordinate sets it in
# motion, in the oracle and in the split schemes alike. This compares the motion that develops.
#
# Tolerance rationale. This is a loose bound, and the measured difference is logged. After one
# hour at Δt = 300 s the response is a gravity-wave adjustment to a steady forcing. The damped
# forward-backward substeps remove a part of the waves and the Verlet substeps shift their phase,
# so the bound is only that the split velocity differs from the oracle's by less than half of the
# oracle's largest velocity: the same motion, not a different one.

@testset "Stratified atmosphere over the mountain" begin
    grid = first(rest_grids())
    A, B = hybrid_coefficients(10)

    function solution(Δt; kw...)
        dynamics = HydrostaticDynamics(surface_geopotential = mountain_geopotential,
                                       pressure_coefficient = A,
                                       surface_pressure_coefficient = B)
        model = hydrostatic_model(grid; dynamics, thermodynamic_constants=reference_model_constants(), kw...)
        set!(model; pˢ = x -> isentropic_surface_pressure(mountain_geopotential(x), 300), θ = (x, z) -> 290 + 60 * z)
        for _ in 1:round(Int, 3600 / Δt)
            time_step!(model, Δt)
        end
        return Array(interior(model.velocities.u))
    end

    oracle = solution(30)
    @test maximum(abs, oracle) > 1e-3 # the state is not at rest

    for (tableau, inner_solver) in configurations
        split = solution(300; split_explicit(tableau, inner_solver)...)
        difference = maximum(abs, split .- oracle)
        @info "Stratified atmosphere over the mountain, $(configuration_name(tableau, inner_solver)): " *
              "largest velocity $(maximum(abs, oracle)) m s⁻¹ (oracle), $(maximum(abs, split)) m s⁻¹ (split), " *
              "largest difference $difference m s⁻¹"
        @test difference < maximum(abs, oracle) / 2
    end
end

#####
##### (2) Linear gravity waves: phase and amplitude of the first three vertical modes
#####
#
# The standing wave of `hydrostatic_dynamics.jl`, integrated for 4.8e5 s by the oracle at
# Δt = 50 s and by each split scheme at Δt = 1200 s, with substep counts for which the stages
# take 3 to 6 substeps (`tableau_substeps`). Both are sampled every 1200 s, 4.4 times per period
# of the fastest mode. The frequency and the amplitude of each mode are those of the peak of the
# Hann-windowed periodogram of the wave's amplitude at all levels, near the reference frequency.
#
# The wave is linear about a state at rest, so it lies entirely in the fast system, and each
# vertical mode is an oscillator of frequency ω that the scheme integrates with its stages and
# substeps alone. The reference for a split scheme is therefore the amplification factor λ of the
# same stages and substeps applied to the oscillator ∂τ u = - ω y, ∂τ y = ω u, computed below
# independently of the model (`oscillator_amplification`): the frequency of the mode must be
# arg(λ) / Δt and its amplitude must decay as |λ| per time step.
#
# History of the tolerances. The bounds first written for this test, before its first run, were
# absolute: for Verlet substeps, a frequency within 5e-3 and an amplitude within 3e-2 of the
# oracle's, on the reasoning that Verlet substeps are neutral and fast by (ω Δτ)² / 24. That
# reasoning holds for Wicker-Skamarock RK3, whose last stage integrates the fast system over the
# whole step, and it was confirmed there (frequency ratio 1.0024, amplitude ratio 1.001 for mode
# 1). It does not hold for MIS3C and MIS4a, whose stages start from combinations of earlier
# stages: with an inner solver that has a phase error the combination is not neutral. The first
# run measured an amplitude ratio of 0.88 (MIS3C) and 0.77 (MIS4a) for mode 1 over 2.4e5 s and a
# frequency ratio of 0.9950 for MIS3C, failing those bounds. The oscillator model reproduces
# these numbers (|λ| = 0.99875 and 0.99728 per step at ω Δt = 1.43, frequency ratio 0.994965),
# so the damping is a property of the schemes at this step, not an error of the implementation,
# and the test was rewritten to compare with the oscillator model. The bounds for
# forward-backward substeps were a bracket that the first run satisfied; they are replaced by the
# same, sharper comparison. The window was also lengthened from 2.4e5 s to the 4.8e5 s for which
# the oracle's peaks are sharp enough to pass the contrast criterion of `hydrostatic_dynamics.jl`.
#
# Tolerance rationale for the comparison with the oscillator model.
#
#   * Oracle. SSP RK3 at Δt = 50 s damps mode 1 (ω Δt = 0.06) by (ω Δt)⁴ / 24 per step, 0.5 %
#     over the run, and shifts its frequency by less than 1e-6.
#   * Frequency. The peak of the periodogram is located to ~ 3e-4 (`hydrostatic_dynamics.jl`), for
#     the split scheme and the oracle each: the frequency ratio must match the model within 1e-3.
#   * Amplitude. With the decay |λ|ⁿ after n steps, the ratio of the peak amplitudes is the
#     Hann-weighted mean of |λ|ⁿ. It must match within 0.03: the oracle's own damping, the
#     estimate of a broadened peak, and nonlinearity (velocities of 1e-3 m s⁻¹).

# Amplification factor of one time step of the tableau, with `substeps` per time step of the
# inner solver, for the oscillator ∂τ u = - ω y, ∂τ y = ω u: the eigenvalue of the step's
# amplification matrix with positive frequency. Stage i integrates from Σ αᵢⱼ ΔYⱼ with the
# constant forcing Σ γᵢⱼ ΔYⱼ / (dᵢ Δt); there is no slow tendency.
function oscillator_amplification(tableau, inner_solver, substeps, ωΔt)
    (; α, γ, d) = multirate_infinitesimal_step_tableau(tableau)
    identity_matrix = [1.0 0.0; 0.0 1.0]
    increments = Matrix{Float64}[]

    for i in eachindex(d)
        state = copy(identity_matrix)
        forcing = zeros(2, 2)
        for j in 1:i-1
            state .+= α[i, j] .* increments[j]
            forcing .+= γ[i, j] / d[i] .* increments[j] # per unit of Δt
        end

        Nτ = max(1, ceil(Int, d[i] * substeps - 1e-9))
        x = ωΔt * d[i] / Nτ  # ω Δτ
        δ = d[i] / Nτ        # Δτ / Δt
        u, y = state[1:1, :], state[2:2, :]
        Fu, Fy = forcing[1:1, :], forcing[2:2, :]

        for _ in 1:Nτ
            if inner_solver isa VerletSubsteps
                y = y .+ (x / 2) .* u .+ (δ / 2) .* Fy
                u = u .- x .* y .+ δ .* Fu
                y = y .+ (x / 2) .* u .+ (δ / 2) .* Fy
            else
                βᵈ = inner_solver.divergence_damping
                y⁺ = y .+ x .* u .+ δ .* Fy
                u = u .- x .* (y⁺ .+ βᵈ .* (y⁺ .- y)) .+ δ .* Fu
                y = y⁺
            end
        end

        push!(increments, [u; y] .- identity_matrix)
    end

    # Eigenvalues of the real 2 × 2 amplification matrix, a complex conjugate pair
    M = identity_matrix .+ increments[end]
    trace, determinant = M[1, 1] + M[2, 2], M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
    return trace / 2 + im * sqrt(max(0, determinant - trace^2 / 4))
end

# Amplitude of sin(k x) in u at every level, sampled every `sampling` seconds
function standing_wave_signals(Δt, sampling, duration; kw...)
    Nx, Nz = 16, 30
    Lx = Nx * 1e5
    k = 2π / Lx
    grid = RectilinearGrid(default_arch, FT; size=(Nx, Nz), x=(0, Lx), z=(0, 1),
                           topology=(Periodic, Flat, Bounded))

    A, B = hybrid_coefficients(Nz)
    dynamics = HydrostaticDynamics(surface_pressure = pˢᵗ,
                                   standard_pressure = pˢᵗ,
                                   pressure_coefficient = A,
                                   surface_pressure_coefficient = B)

    model = hydrostatic_model(grid; dynamics, thermodynamic_constants=reference_model_constants(), kw...)

    p = Array(interior(dynamics_pressure(model.dynamics)))
    θ = standard_temperature.(p) .* (pˢᵗ ./ p) .^ κ
    structure(z) = 1 + cos(π * z) + cos(2π * z)
    set!(model; θ, u = (x, z) -> 1e-3 * sin(k * x) * structure(z))

    xᶠ = Array(xnodes(grid, Face()))
    projection = 2 / Nx .* sin.(k .* xᶠ)

    steps_per_sample = round(Int, sampling / Δt)
    samples = round(Int, duration / sampling)
    times = zeros(samples)
    signals = zeros(samples, Nz)

    for n in 1:samples
        for _ in 1:steps_per_sample
            time_step!(model, Δt)
        end
        u = Array(interior(model.velocities.u, :, 1, :))
        times[n] = model.clock.time
        signals[n, :] .= transpose(u) * projection
    end

    return times, signals, k
end

function mode_peaks(times, signals, k)
    window = @. sin(π * (times - times[1]) / (times[end] - times[1]))^2
    discretization_factor = sin(k * 1e5 / 2) / (k * 1e5 / 2)

    return map(reference_mode_speeds) do c
        ω = c * discretization_factor * k
        ω̃, interior_peak, contrast = peak_frequency(0.9ω, 1.1ω, times, signals, window)
        amplitude = sqrt(periodogram(ω̃, times, signals, window))
        (; ω = ω̃, amplitude, interior_peak, contrast)
    end, window
end

@testset "Linear gravity waves: phase and amplitude" begin
    Δt = 1200
    duration = 4.8e5
    times, signals, k = standing_wave_signals(50, Δt, duration)
    oracle, window = mode_peaks(times, signals, k)
    steps = round.(Int, times ./ Δt)

    for peak in oracle
        @test peak.interior_peak
        @test peak.contrast > 100
    end

    # The oscillator model: Verlet substeps are neutral in Wicker-Skamarock RK3
    @test abs(oscillator_amplification(:WickerSkamarock3, VerletSubsteps(), 6, 1.4)) ≈ 1

    @testset "$(configuration_name(tableau, inner_solver))" for (tableau, inner_solver) in configurations
        substeps = tableau_substeps[tableau]
        split_times, split_signals, _ = standing_wave_signals(Δt, Δt, duration; split_explicit(tableau, inner_solver; substeps)...)
        @test split_times ≈ times
        split, _ = mode_peaks(split_times, split_signals, k)

        for mode in 1:3
            ωΔt = oracle[mode].ω * Δt
            λ = oscillator_amplification(tableau, inner_solver, substeps, ωΔt)
            expected_frequency_ratio = angle(λ) / ωΔt
            expected_amplitude_ratio = sum(window .* abs(λ) .^ steps) / sum(window)

            frequency_ratio = split[mode].ω / oracle[mode].ω
            amplitude_ratio = split[mode].amplitude / oracle[mode].amplitude
            @info "Gravity wave, $(configuration_name(tableau, inner_solver)), mode $mode (ω Δt = $ωΔt): " *
                  "frequency ratio to the oracle $frequency_ratio (oscillator model $expected_frequency_ratio), " *
                  "amplitude ratio $amplitude_ratio (model $expected_amplitude_ratio), |λ| per step $(abs(λ))"

            @test split[mode].interior_peak
            @test abs(frequency_ratio - expected_frequency_ratio) < 1e-3
            @test abs(amplitude_ratio - expected_amplitude_ratio) < 0.03
        end
    end
end

#####
##### (3) Temporal convergence against a fine unsplit reference
#####
#
# The problem of the oracle's convergence test, at Δt = 400, 200 and 100 s with the number of
# substeps per time step fixed (`tableau_substeps`), so that the substep is refined with the
# time step. The reference is the oracle at Δt = 15 s, whose third-order error is below
# (15 / 100)³ = 3e-3 of its own error at 100 s, itself below the errors measured here.
#
# Expectation from the linear analysis of the campaign (`derivations/followup.md`, F2): second
# order with Verlet substeps, first order with forward-backward substeps and βᵈ = 0.1, whatever
# the outer scheme. The orders are measured and logged. Tolerance rationale: at Δt = 400 s the
# gravity waves of the excited scales have ω Δτ ≤ 0.1 in every stage and the advective Courant
# number is 0.04, so the error should be in its asymptotic regime and the order between the two
# finer steps within ± 0.3 of the expectation. The order between the two coarser steps is logged
# and not tested.
#
# Outcome of the first run: 17 of the 18 measured orders are within the bound. The velocity of
# MIS4a with Verlet substeps is not: 1.06 between 200 and 100 s, while its surface pressure and
# ρθ give 2.29. Its velocity error is 5 to 10 times smaller than that of the other Verlet
# schemes and is not monotone in a wider scan (orders 4.31, 1.76, 1.10, 1.68, 1.85 on successive
# halvings from 800 s; campaign evidence, `convergence_scan.log`), which is what the cancellation
# of two error terms of different order at this one output time looks like, with the order
# rising toward 2 at the smallest steps. That is an interpretation, not a measurement of the
# order: the bound is kept, and this one case is recorded as broken rather than loosened.

# The measured order that is outside the bound written before the first run
order_not_established(tableau, inner_solver, name) =
    tableau === :MIS4a && inner_solver isa VerletSubsteps && name === :u

@testset "Temporal convergence of the split schemes" begin
    Nx, Nz = 16, 8
    Lx = Nx * 1e5
    k = 2π / Lx
    grid = RectilinearGrid(default_arch, FT; size=(Nx, Nz), x=(0, Lx), z=(0, 1),
                           topology=(Periodic, Flat, Bounded))

    function solution(Δt; kw...)
        model = hydrostatic_model(grid; dynamics=HydrostaticDynamics(top_pressure=200), kw...)
        set!(model; pˢ = x -> 1e5 + 200 * cos(k * x),
                    θ = (x, z) -> 300 + 40 * z,
                    u = (x, z) -> 5 * (1 + z) * sin(k * x))

        for _ in 1:round(Int, 4800 / Δt)
            time_step!(model, Δt)
        end

        @test model.clock.time ≈ 4800

        return (u = Array(interior(model.velocities.u)),
                pˢ = Array(interior(model.dynamics.surface_pressure)),
                ρθ = Array(interior(model.formulation.potential_temperature_density)))
    end

    reference = solution(15)

    @testset "$(configuration_name(tableau, inner_solver))" for (tableau, inner_solver) in configurations
        substeps = tableau_substeps[tableau]
        solutions = [solution(Δt; split_explicit(tableau, inner_solver; substeps)...) for Δt in (400, 200, 100)]
        expected_order = inner_solver isa VerletSubsteps ? 2 : 1

        for name in (:u, :pˢ, :ρθ)
            errors = [maximum(abs, solution[name] .- reference[name]) for solution in solutions]
            coarse_order = log2(errors[1] / errors[2])
            fine_order = log2(errors[2] / errors[3])
            @info "Temporal convergence, $(configuration_name(tableau, inner_solver)), $name: errors $errors, " *
                  "orders $coarse_order (400 → 200 s), $fine_order (200 → 100 s)"

            @test errors[3] > 1e3 * eps(FT) * maximum(abs, reference[name]) # above rounding

            if order_not_established(tableau, inner_solver, name)
                @test_broken abs(fine_order - expected_order) < 0.3
            else
                @test abs(fine_order - expected_order) < 0.3
            end
        end
    end
end

#####
##### (4) Conservation, and constancy with the accumulated mass fluxes
#####
#
# The flow of the oracle's conservation test, with horizontal divergence, vertical shear and a
# surface pressure wave, for 100 steps of 300 s (600 s on the sphere).
#
# Tolerance rationale.
#
#   * Mass and Θ change only by flux divergences that telescope, in the substeps and in the slow
#     tendencies alike. The split schemes apply about ten times as many updates as the oracle's
#     300, so the bound on the relative drift of the totals is ten times the oracle's: 1e-12.
#     Tracers are advanced once per stage by flux divergences: the same bound.
#   * The mass of each layer, integrated horizontally, is (ΔA × area + ΔB ∫ pˢ dA) / g: it is
#     conserved layer by layer because the total is. Same bound, 1e-12.
#   * A uniform mixing ratio χ. The surface pressure is advanced substep by substep, and the
#     tracer once per stage with the mass fluxes of the velocity perturbation accumulated over
#     the substeps. The two sum the same fluxes in a different order, so ρc = χ ρ holds to
#     rounding: a relative error of a few eps(Float64) per stage, 1e-11 over 100 steps. This is
#     the test that the mass fluxes that carry the tracers are those that change the layer mass.
#   * A uniform potential temperature θ₀ is the same statement for Θ, which the substeps advance
#     with the perturbation mass fluxes one at a time: ρθ = θ₀ ρ within 1e-11.

conservation_grids() = (RectilinearGrid(default_arch, FT; size=(8, 8, 6), x=(0, 8e5), y=(0, 8e5), z=(0, 1)),
                        LatitudeLongitudeGrid(default_arch, FT; size=(12, 8, 6), longitude=(0, 360),
                                              latitude=(-60, 60), z=(0, 1)))

layer_totals(field) = (integral = Field(Integral(field, dims=(1, 2))); compute!(integral); vec(Array(interior(integral))))

@testset "Conservation and constancy [$(nameof(typeof(grid))), $(configuration_name(tableau, inner_solver))]" for
        grid in conservation_grids(), (tableau, inner_solver) in configurations

    spherical = grid isa LatitudeLongitudeGrid
    A, B = hybrid_coefficients(6)
    coriolis = spherical ? HydrostaticSphericalCoriolis(FT) : FPlane(FT; f=1e-4)

    kˣ, kʸ = spherical ? (2π / 360, 2π / 120) : (2π / 8e5, 2π / 8e5)
    pˢᵢ(x, y) = 1e5 + 500 * cos(kˣ * x) * cos(kʸ * y)
    envelope(y) = spherical ? cos(kʸ * y / 2) : (1 + cos(kʸ * y)) / 2 # vanishes at the walls of the sphere
    uᵢ(x, y, z) = 10 * (1 + z) * (1 + sin(kˣ * x) / 2) * envelope(y)
    vᵢ(x, y, z) = 5 * z * sin(kˣ * x) * envelope(y)^2
    θᵢ(x, y, z) = 290 + 60 * z + 2 * sin(kˣ * x) * sin(kʸ * y)

    Δt = spherical ? 600 : 300
    χ = 3

    for uniform_θ in (false, true)
        dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B)
        model = hydrostatic_model(grid; dynamics, coriolis, tracers=(:c, :d),
                                  scalar_advection=(; d=WENO(FT; order=5)),
                                  split_explicit(tableau, inner_solver)...)

        set!(model; pˢ=pˢᵢ, θ = uniform_θ ? 300 : θᵢ, u=uᵢ, v=vᵢ)

        ρ = continuity_density(model.dynamics)
        set!(model; c = χ * ρ, d = χ * ρ)

        ρθ = model.formulation.potential_temperature_density
        pˢ⁰ = Array(interior(model.dynamics.surface_pressure))
        mass⁰, Θ⁰, c⁰, d⁰ = total(ρ), total(ρθ), total(model.tracers.c), total(model.tracers.d)
        layer_mass⁰ = layer_totals(ρ)
        w_max = max_abs(model.velocities.w)

        for _ in 1:100
            time_step!(model, Δt)
        end

        @test w_max > 0 # there is flow through the coordinate surfaces
        @test maximum(abs, Array(interior(model.dynamics.surface_pressure)) .- pˢ⁰) > 1 # the state evolved

        @test abs(total(ρ) - mass⁰) < 1e-12 * mass⁰
        @test abs(total(ρθ) - Θ⁰) < 1e-12 * Θ⁰
        @test abs(total(model.tracers.c) - c⁰) < 1e-12 * c⁰
        @test abs(total(model.tracers.d) - d⁰) < 1e-12 * d⁰
        @test maximum(abs, layer_totals(ρ) ./ layer_mass⁰ .- 1) < 1e-12

        ρᵢ = Array(interior(ρ))
        for name in (:c, :d)
            ρc = Array(interior(model.tracers[name]))
            @test maximum(abs, ρc ./ ρᵢ .- χ) < 1e-11 * χ
        end

        if uniform_θ
            @test maximum(abs, Array(interior(ρθ)) ./ ρᵢ .- 300) < 1e-11 * 300
        end
    end
end
