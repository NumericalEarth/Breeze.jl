include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.HydrostaticEquations: HydrostaticEquations, HydrostaticModel, VerletSubsteps
using Oceananigans
using Oceananigans.Advection: VectorInvariant
using Oceananigans.TimeSteppers: update_state!
using Test

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

#####
##### Construction
#####

@testset "HydrostaticDynamics construction" begin
    grid = RectilinearGrid(default_arch, FT; size=(4, 4, 5), x=(0, 1e5), y=(0, 1e5), z=(0, 1))

    model = hydrostatic_model(grid; dynamics=HydrostaticDynamics(top_pressure=200))
    @test model isa HydrostaticModel
    @test keys(Oceananigans.prognostic_fields(model)) == (:pˢ, :u, :v, :ρθ, :ρqᵛ)
    @test model.velocities.u === model.momentum.u

    # The layer masses sum to the mass of the column: (pˢ - top_pressure) / g
    ρ = Array(interior(continuity_density(model.dynamics)))
    Δz = 1 / 5
    constants = model.thermodynamic_constants
    @test sum(ρ[1, 1, :]) * Δz ≈ (101325 - 200) / constants.gravitational_acceleration

    @test_throws ArgumentError AtmosphereModel(grid; dynamics=HydrostaticDynamics())
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=HydrostaticDynamics(), formulation=:StaticEnergy)
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=HydrostaticDynamics(), timestepper=:RungeKutta3)
    @test_throws ArgumentError HydrostaticDynamics(pressure_coefficient=[0, 1])

    A, B = hybrid_coefficients(5)
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=HydrostaticDynamics(pressure_coefficient=A[1:5],
                                                                                    surface_pressure_coefficient=B[1:5]))
end

#####
##### Unsupported configurations are rejected
#####

@testset "HydrostaticDynamics rejects what it does not support" begin
    grid = RectilinearGrid(default_arch, FT; size=(4, 4, 5), x=(0, 1e5), y=(0, 1e5), z=(0, 1))
    dynamics() = HydrostaticDynamics(top_pressure=200)

    # A Coriolis force with a horizontal component of the rotation would be handed the velocity
    # through the coordinate surfaces in place of the vertical velocity
    for coriolis in (ConstantCartesianCoriolis(FT; fx=0, fy=1e-4, fz=1e-4),
                     NonTraditionalBetaPlane(FT; latitude=45),
                     SphericalCoriolis(FT))
        @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), coriolis)
    end

    for coriolis in (nothing, FPlane(FT; f=1e-4), BetaPlane(FT; f₀=1e-4, β=1e-11))
        @test hydrostatic_model(grid; dynamics=dynamics(), coriolis) isa HydrostaticModel
    end

    # Forcings of the velocities whose effect has not been verified
    acceleration(x, y, z, t) = 1e-4
    discrete_acceleration(i, j, k, grid, clock, fields) = 1e-4
    dependent_acceleration(x, y, z, t, u) = - 1e-4 * u
    unverified = (Forcing(discrete_acceleration, discrete_form=true),
                  Forcing(dependent_acceleration, field_dependencies=:u),
                  Forcing((x, y, z, t, a) -> a, parameters=1e-4),
                  (acceleration, Forcing(discrete_acceleration, discrete_form=true)),
                  SubsidenceForcing(z -> 0),
                  geostrophic_forcings(z -> 10, z -> 0).u)

    for forcing in unverified, key in (:u, :ρu, :v, :ρv)
        @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), coriolis=FPlane(FT; f=1e-4),
                                                     forcing=NamedTuple{(key,)}((forcing,)))
    end

    # A forcing that returns a density-weighted tendency may not be supplied as an acceleration
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), forcing=(; u=held_suarez_forcings().ρu))

    # There is no vertical momentum to force
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), forcing=(; ρw=acceleration))
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), forcing=(; w=acceleration))

    for key in (:u, :ρu, :v, :ρv), forcing in (acceleration, (acceleration, acceleration))
        @test hydrostatic_model(grid; dynamics=dynamics(), forcing=NamedTuple{(key,)}((forcing,))) isa HydrostaticModel
    end

    # The equations are dry: water may be neither set nor forced
    model = hydrostatic_model(grid; dynamics=dynamics())
    @test_throws ArgumentError set!(model; θ=300, qᵗ=1e-3)
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), forcing=(; ρqᵗ=acceleration))
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(), microphysics=SaturationAdjustment())
end

#####
##### No route by which water enters the dry model
#####
#
# Regression for a hole in the guards. Before it was closed, a flux of moisture through a
# boundary, supplied under `ρqᵗ` or `ρqᵛ`, was accepted, and ten steps with a flux of
# 1e-6 kg m⁻² s⁻¹ left a moisture density of 5e-4 in the dry model, with SSP RK3 and with the
# split-explicit stepper alike (campaign evidence, `guard_probe_BEFORE_fix.log`). A forcing of
# the moisture was rejected whatever its form, `Returns(value)` included, but only because a
# forcing that the user supplies is wrapped: the guard itself exempted every `Returns` by its
# type. Three guards now keep the model dry, and each is tested:
#
#   1. the forcing of the moisture density must be identically zero, by value;
#   2. a boundary condition of the moisture density must carry no condition or a zero one;
#   3. the moisture density must be zero each time the state is updated, so that water that
#      enters by a route the first two do not cover stops the integration.

const dry_timesteppers = ((;), (timestepper = :GravityWaveMultirateInfinitesimalStep,
                                timestepper_kwargs = (tableau = :MIS3C, inner_solver = VerletSubsteps())))

@testset "Moisture sources are rejected [$(isempty(timestepper) ? :SSPRungeKutta3 : timestepper.timestepper)]" for
        timestepper in dry_timesteppers

    grid = RectilinearGrid(default_arch, FT; size=(4, 4, 5), x=(0, 4e5), y=(0, 4e5), z=(0, 1))
    build(; kw...) = hydrostatic_model(grid; dynamics=HydrostaticDynamics(top_pressure=200), timestepper..., kw...)

    # (1) Forcing, under the interface key, its specific alias and the prognostic key, by value
    for key in (:ρqᵗ, :qᵗ, :ρqᵛ), forcing in (Returns(1e-6), Returns(-1e-6), (x, y, z, t) -> 1e-6)
        @test_throws ArgumentError build(forcing=NamedTuple{(key,)}((forcing,)))
    end

    # The guard reads the value of a `Returns`, not its type
    @test HydrostaticEquations.is_identically_zero(Returns(0.0))
    @test !HydrostaticEquations.is_identically_zero(Returns(1e-6))
    @test !HydrostaticEquations.is_identically_zero((x, y, z, t) -> 0.0)

    # (2) Boundary sources: fluxes, values and gradients, on either key
    sources = (FluxBoundaryCondition(-1e-6), FluxBoundaryCondition(1e-6), ValueBoundaryCondition(1e-3),
               GradientBoundaryCondition(1e-3), FluxBoundaryCondition((x, y, t) -> 1e-6))
    for key in (:ρqᵗ, :ρqᵛ), source in sources, side in (:bottom, :top)
        conditions = FieldBoundaryConditions(; NamedTuple{(side,)}((source,))...)
        @test_throws ArgumentError build(boundary_conditions=NamedTuple{(key,)}((conditions,)))
    end

    # No source and a flux that is identically zero are kept, and the model stays dry
    for model in (build(), build(boundary_conditions=(; ρqᵗ=FieldBoundaryConditions(bottom=FluxBoundaryCondition(0.0)))))
        set!(model; θ=300, u=1)
        for _ in 1:10
            time_step!(model, 10)
        end
        @test iszero(minimum(model.moisture_density))
        @test iszero(maximum(model.moisture_density))
    end

    # (3) The evolving state: water that appears by any other route stops the time step, in
    # the first `update_state!` that sees it, and the model can be made dry again
    model = build()
    set!(model; θ=300)
    time_step!(model, 10)
    fill!(model.moisture_density, 1e-9)
    @test_throws ArgumentError time_step!(model, 10)
    @test_throws ArgumentError update_state!(model)

    fill!(model.moisture_density, 0)
    @allowscalar interior(model.moisture_density)[2, 2, 2] = -1e-12 # one cell, either sign
    @test_throws ArgumentError update_state!(model)

    fill!(model.moisture_density, 0)
    update_state!(model)
    time_step!(model, 10)
    @test iszero(maximum(abs, model.moisture_density))
end

#####
##### Every layer of every column must have a positive mass
#####
#
# The hybrid coefficients are validated at the standard pressure, 1e5 Pa, where the z-faces of
# this coordinate are at 1e5, 9.5e4 and 200 Pa. Under a surface pressure of 8e4 Pa they are at
# 8e4, 9.4e4 and 200 Pa: the lower layer has the pressure thickness -1.4e4 Pa.

@testset "Layers without mass are rejected" begin
    grid = RectilinearGrid(default_arch, FT; size=(4, 4, 2), x=(0, 1e5), y=(0, 1e5), z=(0, 1))
    A = [0, 90000, 200]
    B = [1, 0.05, 0]
    dynamics(pˢ) = HydrostaticDynamics(surface_pressure=pˢ, pressure_coefficient=A, surface_pressure_coefficient=B)

    # Valid at the standard pressure, invalid from the start at 8e4 Pa
    @test hydrostatic_model(grid; dynamics=dynamics(1e5)) isa HydrostaticModel
    @test_throws ArgumentError hydrostatic_model(grid; dynamics=dynamics(8e4))

    # Invalid when set, in one column only
    model = hydrostatic_model(grid; dynamics=dynamics(1e5))
    @test_throws ArgumentError set!(model; pˢ=8e4, θ=300)
    @test_throws ArgumentError set!(model; pˢ = (x, y) -> x < 2e4 && y < 2e4 ? 8e4 : 1e5, θ=300)

    # Invalid during a time step: the diagnosis of the layer mass stops the integration
    set!(model; pˢ=1e5, θ=300)
    @test minimum(continuity_density(model.dynamics)) > 0
    fill!(model.dynamics.surface_pressure, 8e4)
    @test_throws DomainError time_step!(model, 1)
end

#####
##### (1) A resting isentropic atmosphere stays at rest
#####
#
# With uniform θ = θ₀ the pressure-gradient acceleration is -∇(Φˢ + cᵖᵈ θ₀ Πˢ) for any terrain and
# any hybrid coefficients, so it vanishes when the surface pressure satisfies Φˢ + cᵖᵈ θ₀ Πˢ =
# constant. Tolerance rationale: the acceleration is a difference of terms of size cᵖᵈ θ₀ Π ~ 3e5
# m² s⁻² over Δx ≥ 5e4 m, so rounding leaves at most ~ 3e5 × 1e-15 / 5e4 ~ 1e-14 m s⁻². Over 100
# steps of 30 s that is a velocity below 1e-10 m s⁻¹.

@testset "Resting isentropic atmosphere over terrain [$(nameof(typeof(grid)))]" for grid in (
        RectilinearGrid(default_arch, FT; size=(32, 10), x=(0, 3.2e6), z=(0, 1),
                        topology=(Periodic, Flat, Bounded)),
        LatitudeLongitudeGrid(default_arch, FT; size=(24, 16, 10), longitude=(0, 360),
                              latitude=(-60, 60), z=(0, 1)))

    θ₀ = 300
    A, B = hybrid_coefficients(10)

    # A 2 km mountain
    mountain(x) = g * 2000 * exp(-((x - 1.6e6) / 3e5)^2)
    mountain(λ, φ) = g * 2000 * exp(-((λ - 180) / 30)^2 - (φ / 20)^2)
    balanced_surface_pressure(x...) = isentropic_surface_pressure(mountain(x...), θ₀)

    dynamics = HydrostaticDynamics(surface_geopotential = mountain,
                                   pressure_coefficient = A,
                                   surface_pressure_coefficient = B)

    coriolis = grid isa LatitudeLongitudeGrid ? HydrostaticSphericalCoriolis(FT) : nothing
    model = hydrostatic_model(grid; dynamics, coriolis, thermodynamic_constants=reference_model_constants())
    set!(model; pˢ=balanced_surface_pressure, θ=θ₀)

    pˢ⁰ = Array(interior(model.dynamics.surface_pressure))
    @test maximum(pˢ⁰) - minimum(pˢ⁰) > 1e4 # the terrain is felt

    for _ in 1:100
        time_step!(model, 30)
    end

    @test max_abs(model.velocities.u) < 1e-10
    @test max_abs(model.velocities.v) < 1e-10
    @test max_abs(model.velocities.w) < 1e-12
    pˢ = Array(interior(model.dynamics.surface_pressure))
    @test maximum(abs, pˢ .- pˢ⁰) < 1e-8 # Pa, of 1e5 Pa
end

#####
##### (2) Conservation of dry mass and Θ, and (3) constancy of a uniform tracer
#####
#
# Tolerance rationale. Mass and Θ change only by flux divergences that telescope, so each of the
# 300 stage updates leaves a relative error of a few eps(Float64) = 2e-16 in the totals, as does
# the summation that measures them: 1e-13 bounds the accumulated drift. A uniform mixing ratio χ
# is advected by the mass fluxes of the continuity equation, so ρc = χ ρ is preserved by each
# stage to rounding and χ may drift by ~ 300 × 2e-16 < 1e-12.

@testset "Conservation and constancy [$(nameof(typeof(grid)))]" for grid in (
        RectilinearGrid(default_arch, FT; size=(8, 8, 6), x=(0, 8e5), y=(0, 8e5), z=(0, 1)),
        LatitudeLongitudeGrid(default_arch, FT; size=(12, 8, 6), longitude=(0, 360),
                              latitude=(-60, 60), z=(0, 1)))

    spherical = grid isa LatitudeLongitudeGrid
    A, B = hybrid_coefficients(6)
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B)
    coriolis = spherical ? HydrostaticSphericalCoriolis(FT) : FPlane(FT; f=1e-4)

    model = hydrostatic_model(grid; dynamics, coriolis, tracers=(:c, :d),
                              scalar_advection=(; d=WENO(FT; order=5)))

    # A flow with horizontal divergence, vertical shear, and a surface pressure wave
    kˣ, kʸ = spherical ? (2π / 360, 2π / 120) : (2π / 8e5, 2π / 8e5)
    pˢᵢ(x, y) = 1e5 + 500 * cos(kˣ * x) * cos(kʸ * y)
    envelope(y) = spherical ? cos(kʸ * y / 2) : (1 + cos(kʸ * y)) / 2 # vanishes at the walls of the sphere
    uᵢ(x, y, z) = 10 * (1 + z) * (1 + sin(kˣ * x) / 2) * envelope(y)
    vᵢ(x, y, z) = 5 * z * sin(kˣ * x) * envelope(y)^2
    θᵢ(x, y, z) = 290 + 60 * z + 2 * sin(kˣ * x) * sin(kʸ * y)

    set!(model; pˢ=pˢᵢ, θ=θᵢ, u=uᵢ, v=vᵢ)

    # A uniform mixing ratio χ is the tracer density χ ρ
    χ = 3
    ρ = continuity_density(model.dynamics)
    set!(model; c = χ * ρ, d = χ * ρ)

    ρθ = model.formulation.potential_temperature_density
    pˢ⁰ = Array(interior(model.dynamics.surface_pressure))
    mass⁰ = total(ρ)
    Θ⁰ = total(ρθ)
    w_max = max_abs(model.velocities.w)

    Δt = spherical ? 200 : 20
    for _ in 1:100
        time_step!(model, Δt)
    end

    @test w_max > 0                                   # there is flow through the coordinate surfaces
    @test abs(total(ρ) - mass⁰) < 1e-13 * mass⁰
    @test abs(total(ρθ) - Θ⁰) < 1e-13 * Θ⁰

    ρ = Array(interior(ρ))
    for name in (:c, :d)
        ρc = Array(interior(model.tracers[name]))
        @test maximum(abs, ρc ./ ρ .- χ) < 1e-12 * χ
    end

    # The state did evolve: the surface pressure moved away from its initial values
    @test maximum(abs, Array(interior(model.dynamics.surface_pressure)) .- pˢ⁰) > 1
end

#####
##### (4) Linear gravity waves: phase speeds of the first three vertical modes
#####
#
# The reference speeds are the three largest eigenvalues of the vertical-structure problem of this
# discretization for a resting standard atmosphere on 30 hybrid levels (the campaign's
# one-dimensional reference model, `vertical_modes.csv`, rows N = 30, state = standard): they
# belong to the same vertical discretization, constants, levels and base state as the model below,
# in the limit of a continuous horizontal coordinate and time.
#
# Tolerance rationale, fixed before the test was first run:
#
#   * Horizontal discretization. On the C-grid a wave of wavenumber k has frequency ω = c k̃ with
#     k̃ = (2 / Δx) sin(k Δx / 2), so the measured speed ω / k is the reference times
#     sin(k Δx / 2) / (k Δx / 2) = 0.99359 for k Δx = π / 8: 0.641 % slow. This is the only
#     discrepancy that is not small.
#   * Time discretization. SSP RK3 has a relative phase error (ω Δt)⁴ / 30 < 1e-5 at ω Δt ≤ 0.12.
#   * Nonlinearity. Velocities of 1e-3 m s⁻¹ against speeds above 100 m s⁻¹: below 1e-5.
#   * Frequency estimate. The peak of a Hann-windowed periodogram over 4e5 s (27 periods of mode
#     3), with the nearest other mode more than 8 resolution widths away: below 3e-4.
#
# So the measured speed, divided by the factor 0.99359, must match the reference within 1e-3, and
# the raw speed must be slow by 0.64 % ± 0.1 %.

@testset "Linear gravity-wave phase speeds" begin
    Nx, Nz = 16, 30
    Δx = 1e5
    Lx = Nx * Δx
    k = 2π / Lx

    grid = RectilinearGrid(default_arch, FT; size=(Nx, Nz), x=(0, Lx), z=(0, 1),
                           topology=(Periodic, Flat, Bounded))

    A, B = hybrid_coefficients(Nz)
    dynamics = HydrostaticDynamics(surface_pressure = pˢᵗ,
                                   standard_pressure = pˢᵗ,
                                   pressure_coefficient = A,
                                   surface_pressure_coefficient = B)

    model = hydrostatic_model(grid; dynamics, thermodynamic_constants=reference_model_constants())

    # Resting standard atmosphere: θ sampled at the layer pressure
    p = Array(interior(dynamics_pressure(model.dynamics)))
    θ = standard_temperature.(p) .* (pˢᵗ ./ p) .^ κ

    # A small standing wave in u, with a vertical structure that projects on several modes
    structure(z) = 1 + cos(π * z) + cos(2π * z)
    uᵢ(x, z) = 1e-3 * sin(k * x) * structure(z)
    set!(model; θ, u=uᵢ)

    Δt = 100
    Nt = 4000
    xᶠ = Array(xnodes(grid, Face()))
    projection = 2 / Nx .* sin.(k .* xᶠ)

    times = zeros(Nt)
    signals = zeros(Nt, Nz) # amplitude of sin(k x) in u at each level
    for n in 1:Nt
        time_step!(model, Δt)
        u = Array(interior(model.velocities.u, :, 1, :))
        times[n] = model.clock.time
        signals[n, :] .= transpose(u) * projection
    end

    window = @. sin(π * (times - times[1]) / (times[end] - times[1]))^2
    discretization_factor = sin(k * Δx / 2) / (k * Δx / 2)

    for (mode, c) in enumerate(reference_mode_speeds)
        ω = c * discretization_factor * k
        ω̃, interior_peak, contrast = peak_frequency(0.9ω, 1.1ω, times, signals, window)
        c̃ = ω̃ / k
        @info "Vertical mode $mode: measured $(c̃) m s⁻¹, reference $c m s⁻¹, " *
              "ratio $(c̃ / c), corrected ratio $(c̃ / (c * discretization_factor))"

        @test interior_peak   # a peak, not the edge of the search window
        @test contrast > 100  # that stands out
        @test abs(c̃ / (c * discretization_factor) - 1) < 1e-3
        @test abs(1 - c̃ / c - 0.00641) < 1e-3
    end
end

#####
##### (5) Temporal convergence of the unsplit reference
#####
#
# SSP RK3 is third order, so the difference between solutions at Δt and Δt / 2 is 8 times the
# difference between those at Δt / 2 and Δt / 4, up to higher-order terms. Tolerance rationale:
# the fastest resolved wave has ω Δt ≈ 2 × 306 / Δx × 120 ≈ 0.73 at the largest step, where the
# next term of the error expansion is relatively O(ω Δt) of the leading one for the grid-scale
# waves but the excited scales have ω Δt ≤ 0.15; the measured order should be 3 within ± 0.2.
#
# The solutions are compared after 4800 s, most of a period of the fastest excited wave. After
# 1200 s the velocity of this standing wave is where its leading error term is small, and its
# measured order approaches 3 from above only slowly (3.26, 3.15, 3.08 on successive halvings
# from Δt = 120 s), while the surface pressure and ρθ are already within 0.05 of 3.

@testset "Temporal convergence" begin
    Nx, Nz = 16, 8
    Lx = Nx * 1e5
    k = 2π / Lx
    grid = RectilinearGrid(default_arch, FT; size=(Nx, Nz), x=(0, Lx), z=(0, 1),
                           topology=(Periodic, Flat, Bounded))

    function solution(Δt)
        model = hydrostatic_model(grid; dynamics=HydrostaticDynamics(top_pressure=200))
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

    coarse, medium, fine = solution(120), solution(60), solution(30)

    for name in (:u, :pˢ, :ρθ)
        coarse_difference = maximum(abs, coarse[name] .- medium[name])
        fine_difference = maximum(abs, medium[name] .- fine[name])
        order = log2(coarse_difference / fine_difference)
        @info "Temporal convergence of $name: differences $coarse_difference, $fine_difference, order $order"

        @test fine_difference > 1e3 * eps(FT) * maximum(abs, fine[name]) # above rounding
        @test abs(order - 3) < 0.2
    end
end
