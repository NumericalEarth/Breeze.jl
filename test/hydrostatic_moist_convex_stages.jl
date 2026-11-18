include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.HydrostaticEquations: VerletSubsteps, column_physics_in_time_step!
using Breeze.TimeSteppers: ConvexScalarStages, RecombinedScalarStages, convex_scalar_stages,
                           largest_euler_step_fraction, multirate_infinitesimal_step_tableau
using Oceananigans
using Oceananigans.Advection: UpwindBiased, BoundsPreservation
using Test

include("hydrostatic_moist_utils.jl")

#####
##### Convex scalar stages of the split-explicit stepper with moist HydrostaticDynamics
#####
#
# With a tableau whose stages are convex combinations of the state at the start of the step and
# of one forward Euler step from the stage before, the split-explicit stepper transports the
# water and the tracers once per stage, with the mass flux that changes the layer mass over the
# stage. Expectations, written before the first run of the tests:
#
#   - the water and a tracer between 0 and 1 stay within the rounding of their bounds with a
#     transport scheme that preserves them over a forward Euler step (first-order upwind, WENO
#     with a flux limiter), at a time step at which the recombined stages of :MIS3C do not;
#   - water (with the rain that reached the ground), dry mass and tracer are conserved to rounding;
#   - uniform mixing ratios of the water and of a tracer stay uniform, when nothing rains.
#
# The bounds are the 64 eps × largest value of the positivity rule of the moist validation and
# the 1e-12 of the constancy test of `hydrostatic_moist.jl`.

const rounding_bound = 64 * eps(Float64)
const budget_bound = 1e-12

convex_ridge(x) = 9.80665 * 1500 * exp(-((x - 8e5) / 2e5)^2)

split_stepper(tableau) = (timestepper = :GravityWaveMultirateInfinitesimalStep,
                          timestepper_kwargs = (; tableau, inner_solver = VerletSubsteps(), moist_gravity_wave_operator = true))

function convex_slice(water_scheme; tableau=:ConvexLinearMIS6, microphysics=warm_rain_microphysics())
    grid = RectilinearGrid(CPU(), Float64; size=(32, 12), x=(0, 1.6e6), z=(0, 1), halo=(5, 5),
                           topology=(Periodic, Flat, Bounded))
    dynamics = HydrostaticDynamics(; surface_geopotential=convex_ridge, top_pressure=2.5e4)
    scalar_advection = isnothing(microphysics) ? (ρθ = WENO(order=5), c = water_scheme()) :
                       (ρθ = WENO(order=5), ρqᵉ = water_scheme(), ρqʳ = water_scheme(), c = water_scheme())
    kwargs = isnothing(microphysics) ? (timestepper = :GravityWaveMultirateInfinitesimalStep,
                                        timestepper_kwargs = (; tableau, inner_solver = VerletSubsteps())) : split_stepper(tableau)
    return AtmosphereModel(grid; dynamics, microphysics, tracers=(:c,), scalar_advection,
                           formulation = moist_formulation(), momentum_advection = VectorInvariant(), kwargs...)
end

# Humid air, a box of rain and a box of tracer, all with sharp edges, in a flow over a ridge
function set_boxes!(model)
    box(x, lower, upper) = lower < x < upper
    rain(x, z) = box(x, 5e5, 7e5) * box(z, 0.2, 0.5) * 2e-3
    set!(model; pˢ=1e5, θ = (x, z) -> 288 + 45z + 2 * exp(-((x - 4e5) / 2e5)^2), u=20)
    set!(model; qᵗ = (x, z) -> box(x, 2e5, 9e5) * 0.016 * exp(-5z) + rain(x, z), qʳ = rain)
    ρ = Array(interior(model.dynamics.density))
    x = Array(xnodes(model.grid, Center()))
    interior(model.tracers.c) .= reshape(Float64.(3e5 .< x .< 6e5), :, 1, 1) .* ρ
    Oceananigans.TimeSteppers.update_state!(model)
    return nothing
end

function integrate_boxes(model, Δt, steps)
    set_boxes!(model)
    fields = (model.moisture_density, model.microphysical_fields.ρqʳ, model.tracers.c)
    surface_rain = model.dynamics.moisture.ledger.surface_rain
    water₀, dry₀, tracer₀ = sum(column_water(model)), sum(column_dry_mass(model)), sum(Array(interior(model.tracers.c)))
    largest = map(f -> maximum(Array(interior(f))), fields)
    lowest = map(f -> minimum(Array(interior(f))), fields)

    for _ in 1:steps
        time_step!(model, Δt)
        largest = map((m, f) -> max(m, maximum(Array(interior(f)))), largest, fields)
        lowest = map((m, f) -> min(m, minimum(Array(interior(f)))), lowest, fields)
    end

    rain = sum(Array(interior(surface_rain)))
    mixing_ratio = Array(interior(model.tracers.c)) ./ Array(interior(model.dynamics.density))

    return (; lowest, largest, rain,
              water = (sum(column_water(model)) + rain - water₀) / water₀,
              dry = sum(column_dry_mass(model)) / dry₀ - 1,
              tracer = sum(Array(interior(model.tracers.c))) / tracer₀ - 1,
              mixing_ratio = extrema(mixing_ratio))
end

@testset "Tableaus with convex scalar stages" begin
    for name in (:ConvexLinearMIS3, :ConvexLinearMIS4, :ConvexLinearMIS5, :ConvexLinearMIS6)
        @test convex_scalar_stages(multirate_infinitesimal_step_tableau(name))
    end

    for name in (:MIS3C, :MIS4a, :WickerSkamarock3)
        @test !convex_scalar_stages(multirate_infinitesimal_step_tableau(name))
    end

    # The longest forward Euler step is the time step over the strong-stability coefficient
    tableau = multirate_infinitesimal_step_tableau(:ConvexLinearMIS6)
    @test largest_euler_step_fraction(tableau) ≈ 1 / tableau.strong_stability_coefficient rtol=1e-6
    @test 0.5 < largest_euler_step_fraction(tableau) < 0.6

    upwind() = UpwindBiased(order=1)
    @test convex_slice(upwind).timestepper.scalar_stages isa ConvexScalarStages
    @test convex_slice(upwind; tableau=:MIS3C).timestepper.scalar_stages isa RecombinedScalarStages
    @test convex_slice(upwind; microphysics=nothing).timestepper.scalar_stages isa RecombinedScalarStages
end

water_schemes = (("first-order upwind", () -> UpwindBiased(order=1)),
                 ("WENO5 with flux limiter", () -> WENO(order=5, bounds=BoundsPreservation(0, 1; flux_limiter=true))),
                 ("WENO5 with flux limiter, no rescaling", () -> WENO(order=5, bounds=BoundsPreservation(0, 1; flux_limiter=true, rescaling=false))))

@testset "Bounds and budgets of convex scalar stages [$name]" for (name, scheme) in water_schemes
    result = integrate_boxes(convex_slice(scheme), 1200, 36)
    @info "Convex stages, $name, 36 steps of 1200 s: minima (ρqᵉ, ρqʳ, ρc) $(result.lowest) of maxima $(result.largest); " *
          "tracer mixing ratio in $(result.mixing_ratio); residuals: water $(result.water), dry mass $(result.dry), " *
          "tracer $(result.tracer); rain $(result.rain)"

    @test result.rain > 0
    @test all(result.lowest .≥ .- rounding_bound .* result.largest)
    @test result.mixing_ratio[1] ≥ - rounding_bound
    @test result.mixing_ratio[2] ≤ 1 + rounding_bound
    @test abs(result.water) ≤ budget_bound
    @test abs(result.dry) ≤ budget_bound
    @test abs(result.tracer) ≤ budget_bound
end

@testset "The recombined stages of MIS3C are not positive" begin
    result = integrate_boxes(convex_slice(() -> UpwindBiased(order=1); tableau=:MIS3C), 600, 24)
    @info "Recombined stages of MIS3C, first-order upwind, 24 steps of 600 s: minima $(result.lowest) of maxima $(result.largest)"
    @test any(result.lowest .< .- 1e6 .* rounding_bound .* result.largest) # the test above discriminates
    @test abs(result.water) ≤ budget_bound
end

@testset "Constancy of uniform mixing ratios with convex scalar stages [$name]" for (name, scheme) in water_schemes
    model = convex_slice(scheme)
    column_physics_in_time_step!(model, false) # nothing rains: the water changes by transport alone
    set!(model; pˢ=1e5, θ = (x, z) -> 300 + 10z, u=20)
    r₀, χ₀ = 3e-3, 2.0
    ρ = Array(interior(model.dynamics.density))
    set!(model; ρqᵉ = r₀ .* ρ)
    ρ = Array(interior(model.dynamics.density))
    interior(model.tracers.c) .= χ₀ .* ρ
    set!(model; ρqᵉ = r₀ .* ρ)

    for _ in 1:72
        time_step!(model, 600)
    end

    ρ = Array(interior(model.dynamics.density))
    water = Array(interior(model.moisture_density)) .+ Array(interior(model.microphysical_fields.ρqʳ))
    water_deviation = maximum(abs, water ./ ρ ./ r₀ .- 1)
    tracer_deviation = maximum(abs, Array(interior(model.tracers.c)) ./ ρ ./ χ₀ .- 1)
    @info "Constancy with convex stages, $name, 72 steps of 600 s: water $water_deviation, tracer $tracer_deviation; " *
          "max |u| = $(max_abs(model.velocities.u))"

    @test max_abs(model.velocities.u) > 5
    @test water_deviation ≤ budget_bound
    @test tracer_deviation ≤ budget_bound
end
