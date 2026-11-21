include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.HydrostaticEquations: VerletSubsteps, column_physics_in_time_step!
using Breeze.TimeSteppers: ConvexScalarStages
using Breeze.PressureLevelDiscretization: PressureLevelVerticalDiscretization, layer_mass
using Oceananigans
using Oceananigans.Advection: BoundsPreservation
using Test

include("hydrostatic_moist_utils.jl")

#####
##### Moist HydrostaticDynamics on a pressure-level grid, with the convex scalar stages of the
##### split-explicit stepper and WENO with a flux limiter
#####
#
# The slice of `hydrostatic_moist_convex_stages.jl` on a `PressureLevelGrid`. The water and the
# tracer are masses per unit area M = (Δp / g) ρc / ρ there, so the budgets are sums of M. The
# bounds are those of that file: 64 eps × the largest value for positivity and 1e-12 for the budgets.

const rounding_bound = 64 * eps(Float64)
const budget_bound = 1e-12

pressure_level_ridge(x) = 9.80665 * 1500 * exp(-((x - 8e5) / 2e5)^2)

function pressure_level_coefficients(Nz; top_pressure = 2.5e4, p₀ = 1e5)
    η = range(1, top_pressure / p₀, length = Nz + 1)
    B = [max(0, (e - 0.4) / 0.6) for e in η]
    A = [e * p₀ - b * p₀ for (e, b) in zip(η, B)]
    A[1] = 0; B[1] = 1; B[end] = 0; A[end] = top_pressure
    return A, B
end

function pressure_level_slice(water_scheme)
    A, B = pressure_level_coefficients(12)
    grid = RectilinearGrid(CPU(), Float64; size=(32, 12), x=(0, 1.6e6), halo=(5, 5),
                           z = PressureLevelVerticalDiscretization(A, B), topology=(Periodic, Flat, Bounded))
    dynamics = HydrostaticDynamics(; surface_geopotential=pressure_level_ridge)
    scalar_advection = (ρθ = WENO(order=5), ρqᵉ = water_scheme(), ρqʳ = water_scheme(), c = water_scheme())
    return AtmosphereModel(grid; dynamics, microphysics=warm_rain_microphysics(), tracers=(:c,), scalar_advection,
                           formulation = moist_formulation(), momentum_advection = VectorInvariant(),
                           timestepper = :GravityWaveMultirateInfinitesimalStep,
                           timestepper_kwargs = (; tableau=:ConvexLinearMIS6, inner_solver=VerletSubsteps(),
                                                   moist_gravity_wave_operator=true))
end

# Σ (Δp / g) ρc / ρ over the domain, per unit area of a cell
function total_mass(model, ρc)
    grid = model.grid
    ρ = model.dynamics.density
    Nx, Ny, Nz = size(grid)
    return sum(layer_mass(i, j, k, grid) * ρc[i, j, k] / ρ[i, j, k] for i in 1:Nx, j in 1:Ny, k in 1:Nz)
end

# Σ Δp / g over the domain, per unit area of a cell: the mass of the air and of its water
function column_mass(model)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    return sum(layer_mass(i, j, k, grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz)
end

flux_limited_weno() = WENO(order=5, bounds=BoundsPreservation(0, 1; flux_limiter=true))

@testset "Moist convex stages on a PressureLevelGrid: bounds and budgets" begin
    model = pressure_level_slice(flux_limited_weno)
    @test model.timestepper.scalar_stages isa ConvexScalarStages

    box(x, lower, upper) = lower < x < upper
    rain(x, z) = box(x, 5e5, 7e5) * box(z, 2000, 5000) * 2e-3
    set!(model; pˢ=1e5, θ = (x, z) -> 288 + 4.5e-3z + 2 * exp(-((x - 4e5) / 2e5)^2), u=20)
    set!(model; qᵗ = (x, z) -> box(x, 2e5, 9e5) * 0.016 * exp(-z / 2000) + rain(x, z), qʳ = rain)
    x = xnodes(model.grid, Center())
    for k in 1:12, i in 1:32
        model.tracers.c[i, 1, k] = (3e5 < x[i] < 6e5) * model.dynamics.density[i, 1, k]
    end
    Oceananigans.TimeSteppers.update_state!(model)

    fields = (model.moisture_density, model.microphysical_fields.ρqʳ, model.tracers.c)
    water(model) = total_mass(model, model.moisture_density) + total_mass(model, model.microphysical_fields.ρqʳ)
    surface_rain = model.dynamics.moisture.ledger.surface_rain
    water₀, tracer₀ = water(model), total_mass(model, model.tracers.c)
    mass₀ = column_mass(model)
    dry₀ = mass₀ - water₀
    largest = map(f -> maximum(interior(f)), fields)
    lowest = map(f -> minimum(interior(f)), fields)

    for _ in 1:24
        time_step!(model, 1200)
        largest = map((m, f) -> max(m, maximum(interior(f))), largest, fields)
        lowest = map((m, f) -> min(m, minimum(interior(f))), lowest, fields)
    end

    rained = sum(interior(surface_rain))
    water_residual = (water(model) + rained - water₀) / water₀
    tracer_residual = total_mass(model, model.tracers.c) / tracer₀ - 1
    # The coordinate is the total mass: the rain through the ground lowers it, and the dry air is what is left
    mass_residual = (column_mass(model) + rained - mass₀) / mass₀
    dry_residual = (column_mass(model) - water(model)) / dry₀ - 1
    r = interior(model.tracers.c) ./ interior(model.dynamics.density)
    @info "Moist convex stages on a PressureLevelGrid, 24 steps of 1200 s: minima $lowest of maxima $largest; " *
          "water $water_residual, mass $mass_residual, dry air $dry_residual, tracer $tracer_residual, tracer mixing ratio $(extrema(r)), rain $rained"

    @test all(isfinite, interior(model.dynamics.surface_pressure))
    @test all(lowest .≥ .- rounding_bound .* largest)
    @test minimum(r) ≥ - rounding_bound
    @test maximum(r) ≤ 1 + rounding_bound
    @test abs(water_residual) ≤ budget_bound
    @test abs(tracer_residual) ≤ budget_bound
    @test abs(mass_residual) ≤ budget_bound
    @test abs(dry_residual) ≤ budget_bound
    @test rained > 0
end
