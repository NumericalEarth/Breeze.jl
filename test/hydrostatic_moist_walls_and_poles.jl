using Breeze
using Oceananigans
using Test

include(joinpath(@__DIR__, "hydrostatic_moist_utils.jl"))

using Oceananigans.BoundaryConditions: PolarValue, ValueBoundaryCondition, FluxBoundaryCondition
using Oceananigans.Operators: Vᶜᶜᶜ, Azᶜᶜᶜ

using Breeze.AtmosphereModels: specific_prognostic_moisture
using Breeze.HydrostaticEquations: MixingRatio, is_dry_boundary_condition

#####
##### Moist `HydrostaticDynamics` on a `LatitudeLongitudeGrid` with walls, and with poles.
#####
##### Float32: a moist model cannot be built on a Float64 `LatitudeLongitudeGrid` yet (the broken
##### test "Moist HydrostaticDynamics on a latitude-longitude grid" of hydrostatic_moist_turbulence.jl).
#####
##### Tolerances, written before the first run of this file. The water of a domain closed by walls
##### changes only by rounding: 1e-5 of the total over a few steps in Float32, where a flux through
##### a wall would be of order 1e-3. A water density may be negative by rounding only, within the
##### bound declared for the bounds-preserving transport of water, 64 eps(FT) times its maximum.
#####

const closed_water_relative = 1e-5
const rounding_factor = 64

# The saturation adjustment with a tolerance that single precision can reach
single_precision_microphysics(FT) =
    OneMomentCloudMicrophysics(FT; cloud_formation = SaturationAdjustment(FT; equilibrium = WarmPhaseEquilibrium(),
                                                                          solver = SecantSolver(FT; abstol=1e-4, maxiter=20)))

function latitude_longitude_moist_model(latitude; FT=Float32, size=(24, 12, 8))
    grid = LatitudeLongitudeGrid(CPU(), FT; size, longitude=(0, 360), latitude, z=(0, 1), halo=(4, 4, 3))
    bounded() = WENO(FT; order=5, bounds=(0, 1))
    return AtmosphereModel(grid; dynamics = HydrostaticDynamics(top_pressure=2.5e4),
                           microphysics = single_precision_microphysics(FT),
                           formulation = moist_formulation(),
                           momentum_advection = WENO(FT; order=5),
                           scalar_advection = (ρθ = WENO(FT; order=5), ρqᵉ = bounded(), ρqʳ = bounded()))
end

prognostic_values(model) = reduce(vcat, [vec(Array(interior(field))) for field in values(Oceananigans.prognostic_fields(model))])

# Water of the grid, in kg per unit of the grid's vertical coordinate times m², with what the rain
# took through the ground
function grid_water(model)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    ρqᵉ = model.moisture_density
    ρqʳ = model.microphysical_fields.ρqʳ
    rain = model.dynamics.moisture.ledger.surface_rain
    air = sum(Float64(ρqᵉ[i, j, k] + ρqʳ[i, j, k]) * Vᶜᶜᶜ(i, j, k, grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz)
    ground = sum(Float64(rain[i, j, 1]) * Azᶜᶜᶜ(i, j, 1, grid) for i in 1:Nx, j in 1:Ny)
    return air + ground
end

@testset "Mixing ratio of the halo cells beyond a wall" begin
    model = latitude_longitude_moist_model((-60, 60))
    set!(model; pˢ = 1e5, θ = (λ, φ, z) -> 290 + 40z, qᵗ = (λ, φ, z) -> 0.015 * (1 - z)^3)
    Nx, Ny, Nz = size(model.grid)
    ρ = model.dynamics.density
    r = MixingRatio(specific_prognostic_moisture(model), ρ)

    # Of a boundary without flux only the first halo cell is filled: the others have no mass
    @test ρ[1, 0, 1] == ρ[1, 1, 1] > 0
    @test ρ[1, -1, 1] == 0
    @test ρ[1, Ny+2, 1] == 0

    # The mixing ratio is a number everywhere a stencil reads it, zero where there is no mass,
    # and unchanged where there is
    @test all(isfinite(r[i, j, k]) for i in 1:Nx, j in -2:Ny+3, k in 1:Nz)
    @test all(r[i, j, k] == 0 for i in 1:Nx, j in (-2, -1, Ny+2, Ny+3), k in 1:Nz)
    q = specific_prognostic_moisture(model)
    @test all(r[i, j, k] == q[i, j, k] for i in 1:Nx, j in 0:Ny+1, k in 1:Nz)
    @test r[1, 1, 1] > 0
end

@testset "Moist flow between walls of latitude: finite, positive, closed" begin
    model = latitude_longitude_moist_model((-60, 60))
    FT = eltype(model.grid)
    set!(model; pˢ = 1e5,
                θ = (λ, φ, z) -> 290 + 40z + 2 * cosd(φ) * sind(2λ),
                qᵗ = (λ, φ, z) -> 0.02 * (1 - z)^3 * (1 + 0.3 * sind(λ)) * cosd(φ),
                u = (λ, φ, z) -> 10 * z^0.3 * cosd(φ) * (1 + 0.2 * sind(2λ)),
                v = (λ, φ, z) -> 3 * z * cosd(1.5φ) * sind(λ))

    Ny = size(model.grid, 2)
    @test all(iszero, Array(interior(model.velocities.v, :, 1, :)))
    @test all(iszero, Array(interior(model.velocities.v, :, Ny+1, :)))

    W₀ = grid_water(model)
    ρqᵉ = model.moisture_density
    ρqʳ = model.microphysical_fields.ρqʳ
    largest = (ρqᵉ = maximum(ρqᵉ), ρqʳ = maximum(ρqʳ))

    for n in 1:8
        time_step!(model, 200)
        largest = (ρqᵉ = max(largest.ρqᵉ, maximum(ρqᵉ)), ρqʳ = max(largest.ρqʳ, maximum(ρqʳ)))
    end

    state = prognostic_values(model)
    @test all(isfinite, state)

    # The rows along the walls, which a mixing ratio that is not a number in the halos reaches first
    for field in (ρqᵉ, ρqʳ, model.formulation.potential_temperature_density, model.velocities.u)
        @test all(isfinite, Array(interior(field, :, 1:3, :)))
        @test all(isfinite, Array(interior(field, :, Ny-2:Ny, :)))
    end

    @test minimum(ρqᵉ) ≥ -rounding_factor * eps(FT) * largest.ρqᵉ
    @test minimum(ρqʳ) ≥ -rounding_factor * eps(FT) * largest.ρqʳ
    @test all(iszero, Array(interior(model.velocities.v, :, 1, :)))

    change = (grid_water(model) - W₀) / W₀
    @info "moist flow between walls" change minimum_ρqᵉ = minimum(ρqᵉ) minimum_ρqʳ = minimum(ρqʳ) rain = maximum(model.dynamics.moisture.ledger.surface_rain)
    @test abs(change) ≤ closed_water_relative
end

@testset "The condition of a pole is not a source" begin
    @test is_dry_boundary_condition(nothing)
    @test !is_dry_boundary_condition(ValueBoundaryCondition(1))
    @test !is_dry_boundary_condition(FluxBoundaryCondition(1))
    @test !is_dry_boundary_condition(FluxBoundaryCondition((x, y, t) -> 0))

    # Dry air on the whole sphere
    grid = LatitudeLongitudeGrid(CPU(), Float64; size=(24, 12, 6), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
    dry = AtmosphereModel(grid; dynamics = HydrostaticDynamics(top_pressure=2.5e4), momentum_advection = VectorInvariant())
    south = dry.moisture_density.boundary_conditions.south
    @test south.condition isa PolarValue
    @test is_dry_boundary_condition(south)
    @test is_dry_boundary_condition(dry.moisture_density.boundary_conditions.north)

    # Moist air on the whole sphere: built, set, and one short step
    model = latitude_longitude_moist_model((-90, 90))
    @test model isa AtmosphereModel
    ρθ = model.formulation.potential_temperature_density
    @test ρθ.boundary_conditions.south.condition isa PolarValue
    @test ρθ.boundary_conditions.north.condition isa PolarValue

    set!(model; pˢ = 1e5, θ = (λ, φ, z) -> 290 + 40z, qᵗ = (λ, φ, z) -> 0.01 * (1 - z)^3 * cosd(φ)^2)
    time_step!(model, 1)
    @test all(isfinite, prognostic_values(model))
    @test minimum(model.moisture_density) ≥ 0
end
