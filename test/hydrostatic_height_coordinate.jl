using Breeze
using Oceananigans
using Test

include("hydrostatic_moist_utils.jl")

using Oceananigans.BoundaryConditions: getbc
using Oceananigans.Grids: znode, rnode
using Oceananigans.MultiRegion: YinYangGrid
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, Δrᶜᶜᶜ, Vᶜᶜᶜ, Azᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: height_above_bottomᶜᶜᶜ, height_above_bottomᶜᶜᶠ
using Oceananigans.Utils: getregion

using Breeze.HydrostaticEquations: HydrostaticEquations, HydrostaticPressureGrid, physical_height_grid,
                                   physical_layer_depth, height_between_layers, warm_rain_column_physics!,
                                   WarmRainColumnPhysics, column_physics_in_time_step!
using Breeze.HydrostaticMoistTurbulence: MoistTurbulentMixing, moist_turbulent_mixing_step!, BulkSurfaceExchange
using Breeze.TimeSteppers: VerletSubsteps

#####
##### The physical height grid of `HydrostaticDynamics(physical_height_grid=true)`.
#####
##### Float64, CPU. Tolerances were declared in derivations/vertical_coordinate/DESIGN.md (hashed
##### before this file and the source existed): the comparisons with `==` are bitwise by design,
##### and a failure of one is a failure, not a tolerance to be widened.
#####

const closed_form_relative = 1e-13   # heights of an isentropic column against cᵖᵈ θ ΔΠ / g
const sum_relative = 1e-14           # Σ Δz against the height of the model top
const reference_grid_relative = 1e-12 # a bulk coefficient on the height grid against its closed form at the physical height

const columns_directory = joinpath(@__DIR__, "hydrostatic_moist_columns")
column_state(name) = read_column(joinpath(columns_directory, name * ".csv"))

column_values(field) = vec(Array(interior(field)))
parents_equal(a, b) = isequal(Array(parent(a)), Array(parent(b)))

regional_models(model) = model.grid isa YinYangGrid ? [getregion(model, r) for r in 1:2] : [model]

# The interior of every prognostic field (and of the given others), region by region
function model_state(model, others...)
    state = Float64[]
    for regional in regional_models(model)
        for field in values(Oceananigans.prognostic_fields(regional))
            append!(state, vec(Array(interior(field))))
        end
    end
    for other in others, r in 1:length(regional_models(model))
        field = model.grid isa YinYangGrid ? getregion(other, r) : other
        append!(state, vec(Array(interior(field))))
    end
    return state
end

# The heights the kernel must have stored, from the fields of a single-region model
function fresh_heights(model)
    grid = model.grid
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    cᵖᵈ = constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration
    θᵨ = isnothing(dynamics.moisture) ? model.formulation.potential_temperature : dynamics.moisture.density_potential_temperature
    Π = dynamics.exner_function
    Πᶠ = dynamics.interface_exner_function
    Nx, Ny, Nz = size(grid)

    Δzᶜ = zeros(Nx, Ny, Nz)
    Δzᶠ = zeros(Nx, Ny, Nz+1)
    zᶠ = zeros(Nx, Ny, Nz+1)

    for i in 1:Nx, j in 1:Ny
        for k in 1:Nz
            Δzᶜ[i, j, k] = physical_layer_depth(i, j, k, θᵨ, Πᶠ, cᵖᵈ, g)
            zᶠ[i, j, k+1] = zᶠ[i, j, k] + Δzᶜ[i, j, k]
        end
        Δzᶠ[i, j, 1] = Δzᶜ[i, j, 1] / 2
        for k in 2:Nz
            Δzᶠ[i, j, k] = height_between_layers(i, j, k, grid, θᵨ, Π, Πᶠ, cᵖᵈ, g)
        end
        Δzᶠ[i, j, Nz+1] = Δzᶜ[i, j, Nz] / 2
    end

    return (; Δzᶜ, Δzᶠ, zᶠ)
end

function stored_heights(model)
    coordinate = model.dynamics.height_coordinate
    return (Δzᶜ = Array(interior(coordinate.layer_depth)),
            Δzᶠ = Array(interior(coordinate.interface_spacing)),
            zᶠ = Array(interior(coordinate.interface_height)))
end

heights_are_current(model) = all(stored_heights(regional) == fresh_heights(regional) for regional in regional_models(model))

function moist_height_column(name; physical_height_grid=true, Φˢ=0, tracers=(), FT=Float64)
    column, header = column_state(name)
    Nz = length(column.k)
    grid = RectilinearGrid(CPU(), FT; size=Nz, x=0, y=0, z=(0, 1), topology=(Flat, Flat, Bounded))
    A = vcat(column.A_lower, column.A_upper[end])
    B = vcat(column.B_lower, column.B_upper[end])
    dynamics = HydrostaticDynamics(; pressure_coefficient=A, surface_pressure_coefficient=B, surface_geopotential=Φˢ,
                                   physical_height_grid)
    model = AtmosphereModel(grid; dynamics, microphysics=warm_rain_microphysics(FT), tracers,
                            formulation = moist_formulation(), momentum_advection = VectorInvariant())
    set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, column.r_e, column.r_r)
    return model
end

function turbulent_height_column(name; e=0.1, kw...)
    model = moist_height_column(name; tracers=:ρe, kw...)
    Nz = size(model.grid, 3)
    ρᵗ = Array(interior(model.dynamics.moisture.total_density))
    interior(model.velocities.u) .= reshape([10 * (k / Nz)^0.3 for k in 1:Nz], 1, 1, Nz)
    interior(model.velocities.v) .= reshape([2 * sin(k) for k in 1:Nz], 1, 1, Nz)
    interior(model.tracers.ρe) .= ρᵗ .* e
    update_state!(model; compute_tendencies=false)
    return model
end

function height_slice(; physical_height_grid=true, microphysics=nothing, size=(16, 10), Φˢ=0, kw...)
    grid = RectilinearGrid(CPU(), Float64; size, x=(0, 1.6e6), z=(0, 1), topology=(Periodic, Flat, Bounded))
    dynamics = HydrostaticDynamics(; surface_geopotential=Φˢ, top_pressure=2.5e4, physical_height_grid)
    moist = isnothing(microphysics) ? (;) : (; microphysics, formulation = moist_formulation())
    return AtmosphereModel(grid; dynamics, momentum_advection = VectorInvariant(), moist..., kw...)
end

bump(x) = exp(-((x - 8e5) / 2e5)^2)

set_dry_slice!(model) = set!(model; pˢ = x -> 1e5 + 300 * bump(x), θ = (x, z) -> 290 + 40z + 2 * bump(x), u = (x, z) -> 5 + 10z)

set_moist_slice!(model) = set!(model; pˢ = x -> 1e5 + 300 * bump(x), θ = (x, z) -> 290 + 40z + 2 * bump(x),
                               qᵗ = (x, z) -> 0.02 * (1 - z)^3 * (1 + bump(x)), u = (x, z) -> 5 + 10z)

#####
##### T1: a static column
#####

@testset "Heights of a static isentropic column [Φˢ = $Φˢ]" for Φˢ in (0, 4905.0)
    Nz = 12
    θ₀ = 300
    grid = RectilinearGrid(CPU(), Float64; size=Nz, x=0, y=0, z=(0, 1), topology=(Flat, Flat, Bounded))
    dynamics = HydrostaticDynamics(top_pressure=2.5e4, surface_geopotential=Φˢ, physical_height_grid=true)
    model = AtmosphereModel(grid; dynamics, momentum_advection=VectorInvariant())
    set!(model; pˢ=9.8e4, θ=θ₀)

    heights = physical_height_grid(model)
    @test heights isa HydrostaticPressureGrid
    @test heights !== model.grid
    @test size(heights) == size(grid)

    constants = model.thermodynamic_constants
    cᵖᵈ = constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration
    θ = model.formulation.potential_temperature
    Π = model.dynamics.exner_function
    Πᶠ = model.dynamics.interface_exner_function
    Φ = model.dynamics.geopotential
    c, f = Center(), Face()

    # The depth of the layers is the hydrostatic integral, bitwise
    @test all(Δzᶜᶜᶜ(1, 1, k, heights) == physical_layer_depth(1, 1, k, θ, Πᶠ, cᵖᵈ, g) for k in 1:Nz)
    @test all(Δzᶜᶜᶠ(1, 1, k, heights) == height_between_layers(1, 1, k, grid, θ, Π, Πᶠ, cᵖᵈ, g) for k in 2:Nz)
    @test all(Δzᶜᶜᶜ(1, 1, k, heights) > 0 for k in 1:Nz)

    # Heights of the faces against the closed form of an isentropic column
    z_closed(k) = cᵖᵈ * θ₀ * (Πᶠ[1, 1, 1] - Πᶠ[1, 1, k]) / g
    face_error = maximum(abs(znode(1, 1, k, heights, c, c, f) - Φˢ / g - z_closed(k)) for k in 1:Nz+1) / z_closed(Nz+1)
    @info "isentropic column" Φˢ z_top = z_closed(Nz+1) face_error
    @test face_error ≤ closed_form_relative
    @test z_closed(Nz+1) > 9e3

    # Heights of the centers are the geopotential of the layers
    @test all(znode(1, 1, k, heights, c, c, c) == Φ[1, 1, k] / g for k in 1:Nz)
    @test all(znode(1, 1, k, heights, c, c, f) < znode(1, 1, k, heights, c, c, c) < znode(1, 1, k+1, heights, c, c, f) for k in 1:Nz)

    total = sum(Δzᶜᶜᶜ(1, 1, k, heights) for k in 1:Nz)
    zᶠ = model.dynamics.height_coordinate.interface_height
    @test abs(total - zᶠ[1, 1, Nz+1]) ≤ sum_relative * total

    # Heights above the ground, as the closures read them
    @test all(height_above_bottomᶜᶜᶠ(1, 1, k, heights) == max(Δzᶜᶜᶜ(1, 1, k-1, heights), zᶠ[1, 1, k]) for k in 2:Nz+1)
    @test all(height_above_bottomᶜᶜᶜ(1, 1, k, heights) ==
              max(Δzᶜᶜᶜ(1, 1, k, heights) / 2, (zᶠ[1, 1, k] + zᶠ[1, 1, k+1]) / 2) for k in 1:Nz)

    # The reference coordinate and the grid of the model are untouched
    @test all(Δrᶜᶜᶜ(1, 1, k, heights) == Δzᶜᶜᶜ(1, 1, k, grid) == 1 / Nz for k in 1:Nz)
    @test all(rnode(1, 1, k, heights, c, c, f) == znode(1, 1, k, grid, c, c, f) for k in 1:Nz+1)
    @test all(Vᶜᶜᶜ(1, 1, k, heights) == Azᶜᶜᶜ(1, 1, k, heights) * Δzᶜᶜᶜ(1, 1, k, heights) for k in 1:Nz)

    # The mass of a layer is the same in both metrics: ρ Δr = (ρ Δr / Δz) Δz = Δp / g
    ρ = model.dynamics.density
    @test all(ρ[1, 1, k] * Δzᶜᶜᶜ(1, 1, k, grid) ≈ (ρ[1, 1, k] * Δrᶜᶜᶜ(1, 1, k, heights) / Δzᶜᶜᶜ(1, 1, k, heights)) * Δzᶜᶜᶜ(1, 1, k, heights)
              for k in 1:Nz)
end

@testset "Heights above the ground do not depend on the height of the ground" begin
    flat = moist_height_column("moist_tropical")
    raised = moist_height_column("moist_tropical"; Φˢ=9810.0)
    @test stored_heights(flat) == stored_heights(raised)
    g = flat.thermodynamic_constants.gravitational_acceleration
    lifted = znode(1, 1, 1, physical_height_grid(raised), Center(), Center(), Face())
    @test lifted == 9810.0 / g
    @test znode(1, 1, 1, physical_height_grid(flat), Center(), Center(), Face()) == 0
end

@testset "A model without the option has no height grid" begin
    model = moist_height_column("moist_tropical"; physical_height_grid=false)
    @test isnothing(model.dynamics.height_coordinate)
    @test_throws ArgumentError physical_height_grid(model)
end

@testset "Heights of a moist column with hybrid coefficients [$name]" for name in ("moist_tropical", "midlatitude_frontal")
    model = moist_height_column(name)
    @test heights_are_current(model)
    heights = physical_height_grid(model)
    constants = model.thermodynamic_constants
    θᵨ = model.dynamics.moisture.density_potential_temperature
    Πᶠ = model.dynamics.interface_exner_function
    Nz = size(model.grid, 3)
    @test all(Δzᶜᶜᶜ(1, 1, k, heights) == physical_layer_depth(1, 1, k, θᵨ, Πᶠ, constants.dry_air.heat_capacity,
                                                             constants.gravitational_acceleration) for k in 1:Nz)

    # The density of the air in kg m⁻³ from the two metrics against the equation of state at the
    # layer pressure: equal up to the difference between the layer mean of the Exner function and
    # the Exner function of the mean pressure (reported, bounded loosely)
    ρᵗ = model.dynamics.moisture.total_density
    p = model.dynamics.pressure
    T = model.temperature
    Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
    Tᵥ(k) = θᵨ[1, 1, k] * model.dynamics.exner_function[1, 1, k]
    mismatch = maximum(abs(ρᵗ[1, 1, k] * Δrᶜᶜᶜ(1, 1, k, heights) / Δzᶜᶜᶜ(1, 1, k, heights) / (p[1, 1, k] / (Rᵈ * Tᵥ(k))) - 1) for k in 1:Nz)
    @info "density from the height metric against the equation of state" name mismatch
    @test mismatch < 0.02
end

#####
##### T2: the heights follow the state at the declared update points
#####

@testset "Heights follow the state [$(isempty(timestepper) ? :SSPRungeKutta3 : timestepper.timestepper)]" for
        timestepper in ((;), (timestepper = :GravityWaveMultirateInfinitesimalStep,
                              timestepper_kwargs = (tableau = :MIS3C, inner_solver = VerletSubsteps())))

    model = height_slice(; timestepper...)
    set_dry_slice!(model)
    @test heights_are_current(model)
    initial = stored_heights(model)

    # A new surface pressure and potential temperature: `set!` ends with `update_state!`
    set!(model; pˢ = x -> 9.9e4 - 200 * bump(x), θ = (x, z) -> 285 + 45z)
    @test heights_are_current(model)
    @test stored_heights(model).Δzᶜ != initial.Δzᶜ

    set_dry_slice!(model)
    @test stored_heights(model) == initial

    for n in 1:4
        time_step!(model, 100)
        @test heights_are_current(model)
    end
    stepped = stored_heights(model)
    @test stepped.Δzᶜ != initial.Δzᶜ
    @test all(stepped.Δzᶜ .> 0)

    # Staleness: the heights belong to the state of the last `update_state!`
    interior(model.formulation.potential_temperature_density) .*= 1.01
    @test stored_heights(model) == stepped
    update_state!(model; compute_tendencies=false)
    @test heights_are_current(model)
    @test stored_heights(model).Δzᶜ != stepped.Δzᶜ
end

#####
##### T3: the option changes nothing in the dynamics
#####

@testset "The dynamics does not read the height grid [$(isempty(timestepper) ? :SSPRungeKutta3 : timestepper.timestepper)]" for
        timestepper in ((;), (timestepper = :GravityWaveMultirateInfinitesimalStep,
                              timestepper_kwargs = (tableau = :MIS3C, inner_solver = VerletSubsteps())))

    states = map((true, false)) do option
        model = height_slice(; physical_height_grid=option, timestepper...)
        set_dry_slice!(model)
        for n in 1:6
            time_step!(model, 100)
        end
        model_state(model)
    end

    @test states[1] == states[2]
    @test all(isfinite, states[1])
end

function moist_slice_run(option; steps=6, Δt=200, substeps=4)
    model = height_slice(; physical_height_grid=option, microphysics=warm_rain_microphysics())
    set_moist_slice!(model)
    column_physics_in_time_step!(model, false)
    simulation = Simulation(model; Δt, stop_iteration=steps, verbose=false)
    add_callback!(simulation, Callback(WarmRainColumnPhysics(; substeps), IterationInterval(1)); name=:physics)
    run!(simulation)
    return model
end

@testset "Moist dynamics with warm rain four times per step: the option changes nothing" begin
    with = moist_slice_run(true)
    without = moist_slice_run(false)
    ledger(model) = reduce(vcat, [vec(Array(interior(field))) for field in values(model.dynamics.moisture.ledger)])

    @test model_state(with) == model_state(without)
    @test ledger(with) == ledger(without)
    @test heights_are_current(with)
    @test maximum(with.dynamics.moisture.ledger.surface_rain) ≥ 0
    @info "moist slice" surface_rain = maximum(with.dynamics.moisture.ledger.surface_rain)
end

#####
##### T6: the sedimentation reads the depth of the layers from the grid
#####

@testset "Sedimentation with the depth of the layers from the height grid [$name]" for
        name in ("moist_tropical_heavy_rain", "midlatitude_frontal")

    results = map((true, false)) do option
        model = moist_height_column(name; physical_height_grid=option)
        for n in 1:3
            warm_rain_column_physics!(model.microphysics, model; Δt=600, substeps=4)
        end
        ledger = [getproperty(model.dynamics.moisture.ledger, field)[1, 1, 1] for field in propertynames(model.dynamics.moisture.ledger)]
        (state = model_state(model), ledger = ledger, current = option ? heights_are_current(model) : true)
    end

    @test results[1].state == results[2].state
    @test results[1].ledger == results[2].ledger
    @test results[1].current
    @info "sedimentation" name surface_rain = results[1].ledger[1] courant = results[1].ledger[4]
    name == "moist_tropical_heavy_rain" && @test results[1].ledger[1] > 0
end

#####
##### T5: the closure and the bulk formulas on the height grid, against the wrapper
#####

function bulk_conditions(Tˢ)
    return (drag = Breeze.BulkDrag(coefficient=1.2e-3, surface_temperature=Tˢ),
            sensible_heat = Breeze.BulkSensibleHeatFlux(coefficient=1.1e-3, surface_temperature=Tˢ),
            vapor = Breeze.BulkVaporFlux(coefficient=1.1e-3, surface_temperature=Tˢ))
end

function mixing_run(name, option; steps=6, Δt=600)
    model = turbulent_height_column(name; physical_height_grid=option)
    closure = TKEBasedTurbulenceClosure(Float64)
    exchange = BulkSurfaceExchange(model, closure; bulk_conditions(301)...)
    mixing = MoistTurbulentMixing(model; closure, surface_exchange=exchange)
    for n in 1:steps
        moist_turbulent_mixing_step!(mixing, model, Δt)
    end
    return model, mixing
end

field_values(fields) = reduce(vcat, [vec(Array(interior(field))) for field in values(fields)])

# The fields of the closure: diffusivities, static stability, mixing lengths and the sink rate
closure_fields(mixing) = map(name -> getproperty(mixing.closure_fields, name), (:Kᵘ, :Kᶜ, :Kᵉ, :N², :ℓᵐ, :ℓˢ, :Lᵉ))

@testset "TKEBasedTurbulenceClosure and bulk fluxes on the height grid against the wrapper [$name]" for
        name in ("moist_tropical", "moist_tropical_clear", "midlatitude_frontal", "dry_subsidence")

    with, mixing_with = mixing_run(name, true)
    without, mixing_without = mixing_run(name, false)

    @test model_state(with) == model_state(without)
    @test field_values(closure_fields(mixing_with)) == field_values(closure_fields(mixing_without))
    @test field_values(mixing_with.surface_fluxes) == field_values(mixing_without.surface_fluxes)
    @test field_values(mixing_with.ledger) == field_values(mixing_without.ledger)

    # The closure did something
    @test maximum(mixing_with.closure_fields.Kᵘ) > 0
    @test mixing_with.surface_fluxes.x_stress[1, 1, 1] != 0

    difference = maximum(abs, model_state(with) .- model_state(without))
    @info "closure on the height grid against the wrapper" name difference κu_max = maximum(mixing_with.closure_fields.Kᵘ)
end

@testset "A bulk coefficient that reads heights from the grid" begin
    Tˢ = 301
    closure = TKEBasedTurbulenceClosure(Float64)
    neutral = Breeze.BulkDrag(coefficient=Breeze.PolynomialCoefficient(stability_function=nothing), surface_temperature=Tˢ)
    polynomial = Breeze.BulkDrag(coefficient=Breeze.PolynomialCoefficient(), surface_temperature=Tˢ)

    # On the grid of labels it is rejected, as before
    labels = turbulent_height_column("moist_tropical_clear"; physical_height_grid=false)
    @test_throws ArgumentError BulkSurfaceExchange(labels, closure; drag=polynomial)

    # On the height grid it is evaluated as it is, at half the depth of the lowest layer in meters.
    # Without a stability function the coefficient is C₁₀ [ln(10 / ℓ) / ln(h / ℓ)]², so the stress
    # on the two grids differs by the ratio of ln(h / ℓ)² at the two heights and by nothing else.
    model = turbulent_height_column("moist_tropical_clear")
    heights = physical_height_grid(model)
    exchange = BulkSurfaceExchange(model, closure; drag=neutral)
    arguments = (model.clock, Oceananigans.fields(model), Breeze.AtmosphereModels.dynamics_thermodynamic_fields(model.dynamics))
    on_heights = getbc(exchange.x_drag, 1, 1, heights, arguments...)
    on_labels = getbc(exchange.x_drag, 1, 1, model.grid, arguments...)
    ℓ = exchange.x_drag.coefficient.roughness_length
    h = Δzᶜᶜᶜ(1, 1, 1, heights) / 2
    h_label = Δzᶜᶜᶜ(1, 1, 1, model.grid) / 2
    expected = on_labels * (log(h_label / ℓ) / log(h / ℓ))^2

    @info "neutral polynomial bulk drag" on_heights expected on_labels half_layer = h
    @test isfinite(on_heights) && on_heights != 0
    @test abs(on_heights - expected) ≤ reference_grid_relative * abs(expected)
    @test 30 < h < 200
    @test 1e-4 < abs(on_heights) < 1   # a stress in N m⁻²

    # With its stability function, and in the mixing
    stable = BulkSurfaceExchange(model, closure; drag=polynomial)
    stress = getbc(stable.x_drag, 1, 1, heights, arguments...)
    @test isfinite(stress) && 1e-4 < abs(stress) < 1
    mixing = MoistTurbulentMixing(model; closure, surface_exchange=stable)
    moist_turbulent_mixing_step!(mixing, model, 600)
    @test mixing.surface_fluxes.x_stress[1, 1, 1] == stress
    @test all(isfinite, model_state(model))
    @info "polynomial bulk drag with stability function" stress
end

#####
##### T4: restart
#####

@testset "Restart: the heights are recomputed, bitwise" begin
    function build()
        model = turbulent_height_column("moist_tropical_heavy_rain")
        closure = TKEBasedTurbulenceClosure(Float64)
        mixing = MoistTurbulentMixing(model; closure, surface_exchange = BulkSurfaceExchange(model, closure; bulk_conditions(301)...))
        simulation = Simulation(model; Δt=600, stop_time=3600, verbose=false)
        add_callback!(simulation, Callback(mixing, IterationInterval(1)); name=:mixing)
        return simulation, model, mixing
    end

    precipitation(model) = [getproperty(model.dynamics.moisture.ledger, name)[1, 1, 1] for name in propertynames(model.dynamics.moisture.ledger)]
    heights(model) = map(field -> Array(parent(field)), (model.dynamics.height_coordinate.layer_depth,
                                                         model.dynamics.height_coordinate.interface_spacing,
                                                         model.dynamics.height_coordinate.interface_height))

    continuous, model, mixing = build()
    run!(continuous)

    directory = mktempdir()
    first_half, first_model, = build()
    first_half.stop_time = 1800
    first_half.output_writers[:checkpointer] = Checkpointer(first_model; schedule=TimeInterval(1800), dir=directory, prefix="heights")
    run!(first_half)
    halfway = heights(first_model)

    restarted, restarted_model, restarted_mixing = build()
    restarted.output_writers[:checkpointer] = Checkpointer(restarted_model; schedule=TimeInterval(1e9), dir=directory, prefix="heights")
    run!(restarted; pickup=true)

    @test restarted_model.clock.time == 3600
    @test restarted_model.clock.iteration == model.clock.iteration
    @test model_state(restarted_model) == model_state(model)
    @test precipitation(restarted_model) == precipitation(model)
    @test field_values(restarted_mixing.ledger) == field_values(mixing.ledger)
    @test all(isequal.(heights(restarted_model), heights(model)))
    @test heights(model)[1] != halfway[1]
    @test heights_are_current(restarted_model)
end

#####
##### T7: Yin-Yang
#####

yin_yang_model(option; tracers=:ρe) =
    AtmosphereModel(YinYangGrid(CPU(), Float64; size=(48, 16, 8), z=(0, 1), halo=(4, 4, 3));
                    dynamics = HydrostaticDynamics(top_pressure=2.5e4, physical_height_grid=option),
                    microphysics = warm_rain_microphysics(), tracers,
                    formulation = moist_formulation(), momentum_advection = VectorInvariant())

function set_yin_yang_state!(model; e=0.3)
    set!(model; pˢ = 1e5,
                θ = (λ, φ, z) -> 290 + 40z + 2 * cosd(φ) * sind(2λ),
                qᵗ = (λ, φ, z) -> 0.015 * (1 - z)^3 * (1 + 0.3 * sind(λ)) * cosd(φ),
                u = (λ, φ, z) -> 10 * z^0.3 * cosd(φ) * (1 + 0.2 * sind(2λ)),
                v = (λ, φ, z) -> 3 * z * cosd(1.5φ) * sind(λ))
    for r in 1:2
        regional = getregion(model, r)
        interior(regional.tracers.ρe) .= e .* interior(regional.dynamics.moisture.total_density)
    end
    update_state!(model; compute_tendencies=false)
    return nothing
end

@testset "Yin-Yang: heights of every region, halos, and neutrality" begin
    model = yin_yang_model(true)
    set_yin_yang_state!(model)
    heights = physical_height_grid(model)
    @test getregion(heights, 1) isa HydrostaticPressureGrid
    @test getregion(heights, 2) isa HydrostaticPressureGrid
    @test heights_are_current(model)

    # The grid of a region reads the arrays of the fields of that region
    for r in 1:2
        regional = getregion(model, r)
        @test getregion(heights, r).z.layer_depth === regional.dynamics.height_coordinate.layer_depth.data
        @test Δzᶜᶜᶜ(3, 4, 2, getregion(heights, r)) == regional.dynamics.height_coordinate.layer_depth[3, 4, 2]
    end

    halo_minimum(field) = minimum(minimum(Array(parent(getregion(field, r)))[:, :, 4:end-3]) for r in 1:2)
    interior_minimum(field) = minimum(minimum(Array(interior(getregion(field, r)))) for r in 1:2)
    coordinate = model.dynamics.height_coordinate
    depth_in_halos = halo_minimum(coordinate.layer_depth)
    @info "Yin-Yang layer depth" interior_minimum = interior_minimum(coordinate.layer_depth) minimum_with_halos = depth_in_halos
    @test interior_minimum(coordinate.layer_depth) > 0
    @test depth_in_halos > 0

    # Dynamics (SSPRK3, warm rain applied by the stepper): the option changes nothing
    states = map((true, false)) do option
        twin = yin_yang_model(option)
        set_yin_yang_state!(twin)
        for n in 1:3
            time_step!(twin, 200)
        end
        option && @test heights_are_current(twin)
        model_state(twin)
    end
    @test states[1] == states[2]
    @test all(isfinite, states[1])
end

@testset "Yin-Yang: the closure on the height grid against the wrapper" begin
    results = map((true, false)) do option
        model = yin_yang_model(option)
        set_yin_yang_state!(model)
        closure = TKEBasedTurbulenceClosure(Float64)
        mixing = MoistTurbulentMixing(model; closure, surface_exchange = BulkSurfaceExchange(model, closure; bulk_conditions(300)...))
        for n in 1:3
            moist_turbulent_mixing_step!(mixing, model, 600)
        end
        κu = maximum(maximum(Array(interior(getregion(mixing.closure_fields.Kᵘ, r)))) for r in 1:2)
        (state = model_state(model, closure_fields(mixing)..., values(mixing.ledger)...), κu = κu)
    end

    @test results[1].state == results[2].state
    @test results[1].κu > 0
    @info "Yin-Yang closure" difference = maximum(abs, results[1].state .- results[2].state) κu = results[1].κu
end
