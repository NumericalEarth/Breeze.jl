using Breeze
using Oceananigans
using Test

include("hydrostatic_moist_utils.jl")

using Oceananigans.TimeSteppers: update_state!

using Breeze.HydrostaticEquations: column_physics_in_time_step!, warm_rain_column_physics!
using Breeze.HydrostaticMoistTurbulence: MoistTurbulentMixing, moist_turbulent_mixing_step!, MoistColumnPhysics

#####
##### The turbulent mixing in the single dispatch of the column physics of moist `HydrostaticDynamics`:
##### the warm rain and the mixing are each applied once per time step, whoever applies them.
##### Expectation, written before the first run: the three ways below agree BITWISE.
#####

const columns_directory = joinpath(@__DIR__, "hydrostatic_moist_columns")

function mixing_column()
    column, header = read_column(joinpath(columns_directory, "moist_tropical_heavy_rain.csv"))
    Nz = length(column.k)
    grid = RectilinearGrid(CPU(), Float64; size=Nz, x=0, y=0, z=(0, 1), topology=(Flat, Flat, Bounded))
    A = vcat(column.A_lower, column.A_upper[end])
    B = vcat(column.B_lower, column.B_upper[end])
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B)
    model = AtmosphereModel(grid; dynamics, microphysics=warm_rain_microphysics(), tracers = :ρe,
                            formulation = moist_formulation(), momentum_advection = VectorInvariant())
    set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, column.r_e, column.r_r)
    interior(model.velocities.u) .= reshape([10 * (k / Nz)^0.3 for k in 1:Nz], 1, 1, Nz)
    interior(model.tracers.ρe) .= 0.1 .* interior(model.dynamics.moisture.total_density)
    update_state!(model; compute_tendencies=false)

    mixing = MoistTurbulentMixing(model)
    set!(mixing.surface_fluxes.vapor_flux, 1e-4)
    set!(mixing.surface_fluxes.surface_temperature, 301)
    set!(mixing.surface_fluxes.x_stress, -0.05)
    return model, mixing
end

state(model) = reduce(vcat, map(field -> vec(Array(interior(field))), values(Oceananigans.prognostic_fields(model))))
ledgers(model, mixing) = vcat([field[1, 1, 1] for field in values(mixing.ledger)],
                              [field[1, 1, 1] for field in values(model.dynamics.moisture.ledger)])

@testset "Turbulent mixing in the single dispatch of the column physics" begin
    Δt, steps = 600, 6

    # The time stepper applies the warm rain; the mixing is a callback after it
    model, mixing = mixing_column()
    simulation = Simulation(model; Δt, stop_iteration=steps, verbose=false)
    add_callback!(simulation, Callback(mixing, IterationInterval(1)); name=:mixing)
    run!(simulation)
    @test model.dynamics.moisture.ledger.surface_rain[1, 1, 1] > 1
    @test mixing.ledger.shear_production[1, 1, 1] > 0

    # One callable applies both, the time stepper neither
    dispatched_model, dispatched_mixing = mixing_column()
    column_physics_in_time_step!(dispatched_model, false)
    dispatched = Simulation(dispatched_model; Δt, stop_iteration=steps, verbose=false)
    add_callback!(dispatched, Callback(MoistColumnPhysics(dispatched_mixing), IterationInterval(1)); name=:column_physics)
    run!(dispatched)
    @test state(dispatched_model) == state(model)
    @test ledgers(dispatched_model, dispatched_mixing) == ledgers(model, mixing)

    # A driver applies both by hand
    manual_model, manual_mixing = mixing_column()
    column_physics_in_time_step!(manual_model, false)
    for n in 1:steps
        time_step!(manual_model, Δt)
        warm_rain_column_physics!(manual_model.microphysics, manual_model)
        moist_turbulent_mixing_step!(manual_mixing, manual_model, Δt)
    end
    @test state(manual_model) == state(model)
    @test ledgers(manual_model, manual_mixing) == ledgers(model, mixing)

    # Without taking the column physics from the time stepper the warm rain would be applied twice
    refused_model, refused_mixing = mixing_column()
    refused = Simulation(refused_model; Δt, stop_iteration=1, verbose=false)
    add_callback!(refused, Callback(MoistColumnPhysics(refused_mixing), IterationInterval(1)); name=:column_physics)
    @test_throws ArgumentError run!(refused)

    # More applications of the warm rain per step change the answer, and stay non-negative
    substepped_model, substepped_mixing = mixing_column()
    column_physics_in_time_step!(substepped_model, false)
    substepped = Simulation(substepped_model; Δt, stop_iteration=steps, verbose=false)
    add_callback!(substepped, Callback(MoistColumnPhysics(substepped_mixing; warm_rain_substeps=8), IterationInterval(1)); name=:column_physics)
    run!(substepped)
    @test state(substepped_model) != state(model)
    @test minimum(substepped_model.moisture_density) ≥ 0
    @test minimum(substepped_model.microphysical_fields.ρqʳ) ≥ 0
    @test minimum(substepped_model.tracers.ρe) ≥ 0
end
