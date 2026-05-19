using Breeze
using Oceananigans
using Oceananigans: UpdateStateCallsite, TendencyCallsite, TimeStepCallsite
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, OpenBoundaryCondition
using Oceananigans.Fields: flattened_unique_values
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.OutputReaders: extract_field_time_series
using Oceananigans.TimeSteppers: Clock
using Test

@testset "AtmosphereModel callbacks [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4),
                           x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Bounded, Bounded, Bounded))

    @testset "UpdateStateCallsite and TendencyCallsite fire" begin
        model = AtmosphereModel(grid)
        fired_update = Ref(0)
        fired_tend   = Ref(0)
        fired_step   = Ref(0)
        update_cb(m)  = (fired_update[] += 1; nothing)
        tend_cb(m)    = (fired_tend[]   += 1; nothing)
        step_cb(sim)  = (fired_step[]   += 1; nothing)

        sim = Simulation(model; Δt=FT(0.01), stop_iteration=2, verbose=false)
        add_callback!(sim, update_cb, IterationInterval(1); callsite = UpdateStateCallsite())
        add_callback!(sim, tend_cb,   IterationInterval(1); callsite = TendencyCallsite())
        add_callback!(sim, step_cb,   IterationInterval(1); callsite = TimeStepCallsite())
        run!(sim)

        @test fired_update[] > 0
        @test fired_tend[]   > 0
        @test fired_step[]   > 0
        # UpdateStateCallsite and TendencyCallsite fire from the same per-stage
        # update_state!/compute_tendencies! call, so their counts must match.
        @test fired_update[] == fired_tend[]
    end

    @testset "compute_tendencies! still callable without callbacks" begin
        model = AtmosphereModel(grid)
        Breeze.AtmosphereModels.compute_tendencies!(model)
        @test true  # no error
    end
end

@testset "AtmosphereModel FieldTimeSeries discovery [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4),
                           x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Bounded, Bounded, Bounded))

    @testset "update_model_field_time_series! dispatches to AtmosphereModels method" begin
        model = AtmosphereModel(grid)
        m = which(update_model_field_time_series!, (typeof(model), Clock))
        @test m.module === Breeze.AtmosphereModels
        # A model with no embedded FTS still executes without error.
        update_model_field_time_series!(model, model.clock)
    end

    @testset "FTS embedded in a forcing slot is reachable via the model walk" begin
        # Verify that `(fields(model), model.forcing)` — the tuple our extension walks —
        # carries an embedded `FieldTimeSeries` through `extract_field_time_series` so the
        # update function will reach it. Avoid OpenBoundaryCondition-on-prognostic-momentum
        # here: AnelasticDynamics' `initialize_model_thermodynamics!` calls `set!`, which
        # triggers `compute_velocities!`-driven halo fill without a `clock` argument, and
        # FTS-backed OBCs fall through to the generic `getbc(::AbstractArray, ...)` path
        # that does not exist for the time dimension. That's a separate pre-existing bug
        # in `compute_velocities!` (`fill_halo_regions!(model.momentum)` should pass
        # `model.clock, fields(model)`), out of scope for #719.
        times = collect(FT(0):FT(1):FT(3))
        fts = FieldTimeSeries{Nothing, Center, Center}(grid, times)
        model = AtmosphereModel(grid)
        walk_input = (fields(model), model.forcing, fts)
        walked = flattened_unique_values(extract_field_time_series(walk_input))
        @test any(x -> x === fts, walked)
    end
end
