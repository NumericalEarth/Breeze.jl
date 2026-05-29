using Breeze
using Oceananigans
using Oceananigans: UpdateStateCallsite, TendencyCallsite, TimeStepCallsite
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, OpenBoundaryCondition
using Oceananigans.Fields: flattened_unique_values
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.OutputReaders: extract_field_time_series, InMemory
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

    @testset "Embedded FTS is reachable via the model walk" begin
        # `(fields(model), model.forcing)` — the tuple the extension walks — must carry
        # an embedded `FieldTimeSeries` through `extract_field_time_series`.
        times = collect(FT(0):FT(1):FT(3))
        fts = FieldTimeSeries{Nothing, Center, Center}(grid, times)
        model = AtmosphereModel(grid)
        walked = flattened_unique_values(extract_field_time_series((fields(model), model.forcing, fts)))
        @test any(x -> x === fts, walked)
    end

    @testset "Chunked FTS Open BC advances through update_state!" begin
        # End-to-end: a partly-in-memory (chunked) FieldTimeSeries Open BC on momentum.
        # As the model steps past the in-memory window, `update_state!` must call
        # `update_model_field_time_series!` (the #719 extension) to slide the backend
        # window forward. Without the extension the call hits the no-op `::AbstractModel`
        # fallback and `backend.start` never moves.
        # Boundary *values* are not asserted: a memory-only chunked FTS has no parent
        # file to reload slices from, so data outside the initial window is unreliable —
        # the time-pointer advance is the #719 signal here.
        cgrid = RectilinearGrid(default_arch; size=(8, 8, 4),
                                x=(0, 1000), y=(0, 1000), z=(0, 200),
                                topology=(Bounded, Bounded, Bounded))
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=FT(300),
                                        surface_pressure=FT(1e5))

        times = [FT(0), FT(10), FT(20), FT(30)]
        fts = FieldTimeSeries{Nothing, Center, Center}(cgrid, times; backend=InMemory(2))
        for n in eachindex(times)
            set!(fts[n], (y, z) -> FT(n) / FT(10))   # small momenta, ~0.1–0.4
        end

        ρu_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(fts))
        model = AtmosphereModel(cgrid; dynamics, boundary_conditions=(; ρu=ρu_bcs))
        set!(model; θ=FT(300), ρ=FT(1.17))

        start_before = fts.backend.start
        # Step past the first in-memory window (t > 10 needs a later chunk).
        time_step!(model, FT(12))
        @test model.clock.iteration == 1
        @test fts.backend.start > start_before          # update_model_field_time_series! fired
        @test !any(isnan, parent(model.momentum.ρu))
    end
end
