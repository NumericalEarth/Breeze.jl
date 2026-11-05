include(joinpath(@__DIR__, "setup.jl"))

using Breeze.AtmosphereModels: radiation_interval, radiation_update_due, radiation_window_lead
using Breeze.TimeSteppers: SSPRungeKutta3, AcousticRungeKutta3
using Dates: DateTime, Millisecond, Second
using Oceananigans: Oceananigans, Clock, IterationInterval, TimeInterval, prognostic_state
using Oceananigans.Utils: SpecifiedTimes, time_difference_seconds
using Test

advance_window_time(time::Number, seconds) = time + seconds
advance_window_time(time::DateTime, seconds) = time + Millisecond(round(Int, 1000seconds))

# Find the next solving step by visiting future first-stage clocks and evaluating the actual
# schedule. This is independent of the interval estimator's rounding and offset calculation.
function next_radiation_step_start(schedule, clock, step_start, first_stage, Δt)
    future_schedule = deepcopy(schedule)
    future_clock = deepcopy(clock)
    future = (; clock=future_clock)
    for step in 0:20
        step == 0 && clock.stage != 1 && continue
        future_clock.iteration = clock.iteration + step
        future_clock.stage = 2
        future_clock.time = advance_window_time(step_start, (step + first_stage) * Δt)
        if radiation_update_due(future_schedule, future_clock, future)
            return advance_window_time(step_start, step * Δt)
        end
    end
    error("No radiation solve found within the test window")
end

@testset "Radiation windows [$FT]" for FT in all_float_types()
    Oceananigans.defaults.FloatType = FT
    Δt = FT(600)
    # Only stage coefficients are needed; no fields or acoustic solver are used here.
    steppers = ((SSPRungeKutta3(FT(1), FT(1/4), FT(2/3), nothing, nothing, nothing), (FT(1), FT(1/2))),
                (AcousticRungeKutta3(FT(1/3), FT(1/2), FT(1), nothing, nothing, nothing, nothing), (FT(1/3), FT(1/2))))

    @testset "Next solving step: $(nameof(typeof(timestepper))), stage $stage, $step_start" for
            (timestepper, (first_stage, second_stage)) in steppers, stage in 1:3,
            step_start in (FT(0), DateTime(2024, 6, 21, 8))
        offset = (FT(0), first_stage, second_stage)[stage] * Δt
        stage_Δt = (1 - second_stage, first_stage, second_stage - first_stage)[stage] * Δt
        clock = Clock(time=advance_window_time(step_start, offset), last_Δt=Δt,
                      last_stage_Δt=stage_Δt, iteration=6, stage=stage)
        lead = radiation_window_lead(clock)
        window_start = advance_window_time(clock.time, -lead)

        # The window must end at the start of the step whose first stage the schedule actuates
        function test_window(schedule)
            duration = radiation_interval(schedule, clock, lead, timestepper)
            next_start = next_radiation_step_start(schedule, clock, step_start, first_stage, Δt)
            @test duration ≈ time_difference_seconds(next_start, window_start) rtol=8eps(FT)
        end

        # Include exact first-stage boundaries for both steppers, where rounding up must
        # not accidentally add an entire time step, including with calendar clocks.
        for interval in (900, 3600, 3700, 3800, 4200)
            next_time = advance_window_time(step_start, interval)
            period = step_start isa DateTime ? Second(interval) : FT(interval)
            for schedule in (TimeInterval(period), SpecifiedTimes([next_time]))
                Oceananigans.initialize!(schedule, (; clock=Clock(time=step_start)))
                before = prognostic_state(schedule)
                test_window(schedule)
                @test prognostic_state(schedule) == before
            end
        end

        for schedule in (IterationInterval(3), IterationInterval(4), IterationInterval(6, offset=1))
            test_window(schedule)
        end
    end

    @testset "Equivalent hourly schedules use the same solar window" begin
        for (timestepper, (first_stage, _)) in steppers
            clock = Clock(time=first_stage * Δt, last_Δt=Δt, last_stage_Δt=first_stage * Δt, iteration=6, stage=2)
            schedule = TimeInterval(FT(3600))
            Oceananigans.initialize!(schedule, (; clock))
            lead = radiation_window_lead(clock)
            @test radiation_interval(schedule, clock, lead, timestepper) ==
                  radiation_interval(IterationInterval(6), clock, lead, timestepper) == FT(3600)
            @test radiation_interval(schedule, clock, lead) == FT(3600)
        end
    end

    @testset "Initialization and unknown timing" begin
        for (timestepper, (first_stage, _)) in steppers, iteration in (0, 6)
            clock = Clock(time=FT(0), last_Δt=Δt, iteration=iteration)
            schedule = TimeInterval(FT(100))
            lead = radiation_window_lead(clock)
            next_start = next_radiation_step_start(schedule, clock, clock.time, first_stage, Δt)
            @test radiation_interval(schedule, clock, lead, timestepper) == next_start
        end

        clock = Clock(time=FT(0))
        @test radiation_interval(TimeInterval(FT(3600)), clock, FT(0)) == FT(3600)
        @test iszero(radiation_interval(TimeInterval(FT(Inf)), clock, FT(0)))
        @test iszero(radiation_interval(SpecifiedTimes(FT[]), clock, FT(0)))

        for (timestepper, _) in steppers, stage in (2, 3)
            clock = Clock(time=FT(0), last_Δt=Δt, iteration=6, stage=stage)
            @test radiation_interval(TimeInterval(FT(3600)), clock, FT(0), timestepper) == FT(3600)
        end
    end
end
