include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: update_radiation_between_solves!, _update_radiation!
using Breeze.CelestialMechanics: cos_solar_zenith_angle
using Dates
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Units
using Oceananigans: UpdateStateCallsite, prognostic_state, restore_prognostic_state!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Utils: SpecifiedTimes
using Test

# Load both radiation extensions
using ClimaComms
using NCDatasets
using RRTMGP
using NumericalRadiation: NumericalRadiation

# A cloudy column, by default at 45°N with an hourly solve and a numeric clock anchored at `epoch`.
function radiating_column(grid; epoch = nothing,
                          optics = GrayOptics(),
                          surface_temperature = 300,
                          solar_position = ApparentSolarPosition(; epoch),
                          schedule = TimeInterval(1hours),
                          clock = Clock(time=zero(eltype(grid))),
                          dynamics = nothing,
                          θ = z -> 300 + 5e-6 * z,
                          qᵗ = z -> 0.020 * exp(-z / 3000))
    constants = ThermodynamicConstants()
    if isnothing(dynamics) || dynamics === PrescribedDynamics
        reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=300)
        dynamics = isnothing(dynamics) ? AnelasticDynamics(reference_state) : PrescribedDynamics(reference_state)
    end
    radiation = RadiativeTransferModel(grid, optics, constants;
                                       surface_temperature, surface_emissivity = 0.98, surface_albedo = 0.1,
                                       solar_position, schedule)

    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    model = AtmosphereModel(grid; dynamics, microphysics, radiation, clock)
    Oceananigans.initialize!(schedule, model) # count the schedule's time from the clock
    set!(model; θ, qᵗ)
    return model
end

solved_cosine(model) = Array(model.radiation.approximate_updates.solved_cos_zenith)

column(field) = vec(Array(interior(field)))
longwave_columns(radiation) = column.((radiation.upwelling_longwave_flux, radiation.downwelling_longwave_flux))
top_downwelling_shortwave(radiation) = last(column(radiation.downwelling_shortwave_flux))

@testset "Approximate radiation updates between solves [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    @testset "Longwave fluxes follow the surface temperature: $(nameof(typeof(optics)))" for optics in
            (GrayOptics(), ClearSkyOptics(), AllSkyOptics(), EcCKDOptics())

        grid = radiation_column_grid(FT, 16, 10kilometers)
        T₀ = Field{Center, Center, Nothing}(grid)
        set!(T₀, 300)
        model = radiating_column(grid; epoch = DateTime(2024, 6, 21, 12), optics, surface_temperature = T₀)
        radiation = model.radiation
        before = longwave_columns(radiation)

        set!(T₀, 305)
        update_radiation_between_solves!(radiation, model)
        updated = longwave_columns(radiation)

        _update_radiation!(radiation, model)
        solved = longwave_columns(radiation)

        # Gray longwave fluxes are linear in the surface emission, so the update matches a solve;
        # spectral ones nearly, since the spectrum of the emission shifts with the temperature
        rtol = optics isa GrayOptics ? sqrt(eps(FT)) : FT(0.02)
        @test updated[1] .- before[1] ≈ solved[1] .- before[1] rtol = rtol
        @test updated[2] .- before[2] ≈ solved[2] .- before[2] rtol = rtol
    end

    @testset "Longwave fluxes follow the surface temperature RRTMGP solves at" begin
        # RRTMGP's Planck tables end at 355 K, into which the solve clamps the surface temperature.
        # Near the top the response comes from a cooler surface, and above it the fluxes stay put.
        grid = radiation_column_grid(FT, 16, 10kilometers)
        T₀ = Field{Center, Center, Nothing}(grid)
        for (solved_temperature, updated_temperature) in ((354.6, 354.9), (354.6, 356), (358, 360))
            set!(T₀, solved_temperature)
            model = radiating_column(grid; epoch = DateTime(2024, 6, 21, 12), optics = ClearSkyOptics(), surface_temperature = T₀)
            radiation = model.radiation
            before = longwave_columns(radiation)

            set!(T₀, updated_temperature)
            update_radiation_between_solves!(radiation, model)
            updated = longwave_columns(radiation)

            _update_radiation!(radiation, model)
            solved = longwave_columns(radiation)

            @test updated[1] .- before[1] ≈ solved[1] .- before[1] rtol = FT(0.02)
            @test updated[2] .- before[2] ≈ solved[2] .- before[2] rtol = FT(0.02)
        end
    end

    @testset "Shortwave fluxes follow the sun" begin
        epoch = DateTime(2024, 6, 21, 10)
        model = radiating_column(radiation_column_grid(FT, 16, 10kilometers); epoch)
        radiation = model.radiation
        S₀ = radiation.solar_constant

        # The solve scales its fluxes to the sun of the moment, and each update moves them with the sun
        @test top_downwelling_shortwave(radiation) ≈ -S₀ * cos_solar_zenith_angle(epoch, 0, 45) rtol = sqrt(eps(FT))

        model.clock.time = 45minutes
        update_radiation_between_solves!(radiation, model)
        @test top_downwelling_shortwave(radiation) ≈ -S₀ * cos_solar_zenith_angle(epoch + Minute(45), 0, 45) rtol = sqrt(eps(FT))
    end

    @testset "A sun rising between solves" begin
        # The sun rises at 45°N on the solstice at about 04:13 UTC on the prime meridian
        epoch = DateTime(2024, 6, 21, 3, 30)
        model = radiating_column(radiation_column_grid(FT, 16, 10kilometers); epoch)
        radiation = model.radiation
        @test all(iszero, column(radiation.downwelling_shortwave_flux))

        model.clock.time = 55minutes
        update_radiation_between_solves!(radiation, model)
        μ₀ = cos_solar_zenith_angle(epoch + Minute(55), 0, 45)
        @test μ₀ > 0
        @test top_downwelling_shortwave(radiation) ≈ -radiation.solar_constant * μ₀ rtol = sqrt(eps(FT))
        @test @allowscalar radiation.downwelling_shortwave_flux[1, 1, 1] < 0
    end

    @testset "Time stepping between solves" begin
        model = radiating_column(radiation_column_grid(FT, 16, 10kilometers); epoch = DateTime(2024, 6, 21, 8))
        radiation = model.radiation
        solved = top_downwelling_shortwave(radiation)

        time_step!(model, 60) # the first step solves again with its time step known
        solved_window_mean = solved_cosine(model)
        for _ in 1:2
            time_step!(model, 60)
        end

        # The morning sun climbs between solves without solving again, and the column absorbs the net flux into it
        @test solved_cosine(model) == solved_window_mean
        @test top_downwelling_shortwave(radiation) < solved
        ℐ_net = column(radiation.upwelling_longwave_flux) .+ column(radiation.downwelling_longwave_flux) .+
                column(radiation.upwelling_shortwave_flux) .+ column(radiation.downwelling_shortwave_flux)
        Δz = 10kilometers / 16
        @test Δz * sum(column(radiation.flux_divergence)) ≈ ℐ_net[1] - ℐ_net[end] rtol = sqrt(eps(FT))
    end

    @testset "Calendar and explicit-time schedules" begin
        grid = radiation_column_grid(FT, 8, 10kilometers)
        epoch = DateTime(2024, 6, 21, 10)
        first_half_hour = solved_cosine(radiating_column(grid; epoch, schedule=TimeInterval(30minutes)))
        second_half_hour = solved_cosine(radiating_column(grid; epoch=epoch + Minute(30), schedule=TimeInterval(30minutes)))

        for (schedule, clock) in ((TimeInterval(Minute(30)), Clock(time=epoch)),
                                  (TimeInterval([30minutes, 1hours]), Clock(time=FT(0))),
                                  (SpecifiedTimes([30minutes, 1hours]), Clock(time=FT(0))),
                                  (TimeInterval([epoch + Minute(30), epoch + Hour(1)]), Clock(time=epoch)))
            model = radiating_column(grid; epoch, schedule, clock)
            @test solved_cosine(model) ≈ first_half_hour

            # Actuate at the first listed time; the next solve is another half hour away.
            model.clock.time += ifelse(model.clock.time isa DateTime, Minute(30), 30minutes)
            model.clock.iteration = 1
            model.clock.stage = 2
            update_state!(model; compute_tendencies=false)
            @test solved_cosine(model) ≈ second_half_hour
            expected = -model.radiation.solar_constant * cos_solar_zenith_angle(epoch + Minute(30), 0, 45)
            @test top_downwelling_shortwave(model.radiation) ≈ expected rtol = sqrt(eps(FT))

            # Exhaust explicit times without trying to sample an infinite interval.
            model.clock.time += ifelse(model.clock.time isa DateTime, Hour(1), 1hours)
            update_state!(model; compute_tendencies=false)
            @test all(isfinite, column(model.radiation.downwelling_shortwave_flux))
        end

        # An unknown interval still follows the moving sun and becomes dark at sunset.
        model = radiating_column(grid; epoch, schedule=TimeInterval(Inf))
        solved = top_downwelling_shortwave(model.radiation)
        @test solved ≈ -model.radiation.solar_constant * cos_solar_zenith_angle(epoch, 0, 45) rtol = sqrt(eps(FT))
        model.clock.time += 1hours
        model.clock.iteration = 1
        update_state!(model; compute_tendencies=false)
        expected = -model.radiation.solar_constant * cos_solar_zenith_angle(epoch + Hour(1), 0, 45)
        @test top_downwelling_shortwave(model.radiation) ≈ expected rtol = sqrt(eps(FT))
        model.clock.time = 14hours
        update_state!(model; compute_tendencies=false)
        @test all(iszero, column(model.radiation.downwelling_shortwave_flux))
        @test all(iszero, column(model.radiation.upwelling_shortwave_flux))
        @test model.radiation.schedule.actuations == 0
    end

    @testset "Daylight at the ends of the radiation interval" begin
        grid = radiation_column_grid(FT, 8, 10kilometers)
        solar_position = DiurnalSolarPosition(FT; latitude=0)
        current_sun(model) = -model.radiation.solar_constant * cos(2π * model.clock.time / solar_position.day_length)

        # The sun sets a minute after the solve.
        model = radiating_column(grid; solar_position, clock=Clock(time=FT(6hours - 1minutes)))
        @test top_downwelling_shortwave(model.radiation) ≈ current_sun(model) rtol = sqrt(eps(FT))

        # The sun rises a minute before the next solve.
        model = radiating_column(grid; solar_position, clock=Clock(time=FT(-7hours + 1minutes)))
        @test iszero(top_downwelling_shortwave(model.radiation))
        model.clock.time = FT(-6hours + 30)
        model.clock.iteration = 1
        update_state!(model; compute_tendencies=false)
        @test top_downwelling_shortwave(model.radiation) ≈ current_sun(model) rtol = sqrt(eps(FT))

        # A night-time solve with no known interval must acquire a transmittance at sunrise.
        model = radiating_column(grid; solar_position, schedule=TimeInterval(Inf), clock=Clock(time=FT(-6hours - 1minutes)))
        model.clock.time = FT(-5hours)
        model.clock.iteration = 1
        update_state!(model; compute_tendencies=false)
        @test top_downwelling_shortwave(model.radiation) ≈ current_sun(model) rtol = sqrt(eps(FT))
        @test any(>(0), column(model.radiation.upwelling_shortwave_flux))
        @test model.radiation.schedule.actuations == 0
    end

    @testset "Shortwave path uses the duration-weighted daylight mean" begin
        grid = radiation_column_grid(FT, 8, 10kilometers)
        solar_position = DiurnalSolarPosition(FT; latitude=0)
        sunlit_time = 24minutes
        angle_span = 2π * sunlit_time / solar_position.day_length
        mean_cosine = (1 - cos(angle_span)) / angle_span
        reference = radiating_column(grid; solar_position=FixedCosineZenith(FT(mean_cosine)))

        # Only 24 minutes of a three-hour interval are sunlit, at either end of the interval.
        for start_time in (-9hours + sunlit_time, 6hours - sunlit_time)
            model = radiating_column(grid; solar_position, schedule=TimeInterval(3hours), clock=Clock(time=FT(start_time)))
            @test only(Array(model.radiation.approximate_updates.solved_cos_zenith)) ≈ mean_cosine rtol = sqrt(eps(FT))
            model.clock.time = FT(ifelse(start_time < 0, -6hours + sunlit_time / 2, 6hours - sunlit_time / 2))
            model.clock.iteration = 1
            update_state!(model; compute_tendencies=false)

            # Normalizing by incoming sunlight isolates the optical path and absorption profile.
            for name in (:upwelling_shortwave_flux, :downwelling_shortwave_flux)
                flux = column(getproperty(model.radiation, name)) / top_downwelling_shortwave(model.radiation)
                expected = column(getproperty(reference.radiation, name)) / top_downwelling_shortwave(reference.radiation)
                @test flux ≈ expected rtol = sqrt(eps(FT))
            end
        end
    end

    @testset "A sun climbing far above the slant path of the last solve solves again" begin
        grid = radiation_column_grid(FT, 8, 10kilometers)
        solar_position = DiurnalSolarPosition(FT; latitude=0)
        sun(time) = cos(2π * time / solar_position.day_length)

        # With no known interval each solve takes the current sun, here a minute above the horizon.
        # Once the sun is four times higher, that slant path would put the beam far too high in the
        # column, and the update solves again.
        model = radiating_column(grid; solar_position, schedule=TimeInterval(Inf), clock=Clock(time=FT(-6hours - 1minutes)))
        model.clock.iteration = 1
        for (time, solved_time) in ((-6hours + 1minutes, -6hours + 1minutes),
                                    (-6hours + 3minutes, -6hours + 1minutes),
                                    (-3hours, -3hours))
            model.clock.time = FT(time)
            update_state!(model; compute_tendencies=false)
            @test only(solved_cosine(model)) ≈ sun(solved_time) rtol = sqrt(eps(FT))
            @test top_downwelling_shortwave(model.radiation) ≈ -model.radiation.solar_constant * sun(time) rtol = sqrt(eps(FT))
        end
        @test model.radiation.schedule.actuations == 0

        reference = radiating_column(grid; solar_position=FixedCosineZenith(FT(sun(-3hours))))
        for name in (:upwelling_shortwave_flux, :downwelling_shortwave_flux)
            flux = column(getproperty(model.radiation, name)) / top_downwelling_shortwave(model.radiation)
            expected = column(getproperty(reference.radiation, name)) / top_downwelling_shortwave(reference.radiation)
            @test flux ≈ expected rtol = sqrt(eps(FT))
        end
    end

    @testset "Sunlight past an underestimated radiation interval solves again" begin
        grid = radiation_column_grid(FT, 8, 10kilometers)
        solar_position = DiurnalSolarPosition(FT; latitude=0)
        ω = 2π / solar_position.day_length
        window_mean(start, duration) = (sin(ω * (start + duration)) - sin(ω * start)) / (ω * duration)

        # An iteration schedule takes its interval from the last time step: here four steps of 10 s,
        # the last 10 s of them sunlit. Should the steps grow, the next solve comes long after sunrise.
        clock = Clock(time=FT(-6hours - 30), last_Δt=FT(10))
        model = radiating_column(grid; solar_position, schedule=IterationInterval(4), clock)
        @test only(solved_cosine(model)) ≈ window_mean(-6hours, 10) rtol = 1e-3

        model.clock.time = FT(-5hours)
        model.clock.iteration = 2
        update_state!(model; compute_tendencies=false)
        @test only(solved_cosine(model)) ≈ window_mean(-5hours, 20) rtol = 1e-3
    end

    @testset "A solve in the time step of sunset averages from the start of the step" for dynamics in
            (nothing, CompressibleDynamics(SplitExplicitTimeDiscretization()))
        grid = radiation_column_grid(FT, 8, 10kilometers)
        solar_position = DiurnalSolarPosition(FT; latitude=0)
        ω = 2π / solar_position.day_length
        Δt = 10

        # The sun sets at 6 h, 7 s into the third time step, which solves. SSP RK3 evaluates the
        # first stage of that step after sunset, and the second before it.
        model = radiating_column(grid; dynamics, solar_position, schedule=IterationInterval(2),
                                 clock=Clock(time=FT(6hours - 2Δt - 7)))
        for _ in 1:2
            time_step!(model, Δt)
        end

        solved = FT[]
        record_solved_cosine(model) = push!(solved, only(solved_cosine(model)))
        callback = Callback(record_solved_cosine, IterationInterval(1); callsite=UpdateStateCallsite())
        time_step!(model, Δt; callbacks=[callback])

        # One solve, averaged over the 7 sunlit seconds of its window from the start of the step,
        # and no daylight solve at the stage that returns to before sunset
        @test length(solved) == 3
        @test all(==(first(solved)), solved)
        @test first(solved) ≈ (1 - cos(7ω)) / 7ω rtol = 1e-3
    end

    @testset "Columns around the globe solve only on schedule" begin
        grid = LatitudeLongitudeGrid(default_arch, FT; size=(24, 12, 4), longitude=(-180, 180), latitude=(-84, 84),
                                     z=(0, 10kilometers))
        θ(λ, φ, z) = 300 + 5e-6 * z
        qᵗ(λ, φ, z) = 0.020 * exp(-z / 3000)
        model = radiating_column(grid; epoch=DateTime(2024, 6, 21), dynamics=PrescribedDynamics, θ, qᵗ)

        # Every solve changes the solved cosine of the columns it lights
        solves = Ref(0)
        last_solved = Ref(solved_cosine(model))
        function count_solves(model)
            current = solved_cosine(model)
            solves[] += current != last_solved[]
            last_solved[] = current
        end
        callback = Callback(count_solves, IterationInterval(1); callsite=UpdateStateCallsite())
        for _ in 1:12
            time_step!(model, 10minutes; callbacks=[callback])
        end

        # The first step solves again with its time step known, and the hourly schedule twice more.
        # Along the terminator no stage finds sunlight its solve did not average over.
        @test solves[] == 3
        @test model.radiation.schedule.actuations == 2
    end

    @testset "Fluxes follow every stage and the completed step" for dynamics in
            (nothing, CompressibleDynamics(SplitExplicitTimeDiscretization()))
        grid = radiation_column_grid(FT, 8, 10kilometers)
        solar_position = DiurnalSolarPosition(FT; latitude=0)
        model = radiating_column(grid; dynamics, solar_position, clock=Clock(time=FT(5hours)))
        stages = Int[]
        function check_sun(model)
            push!(stages, model.clock.stage)
            expected = -model.radiation.solar_constant * cos(2π * model.clock.time / solar_position.day_length)
            @test top_downwelling_shortwave(model.radiation) ≈ expected rtol = sqrt(eps(FT))
        end
        callback = Callback(check_sun, IterationInterval(1); callsite=UpdateStateCallsite())
        time_step!(model, 10; callbacks=[callback])
        @test stages == [1, 2, 3, 1]
        @test model.radiation.schedule.actuations == 0 # every stage updates between solves
    end

    @testset "Older checkpoints refresh missing radiation caches: $schedule_kind" for schedule_kind in (:hourly, :every_step)
        make_schedule() = schedule_kind === :hourly ? TimeInterval(1hours) : IterationInterval(1)
        grid = radiation_column_grid(FT, 8, 10kilometers)
        solar_position = DiurnalSolarPosition(FT; latitude=0, noon_offset=12hours)
        model = radiating_column(grid; solar_position, schedule=make_schedule())
        model.clock.time = FT(12hours)
        model.clock.iteration = 10
        model.clock.stage = 2
        update_state!(model; compute_tendencies=false)
        model.clock.stage = 1
        saved = prognostic_state(model)
        legacy_radiation = (; (name => value for (name, value) in pairs(saved.radiation)
                              if name ∉ (:approximate_updates, :cache_valid))...)
        without_cache = merge(saved, (; radiation=legacy_radiation))
        without_radiation = (; (name => value for (name, value) in pairs(saved) if name !== :radiation)...)
        without_validity = merge(saved, (; radiation=(; (name => value for (name, value) in pairs(saved.radiation)
                                                         if name !== :cache_valid)...)))
        array_validity = merge(saved, (; radiation=merge(saved.radiation, (; cache_valid=[true]))))

        for checkpoint in (without_cache, without_radiation, without_validity, array_validity)
            restored = radiating_column(grid; solar_position, schedule=make_schedule())
            @test iszero(top_downwelling_shortwave(restored.radiation))
            restore_prognostic_state!(restored, checkpoint)
            # A second checkpoint before any state update must preserve invalidity too.
            intermediate = prognostic_state(restored)
            restored = radiating_column(grid; solar_position, schedule=make_schedule())
            restore_prognostic_state!(restored, intermediate)
            schedule_state = prognostic_state(restored.radiation.schedule)
            update_state!(restored; compute_tendencies=false)
            @test top_downwelling_shortwave(restored.radiation) ≈ -restored.radiation.solar_constant
            for (restored_flux, expected_flux) in zip(longwave_columns(restored.radiation), longwave_columns(model.radiation))
                @test restored_flux ≈ expected_flux rtol = sqrt(eps(FT))
            end
            @test prognostic_state(restored.radiation.schedule) == schedule_state
        end
    end

    @testset "Restart between solves with a changing surface temperature" begin
        grid = radiation_column_grid(FT, 8, 10kilometers)
        epoch = DateTime(2024, 6, 21, 10)
        T₀ = Field{Center, Center, Nothing}(grid)
        set!(T₀, 300)
        continuous = radiating_column(grid; epoch, surface_temperature=T₀, schedule=IterationInterval(10))
        for _ in 1:2
            time_step!(continuous, 10)
        end
        set!(T₀, 305)
        time_step!(continuous, 10)
        saved = prognostic_state(continuous)

        T₀_restored = Field{Center, Center, Nothing}(grid)
        set!(T₀_restored, 305)
        restored = radiating_column(grid; epoch, surface_temperature=T₀_restored, schedule=IterationInterval(10))
        restore_prognostic_state!(restored, saved)
        update_state!(restored)

        set!(T₀, 307)
        set!(T₀_restored, 307)
        for model in (continuous, restored)
            for _ in 1:2
                time_step!(model, 10)
            end
        end

        @test longwave_columns(restored.radiation) == longwave_columns(continuous.radiation)
        @test column(restored.radiation.downwelling_shortwave_flux) == column(continuous.radiation.downwelling_shortwave_flux)
        for (original_field, restored_field) in zip(Oceananigans.prognostic_fields(continuous), Oceananigans.prognostic_fields(restored))
            @test column(original_field) == column(restored_field)
        end
    end
end
