include(joinpath(@__DIR__, "setup.jl"))

#####
##### Tests for CelestialMechanics solar geometry
#####
##### Two things are checked:
#####
##### 1. Known-value regression for `Dates.DateTime`, so the numbers themselves are pinned.
##### 2. That the solar geometry accepts ANY `Dates.AbstractDateTime`, not only the concrete
#####    `Dates.DateTime`. Datetime types that carry non-`Int64` fields — Reactant's
#####    `ReactantDateTime`, whose milliseconds are a traced number — are `AbstractDateTime`s, and
#####    before the annotations were widened they hit a `MethodError` here rather than anything
#####    intrinsic to tracing.
#####
##### The second group deliberately uses a locally defined `AbstractDateTime` rather than Reactant,
##### so it tests the dispatch contract with no extra dependency and runs everywhere.
#####

using Breeze
using Breeze.CelestialMechanics: cos_solar_zenith_angle, day_of_year, hour_angle,
                                 solar_declination, equation_of_time
using Dates
using Dates: AbstractDateTime, DateTime, UTInstant, Millisecond
using Test

#####
##### A minimal `AbstractDateTime` that is NOT `Dates.DateTime`
#####
##### Mirrors the shape of Reactant's `ReactantDateTime`: a wrapper whose millisecond field is
##### parameterized, so it cannot be a `Dates.DateTime` even though it means the same instant.
#####

struct WrappedDateTime{I} <: AbstractDateTime
    instant :: UTInstant{Millisecond}
    tag :: I
end

WrappedDateTime(dt::DateTime) = WrappedDateTime(dt.instant, nothing)

# The accessors the solar geometry actually calls. `Dates` defines these on the concrete `DateTime`
# rather than on `AbstractDateTime` — `dayofyear` routes through `Dates.days`, and `hour`/`minute`/
# `second` are `DateTime` methods — so a new `AbstractDateTime` has to supply them. Reactant's
# `ReactantDateTime` does the same thing in its own extension. Delegating keeps this test about
# DISPATCH (does the widened signature accept a non-DateTime?) rather than about date arithmetic.
_as_datetime(dt::WrappedDateTime) = DateTime(dt.instant)

Dates.value(dt::WrappedDateTime) = dt.instant.periods.value
Dates.days(dt::WrappedDateTime) = Dates.days(_as_datetime(dt))
Dates.hour(dt::WrappedDateTime) = Dates.hour(_as_datetime(dt))
Dates.minute(dt::WrappedDateTime) = Dates.minute(_as_datetime(dt))
Dates.second(dt::WrappedDateTime) = Dates.second(_as_datetime(dt))

@testset "Solar zenith angle" begin

    @testset "Known values for DateTime" begin
        # Northern summer solstice, noon UTC at the prime meridian: sun high overhead at 23.5°N.
        solstice = DateTime(2024, 6, 21, 12, 0, 0)
        @test day_of_year(solstice) == 173
        @test solar_declination(day_of_year(solstice)) ≈ 0.4090 atol=1e-3   # ≈ +23.4°

        # At 23.5°N the solstice sun is within a degree of the zenith, so cos(θ_z) ≈ 1.
        @test cos_solar_zenith_angle(solstice, 0, 23.5) ≈ 1 atol=1e-3

        # Same instant on the opposite side of the planet is night: sun below the horizon.
        @test cos_solar_zenith_angle(solstice, 180, 23.5) < 0

        # The hour angle is zero at solar noon, up to the equation of time.
        @test abs(hour_angle(solstice, 0)) < 0.03                            # ≲ 1.7°
        @test abs(equation_of_time(day_of_year(solstice))) < 5               # minutes
    end

    @testset "Poles and horizon" begin
        # Polar night: the North Pole in December never sees the sun.
        polar_night = DateTime(2024, 12, 21, 12, 0, 0)
        @test cos_solar_zenith_angle(polar_night, 0, 89) < 0

        # Polar day: the same pole in June is lit at every hour.
        polar_day = DateTime(2024, 6, 21)
        @test all(cos_solar_zenith_angle(polar_day + Dates.Hour(h), 0, 89) > 0 for h in 0:23)
    end

    @testset "Accepts any AbstractDateTime" begin
        datetimes = (DateTime(2024, 6, 21, 12, 0, 0),
                     DateTime(2025, 12, 7, 12, 0, 0),
                     DateTime(2024, 1, 1, 0, 0, 0),
                     DateTime(2024, 2, 29, 18, 30, 0))   # leap day

        for dt in datetimes, (λ, φ) in ((0, 45), (-125, 47), (150, -30))
            wrapped = WrappedDateTime(dt)

            # A non-DateTime AbstractDateTime must dispatch...
            @test day_of_year(wrapped) == day_of_year(dt)
            @test hour_angle(wrapped, λ) == hour_angle(dt, λ)

            # ...and produce bit-identical geometry, since it denotes the same instant.
            @test cos_solar_zenith_angle(wrapped, λ, φ) == cos_solar_zenith_angle(dt, λ, φ)
        end
    end

    @testset "Single-column grid dispatch accepts AbstractDateTime" begin
        grid = RectilinearGrid(CPU(); size = 4, x = -125, y = 47, z = (0, 1000),
                               topology = (Flat, Flat, Bounded))
        dt = DateTime(2025, 12, 7, 12, 0, 0)

        @test cos_solar_zenith_angle(1, 1, grid, WrappedDateTime(dt)) ==
              cos_solar_zenith_angle(1, 1, grid, dt)
    end
end
