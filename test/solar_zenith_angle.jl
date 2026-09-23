include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "supposition_setup.jl"))

using Breeze
using Breeze.CelestialMechanics: cos_solar_zenith_angle, day_of_year, hour_angle,
                                 solar_declination, equation_of_time
using Dates
using Dates: AbstractDateTime, DateTime, UTInstant, Millisecond
using Test

struct WrappedDateTime{I} <: AbstractDateTime
    instant :: UTInstant{Millisecond}
    tag :: I
end

WrappedDateTime(dt::DateTime) = WrappedDateTime(dt.instant, nothing)

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

@testset "Solar geometry properties" begin
    datetimes = map(k -> DateTime(2000, 1, 1) + Millisecond(k), Data.Integers(0, 946_080_000_000))  # 30 years

    # cos θ_z is a cosine, the hour angle advances 15° per hour so longitude is 360°-periodic...
    @breeze_check function cos_zenith_angle_is_bounded_and_periodic(dt = datetimes,
                                                                    longitude = spstn_floats(Float64; lo=-180, hi=180),
                                                                    latitude = spstn_floats(Float64; lo=-90, hi=90))
        c = cos_solar_zenith_angle(dt, longitude, latitude)
        c_shifted = cos_solar_zenith_angle(dt, longitude + 360, latitude)
        return -1 <= c <= 1 && isapprox(c, c_shifted; atol=1e-9)
    end

    # ...and Spencer's (1971) declination stays within the obliquity of the ecliptic (23.45° ≈ 0.409 rad).
    @breeze_check function declination_within_obliquity(day = Data.Integers(1, 366))
        return abs(solar_declination(day)) <= 0.41
    end
end
