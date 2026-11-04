include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: average_solve_cos_zenith!, sunlit_cosine_integral
using Breeze.CelestialMechanics: cos_solar_zenith_angle
using Dates
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.Units
using Test

# Independently sample the instantaneous solar position at one-second intervals.
function sampled_daylight_mean(epoch, longitude, latitude, interval)
    cosines = [cos_solar_zenith_angle(epoch + Second(t), longitude, latitude) for t in 0:interval-1]
    sunlit = filter(>(0), cosines)
    return isempty(sunlit) ? 0.0 : sum(sunlit) / length(sunlit)
end

@testset "Sunlit duration averages [$(FT)]" for FT in all_float_types()
    grid = radiation_column_grid(FT, 2, 1kilometers)
    cosine = on_architecture(default_arch, zeros(FT, 1))
    duration = similar(cosine)
    rtol = sqrt(eps(FT))
    @test @inferred(sunlit_cosine_integral(FT(0.1), FT(0.9), FT(-π), FT(2π))) isa Tuple{FT, FT}

    @testset "Diurnal sunrise, sunset, and whole days" begin
        solar_position = DiurnalSolarPosition(FT; latitude=0)
        daylight = 24minutes
        angle = 2π * daylight / 1days
        expected = (1 - cos(angle)) / angle
        for first_time in (-9hours + daylight, 6hours - daylight)
            average_solve_cos_zenith!(cosine, duration, solar_position, grid, (; time=FT(first_time)), 3hours)
            @test only(Array(cosine)) ≈ expected rtol = rtol
            @test only(Array(duration)) ≈ daylight rtol = rtol
        end

        # The full-day mean over daylight is 2/π, irrespective of the start time or number of days.
        for rotations in (1, 3), first_time in (-5hours, 0, 10hours)
            average_solve_cos_zenith!(cosine, duration, solar_position, grid, (; time=FT(first_time)), rotations * 1days)
            @test only(Array(cosine)) ≈ 2 / π rtol = rtol
            @test only(Array(duration)) ≈ rotations * 12hours rtol = rtol
        end

        average_solve_cos_zenith!(cosine, duration, solar_position, grid, (; time=FT(10hours)), 4hours)
        @test iszero(only(Array(cosine)))
        @test iszero(only(Array(duration)))
    end

    @testset "Polar day and night" for latitude in (-90, 90), declination in (-23.5, 23.5)
        solar_position = DiurnalSolarPosition(FT; latitude, declination)
        average_solve_cos_zenith!(cosine, duration, solar_position, grid, (; time=FT(0)), 2days)
        expected = max(0, sind(latitude) * sind(declination))
        @test only(Array(cosine)) ≈ expected rtol = rtol
        @test only(Array(duration)) ≈ ifelse(expected > 0, 2days, 0) rtol = rtol
    end

    @testset "Apparent sun at date boundaries and high latitudes" for
            (epoch, interval, coordinate) in ((DateTime(2024, 6, 21, 3), 3hours, (0, 45)),
                                            (DateTime(2024, 6, 21, 23), 10hours, (120, 45)),
                                            (DateTime(2024, 12, 31, 20), 1days, (-70, -30)),
                                            (DateTime(2024, 12, 21, 0), 1days, (0, 66.55)),
                                            (DateTime(2024, 6, 21, 0), 1days, (0, 80)))
        solar_position = ApparentSolarPosition(; coordinate, epoch)
        average_solve_cos_zenith!(cosine, duration, solar_position, grid, (; time=FT(0)), interval)
        expected = sampled_daylight_mean(epoch, coordinate..., Int(interval))
        @test only(Array(cosine)) ≈ expected rtol = 1e-3 atol = 1e-6
    end

    @testset "Per-column apparent sun" begin
        grid = LatitudeLongitudeGrid(default_arch, FT; size=(4, 3, 2),
                                     longitude=(-180, 180), latitude=(-75, 75), z=(0, 1kilometers))
        cosines = on_architecture(default_arch, zeros(FT, 12))
        durations = similar(cosines)
        epoch = DateTime(2024, 6, 21, 22)
        solar_position = ApparentSolarPosition(; epoch)
        average_solve_cos_zenith!(cosines, durations, solar_position, grid, (; time=FT(0)), 8hours)
        actual = reshape(Array(cosines), 4, 3)
        for (i, longitude) in enumerate((-135, -45, 45, 135)), (j, latitude) in enumerate((-50, 0, 50))
            expected = sampled_daylight_mean(epoch, longitude, latitude, Int(8hours))
            @test actual[i, j] ≈ expected rtol = 1e-3 atol = 1e-6
        end
    end
end
