using Breeze.CelestialMechanics: day_of_year, solar_declination, hour_angle
using Dates: Date, DateTime, Day, Millisecond

# Integrate a + b cos(ω) only while the sun is above the horizon. Clipping to sunrise and sunset
# weights a partially sunlit interval by its actual duration, even when daylight is very brief.
@inline function sunlit_segment(cosine_offset, cosine_amplitude, first_angle, last_angle, horizon_angle)
    first_sunlit_angle = clamp(first_angle, -horizon_angle, horizon_angle)
    last_sunlit_angle = clamp(last_angle, -horizon_angle, horizon_angle)
    sunlit_angle = max(0, last_sunlit_angle - first_sunlit_angle)
    midpoint = (first_sunlit_angle + last_sunlit_angle) / 2
    integral = cosine_offset * sunlit_angle + 2cosine_amplitude * cos(midpoint) * sin(sunlit_angle / 2)
    return max(0, integral), sunlit_angle
end

@inline function sunlit_cosine_integral(cosine_offset, cosine_amplitude, first_angle, angle_span)
    half_cycle = oftype(first_angle, π)
    cycle = 2half_cycle
    safe_amplitude = ifelse(cosine_amplitude > 0, cosine_amplitude, one(cosine_amplitude))
    horizon_angle = acos(clamp(-cosine_offset / safe_amplitude, -1, 1))
    horizon_angle = ifelse(cosine_amplitude > 0, horizon_angle,
                          ifelse(cosine_offset > 0, half_cycle, zero(half_cycle)))

    first_angle = mod(first_angle + half_cycle, cycle) - half_cycle
    last_angle = first_angle + angle_span
    midnights = floor((last_angle + half_cycle) / cycle)
    last_angle -= midnights * cycle

    same_day_integral, same_day_duration = sunlit_segment(cosine_offset, cosine_amplitude, first_angle, last_angle, horizon_angle)
    first_integral, first_duration = sunlit_segment(cosine_offset, cosine_amplitude, first_angle, half_cycle, horizon_angle)
    last_integral, last_duration = sunlit_segment(cosine_offset, cosine_amplitude, -half_cycle, last_angle, horizon_angle)
    day_integral, day_duration = sunlit_segment(cosine_offset, cosine_amplitude, -half_cycle, half_cycle, horizon_angle)
    whole_days = max(0, midnights - 1)

    integral = ifelse(midnights == 0, same_day_integral, first_integral + whole_days * day_integral + last_integral)
    duration = ifelse(midnights == 0, same_day_duration, first_duration + whole_days * day_duration + last_duration)
    return integral, duration
end

# Each method averages over the `interval` seconds that begin `lead` seconds before `clock`.
# Fixed and user-defined solar positions retain the instantaneous angle unless they provide an average.
average_solve_cos_zenith!(cos_zenith, sunlit_duration, solar_position, grid, clock, interval, lead=0) = nothing

function average_solve_cos_zenith!(cos_zenith, sunlit_duration, solar_position::DiurnalSolarPosition, grid, clock, interval, lead=0)
    angular_speed = 2π / solar_position.day_length
    first_angle = angular_speed * (clock.time - lead - solar_position.noon_offset)
    cosine_offset = sind(solar_position.latitude) * sind(solar_position.declination)
    cosine_amplitude = cosd(solar_position.latitude) * cosd(solar_position.declination)
    integral, duration = sunlit_cosine_integral(cosine_offset, cosine_amplitude, first_angle, angular_speed * interval)
    mean_cosine = integral / ifelse(duration > 0, duration, one(duration))
    fill!(cos_zenith, convert(eltype(grid), mean_cosine))
    fill!(sunlit_duration, convert(eltype(grid), duration / angular_speed))
    return nothing
end

function average_solve_cos_zenith!(cos_zenith, sunlit_duration, solar_position::ApparentSolarPosition, grid, clock, interval, lead=0)
    datetime = compute_datetime(clock.time, solar_position.epoch) - Millisecond(round(Int, 1000lead))
    remaining = interval
    fill!(cos_zenith, 0)
    fill!(sunlit_duration, 0)

    # The apparent-sun model changes declination and equation of time at UTC midnight. Integrate
    # each calendar day separately, including intervals crossing the date or year boundary.
    while remaining > 0
        midnight = DateTime(Date(datetime) + Day(1))
        duration = min(remaining, time_difference_seconds(midnight, datetime))
        declination = solar_declination(day_of_year(datetime))
        first_angle = hour_angle(datetime, 0)
        FT = eltype(grid)
        coordinate = convert_solar_coordinate(FT, solar_position.coordinate)
        launch!(architecture(grid), grid, :xy, _integrate_apparent_zenith!,
                cos_zenith, sunlit_duration, grid, coordinate, FT(declination), FT(first_angle), FT(duration))
        remaining -= duration
        datetime = midnight
    end

    cos_zenith ./= ifelse.(sunlit_duration .> 0, sunlit_duration, 1)
    return nothing
end

convert_solar_coordinate(FT, ::Nothing) = nothing
convert_solar_coordinate(FT, coordinate::Tuple) = FT.(coordinate)

@inline solar_coordinate(i, j, grid, coordinate::Tuple) = coordinate
@inline solar_coordinate(i, j, grid, ::Nothing) = (λnode(i, j, 1, grid, Center(), Center(), Center()),
                                                  φnode(i, j, 1, grid, Center(), Center(), Center()))

@kernel function _integrate_apparent_zenith!(cos_zenith, sunlit_duration, grid, coordinate, declination, first_angle, duration)
    i, j = @index(Global, NTuple)
    longitude, latitude = solar_coordinate(i, j, grid, coordinate)
    cosine_offset = sind(latitude) * sin(declination)
    cosine_amplitude = cosd(latitude) * cos(declination)
    angular_speed = 2oftype(first_angle, π) / 86400
    integral, sunlit_angle = sunlit_cosine_integral(cosine_offset, cosine_amplitude,
                                                  first_angle + deg2rad(longitude), angular_speed * duration)
    c = column_index(i, j, grid.Nx)
    @inbounds begin
        cos_zenith[c] += integral / angular_speed
        sunlit_duration[c] += sunlit_angle / angular_speed
    end
end
