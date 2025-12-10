#####
##### Solar zenith angle calculation
#####
##### Computes the cosine of the solar zenith angle from:
##### - DateTime (from model clock)
##### - Latitude (from grid y-coordinate)
##### - Longitude (from grid x-coordinate)
#####

"""
    day_of_year(dt::DateTime)

Return the day of year (1-365/366) for a given DateTime.
"""
day_of_year(dt::DateTime) = Dates.dayofyear(dt)

"""
    solar_declination(day_of_year)

Compute the solar declination angle (in radians) for a given day of year.
Uses the approximation from Spencer (1971).
"""
function solar_declination(day_of_year)
    # Fractional year in radians
    γ = 2π * (day_of_year - 1) / 365

    # Solar declination (radians) - Spencer (1971) approximation
    δ = 0.006918 - 0.399912 * cos(γ) + 0.070257 * sin(γ) -
        0.006758 * cos(2γ) + 0.000907 * sin(2γ) -
        0.002697 * cos(3γ) + 0.00148 * sin(3γ)

    return δ
end

"""
    equation_of_time(day_of_year)

Compute the equation of time (in minutes) for a given day of year.
This accounts for the difference between mean solar time and apparent solar time.
"""
function equation_of_time(day_of_year)
    # Fractional year in radians
    γ = 2π * (day_of_year - 1) / 365

    # Equation of time in minutes - Spencer (1971)
    eot = 229.18 * (0.000075 + 0.001868 * cos(γ) - 0.032077 * sin(γ) -
                    0.014615 * cos(2γ) - 0.040849 * sin(2γ))

    return eot
end

"""
    hour_angle(datetime::DateTime, longitude)

Compute the hour angle (in radians) for a given datetime and longitude.
The hour angle is zero at solar noon and increases by 15° per hour.
"""
function hour_angle(datetime::DateTime, longitude)
    # Get UTC hour as a decimal
    hour_utc = Dates.hour(datetime) + Dates.minute(datetime) / 60 + Dates.second(datetime) / 3600

    # Day of year for equation of time
    doy = day_of_year(datetime)
    eot = equation_of_time(doy)

    # Time offset due to longitude (in hours, 15° per hour)
    time_offset = longitude / 15

    # True solar time (in hours)
    solar_time = hour_utc + time_offset + eot / 60

    # Hour angle: 0 at solar noon, increases by 15° per hour
    # Convert to radians
    ω = deg2rad(15 * (solar_time - 12))

    return ω
end

"""
    cos_solar_zenith_angle(datetime::DateTime, latitude, longitude)

Compute the cosine of the solar zenith angle for a given datetime and location.

The solar zenith angle θ_z satisfies:
    cos(θ_z) = sin(φ) sin(δ) + cos(φ) cos(δ) cos(ω)

where:
- φ is the latitude
- δ is the solar declination
- ω is the hour angle

Returns a value between -1 and 1. Negative values indicate the sun is below the horizon.
"""
function cos_solar_zenith_angle(datetime::DateTime, latitude, longitude)
    φ = deg2rad(latitude)
    doy = day_of_year(datetime)
    δ = solar_declination(doy)
    ω = hour_angle(datetime, longitude)

    cos_θz = sin(φ) * sin(δ) + cos(φ) * cos(δ) * cos(ω)

    return cos_θz
end

"""
    cos_solar_zenith_angle(grid, datetime::DateTime)

Compute the cosine of the solar zenith angle for the grid's location.
For single-column grids, extracts latitude from y-coordinate and longitude from x-coordinate.
"""
function cos_solar_zenith_angle(grid, datetime::DateTime)
    # For a single-column grid with Flat horizontal topology,
    # the x and y coordinates represent longitude and latitude
    TX, TY, TZ = topology(grid)

    if TX == Flat && TY == Flat
        # Single column: x is longitude, y is latitude
        λ = grid.xᶜᵃᵃ[1]  # longitude
        φ = grid.yᵃᶜᵃ[1]  # latitude
        return cos_solar_zenith_angle(datetime, φ, λ)
    else
        error("cos_solar_zenith_angle for multi-column grids not yet implemented")
    end
end

