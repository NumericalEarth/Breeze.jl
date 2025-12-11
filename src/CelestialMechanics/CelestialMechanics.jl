module CelestialMechanics

export cos_solar_zenith_angle,
       solar_declination,
       equation_of_time,
       hour_angle,
       day_of_year

using Dates: DateTime, Dates

include("solar_zenith_angle.jl")

end # module

