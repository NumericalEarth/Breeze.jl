module CelestialMechanics

export cos_solar_zenith_angle,
       solar_declination,
       equation_of_time,
       hour_angle,
       day_of_year

using Dates: DateTime, Dates

using Oceananigans.Grids: RectilinearGrid, Flat, Bounded, Center, xnode, ynode

include("solar_zenith_angle.jl")

end # module

