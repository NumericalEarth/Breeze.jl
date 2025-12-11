#####
##### Grid-specific solar zenith angle wrapper
#####
##### This file provides the grid-specific interface for computing
##### the solar zenith angle from an Oceananigans grid.
##### The core celestial mechanics are in Breeze.CelestialMechanics.
#####

import Breeze.CelestialMechanics: cos_solar_zenith_angle

"""
    cos_solar_zenith_angle(grid, datetime::DateTime)

Compute the cosine of the solar zenith angle for the grid's location.

For single-column grids with `Flat` horizontal topology,
extracts latitude from the y-coordinate and longitude from the x-coordinate.
"""
function cos_solar_zenith_angle(grid::AbstractGrid, datetime::DateTime)
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
