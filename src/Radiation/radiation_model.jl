abstract type AbstractRadiationModel end

"""
    update_radative_fluxes!(model)

Update radiative fluxes for the given `GrayRadiationModel` by running the
longwave and shortwave two-stream solvers with the current atmospheric state
and boundary conditions.
"""
function update_radative_fluxes!(::AbstractRadiationModel)
    throw("Not implemented")
end
