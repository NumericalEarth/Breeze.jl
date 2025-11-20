"""
Integration of RadiativeTransferModel with AtmosphereModel.
"""

using Oceananigans: CenterField
using Oceananigans.Operators: ℑzᵃᵃᶜ

import ..AtmosphereModels: AtmosphereModel, AnelasticFormulation
import ..grid_conversion: reshape_from_columns

"""
    _radiative_heating_rate(i, j, k, grid, rtm::RadiativeTransferModel, 
                          reference_density, thermo)

Compute the radiative heating rate contribution to moist static energy tendency
at grid point (i, j, k).

The heating rate is computed from flux differences:
    hr = g * (flux_net[k+1] - flux_net[k]) / (cp * Δp)

and the contribution to energy density tendency is:
    ρᵣ * hr

where ρᵣ is the reference density.
"""
@inline function _radiative_heating_rate(i, j, k, grid, rtm, reference_density, thermo)
    # Get net fluxes
    flux_net_lw = rtm.flux_lw.flux_net
    flux_net_sw = rtm.flux_sw.flux_net
    
    # Combine longwave and shortwave
    flux_net_total = flux_net_lw .+ flux_net_sw
    
    # Convert grid indices to column index
    nx, ny, nz = size(grid)
    icol = (j - 1) * nx + i
    
    # Get flux at levels k and k+1 (fluxes are at levels, k is cell index)
    # Level k corresponds to bottom of cell k
    # Level k+1 corresponds to top of cell k
    flux_bottom = @inbounds flux_net_total[k, icol]
    flux_top = @inbounds flux_net_total[k+1, icol]
    
    # Get pressure difference from atmospheric state
    p_lev = rtm.atmospheric_state.p_lev
    Δp = @inbounds p_lev[k+1, icol] - p_lev[k, icol]
    
    # Get constants from thermodynamics
    g = thermo.gravitational_acceleration
    cp = thermo.dry_air.heat_capacity
    
    # Compute heating rate per unit mass
    hr = g * (flux_top - flux_bottom) / (cp * Δp)
    
    # Get reference density at this point
    ρᵣ = @inbounds reference_density[i, j, k]
    
    # Return contribution to energy density tendency
    return ρᵣ * hr
end

