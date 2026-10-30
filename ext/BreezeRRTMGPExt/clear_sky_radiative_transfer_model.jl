#####
##### Clear-sky (gas optics) RadiativeTransferModel: full-spectrum RRTMGP radiative transfer model
#####

using Oceananigans.Grids: AbstractGrid

using Breeze.AtmosphereModels: AtmosphereModels, ClearSkyOptics
using Breeze.Thermodynamics: ThermodynamicConstants

using RRTMGP: ClearSkyRadiation

"""
$(TYPEDSIGNATURES)

Construct a clear-sky (gas-only) full-spectrum `RadiativeTransferModel` for the given grid.

This constructor requires that `NCDatasets` is loadable in the user environment because
RRTMGP loads lookup tables from netCDF via an extension.

# Keyword Arguments
$(full_spectrum_keywords_docstring)
"""
function AtmosphereModels.RadiativeTransferModel(grid::AbstractGrid,
                                                 ::ClearSkyOptics,
                                                 constants::ThermodynamicConstants;
                                                 keywords...)

    # No cloud state and no effective radii
    return full_spectrum_radiative_transfer_model(grid, ClearSkyRadiation(false), constants,
                                                  nothing, nothing, nothing; keywords...)
end

# Clear-sky has no cloud state to update
update_rrtmgp_cloud_state!(::Nothing, model, liquid_effective_radius, ice_effective_radius) = nothing
