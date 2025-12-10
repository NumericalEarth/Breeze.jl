#####
##### Radiation interface for AtmosphereModel
#####
##### This file defines stub functions that are implemented by radiation extensions
##### (e.g., BreezeRRTMGPExt).
#####

"""
    update_radiation!(radiation, model)

Update the radiative fluxes from the current model state.

This is a stub function that does nothing by default. It is extended by
radiation extensions (e.g., BreezeRRTMGPExt) to compute radiative transfer.
"""
update_radiation!(radiation, model) = nothing

