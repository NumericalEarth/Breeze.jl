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

struct GrayRadiation{LW, SW, AS, OT, LA, ST, SE, SA, SC, F}
    longwave_solver :: LW
    shortwave_solver :: SW
    atmospheric_state :: AS
    optical_thickness :: OT
    latitude :: LA
    upwelling_longwave_flux :: F
    downwelling_longwave_flux :: F
    downwelling_shortwave_flux :: F  # Direct beam only for no-scattering solver
    surface_temperature :: ST  # Scalar or 2D field
    surface_emissivity :: SE   # Scalar
    surface_albedo :: SA       # Scalar or 2D field
    solar_constant :: SC       # Scalar
end