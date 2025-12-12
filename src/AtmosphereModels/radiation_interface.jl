#####
##### Radiation interface for AtmosphereModel
#####
##### This file defines stub functions that are implemented by radiation extensions
##### (e.g., BreezeRRTMGPExt).
#####

"""
    $(TYPEDSIGNATURES)

Update the radiative fluxes from the current model state.

This is a stub function that does nothing by default. It is extended by
radiation extensions (e.g., BreezeRRTMGPExt) to compute radiative transfer.
"""
update_radiation!(radiation, model) = nothing

struct GrayRadiativeTransferModel{FT, C, E, SP, OT, AS, LW, SW, F}
    solar_constant :: FT # Scalar
    coordinate :: C # coordinates (for RectilinearGrid) for computing the solar zenith angle
    epoch :: E # optional epoch for computing time with floating-point clocks
    surface_properties :: SP
    optical_thickness :: OT
    atmospheric_state :: AS
    longwave_solver :: LW
    shortwave_solver :: SW
    upwelling_longwave_flux :: F
    downwelling_longwave_flux :: F
    downwelling_shortwave_flux :: F
end

struct SurfaceRadiativeProperties{ST, SE, SA, DW}
    surface_temperature :: ST  # Scalar or 2D field
    surface_emissivity :: SE   # Scalar
    direct_surface_albedo :: SA  # Scalar or 2D field
    diffuse_surface_albedo :: DW  # Scalar or 2D field
end

Base.summary(radiation::GrayRadiativeTransferModel) = "GrayRadiativeTransferModel"

function Base.show(io::IO, radiation::GrayRadiativeTransferModel)
    print(io, summary(radiation), "\n",
          "├── solar_constant: ", prettysummary(radiation.solar_constant))

          # "├── surface_temperature: ", radiation.surface_properties.surface_temperature, " K\n",
          # "├── surface_emissivity: ", radiation.surface_emissivity, "\n",
          # "└── surface_albedo: ", radiation.surface_properties.surface_albedo, "\n",
end
