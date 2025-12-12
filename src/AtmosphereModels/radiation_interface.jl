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

"""
    GrayRadiativeTransferModel

Gray atmosphere radiative transfer model using RRTMGP.jl.

Uses the O'Gorman and Schneider (2008) optical thickness parameterization
for both longwave and shortwave radiation.

# Fields

- `longwave_solver`: RRTMGP longwave RTE solver
- `shortwave_solver`: RRTMGP shortwave RTE solver (no scattering)
- `atmospheric_state`: RRTMGP atmospheric state arrays
- `optical_thickness`: Gray optical thickness parameterization
- `latitude`: Latitude for solar zenith angle calculation (degrees)
- `upwelling_longwave_flux`: ZFaceField for upwelling longwave flux (W/m²)
- `downwelling_longwave_flux`: ZFaceField for downwelling longwave flux (W/m²)
- `downwelling_shortwave_flux`: ZFaceField for direct beam shortwave flux (W/m²)
- `surface_temperature`: Surface temperature (K), scalar or 2D field
- `surface_emissivity`: Surface emissivity (0-1)
- `surface_albedo`: Surface albedo (0-1), scalar or 2D field
- `solar_constant`: Top-of-atmosphere solar flux (W/m²)

!!! note "RRTMGP Extension"
    This type requires the RRTMGP.jl package to be loaded. The constructor
    is provided by the `BreezeRRTMGPExt` extension.

# Example

```julia
using RRTMGP  # Load to enable GrayRadiativeTransferModel constructor

radiation = GrayRadiativeTransferModel(grid, constants;
                                       surface_temperature = 300,
                                       surface_emissivity = 0.98,
                                       surface_albedo = 0.1,
                                       solar_constant = 1361)
```

See also: [`update_radiation!`](@ref)
"""
struct GrayRadiativeTransferModel{LW, SW, AS, OT, LA, ST, SE, SA, SC, F}
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

Base.summary(radiation::GrayRadiativeTransferModel) = "GrayRadiativeTransferModel"

function Base.show(io::IO, radiation::GrayRadiativeTransferModel)
    print(io, summary(radiation), "\n",
          "├── surface_temperature: ", radiation.surface_temperature, " K\n",
          "├── surface_emissivity: ", radiation.surface_emissivity, "\n",
          "├── surface_albedo: ", radiation.surface_albedo, "\n",
          "└── solar_constant: ", radiation.solar_constant, " W/m²")
end
