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

struct RadiativeTransferModel{OT, FT, C, E, SP, AS, LW, SW, F}
    optics :: OT
    solar_constant :: FT # Scalar
    coordinate :: C # coordinates (for RectilinearGrid) for computing the solar zenith angle
    epoch :: E # optional epoch for computing time with floating-point clocks
    surface_properties :: SP
    atmospheric_state :: AS
    longwave_solver :: LW
    shortwave_solver :: SW
    upwelling_longwave_flux :: F
    downwelling_longwave_flux :: F
    downwelling_shortwave_flux :: F
end

"""
$(TYPEDSIGNATURES)

Return a RadiativeTransferModel on `grid` using the `optics` configuration for radiative transfer.
The `parameters` argument provides physical constants for the radiative transfer solver.
"""
function RadiativeTransferModel(grid, optics, parameters=nothing; kw...)
    msg = "RadiativeTransferModel requires RRTMGP. Valid optics are:\n" *
          "  - GrayOpticalThicknessOGorman2008(FT)\n" *
          "  - RRTMGPGasOptics()\n" *
          "Received: $optics"
    throw(ArgumentError(msg))
    return nothing
end

"""
    BackgroundAtmosphericComposition

Constant (spatially uniform) volume mixing ratios (VMR) for radiatively active gases.
All values are dimensionless molar fractions.

# Fields
- Major atmospheric constituents: `N₂`, `O₂`, `CO₂`, `CH₄`, `N₂O`, `CO`, `NO₂`, `O₃`
- Halocarbons: `CFC₁₁`, `CFC₁₂`, `CFC₂₂`, `CCl₄`, `CF₄`
- Hydrofluorocarbons: `HFC₁₂₅`, `HFC₁₃₄ₐ`, `HFC₁₄₃ₐ`, `HFC₂₃`, `HFC₃₂`

Defaults are approximate modern atmospheric values for major gases; halocarbons default to zero.

Note: H₂O is computed from the model's prognostic moisture field, not specified here.
"""
Base.@kwdef struct BackgroundAtmosphericComposition{FT}
    # Major atmospheric constituents
    N₂  :: FT = 0.78084      # Nitrogen (~78%)
    O₂  :: FT = 0.20946      # Oxygen (~21%)
    CO₂ :: FT = 420e-6       # Carbon dioxide (~420 ppm)
    CH₄ :: FT = 1.8e-6       # Methane (~1.8 ppm)
    N₂O :: FT = 0.33e-6      # Nitrous oxide (~330 ppb)
    CO  :: FT = 0.0          # Carbon monoxide
    NO₂ :: FT = 0.0          # Nitrogen dioxide
    O₃  :: FT = 0.0          # Ozone (often specified as a profile)

    # Chlorofluorocarbons (CFCs)
    CFC₁₁ :: FT = 0.0        # Trichlorofluoromethane
    CFC₁₂ :: FT = 0.0        # Dichlorodifluoromethane
    CFC₂₂ :: FT = 0.0        # Chlorodifluoromethane

    # Other halocarbons
    CCl₄ :: FT = 0.0         # Carbon tetrachloride
    CF₄  :: FT = 0.0         # Carbon tetrafluoride

    # Hydrofluorocarbons (HFCs)
    HFC₁₂₅  :: FT = 0.0      # Pentafluoroethane
    HFC₁₃₄ₐ :: FT = 0.0      # 1,1,1,2-Tetrafluoroethane
    HFC₁₄₃ₐ :: FT = 0.0      # 1,1,1-Trifluoroethane
    HFC₂₃   :: FT = 0.0      # Trifluoromethane
    HFC₃₂   :: FT = 0.0      # Difluoromethane
end

"""
$(TYPEDSIGNATURES)

Configuration for RRTMGP full-spectrum **gas optics** (clear-sky).

This object is intentionally defined in Breeze (so users can configure it without importing
RRTMGP internals), but it is **only usable** when the RRTMGP extension is active.

The `background_composition` field specifies the gas volume mixing ratios for radiatively
active species (except H₂O, which is computed from the model's moisture field).
"""
struct RRTMGPGasOptics{BA}
    background_composition :: BA
end

RRTMGPGasOptics() = RRTMGPGasOptics(BackgroundAtmosphericComposition{Float64}())

struct SurfaceRadiativeProperties{ST, SE, SA, DW}
    surface_temperature :: ST  # Scalar or 2D field
    surface_emissivity :: SE   # Scalar
    direct_surface_albedo :: SA  # Scalar or 2D field
    diffuse_surface_albedo :: DW  # Scalar or 2D field
end

Base.summary(radiation::RadiativeTransferModel) = "RadiativeTransferModel"

function Base.show(io::IO, radiation::RadiativeTransferModel)
    print(io, summary(radiation), "\n",
          "├── solar_constant: ", prettysummary(radiation.solar_constant), " W m⁻²\n",
          "├── surface_temperature: ", radiation.surface_properties.surface_temperature, " K\n",
          "├── surface_emissivity: ", radiation.surface_properties.surface_emissivity, "\n",
          "├── direct_surface_albedo: ", radiation.surface_properties.direct_surface_albedo, "\n",
          "└── diffuse_surface_albedo: ", radiation.surface_properties.diffuse_surface_albedo)
end
