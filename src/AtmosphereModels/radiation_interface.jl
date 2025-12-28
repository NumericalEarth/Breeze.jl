#####
##### Radiation interface for AtmosphereModel
#####
##### This file defines stub functions that are implemented by radiation extensions
##### (e.g., BreezeRRTMGPExt).
#####

using Oceananigans.Grids: AbstractGrid

"""
$(TYPEDSIGNATURES)

Update the radiative fluxes from the current model state.

This is a stub function that does nothing by default. It is extended by
radiation extensions (e.g., BreezeRRTMGPExt) to compute radiative transfer.
"""
update_radiation!(radiation, model) = nothing

struct RadiativeTransferModel{FT<:Number, C, E, SP, BA, AS, LW, SW, F}
    solar_constant :: FT # Scalar
    coordinate :: C # coordinates (for RectilinearGrid) for computing the solar zenith angle
    epoch :: E # optional epoch for computing time with floating-point clocks
    surface_properties :: SP
    background_atmosphere :: BA # BackgroundAtmosphere or Nothing (for gray)
    atmospheric_state :: AS
    longwave_solver :: LW
    shortwave_solver :: SW
    upwelling_longwave_flux :: F
    downwelling_longwave_flux :: F
    downwelling_shortwave_flux :: F
end

"""
$(TYPEDSIGNATURES)

Construct a `RadiativeTransferModel` on `grid` using the specified `optics_flavor`.

Valid optics flavors are:
- `:gray` - Gray atmosphere radiation (O'Gorman & Schneider 2008)
- `:clear_sky` - Full-spectrum clear-sky radiation using RRTMGP gas optics

The `constants` argument provides physical constants for the radiative transfer solver.

# Example

```julia
rtm = RadiativeTransferModel(grid, :gray, constants;
    surface_temperature = 300,
    surface_albedo = 0.1)

rtm = RadiativeTransferModel(grid, :clear_sky, constants;
    surface_temperature = 300,
    surface_albedo = 0.1,
    background_atmosphere = BackgroundAtmosphere(CO₂ = 400e-6))
```
"""
function RadiativeTransferModel(grid, optics_flavor::Symbol, args...; kw...)
    RadiativeTransferModel(grid, Val(optics_flavor), args...; kw...)
end

# Fallback for unknown optics flavors or when extension is not loaded
function RadiativeTransferModel(grid::AbstractGrid, ::Val{S}, args...; kw...) where S
    msg = "Unknown optics flavor :$S. Valid options are :gray, :clear_sky.\n" *
          "Make sure RRTMGP.jl is loaded (e.g., `using RRTMGP`)."
    throw(ArgumentError(msg))
end

"""
    BackgroundAtmosphere

Constant (spatially uniform) volume mixing ratios (VMR) for radiatively active gases.
All values are dimensionless molar fractions.

# Fields
- Major atmospheric constituents: `N₂`, `O₂`, `CO₂`, `CH₄`, `N₂O`, `CO`, `NO₂`, `O₃`
- Halocarbons: `CFC₁₁`, `CFC₁₂`, `CFC₂₂`, `CCl₄`, `CF₄`
- Hydrofluorocarbons: `HFC₁₂₅`, `HFC₁₃₄ₐ`, `HFC₁₄₃ₐ`, `HFC₂₃`, `HFC₃₂`

Defaults are approximate modern atmospheric values for major gases; halocarbons default to zero.

Note: H₂O is computed from the model's prognostic moisture field, not specified here.
"""
Base.@kwdef struct BackgroundAtmosphere{FT}
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
