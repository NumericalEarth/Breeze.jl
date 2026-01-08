#####
##### Radiation interface for AtmosphereModel
#####
##### This file defines stub functions that are implemented by radiation extensions
##### (e.g., BreezeRRTMGPExt).
#####

using Oceananigans.Grids: AbstractGrid
using InteractiveUtils: subtypes

"""
$(TYPEDSIGNATURES)

Update the radiative fluxes from the current model state.

This is a stub function that does nothing by default. It is extended by
radiation extensions (e.g., BreezeRRTMGPExt) to compute radiative transfer.
"""
update_radiation!(radiation, model) = nothing

struct RadiativeTransferModel{FT<:Number, C, E, SP, BA, AS, LW, SW, F, LER, IER}
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
    liquid_effective_radius :: LER # Model for cloud liquid effective radius (Nothing for gray/clear-sky)
    ice_effective_radius :: IER    # Model for cloud ice effective radius (Nothing for gray/clear-sky)
end

"""
$(TYPEDEF)

Abstract type representing optics for [`RadiativeTransferModel`](@ref).
"""
abstract type AbstractOptics end
"""
$(TYPEDEF)

Type representing gray atmosphere radiation ([O'Gorman & Schneider 2008](@cite OGormanSchneider2008)), can be used as optics argument in [`RadiativeTransferModel`](@ref).
"""
struct GrayOptics <: AbstractOptics end
"""
$(TYPEDEF)

Type representing full-spectrum clear-sky radiation using RRTMGP gas optics, can be used as optics argument in [`RadiativeTransferModel`](@ref).
"""
struct ClearSkyOptics <: AbstractOptics end

"""
$(TYPEDEF)

Type representing full-spectrum all-sky (cloudy) radiation using RRTMGP gas and cloud optics,
can be used as optics argument in [`RadiativeTransferModel`](@ref).

All-sky radiation includes scattering by cloud liquid and ice particles, requiring
cloud water path, cloud fraction, and effective radius inputs from the microphysics scheme.
"""
struct AllSkyOptics <: AbstractOptics end

"""
$(TYPEDSIGNATURES)

Construct a `RadiativeTransferModel` on `grid` using the specified `optics`.

Valid optics types are:
- [`GrayOptics()`](@ref) - Gray atmosphere radiation ([O'Gorman & Schneider 2008](@cite OGormanSchneider2008))
- [`ClearSkyOptics()`](@ref) - Full-spectrum clear-sky radiation using RRTMGP gas optics
- [`AllSkyOptics()`](@ref) - Full-spectrum all-sky (cloudy) radiation using RRTMGP gas and cloud optics

The `constants` argument provides physical constants for the radiative transfer solver.

# Example

```jldoctest
julia> using Breeze, Oceananigans.Units, RRTMGP, NCDatasets

julia> grid = RectilinearGrid(; size=16, x=0, y=45, z=(0, 10kilometers),
                              topology=(Flat, Flat, Bounded));

julia> RadiativeTransferModel(grid, GrayOptics(), ThermodynamicConstants();
                              surface_temperature = 300,
                              surface_albedo = 0.1)
RadiativeTransferModel
├── solar_constant: 1361.0 W m⁻²
├── surface_temperature: ConstantField(300.0) K
├── surface_emissivity: ConstantField(0.98)
├── direct_surface_albedo: ConstantField(0.1)
└── diffuse_surface_albedo: ConstantField(0.1)

julia> RadiativeTransferModel(grid, ClearSkyOptics(), ThermodynamicConstants();
                              surface_temperature = 300,
                              surface_albedo = 0.1,
                              background_atmosphere = BackgroundAtmosphere(CO₂ = 400e-6))
RadiativeTransferModel
├── solar_constant: 1361.0 W m⁻²
├── surface_temperature: ConstantField(300.0) K
├── surface_emissivity: ConstantField(0.98)
├── direct_surface_albedo: ConstantField(0.1)
└── diffuse_surface_albedo: ConstantField(0.1)
```
"""
function RadiativeTransferModel(grid::AbstractGrid, optics, args...; kw...)
    msg = "Unknown optics $(optics). Valid options are $(join(string.(subtypes(AbstractOptics)) .* "()", ", ")).\n" *
          "Make sure RRTMGP.jl is loaded (e.g., `using RRTMGP`)."
    return throw(ArgumentError(msg))
end

"""
$(TYPEDEF)

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
    N₂O :: FT = 330e-9       # Nitrous oxide (~330 ppb)
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
          "├── direct_surface_albedo: ", radiation.surface_properties.direct_surface_albedo, "\n")

    # Show effective radius models if present (for all-sky optics)
    if !isnothing(radiation.liquid_effective_radius)
        print(io, "├── liquid_effective_radius: ", radiation.liquid_effective_radius, "\n",
                  "├── ice_effective_radius: ", radiation.ice_effective_radius, "\n")
    end

    print(io, "└── diffuse_surface_albedo: ", radiation.surface_properties.diffuse_surface_albedo)
end
