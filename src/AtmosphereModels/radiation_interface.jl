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

Return a RadiativeTransferModel on `grid` with thermodynamic `constants` and using
the `optics` configuration for radiative transfer.
"""
function RadiativeTransferModel(grid, constants, optics; kw...)
    msg = "At the moment, RadiativeTransferModel requires RRTMGP and is only valid with
          optics = GrayOpticalThicknessOGorman2008(FT). Received $optics."
    throw(ArgumentError(msg))
    return nothing
end

"""
$(TYPEDSIGNATURES)

Configuration for RRTMGP full-spectrum **gas optics** (clear-sky).

This object is intentionally defined in Breeze (so users can configure it without importing
RRTMGP internals), but it is **only usable** when the RRTMGP extension is active.

# Keyword Arguments
All keywords correspond to volume mixing ratios (VMR) and are dimensionless.
Defaults are reasonable modern values for the major constituents and `0` for trace halocarbons.

Notes:
- H₂O and O₃ are treated as prognostic / profile fields in the solver setup (H₂O from Breeze
  moisture; O₃ defaults to a constant here but can be upgraded later).
"""
struct RRTMGPGasOptics{GV}
    gas_vmr :: GV
end

@inline function default_rrtmgp_gas_vmr(::Type{FT}) where {FT}
    return (;
        # Major/background gases
        n2  = FT(0.78084),
        o2  = FT(0.20946),
        co2 = FT(420e-6),
        ch4 = FT(1.8e-6),
        n2o = FT(0.33e-6),
        co  = FT(0),
        no2 = FT(0),
        o3  = FT(0),

        # Halocarbons / trace gases (default off)
        cfc11   = FT(0),
        cfc12   = FT(0),
        cfc22   = FT(0),
        ccl4    = FT(0),
        cf4     = FT(0),
        hfc125  = FT(0),
        hfc134a = FT(0),
        hfc143a = FT(0),
        hfc23   = FT(0),
        hfc32   = FT(0),
    )
end

"""
$(TYPEDSIGNATURES)

Construct `RRTMGPGasOptics` with default gas volume mixing ratios, optionally overridden by keywords.
"""
function RRTMGPGasOptics(::Type{FT}; kwargs...) where {FT}
    defaults = default_rrtmgp_gas_vmr(FT)
    overrides = NamedTuple{keys(kwargs)}(map(x -> convert(FT, x), values(kwargs)))
    return RRTMGPGasOptics(merge(defaults, overrides))
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
