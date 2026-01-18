#####
##### Radiation interface for AtmosphereModel
#####
##### This file defines stub functions that are implemented by radiation extensions
##### (e.g., BreezeRRTMGPExt).
#####

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: ConstantField
using InteractiveUtils: subtypes

using Oceananigans.Utils: IterationInterval

"""
$(TYPEDSIGNATURES)

Update the radiative fluxes from the current model state.

This function checks the radiation schedule and only updates if the schedule
returns true. The actual radiation computation is dispatched to `_update_radiation!(rtm, model)`.

Radiation is always computed on the first iteration (iteration 0) to ensure
valid radiative fluxes before the first time step.
"""
function update_radiation!(rtm, model)
    isnothing(rtm) && return nothing
    # Always compute on first iteration, then follow schedule
    first_iteration = model.clock.iteration == 0
    if first_iteration || rtm.schedule(model)
        _update_radiation!(rtm, model)
    end
    return nothing
end

# Fallback: no radiation
update_radiation!(::Nothing, model) = nothing

# Internal function that actually computes radiation (implemented by extensions)
_update_radiation!(::Nothing, model) = nothing

struct RadiativeTransferModel{FT<:Number, C, E, SP, BA, AS, LW, SW, F, LER, IER, S}
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
    schedule :: S  # Update schedule (default: IterationInterval(1) = every step)
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

Volume mixing ratios (VMR) for radiatively active gases.
All values are dimensionless molar fractions.

Each gas can be specified as:
- A `Number` (automatically wrapped in `ConstantField`)
- A `ConstantField` for uniform mixing ratios
- A `Field` for spatially varying profiles (e.g., ozone with height dependence)

# Fields
- Major atmospheric constituents: `N₂`, `O₂`, `CO₂`, `CH₄`, `N₂O`, `CO`, `NO₂`, `O₃`
- Halocarbons: `CFC₁₁`, `CFC₁₂`, `CFC₂₂`, `CCl₄`, `CF₄`
- Hydrofluorocarbons: `HFC₁₂₅`, `HFC₁₃₄ₐ`, `HFC₁₄₃ₐ`, `HFC₂₃`, `HFC₃₂`

Defaults are approximate modern atmospheric values for major gases; halocarbons default to zero.

Note: H₂O is computed from the model's prognostic moisture field, not specified here.
"""
struct BackgroundAtmosphere{N2, O2, CO2, CH4, N2O, CO, NO2, O3, CFC11, CFC12, CFC22, CCL4, CF4, HFC125, HFC134A, HFC143A, HFC23, HFC32}
    # Major atmospheric constituents
    N₂  :: N2
    O₂  :: O2
    CO₂ :: CO2
    CH₄ :: CH4
    N₂O :: N2O
    CO  :: CO
    NO₂ :: NO2
    O₃  :: O3

    # Chlorofluorocarbons (CFCs)
    CFC₁₁ :: CFC11
    CFC₁₂ :: CFC12
    CFC₂₂ :: CFC22

    # Other halocarbons
    CCl₄ :: CCL4
    CF₄  :: CF4

    # Hydrofluorocarbons (HFCs)
    HFC₁₂₅  :: HFC125
    HFC₁₃₄ₐ :: HFC134A
    HFC₁₄₃ₐ :: HFC143A
    HFC₂₃   :: HFC23
    HFC₃₂   :: HFC32
end

"""
    BackgroundAtmosphere(FT=Oceananigans.defaults.FloatType; kwargs...)

Construct a `BackgroundAtmosphere` with volume mixing ratios for radiatively active gases.
All values are dimensionless molar fractions.

Each gas can be specified as:
- A `Number` (automatically wrapped in `ConstantField`)
- A `ConstantField` for uniform mixing ratios
- A `Field` for spatially varying profiles (e.g., ozone with height dependence)

Uses `Oceananigans.defaults.FloatType` by default for numeric constants.

# Keyword Arguments
- Major atmospheric constituents: `N₂`, `O₂`, `CO₂`, `CH₄`, `N₂O`, `CO`, `NO₂`, `O₃`
- Halocarbons: `CFC₁₁`, `CFC₁₂`, `CFC₂₂`, `CCl₄`, `CF₄`
- Hydrofluorocarbons: `HFC₁₂₅`, `HFC₁₃₄ₐ`, `HFC₁₄₃ₐ`, `HFC₂₃`, `HFC₃₂`

Defaults are approximate modern atmospheric values for major gases; halocarbons default to zero.
Note: H₂O is computed from the model's prognostic moisture field, not specified here.
"""
function BackgroundAtmosphere(FT::DataType = Oceananigans.defaults.FloatType;
                              N₂  = 0.78084,      # Nitrogen (~78%)
                              O₂  = 0.20946,      # Oxygen (~21%)
                              CO₂ = 420e-6,       # Carbon dioxide (~420 ppm)
                              CH₄ = 1.8e-6,       # Methane (~1.8 ppm)
                              N₂O = 330e-9,       # Nitrous oxide (~330 ppb)
                              CO  = 0.0,          # Carbon monoxide
                              NO₂ = 0.0,          # Nitrogen dioxide
                              O₃  = 0.0,          # Ozone (often specified as a profile)
                              CFC₁₁ = 0.0,        # Trichlorofluoromethane
                              CFC₁₂ = 0.0,        # Dichlorodifluoromethane
                              CFC₂₂ = 0.0,        # Chlorodifluoromethane
                              CCl₄ = 0.0,         # Carbon tetrachloride
                              CF₄  = 0.0,         # Carbon tetrafluoride
                              HFC₁₂₅  = 0.0,      # Pentafluoroethane
                              HFC₁₃₄ₐ = 0.0,      # 1,1,1,2-Tetrafluoromethane
                              HFC₁₄₃ₐ = 0.0,      # 1,1,1-Trifluoroethane
                              HFC₂₃   = 0.0,      # Trifluoromethane
                              HFC₃₂   = 0.0)      # Difluoromethane

    return BackgroundAtmosphere(
        wrap_vmr(FT, N₂),
        wrap_vmr(FT, O₂),
        wrap_vmr(FT, CO₂),
        wrap_vmr(FT, CH₄),
        wrap_vmr(FT, N₂O),
        wrap_vmr(FT, CO),
        wrap_vmr(FT, NO₂),
        wrap_vmr(FT, O₃),
        wrap_vmr(FT, CFC₁₁),
        wrap_vmr(FT, CFC₁₂),
        wrap_vmr(FT, CFC₂₂),
        wrap_vmr(FT, CCl₄),
        wrap_vmr(FT, CF₄),
        wrap_vmr(FT, HFC₁₂₅),
        wrap_vmr(FT, HFC₁₃₄ₐ),
        wrap_vmr(FT, HFC₁₄₃ₐ),
        wrap_vmr(FT, HFC₂₃),
        wrap_vmr(FT, HFC₃₂))
end

# Wrap numbers in ConstantField, pass fields through unchanged
wrap_vmr(FT, x::Number) = ConstantField(convert(FT, x))
wrap_vmr(FT, f) = f  # Already a field

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
