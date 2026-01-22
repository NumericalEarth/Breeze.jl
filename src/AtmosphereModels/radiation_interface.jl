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

RRTMGP supports spatially-varying VMR only for H₂O (computed from model moisture)
and O₃. All other gases use global mean values.

# Fields
- **Constant gases** (global mean only): `N₂`, `O₂`, `CO₂`, `CH₄`, `N₂O`, `CO`, `NO₂`
- **Halocarbons**: `CFC₁₁`, `CFC₁₂`, `CFC₂₂`, `CCl₄`, `CF₄`
- **Hydrofluorocarbons**: `HFC₁₂₅`, `HFC₁₃₄ₐ`, `HFC₁₄₃ₐ`, `HFC₂₃`, `HFC₃₂`
- **Spatially-varying**: `O₃` - can be a constant or a function for height-dependent profiles

Defaults are approximate modern atmospheric values for major gases; halocarbons default to zero.

Note: H₂O is computed from the model's prognostic moisture field, not specified here.

The `BackgroundAtmosphere` constructor does not require a grid. When passed to
[`RadiativeTransferModel`](@ref), the O₃ field is materialized using the grid.
This allows users to seamlessly switch between constant and function-based concentrations.
"""
struct BackgroundAtmosphere{N2, O2, CO2, CH4, N2O, CO, NO2, O3, CFC11, CFC12, CFC22, CCL4, CF4, HFC125, HFC134A, HFC143A, HFC23, HFC32}
    # Major atmospheric constituents (constant - RRTMGP only supports global mean)
    N₂  :: N2
    O₂  :: O2
    CO₂ :: CO2
    CH₄ :: CH4
    N₂O :: N2O
    CO  :: CO
    NO₂ :: NO2

    # Ozone - can vary spatially (RRTMGP supports per-layer O₃)
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
    BackgroundAtmosphere(; kwargs...)

Construct a `BackgroundAtmosphere` with volume mixing ratios for radiatively active gases.
All values are dimensionless molar fractions.

RRTMGP supports spatially-varying VMR only for H₂O and O₃. Other gases use global means.

- **Constant gases**: Specify as numbers
- **O₃**: Can be a Number or Function for height-dependent profiles

# Keyword Arguments
- Constant gases: `N₂`, `O₂`, `CO₂`, `CH₄`, `N₂O`, `CO`, `NO₂`
- Halocarbons: `CFC₁₁`, `CFC₁₂`, `CFC₂₂`, `CCl₄`, `CF₄`
- Hydrofluorocarbons: `HFC₁₂₅`, `HFC₁₃₄ₐ`, `HFC₁₄₃ₐ`, `HFC₂₃`, `HFC₃₂`
- Spatially-varying: `O₃` (can be Number or Function)

Defaults are approximate modern atmospheric values; halocarbons default to zero.
Note: H₂O is computed from the model's prognostic moisture field.

# Example

```julia
# Constant ozone
background = BackgroundAtmosphere(CO₂ = 400e-6)

# Height-varying ozone (function of z in meters)
tropical_ozone(z) = 30e-9 * (1 + z / 10000)
background = BackgroundAtmosphere(CO₂ = 400e-6, O₃ = tropical_ozone)
```
"""
function BackgroundAtmosphere(; N₂  = 0.78084,      # Nitrogen (~78%)
                                O₂  = 0.20946,      # Oxygen (~21%)
                                CO₂ = 420e-6,       # Carbon dioxide (~420 ppm)
                                CH₄ = 1.8e-6,       # Methane (~1.8 ppm)
                                N₂O = 330e-9,       # Nitrous oxide (~330 ppb)
                                CO  = 0.0,          # Carbon monoxide
                                NO₂ = 0.0,          # Nitrogen dioxide
                                O₃  = 0.0,          # Ozone (can be profile function)
                                CFC₁₁ = 0.0,        # Trichlorofluoromethane
                                CFC₁₂ = 0.0,        # Dichlorodifluoromethane
                                CFC₂₂ = 0.0,        # Chlorodifluoromethane
                                CCl₄ = 0.0,         # Carbon tetrachloride
                                CF₄  = 0.0,         # Carbon tetrafluoride
                                HFC₁₂₅  = 0.0,      # Pentafluoroethane
                                HFC₁₃₄ₐ = 0.0,      # 1,1,1,2-Tetrafluoroethane
                                HFC₁₄₃ₐ = 0.0,      # 1,1,1-Trifluoroethane
                                HFC₂₃   = 0.0,      # Trifluoromethane
                                HFC₃₂   = 0.0)      # Difluoromethane

    return BackgroundAtmosphere(N₂, O₂, CO₂, CH₄, N₂O, CO, NO₂, O₃,
                                CFC₁₁, CFC₁₂, CFC₂₂, CCl₄, CF₄,
                                HFC₁₂₅, HFC₁₃₄ₐ, HFC₁₄₃ₐ, HFC₂₃, HFC₃₂)
end

using Oceananigans.Fields: field

"""
    materialize_background_atmosphere(atm::BackgroundAtmosphere, grid)

Materialize a `BackgroundAtmosphere` by converting O₃ functions to fields and
converting constant gases to the grid's float type.

This is called internally by [`RadiativeTransferModel`](@ref) constructors.
"""
function materialize_background_atmosphere(atm::BackgroundAtmosphere, grid)
    FT = eltype(grid)

    # O₃ can be Number, Function, or Field - use `field` to wrap appropriately
    # Location (Nothing, Nothing, Center) for z-varying profiles
    O₃_field = field((Nothing, Nothing, Center), atm.O₃, grid)

    return BackgroundAtmosphere(
        convert(FT, atm.N₂),
        convert(FT, atm.O₂),
        convert(FT, atm.CO₂),
        convert(FT, atm.CH₄),
        convert(FT, atm.N₂O),
        convert(FT, atm.CO),
        convert(FT, atm.NO₂),
        O₃_field,
        convert(FT, atm.CFC₁₁),
        convert(FT, atm.CFC₁₂),
        convert(FT, atm.CFC₂₂),
        convert(FT, atm.CCl₄),
        convert(FT, atm.CF₄),
        convert(FT, atm.HFC₁₂₅),
        convert(FT, atm.HFC₁₃₄ₐ),
        convert(FT, atm.HFC₁₄₃ₐ),
        convert(FT, atm.HFC₂₃),
        convert(FT, atm.HFC₃₂))
end

# Materialization is idempotent for already-materialized atmospheres
materialize_background_atmosphere(::Nothing, grid) = nothing

struct SurfaceRadiativeProperties{ST, SE, SA, DW}
    surface_temperature :: ST  # Scalar or 2D field
    surface_emissivity :: SE   # Scalar
    direct_surface_albedo :: SA  # Scalar or 2D field
    diffuse_surface_albedo :: DW  # Scalar or 2D field
end

Base.summary(::RadiativeTransferModel) = "RadiativeTransferModel"

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
