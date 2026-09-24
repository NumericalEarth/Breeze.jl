#####
##### Radiation interface for AtmosphereModel
#####
##### This file defines stub functions that are implemented by radiation extensions
##### (e.g., BreezeRRTMGPExt).
#####

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: ConstantField
using InteractiveUtils: subtypes

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

# Extract the radiation flux divergence field from radiation (nothing-safe)
radiation_flux_divergence(::Nothing) = nothing
radiation_flux_divergence(radiation) = radiation.flux_divergence

# Inline accessor for use inside tendency kernels
@inline radiation_flux_divergence(i, j, k, grid, ::Nothing) = zero(eltype(grid))
@inline radiation_flux_divergence(i, j, k, grid, flux_divergence) = @inbounds flux_divergence[i, j, k]

struct RadiativeTransferModel{FT<:Number, SOP, SP, BA, AS, LW, SW, F, H, LER, IER, S}
    solar_constant :: FT # Scalar
    solar_position :: SOP # AbstractSolarPosition: how to obtain cos(θ_z) on each update
    surface_radiation :: SP
    background_atmosphere :: BA # BackgroundAtmosphere or Nothing (for gray)
    atmospheric_state :: AS
    longwave_solver :: LW
    shortwave_solver :: SW
    upwelling_longwave_flux :: F
    downwelling_longwave_flux :: F
    upwelling_shortwave_flux :: F # Zero for non-scattering gray optics
    downwelling_shortwave_flux :: F
    flux_divergence :: H # Center field: -dF_net/dz in W/m³
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

Type representing gray atmosphere radiation ([O'Gorman & Schneider 2008](@cite OGormanSchneider2008)),
can be used as optics argument in [`RadiativeTransferModel`](@ref). Solved by RRTMGP.jl's gray
solver, which requires `using RRTMGP, ClimaComms, NCDatasets`.

# References

* O'Gorman, P. A. and Schneider, T. (2008). The hydrological cycle over a wide range of climates simulated
    with an idealized GCM. Journal of Climate, 21, 3815–3832.
"""
struct GrayOptics <: AbstractOptics end
"""
$(TYPEDEF)

Type representing full-spectrum clear-sky radiation using RRTMGP gas optics, can be used as optics
argument in [`RadiativeTransferModel`](@ref). Requires `using RRTMGP, ClimaComms, NCDatasets`.
"""
struct ClearSkyOptics <: AbstractOptics end

"""
$(TYPEDEF)

Type representing full-spectrum all-sky (cloudy) radiation using RRTMGP gas and cloud optics,
can be used as optics argument in [`RadiativeTransferModel`](@ref). Requires
`using RRTMGP, ClimaComms, NCDatasets`.

All-sky radiation includes scattering by cloud liquid and ice particles, requiring
cloud water path, cloud fraction, and effective radius inputs from the microphysics scheme.
"""
struct AllSkyOptics <: AbstractOptics end

#####
##### ecCKD optics (NumericalRadiation backend)
#####

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Cloud single-scattering tables for [`EcCKDOptics`](@ref): one table per condensed phase, each
either a `Symbol` naming a table shipped with the ecRad data (`:mie_droplet` for spherical liquid
droplets, `:baum_general_habit_mixture` for the Baum general-habit ice mixture) or a path to a
netCDF file in the same format.

```jldoctest
julia> using Breeze

julia> CloudScatteringTables()
CloudScatteringTables(liquid=:mie_droplet, ice=:baum_general_habit_mixture)
```
"""
struct CloudScatteringTables{L, I}
    "Liquid droplet scattering table: a `Symbol` selector or a file path"
    liquid :: L
    "Ice particle scattering table: a `Symbol` selector or a file path"
    ice :: I
end

CloudScatteringTables(; liquid = :mie_droplet, ice = :baum_general_habit_mixture) = CloudScatteringTables(liquid, ice)

Base.show(io::IO, tables::CloudScatteringTables) =
    print(io, "CloudScatteringTables(liquid=", repr(tables.liquid), ", ice=", repr(tables.ice), ")")

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Full-spectrum radiation with the ecCKD correlated-*k* gas optics of
[Hogan & Matricardi (2022)](@cite HoganMatricardi2022), solved column by column by
NumericalRadiation.jl. Can be used as the optics argument of [`RadiativeTransferModel`](@ref)
once the extension is loaded with `using NumericalRadiation: NumericalRadiation` and
`using NCDatasets` (NumericalRadiation exports its own `ThermodynamicConstants`, so it is
loaded qualified next to `using Breeze`).

The solved column reaches beyond the top of the grid through a [`ColumnExtension`](@ref), so that
the downwelling fluxes at the top of the domain include the emission and absorption of the
atmosphere above it.

```jldoctest
julia> using Breeze

julia> EcCKDOptics()
EcCKDOptics
├── gas_model: :climate_32x32
└── clouds: nothing

julia> EcCKDOptics(:climate_64x64, clouds = CloudScatteringTables())
EcCKDOptics
├── gas_model: :climate_64x64
└── clouds: CloudScatteringTables(liquid=:mie_droplet, ice=:baum_general_habit_mixture)
```
"""
struct EcCKDOptics{M, C} <: AbstractOptics
    """
    Gas optics model: a `Symbol` or `String` selecting a reference ecCKD model shipped with the ecRad
    data (`:climate_32x32`, `:climate_64x64`, ...), a `(longwave = path, shortwave = path)` pair of
    ecCKD definition files, or a preloaded `NumericalRadiation` gas optics model
    """
    gas_model :: M
    "[`CloudScatteringTables`](@ref) for all-sky radiation, or `nothing` for clear sky"
    clouds :: C
end

"""
$(TYPEDSIGNATURES)

Construct [`EcCKDOptics`](@ref) with the reference `gas_model` (default `:climate_32x32`) and,
for all-sky radiation, [`CloudScatteringTables`](@ref) as `clouds` (default `nothing`: clear sky).
"""
EcCKDOptics(gas_model = :climate_32x32; clouds = nothing) = EcCKDOptics(gas_model, clouds)

# Selectors and paths are shown as literals; a preloaded gas optics model by its summary
optics_selector_string(x::Union{Symbol, AbstractString, NamedTuple}) = repr(x)
optics_selector_string(x) = summary(x)

Base.show(io::IO, optics::EcCKDOptics) =
    print(io, "EcCKDOptics", "\n",
          "├── gas_model: ", optics_selector_string(optics.gas_model), "\n",
          "└── clouds: ", optics.clouds)

"""
$(TYPEDSIGNATURES)

Construct a `RadiativeTransferModel` on `grid` using the specified `optics`.

Valid optics types are:
- [`GrayOptics()`](@ref) - Gray atmosphere radiation ([O'Gorman & Schneider 2008](@cite OGormanSchneider2008))
- [`ClearSkyOptics()`](@ref) - Full-spectrum clear-sky radiation using RRTMGP gas optics
- [`AllSkyOptics()`](@ref) - Full-spectrum all-sky (cloudy) radiation using RRTMGP gas and cloud optics
- [`EcCKDOptics()`](@ref) - Full-spectrum clear- or all-sky radiation using ecCKD gas optics via
  NumericalRadiation.jl (requires `using NumericalRadiation: NumericalRadiation` and `using NCDatasets`)

The RRTMGP optics require `using RRTMGP, ClimaComms, NCDatasets`. NumericalRadiation exports its own
`ThermodynamicConstants`, so load it qualified (`using NumericalRadiation: NumericalRadiation`) next to
`using Breeze`.

The `constants` argument provides physical constants for the radiative transfer solver.

# Solar position

The `solar_position` keyword controls how the cosine of the solar zenith angle is
obtained on each radiation update. See [`AbstractSolarPosition`](@ref) and its subtypes:

- [`ApparentSolarPosition`](@ref) (default) — time-varying, computed from the model
  clock and grid (or explicit) longitude/latitude. Supports `DateTime` clocks and
  floating-point clocks resolved against an `epoch`.
- [`FixedCosineZenith`](@ref) — constant cos(θ_z), clock-independent. Appropriate
  for idealized radiative-convective equilibrium studies.

# Example

```jldoctest
julia> using Breeze, Oceananigans.Units, RRTMGP, NCDatasets

julia> using NumericalRadiation: NumericalRadiation

julia> grid = RectilinearGrid(; size=16, x=0, y=45, z=(0, 10kilometers),
                              topology=(Flat, Flat, Bounded));

julia> RadiativeTransferModel(grid, GrayOptics(), ThermodynamicConstants();
                              surface_temperature = 300,
                              surface_albedo = 0.1)
RadiativeTransferModel
├── solar_constant: 1361.0 W m⁻²
├── solar_position: ApparentSolarPosition(coordinate=(0.0, 45.0), epoch=<from clock>)
├── surface_temperature: ConstantField(300.0) K
├── surface_emissivity: ConstantField(0.98)
├── direct_surface_albedo: ConstantField(0.1)
└── diffuse_surface_albedo: ConstantField(0.1)

julia> RadiativeTransferModel(grid, GrayOptics(), ThermodynamicConstants();
                              surface_temperature = 300,
                              surface_albedo = 0.1,
                              solar_position = FixedCosineZenith(0.5))
RadiativeTransferModel
├── solar_constant: 1361.0 W m⁻²
├── solar_position: FixedCosineZenith(cos_zenith = 0.5)
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
├── solar_position: ApparentSolarPosition(coordinate=(0.0, 45.0), epoch=<from clock>)
├── surface_temperature: ConstantField(300.0) K
├── surface_emissivity: ConstantField(0.98)
├── direct_surface_albedo: ConstantField(0.1)
└── diffuse_surface_albedo: ConstantField(0.1)

julia> RadiativeTransferModel(grid, EcCKDOptics(), ThermodynamicConstants();
                              surface_temperature = 300,
                              surface_albedo = 0.1)
RadiativeTransferModel
├── solar_constant: 1361.0 W m⁻²
├── solar_position: ApparentSolarPosition(coordinate=(0.0, 45.0), epoch=<from clock>)
├── surface_temperature: ConstantField(300.0) K
├── surface_emissivity: ConstantField(0.98)
├── direct_surface_albedo: ConstantField(0.1)
├── liquid_effective_radius: ConstantRadiusParticles{Float64}(1.0e-5)
├── ice_effective_radius: ConstantRadiusParticles{Float64}(3.0e-5)
├── diffuse_surface_albedo: ConstantField(0.1)
├── optics: EcCKDOptics with 32 longwave and 32 shortwave g-points, clear sky
└── column_extension: 40 layers from 10000.0 m to 65000.0 m
```

# References

* O'Gorman, P. A. and Schneider, T. (2008). The hydrological cycle over a wide range of climates simulated
    with an idealized GCM. Journal of Climate, 21, 3815–3832.
"""
function RadiativeTransferModel(grid::AbstractGrid, optics, args...; kw...)
    msg = "Unknown optics $(optics). Valid options are $(join(string.(subtypes(AbstractOptics)) .* "()", ", ")).\n" *
          "Make sure RRTMGP.jl is loaded (e.g., `using RRTMGP`)."
    return throw(ArgumentError(msg))
end

# The NumericalRadiation extension replaces this with the constructor proper
function RadiativeTransferModel(grid::AbstractGrid, optics::EcCKDOptics, args...; kw...)
    # NumericalRadiation exports its own `ThermodynamicConstants`, which clashes with Breeze's
    # when both are loaded unqualified, so the hint loads it qualified
    msg = "EcCKDOptics requires the NumericalRadiation extension: load NumericalRadiation and " *
          "NCDatasets with `using NumericalRadiation: NumericalRadiation` and `using NCDatasets`."
    return throw(ArgumentError(msg))
end

"""
    materialize_surface_property(x, grid [, solar_position])

Convert a surface property (albedo, emissivity) to the form the radiative-transfer
solver stores: a `Number` becomes a grid-eltype scalar and a `Field` passes through.
Extend the three-argument form for property sources that must be resolved against the
grid and the solar `epoch` (e.g. an observed-albedo dataset); it falls back to the
two-argument form.
"""
materialize_surface_property(x, grid, solar_position) = materialize_surface_property(x, grid)
materialize_surface_property(x::Number, grid) = convert(eltype(grid), x)
materialize_surface_property(x::Oceananigans.Field, grid) = x

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
$(TYPEDSIGNATURES)

An idealized climatological ozone volume mixing ratio (mol/mol) as a function of height
`z` (m): a weak tropospheric background increasing toward the tropopause, blended into a
Gaussian stratospheric layer peaking near 25 km. Keeps the stratospheric column near
radiative balance in deep-column simulations — without ozone the upper column is far from
radiative equilibrium and destabilizes when the spectral fluxes recompute. Not a substitute
for an observed or model ozone climatology.
"""
@inline function standard_ozone_profile(z)
    troposphere_O₃  = 3e-8 * (1 + 0.5 * z / 1e3)
    stratosphere_O₃ = 8e-6 * exp(-((z - 25e3) / 5e3)^2)
    χˢᵗ = 1 / (1 + exp(-(z - 15e3) / 2))
    return troposphere_O₃ * (1 - χˢᵗ) + stratosphere_O₃ * χˢᵗ
end

"""
$(TYPEDSIGNATURES)

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

Defaults are approximate modern atmospheric values; halocarbons default to zero, and ozone
defaults to [`standard_ozone_profile`](@ref) (pass `O₃ = 0` for an ozone-free atmosphere).
Note: H₂O is computed from the model's prognostic moisture field.

# Example

```jldoctest
julia> using Breeze

julia> background = BackgroundAtmosphere(CO₂ = 400e-6)
BackgroundAtmosphere with 6 active gases:
  N₂ = 0.78084
  O₂ = 0.20946
  CO₂ = 400.0 ppm
  CH₄ = 1.8 ppm
  N₂O = 330.0 ppb
  O₃ = standard_ozone_profile (generic function with 1 method)

julia> tropical_ozone(z) = 30e-9 * (1 + z / 10000);

julia> background = BackgroundAtmosphere(CO₂ = 400e-6, O₃ = tropical_ozone)
BackgroundAtmosphere with 6 active gases:
  N₂ = 0.78084
  O₂ = 0.20946
  CO₂ = 400.0 ppm
  CH₄ = 1.8 ppm
  N₂O = 330.0 ppb
  O₃ = tropical_ozone (generic function with 1 method)
```
"""
function BackgroundAtmosphere(; N₂  = 0.78084,      # Nitrogen (~78%)
                                O₂  = 0.20946,      # Oxygen (~21%)
                                CO₂ = 420e-6,       # Carbon dioxide (~420 ppm)
                                CH₄ = 1.8e-6,       # Methane (~1.8 ppm)
                                N₂O = 330e-9,       # Nitrous oxide (~330 ppb)
                                CO  = 0.0,          # Carbon monoxide
                                NO₂ = 0.0,          # Nitrogen dioxide
                                O₃  = standard_ozone_profile,   # Ozone (Number or profile function; 0 disables)
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

function _vmr_string(value::Number)
    value == 0 && return nothing
    if value ≥ 0.001
        return string(round(value, sigdigits=5))
    elseif value ≥ 1e-6
        return string(round(value * 1e6, sigdigits=4), " ppm")
    elseif value ≥ 1e-9
        return string(round(value * 1e9, sigdigits=4), " ppb")
    else
        return string(value)
    end
end

_vmr_string(value) = summary(value)

function Base.show(io::IO, atm::BackgroundAtmosphere)
    gases = [:N₂, :O₂, :CO₂, :CH₄, :N₂O, :CO, :NO₂, :O₃,
             :CFC₁₁, :CFC₁₂, :CFC₂₂, :CCl₄, :CF₄,
             :HFC₁₂₅, :HFC₁₃₄ₐ, :HFC₁₄₃ₐ, :HFC₂₃, :HFC₃₂]

    nonzero = Tuple{Symbol, String}[]
    for name in gases
        val = getfield(atm, name)
        s = _vmr_string(val)
        s !== nothing && push!(nonzero, (name, s))
    end

    print(io, "BackgroundAtmosphere with $(length(nonzero)) active gases:")
    for (name, s) in nonzero
        print(io, "\n  ", name, " = ", s)
    end
end

using Oceananigans.Fields: field

"""
$(TYPEDSIGNATURES)

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

struct SurfaceRadiation{ST, SE, SA, DW}
    surface_temperature :: ST  # Scalar or 2D field
    surface_emissivity :: SE   # Scalar or 2D field
    direct_surface_albedo :: SA  # Scalar or 2D field
    diffuse_surface_albedo :: DW  # Scalar or 2D field
end

Base.summary(::RadiativeTransferModel) = "RadiativeTransferModel"

# The lines every backend shares, up to (not including) the final diffuse-albedo line, so a
# backend with more to show (an extension's optics, say) can append its own lines after them.
function show_radiation_summary(io::IO, radiation::RadiativeTransferModel)
    print(io, summary(radiation), "\n",
          "├── solar_constant: ", prettysummary(radiation.solar_constant), " W m⁻²\n",
          "├── solar_position: ", radiation.solar_position, "\n")

    if radiation.surface_radiation.surface_temperature isa ConstantField
        print(io, "├── surface_temperature: ", radiation.surface_radiation.surface_temperature, " K\n",)
    else
        print(io, "├── surface_temperature: ", summary(radiation.surface_radiation.surface_temperature), "\n")
    end

    print(io, "├── surface_emissivity: ", radiation.surface_radiation.surface_emissivity, "\n",
              "├── direct_surface_albedo: ", radiation.surface_radiation.direct_surface_albedo, "\n")

    # Show effective radius models if present (for all-sky optics)
    if !isnothing(radiation.liquid_effective_radius)
        print(io, "├── liquid_effective_radius: ", radiation.liquid_effective_radius, "\n",
                  "├── ice_effective_radius: ", radiation.ice_effective_radius, "\n")
    end

    return nothing
end

function Base.show(io::IO, radiation::RadiativeTransferModel)
    show_radiation_summary(io, radiation)
    print(io, "└── diffuse_surface_albedo: ", radiation.surface_radiation.diffuse_surface_albedo)
end

#####
##### Backend-agnostic column helpers shared by every radiation extension
#####

using Oceananigans.Architectures: architecture
using Oceananigans.Grids: xnode, ynode, znode, Center, Face
using Breeze.CelestialMechanics: SingleColumnGrid

"""
$(TYPEDSIGNATURES)

The column index `c = i + (j - 1) Nx` of horizontal cell `(i, j)`: every column-based radiation
backend stores its per-column arrays in this order.
"""
@inline column_index(i, j, Nx) = i + (j - 1) * Nx

#####
##### Solar position: infer (λ, φ) from a single-column grid when the user gave none
#####

# Single-column grids: infer (λ, φ) from the grid when the user didn't pass one
maybe_infer_solar_position(sp::ApparentSolarPosition{Nothing}, grid::SingleColumnGrid) =
    ApparentSolarPosition(grid_inferred_coordinate(grid), sp.epoch)

function grid_inferred_coordinate(grid::SingleColumnGrid)
    λ = xnode(1, 1, 1, grid, Center(), Center(), Center())
    φ = ynode(1, 1, 1, grid, Center(), Center(), Center())
    return (λ, φ)
end

# Otherwise (3D grid or explicit coordinate): leave as-is — per-column kernels
# will read λ/φ from the grid as needed.
maybe_infer_solar_position(sp::AbstractSolarPosition, grid) = sp

#####
##### Surface properties: validation and materialization
#####

"""
$(TYPEDSIGNATURES)

The scalar behind a surface property that is constant in space and time, or `nothing` when the
property carries no such scalar.

A `ConstantField` is a scalar in a field's clothing — its value cannot change — so it reports the
value it holds. A general `Field` reports `nothing`: it may be rewritten between radiation updates,
so there is no single value to speak of.
"""
surface_fraction_scalar(x::Number) = x
surface_fraction_scalar(x::ConstantField) = surface_fraction_scalar(x.constant)
surface_fraction_scalar(x) = nothing

"""
$(TYPEDSIGNATURES)

Throw an `ArgumentError` for any keyword whose value is a spatially uniform scalar outside ``[0, 1]``.

Emissivity and albedo are fractions, so a scalar outside the unit interval is a user error — an albedo
given in percent, say — worth rejecting at construction rather than carrying into the solver. A
property with no single value (a `Field`, a dataset, `nothing`) passes through, since a check at
construction says nothing about what it holds at the next solve.
"""
function validate_surface_fractions(; kw...)
    for (name, value) in kw
        x = surface_fraction_scalar(value)
        isnothing(x) || 0 <= x <= 1 ||
            throw(ArgumentError("`$name` must lie in [0, 1]; received $x."))
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Wrap a scalar surface property in a `ConstantField` of the working precision, passing anything
already field-valued through unchanged, so that emissivity and both albedos are uniformly
field-valued whether the user supplied a number, a field, or a dataset.
"""
constant_field_property(x::Number, FT) = ConstantField(convert(FT, x))
constant_field_property(x, FT) = x

"""
$(TYPEDSIGNATURES)

Resolve the albedo keywords of a `RadiativeTransferModel` constructor into a
`(direct, diffuse)` pair of materialized surface albedos.

Either `surface_albedo` alone (used for both the direct and the diffuse albedo) or *both*
`direct_surface_albedo` and `diffuse_surface_albedo` must be given; any other combination is
an `ArgumentError`. Each is passed through [`materialize_surface_property`](@ref) with
`grid` and `solar_position`.
"""
function resolve_surface_albedos(surface_albedo, direct_surface_albedo, diffuse_surface_albedo, grid, solar_position)
    error_msg = "Must either provide surface_albedo or *both* of
                 direct_surface_albedo and diffuse_surface_albedo"

    if !isnothing(surface_albedo)
        if !isnothing(direct_surface_albedo) || !isnothing(diffuse_surface_albedo)
            throw(ArgumentError(error_msg))
        end

        surface_albedo = materialize_surface_property(surface_albedo, grid, solar_position)
        return surface_albedo, surface_albedo

    elseif !isnothing(diffuse_surface_albedo) && !isnothing(direct_surface_albedo)
        direct_surface_albedo = materialize_surface_property(direct_surface_albedo, grid, solar_position)
        diffuse_surface_albedo = materialize_surface_property(diffuse_surface_albedo, grid, solar_position)
        return direct_surface_albedo, diffuse_surface_albedo
    end

    throw(ArgumentError(error_msg))
end

# The constructors accept `surface_temperature = nothing` so that a coupled model can bind
# its interface surface temperature after construction; solving without one is an error.
function assert_bound_surface_temperature(rtm)
    isnothing(rtm.surface_radiation.surface_temperature) && throw(ArgumentError(
        "This RadiativeTransferModel has no surface temperature: construct it with " *
        "`surface_temperature = ...`, or bind one before the first radiation update " *
        "(coupled models wire their interface surface temperature automatically)."))
    return nothing
end

#####
##### Boundary-face pressure and temperature
#####
##### Column solvers need pressure and temperature on the bottom face (k = 1) and the top face
##### (k = Nz + 1), which no interior interpolation reaches. Rather than inheriting whatever the
##### halo carries, extrapolate from the adjacent cells: pressure hydrostatically over the half
##### cell, `∂p/∂z = -ρ g`, and temperature linearly through the two nearest cell centers.
#####

"""
$(TYPEDSIGNATURES)

Pressure on the bottom face of column `(i, j)`, extrapolated hydrostatically from the lowest
cell center: `p₁ + ρ₁ g (z₁ᶜ - z₁ᶠ)`.
"""
@inline function bottom_face_pressure(i, j, grid, p, ρ, g)
    zᶜ = znode(i, j, 1, grid, Center(), Center(), Center())
    zᶠ = znode(i, j, 1, grid, Center(), Center(), Face())
    return @inbounds p[i, j, 1] + ρ[i, j, 1] * g * (zᶜ - zᶠ)
end

"""
$(TYPEDSIGNATURES)

Pressure on the top face of column `(i, j)`, extrapolated hydrostatically from the highest
cell center: `p_Nz - ρ_Nz g (z_{Nz+1}ᶠ - z_Nzᶜ)`.
"""
@inline function top_face_pressure(i, j, grid, p, ρ, g)
    Nz = size(grid, 3)
    zᶜ = znode(i, j, Nz, grid, Center(), Center(), Center())
    zᶠ = znode(i, j, Nz+1, grid, Center(), Center(), Face())
    return @inbounds p[i, j, Nz] - ρ[i, j, Nz] * g * (zᶠ - zᶜ)
end

# Linear extrapolation of a cell-centered field from the centers of cells `k₁` and `k₂` to the face at `zᶠ`.
@inline function extrapolate_to_face(i, j, k₁, k₂, zᶠ, grid, T)
    z₁ = znode(i, j, k₁, grid, Center(), Center(), Center())
    z₂ = znode(i, j, k₂, grid, Center(), Center(), Center())
    @inbounds T₁ = T[i, j, k₁]
    @inbounds T₂ = T[i, j, k₂]
    return T₁ + (T₂ - T₁) * (zᶠ - z₁) / (z₂ - z₁)
end

"""
$(TYPEDSIGNATURES)

Temperature on the bottom face of column `(i, j)`, extrapolated linearly from cells 1 and 2.
"""
@inline function bottom_face_temperature(i, j, grid, T)
    zᶠ = znode(i, j, 1, grid, Center(), Center(), Face())
    return extrapolate_to_face(i, j, 1, 2, zᶠ, grid, T)
end

"""
$(TYPEDSIGNATURES)

Temperature on the top face of column `(i, j)`, extrapolated linearly from cells `Nz-1` and `Nz`.
"""
@inline function top_face_temperature(i, j, grid, T)
    Nz = size(grid, 3)
    zᶠ = znode(i, j, Nz+1, grid, Center(), Center(), Face())
    return extrapolate_to_face(i, j, Nz-1, Nz, zᶠ, grid, T)
end

#####
##### Radiation flux divergence from the four flux fields
#####

"""
$(TYPEDSIGNATURES)

Compute `rtm.flux_divergence = -∂F_net/∂z` (W m⁻³) from the four `ZFaceField` fluxes of `rtm`,
with `F_net` the sum of the up- and downwelling longwave and shortwave fluxes, all signed
positive upward (downwelling fluxes are stored negative).
"""
function compute_radiation_flux_divergence!(rtm, grid)
    arch = architecture(grid)
    ℐ_lw_up = rtm.upwelling_longwave_flux
    ℐ_lw_dn = rtm.downwelling_longwave_flux
    ℐ_sw_up = rtm.upwelling_shortwave_flux
    ℐ_sw_dn = rtm.downwelling_shortwave_flux
    flux_div = rtm.flux_divergence
    launch!(arch, grid, :xyz, _compute_radiation_flux_divergence!,
            flux_div, ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, grid)
    return nothing
end

@kernel function _compute_radiation_flux_divergence!(flux_div, ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, grid)
    i, j, k = @index(Global, NTuple)
    # Net flux at faces k and k+1 (positive upward)
    @inbounds begin
        F_k  = ℐ_lw_up[i, j, k]   + ℐ_lw_dn[i, j, k]   + ℐ_sw_up[i, j, k]   + ℐ_sw_dn[i, j, k]
        F_k1 = ℐ_lw_up[i, j, k+1] + ℐ_lw_dn[i, j, k+1] + ℐ_sw_up[i, j, k+1] + ℐ_sw_dn[i, j, k+1]
    end
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    # Flux divergence: -dF/dz (positive when flux convergence warms)
    @inbounds flux_div[i, j, k] = -(F_k1 - F_k) / Δz
end

#####
##### Column extension above the grid top
#####
##### A large-eddy simulation typically spans a few kilometers, but the radiative fluxes at its top
##### depend on the whole atmosphere above: the stratosphere emits longwave radiation downward and
##### ozone absorbs shortwave radiation on the way down. The column extension describes that
##### atmosphere so that a column solver can append it above the grid and solve the whole column.
#####

# The U.S. Standard Atmosphere 1976: base geopotential heights (m) and temperatures (K) of the
# seven layers between 0 and 86 km, with a constant lapse rate inside each layer.
const ISA_1976_BASE_HEIGHTS = (0, 11e3, 20e3, 32e3, 47e3, 51e3, 71e3, 84852)
const ISA_1976_BASE_TEMPERATURES = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946)

"""
$(TYPEDSIGNATURES)

The temperature (K) of the U.S. Standard Atmosphere 1976 ([NOAA, NASA & USAF (1976)](@cite USStandardAtmosphere1976))
at height `z` (m): piecewise linear through the seven layers between 0 and 86 km, with the
troposphere cooling at 6.5 K km⁻¹ from 288.15 K at the surface to 216.65 K at 11 km, an isothermal
lower stratosphere, warming to 270.65 K at the stratopause (47–51 km), and cooling again through
the mesosphere to 186.946 K at 84.852 km. Held constant above that, and `z` is taken as the
geopotential height.

```jldoctest
julia> using Breeze

julia> standard_atmosphere_temperature(0)
288.15

julia> standard_atmosphere_temperature(11e3)
216.65

julia> standard_atmosphere_temperature(50e3)
270.65
```
"""
function standard_atmosphere_temperature(z)
    FT = float(typeof(z))
    zᶜ = clamp(FT(z), FT(first(ISA_1976_BASE_HEIGHTS)), FT(last(ISA_1976_BASE_HEIGHTS)))
    T = FT(last(ISA_1976_BASE_TEMPERATURES))

    for n in 1:length(ISA_1976_BASE_HEIGHTS)-1
        z₁ = FT(ISA_1976_BASE_HEIGHTS[n])
        z₂ = FT(ISA_1976_BASE_HEIGHTS[n+1])
        T₁ = FT(ISA_1976_BASE_TEMPERATURES[n])
        T₂ = FT(ISA_1976_BASE_TEMPERATURES[n+1])
        Tₙ = T₁ + (T₂ - T₁) * (zᶜ - z₁) / (z₂ - z₁)
        T = ifelse(z₁ <= zᶜ < z₂, Tₙ, T)
    end

    return T
end

"""
$(TYPEDSIGNATURES)

An idealized specific humidity (kg kg⁻¹) at height `z` (m): 10 g kg⁻¹ at the surface decaying
with a 2.5 km scale height, floored at 3 mg kg⁻¹ (about 5 ppmv) for the dry stratosphere,

```math
qᵛ(z) = max(10⁻² exp(-z / 2500 m), 3 × 10⁻⁶) .
```

```jldoctest
julia> using Breeze.AtmosphereModels: standard_specific_humidity

julia> standard_specific_humidity(0)
0.01

julia> standard_specific_humidity(50e3)
3.0e-6
```
"""
function standard_specific_humidity(z)
    FT = float(typeof(z))
    return max(FT(1e-2) * exp(-FT(z) / FT(2.5e3)), FT(3e-6))
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

The atmosphere above the top of the grid, as seen by a column radiation solver.

Radiation is solved on the grid's `Nz` layers plus `layers` extension layers stacked from the grid's
top face `z_top` to `top`, geometrically stretched from a first layer as thick as the grid's top
layer: with ratio `r` between successive layers, `Δz₁ (rᴺ - 1) / (r - 1) = top - z_top` (see
[`column_extension_faces`](@ref)). No layers are added when the grid already reaches `top`.

The extension temperature follows `temperature(z)`, anchored to the temperature `T_top` on the grid's
top face so that the profile is continuous there and relaxes to `temperature(z)` over `blending_height`,

```math
Tₑ(z) = temperature(z) + (T_top - temperature(z_top)) exp(-(z - z_top) / blending_height) ,
```

with `blending_height = 0` disabling the anchor. Pressure is hydrostatic above the top face, the
specific humidity follows `specific_humidity(z)`, and the ozone mole fraction follows
`ozone_mole_fraction(z)` (a function or a number), or the `O₃` of the [`BackgroundAtmosphere`](@ref)
when `ozone_mole_fraction` is `nothing`. Extension layers are clear.

```jldoctest
julia> using Breeze

julia> ColumnExtension()
ColumnExtension{Float64}
├── top: 65000.0 m
├── layers: 40
├── blending_height: 1000.0 m
├── temperature: standard_atmosphere_temperature (generic function with 1 method)
├── specific_humidity: standard_specific_humidity (generic function with 1 method)
└── ozone_mole_fraction: nothing

julia> ColumnExtension(Float32, top = 80e3, layers = 60, ozone_mole_fraction = 1e-6)
ColumnExtension{Float32}
├── top: 80000.0 m
├── layers: 60
├── blending_height: 1000.0 m
├── temperature: standard_atmosphere_temperature (generic function with 1 method)
├── specific_humidity: standard_specific_humidity (generic function with 1 method)
└── ozone_mole_fraction: 1.0e-6
```
"""
struct ColumnExtension{FT, T, Q, O}
    "Height of the top of the extended column [m]"
    top :: FT
    "Number of extension layers between the grid top and `top`"
    layers :: Int
    "Height over which the temperature anchor at the grid top decays [m]; `0` disables it"
    blending_height :: FT
    "Temperature profile `z -> K`"
    temperature :: T
    "Specific humidity profile `z -> kg kg⁻¹`"
    specific_humidity :: Q
    "Ozone mole fraction `z -> mol mol⁻¹`, a number, or `nothing` (`BackgroundAtmosphere.O₃`)"
    ozone_mole_fraction :: O
end

"""
$(TYPEDSIGNATURES)

Construct a [`ColumnExtension`](@ref) of float type `FT` reaching `top` (default 65 km) with
`layers` layers (default 40), a `blending_height` of 1 km for the temperature anchor, and
the [`standard_atmosphere_temperature`](@ref) and `standard_specific_humidity` profiles.
"""
function ColumnExtension(FT = Oceananigans.defaults.FloatType;
                         top = 65e3,
                         layers = 40,
                         blending_height = 1e3,
                         temperature = standard_atmosphere_temperature,
                         specific_humidity = standard_specific_humidity,
                         ozone_mole_fraction = nothing)

    layers >= 1 || throw(ArgumentError("`layers` must be at least 1; received $layers."))
    blending_height >= 0 || throw(ArgumentError("`blending_height` must be non-negative; received $blending_height."))
    return ColumnExtension(convert(FT, top), Int(layers), convert(FT, blending_height),
                           temperature, specific_humidity, convert_number(ozone_mole_fraction, FT))
end

# A constant ozone mole fraction takes the extension's float type; profiles pass through
convert_number(x::Number, FT) = convert(FT, x)
convert_number(x, FT) = x

profile_string(x::Number) = prettysummary(x)
profile_string(::Nothing) = "nothing"
profile_string(x) = summary(x)

Base.summary(::ColumnExtension{FT}) where FT = "ColumnExtension{$FT}"

Base.show(io::IO, extension::ColumnExtension) =
    print(io, summary(extension), "\n",
          "├── top: ", prettysummary(extension.top), " m\n",
          "├── layers: ", extension.layers, "\n",
          "├── blending_height: ", prettysummary(extension.blending_height), " m\n",
          "├── temperature: ", profile_string(extension.temperature), "\n",
          "├── specific_humidity: ", profile_string(extension.specific_humidity), "\n",
          "└── ozone_mole_fraction: ", profile_string(extension.ozone_mole_fraction))

# The depth spanned by `N` geometrically stretched layers of ratio `r` starting from `Δz₁`,
# Δz₁ (1 + r + … + rᴺ⁻¹), summed term by term so that it is regular at r = 1.
function geometric_stack_depth(Δz₁, r, N)
    depth = zero(Δz₁)
    Δz = Δz₁
    for _ in 1:N
        depth += Δz
        Δz *= r
    end
    return depth
end

"""
$(TYPEDSIGNATURES)

The ratio `r` between successive layers of a geometric stretching of `N` layers whose first layer
is `Δz₁` thick and which together span `depth`, i.e. the root of

```math
Δz₁ (1 + r + ⋯ + rᴺ⁻¹) = depth ,
```

found by bisection. The left-hand side increases monotonically with `r` from `Δz₁` (as `r → 0`), so the
root exists and is unique for any `depth > Δz₁`; it is greater than one when `depth > N Δz₁` (layers
grow with height) and less than one otherwise. For `N = 1` the ratio is immaterial and `1` is returned.
"""
function geometric_stretching_ratio(Δz₁, depth, N)
    FT = float(promote_type(typeof(Δz₁), typeof(depth)))
    Δz₁ = FT(Δz₁)
    depth = FT(depth)

    N == 1 && return one(FT)
    depth > Δz₁ || throw(ArgumentError("`depth` ($depth) must exceed the first layer thickness ($Δz₁) for N = $N > 1 layers."))

    residual(r) = geometric_stack_depth(Δz₁, r, N) - depth

    # Bracket the root: the residual is negative as r → 0 and grows without bound with r.
    lo = zero(FT)
    hi = one(FT)
    while residual(hi) < 0
        hi *= 2
    end

    # Bisect until the bracket cannot be halved in floating point
    while true
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        if residual(mid) < 0
            lo = mid
        else
            hi = mid
        end
    end

    return hi
end

"""
$(TYPEDSIGNATURES)

The face heights of the extension layers of `extension` above a grid whose top face is at `z_top`
and whose top layer is `Δz_top` thick: a vector of `Nₑ + 1` heights starting at `z_top` and ending
exactly at `extension.top`, with the first layer `Δz_top` thick and successive layers stretched by
the [`geometric_stretching_ratio`](@ref). `Nₑ = extension.layers`, or `0` (a single face at `z_top`)
when the grid already reaches `extension.top`.

```jldoctest
julia> using Breeze

julia> using Breeze.AtmosphereModels: column_extension_faces

julia> extension = ColumnExtension(top = 3100, layers = 3);

julia> column_extension_faces(extension, 3000, 10)
4-element Vector{Float64}:
 3000.0
 3010.0
 3035.4138126514913
 3100.0
```
"""
function column_extension_faces(extension::ColumnExtension{FT}, z_top, Δz_top) where FT
    z_top = FT(z_top)
    top = extension.top
    z_top >= top && return [z_top]

    Nₑ = extension.layers
    Δz₁ = FT(Δz_top)
    r = geometric_stretching_ratio(Δz₁, top - z_top, Nₑ)

    faces = Vector{FT}(undef, Nₑ + 1)
    faces[1] = z_top
    Δz = Δz₁
    for m in 1:Nₑ
        faces[m+1] = faces[m] + Δz
        Δz *= r
    end

    # The bisection lands within rounding of `top`; make the last face exact.
    faces[end] = top

    return faces
end
