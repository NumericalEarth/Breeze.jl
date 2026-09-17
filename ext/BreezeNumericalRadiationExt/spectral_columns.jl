#####
##### Spectral column workspace
#####
##### Every column of the grid, extended above the grid top, is stored as one row of the arrays
##### below in NumericalRadiation's top-down convention: column `c = i + (j - 1) Nx`, and with
##### `N = Nz + Nₑ` layers (`Nₑ` extension layers), grid cell `k` is column layer `N + 1 - k` and
##### grid face `k` is column interface `N + 2 - k`. Gas amounts are molar column amounts per layer
##### (mol m⁻²), water paths are kg m⁻², and the fluxes carry NumericalRadiation's sign convention
##### (each stream positive in its own direction).
#####

"""
$(TYPEDEF)

Per-column state, interpolation stencils, gas amounts, fluxes and scratch space for the ecCKD
radiation of every column of the grid, laid out as `(Nc, N)` layer and `(Nc, N + 1)` interface
arrays (`Nc` columns, `N` layers) in NumericalRadiation's top-down order.

Fields:
- `pressure_layers`: Layer pressure [Pa], `(Nc, N)`
- `temperature_layers`: Layer temperature [K], `(Nc, N)`
- `pressure_interfaces`: Interface pressure [Pa], `(Nc, N + 1)`
- `temperature_interfaces`: Interface temperature [K], `(Nc, N + 1)`
- `pressure_index`: Lower index of the pressure bracket of the gas optics stencil, `(Nc, N)`
- `pressure_weight`: Weight of the pressure bracket of the gas optics stencil, `(Nc, N)`
- `temperature_index`: Lower index of the temperature bracket of the gas optics stencil, `(Nc, N)`
- `temperature_weight`: Weight of the temperature bracket of the gas optics stencil, `(Nc, N)`
- `water_vapor_index`: Lower index of the H₂O bracket of the gas optics stencil, `(Nc, N)`
- `water_vapor_weight`: Weight of the H₂O bracket of the gas optics stencil, `(Nc, N)`
- `source_index`: Lower index of the Planck source bracket at the interfaces, `(Nc, N + 1)`
- `source_weight`: Weight of the Planck source bracket at the interfaces, `(Nc, N + 1)`
- `dry_air`: Dry air (composite gas) molar amount [mol m⁻²], `(Nc, N)`
- `water_vapor`: Water vapor molar amount [mol m⁻²], `(Nc, N)`
- `ozone`: Ozone molar amount [mol m⁻²], `(Nc, N)`
- `liquid_water_path`: Cloud liquid water path [kg m⁻²], `(Nc, N)`
- `ice_water_path`: Cloud ice water path [kg m⁻²], `(Nc, N)`
- `cos_zenith`: Cosine of the solar zenith angle, `(Nc,)`
- `longwave_up`: Upwelling longwave flux [W m⁻²], `(Nc, N + 1)`
- `longwave_down`: Downwelling longwave flux [W m⁻²], positive downward, `(Nc, N + 1)`
- `shortwave_up`: Upwelling shortwave flux [W m⁻²], `(Nc, N + 1)`
- `shortwave_down`: Downwelling shortwave flux [W m⁻²], positive downward, `(Nc, N + 1)`
- `transmittance`: Longwave layer transmittance scratch, `(Nc, N)`
- `source_up`: Longwave upward layer source scratch, `(Nc, N)`
- `shortwave`: Shortwave adding-method scratch: five `(Nc, N)` layer arrays and two `(Nc, N + 1)`
  interface arrays
- `extension`: The materialized column extension above the grid top, or `nothing`
"""
struct SpectralColumns{FT, AI, AF, V, SW, X}
    pressure_layers :: AF
    temperature_layers :: AF
    pressure_interfaces :: AF
    temperature_interfaces :: AF
    pressure_index :: AI
    pressure_weight :: AF
    temperature_index :: AI
    temperature_weight :: AF
    water_vapor_index :: AI
    water_vapor_weight :: AF
    source_index :: AI
    source_weight :: AF
    dry_air :: AF
    water_vapor :: AF
    ozone :: AF
    liquid_water_path :: AF
    ice_water_path :: AF
    cos_zenith :: V
    longwave_up :: AF
    longwave_down :: AF
    shortwave_up :: AF
    shortwave_down :: AF
    transmittance :: AF
    source_up :: AF
    shortwave :: SW
    extension :: X
end

const SHORTWAVE_LAYER_SCRATCH = (:reflectance, :transmittance, :direct_reflectance,
                                 :direct_diffuse_transmittance, :direct_flux)
const SHORTWAVE_INTERFACE_SCRATCH = (:stack_albedo, :source)

# The float type is that of the layer arrays; every other parameter follows from the fields.
function SpectralColumns{FT}(args...) where FT
    fields = NamedTuple{fieldnames(SpectralColumns)}(args)
    AI = typeof(fields.pressure_index)
    AF = typeof(fields.pressure_layers)
    V = typeof(fields.cos_zenith)
    SW = typeof(fields.shortwave)
    X = typeof(fields.extension)
    return SpectralColumns{FT, AI, AF, V, SW, X}(args...)
end

"""
$(TYPEDSIGNATURES)

Allocate zeroed [`SpectralColumns`](@ref) of float type `FT` for `Nc` columns of `N` layers on
architecture `arch`, carrying `extension` (a [`MaterializedColumnExtension`](@ref) or `nothing`).
"""
function SpectralColumns(arch, FT, Nc, N, extension)
    layers() = on_architecture(arch, zeros(FT, Nc, N))
    interfaces() = on_architecture(arch, zeros(FT, Nc, N + 1))
    layer_indices() = on_architecture(arch, zeros(Int32, Nc, N))
    interface_indices() = on_architecture(arch, zeros(Int32, Nc, N + 1))

    cos_zenith = on_architecture(arch, zeros(FT, Nc))

    shortwave = NamedTuple{(SHORTWAVE_LAYER_SCRATCH..., SHORTWAVE_INTERFACE_SCRATCH...)}(
        (ntuple(_ -> layers(), length(SHORTWAVE_LAYER_SCRATCH))...,
         ntuple(_ -> interfaces(), length(SHORTWAVE_INTERFACE_SCRATCH))...))

    return SpectralColumns{FT}(layers(), layers(), interfaces(), interfaces(),
                               layer_indices(), layers(), layer_indices(), layers(), layer_indices(), layers(),
                               interface_indices(), interfaces(),
                               layers(), layers(), layers(), layers(), layers(),
                               cos_zenith,
                               interfaces(), interfaces(), interfaces(), interfaces(),
                               layers(), layers(),
                               shortwave, extension)
end

Adapt.adapt_structure(to, columns::SpectralColumns{FT}) where FT =
    SpectralColumns{FT}((adapt(to, getfield(columns, name)) for name in fieldnames(SpectralColumns))...)

Base.eltype(::SpectralColumns{FT}) where FT = FT
Base.eltype(::Type{<:SpectralColumns{FT}}) where FT = FT

"""
$(TYPEDSIGNATURES)

The number of layers of the extended columns, `N = Nz + Nₑ`.
"""
@inline number_of_layers(columns::SpectralColumns) = size(columns.pressure_layers, 2)

"""
$(TYPEDSIGNATURES)

The number of columns, `Nc = Nx Ny`.
"""
@inline number_of_columns(columns::SpectralColumns) = size(columns.pressure_layers, 1)
