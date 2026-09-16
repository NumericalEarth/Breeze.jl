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
$(TYPEDFIELDS)

Per-column state, interpolation stencils, gas amounts, fluxes and scratch space for the ecCKD
radiation of every column of the grid, laid out as `(Nc, N)` layer and `(Nc, N + 1)` interface
arrays (`Nc` columns, `N` layers) in NumericalRadiation's top-down order.
"""
struct SpectralColumns{FT, AI, AF, V, SW, X}
    "Layer pressure [Pa], `(Nc, N)`"
    pressure_layers :: AF
    "Layer temperature [K], `(Nc, N)`"
    temperature_layers :: AF
    "Interface pressure [Pa], `(Nc, N + 1)`"
    pressure_interfaces :: AF
    "Interface temperature [K], `(Nc, N + 1)`"
    temperature_interfaces :: AF
    "Lower index of the pressure bracket of the gas optics stencil, `(Nc, N)`"
    stencil_ip :: AI
    "Weight of the pressure bracket of the gas optics stencil, `(Nc, N)`"
    stencil_wp :: AF
    "Lower index of the temperature bracket of the gas optics stencil, `(Nc, N)`"
    stencil_it :: AI
    "Weight of the temperature bracket of the gas optics stencil, `(Nc, N)`"
    stencil_wt :: AF
    "Lower index of the H₂O bracket of the gas optics stencil, `(Nc, N)`"
    stencil_ih :: AI
    "Weight of the H₂O bracket of the gas optics stencil, `(Nc, N)`"
    stencil_wh :: AF
    "Lower index of the Planck source bracket at the interfaces, `(Nc, N + 1)`"
    source_is :: AI
    "Weight of the Planck source bracket at the interfaces, `(Nc, N + 1)`"
    source_ws :: AF
    "Dry air (composite gas) molar amount [mol m⁻²], `(Nc, N)`"
    dry_air :: AF
    "Water vapor molar amount [mol m⁻²], `(Nc, N)`"
    water_vapor :: AF
    "Ozone molar amount [mol m⁻²], `(Nc, N)`"
    ozone :: AF
    "Cloud liquid water path [kg m⁻²], `(Nc, N)`"
    liquid_water_path :: AF
    "Cloud ice water path [kg m⁻²], `(Nc, N)`"
    ice_water_path :: AF
    "Cosine of the solar zenith angle, `(Nc,)`"
    cos_zenith :: V
    "Upwelling longwave flux [W m⁻²], `(Nc, N + 1)`"
    flux_up_lw :: AF
    "Downwelling longwave flux [W m⁻²], positive downward, `(Nc, N + 1)`"
    flux_down_lw :: AF
    "Upwelling shortwave flux [W m⁻²], `(Nc, N + 1)`"
    flux_up_sw :: AF
    "Downwelling shortwave flux [W m⁻²], positive downward, `(Nc, N + 1)`"
    flux_down_sw :: AF
    "Longwave layer transmittance scratch, `(Nc, N)`"
    transmittance :: AF
    "Longwave upward layer source scratch, `(Nc, N)`"
    source_up :: AF
    "Shortwave adding-method scratch: five `(Nc, N)` layer arrays and two `(Nc, N + 1)` interface arrays"
    shortwave :: SW
    "The materialized column extension above the grid top, or `nothing`"
    extension :: X
end

const SHORTWAVE_LAYER_SCRATCH = (:reflectance, :transmittance, :direct_reflectance,
                                 :direct_diffuse_transmittance, :direct_transmittance)
const SHORTWAVE_INTERFACE_SCRATCH = (:stack_albedo, :source)

# The float type is that of the layer arrays; every other parameter follows from the fields.
function SpectralColumns{FT}(args...) where FT
    fields = NamedTuple{fieldnames(SpectralColumns)}(args)
    AI = typeof(fields.stencil_ip)
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
