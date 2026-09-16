#####
##### ecCKD RadiativeTransferModel: full-spectrum radiation through NumericalRadiation.jl
#####

"""
$(TYPEDEF)
$(TYPEDFIELDS)

The longwave half of an ecCKD radiation model: the gas optics model, the cloud optics (`nothing`
for clear sky), the longwave spectral weights, and the mole fractions of the well-mixed gases
relative to dry air.
"""
struct EcCKDLongwave{M, C, W, FT}
    "ecCKD gas optics model"
    gas_model :: M
    "Longwave cloud optics, or `nothing` for clear sky"
    cloud :: C
    "Longwave spectral weights"
    weights :: W
    "Mole fractions of the well-mixed gases relative to dry air"
    mole_fractions :: NamedTuple{(:co2, :ch4, :n2o, :cfc11, :cfc12), NTuple{5, FT}}
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

The shortwave half of an ecCKD radiation model: the gas optics model, the cloud optics (`nothing`
for clear sky), the shortwave spectral weights, and the solar constant.
"""
struct EcCKDShortwave{M, C, W, FT}
    "ecCKD gas optics model"
    gas_model :: M
    "Shortwave cloud optics, or `nothing` for clear sky"
    cloud :: C
    "Shortwave spectral weights"
    weights :: W
    "Solar constant [W m⁻²]"
    solar_constant :: FT
end

# Dispatch on `atmospheric_state::SpectralColumns`: strictly more specific than the RRTMGP
# clear-sky model (slot 4 `BackgroundAtmosphere`) and disjoint from its all-sky model.
const EcCKDRadiativeTransferModel = RadiativeTransferModel{<:Any, <:Any, <:Any, <:BackgroundAtmosphere, <:SpectralColumns}

# The gases carried by the staged columns, in the order of the ecCKD tables: dry air (the
# `composite` of N₂, O₂ and the trace gases the tables fold into it), the two spatially varying
# gases, and the well-mixed gases of `BackgroundAtmosphere` the reference models tabulate.
const ECCKD_GAS_NAMES = (:composite, :h2o, :o3, :co2, :ch4, :n2o, :cfc11, :cfc12)

# `BackgroundAtmosphere` gases the reference ecCKD models do not tabulate: rejecting a nonzero
# value is better than silently ignoring it.
const UNSUPPORTED_BACKGROUND_GASES = (:CO, :NO₂, :CFC₂₂, :CCl₄, :CF₄, :HFC₁₂₅, :HFC₁₃₄ₐ, :HFC₁₄₃ₐ, :HFC₂₃, :HFC₃₂)

function validate_background_gases(background::BackgroundAtmosphere)
    for name in UNSUPPORTED_BACKGROUND_GASES
        value = getproperty(background, name)
        value == 0 || throw(ArgumentError("`BackgroundAtmosphere.$name = $value` is not supported by the ecCKD " *
                                          "gas optics, which tabulate $(join(ECCKD_GAS_NAMES[2:end], ", ")) " *
                                          "(N₂ and O₂ form the composite gas); set it to zero."))
    end
    return nothing
end

# The ecCKD tables live in netCDF files that NumericalRadiation reads through its NCDatasets
# extension; without it the reader throws an `ArgumentError` naming NCDatasets, which is
# rewrapped here to say what to do in terms of this constructor.
function read_ecckd_tables(read)
    try
        return read()
    catch err
        if err isa ArgumentError && occursin("NCDatasets", err.msg)
            throw(ArgumentError("EcCKDOptics reads its gas optics tables from netCDF files, which requires NCDatasets:\n\n" *
                                "    using NCDatasets\n\nand then construct RadiativeTransferModel again."))
        end
        rethrow()
    end
end

load_gas_optics_model(selector::Union{Symbol, AbstractString}) =
    read_ecckd_tables(() -> read_reference_ecckd_gas_optics(selector; names = ECCKD_GAS_NAMES))

load_gas_optics_model(paths::NamedTuple) =
    read_ecckd_tables(() -> read_ecckd_tabulated_gas_optics(paths.longwave, paths.shortwave; names = ECCKD_GAS_NAMES))

load_gas_optics_model(model::Union{EcCKDTabulatedGasOpticsModel, EcCKDGasOpticsModel}) = model

# Warn once at construction when the extension's temperatures leave the Planck source table, where
# the interpolation holds the table edge (a gray `σT⁴` model has no table and no edge).
function warn_source_table_range(gas_model::EcCKDTabulatedGasOpticsModel, extension::MaterializedColumnExtension)
    source_grid = gas_model.longwave_source_temperature_grid
    isnothing(source_grid) && return nothing
    T = extension.temperature_interfaces
    (minimum(T) < first(source_grid) || maximum(T) > last(source_grid)) &&
        @warn "The column extension temperatures span $(extrema(T)) K, outside the ecCKD Planck source table " *
              "$(first(source_grid))–$(last(source_grid)) K; the table edge is used beyond it."
    return nothing
end

warn_source_table_range(gas_model, extension) = nothing

effective_radius_model(model::ConstantRadiusParticles, FT) = ConstantRadiusParticles(convert(FT, model.radius))
effective_radius_model(model, FT) =
    throw(ArgumentError("EcCKDOptics supports only `ConstantRadiusParticles` effective radius models for now; " *
                        "received $(summary(model)). Variable effective radii are a planned follow-up."))

"""
$(TYPEDSIGNATURES)

Construct a full-spectrum `RadiativeTransferModel` on `grid` with the ecCKD gas optics of
`optics::EcCKDOptics`, solved column by column by NumericalRadiation.jl. Radiation is solved
on the grid's layers plus the layers of `column_extension` above the grid top (a
[`ColumnExtension`](@ref), or `nothing` to stop at the grid top).

The gas optics tables are read from netCDF files, which requires `using NCDatasets`.

# Keyword Arguments
- `background_atmosphere`: Background atmospheric gas composition (default: `BackgroundAtmosphere()`).
  O₃ can be a Number or Function of `z`; CO₂, CH₄, N₂O, CFC₁₁ and CFC₁₂ are well-mixed. The
  remaining gases of `BackgroundAtmosphere` are not tabulated by the ecCKD models and must be zero.
- `surface_temperature`: Surface temperature in Kelvin, a `Number` or 2D `Field`. Default: `nothing` —
  bind one before the first radiation update (a coupled model wires its interface surface
  temperature into the radiation automatically).
- `solar_position`: Specification of the solar zenith angle. See [`AbstractSolarPosition`](@ref) and its subtypes:
  - [`ApparentSolarPosition`](@ref) (default) — time-varying, computed from the model clock and grid (or explicit) longitude/latitude.
  - [`FixedCosineZenith`](@ref) — constant cos(θ_z), independent of the clock.
- `surface_emissivity`: Surface emissivity, 0-1 (default: 0.98). Can be scalar or 2D field.
- `surface_albedo`: Surface albedo, 0-1. Can be scalar or 2D field.
                    Alternatively, provide both `direct_surface_albedo` and `diffuse_surface_albedo`.
- `direct_surface_albedo`: Direct surface albedo, 0-1. Can be scalar or 2D field.
- `diffuse_surface_albedo`: Diffuse surface albedo, 0-1. Can be scalar or 2D field.
- `solar_constant`: Top-of-atmosphere solar flux in W/m² (default: 1361)
- `schedule`: When to recompute the fluxes (default: `IterationInterval(1)`, every iteration)
- `liquid_effective_radius`: Model for cloud liquid effective radius in meters (default: `ConstantRadiusParticles(10e-6)`)
- `ice_effective_radius`: Model for cloud ice effective radius in meters (default: `ConstantRadiusParticles(30e-6)`)
- `column_extension`: The atmosphere above the grid top (default: `ColumnExtension(eltype(grid))`;
  `nothing` solves the grid's column only, with no atmosphere above it)
"""
function AtmosphereModels.RadiativeTransferModel(grid::AbstractGrid,
                                                 optics::EcCKDOptics,
                                                 constants::ThermodynamicConstants;
                                                 background_atmosphere = BackgroundAtmosphere(),
                                                 surface_temperature = nothing,
                                                 solar_position::AbstractSolarPosition = ApparentSolarPosition(),
                                                 surface_emissivity = 0.98,
                                                 surface_albedo = nothing,
                                                 direct_surface_albedo = nothing,
                                                 diffuse_surface_albedo = nothing,
                                                 solar_constant = 1361,
                                                 schedule = IterationInterval(1),
                                                 liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                                 ice_effective_radius = ConstantRadiusParticles(30e-6),
                                                 column_extension = ColumnExtension(eltype(grid)))

    FT = eltype(grid)
    arch = architecture(grid)

    isnothing(optics.clouds) ||
        throw(ArgumentError("All-sky ecCKD radiation (`EcCKDOptics(clouds = ...)`) is not available yet; " *
                            "use `EcCKDOptics(clouds = nothing)` for clear-sky radiation."))

    solar_position = maybe_infer_solar_position(solar_position, grid)

    validate_surface_fractions(; surface_emissivity, surface_albedo,
                                 direct_surface_albedo, diffuse_surface_albedo)

    validate_background_gases(background_atmosphere)

    liquid_effective_radius = effective_radius_model(liquid_effective_radius, FT)
    ice_effective_radius = effective_radius_model(ice_effective_radius, FT)

    direct_surface_albedo, diffuse_surface_albedo =
        resolve_surface_albedos(surface_albedo, direct_surface_albedo, diffuse_surface_albedo, grid, solar_position)

    surface_emissivity = materialize_surface_property(surface_emissivity, grid, solar_position)

    # Gas optics tables, then the extension sampled on the host (it needs the background's O₃
    # as a profile, before materialization turns it into a field on the grid)
    host_gas_model = load_gas_optics_model(optics.gas_model)
    extension = materialize_column_extension(column_extension, grid, background_atmosphere)
    warn_source_table_range(host_gas_model, extension)
    gas_model = adapt(array_type(arch), host_gas_model)

    background_atmosphere = materialize_background_atmosphere(background_atmosphere, grid)

    Nx, Ny, Nz = size(grid)
    Nc = Nx * Ny
    N = Nz + number_of_extension_layers(extension)
    columns = SpectralColumns(arch, FT, Nc, N, extension)
    initialize_cos_zenith!(columns.cos_zenith, solar_position)

    surface_emissivity = constant_field_property(surface_emissivity, FT)
    direct_surface_albedo = constant_field_property(direct_surface_albedo, FT)
    diffuse_surface_albedo = constant_field_property(diffuse_surface_albedo, FT)
    surface_temperature = constant_field_property(surface_temperature, FT)

    surface_radiation = SurfaceRadiation(surface_temperature, surface_emissivity,
                                         direct_surface_albedo, diffuse_surface_albedo)

    mole_fractions = (co2 = convert(FT, background_atmosphere.CO₂),
                      ch4 = convert(FT, background_atmosphere.CH₄),
                      n2o = convert(FT, background_atmosphere.N₂O),
                      cfc11 = convert(FT, background_atmosphere.CFC₁₁),
                      cfc12 = convert(FT, background_atmosphere.CFC₁₂))

    longwave = EcCKDLongwave(gas_model, nothing, gas_model.longwave_weights, mole_fractions)
    shortwave = EcCKDShortwave(gas_model, nothing, gas_model.shortwave_weights, convert(FT, solar_constant))

    upwelling_longwave_flux = ZFaceField(grid)
    downwelling_longwave_flux = ZFaceField(grid)
    upwelling_shortwave_flux = ZFaceField(grid)
    downwelling_shortwave_flux = ZFaceField(grid)
    flux_divergence = CenterField(grid)

    return RadiativeTransferModel(convert(FT, solar_constant),
                                  solar_position,
                                  surface_radiation,
                                  background_atmosphere,
                                  columns,
                                  longwave,
                                  shortwave,
                                  upwelling_longwave_flux,
                                  downwelling_longwave_flux,
                                  upwelling_shortwave_flux,
                                  downwelling_shortwave_flux,
                                  flux_divergence,
                                  liquid_effective_radius,
                                  ice_effective_radius,
                                  schedule)
end

number_of_longwave_g_points(rtm::EcCKDRadiativeTransferModel) = length(rtm.longwave_solver.weights)
number_of_shortwave_g_points(rtm::EcCKDRadiativeTransferModel) = length(rtm.shortwave_solver.weights)

optics_summary(rtm::EcCKDRadiativeTransferModel) =
    string("EcCKDOptics with ", number_of_longwave_g_points(rtm), " longwave and ",
           number_of_shortwave_g_points(rtm), " shortwave g-points, ",
           isnothing(rtm.longwave_solver.cloud) ? "clear sky" : "all sky")

function Base.show(io::IO, rtm::EcCKDRadiativeTransferModel)
    show_radiation_summary(io, rtm)
    print(io, "├── diffuse_surface_albedo: ", rtm.surface_radiation.diffuse_surface_albedo, "\n",
              "├── optics: ", optics_summary(rtm), "\n",
              "└── column_extension: ", extension_summary(rtm.atmospheric_state.extension))
end
