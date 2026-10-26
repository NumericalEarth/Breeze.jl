#####
##### ecCKD RadiativeTransferModel: full-spectrum radiation through NumericalRadiation.jl
#####

"""
$(TYPEDEF)

The longwave half of an ecCKD radiation model.

Fields:
- `gas_model`: ecCKD gas optics model
- `cloud`: Longwave cloud optics, or `nothing` for clear sky
- `weights`: Longwave spectral weights
- `mole_fractions`: Mole fractions of the well-mixed gases relative to dry air
"""
struct EcCKDLongwave{M, C, W, FT}
    gas_model :: M
    cloud :: C
    weights :: W
    mole_fractions :: NamedTuple{(:CO₂, :CH₄, :N₂O, :CFC₁₁, :CFC₁₂), NTuple{5, FT}}
end

"""
$(TYPEDEF)

The shortwave half of an ecCKD radiation model.

Fields:
- `gas_model`: ecCKD gas optics model
- `cloud`: Shortwave cloud optics, or `nothing` for clear sky
- `weights`: Shortwave spectral weights
- `solar_constant`: Solar constant [W m⁻²], a number or an `(Nx, Ny)` array of per-column values
"""
struct EcCKDShortwave{M, C, W, S}
    gas_model :: M
    cloud :: C
    weights :: W
    solar_constant :: S
end

# Dispatch on `atmospheric_state::ColumnWorkspace`: strictly more specific than the RRTMGP
# clear-sky model (slot 4 `BackgroundAtmosphere`) and disjoint from its all-sky model.
const EcCKDRadiativeTransferModel = RadiativeTransferModel{<:Any, <:Any, <:Any, <:BackgroundAtmosphere, <:ColumnWorkspace}

# The gases of the column optics, in the order of the ecCKD tables: dry air (N₂, O₂ and the trace
# gases the tables fold into it), the two spatially varying gases, and the well-mixed gases of
# `BackgroundAtmosphere` the reference models tabulate.
const ECCKD_GAS_NAMES = (:dry_air, :H₂O, :O₃, :CO₂, :CH₄, :N₂O, :CFC₁₁, :CFC₁₂)

# `BackgroundAtmosphere` gases the reference ecCKD models do not tabulate: rejecting a nonzero
# value is better than silently ignoring it.
const UNSUPPORTED_BACKGROUND_GASES = (:CO, :NO₂, :CFC₂₂, :CCl₄, :CF₄, :HFC₁₂₅, :HFC₁₃₄ₐ, :HFC₁₄₃ₐ, :HFC₂₃, :HFC₃₂)

function validate_background_gases(background::BackgroundAtmosphere)
    for name in UNSUPPORTED_BACKGROUND_GASES
        value = getproperty(background, name)
        value == 0 || throw(ArgumentError("`BackgroundAtmosphere.$name = $value` is not supported by the ecCKD " *
                                          "gas optics, which tabulate $(join(ECCKD_GAS_NAMES[2:end], ", ")) " *
                                          "(N₂ and O₂ form the dry air); set it to zero."))
    end
    return nothing
end

# The ecCKD tables live in netCDF files that NumericalRadiation reads through its NCDatasets
# extension; without it the reader throws an `ArgumentError` naming NCDatasets, which is
# rewrapped here to say what to do in terms of this constructor.
function read_ecckd_tables(load)
    try
        return load()
    catch err
        if err isa ArgumentError && occursin("NCDatasets", err.msg)
            throw(ArgumentError("EcCKDOptics reads its gas optics tables from netCDF files, which requires NCDatasets:\n\n" *
                                "    using NCDatasets\n\nand then construct RadiativeTransferModel again."))
        end
        rethrow()
    end
end

# The tables are read in the grid's float type; a preloaded model is converted to it (and to
# the grid's architecture) by `adapt` below.
load_gas_optics_model(FT, selector::Union{Symbol, AbstractString}) =
    read_ecckd_tables(() -> read_reference_ecckd_gas_optics(FT, selector; names = ECCKD_GAS_NAMES))

load_gas_optics_model(FT, paths::NamedTuple) =
    read_ecckd_tables(() -> read_ecckd_tabulated_gas_optics(FT, paths.longwave, paths.shortwave; names = ECCKD_GAS_NAMES))

load_gas_optics_model(FT, model::Union{EcCKDTabulatedGasOpticsModel, EcCKDGasOpticsModel}) = model

# Warn once at construction when the extension's temperatures leave the Planck source table, where
# the interpolation holds the table edge (a gray `σT⁴` model has no table and no edge). The
# interface temperatures interpolate the layer temperatures, so they share their range.
function warn_source_table_range(gas_model::EcCKDTabulatedGasOpticsModel, extension::OverlyingAtmosphereColumn)
    source_grid = gas_model.longwave_source_temperature_grid
    isnothing(source_grid) && return nothing
    T = Array(extension.T)
    (minimum(T) < first(source_grid) || maximum(T) > last(source_grid)) &&
        @warn "The overlying atmosphere's temperatures span $(extrema(T)) K, outside the ecCKD Planck source table " *
              "$(first(source_grid))–$(last(source_grid)) K; the table edge is used beyond it."
    return nothing
end

warn_source_table_range(gas_model, extension) = nothing

# A number becomes a number of the grid's float type; per-column values become an `(Nx, Ny)` array on the grid's architecture
materialize_solar_constant(solar_constant::Number, grid) = convert(eltype(grid), solar_constant)

function materialize_solar_constant(solar_constant::AbstractArray, grid)
    size(solar_constant) == size(grid)[1:2] ||
        throw(ArgumentError("expected an (Nx, Ny) = $(size(grid)[1:2]) array of per-column solar constants, got size $(size(solar_constant))"))
    return on_architecture(architecture(grid), convert(Array{eltype(grid)}, solar_constant))
end

effective_radius_model(FT, model::ConstantRadiusParticles) = ConstantRadiusParticles(convert(FT, model.radius))
effective_radius_model(FT, model) =
    throw(ArgumentError("EcCKDOptics supports only `ConstantRadiusParticles` effective radius models for now; " *
                        "received $(summary(model)). Variable effective radii are a planned follow-up."))

"""
$(TYPEDSIGNATURES)

Construct a full-spectrum `RadiativeTransferModel` on `grid` with the ecCKD gas optics of
`optics::EcCKDOptics`, solved column by column by NumericalRadiation.jl. Radiation is solved
on the grid's layers plus the layers of `overlying_atmosphere` above the grid top (a
[`OverlyingAtmosphereColumn`](@ref), or `nothing` to stop at the grid top).

With `optics.clouds::CloudScatteringTables` the radiation is all-sky: the cloud liquid and ice
of every grid cell (from the model's microphysics) scatter and absorb with the scattering tables
mapped onto the ecCKD g points at the constant effective radii of `liquid_effective_radius` and
`ice_effective_radius`; the shortwave folds the cloud scattering into the two-stream solution and
the longwave adds the cloud absorption `κ (1 - ω) W` only and neglects longwave cloud scattering
(ecRad's `do_lw_cloud_scattering = false` setting, and what RRTMGP's no-scattering longwave
solver does; scattering is a planned follow-up). With `optics.clouds = nothing` the sky is clear
whatever the microphysics holds.

The gas optics and cloud scattering tables are read from netCDF files, which requires
`using NCDatasets`. The column kernel takes gravity and the molar masses from `constants`;
the Stefan–Boltzmann constant of the gray Planck source (used by g points without a source
table) is carried by the gas optics model.

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
- `solar_constant`: Top-of-atmosphere solar flux in W/m² (default: 1361), a scalar or an `(Nx, Ny)` array of per-column values
- `schedule`: When to recompute the fluxes (default: `IterationInterval(1)`, every iteration)
- `liquid_effective_radius`: Model for cloud liquid effective radius in meters (default: `ConstantRadiusParticles(10e-6)`);
  only `ConstantRadiusParticles` for now
- `ice_effective_radius`: Model for cloud ice effective radius in meters (default: `ConstantRadiusParticles(30e-6)`);
  only `ConstantRadiusParticles` for now
- `overlying_atmosphere`: The atmosphere above the grid top (default: `OverlyingAtmosphereColumn(eltype(grid))`;
  `nothing` solves the grid's column only, with no atmosphere above it)
- `column_batches`: Number of batches to split the radiation solve into, trading speed for memory
  (default: `nothing`, one unbatched solve). The per-column state and scratch are then stored for
  one batch only. Batches are whole j-rows of `cld(Ny, column_batches)` rows each, as for the
  RRTMGP optics; when the rows do not divide `Ny`, the final batch overlaps the one before it. Each
  batch is a separate kernel launch of a register-limited kernel, so on a GPU smaller batches cost
  time: keep batches to several times the number of columns the GPU solves at once.
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
                                                 overlying_atmosphere = OverlyingAtmosphereColumn(eltype(grid)),
                                                 column_batches = nothing)

    FT = eltype(grid)
    arch = architecture(grid)

    solar_position = maybe_infer_solar_position(solar_position, grid)
    solar_constant = materialize_solar_constant(solar_constant, grid)

    validate_surface_fractions(; surface_emissivity, surface_albedo,
                                 direct_surface_albedo, diffuse_surface_albedo)

    validate_background_gases(background_atmosphere)

    liquid_effective_radius = effective_radius_model(FT, liquid_effective_radius)
    ice_effective_radius = effective_radius_model(FT, ice_effective_radius)

    direct_surface_albedo, diffuse_surface_albedo =
        resolve_surface_albedos(surface_albedo, direct_surface_albedo, diffuse_surface_albedo, grid, solar_position)

    surface_emissivity = materialize_surface_property(surface_emissivity, grid, solar_position)

    # Gas optics tables, then the extension sampled on the host (it needs the background's O₃
    # as a profile, before materialization turns it into a field on the grid)
    host_gas_model = load_gas_optics_model(FT, optics.gas_model)
    extension = materialize_overlying_atmosphere(overlying_atmosphere, grid, background_atmosphere)
    warn_source_table_range(host_gas_model, extension)
    gas_model = adapt(array_type(arch){FT}, host_gas_model)

    longwave_cloud, shortwave_cloud = load_cloud_optics(FT, optics.clouds, optics.gas_model,
                                                        liquid_effective_radius.radius, ice_effective_radius.radius, arch)
    validate_cloud_g_points(longwave_cloud, gas_model.longwave_weights, "longwave")
    validate_cloud_g_points(shortwave_cloud, gas_model.shortwave_weights, "shortwave")

    background_atmosphere = materialize_background_atmosphere(background_atmosphere, grid)

    Nx, Ny, Nz = size(grid)
    N = Nz + number_of_extension_layers(extension)
    rows = resolve_column_batch_rows(column_batches, Ny)
    workspace = ColumnWorkspace(arch, FT, Nx, rows, Nx * Ny, N, gas_model, longwave_cloud, extension)
    initialize_cos_zenith!(workspace.cos_zenith, solar_position)

    surface_emissivity = constant_field_property(surface_emissivity, FT)
    direct_surface_albedo = constant_field_property(direct_surface_albedo, FT)
    diffuse_surface_albedo = constant_field_property(diffuse_surface_albedo, FT)
    surface_temperature = constant_field_property(surface_temperature, FT)

    surface_radiation = SurfaceRadiation(surface_temperature, surface_emissivity,
                                         direct_surface_albedo, diffuse_surface_albedo)

    mole_fractions = (CO₂ = convert(FT, background_atmosphere.CO₂),
                      CH₄ = convert(FT, background_atmosphere.CH₄),
                      N₂O = convert(FT, background_atmosphere.N₂O),
                      CFC₁₁ = convert(FT, background_atmosphere.CFC₁₁),
                      CFC₁₂ = convert(FT, background_atmosphere.CFC₁₂))

    longwave = EcCKDLongwave(gas_model, longwave_cloud, gas_model.longwave_weights, mole_fractions)
    shortwave = EcCKDShortwave(gas_model, shortwave_cloud, gas_model.shortwave_weights, solar_constant)

    upwelling_longwave_flux = ZFaceField(grid)
    downwelling_longwave_flux = ZFaceField(grid)
    upwelling_shortwave_flux = ZFaceField(grid)
    downwelling_shortwave_flux = ZFaceField(grid)
    flux_divergence = CenterField(grid)

    return RadiativeTransferModel(solar_constant,
                                  solar_position,
                                  surface_radiation,
                                  background_atmosphere,
                                  workspace,
                                  longwave,
                                  shortwave,
                                  upwelling_longwave_flux,
                                  downwelling_longwave_flux,
                                  upwelling_shortwave_flux,
                                  downwelling_shortwave_flux,
                                  flux_divergence,
                                  liquid_effective_radius,
                                  ice_effective_radius,
                                  column_batches,
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
              "└── overlying_atmosphere: ", extension_summary(rtm.atmospheric_state.extension))
end
