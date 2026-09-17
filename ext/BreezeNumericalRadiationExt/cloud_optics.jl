#####
##### Cloud optics: scattering tables mapped onto the ecCKD g points
#####
##### All-sky radiation folds the cloud liquid and ice of every layer into the layer optics of
##### `layer_optics.jl` through NumericalRadiation's per-g-point `SpectralCloudOptics`. Each phase
##### is read once from an ecRad scattering table (Mie droplets, Baum general-habit-mixture ice),
##### mapped onto the g points of the selected gas model's longwave and shortwave spectral
##### definitions at the model's constant effective radius, and moved to the grid's architecture.
#####

# The scattering tables shipped with the ecRad data, keyed by the selectors of `CloudScatteringTables`
const CLOUD_SCATTERING_FILES = (mie_droplet = "mie_droplet_scattering.nc",
                                baum_general_habit_mixture = "baum-general-habit-mixture_ice_scattering.nc")

function cloud_scattering_table_path(selector::Symbol)
    haskey(CLOUD_SCATTERING_FILES, selector) ||
        throw(ArgumentError("Unknown cloud scattering table `:$selector`; the ecRad data ship " *
                            join(map(repr, keys(CLOUD_SCATTERING_FILES)), ", ") *
                            ". Alternatively pass the path of a scattering table in ecRad's format."))
    return ecrad_data_file(getproperty(CLOUD_SCATTERING_FILES, selector))
end

cloud_scattering_table_path(path::AbstractString) = String(path)

# The spectral mappings live in the ecCKD definition files of the gas model: the reference
# selectors resolve to those files and a `(longwave, shortwave)` pair of paths is used as is. A
# preloaded gas optics model carries no definition files, so it cannot take clouds.
spectral_mapping_paths(selector::Union{Symbol, AbstractString}) = reference_ecckd_definition_paths(selector)
spectral_mapping_paths(paths::NamedTuple) = paths
spectral_mapping_paths(gas_model) =
    throw(ArgumentError("All-sky ecCKD radiation maps the cloud scattering tables onto the g points of the " *
                        "gas model's ecCKD definition files, which a preloaded $(summary(gas_model)) does not " *
                        "carry; pass `gas_model` as a reference selector (`:climate_32x32`) or as " *
                        "`(longwave = path, shortwave = path)` when `clouds` are given."))

# One phase on one spectral region: the table mapped onto the g points at the model's constant
# effective radius, in the grid's float type and on its architecture
function spectral_cloud_optics(FT, table, mapping, effective_radius, arch)
    cloud = SpectralCloudOptics(FT, table, mapping; effective_radius)
    return adapt(array_type(arch){FT}, cloud)
end

"""
$(TYPEDSIGNATURES)

Read the liquid and ice scattering tables of `tables::CloudScatteringTables` and map them onto
the longwave and shortwave g points of the gas model selected by `gas_model` (a reference selector
or `(longwave = path, shortwave = path)` definition files) at the constant effective radii
`liquid_radius` and `ice_radius` (m), returning `(longwave, shortwave)` `(liquid, ice)` pairs of
`NumericalRadiation.SpectralCloudOptics` in float type `FT` on architecture `arch`.

Clear sky (`tables::Nothing`) returns `(nothing, nothing)`: the layer optics then add no cloud.
"""
function load_cloud_optics(FT, tables::CloudScatteringTables, gas_model, liquid_radius, ice_radius, arch)
    liquid_table = read_ecckd_tables(() -> read_cloud_scattering_table(cloud_scattering_table_path(tables.liquid)))
    ice_table = read_ecckd_tables(() -> read_cloud_scattering_table(cloud_scattering_table_path(tables.ice)))

    paths = spectral_mapping_paths(gas_model)
    longwave_mapping = read_ecckd_tables(() -> read_ecckd_spectral_mapping(paths.longwave))
    shortwave_mapping = read_ecckd_tables(() -> read_ecckd_spectral_mapping(paths.shortwave))

    longwave = (liquid = spectral_cloud_optics(FT, liquid_table, longwave_mapping, liquid_radius, arch),
                ice = spectral_cloud_optics(FT, ice_table, longwave_mapping, ice_radius, arch))

    shortwave = (liquid = spectral_cloud_optics(FT, liquid_table, shortwave_mapping, liquid_radius, arch),
                 ice = spectral_cloud_optics(FT, ice_table, shortwave_mapping, ice_radius, arch))

    return longwave, shortwave
end

load_cloud_optics(FT, ::Nothing, gas_model, liquid_radius, ice_radius, arch) = (nothing, nothing)

# The cloud optics must be mapped onto the same g points the gas model integrates over
function validate_cloud_g_points(cloud::NamedTuple, weights, region)
    Ngpoints = length(weights)
    for phase in (cloud.liquid, cloud.ice)
        size(phase.mass_extinction_coefficient, 1) == Ngpoints ||
            throw(ArgumentError("The $region cloud optics are mapped onto " *
                                "$(size(phase.mass_extinction_coefficient, 1)) g points but the gas model has $Ngpoints"))
    end
    return nothing
end

validate_cloud_g_points(::Nothing, weights, region) = nothing
