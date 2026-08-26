#####
##### P3 lookup table reader
#####
##### Reads the ASCII tables laid out in `lookup_table_format.jl` and builds the
##### `IceProperties` container the scheme evaluates. The
##### rain 1D integrals are absent from those files and are generated here with
##### Julia quadrature instead.
#####

export read_lookup_tables

using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Utils: TabulatedFunction

#####
##### Main entry point
#####

"""
$(TYPEDSIGNATURES)

Read the P3 lookup tables from their ASCII files and construct a complete
`PredictedParticlePropertiesMicrophysics` with tabulated ice integrals.

Nothing is fetched here: `directory` must already hold the table file, and it is an error
if it does not. The tables themselves are not part of the source tree — they are a lazy
`Pkg` artifact (`P3_lookup_tables` in Breeze's `Artifacts.toml`, a tarball pinned by
SHA-256), which `PredictedParticlePropertiesMicrophysics` resolves to a path with
`artifact"P3_lookup_tables"` and passes here. `Pkg` downloads and caches it on first use.

Rain 1D tables (velocity, evaporation) are generated from Julia quadrature
since they are not included in the ASCII table files.

# Arguments

- `directory`: Path to directory containing the ASCII table file
  (`p3_lookupTable_1.dat-v6.9-2momI`).

# Keyword Arguments

- `FT`: Float type (default `Float64`)
- `arch`: Architecture for GPU transfer (default `CPU()`)
- `thermodynamic_constants`: Source of shared phase and dry-air properties.
"""
function read_lookup_tables(directory::AbstractString;
                            FT::DataType = Oceananigans.defaults.FloatType,
                            arch = CPU(),
                            thermodynamic_constants = ThermodynamicConstants(FT),
                            minimum_mass_mixing_ratio = 1e-14,
                            minimum_number_mixing_ratio = 1e-16,
                            precipitation_boundary_condition = nothing,
                            negative_moisture_correction = SpeciesBorrowing(),
                            aerosol = nothing,
                            cloud = nothing,
                            process_rates = nothing,
                            warm_rain_scheme = KhairoutdinovKogan2000())

    table1_file = joinpath(directory, "p3_lookupTable_1.dat-v6.9-2momI")
    isfile(table1_file) || error("2momI table not found: $table1_file")

    # Parse Table 1 (ice integrals + embedded rain-ice collection)
    table1_fields, table2_fields = parse_lookup_table_file(table1_file, FT)

    # Build TabulatedFunction objects
    ice_tables_4d = build_table_1_functions(table1_fields, FT, arch)
    rain_ice_tables = build_table_2_functions(table2_fields, FT, arch)

    # Build IceProperties with tabulated fields
    ice = build_ice_properties_from_tables(ice_tables_4d, rain_ice_tables, FT;
                                           thermodynamic_constants)

    # Generate rain 1D tables from Julia quadrature
    rain_base = RainProperties(FT)
    rain = tabulate_rain_from_quadrature(rain_base, arch, FT)

    # Construct full scheme
    cloud = isnothing(cloud) ? CloudDropletProperties(FT) : cloud
    input_process_rates = if isnothing(process_rates)
        ProcessRateParameters(FT; thermodynamic_constants)
    else
        process_rates
    end

    return PredictedParticlePropertiesMicrophysics(
        FT(minimum_mass_mixing_ratio),
        FT(minimum_number_mixing_ratio),
        ice,
        rain,
        cloud,
        input_process_rates,
        precipitation_boundary_condition,
        negative_moisture_correction,
        aerosol,
        warm_rain_scheme
    )
end

#####
##### Build Table 1 (4D) TabulatedFunction objects
#####

function build_table_1_functions(table1_fields::Dict, FT::Type, arch)
    ranges = ice_integrals_axes(FT)

    result = Dict{Symbol, RimeDensityIndexedTable4D}()
    for (name, data) in table1_fields
        table = make_lookup_table(data, ranges, arch)
        result[name] = RimeDensityIndexedTable4D(table)
    end
    return result
end

#####
##### Build Table 2 (5D) TabulatedFunction objects
#####

function build_table_2_functions(table2_fields::Dict, FT::Type, arch)
    ranges = rain_ice_collection_axes(FT)

    result = Dict{Symbol, RimeDensityIndexedTable5D}()
    for (name, data) in table2_fields
        table = make_lookup_table(data, ranges, arch)
        result[name] = RimeDensityIndexedTable5D(table)
    end
    return result
end

#####
##### Build IceProperties from the tabulated ice integrals
#####

function build_ice_properties_from_tables(ice_4d, rain_ice, FT;
                                          thermodynamic_constants = ThermodynamicConstants(FT))
    # Start from default IceProperties for physical constants
    ice_base = IceProperties(FT; thermodynamic_constants)

    # Build sub-structs with tabulated fields replacing integral placeholders
    fall_speed = IceFallSpeed(
        ice_base.fall_speed.reference_air_density,
        ice_4d[:number_weighted],
        ice_4d[:mass_weighted],
    )

    deposition = IceDeposition(
        ice_4d[:ventilation],
        ice_4d[:ventilation_enhanced],
        ice_4d[:small_ice_ventilation_constant],
        ice_4d[:small_ice_ventilation_reynolds],
        ice_4d[:large_ice_ventilation_constant],
        ice_4d[:large_ice_ventilation_reynolds]
    )

    bulk_properties = IceBulkProperties(
        ice_base.bulk_properties.maximum_mean_diameter,
        ice_base.bulk_properties.minimum_mean_diameter,
        ice_4d[:effective_radius],
        ice_4d[:mean_diameter],
        ice_4d[:mean_density],
        ice_4d[:reflectivity],
        ice_4d[:slope_parameter],
        ice_4d[:shape_parameter],
        ice_4d[:shedding]
    )

    collection = IceCollection(
        ice_4d[:aggregation],
        ice_4d[:cloud_collection],
        ice_4d[:cloud_aerosol_collection],
        ice_4d[:ice_aerosol_collection]
    )

    lambda_limiter = IceLambdaLimiter(
        ice_4d[:small_q],
        ice_4d[:large_q]
    )

    ice_rain_coll = IceRainCollection(
        rain_ice[:rain_mass],
        rain_ice[:rain_number],
    )

    return IceProperties(
        ice_base.minimum_rime_density,
        ice_base.maximum_rime_density,
        ice_base.maximum_shape_parameter,
        fall_speed,
        deposition,
        bulk_properties,
        collection,
        lambda_limiter,
        ice_rain_coll)
end

#####
##### Rain tabulation from Julia quadrature
#####
##### Rain 1D tables are NOT in the ASCII files. We generate them
##### from Julia quadrature (extracted from tabulation.jl).
#####

function tabulate_rain_from_quadrature(rain::RainProperties, arch=CPU(),
                                       FT::DataType = Oceananigans.defaults.FloatType;
                                       lambda_points::Int = 200,
                                       log_lambda_range = (FT(2.5), FT(5.5)),
                                       quadrature_points::Int = 128)

    vel_mass_eval = RainMassWeightedVelocityEvaluator(FT; n_points=quadrature_points)
    vel_num_eval = RainNumberWeightedVelocityEvaluator(FT; n_points=quadrature_points)
    evap_eval = RainEvaporationVentilationEvaluator(FT; n_points=quadrature_points)

    tab_vel_mass = TabulatedFunction(vel_mass_eval, arch, FT;
                                     range=log_lambda_range, points=lambda_points)
    tab_vel_num = TabulatedFunction(vel_num_eval, arch, FT;
                                    range=log_lambda_range, points=lambda_points)
    tab_evap = TabulatedFunction(evap_eval, arch, FT;
                                 range=log_lambda_range, points=lambda_points)

    return RainProperties(
        rain.maximum_mean_diameter,
        rain.fall_speed_coefficient,
        rain.fall_speed_exponent,
        rain.shape_parameter,
        tab_vel_num,
        tab_vel_mass,
        tab_evap
    )
end
