#####
##### Fortran P3 Lookup Table Reader
#####
##### Reads ASCII lookup table files from the Fortran P3 implementation
##### and constructs Julia table objects for use in PredictedParticlePropertiesMicrophysics.
#####

export read_fortran_lookup_tables

using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Utils: TabulatedFunction

#####
##### Main entry point
#####

"""
$(TYPEDSIGNATURES)

Read Fortran P3 lookup tables from ASCII files and construct a complete
`PredictedParticlePropertiesMicrophysics` with tabulated ice integrals.

If the table files are not found in `directory`, they are automatically
downloaded from the P3-microphysics GitHub repository (approximately 28 MB
compressed). This only happens once; subsequent calls read from disk.

Auto-detects 2-moment vs 3-moment ice from file presence in `directory`.
When `three_moment_ice=nothing` (default), prefers 3-moment if available.

Rain 1D tables (velocity, evaporation) are generated from Julia quadrature
since they are not included in the Fortran ASCII files.

# Arguments

- `directory`: Path to directory containing Fortran table files
  (`p3_lookupTable_1.dat-v6.9-3momI`, `p3_lookupTable_1.dat-v6.9-2momI`,
   `p3_lookupTable_3.dat-v1.4`).

# Keyword Arguments

- `FT`: Float type (default `Float64`)
- `arch`: Architecture for GPU transfer (default `CPU()`)
- `three_moment_ice`: Force 2-moment (`false`) or 3-moment (`true`),
  or auto-detect (`nothing`)
"""
function read_fortran_lookup_tables(directory::AbstractString;
                                    FT::Type{<:AbstractFloat} = Float64,
                                    arch = CPU(),
                                    three_moment_ice::Union{Bool, Nothing} = nothing,
                                    water_density = 1000,
                                    precipitation_boundary_condition = nothing,
                                    aerosol = nothing,
                                    cloud = nothing,
                                    process_rates = nothing,
                                    warm_rain_scheme = KhairoutdinovKogan2000())

    # Auto-detect 2momI vs 3momI from file presence
    file_3momI = joinpath(directory, "p3_lookupTable_1.dat-v6.9-3momI")
    file_2momI = joinpath(directory, "p3_lookupTable_1.dat-v6.9-2momI")
    file_table3 = joinpath(directory, "p3_lookupTable_3.dat-v1.4")

    has_3momI = isfile(file_3momI)
    has_2momI = isfile(file_2momI)

    three_moment = if isnothing(three_moment_ice)
        has_3momI
    else
        three_moment_ice
    end

    if three_moment && !has_3momI
        error("3momI table not found: $file_3momI")
    end
    if !three_moment && !has_2momI
        error("2momI table not found: $file_2momI")
    end

    table1_file = three_moment ? file_3momI : file_2momI

    # Parse Table 1 (ice integrals + embedded rain-ice collection)
    table1_fields, table2_fields = parse_fortran_table_1(table1_file, three_moment, FT)

    # Parse Table 3 (3momI only)
    table3_fields = if three_moment && isfile(file_table3)
        parse_fortran_table_3(file_table3, FT)
    else
        # Fortran hard-stops when Table 3 is missing in 3-moment mode.
        # Match that behavior — a silent fallback to 2-moment μ lookup would give
        # different results than the 3-moment μ diagnostic.
        three_moment && error("3-moment mode requested but Table 3 file not found: $file_table3. " *
                              "Ensure the file is present and unzipped.")
        nothing
    end

    # Build TabulatedFunction objects
    ice_tables_5d = build_table_1_functions(table1_fields, three_moment, FT, arch)
    rain_ice_tables = build_table_2_functions(table2_fields, three_moment, FT, arch)
    table3_objs = if !isnothing(table3_fields)
        build_table_3_functions(table3_fields, FT, arch)
    else
        nothing
    end

    # Assemble P3 lookup table structs
    ice_integrals_tab, rain_ice_collection_tab, three_moment_shape_tab = assemble_lookup_tables(
        ice_tables_5d, rain_ice_tables, table3_objs, three_moment)

    # Build IceProperties with tabulated fields
    ice = build_ice_properties_from_tables(
        ice_tables_5d, rain_ice_tables, table3_objs,
        ice_integrals_tab, rain_ice_collection_tab, three_moment_shape_tab, three_moment, FT)

    # Generate rain 1D tables from Julia quadrature
    rain_base = RainProperties(FT)
    rain = tabulate_rain_from_quadrature(rain_base, arch, FT)

    # Construct full scheme
    cloud = isnothing(cloud) ? CloudDropletProperties(FT) : cloud
    input_process_rates = isnothing(process_rates) ? ProcessRateParameters(FT) : process_rates

    return PredictedParticlePropertiesMicrophysics(
        FT(water_density),
        FT(1e-14),   # minimum_mass_mixing_ratio
        FT(1e-16),   # minimum_number_mixing_ratio
        ice,
        rain,
        cloud,
        input_process_rates,
        precipitation_boundary_condition,
        aerosol,
        warm_rain_scheme
    )
end

#####
##### Build Table 1 (5D) TabulatedFunction objects
#####

function build_table_1_functions(table1_fields::Dict, three_moment::Bool,
                                 FT::Type, arch)
    ranges = fortran_table_1_ranges(FT)

    result = Dict{Symbol, FortranTabulatedFunction5D}()
    for (name, data) in table1_fields
        # For 2momI, the mu dimension has size 1. The TabulatedFunction
        # still needs 5D, which works because inv_delta for that axis is 0.
        table = make_fortran_tabulated_function(data, ranges, arch)
        result[name] = FortranTabulatedFunction5D(table)
    end
    return result
end

#####
##### Build Table 2 (6D) TabulatedFunction objects
#####

function build_table_2_functions(table2_fields::Dict, three_moment::Bool,
                                 FT::Type, arch)
    ranges = fortran_table_2_ranges(FT)

    result = Dict{Symbol, FortranTabulatedFunction6D}()
    for (name, data) in table2_fields
        table = make_fortran_tabulated_function(data, ranges, arch)
        result[name] = FortranTabulatedFunction6D(table)
    end
    return result
end

#####
##### Build Table 3 (5D) TabulatedFunction objects
#####

function build_table_3_functions(table3_fields::Dict, FT::Type, arch)
    ranges = fortran_table_3_ranges(FT)

    result = Dict{Symbol, FortranTabulatedFunction3}()
    for (name, data) in table3_fields
        table = make_fortran_tabulated_function(data, ranges, arch)
        result[name] = FortranTabulatedFunction3(table)
    end
    return result
end

#####
##### Assemble P3 lookup table structs
#####

function assemble_lookup_tables(ice_5d, rain_ice, table3_objs, three_moment)
    # P3IceIntegralsTable: groups of ice integrals
    fall_speed = (
        number_weighted = ice_5d[:number_weighted],
        mass_weighted = ice_5d[:mass_weighted],
        reflectivity_weighted = three_moment ? ice_5d[:reflectivity_weighted] : nothing
    )

    deposition = (
        ventilation = ice_5d[:ventilation],
        ventilation_enhanced = ice_5d[:ventilation_enhanced],
        small_ice_ventilation_constant = ice_5d[:small_ice_ventilation_constant],
        small_ice_ventilation_reynolds = ice_5d[:small_ice_ventilation_reynolds],
        large_ice_ventilation_constant = ice_5d[:large_ice_ventilation_constant],
        large_ice_ventilation_reynolds = ice_5d[:large_ice_ventilation_reynolds],
    )

    bulk_properties = (
        effective_radius = ice_5d[:effective_radius],
        mean_diameter = ice_5d[:mean_diameter],
        mean_density = ice_5d[:mean_density],
        reflectivity = ice_5d[:reflectivity],
        shedding = ice_5d[:shedding],
    )

    collection = (
        aggregation = ice_5d[:aggregation],
        rain_collection = ice_5d[:rain_collection],
        cloud_aerosol_collection = ice_5d[:cloud_aerosol_collection],
        ice_aerosol_collection = ice_5d[:ice_aerosol_collection],
    )

    sixth_moment = if three_moment
        # The Fortran table file stores D ≤ D_crit filtered melt Z integrals
        # (f1pr32/f1pr33) in the m6_melt1/m6_melt2 columns. The non-liquid-fraction
        # zimlt path in Fortran reuses deposition tables and is dead code (log_full3mom
        # = .false.). We set melt_all1/melt_all2 to the same file values; the all-D
        # distinction only applies in the Julia-native quadrature path.
        (rime = ice_5d[:m6_rime],
         deposition = ice_5d[:m6_deposition],
         deposition1 = ice_5d[:m6_deposition1],
         melt1 = ice_5d[:m6_melt1],
         melt2 = ice_5d[:m6_melt2],
         melt_all1 = ice_5d[:m6_melt1],
         melt_all2 = ice_5d[:m6_melt2],
         aggregation = ice_5d[:m6_aggregation],
         shedding = ice_5d[:m6_shedding],
         sublimation = ice_5d[:m6_sublimation],
         sublimation1 = ice_5d[:m6_sublimation1])
    else
        nothing
    end

    lambda_limiter = (
        small_q = ice_5d[:small_q],
        large_q = ice_5d[:large_q],
    )

    ice_rain = (
        number = rain_ice[:rain_number],
        mass = rain_ice[:rain_mass],
        sixth_moment = three_moment ? rain_ice[:rain_sixth_moment] : nothing
    )

    ice_integrals_tab = P3IceIntegralsTable(fall_speed, deposition, bulk_properties,
                                              collection, sixth_moment, lambda_limiter, ice_rain)

    rain_ice_collection_tab = P3RainIceCollectionTable(
        rain_ice[:rain_mass],
        rain_ice[:rain_number],
        three_moment ? rain_ice[:rain_sixth_moment] : nothing
    )

    three_moment_shape_tab = if !isnothing(table3_objs)
        P3ThreeMomentShapeTable(
            table3_objs[:shape],
            nothing,  # slope is not in Table 3 file
            table3_objs[:mean_density]
        )
    else
        nothing
    end

    return ice_integrals_tab, rain_ice_collection_tab, three_moment_shape_tab
end

#####
##### Build IceProperties from Fortran tables
#####

function build_ice_properties_from_tables(ice_5d, rain_ice, table3_objs,
                                          ice_integrals_tab, rain_ice_collection_tab, three_moment_shape_tab,
                                          three_moment, FT)
    # Start from default IceProperties for physical constants
    ice_base = IceProperties(FT)

    # Build sub-structs with tabulated fields replacing integral placeholders
    fall_speed = IceFallSpeed(
        ice_base.fall_speed.reference_air_density,
        ice_5d[:number_weighted],
        ice_5d[:mass_weighted],
        three_moment ? ice_5d[:reflectivity_weighted] : nothing
    )

    deposition = IceDeposition(
        ice_5d[:ventilation],
        ice_5d[:ventilation_enhanced],
        ice_5d[:small_ice_ventilation_constant],
        ice_5d[:small_ice_ventilation_reynolds],
        ice_5d[:large_ice_ventilation_constant],
        ice_5d[:large_ice_ventilation_reynolds]
    )

    bulk_properties = IceBulkProperties(
        ice_base.bulk_properties.maximum_mean_diameter,
        ice_base.bulk_properties.minimum_mean_diameter,
        ice_5d[:effective_radius],
        ice_5d[:mean_diameter],
        ice_5d[:mean_density],
        ice_5d[:reflectivity],
        ice_5d[:slope_parameter],
        ice_5d[:shape_parameter],
        ice_5d[:shedding]
    )

    collection = IceCollection(
        ice_base.collection.ice_rain_collection_efficiency,
        ice_5d[:aggregation],
        ice_5d[:rain_collection],
        ice_5d[:cloud_aerosol_collection],
        ice_5d[:ice_aerosol_collection]
    )

    sixth_moment = if three_moment
        # Fortran file stores D ≤ D_crit melt integrals (f1pr32/f1pr33).
        # Set melt_all1/melt_all2 to the same values (all-D distinction is Julia-native only).
        IceSixthMoment(
            ice_5d[:m6_rime],
            ice_5d[:m6_deposition],
            ice_5d[:m6_deposition1],
            ice_5d[:m6_melt1],
            ice_5d[:m6_melt2],
            ice_5d[:m6_melt1],   # melt_all1 = same as melt1 (all-D distinction is quadrature-only)
            ice_5d[:m6_melt2],   # melt_all2 = same as melt2 (all-D distinction is quadrature-only)
            ice_5d[:m6_shedding],
            ice_5d[:m6_aggregation],
            ice_5d[:m6_sublimation],
            ice_5d[:m6_sublimation1]
        )
    else
        IceSixthMoment(nothing, nothing, nothing, nothing,
                        nothing, nothing, nothing, nothing,
                        nothing, nothing, nothing)
    end

    lambda_limiter = IceLambdaLimiter(
        ice_5d[:small_q],
        ice_5d[:large_q]
    )

    ice_rain_coll = IceRainCollection(
        rain_ice[:rain_mass],
        rain_ice[:rain_number],
        three_moment ? rain_ice[:rain_sixth_moment] : nothing
    )

    lookup_tables = P3LookupTables(ice_integrals_tab, rain_ice_collection_tab, three_moment_shape_tab)

    return IceProperties(
        ice_base.minimum_rime_density,
        ice_base.maximum_rime_density,
        ice_base.maximum_shape_parameter,
        ice_base.minimum_reflectivity,
        fall_speed,
        deposition,
        bulk_properties,
        collection,
        sixth_moment,
        lambda_limiter,
        ice_rain_coll;
        lookup_tables
    )
end

#####
##### Rain tabulation from Julia quadrature
#####
##### Rain 1D tables are NOT in the Fortran files. We generate them
##### from Julia quadrature (extracted from tabulation.jl).
#####

function tabulate_rain_from_quadrature(rain::RainProperties, arch=CPU(),
                                       FT::Type{<:AbstractFloat} = Float64;
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
