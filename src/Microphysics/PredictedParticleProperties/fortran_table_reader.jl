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
##### Fortran table dimension constants
#####

const FORTRAN_N_QNORM = 50
const FORTRAN_N_FR = 4
const FORTRAN_N_FL = 4
const FORTRAN_N_RHOR = 5
const FORTRAN_N_ZNORM = 11
const FORTRAN_N_DRSCALE = 30
const FORTRAN_N_ZNORM_TABLE3 = 80

#####
##### Axis range constants (matching Fortran P3 grid generation)
#####
##### Table 1 mass axis: log_m = (i * 0.1 * log10(800)) - 18  for i = 11..60
##### Table 2 rain axis: log_lambda_r = -log10(1.24^j * 10e-6) for j = 1..30
##### Table 3 z-norm axis: log_zq = j * log10(2.1) - 23 for j = 1..80
#####

const LOG_MASS_MIN = 11 * 0.1 * log10(800) - 18
const LOG_MASS_MAX = 60 * 0.1 * log10(800) - 18
const LOG_LAMBDA_R_MIN = -log10(1.24^30 * 10e-6)
const LOG_LAMBDA_R_MAX = -log10(1.24^1 * 10e-6)
const LOG_ZQ_MIN = log10(2.1) - 23
const LOG_ZQ_MAX = 80 * log10(2.1) - 23

#####
##### Table 3 q-norm axis (same mass axis as Table 1)
#####

const LOG_QNORM_MIN = LOG_MASS_MIN
const LOG_QNORM_MAX = LOG_MASS_MAX

#####
##### Main entry point
#####

"""
$(TYPEDSIGNATURES)

Read Fortran P3 lookup tables from ASCII files and construct a complete
`PredictedParticlePropertiesMicrophysics` with tabulated ice integrals.

Auto-detects 2-moment vs 3-moment ice from file presence in `directory`.
When `three_moment_ice=nothing` (default), prefers 3-moment if available.

Rain 1D tables (velocity, evaporation) are generated from Julia quadrature
since they are not included in the Fortran ASCII files.

# Arguments

- `directory`: Path to directory containing Fortran table files
  (`p3_lookupTable_1.dat-v6.9-3momI`, `p3_lookupTable_1.dat-v6.9-2momI`,
   `p3_lookupTable_3.dat-v1.4`)

# Keyword Arguments

- `FT`: Float type (default `Float64`)
- `arch`: Architecture for GPU transfer (default `CPU()`)
- `three_moment_ice`: Force 2-moment (`false`) or 3-moment (`true`),
  or auto-detect (`nothing`)
"""
function read_fortran_lookup_tables(directory::AbstractString;
                                    FT::Type{<:AbstractFloat} = Float64,
                                    arch = CPU(),
                                    three_moment_ice::Union{Bool, Nothing} = nothing)

    # Auto-detect 2momI vs 3momI from file presence
    file_3momI = joinpath(directory, "p3_lookupTable_1.dat-v6.9-3momI")
    file_2momI = joinpath(directory, "p3_lookupTable_1.dat-v6.9-2momI")
    file_table3 = joinpath(directory, "p3_lookupTable_3.dat-v1.4")

    has_3momI = isfile(file_3momI)
    has_2momI = isfile(file_2momI)

    if isnothing(three_moment_ice)
        three_moment = has_3momI
    else
        three_moment = three_moment_ice
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

    # Assemble P3LookupTable structs
    table_1, table_2, table_3 = assemble_lookup_tables(
        ice_tables_5d, rain_ice_tables, table3_objs, three_moment)

    # Build IceProperties with tabulated fields
    ice = build_ice_properties_from_tables(
        ice_tables_5d, rain_ice_tables, table3_objs,
        table_1, table_2, table_3, three_moment, FT)

    # Generate rain 1D tables from Julia quadrature
    rain_base = RainProperties(FT)
    rain = tabulate_rain_from_quadrature(rain_base, arch, FT)

    # Construct full scheme
    cloud = CloudDropletProperties(FT)
    process_rates = ProcessRateParameters(FT)

    return PredictedParticlePropertiesMicrophysics(
        FT(1000),    # water_density
        FT(1e-14),   # minimum_mass_mixing_ratio
        FT(1e-16),   # minimum_number_mixing_ratio
        ice,
        rain,
        cloud,
        process_rates,
        nothing      # precipitation_boundary_condition
    )
end

#####
##### Fortran Table 1 parser
#####

"""
Parse the Fortran Table 1 ASCII file.

Returns two dictionaries:
- `table1_fields`: Dict of Symbol => Array{FT,5} for ice integrals
  with axes (i_Qnorm, i_Fr, i_Fl, i_rhor, i_Znorm)
- `table2_fields`: Dict of Symbol => Array{FT,6} for rain-ice collection
  with axes (i_Qnorm, i_Drscale_reversed, i_Fr, i_Fl, i_rhor, i_Znorm)
"""
function parse_fortran_table_1(filepath::AbstractString, three_moment::Bool, FT::Type)
    lines = readlines(filepath)

    # Number of shape parameter points (mu axis)
    n_mu = three_moment ? FORTRAN_N_ZNORM : 1
    n_q = FORTRAN_N_QNORM
    n_fr = FORTRAN_N_FR
    n_fl = FORTRAN_N_FL
    n_rhor = FORTRAN_N_RHOR
    n_dr = FORTRAN_N_DRSCALE

    # Column names for ice data
    col_names_3momI = [
        :number_weighted, :mass_weighted, :aggregation, :rain_collection,
        :ventilation, :effective_radius, :small_q, :large_q,
        :reflectivity, :ventilation_enhanced, :mean_diameter, :mean_density,
        :reflectivity_weighted, :_skip_lambda_i, :_skip_mu_i,
        :small_ice_ventilation_constant, :small_ice_ventilation_reynolds,
        :large_ice_ventilation_constant, :large_ice_ventilation_reynolds,
        :shedding,
        :m6_rime, :m6_deposition, :m6_deposition1,
        :m6_melt1, :m6_melt2, :m6_aggregation, :m6_shedding,
        :m6_sublimation, :m6_sublimation1,
        :cloud_aerosol_collection, :ice_aerosol_collection
    ]

    col_names_2momI = [
        :number_weighted, :mass_weighted, :aggregation, :rain_collection,
        :ventilation, :effective_radius, :small_q, :large_q,
        :reflectivity, :ventilation_enhanced, :mean_diameter, :mean_density,
        :_skip_lambda_i, :_skip_mu_i,
        :small_ice_ventilation_constant, :small_ice_ventilation_reynolds,
        :large_ice_ventilation_constant, :large_ice_ventilation_reynolds,
        :shedding,
        :cloud_aerosol_collection, :ice_aerosol_collection
    ]

    col_names = three_moment ? col_names_3momI : col_names_2momI

    # Allocate arrays for ice integrals: (Qnorm, Fr, Fl, rhor, mu)
    table1_fields = Dict{Symbol, Array{FT, 5}}()
    for name in col_names
        startswith(String(name), "_skip") && continue
        table1_fields[name] = zeros(FT, n_q, n_fr, n_fl, n_rhor, n_mu)
    end

    # Allocate arrays for rain-ice collection: (Qnorm, Drscale, Fr, Fl, rhor, mu)
    rain_names_3momI = [:rain_number, :rain_mass, :rain_sixth_moment]
    rain_names_2momI = [:rain_number, :rain_mass]
    rain_names = three_moment ? rain_names_3momI : rain_names_2momI

    table2_fields = Dict{Symbol, Array{FT, 6}}()
    for name in rain_names
        table2_fields[name] = zeros(FT, n_q, n_dr, n_fr, n_fl, n_rhor, n_mu)
    end

    # Parse data lines (skip header line 1 and blank line 2)
    line_idx = 3  # 1-indexed; line 3 is first data line
    n_ice_idx = three_moment ? 5 : 4
    n_rain_idx = 4

    # Loop nesting order:
    # 3momI: i_Znorm(1..11) -> i_rhor(1..5) -> i_Fr(1..4) -> i_Fl(1..4) -> {ice, rain}
    # 2momI: i_rhor(1..5) -> i_Fr(1..4) -> i_Fl(1..4) -> {ice, rain}
    for i_mu in 1:n_mu
        for i_rhor in 1:n_rhor
            for i_fr in 1:n_fr
                for i_fl in 1:n_fl
                    # Read 50 ice rows
                    for i_q in 1:n_q
                        vals = parse_fortran_line(lines[line_idx])
                        line_idx += 1
                        # Skip index columns, read data columns
                        data_offset = n_ice_idx
                        for (col_idx, name) in enumerate(col_names)
                            v = vals[data_offset + col_idx]
                            if !startswith(String(name), "_skip")
                                table1_fields[name][i_q, i_fr, i_fl, i_rhor, i_mu] = FT(v)
                            end
                        end
                    end

                    # Read 50 * 30 = 1500 rain-ice rows
                    for i_q in 1:n_q
                        for i_dr in 1:n_dr
                            vals = parse_fortran_line(lines[line_idx])
                            line_idx += 1
                            data_offset = n_rain_idx
                            # CRITICAL: reverse the Drscale axis
                            # Fortran i_Drscale=1 -> largest lambda_r -> Julia index n_dr
                            # Fortran i_Drscale=30 -> smallest lambda_r -> Julia index 1
                            j_dr = n_dr - i_dr + 1
                            for (col_idx, name) in enumerate(rain_names)
                                v = vals[data_offset + col_idx]
                                # Rain number and mass stored as log10 in file
                                # Sixth moment is NOT log10
                                table2_fields[name][i_q, j_dr, i_fr, i_fl, i_rhor, i_mu] = FT(v)
                            end
                        end
                    end
                end
            end
        end
    end

    return table1_fields, table2_fields
end

#####
##### Fortran Table 3 parser
#####

"""
Parse the Fortran Table 3 ASCII file.

Returns a Dict of Symbol => Array{FT,5} for Table 3 fields
with axes (i_Znorm, i_rhor, i_Qnorm, i_Fr, i_Fl).
"""
function parse_fortran_table_3(filepath::AbstractString, FT::Type)
    lines = readlines(filepath)

    n_z = FORTRAN_N_ZNORM_TABLE3
    n_rhor = FORTRAN_N_RHOR
    n_q = FORTRAN_N_QNORM
    n_fr = FORTRAN_N_FR
    n_fl = FORTRAN_N_FL

    # 4 data columns after 5 index integers; only first 2 are used
    table3_fields = Dict{Symbol, Array{FT, 5}}()
    table3_fields[:shape] = zeros(FT, n_z, n_rhor, n_q, n_fr, n_fl)
    table3_fields[:mean_density] = zeros(FT, n_z, n_rhor, n_q, n_fr, n_fl)

    line_idx = 3  # skip header and blank line
    n_idx = 5

    # Loop nesting: i_Znorm(1..80) -> i_rhor(1..5) -> i_Fr(1..4) -> i_Fl(1..4) -> i_Qnorm(1..50)
    for i_z in 1:n_z
        for i_rhor in 1:n_rhor
            for i_fr in 1:n_fr
                for i_fl in 1:n_fl
                    for i_q in 1:n_q
                        vals = parse_fortran_line(lines[line_idx])
                        line_idx += 1
                        table3_fields[:shape][i_z, i_rhor, i_q, i_fr, i_fl] = FT(vals[n_idx + 1])
                        table3_fields[:mean_density][i_z, i_rhor, i_q, i_fr, i_fl] = FT(vals[n_idx + 2])
                    end
                end
            end
        end
    end

    return table3_fields
end

#####
##### Line parsing
#####

"""
Parse a single Fortran-formatted data line into a vector of Float64.
Handles Fortran-style scientific notation (e.g., `0.12345E+06`).
Integer fields are also parsed as Float64.
"""
function parse_fortran_line(line::AbstractString)
    tokens = split(strip(line))
    return [parse(Float64, t) for t in tokens]
end

#####
##### TabulatedFunction construction from raw arrays
#####

"""
Build a TabulatedFunction{N} directly from a pre-computed data array and axis ranges.
"""
function make_fortran_tabulated_function(data::Array{FT, N}, ranges, arch) where {FT, N}
    points = size(data)
    inv_delta = map(ranges, points) do (lo, hi), n
        ifelse(n == 1, zero(FT), FT(1) / ((FT(hi) - FT(lo)) / (n - 1)))
    end
    gpu_data = on_architecture(arch, data)
    return TabulatedFunction{N, Nothing, typeof(gpu_data), typeof(ranges), typeof(inv_delta)}(
        nothing, gpu_data, ranges, inv_delta)
end

#####
##### Build Table 1 (5D) TabulatedFunction objects
#####

function fortran_table_1_ranges(FT)
    # Axes: (log_mass, Fr, Fl, rho_index, mu)
    # Fr in [0, 1] with 4 points, Fl in [0, 1] with 4 points
    # rho_index in [1, 5] with 5 points (piecewise transform applied by wrapper)
    # mu in [0, 20] with 11 points (3momI), or trivial for 2momI
    return (
        (FT(LOG_MASS_MIN), FT(LOG_MASS_MAX)),
        (FT(0), FT(1)),
        (FT(0), FT(1)),
        (FT(1), FT(5)),
        (FT(0), FT(20))
    )
end

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

function fortran_table_2_ranges(FT)
    # Axes: (log_mass, log_lambda_r, Fr, Fl, rho_index, mu)
    return (
        (FT(LOG_MASS_MIN), FT(LOG_MASS_MAX)),
        (FT(LOG_LAMBDA_R_MIN), FT(LOG_LAMBDA_R_MAX)),
        (FT(0), FT(1)),
        (FT(0), FT(1)),
        (FT(1), FT(5)),
        (FT(0), FT(20))
    )
end

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

function fortran_table_3_ranges(FT)
    # Axes: (log_znorm, rho_index, log_qnorm, Fr, Fl)
    return (
        (FT(LOG_ZQ_MIN), FT(LOG_ZQ_MAX)),
        (FT(1), FT(5)),
        (FT(LOG_QNORM_MIN), FT(LOG_QNORM_MAX)),
        (FT(0), FT(1)),
        (FT(0), FT(1))
    )
end

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
##### Assemble P3LookupTable structs
#####

function assemble_lookup_tables(ice_5d, rain_ice, table3_objs, three_moment)
    # P3LookupTable1: groups of ice integrals
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
        (rime = ice_5d[:m6_rime],
         deposition = ice_5d[:m6_deposition],
         deposition1 = ice_5d[:m6_deposition1],
         melt1 = ice_5d[:m6_melt1],
         melt2 = ice_5d[:m6_melt2],
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

    table_1 = P3LookupTable1(fall_speed, deposition, bulk_properties,
                              collection, sixth_moment, lambda_limiter, ice_rain)

    table_2 = P3LookupTable2(
        rain_ice[:rain_mass],
        rain_ice[:rain_number],
        three_moment ? rain_ice[:rain_sixth_moment] : nothing
    )

    table_3 = if !isnothing(table3_objs)
        P3LookupTable3(
            table3_objs[:shape],
            nothing,  # slope is not in Table 3 file
            table3_objs[:mean_density]
        )
    else
        nothing
    end

    return table_1, table_2, table_3
end

#####
##### Build IceProperties from Fortran tables
#####

function build_ice_properties_from_tables(ice_5d, rain_ice, table3_objs,
                                          table_1, table_2, table_3,
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
        ice_base.deposition.thermal_conductivity,
        ice_base.deposition.vapor_diffusivity,
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
        ice_base.bulk_properties.slope,     # not tabulated from Fortran
        ice_base.bulk_properties.shape,     # not tabulated from Fortran
        ice_5d[:shedding]
    )

    collection = IceCollection(
        ice_base.collection.ice_cloud_collection_efficiency,
        ice_base.collection.ice_rain_collection_efficiency,
        ice_5d[:aggregation],
        ice_5d[:rain_collection],
        ice_5d[:cloud_aerosol_collection],
        ice_5d[:ice_aerosol_collection]
    )

    sixth_moment = if three_moment
        IceSixthMoment(
            ice_5d[:m6_rime],
            ice_5d[:m6_deposition],
            ice_5d[:m6_deposition1],
            ice_5d[:m6_melt1],
            ice_5d[:m6_melt2],
            ice_5d[:m6_shedding],
            ice_5d[:m6_aggregation],
            ice_5d[:m6_sublimation],
            ice_5d[:m6_sublimation1]
        )
    else
        IceSixthMoment(nothing, nothing, nothing, nothing,
                        nothing, nothing, nothing, nothing, nothing)
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

    lookup_tables = P3LookupTables(table_1, table_2, table_3)

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
