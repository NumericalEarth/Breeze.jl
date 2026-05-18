#####
##### Fortran P3 Lookup Table Reader
#####
##### Reads ASCII lookup table files from the Fortran P3 implementation
##### and constructs Julia table objects for use in PredictedParticlePropertiesMicrophysics.
#####

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
##### Line parsing
#####

"""
$(TYPEDSIGNATURES)

Parse a single Fortran-formatted data line into a vector of Float64.
Handles Fortran-style scientific notation (e.g., `0.12345E+06`).
Integer fields are also parsed as Float64.
"""
function parse_fortran_line(line::AbstractString)
    tokens = split(strip(line))
    return [parse(Float64, t) for t in tokens]
end

#####
##### Fortran Table 1 parser
#####

"""
$(TYPEDSIGNATURES)

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
        :reflectivity_weighted, :slope_parameter, :shape_parameter,
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
        :slope_parameter, :shape_parameter,
        :small_ice_ventilation_constant, :small_ice_ventilation_reynolds,
        :large_ice_ventilation_constant, :large_ice_ventilation_reynolds,
        :shedding,
        :cloud_aerosol_collection, :ice_aerosol_collection
    ]

    col_names = three_moment ? col_names_3momI : col_names_2momI

    # Allocate arrays for ice integrals: (Qnorm, Fr, Fl, rhor, mu)
    table1_fields = Dict{Symbol, Array{FT, 5}}()
    for name in col_names
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
                            table1_fields[name][i_q, i_fr, i_fl, i_rhor, i_mu] = FT(v)
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
$(TYPEDSIGNATURES)

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
##### TabulatedFunction construction from raw arrays
#####

"""
$(TYPEDSIGNATURES)

Build a TabulatedFunction{N} directly from a pre-computed data array and axis ranges.
"""
function make_fortran_tabulated_function(data::Array{FT, N}, ranges, arch) where {FT, N}
    points = size(data)
    inv_delta = map(ranges, points) do (lo, hi), n
        ifelse(n == 1, zero(FT), FT(1) / ((FT(hi) - FT(lo)) / (n - 1)))
    end
    gpu_data = on_architecture(arch, data)
    return make_tabulated_function(Val(N), gpu_data, ranges, inv_delta)
end

# 1D–5D fall through to Oceananigans' parametric TabulatedFunction, which owns the
# corresponding call methods. 6D is Breeze-owned (commit 1f0234a moved off
# TabulatedFunction{6} to eliminate type piracy), so the 6D path must construct
# our owned struct — otherwise the resulting object has no call method and the
# GPU compiler emits jl_f_throw_methoderror deep inside the rain-ice collection
# lookup chain.
@inline function make_tabulated_function(::Val{N}, gpu_data, ranges, inv_delta) where {N}
    return TabulatedFunction{N, Nothing, typeof(gpu_data), typeof(ranges), typeof(inv_delta)}(
        nothing, gpu_data, ranges, inv_delta)
end

@inline make_tabulated_function(::Val{6}, gpu_data, ranges, inv_delta) =
    TabulatedFunction6D{Nothing, typeof(gpu_data), typeof(ranges), typeof(inv_delta)}(
        nothing, gpu_data, ranges, inv_delta)

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
