#####
##### P3 lookup table format
#####
##### Layout of the P3 ice ASCII lookup tables and the parser that reads them.
##### The tables ship with the reference P3 distribution as whitespace-separated
##### text; nothing about the format depends on the language that wrote them.
##### A single file holds two blocks:
#####
#####   Table 1: ice integrals,          axes (Q̄, Fᶠ, Fˡ, ρᶠ index, μⁱ)
#####   Table 2: rain-ice collection,    axes (Q̄, λ_r, Fᶠ, Fˡ, ρᶠ index, μⁱ)
#####
##### The rime-density axis is stored as an index 1..5 over the non-uniform grid
##### {50, 250, 450, 650, 900} kg/m³; `RimeDensityIndexedTable5D`/`6D` apply the
##### coordinate transform on evaluation.
#####

using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Utils: TabulatedFunction

#####
##### Table dimension constants
#####

const N_NORMALIZED_MASS = 50
const N_RIME_FRACTION = 4
const N_LIQUID_FRACTION = 4
const N_RIME_DENSITY = 5
const N_RAIN_SLOPE = 30

#####
##### Axis range constants (matching the reference P3 grid generation)
#####
##### Table 1 mass axis: log_m = (i * 0.1 * log10(800)) - 18  for i = 11..60
##### Table 2 rain axis: log_lambda_r = -log10(1.24^j * 10e-6) for j = 1..30
#####

const LOG_MASS_MIN = 11 * 0.1 * log10(800) - 18
const LOG_MASS_MAX = 60 * 0.1 * log10(800) - 18
const LOG_LAMBDA_R_MIN = -log10(1.24^30 * 10e-6)
const LOG_LAMBDA_R_MAX = -log10(1.24^1 * 10e-6)

#####
##### Line parsing
#####

"""
$(TYPEDSIGNATURES)

Parse one whitespace-separated data line of a table file into a vector of
Float64. Handles the `E`-exponent scientific notation the files are written in
(e.g. `0.12345E+06`); integer fields are parsed as Float64 too.
"""
function parse_table_line(line::AbstractString)
    tokens = split(strip(line))
    return [parse(Float64, t) for t in tokens]
end

#####
##### Table file parser
#####

"""
$(TYPEDSIGNATURES)

Parse the P3 ice ASCII table file, which carries both table blocks.

Returns two dictionaries:
- `table1_fields`: Dict of Symbol => Array{FT,5} for ice integrals
  with axes (i_Qnorm, i_Fr, i_Fl, i_rhor, i_mu)
- `table2_fields`: Dict of Symbol => Array{FT,6} for rain-ice collection
  with axes (i_Qnorm, i_Drscale_reversed, i_Fr, i_Fl, i_rhor, i_mu)
"""
function parse_lookup_table_file(filepath::AbstractString, FT::Type)
    lines = readlines(filepath)

    # Number of shape parameter points (mu axis)
    n_mu = 1
    n_q = N_NORMALIZED_MASS
    n_fr = N_RIME_FRACTION
    n_fl = N_LIQUID_FRACTION
    n_rhor = N_RIME_DENSITY
    n_dr = N_RAIN_SLOPE

    # Column names for ice data.
    # Column 4 (`cloud_collection`) is Fortran `f1pr04`, the ice-cloud-water
    # sweep-out integral ∫ V(D) A(D) N'(D) dD. Ice-*rain* collection is not in
    # the 5D ice block: it needs the rain slope parameter as an extra coordinate
    # and lives in the 6D rain-ice block embedded later in the same Fortran
    # Lookup Table 1 file (`rain_number` / `rain_mass`).
    col_names = [
        :number_weighted, :mass_weighted, :aggregation, :cloud_collection,
        :ventilation, :effective_radius, :small_q, :large_q,
        :reflectivity, :ventilation_enhanced, :mean_diameter, :mean_density,
        :slope_parameter, :shape_parameter,
        :small_ice_ventilation_constant, :small_ice_ventilation_reynolds,
        :large_ice_ventilation_constant, :large_ice_ventilation_reynolds,
        :shedding,
        :cloud_aerosol_collection, :ice_aerosol_collection
    ]

    # Allocate arrays for ice integrals: (Qnorm, Fr, Fl, rhor, mu)
    table1_fields = Dict{Symbol, Array{FT, 5}}()
    for name in col_names
        table1_fields[name] = zeros(FT, n_q, n_fr, n_fl, n_rhor, n_mu)
    end

    # Allocate arrays for rain-ice collection: (Qnorm, Drscale, Fr, Fl, rhor, mu)
    rain_names = [:rain_number, :rain_mass]

    table2_fields = Dict{Symbol, Array{FT, 6}}()
    for name in rain_names
        table2_fields[name] = zeros(FT, n_q, n_dr, n_fr, n_fl, n_rhor, n_mu)
    end

    # Parse data lines (skip header line 1 and blank line 2)
    line_idx = 3  # 1-indexed; line 3 is first data line
    n_ice_idx = 4
    n_rain_idx = 4

    # Loop nesting order:
    # i_rhor(1..5) -> i_Fr(1..4) -> i_Fl(1..4) -> {ice, rain}
    for i_mu in 1:n_mu
        for i_rhor in 1:n_rhor
            for i_fr in 1:n_fr
                for i_fl in 1:n_fl
                    # Read 50 ice rows
                    for i_q in 1:n_q
                        vals = parse_table_line(lines[line_idx])
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
                            vals = parse_table_line(lines[line_idx])
                            line_idx += 1
                            data_offset = n_rain_idx
                            # CRITICAL: reverse the Drscale axis
                            # Fortran i_Drscale=1 -> largest lambda_r -> Julia index n_dr
                            # Fortran i_Drscale=30 -> smallest lambda_r -> Julia index 1
                            j_dr = n_dr - i_dr + 1
                            for (col_idx, name) in enumerate(rain_names)
                                v = vals[data_offset + col_idx]
                                # Rain number and mass stored as log10 in file
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
##### TabulatedFunction construction from raw arrays
#####

"""
$(TYPEDSIGNATURES)

Build a TabulatedFunction{N} directly from a pre-computed data array and axis ranges.
"""
function make_lookup_table(data::Array{FT, N}, ranges, arch) where {FT, N}
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
    TabulatedFunction6D{typeof(gpu_data), typeof(ranges), typeof(inv_delta)}(
        gpu_data, ranges, inv_delta)

function ice_integrals_axes(FT)
    # Axes: (log_mass, Fᶠ, Fˡ, rime-density index, μⁱ). The two mass fractions span
    # [0, 1], the rime-density index runs 1..5 over the non-uniform ρᶠ grid (the
    # wrapper applies the transform), and the μⁱ axis spans [0, 20] but is trivial
    # here — the 2-moment tables carry a single μⁱ point.
    return (
        (FT(LOG_MASS_MIN), FT(LOG_MASS_MAX)),
        (FT(0), FT(1)),
        (FT(0), FT(1)),
        (FT(1), FT(5)),
        (FT(0), FT(20))
    )
end

function rain_ice_collection_axes(FT)
    # Axes: (log_mass, log_lambda_r, Fᶠ, Fˡ, rime-density index, μⁱ); the last four
    # match `ice_integrals_axes`.
    return (
        (FT(LOG_MASS_MIN), FT(LOG_MASS_MAX)),
        (FT(LOG_LAMBDA_R_MIN), FT(LOG_LAMBDA_R_MAX)),
        (FT(0), FT(1)),
        (FT(0), FT(1)),
        (FT(1), FT(5)),
        (FT(0), FT(20))
    )
end
